"""
Full-data readout swap: shared definitions.

Everything is imported from readout_swap/pilot/common.py (feature conventions, readouts R1 to R4) except:
  R1n      R1 with every DeepGaze LayerNorm (image-wide: one mean and variance over all channels and positions)
           replaced by a fixed per-channel normalisation y = (x - mu_c) / sd_c * w_c + b_c (w, b learnable as in
           DeepGaze's LayerNorm affine). Input layer: mu_c, sd_c over all training images x tokens (stats_full.py).
           Hidden layers (saliency net 8 and 16 channels, fixation-selection net 1 and 128 channels): mu_c, sd_c set
           once at initialisation from a fixed calibration set of training images, layer by layer, then frozen
           (a fixed affine map before a learned 1x1 conv is a reparametrisation, so this only sets the starting
           scale). At test time no statistic depends on the test image: the readout is pointwise up to the bilinear
           upsampling and the fixed blur.
  features computed on the fly (float32, as the sweep) instead of float16 caches; device from V3_DEVICE (cpu, mps or
  cuda; the published run used mps).
Default output directory: <results>/v3/gpu/full (scores/, stats/ and the training records ckpt/*.json are distributed;
ckpt/*.pt, state/ and maps/ are not).
"""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

RO2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot")
if RO2 not in sys.path:
    sys.path.insert(0, RO2)
import common as C2                                     # noqa: E402  (sets 3 threads, offline HF)
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F  # noqa: E402
from common import (AB, FSETS, BB_FSETS, BACKBONES, Extractor, PointwiseReadout, make_readout, read_rgb,  # noqa: E402
                    mit_heldout, CC, NSS, S, SIGMA_FRAC, LP)
V3 = str(LP.V3)                                          # result tree
OUT_DEFAULT = str(LP.V3 / "gpu" / "full")

DEVICE = os.environ.get("V3_DEVICE", "cpu")
C2.DEV = DEVICE                                          # Extractor reads common.DEV at call time
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
ARCHS_ALL = ("R1n", "R1", "R2", "R3", "R4")
SEEDS = (0, 1, 2)
CALIB_N, CALIB_SEED = 256, 2026


# ------------------------------------------------------------------ paths
def paths(out):
    d = dict(out=out, ckpt=os.path.join(out, "ckpt"), state=os.path.join(out, "state"), stats=os.path.join(out, "stats"),
             scores=os.path.join(out, "scores"), maps=os.path.join(out, "maps"), done=os.path.join(out, "done"),
             logs=os.path.join(out, "logs"))
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d


def log(path, msg):
    msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(path, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------------ data
def train_items(n_train=None):
    """The sweep's training set (training-cache meta.json, all items with fixations: 12,803) in meta order, with indices
    into the training-cache maps.npy. n_train (dry runs only): a seeded subset, numpy RandomState(0) without replacement, sorted."""
    meta = json.load(open(os.path.join(AB.WORK, "meta.json")))
    its = meta["items"]
    keep = [i for i in range(len(its)) if len(meta["fixes"][i]) > 0]
    if n_train:
        keep = sorted(np.random.RandomState(0).choice(keep, n_train, replace=False).tolist())
    return keep, [its[i]["image"] for i in keep]


def train_maps():
    return np.load(os.path.join(AB.WORK, "maps.npy"), mmap_mode="r")


# ------------------------------------------------------------------ features (on the fly)
def make_extractor(bb):
    """v2 Extractor on DEVICE. Off the CPU, float64 buffers (deepgaze_pytorch Normalizer mean/std, used by the IIE
    backbone) are cast to float32 first, since MPS has no float64; on the CPU the v2 object is used unchanged."""
    if DEVICE == "cpu":
        return Extractor(bb)
    C2.DEV = "cpu"
    try:
        ex = Extractor(bb)
    finally:
        C2.DEV = DEVICE
    for mod in ex.m.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.dtype == torch.float64:
                mod._buffers[name] = buf.float()
    ex.m = ex.m.to(DEVICE).eval()
    return ex


def prep_batch(ex, img_paths, s=S):
    """CPU part of Extractor.__call__ (read, INTER_AREA resize, normalise): run in a loader thread."""
    return torch.stack([ex.prep(read_rgb(p), s) for p in img_paths])


@torch.no_grad()
def feats(ex, x, s):
    """GPU part of Extractor.__call__ (readout_swap/pilot/common.py), same code path; x from prep_batch."""
    x = x.to(DEVICE)
    out = {}
    if ex.bb == "iie":
        fs = ex.m(x)
        g = math.ceil(s / 16)
        out["iie_shapenetc"] = torch.cat([F.interpolate(t, [g, g]) for t in fs], 1).float()
        return out
    o = ex.m(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True)
    hs = o.hidden_states
    cls = BACKBONES[ex.bb]["cls"]
    for fs in BB_FSETS[ex.bb]:
        t = hs[FSETS[fs]["layer"]]
        t = t[:, 1:, :] if cls else t
        g = int(round(t.shape[1] ** 0.5))
        out[fs] = t.transpose(1, 2).reshape(t.shape[0], -1, g, g).contiguous()
    return out


class Loader:
    """Prefetches batches in background threads (cv2 releases the GIL), in the given order.
    Default job: prep_batch(ex, img_paths, s); or any callable fn(item) via fn=."""

    def __init__(self, ex, batches, s=S, workers=2, ahead=3, fn=None):
        self.batches = batches
        self.fn = fn or (lambda b: prep_batch(ex, b, s))
        self.pool = ThreadPoolExecutor(workers)
        self.ahead = ahead

    def __iter__(self):
        futs = []
        try:
            for b in self.batches:
                futs.append(self.pool.submit(self.fn, b))
                if len(futs) > self.ahead:
                    yield futs.pop(0).result()
            for f in futs:
                yield f.result()
        finally:
            self.pool.shutdown(cancel_futures=True)


# ------------------------------------------------------------------ R1n
class FixedNorm(nn.Module):
    """Per-channel normalisation with fixed statistics (buffers) and DeepGaze's learnable per-channel affine."""

    def __init__(self, features, eps=1e-12):
        super().__init__()
        self.features, self.eps = features, eps
        self.weight = nn.Parameter(torch.ones(features)); self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("mu", torch.zeros(features)); self.register_buffer("sd", torch.ones(features))
        self.register_buffer("is_set", torch.zeros((), dtype=torch.long))
        self.calib = False; self._acc = None

    def set_stats(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32)); self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))
        self.is_set.fill_(1)

    def forward(self, x):
        if self.calib:                                     # accumulate per-channel moments (float64, CPU)
            xd = x.detach().cpu().double().transpose(0, 1).reshape(self.features, -1)   # MPS has no float64
            a = (xd.sum(1), (xd * xd).sum(1), xd.shape[1])
            self._acc = a if self._acc is None else (self._acc[0] + a[0], self._acc[1] + a[1], self._acc[2] + a[2])
        v = lambda t: t[None, :, None, None]
        return (x - v(self.mu)) / v(self.sd) * v(self.weight) + v(self.bias)

    def finish_calib(self):
        s, ss, n = self._acc
        mu = s / n; var = (ss / n - mu * mu).clamp_min(0)
        self.set_stats(mu.float(), torch.sqrt(var + self.eps).float())
        self.calib = False; self._acc = None


class FixedNormMultiInput(nn.Module):
    """Drop-in for deepgaze_pytorch.layers.LayerNormMultiInput([1, 0])."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        for k, f in enumerate(features):
            if f:
                setattr(self, f"layernorm_part{k}", FixedNorm(f))

    def forward(self, tensors):
        return [None if not c else getattr(self, f"layernorm_part{k}")(t) for k, (c, t) in enumerate(zip(self.features, tensors))]


class PointwiseReadoutFixedNorm(PointwiseReadout):
    """R1n. Built with R1's constructors first (identical conv initialisation for a given seed: LayerNorm init does
    not draw random numbers), then each LayerNorm is swapped in place for a FixedNorm of the same width."""

    def __init__(self, cin, sigma_frac=SIGMA_FRAC):
        super().__init__(cin, sigma_frac)
        sn, fn = self.saliency_network, self.fixation_selection_network
        sn.layernorm0 = FixedNorm(cin); sn.layernorm1 = FixedNorm(8); sn.layernorm2 = FixedNorm(16)
        fn.layernorm0 = FixedNormMultiInput([1, 0]); fn.layernorm1 = FixedNorm(128)

    def norms(self):
        """FixedNorms in forward order (input layer first)."""
        sn, fn = self.saliency_network, self.fixation_selection_network
        return [sn.layernorm0, sn.layernorm1, sn.layernorm2, fn.layernorm0.layernorm_part0, fn.layernorm1]


def make_head(arch, cin):
    return PointwiseReadoutFixedNorm(cin) if arch == "R1n" else make_readout(arch, cin)


def load_chstats(P, fset):
    d = np.load(os.path.join(P["stats"], f"{fset}_chstats.npz"))
    return d["mean"], d["std"]


@torch.no_grad()
def calibrate_r1n(m, calib_feats, chstats, logf=None, chunk=32):
    """Input FixedNorm from the full-training-set channel statistics; hidden FixedNorms one at a time, in forward
    order, from the calibration features (each layer sees the already-fixed earlier layers)."""
    norms = m.norms()
    norms[0].set_stats(*chstats)
    was = m.training; m.eval()
    for k, nm in enumerate(norms[1:], 1):
        nm.calib = True
        for b in range(0, len(calib_feats), chunk):
            m(calib_feats[b:b + chunk].to(DEVICE), (S, S))
        nm.finish_calib()
        if logf:
            log(logf, f"  R1n calib layer {k} ({nm.features} ch): mu {nm.mu.mean().item():+.4f} sd {nm.sd.mean().item():.4f}")
    m.train(was)
    assert all(int(n.is_set) == 1 for n in norms)


def tag_of(fset, arch, seed):
    return f"{fset}_{arch}_s{seed}"
