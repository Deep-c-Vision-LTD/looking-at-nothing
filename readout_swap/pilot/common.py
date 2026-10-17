"""
Readout swap, shared definitions (first used by the CPU pilot with 3,000 training images; the full-data runs in
readout_swap/ import everything from here).

Feature sets (conventions copied from models/sweep.py):
  dinov2_L12, dinov2_L6  facebook/dinov2-base hidden_states[12] / [6] (no final LayerNorm, CLS dropped)
  siglip_L24             google/siglip-large-patch16-256 hidden_states[24] (no post-LayerNorm, no CLS)
  iie_shapenetc          DeepGaze IIE backbone 0 (RGBShapeNetC) at IIE's six readout layers, nearest-resized to the
                         IIE readout grid ceil(input / 16), 2,048 channels
Readouts:
  R1 DeepGaze IIE 1x1 readout + learned centre bias + fixed Gaussian blur
  R2 ASPP (1, 3, 6, 12) without global pooling
  R3 one 3x3 dilation-1 branch, no global pooling
  R4 sweep readout (ASPP 1, 3, 6, 12 + global pooling), copied exactly
CPU only: 3 threads. ab_objective is imported for its loss and paths only (its DEV is never used here).
Pilot outputs (feature caches, checkpoints, maps, scores) go to <results>/v2/readout/ (not distributed).
"""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")   # =0 to download
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(3)
cv2.setNumThreads(1)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)
import paths as LP                   # noqa: E402
RO = str(LP.V2 / "readout")          # pilot outputs
V2 = str(LP.V2)
import ab_objective as AB            # noqa: E402  loss_A, data paths (AB.DEV is NOT used)
import gate_kanizsa as gk            # noqa: E402  rel_inside
from src.model_aspp import ASPPModule  # noqa: E402
from src.data import build_mit1003     # noqa: E402
from src.metrics import CC, NSS, AUC_Judd  # noqa: E402

DEV = "cpu"
S = 336                                    # sweep training resolution
CACHE = os.path.join(RO, "cache"); CKPT = os.path.join(RO, "ckpt"); MAPS = os.path.join(RO, "maps")
SCORES = os.path.join(RO, "scores"); LOGS = str(LP.LOGS)
for d in (CACHE, CKPT, MAPS, SCORES, LOGS, os.path.join(CACHE, "stim")):
    os.makedirs(d, exist_ok=True)

BACKBONES = {
    "dinov2": dict(hf="facebook/dinov2-base", cls=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "siglip": dict(hf="google/siglip-large-patch16-256", cls=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
FSETS = {
    "dinov2_L12": dict(bb="dinov2", layer=12, cin=768),
    "dinov2_L6": dict(bb="dinov2", layer=6, cin=768),
    "siglip_L24": dict(bb="siglip", layer=24, cin=1024),
    "iie_shapenetc": dict(bb="iie", layer=None, cin=2048),
}
BB_FSETS = {"dinov2": ("dinov2_L12", "dinov2_L6"), "siglip": ("siglip_L24",), "iie": ("iie_shapenetc",)}
IIE_LAYERS = ['1.module.layer3.0.conv2', '1.module.layer3.3.conv2', '1.module.layer3.5.conv1',
              '1.module.layer3.5.conv2', '1.module.layer4.1.conv2', '1.module.layer4.2.conv2']
ARCHS = ("R1", "R2", "R3", "R4")
SEEDS = (0, 1, 2)
# R1 blur: mean learned sigma of the 120 DeepGaze IIE finalizers (10.13 px at IIE's ~512 px network input)
IIE_SIGMA_PX, IIE_NET_PX = 10.1337, 512.0
SIGMA_FRAC = IIE_SIGMA_PX / IIE_NET_PX


# ------------------------------------------------------------------ feature extraction
class Extractor:
    """Maps a list of RGB uint8 images to {fset: float32 tensor (B, C, g, g)} at a square input size."""

    def __init__(self, bb):
        self.bb = bb
        if bb == "dinov2":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained(BACKBONES[bb]["hf"]).to(DEV).eval()
        elif bb == "siglip":
            from transformers import SiglipVisionModel
            self.m = SiglipVisionModel.from_pretrained(BACKBONES[bb]["hf"]).to(DEV).eval()
        else:
            from deepgaze_pytorch.features.shapenet import RGBShapeNetC
            from deepgaze_pytorch.modules import FeatureExtractor
            net = RGBShapeNetC()
            # use the weights stored in the pretrained DeepGaze IIE checkpoint (backbone 0)
            sd = torch.load(os.path.join(torch.hub.get_dir(), "checkpoints", "deepgaze2e.pth"), map_location="cpu")
            pre = "models.0.features.features."
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            missing, unexpected = net.load_state_dict(sub, strict=False)
            assert not unexpected and all("num_batches_tracked" in k for k in missing), (missing[:5], unexpected[:5])
            self.m = FeatureExtractor(net, IIE_LAYERS).to(DEV).eval()
        if bb in BACKBONES:
            self.mean = np.array(BACKBONES[bb]["mean"], np.float32); self.std = np.array(BACKBONES[bb]["std"], np.float32)

    def prep(self, rgb, s):
        x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.bb == "iie":
            return torch.from_numpy(x.transpose(2, 0, 1))                 # 0..255; IIE Normalizer divides by 255
        return torch.from_numpy(((x / 255 - self.mean) / self.std).transpose(2, 0, 1))

    @torch.no_grad()
    def __call__(self, rgbs, s=S, hidden_index=None):
        x = torch.stack([self.prep(r, s) for r in rgbs]).to(DEV)
        out = {}
        if self.bb == "iie":
            fs = self.m(x)
            g = math.ceil(s / 16)
            out["iie_shapenetc"] = torch.cat([F.interpolate(t, [g, g]) for t in fs], 1).float()
            return out
        o = self.m(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True)
        hs = o.hidden_states
        cls = BACKBONES[self.bb]["cls"]
        def grid(t):
            t = t[:, 1:, :] if cls else t
            g = int(round(t.shape[1] ** 0.5))
            return t.transpose(1, 2).reshape(t.shape[0], -1, g, g).contiguous()
        for fs in BB_FSETS[self.bb]:
            out[fs] = grid(hs[FSETS[fs]["layer"]])
        if hidden_index is not None:                                      # for convention checks only
            for k in hidden_index:
                out[f"hs{k}"] = grid(hs[k])
            out["last_hidden_state"] = grid(o.last_hidden_state)
        return out


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ readouts
class ASPPNoGAP(nn.Module):
    """ASPPModule without the global-pooling branch (identical branches and projection otherwise)."""

    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU()))
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class ASPPReadout(nn.Module):
    """R4 is sweep.Readout verbatim (same parameter names, so sweep checkpoints load); R2/R3 swap the ASPP block."""

    def __init__(self, cin, arch="R4"):
        super().__init__()
        if arch == "R4":
            self.aspp = ASPPModule(cin, 128, (1, 3, 6, 12))
        elif arch == "R2":
            self.aspp = ASPPNoGAP(cin, 128, (1, 3, 6, 12))
        elif arch == "R3":
            self.aspp = ASPPNoGAP(cin, 128, (1,))
        else:
            raise ValueError(arch)
        self.head = nn.Conv2d(128, 1, 1)
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True

    def forward(self, f, out_hw):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)


class PointwiseReadout(nn.Module):
    """R1: DeepGaze IIE's per-component readout (all 1x1), learned centre bias, bilinear upsampling, fixed blur."""

    def __init__(self, cin, sigma_frac=SIGMA_FRAC):
        super().__init__()
        from deepgaze_pytorch.deepgaze2e import build_saliency_network, build_fixation_selection_network
        self.saliency_network = build_saliency_network(cin)
        self.fixation_selection_network = build_fixation_selection_network()
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True
        self.sigma_frac = sigma_frac

    def forward(self, f, out_hw):
        from deepgaze_pytorch.layers import gaussian_filter_1d
        s = self.fixation_selection_network((self.saliency_network(f), None))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        s = F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)
        for dim, n in ((2, out_hw[0]), (3, out_hw[1])):
            s = gaussian_filter_1d(s, dim=dim, sigma=self.sigma_frac * n, truncate=3, padding_mode="replicate")
        return s


def make_readout(arch, cin):
    """arch R1..R4; a suffix (e.g. R1x30ep, an exploratory longer-trained run) keeps the architecture of arch[:2]."""
    return PointwiseReadout(cin) if arch[:2] == "R1" else ASPPReadout(cin, arch[:2])


def ckpt_path(fset, arch, seed):
    return os.path.join(CKPT, f"{fset}_{arch}_s{seed}.pt")


def load_readout(fset, arch, seed):
    """seed == "sweep": the original full-data sweep readout (R4 only; reference and convention check)."""
    m = make_readout(arch, FSETS[fset]["cin"]).to(DEV)
    if seed == "sweep":
        assert arch == "R4" and FSETS[fset]["bb"] in BACKBONES
        p = os.path.join(str(LP.LAYER_READOUTS), FSETS[fset]["bb"], f"readout_L{FSETS[fset]['layer']}.pt")
    else:
        p = ckpt_path(fset, arch, seed)
    m.load_state_dict(torch.load(p, map_location=DEV)); m.eval()
    return m


# ------------------------------------------------------------------ data
def train_subset():
    p = os.path.join(CACHE, "train_subset.json")
    if os.path.exists(p):
        return json.load(open(p))
    meta = json.load(open(os.path.join(AB.WORK, "meta.json")))
    its = meta["items"]
    keep = [i for i in range(len(its)) if len(meta["fixes"][i]) > 0]       # sweep's filter (all 12,803 pass)
    idx = sorted(np.random.RandomState(0).choice(keep, 3000, replace=False).tolist())
    rec = dict(n_pool=len(keep), seed=0, idx=idx, images=[its[i]["image"] for i in idx], src=[its[i]["src"] for i in idx])
    json.dump(rec, open(p, "w"))
    return rec


def mit_heldout():
    return build_mit1003(f"{AB.SR}/mit1003")[-200:]


def stim_items(stimset):
    if stimset == "stim_v2":
        D = str(LP.STIM_V2)
        items = json.load(open(os.path.join(D, "items.json")))
        sel = [dict(it, illusion="kanizsa", img=os.path.join(D, "img", it["name"] + ".png"),
                    aoi=os.path.join(D, "img", it["name"] + "_aoi.npz"))
               for it in items if it["sr"] in (0.3, 0.5, 0.7) and it["cond"] in ("illusory", "control", "control90")]
        return sel
    D = str(LP.STIM_V3)
    items = json.load(open(os.path.join(D, "items.json")))
    return [dict(it, img=os.path.join(D, "img", it["name"] + ".png"), aoi=os.path.join(D, "img", it["aoi"])) for it in items]


def log(path, msg):
    with open(path, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)
