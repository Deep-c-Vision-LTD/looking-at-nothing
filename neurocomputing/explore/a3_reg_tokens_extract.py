"""
A3 (iv) registers-checkpoint token components, extraction step (RUN_PLAN.md Step 2, T1). Exploratory analysis of
REANALYSIS_RULE.md Section 6, item 8 (iv). Written 18 November 2026 after the rule was fixed and public; not yet run.

Rule text (Section 6, item 8, quoted verbatim):
  "A3 readout-free token analysis (decision D5; Critic fix 6): interior index E_int on pixel-identical interior tokens at
  every block of DINOv2-B, CLIP-L and SigLIP-L (336, 448 px and size average). (i) Analysis of record: the rule of 15
  October 2026 (figure-specific if the interior lower bound exceeds the upper bounds of both the background null and the
  control90 null). (ii) Post hoc: the near-exterior null at matched inducer distance (`encoding_nulls.py`). (iii) The
  paired interior-minus-near-exterior excess per configuration with a WCU interval and its retention under mis20 (BCa,
  stability flag). (iv) Registers checkpoint token components, required: all 12 blocks of
  `facebook/dinov2-with-registers-base` at 336 and 448 px, CLS and 4 register tokens dropped, same code path, both
  nulls; they are the only readout-free test of whether the block-12 change between checkpoints exists at the token
  level. (v) The comparison table for the 4 swap feature sets and the 18 sweep blocks (token verdict under each null,
  saliency E with interval, S20). No headline dissociation is drawn from either null."

This script is a copy of looking-at-nothing/analysis/encoding/run_vit.py (= v2/encoding/run_vit.py, the code that wrote
v2/encoding/tokens) with the helper definitions it imports from common_enc.py copied verbatim below (K_IMGS, E_IMGS,
K_TESTS, E_TESTS, BG_MIN_PX, configs, figure_dist, footprints, region_masks, patch_equal, read_rgb). Changes, nothing
else:
  - backbone facebook/dinov2-with-registers-base (13 hidden states = embeddings + 12 blocks, patch 14, ImageNet
    normalisation); the first 5 tokens (CLS and the 4 registers) are dropped instead of CLS alone;
  - the model is called as the registers path of gpu/recommon.py and nn/registers/norm_diag.py call it
    (m(pixel_values=x, output_hidden_states=True); the registers model always interpolates its position embeddings and
    takes no interpolate_pos_encoding argument);
  - both sizes (336, 448) in one process; output folder neuro/reanalysis/A3_tokens/registers/dinov2reg_s<S>/
    (RUN_PLAN T1), log in neuro/reanalysis/A3_tokens/registers/logs/; nothing is written outside neuro/;
  - require_public() first (rule Section 8.1); torch threads from OMP_NUM_THREADS (4); no ab_objective import, CPU only,
    and excommon.metal_guard() after the model load.
Output per configuration (G x G tokens, row-major), as run_vit.py: dot [13, T, G*G], nt [13, T, G*G], nrc [13, G*G],
same [I-1, G*G], inside_full [G*G], far [G*G], imgs, tests, grid = [G, P, S], plus dropped_tokens = 5.
Usage (after the rule is public; CPU, one process, about 0.2 h):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a3_reg_tokens_extract.py [max_configs]
"""
import json, math, os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
from scipy.ndimage import distance_transform_edt                          # noqa: E402
import excommon as X                                                      # noqa: E402

HF = "facebook/dinov2-with-registers-base"
NET, P, N_HS, DROP = "dinov2reg", 14, 13, 5
MEAN, STD = np.array((0.485, 0.456, 0.406), np.float32), np.array((0.229, 0.224, 0.225), np.float32)
OUT_ROOT = os.path.join(X.RE, "A3_tokens", "registers")
STIM = X.STIM_V3
W, H = 1024, 768

# ------------------------------------------------------------------ verbatim from analysis/encoding/common_enc.py
BG_MIN_PX = 80.0                       # background tokens: every footprint pixel > 80 source px from the figure
K_IMGS = ("control180", "illusory", "control90", "realC", "mis10", "mis20", "mis30")
E_IMGS = ("control_end", "illusory", "realC")
K_TESTS = ("illusory", "control90", "mis10", "mis20", "mis30")   # control90 is the null "test"
E_TESTS = ("illusory",)


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def items():
    return json.load(open(os.path.join(STIM, "items.json")))


def configs():
    """Ordered configuration records: Kanizsa (60) then Ehrenstein (40), with image paths and AOI path."""
    out = []
    for it in items():
        if it["cond"] != "illusory":
            continue
        ill = it["illusion"]
        imgs = K_IMGS if ill == "kanizsa" else E_IMGS
        cell = (it["shape"], it["sr"], it["pol"]) if ill == "kanizsa" else (it["R"], it["n"], it["pol"])
        out.append(dict(config=it["config"], illusion=ill, cell=cell, pos=it["pos"], pol=it["pol"],
                        imgs={c: os.path.join(STIM, "img", it["siblings"][c] + ".png") for c in imgs},
                        aoi=os.path.join(STIM, "img", it["aoi"])))
    return out


def figure_dist(aoi):
    """Euclidean distance (source px) of every source pixel to the figure (inside AOI | inducer AOI)."""
    fig = aoi["inside"] | aoi["ind"]
    return distance_transform_edt(~fig)


def footprints(n_rows, n_cols, cell_h, cell_w, sy, sx, oy=0.0, ox=0.0):
    """Source-pixel index ranges [y0, y1) x [x0, x1) of grid cells of cell_h x cell_w network pixels.
    Network pixel u covers source [u * s, (u + 1) * s) (cv2 INTER_AREA and nearest x2 both read only pixels in it).
    oy, ox = network-pixel offset of cell 0 (negative for CNN stride cells centred on unit centres)."""
    out = np.zeros((n_rows, n_cols, 4), np.int64)
    for r in range(n_rows):
        a0 = max(0.0, (oy + r * cell_h) * sy); a1 = min(H, (oy + (r + 1) * cell_h) * sy)
        for c in range(n_cols):
            b0 = max(0.0, (ox + c * cell_w) * sx); b1 = min(W, (ox + (c + 1) * cell_w) * sx)
            out[r, c] = (int(math.floor(a0)), int(math.ceil(a1)), int(math.floor(b0)), int(math.ceil(b1)))
    return out


def region_masks(fp, aoi, dist):
    """Per cell: entirely inside the 'inside' AOI; every pixel > BG_MIN_PX from the figure."""
    ins = aoi["inside"]
    n_r, n_c = fp.shape[:2]
    inside_full = np.zeros((n_r, n_c), bool); far = np.zeros((n_r, n_c), bool)
    for r in range(n_r):
        for c in range(n_c):
            y0, y1, x0, x1 = fp[r, c]
            if y1 <= y0 or x1 <= x0:
                continue
            inside_full[r, c] = ins[y0:y1, x0:x1].all()
            far[r, c] = dist[y0:y1, x0:x1].min() > BG_MIN_PX
    return inside_full, far


def patch_equal(a, b, G, P):
    """True where the P x P input patch of token (gy, gx) is pixel-identical (all channels) in a and b (S x S x 3)."""
    return (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1).reshape(G, P, G, P).max((1, 3)) == 0)


# ------------------------------------------------------------------ run_vit.py, adapted (see the header)
def log(path, msg):
    with open(path, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def token_components(hs, names, tests, drop=DROP):
    """run_vit.py inner loop on a tuple of hidden states (each [I, drop + G*G, C]): per hidden state, the projection of
    every test's token difference onto the realC difference (dot), the test's squared norm (nt) and realC's (nrc).
    Torch tensors take run_vit.py's torch double path; numpy arrays (unit tests) the same arithmetic in numpy float64."""
    G2 = hs[0].shape[1] - drop
    ir = names.index("realC"); it = [names.index(t) for t in tests]
    dot = np.zeros((len(hs), len(tests), G2), np.float32); nt = np.zeros_like(dot)
    nrc = np.zeros((len(hs), G2), np.float32)
    for L, h in enumerate(hs):
        if isinstance(h, np.ndarray):
            f = h[:, drop:, :].astype(np.float64)
            d = f - f[0:1]
            drc = d[ir]
            nrc[L] = (drc * drc).sum(-1)
            for j, i in enumerate(it):
                dot[L, j] = (d[i] * drc).sum(-1); nt[L, j] = (d[i] * d[i]).sum(-1)
            continue
        f = h[:, drop:, :].double()                               # run_vit.py: (h[:, 1:, :] if cls else h).double()
        d = f - f[0:1]
        drc = d[ir]
        nrc[L] = (drc * drc).sum(-1).numpy()
        for j, i in enumerate(it):
            dot[L, j] = (d[i] * drc).sum(-1).numpy(); nt[L, j] = (d[i] * d[i]).sum(-1).numpy()
    return dot, nt, nrc


def run(max_configs=None):
    import cv2, torch
    from transformers import AutoModel
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cv2.setNumThreads(1)
    os.makedirs(os.path.join(OUT_ROOT, "logs"), exist_ok=True)
    m = AutoModel.from_pretrained(HF).to("cpu").eval()
    assert m.config.num_register_tokens == DROP - 1, m.config.num_register_tokens
    X.metal_guard()
    cfgs = configs()[:max_configs] if max_configs else configs()
    for S in X.SIZES:
        G = S // P
        assert G * P == S, (S, P)
        OUT = os.path.join(OUT_ROOT, f"{NET}_s{S}"); os.makedirs(OUT, exist_ok=True)
        LOG = os.path.join(OUT_ROOT, "logs", f"encoding_{NET}_s{S}.log")
        log(LOG, f"=== start {NET} s{S} grid {G}x{G} {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {os.getpid()})")
        fp = footprints(G, G, P, P, H / S, W / S)
        t0 = time.time()
        with torch.no_grad():
            for k, cf in enumerate(cfgs):
                path = os.path.join(OUT, cf["config"] + ".npz")
                if os.path.exists(path):
                    continue
                names = K_IMGS if cf["illusion"] == "kanizsa" else E_IMGS
                tests = K_TESTS if cf["illusion"] == "kanizsa" else E_TESTS
                imgs = [cv2.resize(read_rgb(cf["imgs"][c]), (S, S), interpolation=cv2.INTER_AREA) for c in names]
                same = np.stack([patch_equal(imgs[i], imgs[0], G, P).ravel() for i in range(1, len(names))])
                aoi = np.load(cf["aoi"])
                inside_full, far = region_masks(fp, aoi, figure_dist(aoi))
                x = torch.stack([torch.from_numpy(((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
                                 for im in imgs])
                hs = m(pixel_values=x, output_hidden_states=True).hidden_states
                assert len(hs) == N_HS and hs[0].shape[1] == DROP + G * G, (len(hs), hs[0].shape)
                dot, nt, nrc = token_components(hs, list(names), list(tests))
                np.savez_compressed(path, dot=dot, nt=nt, nrc=nrc, same=same, inside_full=inside_full.ravel(),
                                    far=far.ravel(), imgs=np.array(names), tests=np.array(tests), grid=np.array([G, P, S]),
                                    dropped_tokens=np.array(DROP))
                if k % 10 == 0 or k == len(cfgs) - 1:
                    log(LOG, f"  {k + 1}/{len(cfgs)} {cf['config']} interior-identical tokens "
                             f"{int((same.all(0) & inside_full.ravel()).sum())} far {int(far.sum())} {time.time() - t0:.0f}s")
        log(LOG, f"=== end {NET} s{S} {time.time() - t0:.0f}s {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if not max_configs:
            open(os.path.join(OUT, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    X.require_public()                                                # rule Section 8.1
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
