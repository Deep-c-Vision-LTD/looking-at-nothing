"""
Readout-free encoding index (transformer tokens inside the figure against matched tokens outside it): shared
definitions.

The projection index uses realC (the real outline drawn over the 180 deg control inducers) as the outline direction,
interior tokens whose input patch is pixel-identical across the compared images, and far-background tokens. realC is
read from stim_v3; the token sets are computed from each token's source-pixel footprint; the background distance is
Euclidean; all hidden states are kept; the per-token components are saved so that any token set can be recomputed
without rerunning a model.

Index for a configuration (tokens of the set pooled):
    E   = sum_tok <f(test) - f(ctrl), f(realC) - f(ctrl)> / sum_tok ||f(realC) - f(ctrl)||^2
    cos = sum_tok <f(test) - f(ctrl), f(realC) - f(ctrl)> / sqrt(sum_tok ||f(test) - f(ctrl)||^2 * sum_tok ||f(realC) - f(ctrl)||^2)
ctrl = control180 (Kanizsa) or control_end (Ehrenstein).
CPU only, 3 threads.
"""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, cv2
from scipy.ndimage import distance_transform_edt

ENC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(ENC))                    # repository root
sys.path.insert(0, ROOT)
import paths as LP                                              # noqa: E402
V2 = str(LP.V2)
STIM = str(LP.STIM_V3)
TOK = str(LP.V2 / "encoding" / "tokens"); LOGS = str(LP.LOGS)   # per-token components (large; not distributed)
for d in (TOK, LOGS):
    os.makedirs(d, exist_ok=True)

W, H = 1024, 768
BG_MIN_PX = 80.0                       # background tokens: every footprint pixel > 80 source px from the figure
SIZES = (336, 448)
B = 10000
BOOT_SEED = 2026

# images per configuration; index 0 is the primary control
K_IMGS = ("control180", "illusory", "control90", "realC", "mis10", "mis20", "mis30")
E_IMGS = ("control_end", "illusory", "realC")
K_TESTS = ("illusory", "control90", "mis10", "mis20", "mis30")   # control90 is the null "test"
E_TESTS = ("illusory",)
# primary layers per backbone (first sweep layer, best-fixation layer, last layer)
PRIMARY = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
SWEEP_LAYERS = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}

VIT = {
    "dinov2": dict(hf="facebook/dinov2-base", patch=14, n_hs=13, cls=True,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "clip": dict(hf="openai/clip-vit-large-patch14", patch=14, n_hs=25, cls=True,
                 mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    "siglip": dict(hf="google/siglip-large-patch16-256", patch=16, n_hs=25, cls=False,
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


def read_rgb(path):
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


def cells(cfgs, ill):
    """[K][P] configuration indices into cfgs (cells sorted, positions sorted)."""
    cl = {}
    for i, c in enumerate(cfgs):
        if c["illusion"] == ill:
            cl.setdefault(c["cell"], []).append((c["pos"], i))
    return [[i for _, i in sorted(v)] for _, v in sorted(cl.items(), key=lambda kv: str(kv[0]))]


# ------------------------------------------------------------------ token / unit sets
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


# ------------------------------------------------------------------ indices and bootstrap
def index_from_sums(dot, nt, nrc, eps=0.0):
    """dot, nt, nrc: sums over the tokens of a set. Returns (E, cos); NaN when the outline direction is zero."""
    if not (nrc > eps):
        return float("nan"), float("nan")
    E = dot / nrc
    cs = dot / math.sqrt(nt * nrc) if nt > 0 else float("nan")
    return E, cs


def pstr(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


class Boot:
    """Cluster bootstrap over cells (with replacement), positions resampled within the sampled cells.
    The same resample indices are used for every statistic of an illusion (paired). NaN configurations (no qualifying
    token or zero outline direction) are ignored in each resample's mean (nanmean)."""

    def __init__(self, cell_idx, seed=BOOT_SEED, b=B):
        self.cells = np.array(cell_idx)                          # [K, P] config indices
        K, P = self.cells.shape
        rng = np.random.RandomState(seed)
        ci = rng.randint(0, K, (b, K)); pi = rng.randint(0, P, (b, K, P))
        self.flat = self.cells[ci[:, :, None], pi].reshape(b, K * P)   # [B, K*P] resampled config indices

    def dist(self, vals):
        """vals: per-config values indexed like cfgs. Returns the bootstrap distribution of the mean."""
        v = np.asarray(vals, np.float64)[self.flat]
        with np.errstate(invalid="ignore"):
            return np.nanmean(v, 1) if np.isnan(v).any() else v.mean(1)

    def summary(self, vals, dist=None):
        v = np.asarray(vals, np.float64)[self.cells.ravel()]
        n = int(np.sum(~np.isnan(v)))
        if n == 0:
            return dict(mean=None, ci=[None, None], p=None, p_str=None, n=0)
        d = self.dist(vals) if dist is None else dist
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        return dict(mean=float(np.nanmean(v)), ci=[float(lo), float(hi)], p=float(p), p_str=pstr(p), n=n)

    def ratio(self, num, den):
        """Paired bootstrap of mean(num) / mean(den) over configurations valid for both."""
        num = np.asarray(num, np.float64).copy(); den = np.asarray(den, np.float64).copy()
        bad = np.isnan(num) | np.isnan(den); num[bad] = np.nan; den[bad] = np.nan
        dn, dd = self.dist(num), self.dist(den)
        v = self.cells.ravel()
        est = float(np.nanmean(num[v]) / np.nanmean(den[v]))
        r = dn / dd
        r = r[np.isfinite(r)]
        lo, hi = np.percentile(r, [2.5, 97.5])
        return dict(ratio=est, ci=[float(lo), float(hi)], n=int(np.sum(~np.isnan(num[v]))))


def log(path, msg):
    with open(path, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)
