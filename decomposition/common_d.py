"""
Shapley decomposition of the Kanizsa effect in DeepGaze IIE and DeepGaze MSDB: shared definitions.

Stimuli: stim_v3 Kanizsa, 60 configurations, illusory vs control180.
Measure: r = mass(inside) / (mass(inside) + mass(ind)) on the non-negative map at display resolution (768 x 1024),
as gate_kanizsa.rel_inside; AOI variants inside (primary), inside_eroded, inside_eroded_all.
Scores are stored as AOI masses of every (hybrid) map, so r for any of the three variants is recomputable.
CPU only, 3 threads. Scores and results go to <results>/v2/decompose (scores/, results.json); maps and the MSDB
feature cache go to its maps/ and cache/ subfolders (large; not distributed).
"""
import json, math, os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, cv2

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import paths as LP  # noqa: E402

V2 = str(LP.V2)
DD = os.path.join(V2, "decompose")
STIM = str(LP.STIM_V3)
LEG = str(LP.LEGACY_RESULTS)                       # first-version scores, for the cross-checks in analyze.py
MAPS = os.path.join(DD, "maps"); SCORES = os.path.join(DD, "scores"); CACHE = os.path.join(DD, "cache")
LOGS = str(LP.LOGS)
for d in (MAPS, SCORES, CACHE, LOGS):
    os.makedirs(d, exist_ok=True)

H, W = 768, 1024
VARIANTS = ("inside", "inside_eroded", "inside_eroded_all")
AOI_KEYS = ("inside", "ind", "inside_eroded", "inside_eroded_all")


def torch_setup():
    import torch
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    return torch


def kanizsa_configs():
    """60 Kanizsa configurations (sorted), each dict with config, cell key, illusory and control180 image names."""
    items = json.load(open(os.path.join(STIM, "items.json")))
    out = []
    for it in items:
        if it["illusion"] == "kanizsa" and it["cond"] == "illusory":
            out.append(dict(config=it["config"], cell=(it["shape"], it["sr"], it["pol"]), pos=it["pos"],
                            shape=it["shape"], sr=it["sr"], pol=it["pol"],
                            ill=it["name"], ctrl=it["siblings"]["control180"]))
    out.sort(key=lambda d: d["config"])
    assert len(out) == 60
    return out


def read_rgb(name):
    return cv2.cvtColor(cv2.imread(os.path.join(STIM, "img", name + ".png")), cv2.COLOR_BGR2RGB)


def read_aoi(name):
    a = np.load(os.path.join(STIM, "img", name + "_aoi.npz"))
    return {k: a[k].astype(bool) for k in AOI_KEYS}


def masses(sal, aoi):
    """AOI masses of the non-negative map (float64), as in gate_kanizsa.rel_inside."""
    s = np.clip(np.asarray(sal, np.float64), 0, None)
    return {k: float(s[aoi[k]].sum()) for k in AOI_KEYS}


def r_from(m, var="inside"):
    a, b = m[var], m["ind"]
    return a / (a + b) if a + b > 0 else float("nan")


# ------------------------------------------------------------------ Shapley
def shapley(n, v):
    """Exact Shapley values for a game on players 0..n-1; v(mask) -> value, v(0) = 0 assumed by the caller's definition."""
    phi = np.zeros(n)
    f = math.factorial
    for i in range(n):
        for m in range(1 << n):
            if m >> i & 1:
                continue
            s = bin(m).count("1")
            w = f(s) * f(n - s - 1) / f(n)
            phi[i] += w * (v(m | (1 << i)) - v(m))
    return phi


def shapley_both(n, r_of_mask):
    """r_of_mask(T): r of the hybrid whose players in T carry illusory features (others control180).
    Insertion game v_ins(S) = r(S) - r(empty); removal game v_rem(S) = r(N) - r(N minus S).
    Returns phi_ins, phi_rem, their mean, the full effect, and single-player insertion / removal effects."""
    N = (1 << n) - 1
    v_ins = lambda S: r_of_mask(S) - r_of_mask(0)
    v_rem = lambda S: r_of_mask(N) - r_of_mask(N & ~S)
    p_ins, p_rem = shapley(n, v_ins), shapley(n, v_rem)
    eff = r_of_mask(N) - r_of_mask(0)
    ins1 = np.array([v_ins(1 << i) for i in range(n)])
    rem1 = np.array([v_rem(1 << i) for i in range(n)])
    return dict(phi_ins=p_ins, phi_rem=p_rem, phi=(p_ins + p_rem) / 2, effect=eff, ins_alone=ins1, rem_alone=rem1)


# ------------------------------------------------------------------ cluster bootstrap (cells with replacement, positions within cells)
B = 10000


def pstr(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".", 1)


def cell_index(cfgs):
    """[K cells][P positions] indices into the sorted config list (12 x 5)."""
    cells = {}
    for i, c in enumerate(cfgs):
        cells.setdefault(c["cell"], []).append(i)
    keys = sorted(cells, key=str)
    idx = np.array([sorted(cells[k], key=lambda j: cfgs[j]["pos"]) for k in keys])
    return idx, keys


def boot_draws(K, P, seed=2026):
    rng = np.random.RandomState(seed)
    ci = rng.randint(0, K, (B, K)); pi = rng.randint(0, P, (B, K, P))
    return ci, pi


def boot_mean(vals_kp, draws):
    ci, pi = draws
    return vals_kp[ci[:, :, None], pi].mean((1, 2))


def summ(vals_kp, draws, nd=4):
    bm = boot_mean(vals_kp, draws)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)))
    return dict(mean=round(float(vals_kp.mean()), nd), ci=[round(float(lo), nd), round(float(hi), nd)], p=float(p), p_str=pstr(p),
                pct_configs_pos=round(float(np.mean(vals_kp > 0) * 100), 1))


def summ_ratio(num_kp, den_kp, draws, nd=3):
    """Ratio of means with the bootstrap CI of the ratio of resampled means."""
    bn, bd = boot_mean(num_kp, draws), boot_mean(den_kp, draws)
    br = bn / bd
    lo, hi = np.percentile(br, [2.5, 97.5])
    return dict(share=round(float(num_kp.mean() / den_kp.mean()), nd), ci=[round(float(lo), nd), round(float(hi), nd)])


def save_json(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)
