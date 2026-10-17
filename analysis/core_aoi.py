"""Where inside the figure does each model's Kanizsa effect sit?
core = inside AOI pixels more than D px (Euclidean) from the inducer-disc AOI; ring = the rest of the inside AOI.
Per image: shares of total map mass (map clipped at 0) in core, ring and inducer AOI; r_core = core / (core + ind).
Effects (test - control180) with the survey's cluster bootstrap (analysis/analyze_v3.py indices).
Reads the saved maps of scoring/score_v3.py (<results>/v2/scores/maps, not distributed) or, for tags A:<fset>|<arch>,
the readout maps of readout_swap/pilot/evaluate.py (<results>/v2/readout/maps/stim_v3).
Usage: CORE_D=30 python core_aoi.py DeepGaze_IIE DeepGaze_MSDB xSoyelNet-ASPP xSoyelNet-ML MSI-Net Edge-energy_null \
           L:dinov2_L12 L:dinov2_L8 L:siglip_L24 L:clip_L12
Writes <results>/v2/verify/core_aoi_D<D>_<n>.json, where <n> is abs(hash(tags)) % 10**6 (Python string hashing is
randomised per process, so set CORE_OUT to fix the file name; the paper's numbers read core_aoi_D30_835295.json)."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, cv2
cv2.setNumThreads(3)
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v3 as A                                   # noqa: E402  (also puts the repository root on sys.path)
import paths as LP                                       # noqa: E402
V2, STIM = str(LP.V2), str(LP.STIM_V3)
D = int(os.environ.get("CORE_D", "30"))
items = [it for it in json.load(open(f"{STIM}/items.json")) if it["illusion"] == "kanizsa"]
CONDS = ["illusory", "control180", "control90", "mis20", "real_filled", "realC"]
cfgs = sorted({it["config"] for it in items})
masks = {}
for c in cfgs:
    a = np.load(f"{STIM}/img/{c}_illusory_aoi.npz"); ins, ind = a["inside"], a["ind"]
    dd = ndimage.distance_transform_edt(~ind)
    core = ins & (dd > D); masks[c] = (core.ravel(), (ins & ~core).ravel(), ind.ravel())
def comps(sal, c):
    s = np.clip(sal.ravel(), 0, None).astype(np.float64); tot = s.sum(); core, ring, ind = masks[c]
    C, R, I = s[core].sum(), s[ring].sum(), s[ind].sum()
    return dict(core=C / tot, ring=R / tot, ind=I / tot, r_core=C / (C + I), r_inside=(C + R) / (C + R + I))
def survey(tag):
    out = {}
    for c in cfgs:
        for cond in CONDS:
            out[f"{c}_{cond}"] = comps(np.load(f"{V2}/scores/maps/{tag}/{c}_{cond}.npz")["sal"].astype(np.float32), c)
    return out
def secA(fname):
    d = np.load(f"{V2}/readout/maps/stim_v3/{fname}.npz"); names = list(d["names"]); mp = d["maps"]
    out = {}
    for c in cfgs:
        for cond in CONDS:
            n = f"{c}_{cond}"; out[n] = comps(cv2.resize(mp[names.index(n)].astype(np.float32), (1024, 768)), c)
    return out
tags = sys.argv[1:]
res = {}
for t in tags:
    if t.startswith("A:"):
        # section A readout: average components over seeds 0-2 and both sizes
        fs, arch = t[2:].split("|")
        parts = [secA(f"{fs}_{arch}_s{s}_px{px}") for s in (0, 1, 2) for px in (336, 448)]
    elif t.startswith("L:"):
        parts = [survey(f"{t[2:]}_s{px}") for px in (336, 448)]
    else:
        parts = [survey(t)]
    res[t] = parts
    print("loaded", t, flush=True)
summ = {}
for t, parts in res.items():
    row = {}
    for test in ["illusory", "control90", "mis20", "real_filled"]:
        for k in ["r_inside", "r_core", "core", "ring", "ind"]:
            e = A.eff(parts, "kanizsa", test, "control180", k)
            row[f"{test}:{k}"] = A.summ("kanizsa", e, pct=False)
    num = A.eff(parts, "kanizsa", "mis20", "control180", "r_core"); den = A.eff(parts, "kanizsa", "illusory", "control180", "r_core")
    row["ratio20_core"] = A.ratio("kanizsa", num, den)
    lev = {cond: float(np.mean(A.arr(parts, "kanizsa", cond, "core"))) for cond in CONDS}
    row["core_share_levels"] = lev
    summ[t] = row
    f = lambda k: f"{row[k]['mean']:+.3f} [{row[k]['ci'][0]:+.3f},{row[k]['ci'][1]:+.3f}]"
    print(f"{t:28s} r_inside {f('illusory:r_inside')} | r_core {f('illusory:r_core')} | dshare core {f('illusory:core')} ring {f('illusory:ring')} ind {f('illusory:ind')} | ratio20_core {row['ratio20_core']['value']:.2f} {row['ratio20_core']['ci']}", flush=True)
os.makedirs(f"{V2}/verify", exist_ok=True)
json.dump(summ, open(os.environ.get("CORE_OUT") or f"{V2}/verify/core_aoi_D{D}_{abs(hash(tuple(tags))) % 10**6}.json", "w"), indent=1)
