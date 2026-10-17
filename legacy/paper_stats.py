"""
Every number quoted in the manuscript, computed from the result files (written to results/stats.json).
Kanizsa (primary control: 180 deg, as in Kavsek 2023; secondary: 90 deg).
Ehrenstein (primary control: control_end, chosen before any model was run; secondary: control_mid).
Illusion index = (illusory - control) / (real - control), computed on condition means (human reference from
Kavsek 2023, all ages pooled: Kanizsa .78/.82/.54, Ehrenstein .88/.91/.74 for illusory/real/control).
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon, spearmanr, linregress
from common import STIM, RESULTS
MODELS = ["DeepGaze IIE", "DeepGaze MSDB", "xSoyelNet-ML", "MSI-Net", "xSoyelNet-ASPP", "Spectral residual", "Edge-energy null"]
SETS = {"kanizsa": ("control", "control90"), "ehrenstein": ("control_end", "control_mid")}
HUMAN = {"kanizsa": (0.78, 0.82, 0.54), "ehrenstein": (0.88, 0.91, 0.74)}
S = (336, 448, 560, 672)

def pair(r, items, ctrl, filt=lambda it: True):
    return np.array([r[it["name"]] - r[it["name"].replace("_illusory", "_" + ctrl)] for it in items if it["cond"] == "illusory" and filt(it)])

out = {"human": {k: dict(illusory=v[0], real=v[1], control=v[2], delta=round(v[0] - v[2], 3),
                         index=round((v[0] - v[2]) / (v[1] - v[2]), 3)) for k, v in HUMAN.items()}, "survey": {}, "layers": {}}
for st, (c1, c2) in SETS.items():
    items = json.load(open(os.path.join(STIM[st], "items.json")))
    for m in MODELS:
        r = json.load(open(os.path.join(RESULTS, st, f"results_{m.replace(' ', '_')}.json")))
        e = {}
        for tag, c in (("primary", c1), ("secondary", c2)):
            x = pair(r, items, c)
            e[tag] = dict(control=c, mean=round(float(x.mean()), 4), sem=round(float(x.std(ddof=1) / np.sqrt(len(x))), 4),
                          pct=round(float(np.mean(x > 0)) * 100, 1), n=len(x), p=float(wilcoxon(x).pvalue))
        lv = {c: float(np.mean([r[it["name"]] for it in items if it["cond"] == c])) for c in ("illusory", "real", c1)}
        e["levels"] = {k: round(v, 3) for k, v in lv.items()}
        e["index"] = round((lv["illusory"] - lv[c1]) / (lv["real"] - lv[c1]), 3) if lv["real"] != lv[c1] else None
        e["real_minus_control"] = round(lv["real"] - lv[c1], 4)
        if st == "kanizsa":
            srs = sorted({it["sr"] for it in items})
            per = [float(pair(r, items, c1, lambda it, s=s: it["sr"] == s).mean()) for s in srs]
            lr = linregress([it["sr"] for it in items if it["cond"] == "illusory"], pair(r, items, c1))
            e["by_sr"] = dict(sr=srs, mean=[round(v, 4) for v in per], slope=round(lr.slope, 3), slope_p=float(lr.pvalue))
            e["by_shape"] = {sh: round(float(pair(r, items, c1, lambda it, sh=sh: it["shape"] == sh).mean()), 4) for sh in ("square", "triangle")}
        else:
            ns = sorted({it["n"] for it in items})
            e["by_n"] = dict(n=ns, mean=[round(float(pair(r, items, c1, lambda it, k=k: it["n"] == k).mean()), 4) for k in ns])
            e["by_R"] = {str(R): round(float(pair(r, items, c1, lambda it, R=R: it["R"] == R).mean()), 4) for R in (70, 100)}
        out["survey"].setdefault(m, {})[st] = e

rows = []
for bb in ("dinov2", "clip", "siglip"):
    k = json.load(open(os.path.join(RESULTS, "layers", bb, "sweep_results.json")))
    eh = json.load(open(os.path.join(RESULTS, "layers", bb, "ehr_results.json")))
    for L in sorted(int(x) for x in k["mit1003"]):
        kz = [k["kanizsa"][f"L{L}_s{s}"]["d180"] for s in S]; ez = [eh[f"L{L}_s{s}"]["d_end"] for s in S]
        rec = dict(layer=L, depth=round(L / (12 if bb == "dinov2" else 24), 3), **{f"mit_{a}": b for a, b in k["mit1003"][str(L)].items()},
                   kanizsa=round(float(np.mean(kz)), 4), kanizsa_by_size=kz,
                   kanizsa_pct=[k["kanizsa"][f"L{L}_s{s}"]["p180"] for s in S],
                   ehrenstein=round(float(np.mean(ez)), 4), ehrenstein_by_size=ez, ehrenstein_pct=[eh[f"L{L}_s{s}"]["p_end"] for s in S])
        out["layers"].setdefault(bb, []).append(rec); rows.append((rec["mit_CC"], rec["kanizsa"], rec["ehrenstein"], rec["depth"]))
A = np.array(rows)
def sp(a, b):
    r = spearmanr(a, b); return dict(rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 4))
out["layer_corr"] = dict(n=len(A), cc_vs_kanizsa=sp(A[:, 0], A[:, 1]), cc_vs_ehrenstein=sp(A[:, 0], A[:, 2]),
                         cc_vs_mean=sp(A[:, 0], (A[:, 1] + A[:, 2]) / 2), kanizsa_vs_ehrenstein=sp(A[:, 1], A[:, 2]),
                         depth_vs_mean=sp(A[:, 3], (A[:, 1] + A[:, 2]) / 2))
# best-CC layer per backbone and first layer
out["layer_summary"] = {bb: dict(best=max(v, key=lambda r: r["mit_CC"]), first=v[0], last=v[-1]) for bb, v in out["layers"].items()}
json.dump(out, open(os.path.join(RESULTS, "stats.json"), "w"), indent=1)

print("HUMAN", out["human"])
for st in SETS:
    print(f"\n== {st}")
    for m in MODELS:
        e = out["survey"][m][st]; p, s = e["primary"], e["secondary"]
        extra = f"slope {e['by_sr']['slope']:+.3f} p={e['by_sr']['slope_p']:.1e} shapes {e['by_shape']}" if st == "kanizsa" else f"byn {e['by_n']['mean']} byR {e['by_R']}"
        print(f"{m:18s} prim {p['mean']:+.3f}±{p['sem']:.3f} {p['pct']:5.1f}% p={p['p']:.1e} | sec {s['mean']:+.3f} {s['pct']:5.1f}% p={s['p']:.1e} | "
              f"lv {e['levels']} idx {e['index']} | {extra}")
print("\nLAYER CORR", out["layer_corr"])
for bb, v in out["layer_summary"].items():
    print(bb, {k: (r["layer"], r["mit_CC"], r["kanizsa"], r["ehrenstein"]) for k, r in v.items()})
