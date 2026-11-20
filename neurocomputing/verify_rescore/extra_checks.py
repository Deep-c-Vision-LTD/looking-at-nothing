"""
Extra independent checks on claims printed in neuro/reanalysis/results/PRIMARY_P1_P4.md that are not single JSON keys:
  - 30 of 30 seed point estimates of the 10 valid P3 cells have R4's sign (rule 1.1 item 2; size-averaged per seed);
  - WCR decision at .05 and WCU interval agree for all 74 linear primary contrasts (PRIMARY_P1_P4.md Section 5);
  - null with the largest point loss L20 is corner_s60 for P1; null square Delta20 point range (P2 sentence);
  - skewness range of the cluster means over the primary linear contrasts.
Reads recompute_stats.json and the raw swap score files; imports only recompute_stats (this folder). Writes
extra_checks.json.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recompute_stats as RS

mine = json.load(open(os.path.join(HERE, "recompute_stats.json")))
keys, grid = RS.make_design(); ksq, gsq = RS.make_design("square")
out = {}

# seed point estimates (size-averaged per seed), 10 valid cells, sign vs R4 point sign
seed_signs = {}
n_same = 0
for f, a in RS.P3_CELLS:
    ref_sign = np.sign(mine["P3"]["reference"][f]["mean"])
    vals = []
    for s in (0, 1, 2):
        d = RS.jl(os.path.join(RS.ROOT, "v3/gpu/full/scores", f"stim_v3_{f}_{a}_s{s}.json"))
        sc = {k: 0.5 * (d["336"]["r_inside"][k] + d["448"]["r_inside"][k]) for k in d["336"]["r_inside"]}
        e = RS.arr(grid, lambda c: sc[f"{c}_illusory"] - sc[f"{c}_control180"]).mean()
        vals.append(float(e)); n_same += int(np.sign(e) == ref_sign)
    seed_signs[f"{f}|{a}"] = vals
out["seed_point_estimates"] = seed_signs
out["n_seed_same_sign_of_30"] = n_same

# WCR vs WCU agreement over the 74 linear primary contrasts
lin = []
for fam in ("P1", "P2"):
    for mod in ("DeepGaze IIE", "DeepGaze MSDB"):
        M = mine[fam][mod]
        lin.append(M.get("L_model") or M.get("Delta_model"))
        lin += list(M["vs_null"].values())
lin += list(mine["P3"]["cells"].values())
agree = sum(((r["p"] < 0.05) == (r["ci"][0] > 0 or r["ci"][1] < 0)) for r in lin)
out["n_linear"] = len(lin); out["n_wcr_wcu_agree"] = agree
sk = [r["skew_cluster_means"] for r in lin]
out["skew_range"] = [min(sk), max(sk)]

# null with largest loss (P1) and null square Delta20 point range (P2)
out["P1_null_with_largest_loss"] = {m: mine["P1"][m]["null_with_largest_loss"] for m in mine["P1"]}
N = RS.null_scores()
dl = {k: float(RS.arr(gsq, lambda c: RS.val(N[k], c, "alt20") - RS.val(N[k], c, "mis20")).mean()) for k in RS.NULLS}
out["P2_null_square_Delta20"] = dl
out["P2_null_square_Delta20_range"] = [min(dl.values()), max(dl.values())]
json.dump(out, open(os.path.join(HERE, "extra_checks.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("seed_point_estimates", "P2_null_square_Delta20")}, indent=1))
