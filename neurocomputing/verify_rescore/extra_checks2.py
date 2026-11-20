"""
Independent checks of two inputs to the P3/P4 decisions and of the sanity-gate claim, from raw files:
  (a) accuracy criterion (rule 2.7): per swap cell, the mean of the three unrounded seed held-out MIT1003 CCs
      (v3/gpu/full/scores/mit_<tag>_s<seed>.json, key CC), rounded to 4 decimals, against round(R4 mean - 0.03, 4)
      with the R4 mean rounded to 4 decimals; the 10 P3 cells must be exactly the valid non-R4 cells;
  (b) my recomputed point estimates against the published 4-decimal values in known_values.json (P1 L20, P1 model
      minus null, P2 square Delta20 and minus null, swap E, U20, A20): |diff| must be <= 5e-5 + 1e-12 (rounding).
Writes extra_checks2.json. Imports only recompute_stats (this folder).
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recompute_stats as RS

out = {}
# (a) validity
cc = {}
for f in RS.FSETS:
    for a in RS.ARCHS:
        cc[(f, a)] = [RS.jl(os.path.join(RS.ROOT, "v3/gpu/full/scores", f"mit_{f}_{a}_s{s}.json"))["CC"] for s in (0, 1, 2)]
valid = {}
for f in RS.FSETS:
    r4 = round(float(np.mean(cc[(f, "R4")])), 4)
    thr = round(r4 - 0.03, 4)
    for a in RS.ARCHS[:-1]:
        m = round(float(np.mean(cc[(f, a)])), 4)
        valid[f"{f}|{a}"] = dict(cc_mean=m, threshold=thr, R4_mean=r4, valid=bool(m >= thr), margin=round(m - thr, 4))
out["validity"] = valid
vset = sorted(k for k, v in valid.items() if v["valid"])
out["valid_cells"] = vset
out["valid_equals_P3_list"] = vset == sorted(f"{f}|{a}" for f, a in RS.P3_CELLS)

# (b) point estimates vs published 4-decimal values
K = json.load(open(os.path.join(RS.RE, "known_values.json")))
mine = json.load(open(os.path.join(HERE, "recompute_stats.json")))
chk = []
def c(name, pub, me):
    chk.append(dict(name=name, published=float(pub), recomputed=float(me), diff=float(abs(pub - me)), ok=bool(abs(pub - me) <= 5e-5 + 1e-12)))
for mod in ("DeepGaze IIE", "DeepGaze MSDB"):
    c(f"P1 {mod} L20", K["P1"][mod]["L20"], mine["P1"][mod]["L_model"]["mean"])
    for nk, v in K["P1"][mod]["diff_vs_nulls"].items():
        c(f"P1 {mod} minus {nk}", v, mine["P1"][mod]["vs_null"][nk]["mean"])
    c(f"P2 {mod} Delta20", K["P2"][mod]["Delta20"], mine["P2"][mod]["Delta_model"]["mean"])
    for nk, v in K["P2"][mod]["diff_vs_nulls"].items():
        c(f"P2 {mod} minus {nk}", v, mine["P2"][mod]["vs_null"][nk]["mean"])
H = {(f, a): RS.head_scores(f, a) for f in RS.FSETS for a in RS.ARCHS}
keys, grid = RS.make_design()
for f in RS.FSETS:
    for a in RS.ARCHS:
        sc = H[(f, a)]
        I = RS.arr(grid, lambda q: RS.val(sc, q, "illusory")).mean(); C = RS.arr(grid, lambda q: RS.val(sc, q, "control180")).mean()
        M = RS.arr(grid, lambda q: RS.val(sc, q, "mis20")).mean(); A = RS.arr(grid, lambda q: RS.val(sc, q, "alt20")).mean()
        kv = K["swap_cells"][f"{f}|{a}"]
        c(f"swap {f}|{a} E", kv["E"], I - C)
        c(f"swap {f}|{a} U20", kv["ratio_uni20"]["value"], (M - C) / (I - C))
        c(f"swap {f}|{a} A20", kv["ratio_alt20"]["value"], (A - C) / (I - C))
out["published_checks"] = chk
out["n_published_checks"] = len(chk)
out["n_published_ok"] = sum(x["ok"] for x in chk)
out["max_published_diff"] = max(x["diff"] for x in chk)
json.dump(out, open(os.path.join(HERE, "extra_checks2.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("published_checks", "validity")}, indent=1))
print({k: v for k, v in valid.items() if not v["valid"] or v["margin"] < 0.005})
