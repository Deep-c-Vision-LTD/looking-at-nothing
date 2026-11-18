"""
Sanity check (nn/ANALYSIS_RULE.md, Section 8): one already-scored sweep readout rescored by nn/score_nn.py must reproduce
the stored per-image scores and the stored summary effects and ratios.
Compares nn/results/sanity/{raw_v2style, raw_v3maps, raw_alt}/<tag>.json with v2/scores/raw, v3/scores/raw_v3maps and
v3/scores/raw (every field, every image), then runs v3/scores/analyze_alt.analyse on the rescored parts and compares with
v3/scores/results.json (size average and per size; inside r, core r, core share). Writes nn/results/sanity/sanity_report.json.
Usage: python sanity_check.py <bb> <L> [sanity_dir]
"""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nn as C                                   # noqa: E402
import analyze_nn as AN                                 # noqa: E402
AA = AN.AA

bb, L = sys.argv[1], int(sys.argv[2])
SD = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(C.RES, "sanity")
rep = dict(readout=C.rname(bb, L), per_image={}, summary={}, per_size={})

for s in C.SIZES:
    t = C.tag(bb, L, s)
    for new_d, old_p in (("raw_v2style", os.path.join(C.V2SC, "raw", t + ".json")),
                         ("raw_v3maps", os.path.join(C.V3SC, "raw_v3maps", t + ".json")),
                         ("raw_alt", os.path.join(C.V3SC, "raw", t + ".json"))):
        new, old = json.load(open(os.path.join(SD, new_d, t + ".json"))), json.load(open(old_p))
        assert set(new) == set(old), (t, new_d, len(new), len(old))
        fields = sorted(next(iter(old.values())).keys())
        mx = {f: max(abs(new[k][f] - old[k][f]) for k in old) for f in fields}
        exact = sum(all(new[k][f] == old[k][f] for f in fields) for k in old)
        rep["per_image"][f"{t} {new_d}"] = dict(n_images=len(old), max_abs_diff=mx, n_bit_identical=exact)

v3 = json.load(open(os.path.join(C.V3SC, "results.json")))["models"][C.rname(bb, L)]
parts = AN.parts_for(bb, L, SD)
new = AA.analyse(parts)


def cmp(o_new, o_old):
    rows = {}
    def put(k, a, b):
        if a is None and b is None:
            return
        rows[k] = dict(new=a, stored=b, diff=(None if a is None or b is None else round(abs(a - b), 6)))
    put("E mean", o_new["aligned"]["mean"], o_old["aligned"]["mean"])
    put("E ci lo", o_new["aligned"]["ci"][0], o_old["aligned"]["ci"][0]); put("E ci hi", o_new["aligned"]["ci"][1], o_old["aligned"]["ci"][1])
    for q in ("ratio_uni", "ratio_alt", "drop_ratio"):
        for th in (10, 20, 30):
            a = o_new.get(q, {}).get(th) if o_new.get(q) else None
            b = o_old.get(q, {}).get(str(th)) if o_old.get(q) else None
            put(f"{q}({th})", None if a is None else a["value"], None if b is None else b["value"])
            if a is not None and b is not None:
                put(f"{q}({th}) ci lo", a["ci"][0], b["ci"][0]); put(f"{q}({th}) ci hi", a["ci"][1], b["ci"][1])
    put("SR slope", o_new["sr_slope_vs_control180"]["slope_per_unit_sr"], o_old["sr_slope_vs_control180"]["slope_per_unit_sr"])
    for th in (20, 30):
        put(f"figure-mass loss({th})", o_new["drop_uni"][th]["mean"], o_old["drop_uni"][str(th)]["mean"])
    return rows


for m in ("r_inside", "r_core30", "f_core30"):
    for sub in ("all", "square", "triangle"):
        rep["summary"][f"{m} {sub}"] = cmp(new[m][sub], v3["results"][m][sub])
for k, s in enumerate(C.SIZES):
    ns = AA.analyse([parts[k]])
    rep["per_size"][str(s)] = cmp(ns["r_inside"]["all"], v3["per_size"][str(s)]["r_inside"]["all"])

all_img = max(v for r in rep["per_image"].values() for v in r["max_abs_diff"].values())
all_sum = max(r["diff"] for blk in list(rep["summary"].values()) + list(rep["per_size"].values()) for r in blk.values() if r["diff"] is not None)
rep["max_abs_diff_per_image_any_field"] = all_img
rep["max_abs_diff_summary_any_number"] = all_sum
rep["pass"] = bool(all_img < 1e-4 and all_sum < 5e-4)
json.dump(rep, open(os.path.join(SD, "sanity_report.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("readout", "max_abs_diff_per_image_any_field", "max_abs_diff_summary_any_number", "pass")}))
print(json.dumps(rep["summary"]["r_inside all"], indent=0))
for k, v in rep["per_image"].items():
    print(k, v["n_images"], "bit-identical", v["n_bit_identical"], "max", max(v["max_abs_diff"].values()))
