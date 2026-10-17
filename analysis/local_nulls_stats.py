"""Statistics for six local nulls with the survey's rules (misalignment ratio, illusion index), using the bootstrap of
analysis/analyze_v3.py. Reads <results>/v2/verify/local_nulls_raw.json, writes local_nulls_stats.json."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v3 as A                                   # noqa: E402  (also puts the repository root on sys.path)
import paths as LP                                       # noqa: E402
V2, STIM = str(LP.V2), str(LP.STIM_V3)
raw = LP.load_json(f"{V2}/verify/local_nulls_raw.json")
out = {}
for m in ["corner_s16", "corner_s25", "corner_s40", "edge_s25", "ink_s25", "ink_s40"]:
    sc = [raw[m]]
    k = A.eff(sc, "kanizsa", "illusory", "control180", "r_inside")
    kea = A.eff(sc, "kanizsa", "illusory", "control180", "r_eroded_all")
    m20 = A.eff(sc, "kanizsa", "mis20", "control180", "r_inside")
    den = A.eff(sc, "kanizsa", "real_filled", "control180", "r_inside")
    e = A.eff(sc, "ehrenstein", "illusory", "control_end", "r_inside")
    eden = A.eff(sc, "ehrenstein", "real_filled", "control_end", "r_inside")
    o = dict(K=A.summ("kanizsa", k), K_eroded_all=A.summ("kanizsa", kea), ratio20=A.ratio("kanizsa", m20, k),
             K_den=A.summ("kanizsa", den), K_index=A.ratio("kanizsa", k, den),
             E=A.summ("ehrenstein", e), E_den=A.summ("ehrenstein", eden), E_index=A.ratio("ehrenstein", e, eden))
    out[m] = o
    print(m, "K", o["K"]["mean"], o["K"]["ci"], "| K eroded_all", o["K_eroded_all"]["mean"], o["K_eroded_all"]["ci"],
          "| ratio20", o["ratio20"]["value"], o["ratio20"]["ci"], "| filled den", o["K_den"]["mean"], o["K_den"]["ci"],
          "| K index", o["K_index"]["value"], o["K_index"]["ci"], "| E", o["E"]["mean"], o["E"]["ci"], "| E den", o["E_den"]["mean"], o["E_den"]["ci"])
json.dump(out, open(f"{V2}/verify/local_nulls_stats.json", "w"), indent=1)
