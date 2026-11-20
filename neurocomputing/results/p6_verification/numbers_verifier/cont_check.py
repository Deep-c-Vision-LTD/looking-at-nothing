import json, sys
import numpy as np
V = json.load(open(sys.argv[1]))
import os
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
srcs = {"g3": os.path.join(RE, "gpu/g3/results_full.json"),
        "a14": os.path.join(RE, "gpu/a14/results_full.json"),
        "reg": os.path.join(ROOT, "nn/registers/full/results_full.json")}
C = {k: json.load(open(p)) for k, p in srcs.items()}
out = []
for c in V["cells"]:
    fs, ar = c["fset"], c["arch"]
    src = None
    for k in ("g3", "a14", "reg"):
        s = C[k].get("stim_v3", {}).get(fs, {}).get(ar)
        if s:
            src = k; break
    if src is None:
        out.append((fs, ar, "no continuity entry")); continue
    k = C[src]["stim_v3"][fs][ar]["kanizsa"]["illusory-control180"]
    a, s336, s448 = k["avg|r_inside"], k.get("336|r_inside"), k.get("448|r_inside")
    agree = np.sign(a["mean"]) == np.sign(c["E_recomputed"]) if a["mean"] != 0 else None
    d4 = abs(round(c["E_recomputed"], 4) - a["mean"])
    seeds_ok = all(abs(round(x, 4) - y) <= 1e-4 + 1e-12 for x, y in zip(c["seed_point_estimates"], a["seed_means"]))
    sz = (abs(round(c["size_E"]["336"], 4) - s336["mean"]) <= 1e-4 + 1e-12, abs(round(c["size_E"]["448"], 4) - s448["mean"]) <= 1e-4 + 1e-12) if s336 else None
    excl = a["ci"][0] > 0 or a["ci"][1] < 0
    out.append((fs, ar, src, f"WCU {c['E_recomputed']:+.4f} [{c['ci_own'][0]:+.4f},{c['ci_own'][1]:+.4f}] | pct {a['mean']:+.4f} {a['ci']} p {a['p']}",
                f"sign_agree={agree}", f"|round4 diff|={d4:.1e}", f"seeds_match={seeds_ok}", f"size_match={sz}",
                f"pct_excludes0={excl}", f"seed_means_pct={a['seed_means']}",
                f"336 {s336['mean'] if s336 else None} {s336['ci'] if s336 else None}; 448 {s448['mean'] if s448 else None} {s448['ci'] if s448 else None}"))
for o in out:
    print(" | ".join(map(str, o)))
