"""
verify_fast_battery.py: completeness check of the fast CPU battery on stim_rigid (RUN_PLAN.md Step 3, R1 to R5).
Written 18 November 2026. Computes no statistic: it counts displays, measures and non-finite values in every score
file the battery wrote, takes the md5 of each file, and writes results/fast_battery_outputs.json. Calls
require_public() first (REANALYSIS_RULE.md Section 8.1), as every script of this re-analysis must.
Read only: scores_rigid/, check_cpu_*/, out/sanity.json, stim_rigid/items.json. Writes only results/.
"""
import hashlib, json, math, os, sys, time
sys.dont_write_bytecode = True
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RE, "scoring"))
import common_rigid as CR  # noqa: E402

n_hashed = CR.require_public(RE)
NAMES = sorted(it["name"] for it in CR.RIGID_ITEMS)
EXTRA = ("r_N", "r_K", "f_polygon", "f_notch", "f_ink")
V3M = CR.MEASURES
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
SWEEP = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
SWAP = [f"{f}_{a}_s{s}" for f in ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")
        for a in ("R1n", "R1", "R2", "R3", "R4") for s in (0, 1, 2)]
REG = [f"dinov2reg_{L}_R4_s{s}" for L in ("L12", "L6") for s in (0, 1, 2)]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def nonfinite(vals):
    return sum(1 for v in vals if not (isinstance(v, (int, float)) and math.isfinite(v)))


def flat_file(p, need):
    """{name: {measure: v}} layout (scores_rigid/<tag>.json, one entry of nulls.json)."""
    d = p if isinstance(p, dict) else json.load(open(p))
    names = sorted(d)
    miss = [m for m in need if any(m not in d[n] for n in names)]
    return dict(n_displays=len(names), names_match=names == NAMES, missing_measures=miss,
                measures=sorted({m for n in names for m in d[n]}),
                nonfinite=nonfinite([v for n in names for v in d[n].values()]))


def head_file(p):
    d = json.load(open(p))
    out = dict(sizes=sorted(d))
    for z in d:
        ms = sorted(d[z])
        nm = {m: sorted(d[z][m]) for m in ms}
        out[z] = dict(measures=ms, all_names_match=all(v == NAMES for v in nm.values()),
                      missing_measures=[m for m in V3M + EXTRA if m not in d[z]],
                      nonfinite=nonfinite([v for m in ms for v in d[z][m].values()]))
    return out


res = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), require_public_hashed_files=n_hashed,
           n_items_expected=len(NAMES), files={}, missing=[])
SR = os.path.join(RE, "scores_rigid")
p = os.path.join(SR, "nulls.json")
if os.path.exists(p):
    d = json.load(open(p))
    res["files"]["scores_rigid/nulls.json"] = dict(md5=md5(p), bytes=os.path.getsize(p), nulls=sorted(d),
                                                   n_nulls=len(d), all_15=sorted(d) == sorted(NULLS),
                                                   per_null={k: flat_file(d[k], V3M + EXTRA) for k in sorted(d)})
else:
    res["missing"].append("scores_rigid/nulls.json")
tags = ["DeepGaze_IIE"] + [f"{b}_L{L}_s{s}" for b, Ls in SWEEP.items() for s in (336, 448) for L in Ls]
for t in tags:
    p = os.path.join(SR, t + ".json")
    if os.path.exists(p):
        res["files"][f"scores_rigid/{t}.json"] = dict(md5=md5(p), bytes=os.path.getsize(p),
                                                       **flat_file(p, V3M + EXTRA + ("r_inside_f32",)))
    else:
        res["missing"].append(f"scores_rigid/{t}.json")
for t in SWAP + REG:
    p = os.path.join(SR, "heads", "scores", f"stim_rigid_{t}.json")
    if os.path.exists(p):
        res["files"][f"scores_rigid/heads/scores/stim_rigid_{t}.json"] = dict(md5=md5(p), bytes=os.path.getsize(p),
                                                                              **head_file(p))
    else:
        res["missing"].append(f"scores_rigid/heads/scores/stim_rigid_{t}.json")
for extra in sorted(os.listdir(SR)) if os.path.isdir(SR) else []:
    if extra.endswith(".json") and f"scores_rigid/{extra}" not in res["files"]:
        res.setdefault("unexpected", []).append(f"scores_rigid/{extra}")
for c in ("check_cpu_dinov2", "check_cpu_siglip", "check_cpu_iie", "check_cpu_dinov2reg"):
    p = os.path.join(RE, c, "eval_check.json")
    if os.path.exists(p):
        e = json.load(open(p))
        res["files"][f"{c}/eval_check.json"] = dict(md5=md5(p), worst=e["worst"], n=e["n"], tol=e["tol"], pub=e["pub"],
                                                    per_file=e["files"])
p = os.path.join(RE, "out", "sanity.json")
if os.path.exists(p):
    s = json.load(open(p))["result"]
    res["files"]["out/sanity.json"] = dict(md5=md5(p), checks=s["checks"], failed=s["failed"], passed=s["passed"])
n_nf = 0
for k, v in res["files"].items():
    if "nonfinite" in v:
        n_nf += v["nonfinite"]
    for z in ("336", "448"):
        if z in v:
            n_nf += v[z]["nonfinite"]
    for q in v.get("per_null", {}).values():
        n_nf += q["nonfinite"]
res["total_nonfinite"] = n_nf
os.makedirs(os.path.join(RE, "results"), exist_ok=True)
json.dump(res, open(os.path.join(RE, "results", "fast_battery_outputs.json"), "w"), indent=1)
print(f"verified: {len(res['files'])} files, missing {len(res['missing'])}, non-finite values {n_nf}")
for m in res["missing"]:
    print("MISSING", m)
