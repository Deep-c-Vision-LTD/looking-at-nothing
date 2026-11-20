"""
compare_msdb_rerun.py: compares the outputs of the second run of explore/a44_logmass.py, a12_kavsek.py and a13_region.py
(18 November 2026, after scores_rigid/DeepGaze_MSDB.json existed; DEVIATIONS.md D6) with the first run (moved to
results/explore/_previous/2026-11-18_before_msdb_rigid/). Computes no statistic: it reads the two JSON records of each
script and reports (a) whether every value outside DeepGaze MSDB's stim_rigid section is identical, (b) the code hashes
recorded by both runs, and (c) DeepGaze MSDB's new stim_rigid values (and IIE's, for reference) at 4 decimals.
Calls analyze_re.require_public() first anyway (rule Section 8.1). Written 18 November 2026; not a hashed file.
Output: results/explore/msdb_rerun_comparison.json (and a printed summary).
Run: nice -n 10 env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $PY -B -I results/scripts/compare_msdb_rerun.py
"""
import json, math, os, sys, time
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RE)
import analyze_re as A                                                    # noqa: E402

A.require_public()
NEW = os.path.join(RE, "results", "explore")
OLD = os.path.join(NEW, "_previous", "2026-11-18_before_msdb_rigid")
FILES = ("A4_4_logmass.json", "A12_kavsek.json", "A13_region.json")


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (present in one run only)")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} (length {len(a)} vs {len(b)})")
        for i, (u, v) in enumerate(zip(a, b)):
            diff(u, v, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}"[:200])


def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def cell(r):
    if not isinstance(r, dict) or r.get("mean") is None:
        return "n/a"
    return f"{f4(r['mean'])} [{f4(r['ci'][0])}, {f4(r['ci'][1])}], p {r.get('p'):.4g}"


res, lines = {}, []
for fn in FILES:
    o, n = json.load(open(os.path.join(OLD, fn))), json.load(open(os.path.join(NEW, fn)))
    ou, nu = o["result"]["units"], n["result"]["units"]
    d_other = []
    for u in sorted(set(ou) | set(nu)):
        if u not in ou or u not in nu:
            d_other.append(f"unit {u} present in one run only")
            continue
        a, b = dict(ou[u]), dict(nu[u])
        if u == "DeepGaze MSDB":
            a.pop("stim_rigid", None); b.pop("stim_rigid", None)
        diff(a, b, u, d_other)
    d_top = []
    diff({k: v for k, v in o["result"].items() if k != "units"}, {k: v for k, v in n["result"].items() if k != "units"},
         "result", d_top)
    d_code = []
    diff(o["code"], n["code"], "code", d_code)
    msdb_old, msdb_new = ou["DeepGaze MSDB"].get("stim_rigid"), nu["DeepGaze MSDB"].get("stim_rigid")
    res[fn] = dict(old_created=o["created_utc"], new_created=n["created_utc"], n_units=len(nu),
                   differences_outside_msdb_stim_rigid=d_other[:50], n_differences_outside_msdb_stim_rigid=len(d_other),
                   differences_top_level=d_top[:20], code_hash_differences=d_code, rule_sha256_equal=o["rule_sha256"] == n["rule_sha256"],
                   public_commit_equal=o["public_commit"] == n["public_commit"],
                   msdb_stim_rigid_old=msdb_old if isinstance(msdb_old, str) else "(values)",
                   msdb_stim_rigid_new=msdb_new, iie_stim_rigid=nu["DeepGaze IIE"].get("stim_rigid"),
                   units_still_not_scored=[u for u, r in nu.items() if isinstance(r.get("stim_rigid"), str)])
    lines.append(f"== {fn}: old {o['created_utc']}, new {n['created_utc']}, {len(nu)} units; differences outside MSDB "
                 f"stim_rigid: {len(d_other)}; top-level: {len(d_top)}; code hash differences: {len(d_code)}; "
                 f"still a string stim_rigid: {res[fn]['units_still_not_scored']}")
    for m in ("DeepGaze MSDB", "DeepGaze IIE"):
        sr = nu[m].get("stim_rigid")
        if isinstance(sr, str):
            lines.append(f"   {m}: {sr}"); continue
        if fn.startswith("A4_4"):
            for reg, per in sr.items():
                lines.append(f"   {m} {reg}: " + "; ".join(f"phi {phi} {cell(v)}" for phi, v in per.items()))
        else:
            key = "E_K" if fn.startswith("A12") else "E_N"
            for phi, v in sr.items():
                extra = f"; E_K minus gain {cell(v['E_K_minus_gain'])}" if key == "E_K" else ""
                lines.append(f"   {m} phi {phi}: {key} {cell(v[key])}{extra}")
out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=A.sha256(os.path.abspath(__file__)),
           old_folder=os.path.relpath(OLD, RE), files=res, summary=lines)
p = os.path.join(NEW, "msdb_rerun_comparison.json")
json.dump(out, open(p, "w"), indent=1)
print("\n".join(lines))
print("wrote", p)
