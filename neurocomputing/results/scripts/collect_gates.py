"""
collect_gates.py: gathers the gate results and wall times of the fast CPU battery (18 November 2026) from the run logs
and check files into one JSON, results/fast_battery_gates.json, so that every number in results/FAST_BATTERY.md points
to a JSON. Parses text only; computes no statistic. Calls require_public() first (REANALYSIS_RULE.md Section 8.1).
Read only: results/run_logs/*, logs/*, check_cpu_*/eval_check.json, out/sanity.json. Writes only results/.
"""
import json, os, re, sys, time
sys.dont_write_bytecode = True
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RE, "scoring"))
import common_rigid as CR  # noqa: E402

n_hashed = CR.require_public(RE)
L = os.path.join(RE, "results", "run_logs")
out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), require_public_hashed_files=n_hashed)

# gate 0: hash and bit-identity checks (text log)
g0 = open(os.path.join(L, "gate0_bit_identity.txt")).read()
out["gate0_bit_identity"] = dict(source="results/run_logs/gate0_bit_identity.txt",
                                 pass_lines=[x for x in g0.splitlines() if x.startswith("PASS")],
                                 rule_sha256_files=re.findall(r"^([0-9a-f]{64})  (\S+REANALYSIS_RULE.sha256)$", g0, re.M),
                                 clone_refs=re.findall(r"^(looking-at-nothing/\.git/\S+): (.*)$", g0, re.M))
out["gate0_post_run"] = dict(source="results/run_logs/gate0_post_run_hashes.txt",
                             text=open(os.path.join(L, "gate0_post_run_hashes.txt")).read().strip())

# S1 sanity (JSON written by analyze_re.py)
s = json.load(open(os.path.join(RE, "out", "sanity.json")))
out["S1_sanity"] = dict(source="out/sanity.json", created_utc=s["created_utc"], checks=s["result"]["checks"],
                        failed=s["result"]["failed"], passed=s["result"]["passed"])

# renderer check-only (summary JSON printed by render_rigid.py --check-only)
t = open(os.path.join(L, "G0_render_check_only.log")).read()
summ = json.loads(t[t.find("\n{") + 1:t.rfind("\n}") + 2])
out["G0_render_check_only"] = dict(source="results/run_logs/G0_render_check_only.log",
                                   **{k: summ[k] for k in ("n_displays", "n_rigid", "n_mis_control180", "n_configs",
                                                           "n_refused", "checks", "n_failures")},
                                   worst={k: v for k, v in summ["worst"].items() if not isinstance(v, dict)})

# built-in pipeline checks of the scoring scripts (rule Section 3.2)
pc = {}
t = open(os.path.join(L, "R1_local_nulls.log")).read()
m = re.search(r"reproduction check on (\d+) stim_v3 displays: max \|difference\| = (\S+)", t)
pc["15 local nulls"] = dict(source="results/run_logs/R1_local_nulls.log (also logs/local_nulls_rigid.log)",
                            displays=int(m.group(1)), max_abs_diff=float(m.group(2)), tol=1e-9,
                            passed=float(m.group(2)) < 1e-9)
for lab, f in (("DeepGaze IIE", "R2_iie.log"), ("sweep dinov2", "R3_sweep_all.log"), ("sweep clip", "R3b_sweep_clip.log"),
               ("sweep siglip", "R3c_sweep_siglip.log")):
    t = open(os.path.join(L, f)).read()
    m = re.search(re.escape(lab) + r": pipeline check on (\d+) stim_v3 images, max \|r_inside - published\| per tag: (.*)", t)
    per = {a.split()[0]: float(a.split()[1]) for a in m.group(2).split(", ")}
    pc[lab] = dict(source=f"results/run_logs/{f} (also logs/score_rigid.log)", displays=int(m.group(1)), per_tag=per,
                   max_abs_diff=max(per.values()), tol=1e-4, passed=max(per.values()) < 1e-4)
for bb in ("dinov2", "siglip", "iie", "dinov2reg"):
    e = json.load(open(os.path.join(RE, f"check_cpu_{bb}", "eval_check.json")))
    pc[f"heads {bb}"] = dict(source=f"check_cpu_{bb}/eval_check.json", worst=e["worst"], values=e["n"], tol=e["tol"],
                             per_file=e["files"], passed=e["n"] > 0 and e["worst"] <= e["tol"])
out["pipeline_checks"] = pc

# wall times and exits (one process at a time)
rows = [r.split("\t") for r in open(os.path.join(L, "wall_times.tsv")).read().strip().splitlines()[1:]]
out["steps"] = [dict(label=r[0], start_utc=r[1], end_utc=r[2], wall_s=int(r[3]), exit=int(r[4]), command=r[5]) for r in rows]
out["metal_client_detected_in_any_step"] = any("METAL CLIENT HELD" in open(os.path.join(L, f)).read()
                                               for f in os.listdir(L) if f.endswith(".log"))
out["scoring_wall_s_R1_to_R5"] = sum(r["wall_s"] for r in out["steps"] if r["label"][:2] in ("R1", "R2", "R3", "R4", "R5"))
json.dump(out, open(os.path.join(RE, "results", "fast_battery_gates.json"), "w"), indent=1)
print("wrote results/fast_battery_gates.json;", "all pipeline checks passed:", all(v["passed"] for v in pc.values()),
      "; metal:", out["metal_client_detected_in_any_step"], "; R1-R5 wall s:", out["scoring_wall_s_R1_to_R5"])
