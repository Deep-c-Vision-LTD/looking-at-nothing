import glob, hashlib, json, os, re, time
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
G = os.path.join(RE, "gpu")
def mt(p): return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p)))
print("== train logs")
for p in sorted(glob.glob(f"{G}/g3/logs/train_*.log") + glob.glob(f"{G}/a14/logs/train_*.log") + glob.glob(f"{G}/logs/train_*.log")):
    t = open(p, errors="replace").read()
    eps = sorted(set(int(x) for x in re.findall(r" epoch (\d+) ", t)))
    print(os.path.relpath(p, G), "epochs", eps, "headers", t.count("=== ") - t.count(" done"), "done", t.count("=== ") and (" done" in t.splitlines()[-1]),
          "resumed", "resumed" in t, "skip", "skip " in t, "stopping", "stopping after" in t, "batch400x", t.count("batch 400/400"))
print("== ckpt records")
for run, n_exp in (("g3", 24), ("a14", 30)):
    pts = sorted(glob.glob(f"{G}/{run}/ckpt/*.pt")); js = sorted(glob.glob(f"{G}/{run}/ckpt/*.json"))
    bad = []
    for j in js:
        d = json.load(open(j))
        if not (d["epochs"] == 6 and len(d["hist"]) == 6 and [h["epoch"] for h in d["hist"]] == list(range(6)) and d["bs"] == 32 and abs(d["lr"] - 5e-4) < 1e-12 and d["n_train"] == 12803):
            bad.append(os.path.basename(j))
    tr = sorted(glob.glob(f"{G}/{run}/done/train_*")); tr = [x for x in tr if not os.path.basename(x).startswith("step_")]
    fin = sorted(set(json.load(open(j))["finished"][:16] for j in js))
    print(run, "pt", len(pts), "json", len(js), "expected", n_exp, "bad records", bad, "train done markers", len(tr), "finished times", fin)
    print("   ckpt mtime range", min(mt(p) for p in pts), max(mt(p) for p in pts))
print("== score files")
for run in ("g3", "a14"):
    fs = sorted(glob.glob(f"{G}/{run}/scores/*.json"))
    print(run, len(fs), "mtime range", min(mt(p) for p in fs), max(mt(p) for p in fs))
reg = sorted(glob.glob(os.path.join(ROOT, "nn/registers/full/scores/*.json")))
print("registers R4", len(reg), "mtime range", min(mt(p) for p in reg), max(mt(p) for p in reg))
print("== markers")
print("DONE_both", os.path.exists(f"{G}/DONE_both"), open(f"{G}/DONE_both").read().strip(), mt(f"{G}/DONE_both"))
P6 = os.path.join(RE, "out/primary_P6.json")
print("primary_P6.json created_utc", json.load(open(P6))["created_utc"], "mtime", mt(P6))
print("eval final logs", mt(f"{G}/logs/eval_dinov2ms_final.log"), mt(f"{G}/logs/eval_dinov2reg_final.log"))
print("== sha256")
n = ok = 0
for line in open(os.path.join(RE, "REANALYSIS_RULE.sha256")):
    parts = line.split()
    if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
        n += 1
        fp = os.path.join(RE, parts[1])
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest() if os.path.exists(fp) else None
        if h == parts[0]: ok += 1
        else: print("  DIFFERS", parts[1])
print("hashed files", n, "unchanged", ok)
rule = hashlib.sha256(open(os.path.join(RE, "REANALYSIS_RULE.md"), "rb").read()).hexdigest()
print("REANALYSIS_RULE.md sha256", rule)
print("analyze_re.py sha256", hashlib.sha256(open(os.path.join(RE, "analyze_re.py"), "rb").read()).hexdigest())
print("== stimulus manifest")
man = json.load(open(os.path.join(RE, "stim_rigid", "manifest.json")))
bad = [f for f, m in man["files"].items() if hashlib.md5(open(os.path.join(RE, "stim_rigid", f), "rb").read()).hexdigest() != m]
print("stim_rigid files", len(man["files"]), "bad", len(bad))
print("== deviations")
t = open(os.path.join(RE, "DEVIATIONS.md")).read()
heads = re.findall(r"^### (D\d+)\. (.*)$", t, re.M)
for h in heads: print(" ", h[0], h[1][:110])
print("DEVIATIONS.md mtime", mt(os.path.join(RE, "DEVIATIONS.md")))
