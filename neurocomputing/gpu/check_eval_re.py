"""
Sanity gates of REANALYSIS_RULE.md Section 3: compare scores written by eval_re.py into <check_dir>/scores with the
published score files (read only). Adapted from nn/registers/check_eval.py; compares every value present in the new
file (so a --stim-limit check run is compared on the images it scored).
  gate 3 (GPU, before G3 and A14): python check_eval_re.py <check_dir>                  (tolerance 1e-6)
  gate 2 (CPU heads, before scoring stim_rigid): python check_eval_re.py <check_dir> --tol 1e-4 [--pub <scores dir>]
Default published directory: v3/gpu/full/scores; registers heads: --pub <ROOT>/nn/registers/full/scores.
Exit 1 if any |difference| exceeds the tolerance or nothing was compared.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser()
ap.add_argument("check_dir"); ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--pub", default=os.path.join(ROOT, "v3", "gpu", "full", "scores"))
a = ap.parse_args()
out, worst, n_all = {}, 0.0, 0
for f in sorted(os.listdir(os.path.join(a.check_dir, "scores"))):
    new = json.load(open(os.path.join(a.check_dir, "scores", f))); pub = json.load(open(os.path.join(a.pub, f)))
    if f.startswith("mit_"):
        d = np.abs(np.array(new["cc_per_image"]) - np.array(pub["cc_per_image"][:len(new["cc_per_image"])]))
        out[f] = dict(CC_new=new["CC"], CC_pub=pub["CC"], n=len(d), max_abs_diff_cc_per_image=float(d.max()))
        worst = max(worst, float(d.max())); n_all += len(d)
    else:
        mx, n = 0.0, 0
        for s in new:
            for m in new[s]:
                for k, v in new[s][m].items():
                    mx = max(mx, abs(v - pub[s][m][k])); n += 1
        out[f] = dict(values=n, max_abs_diff=mx); worst = max(worst, mx); n_all += n
json.dump(dict(tol=a.tol, pub=a.pub, files=out, worst=worst, n=n_all), open(os.path.join(a.check_dir, "eval_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1)); print("worst", worst, "values", n_all, "tol", a.tol)
sys.exit(0 if (n_all > 0 and worst <= a.tol) else 1)
