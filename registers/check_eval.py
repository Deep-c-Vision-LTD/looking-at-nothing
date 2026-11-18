"""Compare the rescored plain checkpoint (check/scores) with the published score files (v3/gpu/full/scores)."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); PUB = os.path.join(os.path.dirname(os.path.dirname(H)), "v3", "gpu", "full", "scores")
out = {}
for f in sorted(os.listdir(os.path.join(H, "check", "scores"))):
    a = json.load(open(os.path.join(H, "check", "scores", f))); b = json.load(open(os.path.join(PUB, f)))
    if f.startswith("mit_"):
        d = np.abs(np.array(a["cc_per_image"]) - np.array(b["cc_per_image"]))
        out[f] = dict(CC_new=a["CC"], CC_pub=b["CC"], max_abs_diff_cc_per_image=float(d.max()))
    else:
        mx, n = 0.0, 0
        for s in b:
            for m in b[s]:
                for k, v in b[s][m].items():
                    mx = max(mx, abs(a[s][m][k] - v)); n += 1
        out[f] = dict(values=n, max_abs_diff=mx)
json.dump(out, open(os.path.join(H, "check", "eval_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
