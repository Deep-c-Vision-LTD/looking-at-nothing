"""Kanizsa illusory - control180 effect by figure position (p0 = image centre), for every per-image score file in
<results>/v2/scores/raw. Writes <results>/v2/verify/position_check.json."""
import json, numpy as np, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as LP  # noqa: E402
V2 = str(LP.V2)
items = json.load(open(f"{LP.STIM_V3}/items.json"))
kcfg = sorted({it["config"] for it in items if it["illusion"] == "kanizsa"})
res = {}
print(f"{'model':22s} " + " ".join(f"{'p'+str(p):>7s}" for p in range(5)) + "   (illusory - control180, inside AOI)")
tags = sorted({os.path.basename(f).split(".json")[0] for f in glob.glob(f"{V2}/scores/raw/*.json*") if ".shard" not in f})
for tag in tags:
    raw = LP.load_json(f"{V2}/scores/raw/{tag}.json")
    by = []
    for p in range(5):
        cf = [c for c in kcfg if c.endswith(f"_p{p}")]
        by.append(np.mean([raw[f"{c}_illusory"]["r_inside"] - raw[f"{c}_control180"]["r_inside"] for c in cf]))
    res[tag] = by
    print(f"{tag:22s} " + " ".join(f"{v:+7.3f}" for v in by))
os.makedirs(f"{V2}/verify", exist_ok=True)
json.dump(res, open(f"{V2}/verify/position_check.json", "w"), indent=1)
