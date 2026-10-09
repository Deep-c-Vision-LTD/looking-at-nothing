"""
Score saliency models on a stimulus set: rel_inside for every image -> results/<set>/results_<model>.json,
then print illusory - control for every control condition.
Usage: python score_models.py kanizsa|ehrenstein ["DeepGaze IIE" "Edge-energy null" ...]   (default: all models)
MSI-Net is scored from maps written by msinet_maps.py (stimuli/<set>/maps_msinet).
"""
import json, os, sys, time
import numpy as np, cv2
from common import STIM, RESULTS, rel_inside
import models

SET = sys.argv[1]; D = STIM[SET]
items = json.load(open(os.path.join(D, "items.json")))
names = sys.argv[2:] or list(models.LOADERS) + ["MSI-Net"]
os.makedirs(os.path.join(RESULTS, SET), exist_ok=True)

for name in names:
    fn = None if name == "MSI-Net" else models.LOADERS[name]()
    path = os.path.join(RESULTS, SET, f"results_{name.replace(' ', '_')}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for i, it in enumerate(items):
        if it["name"] in res: continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
        a = np.load(os.path.join(D, "img", it["name"] + "_aoi.npz"))
        sal = (cv2.imread(os.path.join(D, "maps_msinet", it["name"] + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
               if fn is None else fn(rgb))
        res[it["name"]] = rel_inside(sal, a["inside"], a["ind"])
        if (i + 1) % 60 == 0:
            json.dump(res, open(path, "w")); print(f"  {name} {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(path, "w"))
    ctrls = sorted({it["cond"] for it in items if it["cond"].startswith("control")})
    d = {c: np.array([res[it["name"]] - res[it["name"].replace("_illusory", "_" + c)] for it in items if it["cond"] == "illusory"]) for c in ctrls}
    print(f"{name:18s} " + "  ".join(f"illusory-{c} {v.mean():+.3f} ({np.mean(v > 0) * 100:.0f}%)" for c, v in d.items()), flush=True)
