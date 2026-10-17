"""
Measures (common_alt.measures) on the saved stim_v3 maps of score_v3.py (<results>/v2/scores/maps/<tag>/<name>.npz,
float16 sal / max) for the 60 Kanizsa configurations of stim_v3 and the conditions illusory, control180, control90,
real, real_filled, mis10/20/30. No model is run. Writes <results>/v3/scores/raw_v3maps/<tag>.json and logs the
largest difference between r_inside from the float16 map and the float32 value in <results>/v2/scores/raw/<tag>.json.
"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_alt as CA     # noqa: E402
SC = os.path.join(CA.V3, "scores")

TAGS = ["DeepGaze_IIE", "DeepGaze_MSDB"] + [f"{bb}_L{L}_s{s}" for bb, Ls in (("dinov2", (2, 8, 12)), ("clip", (4, 12, 24)),
                                                                              ("siglip", (4, 8, 24))) for L in Ls for s in (336, 448)]
OUT = os.path.join(SC, "raw_v3maps"); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(CA.V3, "logs"), exist_ok=True)
LOG = os.path.join(CA.V3, "logs", "measure_v3maps.log")


def log(m):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")
    print(m, flush=True)


for tag in TAGS:
    p = os.path.join(OUT, tag + ".json")
    if os.path.exists(p):
        continue
    v2raw = CA.LP.load_json(os.path.join(CA.V2, "scores", "raw", tag + ".json"))
    res, dmax = {}, 0.0
    t0 = time.time()
    for cfg in CA.CONFIGS:
        a = CA.aoi(cfg)
        for cond in CA.V3_CONDS:
            nm = f"{cfg}_{cond}"
            sal = np.load(os.path.join(CA.V2, "scores", "maps", tag, nm + ".npz"))["sal"]
            rec = CA.measures(sal, a)
            dmax = max(dmax, abs(rec["r_inside"] - v2raw[nm]["r_inside"]))
            res[nm] = rec
    json.dump(res, open(p, "w"))
    log(f"{tag}: {len(res)} maps, max |r_inside(float16 map) - v2 raw| = {dmax:.2e}, {time.time() - t0:.0f}s")
