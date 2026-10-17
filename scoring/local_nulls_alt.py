"""
Local non-configural nulls on stim_v3 (Kanizsa: illusory, control180, control90, real, real_filled, mis10/20/30) and on
stim_alt (alt10/20/30, phases a and b). Null maps exactly as scoring/local_nulls.py (function maps() copied
verbatim): Gaussian-blurred (sigma 8, 16, 25, 40, 60 px) ink contrast |I - background| / 255, Sobel edge energy and
Harris corner response (blockSize 5, ksize 3, k 0.04, clipped at 0) of the grey image. Measures: common_alt.measures on
the float32 null map (the null maps are not quantised; v2 scored them in float32).
Reproduction check: r_inside on the stim_v3 images against <results>/v2/verify/local_nulls_raw.json (logged).
Writes <results>/v3/scores/raw_nulls.json ({null: {image name: measures}}).
"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, cv2
cv2.setNumThreads(3)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_alt as CA     # noqa: E402
SC = os.path.join(CA.V3, "scores"); os.makedirs(SC, exist_ok=True); os.makedirs(os.path.join(CA.V3, "logs"), exist_ok=True)

SIG = [8, 16, 25, 40, 60]
def maps(g, bg):                                   # verbatim from scoring/local_nulls.py
    ink = np.abs(g - bg) / 255.0
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    out = {}
    for s in SIG:
        out[f"ink_s{s}"] = cv2.GaussianBlur(ink, (0, 0), s)
        out[f"edge_s{s}"] = cv2.GaussianBlur(edge, (0, 0), s)
        out[f"corner_s{s}"] = cv2.GaussianBlur(har, (0, 0), s)
    return out


v2null = CA.LP.load_json(os.path.join(CA.V2, "verify", "local_nulls_raw.json"))
jobs = [(f"{c}_{cond}", os.path.join(CA.STIM_V3, "img", f"{c}_{cond}.png")) for c in CA.CONFIGS for cond in CA.V3_CONDS]
jobs += [(f"{c}_{cond}", os.path.join(CA.STIM_ALT, "img", f"{c}_{cond}.png")) for c in CA.CONFIGS for cond in CA.ALT_CONDS]
raw, dmax = {}, 0.0
t0 = time.time()
for k, (nm, path) in enumerate(jobs):
    cfg = nm.rsplit("_", 1)[0] if not nm.endswith("_real_filled") else nm[: -len("_real_filled")]
    assert cfg in CA.CONFIGS, nm
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = 255.0 if "_dark_" in nm else 0.0
    a = CA.aoi(cfg)
    for m, s in maps(g, bg).items():
        rec = CA.measures(s, a)
        if nm in v2null[m]:
            I, D = s[a["inside"]].sum(), s[a["ind"]].sum()                 # v2 arithmetic (float32 sums)
            dmax = max(dmax, abs(float(I / (I + D)) - v2null[m][nm]["r_inside"]), abs(rec["r_inside"] - v2null[m][nm]["r_inside"]))
        raw.setdefault(m, {})[nm] = rec
    if k % 100 == 0:
        print(k, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
json.dump(raw, open(os.path.join(SC, "raw_nulls.json"), "w"))
msg = f"local nulls: {len(jobs)} images x {len(raw)} nulls; max |r_inside - v2/verify/local_nulls_raw.json| = {dmax:.2e}; {time.time() - t0:.0f}s"
open(os.path.join(CA.V3, "logs", "local_nulls_alt.log"), "a").write(time.strftime("%H:%M:%S ") + msg + "\n")
print(msg)
