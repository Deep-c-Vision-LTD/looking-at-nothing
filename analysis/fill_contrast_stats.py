"""Kanizsa illusion index for DeepGaze IIE and three local nulls at fill contrast 25 / 50 / 75 %.
Reads <results>/v2/scores/raw/DeepGaze_IIE.json, <results>/v2/verify/fill_variants/iie_scores.json (scoring/
fill_contrast_iie.py), <results>/v2/verify/local_nulls_raw.json and the fill-variant images in stimuli/fill_variants.
Writes <results>/v2/verify/fill_contrast_stats.json."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v3 as A                                   # noqa: E402  (also puts the repository root on sys.path)
import paths as LP                                       # noqa: E402
V2, STIM = str(LP.V2), str(LP.STIM_V3)
FV = str(LP.FILL_VARIANTS)
iie = LP.load_json(f"{V2}/scores/raw/DeepGaze_IIE.json"); fv = json.load(open(f"{V2}/verify/fill_variants/iie_scores.json"))
for k, v in fv.items(): iie[k.replace("real_filled_c", "rf")] = v
iie_rf = {**iie}
for k in list(iie):
    if k.endswith("_real_filled"): iie_rf[k[:-len("real_filled")] + "rf50"] = iie[k]
out = {}
def index(sc, lvl):
    k = A.eff([sc], "kanizsa", "illusory", "control180", "r_inside")
    cells = A.DESIGN["kanizsa"][1]
    den = np.array([[sc[f"{c}_rf{lvl}"]["r_inside"] - sc[f"{c}_control180"]["r_inside"] for c in cell] for cell in cells])
    return A.summ("kanizsa", den, pct=False), A.ratio("kanizsa", k, den)
for lvl in (25, 50, 75):
    d, r = index(iie_rf, lvl); out[f"IIE_c{lvl}"] = dict(den=d, index=r)
    print(f"DeepGaze IIE fill contrast {lvl}%: denominator {d['mean']:+.3f} {d['ci']}  Kanizsa index {r['value']:.2f} {r['ci']}  (human 0.857)")
# local nulls on the same fill variants
raw = LP.load_json(f"{V2}/verify/local_nulls_raw.json")
items = [it for it in json.load(open(f"{STIM}/items.json")) if it["illusion"] == "kanizsa" and it["cond"] == "illusory"]
for m in ("corner_s25", "ink_s25", "edge_s25"):
    sig = int(m.split("_s")[1]); sc = dict(raw[m])
    for it in items:
        c = it["config"]; sc[f"{c}_rf50"] = raw[m][f"{c}_real_filled"]
        a = np.load(f"{STIM}/img/{it['aoi']}"); ins, ind = a["inside"], a["ind"]
        for lvl in (25, 75):
            g = cv2.imread(f"{FV}/img/{c}_real_filled_c{lvl}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
            bg = 255.0 if it["pol"] == "dark" else 0.0
            base = {"corner": np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None), "ink": np.abs(g - bg) / 255.0,
                    "edge": np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))}[m.split("_")[0]]
            s = cv2.GaussianBlur(base, (0, 0), sig); I, D = s[ins].sum(), s[ind].sum()
            sc[f"{c}_rf{lvl}"] = dict(r_inside=float(I / (I + D)))
    for lvl in (25, 50, 75):
        d, r = index(sc, lvl); out[f"{m}_c{lvl}"] = dict(den=d, index=r)
        print(f"{m:10s} fill contrast {lvl}%: denominator {d['mean']:+.3f} {d['ci']}  Kanizsa index {r['value']:.2f} {r['ci']}")
json.dump(out, open(f"{V2}/verify/fill_contrast_stats.json", "w"), indent=1)
