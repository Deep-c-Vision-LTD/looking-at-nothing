"""
The 15 local nulls on the whole-display rotation displays (neuro/reanalysis/stim_rigid). Adapted from
v3/scores/local_nulls_alt.py: null maps verbatim (maps(), itself verbatim from v2/verify/local_nulls.py), measures of
common_rigid.measures (verbatim common_alt.measures) on the float32 null maps with each display's own rotated AOIs.
Reproduction check first: the same code on 8 stim_v3 displays must reproduce v3/scores/raw_nulls.json exactly
(|difference| < 1e-9 for every measure; REANALYSIS_RULE.md Section 3.2).
Writes neuro/reanalysis/scores_rigid/nulls.json ({null: {image name: measures}}). No torch import.
Changes of 18 November 2026 (rule review, before the rule was fixed): require_public() runs first (nothing is scored
before the rule is fixed and public); the exploratory AOI variants of common_rigid.measures_extra are stored with every
display; the set has 660 displays (480 rigid rotations, 180 orientation-matched controls mis<theta>_control180).
"""
import json, os, sys, time
sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
import numpy as np, cv2
cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common_rigid as CR     # noqa: E402

SIG = [8, 16, 25, 40, 60]
def maps(g, bg):                                   # verbatim from v2/verify/local_nulls.py
    ink = np.abs(g - bg) / 255.0
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    out = {}
    for s in SIG:
        out[f"ink_s{s}"] = cv2.GaussianBlur(ink, (0, 0), s)
        out[f"edge_s{s}"] = cv2.GaussianBlur(edge, (0, 0), s)
        out[f"corner_s{s}"] = cv2.GaussianBlur(har, (0, 0), s)
    return out


def check():
    """Recompute 8 published stim_v3 null scores (read only: v3/scores/raw_nulls.json, v3/stim_alt AOIs)."""
    pub = json.load(open(os.path.join(CR.V3, "scores", "raw_nulls.json")))
    names = ["square_sr30_dark_p0_illusory", "square_sr50_light_p2_control180", "triangle_sr70_dark_p4_mis20",
             "triangle_sr30_light_p1_control90", "square_sr70_dark_p3_mis10", "triangle_sr50_dark_p0_illusory",
             "square_sr50_dark_p4_mis30", "triangle_sr70_light_p2_control180"]
    worst = 0.0
    for nm in names:
        cfg = nm.rsplit("_", 1)[0]
        a = np.load(os.path.join(CR.V3, "stim_alt", "aoi", cfg + "_aoi.npz"))
        a = {k: a[k] for k in a.files}
        g = cv2.imread(os.path.join(CR.STIM_V3, "img", nm + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        bg = 255.0 if "_dark_" in nm else 0.0
        for m, s in maps(g, bg).items():
            rec = CR.measures(s, a)
            worst = max(worst, max(abs(rec[k] - pub[m][nm][k]) for k in rec))
    return worst


if __name__ == "__main__":
    CR.require_public()                            # REANALYSIS_RULE.md Section 8.1
    os.makedirs(CR.SCORES, exist_ok=True); os.makedirs(CR.LOGS, exist_ok=True)
    w = check()
    print(f"reproduction check on 8 stim_v3 displays: max |difference| = {w:.2e}", flush=True)
    assert w < 1e-9, w
    raw, t0 = {}, time.time()
    for k, it in enumerate(CR.RIGID_ITEMS):
        g = cv2.imread(os.path.join(CR.STIM_RIGID, "img", it["name"] + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        bg = 255.0 if it["pol"] == "dark" else 0.0
        a = CR.aoi(it["config"])
        for m, s in maps(g, bg).items():
            rec = CR.measures(s, a); rec.update(CR.measures_extra(s, a, it))
            raw.setdefault(m, {})[it["name"]] = rec
        if k % 100 == 0:
            print(k, len(CR.RIGID_ITEMS), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(raw, open(os.path.join(CR.SCORES, "nulls.json"), "w"))
    msg = f"local nulls on stim_rigid: {len(CR.RIGID_ITEMS)} images x {len(raw)} nulls; check {w:.2e}; {time.time() - t0:.0f}s"
    open(os.path.join(CR.LOGS, "local_nulls_rigid.log"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg)
