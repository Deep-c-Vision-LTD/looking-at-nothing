"""Local non-configural nulls (blurred ink, edge energy, Harris corners at sigma 8, 16, 25, 40, 60 px) on stim_v3.
Question: which signs, misalignment ratios and real/realC/filled effects can purely local image structure produce?
Writes <results>/v2/verify/local_nulls_raw.json (per image: r_inside, r_eroded_all) and local_nulls_summary.json."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "3"
import json, numpy as np, cv2
cv2.setNumThreads(3)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as LP  # noqa: E402
V2, STIM = str(LP.V2), str(LP.STIM_V3)
os.makedirs(f"{V2}/verify", exist_ok=True)
items = json.load(open(f"{STIM}/items.json"))
SIG = [8, 16, 25, 40, 60]
def maps(g, bg):
    ink = np.abs(g - bg) / 255.0
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    out = {}
    for s in SIG:
        out[f"ink_s{s}"] = cv2.GaussianBlur(ink, (0, 0), s)
        out[f"edge_s{s}"] = cv2.GaussianBlur(edge, (0, 0), s)
        out[f"corner_s{s}"] = cv2.GaussianBlur(har, (0, 0), s)
    return out
raw = {}
for k, it in enumerate(items):
    g = cv2.imread(f"{STIM}/img/{it['name']}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = 255.0 if it["pol"] == "dark" else 0.0
    a = np.load(f"{STIM}/img/{it['aoi']}")
    ins, ind, iea = a["inside"], a["ind"], a["inside_eroded_all"]
    for m, s in maps(g, bg).items():
        I, D, E = s[ins].sum(), s[ind].sum(), s[iea].sum()
        raw.setdefault(m, {})[it["name"]] = dict(r_inside=float(I / (I + D)), r_eroded_all=float(E / (E + D)))
    if k % 100 == 0: print(k, flush=True)
json.dump(raw, open(f"{V2}/verify/local_nulls_raw.json", "w"))
kcfg = sorted({it["config"] for it in items if it["illusion"] == "kanizsa"})
ecfg = sorted({it["config"] for it in items if it["illusion"] == "ehrenstein"})
def e(r, cf, t, c, key="r_inside"): return np.mean([r[f"{x}_{t}"][key] - r[f"{x}_{c}"][key] for x in cf])
summ = {}
print(f"\n{'null':12s} {'K ill':>7s} {'sq':>7s} {'tri':>7s} {'r20':>6s} {'r20sq':>6s} {'r20tri':>6s} {'c90':>7s} {'real':>7s} {'realC':>7s} {'filled':>7s} | {'E ill':>7s} {'E fill':>7s} {'E realC':>7s} | {'K ill eroded_all':>8s}")
for m, r in raw.items():
    sq = [c for c in kcfg if c.startswith("square")]; tr = [c for c in kcfg if c.startswith("triangle")]
    d = dict(K_ill=e(r, kcfg, "illusory", "control180"), K_sq=e(r, sq, "illusory", "control180"), K_tri=e(r, tr, "illusory", "control180"),
             r20=e(r, kcfg, "mis20", "control180") / e(r, kcfg, "illusory", "control180"),
             r20sq=e(r, sq, "mis20", "control180") / e(r, sq, "illusory", "control180"),
             r20tri=e(r, tr, "mis20", "control180") / e(r, tr, "illusory", "control180"),
             c90=e(r, kcfg, "control90", "control180"), real=e(r, kcfg, "real", "control180"), realC=e(r, kcfg, "realC", "control180"),
             filled=e(r, kcfg, "real_filled", "control180"), E_ill=e(r, ecfg, "illusory", "control_end"),
             E_fill=e(r, ecfg, "real_filled", "control_end"), E_realC=e(r, ecfg, "realC", "control_end"),
             K_ill_ea=e(r, kcfg, "illusory", "control180", "r_eroded_all"))
    summ[m] = d
    print(f"{m:12s} {d['K_ill']:+7.3f} {d['K_sq']:+7.3f} {d['K_tri']:+7.3f} {d['r20']:6.2f} {d['r20sq']:6.2f} {d['r20tri']:6.2f} {d['c90']:+7.3f} {d['real']:+7.3f} {d['realC']:+7.3f} {d['filled']:+7.3f} | {d['E_ill']:+7.3f} {d['E_fill']:+7.3f} {d['E_realC']:+7.3f} | {d['K_ill_ea']:+8.3f}")
json.dump(summ, open(f"{V2}/verify/local_nulls_summary.json", "w"), indent=1)
