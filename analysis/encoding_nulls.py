"""Is the interior encoding index figure-specific, or does it reflect proximity to the changed inducers?
Recomputes E = sum<d_test, d_rc> / sum||d_rc||^2 from the saved per-token components on four token sets:
  int       = the primary set (pixel-identical across all 7 images, footprint inside the 'inside' AOI)
  int_far   = int tokens whose footprint is > 30 px from every inducer-disc pixel
  int_near  = int tokens within 30 px of an inducer disc
  ext_near  = pixel-identical tokens entirely OUTSIDE the figure (no inside or inducer pixel), with footprint distance to
              the inducer discs no larger than the largest such distance among this configuration's int tokens
              (a near-but-outside null at matched inducer distance).
Averaged over the 336 and 448 px inputs; cluster bootstrap of analysis/encoding/common_enc.Boot. Reads the token
components of analysis/encoding/run_vit.py; writes <results>/v2/verify/encoding_nulls.json."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "3"
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoding"))
from common_enc import configs, cells, Boot, footprints, TOK, VIT, H, W, V2  # noqa: E402
CFGS = configs(); NC = len(CFGS)
KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
BOOT = Boot(cells(CFGS, "kanizsa"))
SETS = ("int", "int_far", "int_near", "ext_near")
geo = {}
def token_geo(cf, G, P, S):
    key = (cf["config"], G, P, S)
    if key in geo: return geo[key]
    a = np.load(cf["aoi"]); ins, ind = a["inside"], a["ind"]; fig = ins | ind
    dI = distance_transform_edt(~ind)
    fp = footprints(G, G, P, P, H / S, W / S)
    full_in = np.zeros(G * G, bool); full_out = np.zeros(G * G, bool); mind = np.full(G * G, np.inf)
    for r in range(G):
        for c in range(G):
            y0, y1, x0, x1 = fp[r, c]; k = r * G + c
            if y1 <= y0 or x1 <= x0: continue
            full_in[k] = ins[y0:y1, x0:x1].all(); full_out[k] = not fig[y0:y1, x0:x1].any(); mind[k] = dI[y0:y1, x0:x1].min()
    geo[key] = (full_in, full_out, mind); return geo[key]
def per_config(net, S):
    nL = VIT[net]["n_hs"]
    E = {s: {t: np.full((NC, nL), np.nan) for t in ("illusory", "mis20", "control90")} for s in SETS}
    cnt = {s: np.zeros(NC, int) for s in SETS}
    for ci in KI:
        cf = CFGS[ci]; z = np.load(os.path.join(TOK, f"{net}_s{S}", cf["config"] + ".npz"))
        G, P, S_ = [int(v) for v in z["grid"]]; same = z["same"].all(0)
        full_in, full_out, mind = token_geo(cf, G, P, S_)
        intm = same & z["inside_full"]
        m = {"int": intm, "int_far": intm & (mind > 30), "int_near": intm & (mind <= 30)}
        dmax = mind[intm].max() if intm.any() else -1
        m["ext_near"] = same & full_out & (mind <= dmax)
        tests = list(z["tests"]); dot = z["dot"].astype(np.float64); nrc = z["nrc"].astype(np.float64)
        for s in SETS:
            cnt[s][ci] = m[s].sum()
            if not m[s].any(): continue
            R = nrc[:, m[s]].sum(-1)
            for t in ("illusory", "mis20", "control90"):
                D = dot[:, tests.index(t)][:, m[s]].sum(-1); ok = R > 0
                E[s][t][ci, ok] = D[ok] / R[ok]
    return E, cnt
out = {}
for net in ("dinov2", "clip", "siglip"):
    Es = [per_config(net, S) for S in (336, 448)]
    nL = VIT[net]["n_hs"]
    Eavg = {s: {t: np.nanmean(np.stack([Es[0][0][s][t], Es[1][0][s][t]]), 0) for t in Es[0][0][s]} for s in SETS}
    cnts = {s: [int(np.median(Es[k][1][s][KI])) for k in (0, 1)] for s in SETS}
    nvalid = {s: [int(np.sum(Es[k][1][s][KI] > 0)) for k in (0, 1)] for s in SETS}
    print(f"\n== {net}: median tokens per config (336, 448): {cnts}; configs with >=1 token: {nvalid}")
    rows = {}
    for L in range(1, nL):
        row = {}
        for s in SETS:
            v = np.full(NC, np.nan); v[KI] = Eavg[s]["illusory"][KI, L]
            row[s] = BOOT.summary(v)
        rows[L] = row
        f = lambda s: f"{row[s]['mean']:+.3f} [{row[s]['ci'][0]:+.3f},{row[s]['ci'][1]:+.3f}]" if row[s]["mean"] is not None else "n/a"
        spec = row["int"]["ci"][0] is not None and row["ext_near"]["ci"][1] is not None and row["int"]["ci"][0] > row["ext_near"]["ci"][1]
        print(f"L{L:2d} int {f('int')}  int_far {f('int_far')}  int_near {f('int_near')}  ext_near {f('ext_near')}  int>ext_near:{spec}")
    out[net] = dict(counts_median=cnts, n_configs=nvalid, layers={L: {s: rows[L][s] for s in SETS} for L in rows})
os.makedirs(os.path.join(V2, "verify"), exist_ok=True)
json.dump(out, open(os.path.join(V2, "verify", "encoding_nulls.json"), "w"), indent=1, default=float)
