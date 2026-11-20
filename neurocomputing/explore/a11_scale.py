"""
A11 effective spatial scale: exploratory analysis of REANALYSIS_RULE.md Section 6, item 9. Written 18 November 2026 after
the rule was fixed and public; not yet run on real data. No torch.

Rule text (Section 6, item 9, quoted verbatim):
  "A11 effective spatial scale (stim_v3 and stim_alt only): for every saved IIE and MSDB map on the 540 stim_v3 Kanizsa
  and 360 stim_alt displays, a non-negative least-squares fit m = a (G_sigma * corner) + b (G_sigma * ink) + c on the
  192 x 256 block-mean grid for sigma in {4, 6, 8, 12, 16, 20, 25, 32, 40, 50, 60} px; reported: the sigma with the
  largest pooled R2 and the R2 profile. The point-spread probe is not run (decided now); the Limitations say so."

Implementation (choices the rule leaves open are stated here, before any fit is run):
  - Maps: the saved DeepGaze IIE and MSDB maps (stim_v3: v2/scores/maps/<tag>/<name>.npz, 540 Kanizsa displays in the
    9 conditions; stim_alt: v3/scores/maps/<tag>/<name>.npz, 360 displays); sal = density / max (float16); each map is
    normalised to mean 1 over the display (a fixed multiple of mass 1, identical for every map, so the pooled R2 is
    the same as for mass 1 and NNLS works at unit scale).
  - Regressors: corner = the Harris response of the local corner null and ink = the ink contrast |I - background| / 255
    of the local ink null (exactly as v2/verify/local_nulls.py maps(): cv2.cornerHarris(I / 255, 5, 3, 0.04) clipped
    at 0), on the grey display read with cv2; G_sigma * x = cv2.GaussianBlur(x, (0, 0), sigma) at full resolution (the
    same call as the nulls). All three regressors and the map are block-averaged over 4 x 4 pixels to the 192 x 256
    grid. Each regressor column is divided by its maximum before the fit (NNLS is equivariant under positive column
    scaling, so the fitted map and R2 are unchanged).
  - Fit per map and sigma: scipy.optimize.nnls on [G*corner, G*ink, 1] (a, b, c >= 0). R2 = 1 - SSE / SST (SST about
    the map's own mean).
  - Pooled R2 per model and sigma = 1 - (sum of SSE over the 900 maps) / (sum of SST over the 900 maps); also by
    stimulus set, and the mean of the per-map R2. Reported: the sigma with the largest pooled R2 per model, the profile,
    and the share of maps whose best fit uses a = 0 or b = 0 at that sigma.
  - Check: r_inside of every saved map against the published per-image value (v3/scores/raw_v3maps for 480 of the 540
    stim_v3 displays, v3/scores/raw for stim_alt), tolerance 1e-6.
Usage (after the rule is public; CPU only, about 10 to 20 min):
  nice -n 10 env OMP_NUM_THREADS=4 $PY -B explore/a11_scale.py
Output: results/explore/A11_scale.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

SIGMAS = (4, 6, 8, 12, 16, 20, 25, 32, 40, 50, 60)
MODELS = ("DeepGaze_IIE", "DeepGaze_MSDB")
V3_CONDS = ("illusory", "control180", "control90", "real", "real_filled", "realC", "mis10", "mis20", "mis30")
ALT_CONDS = tuple(f"alt{t}{ph}" for t in (10, 20, 30) for ph in "ab")
BH, BW = 192, 256


def block_mean(x):
    x = np.asarray(x, np.float64)
    assert x.shape == (X.H, X.W), x.shape
    return x.reshape(BH, X.H // BH, BW, X.W // BW).mean((1, 3))


def regressors(g, bg, sigmas=SIGMAS):
    """{sigma: (corner, ink)} block-mean regressors from the grey display g (float32) and its background grey level."""
    import cv2
    ink = np.abs(g - bg) / 255.0
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    return {s: (block_mean(cv2.GaussianBlur(har, (0, 0), s)), block_mean(cv2.GaussianBlur(ink, (0, 0), s)))
            for s in sigmas}


def fit(y, corner, ink):
    """NNLS of y on [corner, ink, 1] (columns scaled to max 1). Returns sse, sst, coefficients (scaled columns)."""
    from scipy.optimize import nnls
    y = np.asarray(y, np.float64).ravel()
    cols = []
    for c in (corner, ink):
        c = np.asarray(c, np.float64).ravel()
        mx = c.max()
        cols.append(c / mx if mx > 0 else c)
    A = np.column_stack(cols + [np.ones_like(y)])
    coef, rnorm = nnls(A, y)
    sse = float(rnorm ** 2)
    sst = float(((y - y.mean()) ** 2).sum())
    return sse, sst, coef


def pooled(rows):
    """rows: list of (sse, sst). Pooled R2 = 1 - sum sse / sum sst."""
    s1 = sum(r[0] for r in rows); s2 = sum(r[1] for r in rows)
    return 1 - s1 / s2 if s2 > 0 else float("nan")


def profile(fits, sigmas=SIGMAS):
    """fits: {sigma: [(sse, sst, coef, stimset), ...]} -> pooled R2 profile, by set, mean per-map R2, best sigma."""
    prof = {str(s): pooled([(f[0], f[1]) for f in fits[s]]) for s in sigmas}
    best = max(sigmas, key=lambda s: prof[str(s)])
    by = {}
    for st in sorted({f[3] for s in sigmas for f in fits[s]}):
        p = {str(s): pooled([(f[0], f[1]) for f in fits[s] if f[3] == st]) for s in sigmas}
        by[st] = dict(profile=p, best_sigma=max(sigmas, key=lambda s: p[str(s)]))
    mean_r2 = {str(s): float(np.mean([1 - f[0] / f[1] for f in fits[s] if f[1] > 0])) for s in sigmas}
    fb = fits[best]
    return dict(pooled_R2_profile=prof, best_sigma=best, best_pooled_R2=prof[str(best)], by_stimset=by,
                mean_per_map_R2=mean_r2, n_maps=len(fb),
                share_a_zero_at_best=float(np.mean([f[2][0] == 0 for f in fb])),
                share_b_zero_at_best=float(np.mean([f[2][1] == 0 for f in fb])))


def jobs():
    D = X.design()
    out = []
    for c in D.configs:
        for cond in V3_CONDS:
            out.append(("stim_v3", c, f"{c}_{cond}", os.path.join(X.STIM_V3, "img", f"{c}_{cond}.png")))
        for cond in ALT_CONDS:
            out.append(("stim_alt", c, f"{c}_{cond}", os.path.join(X.STIM_ALT, "img", f"{c}_{cond}.png")))
    return out


def main():
    X.require_public()                                                # rule Section 8.1
    import cv2
    cv2.setNumThreads(min(4, int(os.environ["OMP_NUM_THREADS"])))
    J = jobs()
    assert len(J) == 900, len(J)
    pubs = {(m, st): X.load_json(os.path.join(X.ROOT, "v3", "scores", "raw_v3maps" if st == "stim_v3" else "raw",
                                              m + ".json")) for m in MODELS for st in ("stim_v3", "stim_alt")}
    K = X.Check()
    fits = {m: {s: [] for s in SIGMAS} for m in MODELS}
    aoi = {}
    for k, (st, c, nm, path) in enumerate(J):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        bg = 255.0 if X.kz_params()[c]["pol"] == "dark" else 0.0
        RG = regressors(g, bg)
        if (st, c) not in aoi:
            a = X.v3_aoi(c) if st == "stim_v3" else X.alt_aoi(c)
            aoi[(st, c)] = (X.idx(a["inside"]), X.idx(a["ind"]))
        for m in MODELS:
            base = os.path.join(X.ROOT, "v2" if st == "stim_v3" else "v3", "scores", "maps", m)
            sal = X.clip64(np.load(os.path.join(base, nm + ".npz"))["sal"])
            pr = pubs[(m, st)]
            K.add("deepgaze", m, X.r_from(sal, *aoi[(st, c)]), None if pr is None or nm not in pr else pr[nm]["r_inside"])
            y = block_mean(sal / sal.mean())
            for s in SIGMAS:
                sse, sst, coef = fit(y, *RG[s])
                fits[m][s].append((sse, sst, coef, st))
        if (k + 1) % 50 == 0:
            X.log(f"  {k + 1}/{len(J)} displays")
    chk = K.report()
    res = dict(definition="NNLS m = a (G_sigma * corner) + b (G_sigma * ink) + c on the 192 x 256 block-mean grid per "
                          "saved map; pooled R2 = 1 - sum SSE / sum SST over the maps of a model",
               sigmas=list(SIGMAS), pipeline_check=chk, point_spread_probe="not run (decided in the rule)",
               models={})
    for m in MODELS:
        name = m.replace("_", " ")
        if not K.unit_ok("deepgaze", m):
            res["models"][name] = dict(status="not computable (saved maps do not reproduce the published r_inside)")
        else:
            res["models"][name] = dict(status="ok", **profile(fits[m]))
    X.write_json("A11_scale.json", res, os.path.abspath(__file__),
                 inputs=["v2/scores/maps/DeepGaze_{IIE,MSDB}", "v3/scores/maps/DeepGaze_{IIE,MSDB}", "v2/stim_v3/img",
                         "v3/stim_alt/img", "v3/scores/raw_v3maps", "v3/scores/raw"])


if __name__ == "__main__":
    main()
