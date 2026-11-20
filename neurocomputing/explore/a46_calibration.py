"""
A4.6 calibration robustness: exploratory analysis of REANALYSIS_RULE.md Section 6, item 6. Written 18 November 2026
after the rule was fixed and public; not yet run on real data.

Rule text (Section 6, item 6, quoted verbatim):
  "A4.6 calibration robustness (stim_v3 and stim_alt only): r recomputed (a) on rank-equalised maps (each pixel replaced
  by its rank within the display divided by the number of pixels, ties averaged) and (b) after transforms with T in
  {0.5, 0.75, 1, 1.5, 2}: DeepGaze density^(1/T); readouts exp(logit / T), with logits re-inferred (heads-only from the
  float16 stim_v3 feature cache where the cache exists; never inverted from the float16 maps). MSI-Net: rank
  equalisation only (decided now; its float maps are not regenerated). Robustness criterion (fixed now): a model or
  cell is "calibration-robust" if the point estimate of E keeps the sign of the published E under rank equalisation and
  under every T; otherwise "calibration-sensitive", and every statement of its sign carries that label. Whether each
  transformed interval excludes 0 is reported."
  Intervals (Section 6, first paragraph): WCU for linear quantities, BCa for ratios with the stability flag.

Implementation:
  - Displays: stim_v3 illusory, control180, control90, mis20 and stim_alt alt20a, alt20b of the 60 Kanizsa
    configurations (E needs only illusory and control180; E90, U20 and A20 are reported alongside).
  - r = M(inside) / (M(inside) + M(inducer discs)) on the transformed display map, as the published measure.
  - DeepGaze: the saved map is density / max (float16); density^(1/T) is computed as sal^(1/T) (r is scale free).
    Rank equalisation of the saved map (ties averaged).
  - Readouts (sweep, swap, registers): logits re-inferred on the CPU by mapsrc (swap on stim_v3: heads only from the
    float16 feature cache; otherwise a backbone pass). exp(logit / T) is computed in float64 at the readout's output
    resolution (s x s, after subtracting the display maximum, which leaves r unchanged) and resized to 1024 x 768 with
    cv2 bilinear interpolation, the published order (map at s x s, then resize). Rank equalisation is applied to the
    resized float32 sigmoid map of the re-inferred logits (no float16 step). Note: for readouts T = 1 gives
    exp(logit), the odds map, which is not the published sigmoid map; the rule's transform is applied as written.
  - MSI-Net: rank equalisation of the 8-bit PNG (stim_v3 only; MSI-Net has no stim_alt maps, so no A20).
  - "pub": r of the published-form map (mapsrc), whose agreement with the published per-image r_inside is the pipeline
    check; the published E is the mean of the published per-image r_inside values (read from the score files).
  - Per configuration: r averaged over seeds and sizes (readouts); E = r(illusory) - r(control180), E90 =
    r(illusory) - r(control90) with WCU interval and WCR p; U20 = E(mis20) / E and A20 = E(alt20) / E (alt20 = mean of
    the phases) as ratios of means with BCa, the stability flag and Fieller (excommon.ratio).
  - Criterion per unit: calibration-robust if sign(E) under rank equalisation and under every T equals the sign of the
    published E (rank only for MSI-Net); else calibration-sensitive. A unit whose pipeline check fails is "not
    computable" (values withheld).
Models: DeepGaze IIE and MSDB, MSI-Net, the 18 sweep readouts, the 20 swap cells, the 2 registers cells.
Usage (after the rule is public; CPU only; about 2 h without load):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a46_calibration.py
      [--groups deepgaze,msinet,sweep,swap,registers]
Output: results/explore/A4_6_calibration.json
"""
import argparse, os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

TS = (0.5, 0.75, 1.0, 1.5, 2.0)
V3_CONDS = ("illusory", "control180", "control90", "mis20")
ALT_CONDS = ("alt20a", "alt20b")


def tkey(T):
    return f"T{T:g}"


def transforms_for(group):
    return ["pub", "rank"] if group == "msinet" else ["pub", "rank"] + [tkey(T) for T in TS]


def deepgaze_r(sal, ins, ind):
    out = {"rank": X.r_from(X.rank_equalise(sal), ins, ind)}
    for T in TS:
        out[tkey(T)] = X.r_from(np.power(sal, 1.0 / T), ins, ind)
    return out


def readout_r(z, sig, ins, ind):
    """z, sig: s x s logit and float32 sigmoid. exp(z / T) at s x s in float64, then bilinear resize to the display."""
    import cv2
    full = cv2.resize(np.asarray(sig, np.float32), (X.W, X.H)).astype(np.float64)
    out = {"rank": X.r_from(X.rank_equalise(full), ins, ind)}
    zm = np.asarray(z, np.float64)
    zm = zm - zm.max()
    for T in TS:
        m = cv2.resize(np.exp(zm / T), (X.W, X.H))
        out[tkey(T)] = X.r_from(np.clip(m, 0, None), ins, ind)
    return out


def fn(ctx, pub, raw):
    d, g = ctx["disp"], ctx["group"]
    ins, ind = d["inside"], d["ind"]
    out = {"pub": X.r_from(pub, ins, ind)}
    if g == "deepgaze":
        out.update(deepgaze_r(raw["sal"], ins, ind))
    elif g == "msinet":
        out["rank"] = X.r_from(X.rank_equalise(raw["png"]), ins, ind)
    else:
        out.update(readout_r(raw["logit"], raw["sig"], ins, ind))
    return out


def stats_for(pd, D, have_alt):
    """pd: {display name: r}. E, E90 (WCU), U20, A20 (BCa ratios)."""
    v = lambda c, cond: pd[X.v3_name(c, cond)]
    E = D.arr(lambda c: v(c, "illusory") - v(c, "control180"))
    out = dict(E=X.lin(E, D), E90=X.lin(D.arr(lambda c: v(c, "illusory") - v(c, "control90")), D),
               U20=X.ratio(D.arr(lambda c: v(c, "mis20") - v(c, "control180")), E, D))
    if have_alt:
        out["A20"] = X.ratio(D.arr(lambda c: 0.5 * (v(c, "alt20a") + v(c, "alt20b")) - v(c, "control180")), E, D)
    else:
        out["A20"] = "not available (no stim_alt maps)"
    return out


def published_E(M, g, u, D):
    """Mean over configurations of the published per-image r_inside (illusory minus control180), averaged over the
    unit's seeds and sizes."""
    seeds, sizes = X.unit_sizes_seeds(g)
    acc = {}
    for sd in seeds:
        for s in sizes:
            pr = M.published_r(g, u, sd, s, "stim_v3")
            for c in D.configs:
                acc.setdefault(c, []).append(pr[X.v3_name(c, "illusory")] - pr[X.v3_name(c, "control180")])
    return float(D.arr(lambda c: np.mean(acc[c])).mean())


def criterion(E_pub, per_transform, group):
    keys = [k for k in transforms_for(group) if k != "pub"]
    sg = np.sign(E_pub)
    keeps = {k: bool(per_transform[k]["E"]["mean"] is not None and np.sign(per_transform[k]["E"]["mean"]) == sg)
             for k in keys}
    return ("calibration-robust" if all(keeps.values()) else "calibration-sensitive"), keeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="deepgaze,msinet,sweep,swap,registers")
    a = ap.parse_args()
    X.require_public()                                                # rule Section 8.1
    import mapsrc as M
    groups = a.groups.split(",")
    D = X.design()
    disps = M.displays("stim_v3", V3_CONDS) + M.displays("stim_alt", ALT_CONDS)
    C, K = M.Engine().run(groups, disps, fn)
    units = {}
    for g in groups:
        for u in X.group_units(g):
            if u not in C.units():
                continue
            if not K.unit_ok(g, u):
                units[u] = dict(group=g, status="not computable (pipeline check failed or no published value)")
                continue
            have_alt = g != "msinet"
            per = {}
            for k in transforms_for(g):
                pd = C.per_display(u, k)
                per[k] = stats_for(pd, D, have_alt)
                e = per[k]["E"]
                per[k]["E_interval_excludes_0"] = X.excl0(e["ci"])
                per[k]["E_interval_side"] = (None if not X.excl0(e["ci"]) else ("positive" if e["ci"][0] > 0 else "negative"))
            Ep = published_E(M, g, u, D)
            lab, keeps = criterion(Ep, per, g)
            units[u] = dict(group=g, status="ok", published_E=Ep, label=lab, sign_kept=keeps, transforms=per)
    res = dict(definition="r on rank-equalised maps and after T in {0.5, 0.75, 1, 1.5, 2} (DeepGaze density^(1/T); "
                          "readouts exp(logit/T), logits re-inferred; MSI-Net rank only); calibration-robust if the point "
                          "estimate of E keeps the sign of the published E under rank equalisation and every T",
               temperatures=list(TS), pipeline_check=K.report(), units=units,
               n_robust=sum(1 for v in units.values() if v.get("label") == "calibration-robust"),
               n_sensitive=sum(1 for v in units.values() if v.get("label") == "calibration-sensitive"),
               n_not_computable=sum(1 for v in units.values() if v.get("status") != "ok"))
    X.write_json("A4_6_calibration.json", res, os.path.abspath(__file__),
                 inputs=["v2/stim_v3", "v3/stim_alt", "v2/scores/maps", "v3/scores/maps", "v2/scores/maps_msinet_png",
                         "v2/readout/cache/stim", "v3/gpu/full/ckpt", "nn/registers/full/ckpt",
                         "sweep/<bb>/readout_L<L>.pt", "published per-image scores"],
                 extra_code=[os.path.join(X.HERE, "mapsrc.py")])


if __name__ == "__main__":
    main()
