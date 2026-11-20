"""
A4.4 log mass: exploratory analysis of REANALYSIS_RULE.md Section 6, item 4. Written 18 November 2026 after the rule
was fixed and public; not yet run on real data.

Rule text (Section 6, item 4, quoted verbatim):
  "A4.4 log mass: each map is normalised to total mass 1 over the display (DeepGaze: the exponentiated log density;
  readouts: the sigmoid map divided by its sum; MSI-Net: the min-max map divided by its sum). For region R, m_R = mass in
  R; the quantity is the mean over configurations of log m_R(illusory) - log m_R(control180). Regions: inside (the
  inside AOI, identical across conditions); notch sectors (per condition: the analytic notch sector, that is the disc
  pixels whose direction from the inducer centre lies within half the notch angle of the notch direction, half-open,
  as `render_rigid.notch_geom`, minus ink); inducer ink (per condition: pixels whose value differs from the background
  by at least 128 grey levels, as `render_rigid.ink_mask`). "Pixel differs from the background" is not used, because
  LANCZOS ringing adds a faint halo up to about 3 px wide, and "disc minus ink" is not used, because the stim_v3
  renderer draws inducers about 0.38 px up and left of their nominal centres, which leaves a rim of uninked disc pixels
  outside the notch. On `stim_rigid` the shares f_inside, f_notch and f_ink are computed in the scoring pass
  (`common_rigid.measures_extra`), so no rigid map is stored."
  Intervals (Section 6, first paragraph): "WCU (12 clusters; ...) for linear quantities"; exploratory, unadjusted.

Implementation:
  - Displays: stim_v3 illusory and control180 of the 60 Kanizsa configurations (phi = 0). Regions per display from
    excommon.v3_regions: inside AOI; notch = render_rigid.notch_geom(phi 0, rotation 0 or 180) minus
    render_rigid.ink_mask(display, background); ink = render_rigid.ink_mask.
  - Map per display: the published-form map of mapsrc (DeepGaze: the saved density / max; MSI-Net: the 8-bit min-max
    PNG; readouts: q16 of the resized sigmoid map, i.e. the map from which every published f_ measure and the stim_rigid
    scoring-pass shares f_inside, f_notch, f_ink were computed; the 15 local nulls: their float32 maps). m_R = mass in R /
    total mass of the display (scale free, so "normalised to total mass 1" needs no separate step).
  - Per configuration: log m_R(illusory) - log m_R(control180). Readouts: log m_R averaged over seeds and over the 336
    and 448 px inputs per display (rule Section 2.2; the average of logs, so the order of averaging does not matter);
    also reported per size (seeds averaged). A zero mass gives log = NaN and the unit's region is "not computable".
  - Interval: WCU with the WCR p (analyze_re.wild_cluster, 12 clusters, 9,999 Webb draws, RandomState(2026)).
  - Pipeline check (mapsrc/excommon.Check): r_inside of every display's map against the published per-image value; a
    unit that fails is reported as "not computable (pipeline check failed)" and its values are withheld.
  - stim_rigid: per phi (10, 20, 30, 45), log f_R(rig illusory) - log f_R(rig control180) from the scoring-pass shares in
    scores_rigid/ (f_inside, f_notch, f_ink), for every unit already scored there; otherwise "not scored yet".
Models: DeepGaze IIE and MSDB, MSI-Net, the 15 local nulls, the 18 sweep readouts, the 20 swap cells (60 heads), the 2
registers cells (6 heads).
Usage (after the rule is public; CPU only):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a44_logmass.py
      [--groups deepgaze,msinet,nulls,sweep,swap,registers] [--no-rigid]
Output: results/explore/A4_4_logmass.json
"""
import argparse, os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

REGIONS = ("inside", "notch", "ink")
CONDS = ("illusory", "control180")
RIGID_KEY = {"inside": "f_inside", "notch": "f_notch", "ink": "f_ink"}


def region_index(cfgs):
    """{(cfg, cond): {region: flat indices}} for the stim_v3 displays."""
    out = {}
    for c in cfgs:
        for cond in CONDS:
            rg = X.v3_regions(c, cond)
            out[(c, cond)] = {r: X.idx(rg[r]) for r in REGIONS}
    return out


def make_fn(REG):
    def fn(ctx, pub, raw):
        d = ctx["disp"]
        tot = float(pub.sum())
        return {f"logm_{r}": X.safe_log(X.mass(pub, ix) / tot) if tot > 0 else float("nan")
                for r, ix in REG[(d["cfg"], d["cond"])].items()}
    return fn


def contrast(C, unit, D, key, size=None):
    pdv = C.per_display(unit, key, size)
    dKP = D.arr(lambda c: pdv[X.v3_name(c, "illusory")] - pdv[X.v3_name(c, "control180")])
    res = X.lin(dKP, D)
    res["n_configs_not_finite"] = int(np.sum(~np.isfinite(dKP)))
    return res


def summarise(C, K, groups, D, rigid=True):
    out = {}
    for g in groups:
        for u in X.group_units(g):
            if u not in C.units():
                continue
            rec = dict(group=g)
            if not K.unit_ok(g, u):
                rec.update(status="not computable (pipeline check failed or no published value to check against)")
                out[u] = rec
                continue
            rec["status"] = "ok"
            st = {}
            for r in REGIONS:
                key = f"logm_{r}"
                st[r] = dict(size_average=contrast(C, u, D, key))
                for s in C.sizes(u, key):
                    st[r][f"px{s}"] = contrast(C, u, D, key, size=s)
            rec["stim_v3"] = st
            if rigid and g == "msinet":
                rec["stim_rigid"] = "not part of the stim_rigid battery (MSI-Net is not scored on stim_rigid)"
            elif rigid:
                rr = {}
                for r in REGIONS:
                    f = RIGID_KEY[r]
                    pc = X.rigid_per_config(g, u, lambda a, b, f=f: X.safe_log(a.get(f)) - X.safe_log(b.get(f)), D)
                    if pc is None:
                        rr = "not scored yet (scores_rigid/ has no complete file for this unit)"
                        break
                    rr[r] = {str(phi): dict(X.lin(v, D), n_configs_not_finite=int(np.sum(~np.isfinite(v))))
                             for phi, v in pc.items()}
                rec["stim_rigid"] = rr
            out[u] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="deepgaze,msinet,nulls,sweep,swap,registers")
    ap.add_argument("--no-rigid", action="store_true")
    a = ap.parse_args()
    X.require_public()                                                # rule Section 8.1, before anything is read
    import mapsrc as M
    groups = a.groups.split(",")
    D = X.design()
    X.log("building regions (inside, notch, ink) for 120 stim_v3 displays")
    REG = region_index(D.configs)
    disps = M.displays("stim_v3", CONDS)
    eng = M.Engine()
    C, K = eng.run(groups, disps, make_fn(REG))
    res = dict(definition="mean over the 60 configurations of log m_R(illusory) - log m_R(control180); m_R = share of the "
                          "display's map mass in region R; WCU interval, WCR p (12 clusters)",
               regions=dict(inside="inside AOI", notch="render_rigid.notch_geom (phi 0) minus ink",
                            ink="render_rigid.ink_mask (>= 128 grey levels from the background)"),
               pipeline_check=K.report(), units=summarise(C, K, groups, D, rigid=not a.no_rigid))
    X.write_json("A4_4_logmass.json", res, os.path.abspath(__file__),
                 inputs=["v2/stim_v3 (displays, AOIs)", "v2/scores/maps (DeepGaze)", "v2/scores/maps_msinet_png",
                         "v2/readout/cache/stim (swap features)", "v3/gpu/full/ckpt", "nn/registers/full/ckpt",
                         "sweep/<bb>/readout_L<L>.pt", "published per-image scores (checks)",
                         "neuro/reanalysis/scores_rigid (if scored)"],
                 extra_code=[os.path.join(X.HERE, "mapsrc.py")])


if __name__ == "__main__":
    main()
