"""
A13 region variant: exploratory analysis of REANALYSIS_RULE.md Section 6, item 11. Written 18 November 2026 after the
rule was fixed and public; not yet run on real data.

Rule text (Section 6, item 11, quoted verbatim):
  "A13 region variant: the notch sectors are assigned to the figure: inside_N = P, inducers_N = D minus P, identical
  across conditions; r_N = M(P) / (M(P) + M(D minus P)); E_N and its support-ratio slope for every model and cell. On
  `stim_rigid`, r_N is computed in the scoring pass."
  Intervals (Section 6, first paragraph): "WCU (12 clusters; ...) for linear quantities, ... two-stage percentile
  (published method) for support-ratio slopes."

Implementation:
  - stim_v3 (phi = 0): P = render_rigid.polygon_exact(stimulus vertices), D = inducer-disc AOI; r_N on the published-form
    map of every display (mapsrc). E_N,i = r_N(illusory) - r_N(control180) per configuration (readouts: r_N averaged
    over seeds and sizes); WCU interval and WCR p.
  - Support-ratio slope: OLS slope of the per-configuration E_N on the support ratio (per unit support ratio), with the
    two-stage percentile interval and bootstrap p on the v2 indices (excommon.slope_pct, a copy of
    v3/scores/analyze_alt.slope without rounding).
  - stim_rigid: per phi, E_N from the scoring-pass r_N (scores_rigid/) with WCU and the slope; "not scored yet" until
    Step 3 has run.
  - Pipeline check as mapsrc (units that fail are withheld).
Models: DeepGaze IIE and MSDB, MSI-Net, the 15 local nulls, the 18 sweep readouts, the 20 swap cells, the 2 registers cells.
Usage (after the rule is public; CPU only):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a13_region.py
      [--groups deepgaze,msinet,nulls,sweep,swap,registers] [--no-rigid]
Output: results/explore/A13_region.json
"""
import argparse, os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

CONDS = ("illusory", "control180")


def make_fn(PD):
    def fn(ctx, pub, raw):
        return dict(r_N=X.r_N(X.pd_masses(pub, PD[ctx["disp"]["cfg"]])))
    return fn


def summarise(C, K, groups, D, rigid=True):
    out = {}
    for g in groups:
        for u in X.group_units(g):
            if u not in C.units():
                continue
            if not K.unit_ok(g, u):
                out[u] = dict(group=g, status="not computable (pipeline check failed or no published value)")
                continue
            pdv = C.per_display(u, "r_N")
            EN = D.arr(lambda c: pdv[X.v3_name(c, "illusory")] - pdv[X.v3_name(c, "control180")])
            rec = dict(group=g, status="ok", stim_v3=dict(E_N=X.lin(EN, D), sr_slope=X.slope_pct(EN, D)))
            if rigid and g == "msinet":
                rec["stim_rigid"] = "not part of the stim_rigid battery"
            elif rigid:
                pc = X.rigid_per_config(g, u, lambda a, b: a.get("r_N", float("nan")) - b.get("r_N", float("nan")), D)
                if pc is None:
                    rec["stim_rigid"] = "not scored yet (scores_rigid/ has no complete file for this unit)"
                else:
                    rec["stim_rigid"] = {str(phi): dict(E_N=X.lin(v, D),
                                                        sr_slope=X.slope_pct(v, D) if np.all(np.isfinite(v)) else None)
                                         for phi, v in pc.items()}
            out[u] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="deepgaze,msinet,nulls,sweep,swap,registers")
    ap.add_argument("--no-rigid", action="store_true")
    a = ap.parse_args()
    X.require_public()                                                # rule Section 8.1
    import mapsrc as M
    groups = a.groups.split(",")
    D = X.design()
    PD = X.pd_index_v3(D.configs)
    C, K = M.Engine().run(groups, M.displays("stim_v3", CONDS), make_fn(PD))
    res = dict(definition="E_N = r_N(illusory) - r_N(control180), r_N = M(P) / (M(P) + M(D minus P)) with AOIs identical "
                          "across conditions; WCU interval and WCR p (12 clusters); support-ratio slope with the two-stage "
                          "percentile interval",
               pipeline_check=K.report(), units=summarise(C, K, groups, D, rigid=not a.no_rigid))
    X.write_json("A13_region.json", res, os.path.abspath(__file__),
                 inputs=["v2/stim_v3 (AOIs)", "map sources of mapsrc.py", "neuro/reanalysis/scores_rigid (if scored)"],
                 extra_code=[os.path.join(X.HERE, "mapsrc.py")])


if __name__ == "__main__":
    main()
