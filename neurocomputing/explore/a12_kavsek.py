"""
A12 Kavsek-style AOIs: exploratory analysis of REANALYSIS_RULE.md Section 6, item 10. Written 18 November 2026 after the
rule was fixed and public; not yet run on real data.

Rule text (Section 6, item 10, quoted verbatim):
  "A12 Kavsek-style AOIs: P = the whole contour area (`render_rigid.polygon_exact`, pixel-centre raster, applied
  identically to stim_v3 at phi = 0), D = inducer discs. Inner AOI: P for illusory and real displays, P minus D for
  control displays; r_K = M(inner) / (M(inner) + M(D minus inner)); E_K = r_K(illusory) - r_K(control180), reported
  with the gain that a uniform map would show from the AOI change alone (computed from the areas, per configuration)
  and E_K minus that gain. On `stim_rigid`, r_K is computed in the scoring pass."
  Intervals (Section 6, first paragraph): WCU (12 clusters) for linear quantities; exploratory, unadjusted.

Implementation:
  - stim_v3 (phi = 0): P = render_rigid.polygon_exact(stimulus vertices), D = the inducer-disc AOI of the configuration;
    masses on the published-form map of every display (mapsrc; the map from which the published measures and the
    stim_rigid scoring-pass r_K were computed). r_K as excommon.r_K (identical to common_rigid.measures_extra).
  - Uniform gain per configuration g_i = |P| / (|P| + |D \\ P|) - |P \\ D| / (|P \\ D| + |D|) (pixel counts).
  - Per configuration: E_K,i = r_K(illusory) - r_K(control180) (readouts: r_K averaged over seeds and sizes) and
    E_K,i - g_i; both with WCU interval and WCR p. The gain is a fixed property of the AOIs; its mean and range are
    reported (a constant has no sampling interval of its own beyond the configuration resampling, so it is given as the
    mean over configurations).
  - stim_rigid: per phi, E_K from the scoring-pass r_K (scores_rigid/), with the gain recomputed from each configuration's
    rotated AOIs (polygon, ind) and E_K minus the gain; "not scored yet" until Step 3 has run.
  - Pipeline check as mapsrc (units that fail are withheld).
Models: DeepGaze IIE and MSDB, MSI-Net, the 15 local nulls, the 18 sweep readouts, the 20 swap cells, the 2 registers cells.
Usage (after the rule is public; CPU only):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a12_kavsek.py
      [--groups deepgaze,msinet,nulls,sweep,swap,registers] [--no-rigid]
Output: results/explore/A12_kavsek.json
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
        d = ctx["disp"]
        return dict(r_K=X.r_K(X.pd_masses(pub, PD[d["cfg"]]), d["cond"]))
    return fn


def gains_v3(PD, D):
    def g(c):
        ix = PD[c]
        return X.uniform_gain(len(ix["P"]), len(ix["DnP"]), len(ix["PnD"]), len(ix["D"]))
    return D.arr(g)


def gains_rigid(D, phi):
    return D.arr(lambda c: X.uniform_gain(*X.rigid_pd_counts(c, phi)))


def gain_summary(G):
    return dict(mean=float(G.mean()), min=float(G.min()), max=float(G.max()))


def summarise(C, K, groups, D, PD, rigid=True):
    G0 = gains_v3(PD, D)
    Gr = {}
    out = {}
    for g in groups:
        for u in X.group_units(g):
            if u not in C.units():
                continue
            if not K.unit_ok(g, u):
                out[u] = dict(group=g, status="not computable (pipeline check failed or no published value)")
                continue
            pdv = C.per_display(u, "r_K")
            EK = D.arr(lambda c: pdv[X.v3_name(c, "illusory")] - pdv[X.v3_name(c, "control180")])
            rec = dict(group=g, status="ok", stim_v3=dict(E_K=X.lin(EK, D), uniform_gain=gain_summary(G0),
                                                           E_K_minus_gain=X.lin(EK - G0, D)))
            if rigid and g == "msinet":
                rec["stim_rigid"] = "not part of the stim_rigid battery"
            elif rigid:
                pc = X.rigid_per_config(g, u, lambda a, b: a.get("r_K", float("nan")) - b.get("r_K", float("nan")), D)
                if pc is None:
                    rec["stim_rigid"] = "not scored yet (scores_rigid/ has no complete file for this unit)"
                else:
                    rr = {}
                    for phi, EKr in pc.items():
                        if phi not in Gr:
                            Gr[phi] = gains_rigid(D, phi)
                        rr[str(phi)] = dict(E_K=X.lin(EKr, D), uniform_gain=gain_summary(Gr[phi]),
                                            E_K_minus_gain=X.lin(EKr - Gr[phi], D))
                    rec["stim_rigid"] = rr
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
    res = dict(definition="E_K = r_K(illusory) - r_K(control180); inner AOI P (illusory) or P minus D (control); "
                          "r_K = M(inner) / (M(inner) + M(D minus inner)); uniform-map gain from the areas per "
                          "configuration; WCU interval, WCR p (12 clusters)",
               pipeline_check=K.report(), units=summarise(C, K, groups, D, PD, rigid=not a.no_rigid))
    X.write_json("A12_kavsek.json", res, os.path.abspath(__file__),
                 inputs=["v2/stim_v3 (AOIs)", "map sources of mapsrc.py", "neuro/reanalysis/stim_rigid/aoi",
                         "neuro/reanalysis/scores_rigid (if scored)"],
                 extra_code=[os.path.join(X.HERE, "mapsrc.py")])


if __name__ == "__main__":
    main()
