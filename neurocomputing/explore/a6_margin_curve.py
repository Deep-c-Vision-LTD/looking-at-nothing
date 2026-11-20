"""
A6 validity margin curve: exploratory analysis of REANALYSIS_RULE.md Section 6, item 7 (first part; RUN_PLAN.md Step 1,
S3). Written 18 November 2026 after the rule was fixed and public; not yet run on real data. No torch; saved scores only.

Rule text (Section 6, item 7, quoted verbatim):
  "A6 validity margin: valid-set composition and sign agreement among valid cells for margins 0 to 0.06 in steps of
  0.005 (known to be flat at 100%; Section 1.1, item 4); absolute CC, NSS and information gain (bits per fixation,
  baseline = log of the mean training map of `ab/maps.npy`) on held-out MIT1003 with the centre bias off, heads-only
  from the cached features."
  Accuracy criterion (Section 2.7): "the mean of the three unrounded seed CCs, rounded to 4 decimals, must be at least
  round(R4 mean - 0.03, 4), where the R4 mean is the mean of R4's three unrounded seed CCs on the same feature set,
  rounded to 4 decimals (`analyze_re.accuracy_valid`)."
  Wording (Section 7.5): "each of the 16 non-reference cells leaves the valid set at its own margin between 0.0020 and
  0.0446, and sign agreement among valid cells is 100% at every margin."
This script covers the margin curve only; the absolute CC, NSS and information gain with the centre bias off (the second
part of item 7, RUN_PLAN S4) need heads-only re-inference on the MIT1003 feature cache and are not part of it.

Implementation:
  - Cells: the 20 swap cells (4 feature sets x R1n, R1, R2, R3, R4); seed CCs from v3/gpu/full/scores/mit_<tag>.json
    ("CC", held-out MIT1003, 336 px, centre bias on); cell mean = mean of the three unrounded seed CCs.
  - E per cell: r(illusory) - r(control180), seeds averaged per configuration, then the 336 and 448 px values averaged
    (analyze_re.head_scores), with the WCU interval and WCR p (analyze_re.wild_cluster, 12 clusters).
  - Margins m = 0, 0.005, ..., 0.060 (13 values, built as round(k x 0.005, 3)): a non-R4 cell is valid at m if
    analyze_re.accuracy_valid(cell mean, R4 mean, margin=m) (the published rounding). Per margin: the valid non-R4 cells
    (composition), their number, and sign agreement with R4 on the same features: (a) point estimates on R4's side
    (the rule's "sign agreement"), (b) also WCU interval excluding 0 on R4's side (reported alongside).
  - Per cell: the CC gap to R4 (unrounded means; and with the published rounding), and the smallest grid margin at
    which it is valid.
Usage (after the rule is public): $PY -B explore/a6_margin_curve.py
Output: results/explore/A6_margin_curve.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
MARGINS = [round(k * 0.005, 3) for k in range(13)]


def seed_ccs(fs, ar):
    out = []
    for s in (0, 1, 2):
        d = X.load_json(os.path.join(X.SWAP_SCORES, f"mit_{fs}_{ar}_s{s}.json"))
        out.append(None if d is None else float(d["CC"]))
    return out


def same_side(ci, sign):
    """The interval excludes 0 on the side of `sign` (+1: lower bound > 0; -1: upper bound < 0)."""
    return bool((sign > 0 and ci[0] > 0) or (sign < 0 and ci[1] < 0))


def curve(cells, margins=MARGINS):
    """cells: {(fs, ar): dict(cc=mean CC, E=mean E, ci=WCU interval)}; R4 cells are the references.
    Returns per-margin composition and sign agreement, and per-cell exit information."""
    rows = []
    non = [k for k in cells if k[1] != "R4"]
    for m in margins:
        valid = [k for k in non if AR.accuracy_valid(cells[k]["cc"], cells[(k[0], "R4")]["cc"], margin=m)]
        agree = [k for k in valid if np.sign(cells[k]["E"]) == np.sign(cells[(k[0], "R4")]["E"])]
        agree_ci = [k for k in valid if same_side(cells[k]["ci"], np.sign(cells[(k[0], "R4")]["E"]))]
        rows.append(dict(margin=m, n_valid=len(valid), valid=[f"{a}|{b}" for a, b in valid],
                         n_sign_agree_point=len(agree), share_sign_agree_point=(len(agree) / len(valid)) if valid else None,
                         n_same_side_interval_excl0=len(agree_ci),
                         share_same_side_interval_excl0=(len(agree_ci) / len(valid)) if valid else None))
    exits = {}
    for k in non:
        r4 = cells[(k[0], "R4")]["cc"]
        first = next((m for m in margins if AR.accuracy_valid(cells[k]["cc"], r4, margin=m)), None)
        exits[f"{k[0]}|{k[1]}"] = dict(cc=cells[k]["cc"], cc_R4=r4, gap_unrounded=r4 - cells[k]["cc"],
                                      gap_published_rounding=round(round(r4, 4) - round(cells[k]["cc"], 4), 4),
                                      smallest_grid_margin_valid=first, valid_at_0p03=AR.accuracy_valid(cells[k]["cc"], r4))
    return rows, exits


def main():
    X.require_public()                                                # rule Section 8.1
    D = X.design()
    cells, seeds = {}, {}
    for fs in X.SWAP_FSETS:
        for ar in X.ARCHS:
            cc = seed_ccs(fs, ar)
            if any(c is None for c in cc):
                sys.exit(f"missing MIT1003 CC file for {fs} {ar}")
            v = AR.getter(AR.head_scores(X.SWAP_SCORES, f"{fs}_{ar}", sets=("stim_v3",)))
            w = X.lin(AR.per_config(D, v, "illusory", "control180"), D)
            cells[(fs, ar)] = dict(cc=float(np.mean(cc)), E=w["mean"], ci=w["ci"], p=w["p"])
            seeds[f"{fs}|{ar}"] = cc
    rows, exits = curve(cells)
    res = dict(margins=MARGINS, curve=rows, cells=exits, seed_ccs=seeds,
               E={f"{a}|{b}": dict(mean=c["E"], ci_wcu=c["ci"], p_wcr=c["p"]) for (a, b), c in cells.items()},
               definition="valid at margin m: round(mean CC, 4) >= round(round(R4 mean, 4) - m, 4); sign agreement = "
                          "point estimate on R4's side (also: WCU interval excluding 0 on R4's side)",
               not_included="absolute CC, NSS and information gain with the centre bias off (rule 6.7, second part)")
    X.write_json("A6_margin_curve.json", res, os.path.abspath(__file__),
                 inputs=["v3/gpu/full/scores/mit_<tag>.json", "v3/gpu/full/scores/stim_v3_<tag>.json"])


if __name__ == "__main__":
    main()
