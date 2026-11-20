"""
S13 block 12 minus the ink and edge nulls: exploratory analysis of REANALYSIS_RULE.md Section 6, item 14. Written
18 November 2026 after the rule was fixed and public; not yet run on real data. No torch; saved scores only.

Rule text (Section 6, item 14, quoted verbatim):
  "S13: for block 12 of the public DINOv2-B checkpoint (R4), the paired difference U20(block 12) - U20(null k) for each
  of the 10 ink and edge nulls (BCa, stability flag) and the ratio-free difference of sign-adjusted losses sign(E) L(20)
  (WCU); restricted to the public checkpoint, with the registers block-12 U20 (0.96 [0.79, 1.16]) printed alongside,
  not as a contrast."
  Definitions (Section 2.3): E(test) = r(test) - r(control180); L(theta) = r(illusory) - r(mis theta); U(theta) =
  E(mis theta) / E(illusory); ratios are ratios of means. Ratio rules (Section 4.2, 4.3): BCa from the two-stage pairs
  cluster bootstrap (v2 indices, 10,000 resamples, RandomState(2026)); a ratio is "unstable" if the |t| of its
  denominator (CR1 se) is below 4, and reportable only if the denominator's 95% WCU interval excludes 0.

Implementation:
  - Block 12 R4 = the swap cell dinov2_L12 R4 (facebook/dinov2-base hidden_states[12], three seeds averaged per
    configuration, then the 336 and 448 px values averaged; v3/gpu/full/scores), the cell whose U20 the rule compares
    with the registers block-12 U20 (Section 1.1, item 7: "plain 0.72 [0.62, 0.81]").
  - Nulls: ink_s{8,16,25,40,60} and edge_s{8,16,25,40,60} (v3/scores/raw_nulls.json).
  - U20 difference: BCa of U20(R4) - U20(null k) on the same (paired) resamples, p by inverting the BCa interval at 0;
    both denominators screened (|t| >= 4 and WCU interval excluding 0), Fieller for both single ratios
    (excommon.ratio_diff, as analyze_re.p4).
  - Ratio-free: per configuration d_i = sign(E_R4) L_R4,i(20) - sign(E_k) L_k,i(20), with sign(E) the sign of the
    unrounded mean E of that model (block 12 and every ink and edge null have negative E); WCU interval and WCR p.
  - Alongside: U20 of block 12 R4 and of each null (BCa, flags); the registers block-12 U20 read from
    nn/registers/registers_results.json (percentile interval as published), printed, not contrasted.
Usage (after the rule is public): $PY -B explore/s13_block12_nulls.py
Output: results/explore/S13_block12_vs_nulls.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
NULLS10 = [f"{k}_s{s}" for k in ("ink", "edge") for s in (8, 16, 25, 40, 60)]


def sign_loss(D, val):
    """sign(E) x [r(illusory) - r(mis20)] per configuration, sign from the unrounded mean E."""
    E = AR.per_config(D, val, "illusory", "control180")
    L = AR.per_config(D, val, "illusory", "mis20")
    sg = float(np.sign(E.mean()))
    return sg * L, sg, float(E.mean())


def s13(D, block12, nulls):
    n1, d1 = AR.ratio_parts(D, block12, "U20")
    L1, s1, E1 = sign_loss(D, block12)
    out = dict(block12=dict(E=E1, sign=s1, U20=X.ratio(n1, d1, D), sign_adjusted_L20=X.lin(L1, D)), vs_null={})
    for k, v in nulls.items():
        n2, d2 = AR.ratio_parts(D, v, "U20")
        L2, s2, E2 = sign_loss(D, v)
        out["vs_null"][k] = dict(null_E=E2, null_sign=s2, null_U20=X.ratio(n2, d2, D),
                                 U20_difference=X.ratio_diff(n1, d1, n2, d2, D),
                                 sign_adjusted_L20_difference=X.lin(L1 - L2, D))
    return out


def registers_u20():
    r = X.load_json(os.path.join(X.ROOT, "nn", "registers", "registers_results.json"))
    if r is None:
        return "registers_results.json not found"
    p = r["blocks"]["12"]["registers"]["pooled"]
    out = {}
    if "alt" in p and "ratio_uni20" in p["alt"]:
        out["alt.ratio_uni20"] = {k: p["alt"]["ratio_uni20"][k] for k in ("value", "ci")}
    if "section_B" in p and "ratio_20" in p["section_B"]:
        out["section_B.ratio_20"] = {k: p["section_B"]["ratio_20"][k] for k in ("value", "ci")}
    out["note"] = "registers block 12 R4 (17 November 2026), percentile interval as published; printed alongside, not a contrast"
    return out


def main():
    X.require_public()                                                # rule Section 8.1
    D = X.design()
    block12 = AR.getter(AR.head_scores(X.SWAP_SCORES, "dinov2_L12_R4"))
    N = AR.null_scores()
    nulls = {k: AR.getter(N[k]) for k in NULLS10}
    res = s13(D, block12, nulls)
    res["registers_block12_U20"] = registers_u20()
    res["definition"] = ("U20(block 12 R4) - U20(null k), BCa (paired), stability flag on both denominators; "
                         "sign(E) L(20) of block 12 minus that of null k, WCU; 10 ink and edge nulls")
    X.write_json("S13_block12_vs_nulls.json", res, os.path.abspath(__file__),
                 inputs=["v3/gpu/full/scores/stim_{v3,alt}_dinov2_L12_R4_s{0,1,2}.json", "v3/scores/raw_nulls.json",
                         "nn/registers/registers_results.json"])


if __name__ == "__main__":
    main()
