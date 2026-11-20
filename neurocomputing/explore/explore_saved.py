"""
explore_saved.py: exploratory analyses from saved outputs, REANALYSIS_RULE.md Section 6 items 1, 2, 3, 5, 7, 14, 17,
18 and 20 (RUN_PLAN.md Step 1, S3). Written 18 November 2026 (about 23:45 UTC), after the rule was fixed and public, after
P1 to P5, R8, the other exploratory scripts and the test-time registers had been run and read (DEVIATIONS.md D9). No
torch; saved per-image scores only; the statistics engine is analyze_re.py (imported read only through excommon).

Rule text (Section 6, quoted verbatim):
  "Intervals: WCU (12 clusters; 6 for shape-specific; 4 for item 18) for linear quantities, BCa for ratios with the
  stability flag, two-stage percentile (published method) for support-ratio slopes."
  "1. A4.1 per size: E, U20, A20, D20 and L(20) at 336 px and at 448 px separately for all 20 swap cells, the registers
  cells and the new cells (stim_v3 and stim_alt)."
  "2. A4.2 90 degree control: E against control90 for the 20 swap cells, IIE, MSDB and the 15 nulls; the 90/180 ratio
  E90 / E180 (paired BCa, stability flag)."
  "3. A4.3 rotation profiles: U, A and L at 10, 20 and 30 degrees, losses as differences; drop ratio D(theta) =
  (r(illusory) - r(alt theta)) / (r(illusory) - r(mis theta)) only where the uniform-loss interval excludes 0;
  support-ratio slope of the per-configuration E (OLS on support ratio); per-seed values and the two-level seed
  bootstrap of `cbb/check_seed_bootstrap.py`; per-seed interval count (29 of 30 among valid cells, 59 of 60 overall;
  Critic fix 4)."
  "5. A4.5 core share: f_core30 = M(core30) / M(display), illusory minus control180, every model, null and cell."
  "7. A6 validity margin: valid-set composition and sign agreement among valid cells for margins 0 to 0.06 in steps of
  0.005 (known to be flat at 100%; Section 1.1, item 4); absolute CC, NSS and information gain (...) on held-out
  MIT1003 with the centre bias off, heads-only from the cached features."
  "14. S13: for block 12 of the public DINOv2-B checkpoint (R4), the paired difference U20(block 12) - U20(null k) for
  each of the 10 ink and edge nulls (BCa, stability flag) and the ratio-free difference of sign-adjusted losses sign(E)
  L(20) (WCU); (...)"
  "17. Below-criterion swap cells and the two excluded SigLIP-L R3 P4 contrasts, with their flags."
  "18. A10 (Critic fix 12): the aligned effect at each support ratio (0.3, 0.5, 0.7; 20 configurations each, 4
  clusters, WCU with full enumeration of 6^4 weight vectors) for all 18 sweep readouts, from the saved scores; labelled
  "4 clusters; approximate" with the smallest attainable p (0.0046) stated; a table in the supplement and a sentence in
  the Results next to the receptive-field test."
  "20. P2 companions: the pooled and triangle Delta(20) of IIE and MSDB (Section 5.2), and the square drop ratios with
  their flags."
  Definitions (Section 2.3): E(test) = r(test) - r(control180); L(theta) = r(illusory) - r(mis theta); U(theta) =
  E(mis theta) / E(illusory); A(theta) = E(alt theta) / E(illusory); ratios are ratios of means; alt = mean of phases a
  and b. Section 2.2: readouts are seed-averaged per configuration, then the 336 and 448 px values averaged.

Items already produced (not redone; recomputed here with their own code and compared leaf by leaf with the stored
outputs): item 7 first part (explore/a6_margin_curve.py -> results/explore/A6_margin_curve.json), item 14
(explore/s13_block12_nulls.py -> results/explore/S13_block12_vs_nulls.json), item 20 (results/scripts/p2_companions.py
-> results/p2_companions_exploratory.json). Item 7 second part (absolute metrics) is RUN_PLAN S4 (explore/a6_absolute.py).

Implementation choices (the rule leaves these open; fixed here before this script read any score):
  - Units. Item 1: the 20 swap cells and the 2 registers R4 cells (dinov2reg_L12, dinov2reg_L6); the new cells (G3, A14)
    are "not run" (no GO_AHEAD; rule 5.6). Item 2: as listed (20 swap cells, IIE, MSDB, 15 nulls). Item 3: the 20 swap
    cells and the 2 registers cells (the units with seeds and per-seed scores; the plan's A4.3 row reads "60 heads").
    Item 5: IIE, MSDB, the 15 nulls, the 20 swap cells, the 2 registers cells and the 18 sweep readouts (every unit with
    a stored f_core30; MSI-Net has no stored f_core30 and is reported as not computable from saved outputs). Item 17:
    the six non-R4 swap cells below the accuracy criterion (computed with analyze_re.accuracy_valid from the stored
    MIT1003 CCs; asserted to be the complement of analyze_re.P3_CELLS) and siglip_L24 R3 minus R4 for U20 and A20.
    Item 18: the 18 sweep readouts (single seed; per configuration the 336 and 448 px values averaged).
  - Item 1 per size: a readout's per-size value is its seed mean per configuration at that size
    (analyze_re.head_scores with one size); the size average is computed too, as a reproduction check against P3 and P4.
    D20 is pooled over shapes (12 clusters, as U20 and A20) and reported only where the L20 WCU interval excludes 0
    (rule 6.3 applied to the D20 of item 1 as well).
  - Item 3: losses as differences: L_uni(theta) = r(illusory) - r(mis theta) and the alternating loss L_alt(theta) =
    r(illusory) - r(alt theta), WCU. Slope: excommon.slope_pct (two-stage percentile, per unit support ratio). Per-seed
    values: for each seed (size-averaged) E and L_uni(10, 20, 30) with WCU and WCR p, E also with the published two-stage
    percentile interval (continuity with the known 29 of 30 and 59 of 60), U and A point values. Per-seed interval
    count: a seed interval counts if it excludes 0 on the side of R4's sign on the same features (R4's seed-averaged E);
    "among valid cells" = the 10 P3 cells x 3 seeds; "overall" = the 20 swap cells x 3 seeds (R4 included). Point-sign
    counts are given alongside (valid, below criterion, overall). Two-level seed bootstrap: the scheme of
    cbb/check_seed_bootstrap.py (clusters with replacement, positions within clusters, RandomState(2026), 10,000
    resamples; seeds with replacement per resample, RandomState(2027)), applied to E on the rule's measure r_inside, with
    percentile intervals as in that script; reproduced first on the script's own measures ("inside", "r_core30") and
    compared with cbb/seed_bootstrap.json (stored at 4 decimals).
  - Item 2 ratio: excommon.ratio (BCa on the v2 two-stage indices, Fieller, stability flag on the E180 denominator,
    reportable only if the E180 WCU interval excludes 0).
  - Item 17: E with WCU and WCR p (unadjusted, outside every family), U20 and A20 (excommon.ratio) and the
    readout-minus-R4 ratio differences (excommon.ratio_diff, as analyze_re.p4 without Holm), with the accuracy record
    (seed-mean CC, threshold, gap) and the stability flags.
  - Item 18: per support ratio a sub-design of the 20 configurations (analyze_re.Design on the filtered configuration
    metadata: 4 clusters = shape x polarity, 5 positions); analyze_re.wild_cluster enumerates all 6^4 = 1,296 weight
    vectors; smallest attainable p 6 / 1,296 = 0.00463.
  - Cross-checks against published values: per-size E, E90, per-seed means and per-seed percentile intervals, U, A, D
    and L at 10, 20, 30 of the swap cells against known_values.json (4 decimals); size-average E, U20 and A20 against
    out/primary_P1_P2_P3_P4.json (exact); sweep E against rule Section 1.1 item 5 (3 decimals).
Usage (after the rule is public): $PY -B -I explore/explore_saved.py
Outputs: results/explore/explore_saved.json and results/explore/explore_saved.md
"""
import os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
SWAP_CELLS = [(f, a) for f in AR.SWAP_FSETS for a in AR.ARCHS]
REG_CELLS = [(f, "R4") for f in X.REG_FSETS]
THETAS = (10, 20, 30)
SRS = (0.3, 0.5, 0.7)
NEW_SETS = ("dinov2reg_L6", "dinov2reg_L12", "dinov2ms_L7L11", "dinov2ms_L11")
SWEEP_UNITS = [(bb, L) for bb in ("dinov2", "clip", "siglip") for L in X.SWEEP_LAYERS[bb]]
KNOWN_SWEEP_E = {"dinov2": (-0.000, +0.051, +0.082, +0.035, -0.029, -0.098),       # rule Section 1.1 item 5
                 "clip": (-0.000, +0.041, +0.060, +0.018, -0.010, +0.014),
                 "siglip": (+0.020, +0.055, -0.021, +0.002, +0.047, +0.145)}
P_MIN_4 = 6 / 6 ** 4


# ====================================================================== loaders (read only)
def score_dir(fs):
    return X.REG_SCORES if fs.startswith("dinov2reg") else X.SWAP_SCORES


def cell_getter(fs, ar, seeds=(0, 1, 2), sizes=("336", "448"), meas=AR.MEAS, sets=("stim_v3", "stim_alt")):
    """value(config, cond) of a readout cell: seeds averaged per display, then the listed sizes averaged."""
    return AR.getter(AR.head_scores(score_dir(fs), f"{fs}_{ar}", seeds=seeds, sets=sets, sizes=sizes, meas=meas))


def sweep_getter(bb, L, meas=AR.MEAS, load=None):
    """Sweep readout (single seed): per display the 336 and 448 px values averaged, stim_v3 and stim_alt."""
    load = load or AR._load
    acc = {}
    for s in X.SIZES:
        for st in ("stim_v3", "stim_alt"):
            p = X.sweep_published_file(bb, L, s, st)
            if p is None:
                raise FileNotFoundError(f"sweep {bb} L{L} s{s} {st}")
            for k, x in load(p).items():
                acc.setdefault(k, []).append(x[meas] if isinstance(x, dict) else x)
    out = {}
    for k, v in acc.items():
        assert len(v) == 2, (bb, L, k, len(v))
        out[k] = float(np.mean(v))
    return AR.getter(out)


def mit_cc(fs, ar):
    return [float(AR._load(os.path.join(score_dir(fs), f"mit_{fs}_{ar}_s{s}.json"))["CC"]) for s in (0, 1, 2)]


# ====================================================================== statistics (engine of analyze_re)
def drop_ratio(num, den, D, label="uniform-loss"):
    """D(theta) = mean(num) / mean(den), num = r(ill) - r(alt), den = r(ill) - r(mis); reported only where the WCU
    interval of the uniform loss (den) excludes 0 (rule 6.3)."""
    w = X.lin(den, D)
    if not X.excl0(w["ci"]):
        return dict(reported=False, reason=f"{label} WCU interval includes 0 (rule 6.3: not computed)",
                    uniform_loss=dict(mean=w["mean"], ci=w["ci"]))
    r = X.ratio(num, den, D)
    r["reported"] = True
    return r


def a41_block(D, v):
    """E, L20, U20, A20, D20 of one value function (one size or the size average)."""
    E = AR.per_config(D, v, "illusory", "control180")
    L = AR.per_config(D, v, "illusory", "mis20")
    nU, dU = AR.ratio_parts(D, v, "U20")
    nA, dA = AR.ratio_parts(D, v, "A20")
    La = AR.per_config(D, v, "illusory", "alt20")
    return dict(E=X.lin(E, D), L20=X.lin(L, D), U20=X.ratio(nU, dU, D), A20=X.ratio(nA, dA, D),
                D20=drop_ratio(La, L, D))


def a42_block(D, v):
    E180 = AR.per_config(D, v, "illusory", "control180")
    E90 = AR.per_config(D, v, "illusory", "control90")
    return dict(E180=X.lin(E180, D), E90=X.lin(E90, D), ratio_E90_over_E180=X.ratio(E90, E180, D))


def a43_profile(D, v):
    E = AR.per_config(D, v, "illusory", "control180")
    out = dict(E=X.lin(E, D), slope_E=X.slope_pct(E, D))
    for th in THETAS:
        Lu = AR.per_config(D, v, "illusory", f"mis{th}")
        La = AR.per_config(D, v, "illusory", f"alt{th}")
        nU = AR.per_config(D, v, f"mis{th}", "control180")
        nA = AR.per_config(D, v, f"alt{th}", "control180")
        out[str(th)] = dict(L_uni=X.lin(Lu, D), L_alt=X.lin(La, D), U=X.ratio(nU, E, D), A=X.ratio(nA, E, D),
                            D=drop_ratio(La, Lu, D))
    return out


def a43_seed(D, v):
    """Per-seed values (one seed, size-averaged): E (WCU, WCR, published percentile), L_uni (WCU), U and A points."""
    E = AR.per_config(D, v, "illusory", "control180")
    out = dict(E=X.lin(E, D), E_percentile=AR.pct_summary(E, D.idx))
    for th in THETAS:
        Lu = AR.per_config(D, v, "illusory", f"mis{th}")
        nU = AR.per_config(D, v, f"mis{th}", "control180")
        nA = AR.per_config(D, v, f"alt{th}", "control180")
        out[str(th)] = dict(L_uni=X.lin(Lu, D), U=float(nU.mean() / E.mean()), A=float(nA.mean() / E.mean()))
    return out


def same_side(ci, sign):
    return bool((sign > 0 and ci[0] > 0) or (sign < 0 and ci[1] < 0))


def two_level(per, idx, B=AR.B_PAIRS, seed_rng=2027):
    """cbb/check_seed_bootstrap.py scheme on per-seed (n, K, P) arrays: plan = seeds averaged first, two-stage cluster
    resample; two-level = also seeds with replacement per resample (RandomState(2027)); percentile intervals."""
    per = np.asarray(per, float)
    ci, pi = idx
    n = per.shape[0]
    si = np.random.RandomState(seed_rng).randint(0, n, (B, n))
    plan = per.mean(0)[ci[:, :, None], pi].mean((1, 2))
    two = np.stack([per[si[:, j]][np.arange(B)[:, None, None], ci[:, :, None], pi] for j in range(n)]).mean(0).mean((1, 2))
    cp = np.percentile(plan, [2.5, 97.5]); ct = np.percentile(two, [2.5, 97.5])
    pos = lambda lo, hi: "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")
    return dict(mean=float(per.mean()), ci_plan=[float(cp[0]), float(cp[1])], ci_two_level=[float(ct[0]), float(ct[1])],
                position_plan=pos(*cp), position_two_level=pos(*ct), changed=pos(*cp) != pos(*ct), n_seeds=int(n))


def sr_design(meta, sr):
    return AR.Design({c: m for c, m in meta.items() if abs(float(m[1]) - sr) < 1e-9})


def a10_unit(Dsr, D, v):
    out = {}
    for sr, Ds in Dsr.items():
        r = AR.wild_cluster(AR.per_config(Ds, v, "illusory", "control180").ravel(), Ds.g)
        r["label"] = "4 clusters; approximate"
        # the engine's tie tolerance (1e-12, absolute) can miss the sign-constant vectors when |t| is very large, so a
        # p below the enumeration floor is a numerical artefact; flagged, the stored p is not altered
        r["p_below_enumeration_floor"] = bool(r["p"] < r["p_min"] - 1e-12)
        out[str(sr)] = r
    out["all_12_clusters"] = X.lin(AR.per_config(D, v, "illusory", "control180"), D)
    return out


# ====================================================================== cross-check helpers
def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, o


def compare(new, stored, skip=()):
    """Leaf-by-leaf comparison of a recomputed object with a stored JSON object (after excommon._clean)."""
    a = dict(leaves(X._clean(new))); b = dict(leaves(stored))
    keys = [k for k in sorted(set(a) | set(b)) if not any(s in k for s in skip)]
    n_num, worst, mism = 0, 0.0, []
    for k in keys:
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            n_num += 1
            d = abs(float(x) - float(y))
            worst = max(worst, d)
            if d != 0:
                mism.append((k, x, y))
        elif x != y:
            mism.append((k, x, y))
    return dict(n_leaves=len(keys), n_numeric=n_num, max_abs_diff=worst, n_mismatch=len(mism), first_mismatches=mism[:10],
                identical=not mism)


def close4(x, ref, tol=5.001e-5):
    """Agreement with a value published at 4 decimals."""
    return x is not None and ref is not None and abs(float(x) - float(ref)) <= tol


# ====================================================================== items
def item1(D, getters):
    res = {}
    for (fs, ar), g in getters.items():
        res[f"{fs}|{ar}"] = {f"px{s}": a41_block(D, g[s]) for s in ("336", "448")}
        res[f"{fs}|{ar}"]["size_average"] = a41_block(D, g["avg"])
    for fs in NEW_SETS:
        res.setdefault("new_cells", {})[fs] = "not run (G3 and A14 wait for GPU/GO_AHEAD; rule 5.6, 8.2)"
    res["registers_note"] = ("dinov2reg_L12 and dinov2reg_L6 R4 only (published 17 November 2026); the G3 registers "
                             "readouts R1n to R3 are new cells and not run")
    return res


def item2(D, units):
    return {name: a42_block(D, v) for name, v in units.items()}


def item3(D, getters, ref_sign, valid_cells):
    prof, seeds, tl = {}, {}, {}
    for (fs, ar), g in getters.items():
        key = f"{fs}|{ar}"
        prof[key] = a43_profile(D, g["avg"])
        seeds[key] = {str(s): a43_seed(D, g[f"seed{s}"]) for s in (0, 1, 2)}
        per = np.stack([AR.per_config(D, g[f"seed{s}"], "illusory", "control180") for s in (0, 1, 2)])
        tl[key] = two_level(per, D.idx)
    counts = {}
    groups = dict(valid=[f"{a}|{b}" for a, b in valid_cells],
                  below_criterion=[f"{a}|{b}" for a, b in SWAP_CELLS if b != "R4" and (a, b) not in valid_cells],
                  reference_R4=[f"{a}|R4" for a in AR.SWAP_FSETS],
                  overall=[f"{a}|{b}" for a, b in SWAP_CELLS])
    for gname, keys in groups.items():
        rec = dict(n_seed_runs=3 * len(keys), n_point_R4_sign=0, n_wcu_excl0_R4_side=0, n_percentile_excl0_R4_side=0,
                   not_counted_wcu=[], not_counted_percentile=[])
        for k in keys:
            sg = ref_sign[k.split("|")[0]]
            for s in ("0", "1", "2"):
                r = seeds[k][s]
                rec["n_point_R4_sign"] += int(np.sign(r["E"]["mean"]) == sg)
                if same_side(r["E"]["ci"], sg):
                    rec["n_wcu_excl0_R4_side"] += 1
                else:
                    rec["not_counted_wcu"].append(dict(cell=k, seed=int(s), mean=r["E"]["mean"], ci_wcu=r["E"]["ci"]))
                if same_side(r["E_percentile"]["ci"], sg):
                    rec["n_percentile_excl0_R4_side"] += 1
                else:
                    rec["not_counted_percentile"].append(dict(cell=k, seed=int(s), mean=r["E_percentile"]["mean"],
                                                              ci_percentile=r["E_percentile"]["ci"]))
        counts[gname] = rec
    return dict(profiles=prof, per_seed=seeds, per_seed_counts=counts, two_level_seed_bootstrap=tl,
                count_definition="a seed's interval counts if it excludes 0 on the side of R4's sign on the same "
                                 "features (R4's seed-averaged E); seeds size-averaged; WCU = rule interval, percentile = "
                                 "published two-stage interval (continuity with rule 1.1 item 2)")


def item5(D, units):
    return {name: X.lin(AR.per_config(D, v, "illusory", "control180"), D) for name, v in units.items()}


def item17(D, heads, cc, valid_cells, ref_sign):
    out = dict(below_criterion={}, siglip_L24_R3_excluded_P4={})
    for fs, ar in SWAP_CELLS:
        if ar == "R4" or (fs, ar) in valid_cells:
            continue
        v, v4 = heads[(fs, ar)], heads[(fs, "R4")]
        E = X.lin(AR.per_config(D, v, "illusory", "control180"), D)
        rec = dict(accuracy=dict(cc=cc[(fs, ar)], cc_R4=cc[(fs, "R4")],
                                 threshold=round(round(cc[(fs, "R4")], 4) - 0.03, 4),
                                 gap_unrounded=cc[(fs, "R4")] - cc[(fs, ar)], valid=False,
                                 flag="below the accuracy criterion (rule 2.7); outside P3 and P4"),
                   E=E, R4_sign=ref_sign[fs], point_on_R4_side=bool(np.sign(E["mean"]) == ref_sign[fs]),
                   wcu_excl0_on_R4_side=same_side(E["ci"], ref_sign[fs]))
        for meas in ("U20", "A20"):
            n1, d1 = AR.ratio_parts(D, v, meas)
            n2, d2 = AR.ratio_parts(D, v4, meas)
            rec[meas] = X.ratio(n1, d1, D)
            rec[f"{meas}_minus_R4"] = X.ratio_diff(n1, d1, n2, d2, D)
        out["below_criterion"][f"{fs}|{ar}"] = rec
    v, v4 = heads[("siglip_L24", "R3")], heads[("siglip_L24", "R4")]
    for meas in ("U20", "A20"):
        n1, d1 = AR.ratio_parts(D, v, meas)
        n2, d2 = AR.ratio_parts(D, v4, meas)
        r = X.ratio_diff(n1, d1, n2, d2, D)
        r["ratio_R3"] = X.ratio(n1, d1, D)
        r["ratio_R4"] = X.ratio(n2, d2, D)
        r["note"] = ("excluded from the P4 family by rule 5.4 (denominator percentile-approximate |t| about 3.0); "
                     "unadjusted, never counted")
        out["siglip_L24_R3_excluded_P4"][f"siglip_L24|R3|{meas}"] = r
    return out


def item18(meta, D, sweeps):
    Dsr = {sr: sr_design(meta, sr) for sr in SRS}
    for sr, Ds in Dsr.items():
        assert Ds.K == 4 and Ds.P == 5, (sr, Ds.K, Ds.P)
    res = {f"{bb}_L{L}": a10_unit(Dsr, D, v) for (bb, L), v in sweeps.items()}
    return dict(units=res, clusters_per_sr=[str(k) for k in Dsr[SRS[0]].keys],
                smallest_attainable_p=P_MIN_4,
                label="4 clusters; approximate (rule 6.18); WCU with all 6^4 = 1,296 Webb weight vectors; "
                      "smallest attainable two-sided p 6 / 1,296 = 0.00463")


# ====================================================================== cross-checks of items 7, 14, 20
def crosscheck_a6(D):
    import a6_margin_curve as A6
    stored = X.load_json(os.path.join(X.RESULTS, "A6_margin_curve.json"))
    cells, seeds = {}, {}
    for fs in AR.SWAP_FSETS:
        for ar in AR.ARCHS:
            ccs = A6.seed_ccs(fs, ar)
            v = AR.getter(AR.head_scores(X.SWAP_SCORES, f"{fs}_{ar}", sets=("stim_v3",)))
            w = X.lin(AR.per_config(D, v, "illusory", "control180"), D)
            cells[(fs, ar)] = dict(cc=float(np.mean(ccs)), E=w["mean"], ci=w["ci"], p=w["p"])
            seeds[f"{fs}|{ar}"] = ccs
    rows, exits = A6.curve(cells)
    new = dict(curve=rows, cells=exits, seed_ccs=seeds,
               E={f"{a}|{b}": dict(mean=c["E"], ci_wcu=c["ci"], p_wcr=c["p"]) for (a, b), c in cells.items()})
    st = {k: stored["result"][k] for k in ("curve", "cells", "seed_ccs", "E")}
    return dict(compare(new, st), stored_file="results/explore/A6_margin_curve.json", stored_created=stored["created_utc"],
                stored_code_sha256=stored["code"].get("explore/a6_margin_curve.py"),
                current_code_sha256=X.sha256_file(os.path.join(X.HERE, "a6_margin_curve.py")))


def crosscheck_s13(D):
    import s13_block12_nulls as S13
    stored = X.load_json(os.path.join(X.RESULTS, "S13_block12_vs_nulls.json"))
    block12 = AR.getter(AR.head_scores(X.SWAP_SCORES, "dinov2_L12_R4"))
    N = AR.null_scores()
    new = S13.s13(D, block12, {k: AR.getter(N[k]) for k in S13.NULLS10})
    st = {k: stored["result"][k] for k in ("block12", "vs_null")}
    return dict(compare(new, st), stored_file="results/explore/S13_block12_vs_nulls.json",
                stored_created=stored["created_utc"], stored_code_sha256=stored["code"].get("explore/s13_block12_nulls.py"),
                current_code_sha256=X.sha256_file(os.path.join(X.HERE, "s13_block12_nulls.py")))


def p2_companions(meta):
    """The computation of results/scripts/p2_companions.py (same calls, same order), for the cross-check."""
    D, Dsq, Dtri = AR.Design(meta), AR.Design(meta, "square"), AR.Design(meta, "triangle")
    models, nulls = AR.load_survey_and_nulls()
    units = list(models.items()) + [("null " + k, v) for k, v in nulls.items()]
    out = {}
    for name, v in units:
        rec = {}
        for lab, des in (("pooled_12_clusters", D), ("square_6_clusters", Dsq), ("triangle_6_clusters", Dtri)):
            rec["Delta20_" + lab] = AR.wild_cluster(AR.per_config(des, v, "alt20", "mis20").ravel(), des.g)
        num = AR.per_config(Dsq, v, "illusory", "alt20")
        den = AR.per_config(Dsq, v, "illusory", "mis20")
        w = AR.wild_cluster(den.ravel(), Dsq.g)
        tden = AR.t_den(den.ravel(), Dsq.g)
        passes = bool(tden >= AR.T_STABLE and (w["ci"][0] > 0 or w["ci"][1] < 0))
        rec["square_uniform_loss_denominator"] = dict(w, t_den=tden)
        rec["square_drop_ratio_D20"] = dict(
            AR.bca(lambda a, b: a / b, [num, den], Dsq.idx), fieller=AR.fieller(num, den, Dsq.g),
            stable=bool(tden >= AR.T_STABLE), denominator_excludes_zero=bool(w["ci"][0] > 0 or w["ci"][1] < 0),
            passes_screen=passes,
            flag=None if passes else ("unstable" if (w["ci"][0] > 0 or w["ci"][1] < 0) else "not reportable"),
            report=("yes" if passes else ("with the flag 'unstable'" if not name.startswith("null") else
                                          "no (null ratios only when the denominator passes the screen)")))
        out[name] = rec
    return out


def crosscheck_p2(meta):
    p = os.path.join(X.RE, "results", "p2_companions_exploratory.json")
    stored = X.load_json(p)
    new = p2_companions(meta)
    sc = os.path.join(X.RE, "results", "scripts", "p2_companions.py")
    return dict(compare(new, stored["result"]), stored_file="results/p2_companions_exploratory.json",
                stored_created=stored["created_utc"], stored_script_sha256=stored["script_sha256"],
                current_script_sha256=X.sha256_file(sc)), new


# ====================================================================== published-value checks
def known_checks(res1, res2, res3, known, primary):
    out = dict(per_size_E=[], E90=[], profiles=[], seed_means=[], seed_percentile_ci=[], size_average_vs_P3P4=[],
               L_mismatch_explained_by_rounded_components=[])
    sc = known["swap_cells"]
    for key, kv in sc.items():
        a = res1[key]
        for s in ("336", "448"):
            out["per_size_E"].append(close4(a[f"px{s}"]["E"]["mean"], kv[f"E_{s}"]))
        out["E90"].append(close4(res2[key]["E90"]["mean"], kv["E90"]))
        p = res3["profiles"][key]
        for th in THETAS:
            pairs = [(p[str(th)]["L_uni"]["mean"], kv.get(f"loss_uni{th}")),
                     (p[str(th)]["U"]["value"], (kv.get(f"ratio_uni{th}") or {}).get("value")),
                     (p[str(th)]["A"]["value"], (kv.get(f"ratio_alt{th}") or {}).get("value"))]
            if p[str(th)]["D"].get("reported"):
                pairs.append((p[str(th)]["D"]["value"], (kv.get(f"drop_ratio{th}") or {}).get("value")))
            out["profiles"] += [close4(a, b) for a, b in pairs if b is not None]
            ref = kv.get(f"loss_uni{th}")
            if ref is not None and not close4(p[str(th)]["L_uni"]["mean"], ref):
                # published loss = difference of the two effects rounded to 4 decimals (as rule D5 found for P1)?
                e, em = p["E"]["mean"], p[str(th)]["U"]["value"] * p["E"]["mean"]
                out["L_mismatch_explained_by_rounded_components"].append(
                    dict(cell=key, theta=th, L=p[str(th)]["L_uni"]["mean"], published=ref,
                         rounded_difference=round(round(e, 4) - round(em, 4), 4),
                         explained=abs(round(round(e, 4) - round(em, 4), 4) - ref) < 5e-9))
        for s in (0, 1, 2):
            r = res3["per_seed"][key][str(s)]
            out["seed_means"].append(close4(r["E"]["mean"], kv["E_seed_means"][s]))
            out["seed_percentile_ci"].append(close4(r["E_percentile"]["ci"][0], kv["E_seed_cis"][s][0]) and
                                             close4(r["E_percentile"]["ci"][1], kv["E_seed_cis"][s][1]))
    P3, P4 = primary["P3"], primary["P4"]["contrasts"]
    diffs = []
    for key in sc:
        fs, ar = key.split("|")
        e = res1[key]["size_average"]["E"]
        ref = P3["cells"].get(key) or (P3["reference"].get(fs) if ar == "R4" else None)
        if ref is not None:
            diffs += [abs(e["mean"] - ref["mean"]), abs(e["ci"][0] - ref["ci"][0]), abs(e["ci"][1] - ref["ci"][1]),
                      abs(e["p"] - ref["p"])]
        for meas in ("U20", "A20"):
            c = P4.get(f"{fs}|{ar}|{meas}")
            if c is not None:
                u = res1[key]["size_average"][meas]
                diffs += [abs(u["value"] - c["ratio_readout"]), abs(u["ci_bca"][0] - c["ratio_readout_bca"]["ci"][0]),
                          abs(u["ci_bca"][1] - c["ratio_readout_bca"]["ci"][1])]
            if ar == "R4":
                c = next((P4[k] for k in P4 if k.startswith(fs + "|") and k.endswith("|" + meas)), None)
                if c is not None:
                    u = res1[key]["size_average"][meas]
                    diffs += [abs(u["value"] - c["ratio_R4"]), abs(u["ci_bca"][0] - c["ratio_R4_bca"]["ci"][0]),
                              abs(u["ci_bca"][1] - c["ratio_R4_bca"]["ci"][1])]
    lm = out.pop("L_mismatch_explained_by_rounded_components")
    summ = {k: dict(n=len(v), n_agree=int(sum(bool(x) for x in v))) for k, v in out.items() if k != "size_average_vs_P3P4"}
    summ["L_mismatches"] = dict(n=len(lm), n_explained_by_rounded_components=int(sum(x["explained"] for x in lm)),
                                rows=lm)
    summ["size_average_vs_P3P4"] = dict(n_compared=len(diffs), max_abs_diff=max(diffs) if diffs else None)
    summ["tolerance_4_decimals"] = 5.001e-5
    return summ


def sweep_source_check(res18):
    """Rule 1.1 item 5 quotes the sweep E from sweep/<bb>/sweep_results.json (8 October 2026; earlier stimulus files,
    conditions 'illusory' and 'control'); recompute it from that file's per-image values and compare (3 decimals), and
    give its difference from the rule-measure E computed here (v3 q16 r_inside on stim_v3)."""
    rows, n_ok = [], 0
    for bb in ("dinov2", "clip", "siglip"):
        src = X.load_json(os.path.join(X.ROOT, "sweep", bb, "sweep_results.json"))
        for i, Ly in enumerate(X.SWEEP_LAYERS[bb]):
            es = []
            for s in X.SIZES:
                raw = src["kanizsa"][f"L{Ly}_s{s}"]["raw"]
                cf = sorted({k.rsplit("_", 1)[0] for k in raw if k.endswith("_illusory")})
                es.append(float(np.mean([raw[c + "_illusory"] - raw[c + "_control"] for c in cf])))
            e_src = float(np.mean(es))
            e_rule = res18["units"][f"{bb}_L{Ly}"]["all_12_clusters"]["mean"]
            ok = abs(e_src - KNOWN_SWEEP_E[bb][i]) <= 5.001e-4
            n_ok += int(ok)
            rows.append(dict(readout=f"{bb}_L{Ly}", E_source_file=e_src, published_3dp=KNOWN_SWEEP_E[bb][i],
                             source_reproduces_published=ok, E_rule_measure=e_rule, rule_minus_source=e_rule - e_src,
                             n_configurations=len(cf)))
    return dict(n=len(rows), n_source_reproduces_published=n_ok,
                max_abs_rule_minus_source=max(abs(r["rule_minus_source"]) for r in rows), rows=rows)


def crosscheck_cbb(D, getters_inside):
    """Reproduce cbb/seed_bootstrap.json (4 decimals) with two_level() on the script's own measures."""
    st = X.load_json(os.path.join(X.ROOT, "cbb", "seed_bootstrap.json"))
    worst, n, bad = 0.0, 0, []
    for (fs, ar), gs in getters_inside.items():
        rec = st["cells"].get(f"{fs} {ar}")
        if rec is None:
            continue
        for var, gl in gs.items():
            per = np.stack([AR.per_config(D, g, "illusory", "control180") for g in gl])
            r = two_level(per, D.idx)
            ref = rec[var]
            for a, b in ((r["mean"], ref["mean"]), (r["ci_plan"][0], ref["ci_plan"][0]), (r["ci_plan"][1], ref["ci_plan"][1]),
                         (r["ci_two_level"][0], ref["ci_two_level"][0]), (r["ci_two_level"][1], ref["ci_two_level"][1])):
                d = abs(a - b); worst = max(worst, d); n += 1
                if d > 5.001e-5:
                    bad.append((fs, ar, var, a, b))
    return dict(n_compared=n, max_abs_diff_vs_4_decimals=worst, n_above_5e_5=len(bad), first_bad=bad[:5],
                file="cbb/seed_bootstrap.json")


# ====================================================================== markdown
def f4(x):
    """4 decimals; values that would print as +/-0.0000 but are not 0 are given in scientific notation."""
    if x is None:
        return "n/a"
    return f"{x:+.2e}" if (x != 0 and abs(x) < 5e-5) else f"{x:+.4f}"


def ci4(c):
    return "n/a" if c is None or c[0] is None else f"[{f4(c[0])}, {f4(c[1])}]"


def rtxt(r):
    if not isinstance(r, dict):
        return str(r)
    if r.get("reported") is False:
        return "not computed (uniform-loss interval includes 0)"
    if r.get("value") is None:
        return "n/a"
    s = f"{r['value']:.4f} {ci4(r.get('ci_bca'))}"
    return s + (f" ({r['flag']})" if r.get("flag") else "")


def ltxt(r):
    return f"{f4(r['mean'])} {ci4(r['ci'])}, p {r['p']:.4g}"


def write_md(res, path, wall):
    L = []
    w = L.append
    w("# Exploratory analyses from saved outputs (`explore_saved.py`, RUN_PLAN S3)\n")
    w("**EXPLORATORY** (REANALYSIS_RULE.md Section 6: unadjusted, never decision-bearing). Rule items 1, 2, 3, 5, 17, 18 "
      "computed here; items 7 (first part), 14 and 20 were produced earlier and are recomputed and compared leaf by leaf. "
      "Linear quantities: mean, 95% WCU interval, WCR p (12 clusters; 4 for item 18). Ratios: value, BCa interval, flag "
      "(stability |t| < 4 or not reportable). All values unrounded in `explore_saved.json`; 4 decimals here.\n")
    w(f"- Run {res['run']['start_utc']} to {res['run']['end_utc']} ({wall:.1f} s compute). require_public: "
      f"{res['run']['hashed_files_verified']} hashed files and every stimulus verified.\n")
    cc = res["crosschecks"]
    w("## 0. Reproduction and cross-checks\n")
    for k in ("item7_A6_margin_curve", "item14_S13", "item20_P2_companions"):
        c = cc[k]
        w(f"- {k}: {c['n_numeric']} numeric leaves, max |difference| {c['max_abs_diff']}, mismatches {c['n_mismatch']} "
          f"(identical: {c['identical']}); stored {c['stored_file']} ({c['stored_created']}).")
    kc = cc["published_values"]
    w("- Published values (4 decimals, known_values.json): " + "; ".join(
        f"{k} {v['n_agree']} of {v['n']}" for k, v in kc.items() if isinstance(v, dict) and "n_agree" in v) + ".")
    w(f"- Size-average E, U20, A20 against out/primary_P1_P2_P3_P4.json: {kc['size_average_vs_P3P4']['n_compared']} values, "
      f"max |difference| {kc['size_average_vs_P3P4']['max_abs_diff']}.")
    cb = cc["cbb_seed_bootstrap"]
    w(f"- Two-level seed bootstrap reproduces cbb/seed_bootstrap.json: {cb['n_compared']} values, max |difference| from "
      f"the 4-decimal stored values {cb['max_abs_diff_vs_4_decimals']:.2e} ({cb['n_above_5e_5']} above 5e-5).")
    lm = kc["L_mismatches"]
    w(f"- Profile losses L_uni outside 4-decimal agreement: {lm['n']}, of which {lm['n_explained_by_rounded_components']} "
      "equal the difference of the two effects rounded to 4 decimals (the published loss is a difference of rounded "
      "values, as D5 found for P1); U, A and D agree.")
    sw = cc["sweep_E_vs_rule_1_1_item5"]
    sf = sw["source_file_check"]
    w(f"- Sweep E (rule measure, stim_v3) against rule 1.1 item 5 (3 decimals): {sw['n_agree']} of {sw['n']}. Rule 1.1 "
      f"item 5 comes from sweep/<bb>/sweep_results.json (8 October; earlier stimulus files and pipeline): recomputed from "
      f"that file {sf['n_source_reproduces_published']} of {sf['n']} agree; largest |rule-measure E minus source E| "
      f"{sf['max_abs_rule_minus_source']:.4f}.\n")

    w("## 1. A4.1 per size (rule 6.1): 20 swap cells and 2 registers cells\n")
    w("| cell | size | E [WCU], p | L20 [WCU] | U20 [BCa] | A20 [BCa] | D20 [BCa] |")
    w("|---|---|---|---|---|---|---|")
    for key, rec in res["item1_A4_1_per_size"].items():
        if not isinstance(rec, dict) or "px336" not in rec:
            continue
        for s in ("px336", "px448"):
            b = rec[s]
            w(f"| {key} | {s[2:]} | {ltxt(b['E'])} | {f4(b['L20']['mean'])} {ci4(b['L20']['ci'])} | {rtxt(b['U20'])} | "
              f"{rtxt(b['A20'])} | {rtxt(b['D20'])} |")
    w("\nNew cells (G3, A14): not run.\n")

    w("## 2. A4.2 90 degree control (rule 6.2)\n")
    w("| unit | E180 [WCU] | E90 [WCU], p | E90 / E180 [BCa] |")
    w("|---|---|---|---|")
    for name, r in res["item2_A4_2_control90"].items():
        w(f"| {name} | {f4(r['E180']['mean'])} {ci4(r['E180']['ci'])} | {ltxt(r['E90'])} | {rtxt(r['ratio_E90_over_E180'])} |")
    s2 = res["item2_summary"]
    w(f"\nE90 has the sign of E180 in {s2['n_same_sign']} of {s2['n']} units; E90 WCU interval excludes 0 in "
      f"{s2['n_e90_excl0']} of {s2['n']}.\n")

    r3 = res["item3_A4_3_profiles"]
    w("## 3. A4.3 rotation profiles, per-seed counts, two-level seed bootstrap (rule 6.3)\n")
    w("Per-seed interval counts (seed intervals excluding 0 on R4's side; seeds size-averaged):\n")
    w("| group | seed runs | point estimate with R4's sign | WCU interval on R4's side | published percentile interval on R4's side |")
    w("|---|---|---|---|---|")
    for g, c in r3["per_seed_counts"].items():
        w(f"| {g} | {c['n_seed_runs']} | {c['n_point_R4_sign']} | {c['n_wcu_excl0_R4_side']} | {c['n_percentile_excl0_R4_side']} |")
    for g in ("valid", "overall"):
        for x in r3["per_seed_counts"][g]["not_counted_wcu"]:
            w(f"- not counted under WCU ({g}): {x['cell']} seed {x['seed']}, {f4(x['mean'])} {ci4(x['ci_wcu'])}")
        for x in r3["per_seed_counts"][g]["not_counted_percentile"]:
            w(f"- not counted under the percentile interval ({g}): {x['cell']} seed {x['seed']}, {f4(x['mean'])} {ci4(x['ci_percentile'])}")
    w("\nProfiles (size average, seeds averaged): L_uni, U, A, D at 10 / 20 / 30 degrees; slope of E per unit support ratio "
      "(two-stage percentile); two-level seed bootstrap interval of E (percentile).\n")
    w("| cell | L_uni 10 / 20 / 30 | U 10 / 20 / 30 | A 10 / 20 / 30 | D 10 / 20 / 30 | slope of E [pct] | E two-level [pct] |")
    w("|---|---|---|---|---|---|---|")
    for key, p in r3["profiles"].items():
        lu = " / ".join(f4(p[str(t)]["L_uni"]["mean"]) + ("*" if X.excl0(p[str(t)]["L_uni"]["ci"]) else "") for t in THETAS)
        u = " / ".join((f"{p[str(t)]['U']['value']:.3f}" + ("u" if p[str(t)]["U"].get("flag") else "")) for t in THETAS)
        a = " / ".join((f"{p[str(t)]['A']['value']:.3f}" + ("u" if p[str(t)]["A"].get("flag") else "")) for t in THETAS)
        d = " / ".join((f"{p[str(t)]['D']['value']:.3f}" + ("u" if p[str(t)]["D"].get("flag") else ""))
                       if p[str(t)]["D"].get("reported") else "nc" for t in THETAS)
        sl = p["slope_E"]
        tl = r3["two_level_seed_bootstrap"][key]
        w(f"| {key} | {lu} | {u} | {a} | {d} | {f4(sl['slope_per_unit_sr'])} {ci4(sl['ci'])} | {ci4(tl['ci_two_level'])} "
          f"({tl['position_two_level']}{', changed' if tl['changed'] else ''}) |")
    w("\n`*` WCU interval excludes 0; `u` ratio flagged (unstable or not reportable); `nc` D not computed (uniform-loss "
      "interval includes 0). Intervals of U, A and D are in the JSON.\n")
    ntl = sum(1 for v in r3["two_level_seed_bootstrap"].values() if v["changed"])
    w(f"Two-level seed bootstrap: the interval of E changes position relative to 0 in {ntl} of "
      f"{len(r3['two_level_seed_bootstrap'])} cells.\n")

    w("## 4. A4.5 core share f_core30, illusory minus control180 (rule 6.5)\n")
    w("| unit | difference [WCU], p |")
    w("|---|---|")
    for name, r in res["item5_A4_5_core_share"].items():
        w(f"| {name} | {ltxt(r)} |" if isinstance(r, dict) and r.get("mean") is not None else f"| {name} | {r} |")
    w("")

    r17 = res["item17_below_criterion_and_excluded_P4"]
    w("## 5. Item 17: below-criterion swap cells and the two excluded SigLIP-L R3 P4 contrasts\n")
    w("| cell | CC (gap to R4) | E [WCU], p | R4 sign, same side | U20 [BCa] | U20 minus R4 [BCa], p | A20 [BCa] | A20 minus R4 [BCa], p |")
    w("|---|---|---|---|---|---|---|---|")
    for key, r in r17["below_criterion"].items():
        a = r["accuracy"]
        def dtx(x):
            return f"{f4(x['diff'])} {ci4(x['ci_bca'])}, p {x['p_bca']:.4g}" + (f" ({x['flag']})" if x.get("flag") else "")
        w(f"| {key} | {a['cc']:.4f} ({a['gap_unrounded']:.4f}) | {ltxt(r['E'])} | {r['R4_sign']:+d}, "
          f"{'yes' if r['wcu_excl0_on_R4_side'] else 'no'} | {rtxt(r['U20'])} | {dtx(r['U20_minus_R4'])} | {rtxt(r['A20'])} | "
          f"{dtx(r['A20_minus_R4'])} |")
    w("")
    for key, r in r17["siglip_L24_R3_excluded_P4"].items():
        w(f"- {key} minus R4: {f4(r['diff'])} {ci4(r['ci_bca'])}, BCa p {r['p_bca']:.4g}, min denominator |t| "
          f"{min(r['t_den_1'], r['t_den_2']):.3f}, flag: {r['flag']}; R3 ratio {rtxt(r['ratio_R3'])}, R4 ratio {rtxt(r['ratio_R4'])}.")
    w("")

    r18 = res["item18_A10_per_support_ratio"]
    w("## 6. A10 per support ratio (rule 6.18): 4 clusters; approximate\n")
    w(f"Aligned effect E = r(illusory) - r(control180) of the 18 sweep readouts (single seed, size average) at each support "
      f"ratio: 20 configurations, 4 clusters (shape x polarity), WCU with all 1,296 Webb weight vectors; smallest attainable "
      f"p {r18['smallest_attainable_p']:.5f}.\n")
    w("| readout | SR 0.3 [WCU], p | SR 0.5 [WCU], p | SR 0.7 [WCU], p | all 60 (12 clusters) [WCU], p |")
    w("|---|---|---|---|---|")
    for name, u in r18["units"].items():
        w(f"| {name} | {ltxt(u['0.3'])} | {ltxt(u['0.5'])} | {ltxt(u['0.7'])} | {ltxt(u['all_12_clusters'])} |")
    w("")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path, flush=True)


# ====================================================================== main
def main():
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_hashed = X.require_public()                                     # rule Section 8.1
    meta = AR.configs_meta()
    D = X.design()
    known = X.load_json(os.path.join(X.RE, "known_values.json"))
    primary = X.load_json(os.path.join(X.RE, "out", "primary_P1_P2_P3_P4.json"))["result"]

    # value functions
    getters = {}
    for fs, ar in SWAP_CELLS + REG_CELLS:
        g = dict(avg=cell_getter(fs, ar), **{s: cell_getter(fs, ar, sizes=(s,)) for s in ("336", "448")})
        for s in (0, 1, 2):
            g[f"seed{s}"] = cell_getter(fs, ar, seeds=(s,))
        getters[(fs, ar)] = g
    heads = {k: getters[k]["avg"] for k in SWAP_CELLS}
    cc = {k: float(np.mean(mit_cc(*k))) for k in SWAP_CELLS}
    valid = [k for k in SWAP_CELLS if k[1] != "R4" and AR.accuracy_valid(cc[k], cc[(k[0], "R4")])]
    assert sorted(valid) == sorted(AR.P3_CELLS), (valid, AR.P3_CELLS)
    ref_sign = {fs: int(np.sign(AR.per_config(D, heads[(fs, "R4")], "illusory", "control180").mean())) for fs in AR.SWAP_FSETS}
    models, nulls = AR.load_survey_and_nulls()
    X.log("loaded scores")

    res = dict(definitions=dict(E="r(illusory) - r(control180)", L="r(illusory) - r(mis theta)",
                                L_alt="r(illusory) - r(alt theta)", U="E(mis theta) / E(illusory)",
                                A="E(alt theta) / E(illusory)", D="(r(illusory) - r(alt theta)) / (r(illusory) - r(mis theta))",
                                measure="r_inside (rule 2.2)"))
    res["item1_A4_1_per_size"] = item1(D, getters)
    X.log("item 1 done")
    u2 = {f"{fs}|{ar}": heads[(fs, ar)] for fs, ar in SWAP_CELLS}
    u2.update(models)
    u2.update({"null " + k: v for k, v in nulls.items()})
    res["item2_A4_2_control90"] = item2(D, u2)
    r2 = res["item2_A4_2_control90"]
    res["item2_summary"] = dict(n=len(r2), n_same_sign=sum(int(np.sign(r["E90"]["mean"]) == np.sign(r["E180"]["mean"]))
                                                         for r in r2.values()),
                                n_e90_excl0=sum(int(X.excl0(r["E90"]["ci"])) for r in r2.values()))
    X.log("item 2 done")
    res["item3_A4_3_profiles"] = item3(D, getters, ref_sign, valid)
    X.log("item 3 done")
    u5 = {name: AR.getter(survey, "f_core30") for name, survey in
          (("DeepGaze IIE", AR.survey_scores("DeepGaze_IIE")), ("DeepGaze MSDB", AR.survey_scores("DeepGaze_MSDB")))}
    N = AR.null_scores()
    u5.update({"null " + k: AR.getter(N[k], "f_core30") for k in AR.NULLS})
    for fs, ar in SWAP_CELLS + REG_CELLS:
        u5[f"{fs}|{ar}"] = cell_getter(fs, ar, meas="f_core30")
    for bb, Ly in SWEEP_UNITS:
        u5[f"sweep {bb}_L{Ly}"] = sweep_getter(bb, Ly, meas="f_core30")
    res["item5_A4_5_core_share"] = item5(D, u5)
    res["item5_A4_5_core_share"]["MSI-Net"] = ("not computable from saved outputs (v2/scores/raw/MSI-Net.json stores no "
                                               "f_core30; its maps are 8-bit PNGs; MSI-Net is not in the rule 2.4 model list)")
    X.log("item 5 done")
    res["item17_below_criterion_and_excluded_P4"] = item17(D, heads, cc, valid, ref_sign)
    X.log("item 17 done")
    sweeps = {(bb, Ly): sweep_getter(bb, Ly) for bb, Ly in SWEEP_UNITS}
    res["item18_A10_per_support_ratio"] = item18(meta, D, sweeps)
    X.log("item 18 done")

    # cross-checks
    cks = dict(item7_A6_margin_curve=crosscheck_a6(D), item14_S13=crosscheck_s13(D))
    cks["item20_P2_companions"], p2new = crosscheck_p2(meta)
    cks["published_values"] = known_checks(res["item1_A4_1_per_size"], res["item2_A4_2_control90"],
                                           res["item3_A4_3_profiles"], known, primary)
    gi = {}
    for fs, ar in SWAP_CELLS:
        gi[(fs, ar)] = {var: [AR.getter(AR.head_scores(X.SWAP_SCORES, f"{fs}_{ar}", seeds=(s,), sets=("stim_v3",), meas=var))
                              for s in (0, 1, 2)] for var in ("inside", "r_core30")}
    cks["cbb_seed_bootstrap"] = crosscheck_cbb(D, gi)
    sw = []
    for bb, Ly in SWEEP_UNITS:
        e = res["item18_A10_per_support_ratio"]["units"][f"{bb}_L{Ly}"]["all_12_clusters"]["mean"]
        ref = KNOWN_SWEEP_E[bb][X.SWEEP_LAYERS[bb].index(Ly)]
        sw.append(abs(e - ref) <= 5.001e-4)
    cks["sweep_E_vs_rule_1_1_item5"] = dict(n=len(sw), n_agree=int(sum(sw)), tolerance=5.001e-4,
                                            source_file_check=sweep_source_check(res["item18_A10_per_support_ratio"]))
    a10 = res["item18_A10_per_support_ratio"]["units"]
    cks["a10_mean_of_sr_means_equals_overall"] = max(
        abs(np.mean([a10[k][str(s)]["mean"] for s in SRS]) - a10[k]["all_12_clusters"]["mean"]) for k in a10)
    res["crosschecks"] = cks
    X.log("cross-checks done")

    wall = time.time() - t0
    res["run"] = dict(start_utc=start, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), compute_s=wall,
                      hashed_files_verified=n_hashed)
    inputs = ["v3/gpu/full/scores/{stim_v3,stim_alt,mit}_<tag>_s<seed>.json", "nn/registers/full/scores/",
              "v3/scores/raw_v3maps, v3/scores/raw (IIE, MSDB, primary sweep)", "nn/results/raw_v3maps, raw_alt (sweep)",
              "v3/scores/raw_nulls.json", "known_values.json", "out/primary_P1_P2_P3_P4.json", "cbb/seed_bootstrap.json",
              "results/explore/A6_margin_curve.json", "results/explore/S13_block12_vs_nulls.json",
              "results/p2_companions_exploratory.json"]
    X.write_json("explore_saved.json", res, os.path.abspath(__file__), inputs=inputs,
                 notes="items 7 (first part), 14 and 20 were produced earlier and are only recomputed and compared here",
                 extra_code=[os.path.join(X.HERE, "a6_margin_curve.py"), os.path.join(X.HERE, "s13_block12_nulls.py"),
                             os.path.join(X.RE, "results", "scripts", "p2_companions.py")])
    write_md(X._clean(res), os.path.join(X.RESULTS, "explore_saved.md"), wall)


if __name__ == "__main__":
    main()
