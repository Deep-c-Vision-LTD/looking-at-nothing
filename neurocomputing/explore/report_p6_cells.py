"""
report_p6_cells.py: the rule 5.6 reporting items that out/primary_P6.json does not carry, and rule Section 6 items 1, 3
and 5, for the G3 and A14 cells of P6 (REANALYSIS_RULE.md Sections 5.6 and 6). Written 20 November 2026, after P6 and the
title decision had been run once (out/primary_P6.json, 01:20:26 UTC) and read (results/PRIMARY_P6.md, with seed point
estimates, per-size E and E against control90 of the two R1 opposite-sign cells; the published-method continuity
statistics of every cell, gpu/g3/report_full.md and gpu/a14/report_full.md written by the GPU runner, had been compared
with P6 for sign and zero exclusion, PRIMARY_P6.md Section 1), so every open choice below was made with those results
known. Exploratory and reporting only: no value computed here is decision-bearing, none changes P6
or the title decision, and the P6 values are only reproduced (gate). Precedent: DEVIATIONS.md D9 (explore_saved.py).
No torch; saved per-image scores only; the statistics engine is analyze_re.py (imported read only through excommon);
the item 1, item 3 and item 5 computations are the functions of explore_saved.py, imported read only and unchanged.

Rule text (quoted verbatim):
  Section 5.6, last bullet: "Reported for every new cell: CC per seed and mean, threshold, validity, E with WCU interval
  and Holm-adjusted p, seed point estimates, per-size E, E against control90, U20 and A20 (BCa, stability flag), and the
  published percentile interval for continuity."
  Section 5.6, accuracy criterion: "(...) Reported as well, never decision-bearing: each `dinov2reg` cell's CC against
  the plain DINOv2-B R4 threshold of the same block (0.6955 for block 12, 0.7122 for block 6). R4 cells are the
  reference and are not counted."
  Section 6, header: "Intervals: WCU (12 clusters; 6 for shape-specific; 4 for item 18) for linear quantities, BCa for
  ratios with the stability flag, two-stage percentile (published method) for support-ratio slopes."
  Section 6, item 1: "A4.1 per size: E, U20, A20, D20 and L(20) at 336 px and at 448 px separately for all 20 swap
  cells, the registers cells and the new cells (stim_v3 and stim_alt)."
  Section 6, item 3: "A4.3 rotation profiles: U, A and L at 10, 20 and 30 degrees, losses as differences; drop ratio
  D(theta) = (r(illusory) - r(alt theta)) / (r(illusory) - r(mis theta)) only where the uniform-loss interval excludes
  0; support-ratio slope of the per-configuration E (OLS on support ratio); per-seed values and the two-level seed
  bootstrap of `cbb/check_seed_bootstrap.py`; per-seed interval count (29 of 30 among valid cells, 59 of 60 overall;
  Critic fix 4)."
  Section 6, item 5: "A4.5 core share: f_core30 = M(core30) / M(display), illusory minus control180, every model, null
  and cell."
  Section 4.1: "for every primary linear contrast the sample skewness of its cluster means (`scipy.stats.skew`) is
  reported next to the result". Section 4.2: "A Fieller interval (...) is reported next to every single ratio".
  Section 4.3: "A ratio is reported only when the 95% WCU interval of its denominator excludes 0 (otherwise "not
  reportable"), and always next to the difference it summarises." Section 4.5: "the published two-stage percentile intervals are reported in the supplement for every
  primary estimate."

Implementation choices (the rule leaves these open; fixed here before this script read any score):
  (a) Units: the 18 new cells (G3: dinov2reg_L6 and dinov2reg_L12 x R1n, R1, R2, R3; A14: dinov2ms_L7L11 and
      dinov2ms_L11 x R1n, R1, R2, R3, R4, where the two A14 R4 cells are new and are the references of their sets) and,
      for context only, the two registers R4 reference cells of 17 November 2026 (nn/registers/full/scores): 20 units,
      16 of them non-R4 cells. The registers R4 units were already in explore_saved items 1 and 3; they are recomputed
      here and must equal results/explore/explore_saved.json (gate G5).
  (b) Score sources as analyze_re.step_primary for P6: dinov2reg R1n to R3 from gpu/g3/scores, dinov2reg R4 from
      nn/registers/full/scores, dinov2ms from gpu/a14/scores. Per display: seeds 0, 1, 2 averaged, then the 336 and
      448 px values averaged (analyze_re.head_scores, rule 2.2); per size: the seed mean at that size; per seed: the two
      sizes of one seed averaged (the conventions of explore_saved.cell_getter). stim_v3 and stim_alt files are both
      read; E and E against control90 use only stim_v3 conditions.
  (c) Validity: analyze_re.accuracy_valid on the unrounded seed-mean MIT1003 CCs (mit_<tag>_s<seed>.json), as P6.
      Holm p, keeps-sign and opposite-sign flags are those of analyze_re.p6 (reproduced, gate G2). Invalid cells are
      reported with every statistic and the flag "below criterion (rule 2.7); reported, not counted"; their Holm p is
      "n/a" (they are outside the rule 5.6 Holm family). R4 rows carry no threshold, validity or Holm p (reference).
  (d) The five items missing from primary_P6.json, size average unless stated:
      per-size E = excommon.lin (analyze_re.wild_cluster: WCU interval and WCR p, 12 clusters) on the per-size values;
      E against control90 = excommon.lin on r(illusory) - r(control90);
      U20 and A20 = excommon.ratio on analyze_re.ratio_parts (BCa on the v2 two-stage indices, Fieller alongside,
      stability flag |t| < 4 on the denominator E, reportable only if E's WCU interval excludes 0): the same calls as
      ratio_readout_bca of analyze_re.p4 (out/primary_P1_P2_P3_P4.json) and as explore_saved;
      continuity interval = analyze_re.pct_summary on E (the published two-stage percentile interval and bootstrap p;
      rule 4.5 names "every primary estimate", and E is the P6 estimate). For per-size E, E against control90, U20 and
      A20 the published percentile values (pct_summary, pct_ratio) are computed as well, only for gate G4, and stored
      in the JSON under "published_percentile_for_gate".
  (e) Rule 5.6 "reported as well": each dinov2reg non-R4 cell's seed-mean CC, rounded to 4 decimals, against the plain
      DINOv2-B R4 threshold of the same block, recomputed from v3/gpu/full/scores (mit_dinov2_L12_R4, mit_dinov2_L6_R4)
      with the published rounding and required to equal the rule's 0.6955 and 0.7122 (gate G3).
  (f) Items 1 and 3: explore_saved.a41_block, a43_profile, a43_seed and two_level, unchanged, so every choice of D9
      applies: D20 pooled over shapes and only where the L20 WCU interval excludes 0; item 3 losses L_uni and L_alt as
      differences (WCU); slope by excommon.slope_pct (two-stage percentile); per-seed values E (WCU, WCR p, published
      percentile) and L_uni (WCU), U and A as points; two-level seed bootstrap = the scheme of
      cbb/check_seed_bootstrap.py on r_inside (RandomState(2026) clusters, RandomState(2027) seeds, 10,000 resamples).
      No stored cbb output exists for these cells, so it is not cross-checked here (D9 reproduced the scheme).
  (g) Per-seed interval count (item 3): a seed interval counts if it excludes 0 on the side of R4's sign on the same
      features (sign of R4's seed-averaged E; for dinov2reg the registers R4 heads of 17 November), as explore_saved.
      Groups per feature set and pooled over the four sets: valid = valid non-R4 cells x 3 seeds; below_criterion;
      reference_R4; overall = every unit of the set x 3 seeds (R4 included); pooled also new_cells_only = the 18 new
      cells x 3 seeds (registers R4 references excluded). Point-sign counts and the published percentile interval are
      given alongside, as explore_saved.
  (h) Markdown: a ratio whose denominator interval includes 0 is printed "not reportable" without a value (rule 4.3,
      as D11 did for A3); the value stays in the JSON with reportable = false. This holds for the per-seed U20 and A20
      points too: their denominator is that seed's E, and they are printed "nr" when its WCU interval includes 0.
      Unstable ratios are printed with the flag. Each size-average U20 and A20 is printed next to the difference it
      summarises (L20 for U20, L_alt20 for A20, WCU) and with its Fieller interval (rules 4.2 and 4.3); E is printed with
      the skewness of its cluster means (rule 4.1, diagnostic). 4 decimals, signed for values and interval bounds alike;
      values below 5e-5 in magnitude in scientific notation (explore_saved.f4). Flags are printed as yes or no.
  (i) Gates (any failure stops the run before any output file is written):
      G1 the reused code is the code that produced explore_saved.json: SHA-256 of explore/explore_saved.py and
         explore/excommon.py equal to the hashes embedded in results/explore/explore_saved.json;
      G2 out/primary_P6.json reproduced in full: analyze_re.p3, p6 and title_decision rerun on the inputs that
         analyze_re.step_primary builds (stim_v3 only, six files per cell); every leaf compared after a JSON round trip;
         numeric |difference| <= 1e-12 and every non-numeric leaf equal;
      G3 per unit, through this script's own loader (stim_v3 and stim_alt), against primary_P6.json at 1e-12: CC mean,
         threshold, validity and status (R4 rows: CC mean, sign and reference defined), E mean, WCU interval, WCR p,
         se, t and cluster-mean skewness; per set: Holm p recomputed with analyze_re.holm over the valid non-R4 cells,
         the keeps-sign and opposite-sign flags and n_valid. primary_P6.json stores neither per-seed CCs nor seed point
         estimates, so the seed point estimates are checked here only through "their mean equals E" (1e-12, internal),
         and the per-seed CCs and the seed point estimates one by one only at 4 decimals in G4 (mit_cc_seed,
         E_seed_mean). Also the rule-fixed registers R4 means (0.7282, 0.7377), thresholds (0.6982, 0.7077) and plain
         thresholds (0.6955, 0.7122), exact;
      G4 the published-method continuity statistics stored at 4 decimals by gpu/analyze_gpu.py (gpu/g3/results_full.json,
         gpu/a14/results_full.json) and by the registers run (nn/registers/full/results_full.json): per-seed and mean
         CC, A14 validity flags, E mean and percentile interval, seed means and per-seed percentile intervals, per-size
         E and intervals, E against control90 and interval, U and A at 10, 20, 30, D where computed, the U20 and A20
         percentile intervals, and the item 5 f_core30 effect and its percentile interval (avg|f_core30); tolerance
         5.001e-5 (explore_saved.close4);
      G5 the registers R4 units' item 1, item 3 and item 5 values equal results/explore/explore_saved.json leaf by leaf
         (1e-12);
      internal: the rule 5.6 per-size E, U20 and A20 equal the corresponding item 1 values exactly.
  (j) Not computable: P6 counts a cell only with its three stim_v3 files and its three MIT1003 files (rule 5.6). A unit
      missing one of these is outside P6 too ("not run") and is reported here as "not computable from saved outputs"
      with the missing files listed (a set whose R4 lacks one: every unit of the set); never silently skipped. A unit
      that has these six files but lacks a stim_alt file is inside the P6 family; leaving it out would change the Holm
      family that G3 compares with P6, so the run stops before any statistic (GateError naming the files) and nothing
      is written. The rule Section 6 items that would apply to these cells but need data the G3 and A14 runs did not
      save (maps, feature caches, stim_rigid scores) are listed with the evidence checked at run time.
  (k) Outputs results/explore/P6_reporting.json and results/explore/P6_reporting.md (task names). Never overwritten: if
      either exists, main exits with code 4 before any computation, and write_outputs refuses as well. Earlier outputs
      are moved by hand to results/explore/_previous/<UTC>_P6_reporting/ (the D9 convention), never deleted.
  (l) Item 5 (A4.5 core share; rule 6.5 "every model, null and cell"): f_core30, illusory minus control180, size
      average, through explore_saved.item5 (unchanged: excommon.lin, WCU and WCR p) on the f_core30 values of the
      stim_v3 files, seeds averaged per display, then the two sizes averaged.
Revised 20 November 2026 (about 03:55 UTC) after an independent code review, before any real-data run of this script:
per-seed U20 and A20 follow rule 4.3 in the markdown; outputs are never overwritten; the G3 description states what is
compared; a unit lacking only stim_alt files stops the run instead of shrinking the Holm family; L20, L_alt20, Fieller
intervals and the skewness of E are printed; item 5 is computed and gated; only a missing stored field (KeyError) is
reported as one; flags print as yes or no; ratio values are signed like their intervals.
Usage (after the rule is public; run under results/scripts/run_step.sh conventions: nice -n 10, OMP_NUM_THREADS=4,
HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, V3_DEVICE=cpu, no Metal client): $PY -B -I explore/report_p6_cells.py
"""
import json, math, os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402
import explore_saved as ES                                                # noqa: E402  (read only, unchanged)

AR = X.AR
FSETS = ("dinov2reg_L6", "dinov2reg_L12", "dinov2ms_L7L11", "dinov2ms_L11")     # order of analyze_re.step_primary
SET_LABEL = {"dinov2reg_L6": "G3; DINOv2-B with registers, block 6; R4 = registers heads of 17 November 2026",
             "dinov2reg_L12": "G3; DINOv2-B with registers, block 12; R4 = registers heads of 17 November 2026",
             "dinov2ms_L7L11": "A14; public DINOv2-B blocks 7 and 11 concatenated; R4 trained in A14",
             "dinov2ms_L11": "A14; public DINOv2-B block 11 alone; R4 trained in A14"}
REG_R4_KEYS = ("dinov2reg_L12|R4", "dinov2reg_L6|R4")
THETAS = ES.THETAS
G3_SCORES = os.path.join(X.RE, "gpu", "g3", "scores")
A14_SCORES = os.path.join(X.RE, "gpu", "a14", "scores")
PATHS = dict(primary=os.path.join(X.RE, "out", "primary_P6.json"),
             explore_saved=os.path.join(X.RESULTS, "explore_saved.json"),
             rf_g3=os.path.join(X.RE, "gpu", "g3", "results_full.json"),
             rf_a14=os.path.join(X.RE, "gpu", "a14", "results_full.json"),
             rf_reg=os.path.join(X.ROOT, "nn", "registers", "full", "results_full.json"),
             maps=(os.path.join(X.RE, "gpu", "g3", "maps"), os.path.join(X.RE, "gpu", "a14", "maps")),
             cache=os.path.join(X.ROOT, "v2", "readout", "cache"),
             rigid_heads=os.path.join(X.RE, "scores_rigid", "heads", "scores"))
RULE_REG = {"dinov2reg_L12": dict(r4=0.7282, thr=0.6982, plain_fs="dinov2_L12", plain_thr=0.6955),   # rule 5.6
            "dinov2reg_L6": dict(r4=0.7377, thr=0.7077, plain_fs="dinov2_L6", plain_thr=0.7122)}
TOL = 1e-12
TOL4 = 5.001e-5
OUT_JSON, OUT_MD = "P6_reporting.json", "P6_reporting.md"
BELOW = "below criterion (rule 2.7); reported, not counted"


class GateError(Exception):
    """A reproduction gate failed: nothing is written."""


# ====================================================================== units and loaders (read only)
def units():
    """[(fset, arch, role)]: the 18 new cells and the 2 registers R4 references, in table order."""
    out = []
    for fs in FSETS:
        for ar in AR.ARCHS:
            if fs.startswith("dinov2reg"):
                role = "reference (registers R4, 17 Nov 2026; context only)" if ar == "R4" else "new cell (G3)"
            else:
                role = "reference (A14 R4; new cell)" if ar == "R4" else "new cell (A14)"
            out.append((fs, ar, role))
    return out


def src_dir(fs, ar):
    """Score folder of a unit, as analyze_re.step_primary (P6)."""
    if fs.startswith("dinov2reg"):
        return X.REG_SCORES if ar == "R4" else G3_SCORES
    return A14_SCORES


def needed_files(fs, ar):
    """{set: [paths]} of one unit: stim_v3, stim_alt and mit files of seeds 0, 1, 2."""
    d = src_dir(fs, ar)
    return {st: [os.path.join(d, f"{st}_{fs}_{ar}_s{s}.json") for s in (0, 1, 2)] for st in ("stim_v3", "stim_alt", "mit")}


def getter(fs, ar, seeds=(0, 1, 2), sizes=("336", "448"), meas=AR.MEAS, sets=("stim_v3", "stim_alt")):
    """value(config, cond): seeds averaged per display, then the listed sizes averaged (analyze_re.head_scores)."""
    return AR.getter(AR.head_scores(src_dir(fs, ar), f"{fs}_{ar}", seeds=seeds, sets=sets, sizes=sizes, meas=meas))


def mit_ccs(fs, ar, folder=None):
    folder = folder or src_dir(fs, ar)
    return [float(AR._load(os.path.join(folder, f"mit_{fs}_{ar}_s{s}.json"))["CC"]) for s in (0, 1, 2)]


def unit_getters(fs, ar):
    g = dict(avg=getter(fs, ar), **{z: getter(fs, ar, sizes=(z,)) for z in ("336", "448")})
    for s in (0, 1, 2):
        g[f"seed{s}"] = getter(fs, ar, seeds=(s,))
    g["f_core30"] = getter(fs, ar, meas="f_core30", sets=("stim_v3",))      # item 5 (illusory and control180 only)
    return g


# ====================================================================== comparison helpers
def _conv(o):
    """As analyze_re.jdump: numpy scalars and arrays to Python."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def roundtrip(o):
    """The object as analyze_re.jdump writes and json reads it (tuples to lists, numpy to Python, inf kept)."""
    return json.loads(json.dumps(o, default=_conv))


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def gate_compare(new, stored, tol=TOL):
    """Leaf-by-leaf comparison: numeric leaves within tol (non-finite leaves must be equal), others equal."""
    a, b = dict(ES.leaves(new)), dict(ES.leaves(stored))
    keys = sorted(set(a) | set(b))
    worst, mism, n_num = 0.0, [], 0
    for k in keys:
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if _num(x) and _num(y):
            n_num += 1
            fx, fy = float(x), float(y)
            if math.isfinite(fx) and math.isfinite(fy):
                d = abs(fx - fy)
                worst = max(worst, d)
                if d > tol:
                    mism.append([k, x, y])
            elif not (fx == fy or (math.isnan(fx) and math.isnan(fy))):
                mism.append([k, x, y])
        elif x != y:
            mism.append([k, x, y])
    return dict(n_leaves=len(keys), n_numeric=n_num, max_abs_diff=worst, n_mismatch=len(mism),
                first_mismatches=mism[:10], tolerance=tol, passed=not mism)


def _fail(gate, detail):
    raise GateError(f"{gate} failed: {detail}")


# ====================================================================== gates
def gate_code(stored_es):
    """G1: explore_saved.py and excommon.py unchanged since they produced explore_saved.json."""
    rec = {}
    for rel in ("explore/explore_saved.py", "explore/excommon.py"):
        cur = X.sha256_file(os.path.join(X.RE, rel))
        st = (stored_es.get("code") or {}).get(rel)
        rec[rel] = dict(current_sha256=cur, sha256_in_explore_saved_json=st, equal=bool(cur == st))
    return dict(files=rec, passed=all(r["equal"] for r in rec.values()))


def p6_inputs():
    """The P6 inputs exactly as analyze_re.step_primary builds them (stim_v3 only; a cell needs its six files)."""
    newsets, cc = {}, {}
    for fs in FSETS:
        heads, ccs = {}, {}
        for ar in AR.ARCHS:
            src = src_dir(fs, ar)
            need = [os.path.join(src, f"stim_v3_{fs}_{ar}_s{s}.json") for s in (0, 1, 2)] + \
                   [os.path.join(src, f"mit_{fs}_{ar}_s{s}.json") for s in (0, 1, 2)]
            if not all(os.path.exists(x) for x in need):
                continue
            heads[ar] = AR.getter(AR.head_scores(src, f"{fs}_{ar}", sets=("stim_v3",)))
            ccs[ar] = float(np.mean([AR._load(os.path.join(src, f"mit_{fs}_{ar}_s{s}.json"))["CC"] for s in (0, 1, 2)]))
        newsets[fs], cc[fs] = heads, ccs
    return newsets, cc


def gate_primary(D, stored_primary):
    """G2: rerun analyze_re.p3, p6 and title_decision as step_primary does for --which P6 and compare every leaf."""
    newsets, cc = p6_inputs()
    p6 = AR.p6(D, newsets, cc)
    heads = AR.load_swap()
    p3 = AR.p3(D, {k: heads[k] for k in AR.P3_CELLS}, {f: heads[(f, "R4")] for f in AR.SWAP_FSETS})
    p3["wording"] = AR.wording_p3(p3)
    title = AR.title_decision(p6, p3)
    new = roundtrip(dict(P3=p3, P6=p6, title=title))
    cmp = gate_compare(new, stored_primary["result"])
    cmp["stored_created_utc"] = stored_primary.get("created_utc")
    return cmp, new


def gate_unit(fs, ar, rec, st6):
    """G3 for one unit against the stored P6 result of its set (st6 = primary P6[fs])."""
    rows = []

    def chk(name, a, b):
        ok = (abs(float(a) - float(b)) <= TOL) if (_num(a) and _num(b)) else (a == b)
        rows.append(dict(check=name, value=a, stored=b, ok=bool(ok)))

    acc, E = rec["accuracy"], rec["rule_5_6"]["E"]
    seedE = rec["rule_5_6"]["seed_point_estimates"]
    if ar == "R4":
        ref = st6["R4"]
        chk("R4 CC mean", acc["cc_mean"], st6["R4_cc"])
        chk("R4 sign", int(np.sign(E["mean"])), st6["R4_sign"])
        chk("R4 reference defined", bool(E["p"] < AR.ALPHA), st6["R4_reference_defined"])
    else:
        ref = st6["cells"][ar]
        chk("CC mean", acc["cc_mean"], ref["cc"])
        chk("threshold", acc["threshold"], ref["threshold"])
        chk("validity", acc["valid"], ref["valid"])
        chk("status", "valid" if acc["valid"] else "below criterion", ref["status"])
    for k in ("mean", "p", "se", "t", "skew_cluster_means"):
        chk(f"E {k}", E[k], ref[k])
    for i in (0, 1):
        chk(f"E WCU ci[{i}]", E["ci"][i], ref["ci"][i])
    chk("mean of the seed point estimates equals E", float(np.mean(seedE)), E["mean"])
    return rows


def gate_sets(recs, st6):
    """G3 at set level: Holm within each set over the valid non-R4 cells (analyze_re.holm), keeps-sign and
    opposite-sign flags (analyze_re.opposite_interval), from this script's E; compared with primary_P6.json."""
    rows = []
    for fs in FSETS:
        r4 = recs.get(f"{fs}|R4")
        if r4 is None or r4.get("not_computable"):
            continue
        E4 = r4["rule_5_6"]["E"]
        ref_ok, sgn = bool(E4["p"] < AR.ALPHA), int(np.sign(E4["mean"]))
        cells = [(a, recs[f"{fs}|{a}"]) for a in AR.ARCHS if a != "R4" and not recs[f"{fs}|{a}"].get("not_computable")]
        valid = [(a, r) for a, r in cells if r["accuracy"]["valid"]]
        if valid:
            adj, rej = AR.holm([r["rule_5_6"]["E"]["p"] for _, r in valid])
            for (a, r), pa, j in zip(valid, adj, rej):
                st = st6[fs]["cells"][a]
                keeps = bool(ref_ok and j and np.sign(r["rule_5_6"]["E"]["mean"]) == sgn)
                rows.append(dict(check=f"{fs} {a} Holm p", value=float(pa), stored=st.get("p_holm"),
                                 ok=st.get("p_holm") is not None and abs(float(pa) - st["p_holm"]) <= TOL))
                rows.append(dict(check=f"{fs} {a} keeps sign", value=keeps, stored=st.get("keeps_sign"),
                                 ok=keeps == st.get("keeps_sign")))
        for a, r in cells:
            opp = AR.opposite_interval(r["rule_5_6"]["E"]["ci"], sgn, ref_ok)
            st = st6[fs]["cells"][a]
            rows.append(dict(check=f"{fs} {a} opposite-sign interval", value=opp, stored=st.get("opposite_sign_interval"),
                             ok=opp == st.get("opposite_sign_interval")))
        rows.append(dict(check=f"{fs} n_valid", value=len(valid), stored=st6[fs]["n_valid"], ok=len(valid) == st6[fs]["n_valid"]))
    return rows


def plain_thresholds():
    """Rule 5.6 'reported as well': the plain DINOv2-B R4 threshold of the same block, recomputed from the published
    MIT1003 files (v3/gpu/full/scores) with the published rounding."""
    plain = {}
    for fs, fx in RULE_REG.items():
        pc = mit_ccs(fx["plain_fs"], "R4", folder=X.SWAP_SCORES)
        plain[fs] = dict(plain_feature_set=fx["plain_fs"], plain_R4_cc_seeds=pc, plain_R4_cc_mean=float(np.mean(pc)),
                         plain_threshold=round(round(float(np.mean(pc)), 4) - 0.03, 4))
    return plain


def gate_rule_fixed(recs, plain):
    """G3: the registers R4 means and thresholds fixed in rule 5.6 (0.7282, 0.7377; 0.6982, 0.7077) and the plain
    DINOv2-B R4 thresholds (0.6955, 0.7122), exact."""
    rows = []
    for fs, fx in RULE_REG.items():
        r4 = recs.get(f"{fs}|R4")
        if r4 is None or r4.get("not_computable"):
            rows.append(dict(check=f"{fs} R4 available", value=False, stored=True, ok=False))
        else:
            m4 = round(r4["accuracy"]["cc_mean"], 4)
            rows.append(dict(check=f"{fs} R4 mean (4 decimals) as fixed in rule 5.6", value=m4, stored=fx["r4"],
                             ok=m4 == fx["r4"]))
            thr = round(m4 - 0.03, 4)
            rows.append(dict(check=f"{fs} threshold as fixed in rule 5.6", value=thr, stored=fx["thr"], ok=thr == fx["thr"]))
        pthr = plain[fs]["plain_threshold"]
        rows.append(dict(check=f"{fs} plain {fx['plain_fs']} R4 threshold as in rule 5.6", value=pthr,
                         stored=fx["plain_thr"], ok=pthr == fx["plain_thr"]))
    return rows


def rf_for(fs, ar, rfs):
    if fs.startswith("dinov2reg"):
        return rfs["reg"] if ar == "R4" else rfs["g3"]
    return rfs["a14"]


def gate_continuity(recs, rfs):
    """G4: published-method continuity statistics (4 decimals) of gpu/analyze_gpu.py and the registers run."""
    cats, fails = {}, []

    def add(cat, key, field, mine, stored, exact=False):
        ok = (mine == stored) if exact else ES.close4(mine, stored)
        c = cats.setdefault(cat, dict(n=0, n_agree=0, max_abs_diff=0.0))
        c["n"] += 1
        c["n_agree"] += int(bool(ok))
        if not exact and _num(mine) and _num(stored):
            c["max_abs_diff"] = max(c["max_abs_diff"], abs(float(mine) - float(stored)))
        if not ok:
            fails.append(dict(category=cat, unit=key, field=field, value=mine, stored=stored))

    def get(d, *ks):
        """d[k1][k2]...; None if any key or list index is missing."""
        for k in ks:
            if isinstance(d, dict) and k in d:
                d = d[k]
            elif isinstance(d, list) and isinstance(k, int) and 0 <= k < len(d):
                d = d[k]
            else:
                return None
        return d

    for key, rec in recs.items():
        if rec.get("not_computable"):
            continue
        fs, ar = key.split("|")
        rf = rf_for(fs, ar, rfs)
        acc, r56, it3 = rec["accuracy"], rec["rule_5_6"], rec["item3"]
        for s in (0, 1, 2):
            add("mit_cc_seed", key, f"seed {s}", acc["cc_seeds"][s], get(rf, "mit", fs, ar, "seeds", str(s)))
        add("mit_cc_mean", key, "mean", acc["cc_mean"], get(rf, "mit", fs, ar, "mean"))
        if rf is rfs["a14"] and ar != "R4":
            add("validity_flag_a14", key, "valid", acc["valid"], get(rf, "mit", fs, ar, "valid"), exact=True)
        k = get(rf, "stim_v3", fs, ar, "kanizsa")
        avg = get(k, "illusory-control180", "avg|r_inside")
        cont = r56["E_continuity_percentile"]
        add("E_mean", key, "avg mean", r56["E"]["mean"], get(avg, "mean"))
        for i in (0, 1):
            add("E_percentile_ci", key, f"avg ci[{i}]", cont["ci"][i], get(avg, "ci", i))
        for s in (0, 1, 2):
            add("E_seed_mean", key, f"seed {s}", r56["seed_point_estimates"][s], get(avg, "seed_means", s))
            ps = it3["per_seed"][str(s)]["E_percentile"]["ci"]
            for i in (0, 1):
                add("E_seed_percentile_ci", key, f"seed {s} ci[{i}]", ps[i], get(avg, "seed_cis", s, i))
        pp = r56["published_percentile_for_gate"]
        for z in ("336", "448"):
            st = get(k, "illusory-control180", f"{z}|r_inside")
            add("E_per_size", key, f"{z} mean", r56[f"E_{z}"]["mean"], get(st, "mean"))
            add("E_per_size", key, f"{z} percentile mean", pp[f"E_{z}"]["mean"], get(st, "mean"))
            for i in (0, 1):
                add("E_per_size_percentile_ci", key, f"{z} ci[{i}]", pp[f"E_{z}"]["ci"][i], get(st, "ci", i))
        st = get(k, "illusory-control90", "avg|r_inside")
        add("E_vs_control90", key, "mean", r56["E_vs_control90"]["mean"], get(st, "mean"))
        for i in (0, 1):
            add("E_vs_control90_percentile_ci", key, f"ci[{i}]", pp["E_vs_control90"]["ci"][i],
                get(st, "ci", i))
        sa = get(rf, "stim_alt", fs, ar, "r_inside")
        add("alt_effect_illusory", key, "mean", r56["E"]["mean"], get(sa, "effect_illusory", "mean"))
        prof = it3["profile"]
        for th in THETAS:
            add("profile_U_A", key, f"U{th}", prof[str(th)]["U"]["value"], get(sa, f"ratio_uni{th}", "value"))
            add("profile_U_A", key, f"A{th}", prof[str(th)]["A"]["value"], get(sa, f"ratio_alt{th}", "value"))
            if prof[str(th)]["D"].get("reported"):
                add("profile_D", key, f"D{th}", prof[str(th)]["D"]["value"], get(sa, f"drop_ratio{th}", "value"))
        for meas, sk in (("U20", "ratio_uni20"), ("A20", "ratio_alt20")):
            add("U20_A20_value", key, meas, r56[meas]["value"], get(sa, sk, "value"))
            for i in (0, 1):
                add("U20_A20_percentile_ci", key, f"{meas} ci[{i}]", pp[meas]["ci"][i],
                    get(sa, sk, "ci", i))
        i5, fc = rec["item5"], get(k, "illusory-control180", "avg|f_core30")
        add("item5_f_core30", key, "mean", i5["core_share"]["mean"], get(fc, "mean"))
        add("item5_f_core30", key, "percentile mean", i5["published_percentile_for_gate"]["mean"], get(fc, "mean"))
        for i in (0, 1):
            add("item5_f_core30_percentile_ci", key, f"ci[{i}]", i5["published_percentile_for_gate"]["ci"][i],
                get(fc, "ci", i))
    return dict(categories=cats, n_failed=len(fails), failures=fails[:30], tolerance=TOL4, passed=not fails)


def gate_explore_saved(stored_es, recs):
    """G5: the registers R4 units of explore_saved items 1, 3 and 5, recomputed here, leaf by leaf."""
    out = {}
    r = stored_es["result"]
    for key in REG_R4_KEYS:
        rec = recs.get(key)
        if rec is None or rec.get("not_computable"):
            out[key] = dict(passed=False, note="not computable here")
            continue
        new = X._clean(dict(item1=rec["item1"], profile=rec["item3"]["profile"], per_seed=rec["item3"]["per_seed"],
                            two_level=rec["item3"]["two_level_seed_bootstrap"], item5=rec["item5"]["core_share"]))
        i3 = r["item3_A4_3_profiles"]
        st = dict(item1=r["item1_A4_1_per_size"].get(key), profile=i3["profiles"].get(key),
                  per_seed=i3["per_seed"].get(key), two_level=i3["two_level_seed_bootstrap"].get(key),
                  item5=r.get("item5_A4_5_core_share", {}).get(key))
        out[key] = gate_compare(new, st)
    return dict(units=out, passed=all(v.get("passed") for v in out.values()))


def gate_internal(recs):
    """The rule 5.6 per-size E, U20 and A20 equal the item 1 values (same functions, same inputs)."""
    worst, n, bad = 0.0, 0, []
    for key, rec in recs.items():
        if rec.get("not_computable"):
            continue
        r56, i1 = rec["rule_5_6"], rec["item1"]
        pairs = [(r56["E_336"], i1["px336"]["E"]), (r56["E_448"], i1["px448"]["E"]),
                 (r56["E"], i1["size_average"]["E"]), (r56["U20"], i1["size_average"]["U20"]),
                 (r56["A20"], i1["size_average"]["A20"])]
        for a, b in pairs:
            c = gate_compare(X._clean(a), X._clean(b))
            n += 1
            worst = max(worst, c["max_abs_diff"])
            if not c["passed"]:
                bad.append(dict(unit=key, mismatches=c["first_mismatches"]))
    return dict(n_compared=n, max_abs_diff=worst, failures=bad[:10], passed=not bad)


# ====================================================================== per-unit computation
def accuracy_record(fs, ar, ccs, cc_r4, plain):
    m = float(np.mean(ccs))
    acc = dict(cc_seeds=ccs, cc_mean=m, cc_mean_4dp=round(m, 4))
    if ar == "R4":
        acc.update(role="reference", threshold=None, valid=None, status="reference")
        return acc
    thr = round(round(cc_r4, 4) - 0.03, 4)
    valid = AR.accuracy_valid(m, cc_r4)
    acc.update(R4_cc_mean=cc_r4, threshold=thr, valid=valid, status="valid" if valid else BELOW,
               gap_to_R4_unrounded=cc_r4 - m, margin_to_threshold_4dp=round(round(m, 4) - thr, 4))
    if fs in plain:
        p = plain[fs]
        acc["plain_R4_threshold_check"] = dict(
            plain_feature_set=p["plain_feature_set"], plain_threshold=p["plain_threshold"],
            at_or_above=bool(round(m, 4) >= p["plain_threshold"]),
            note="rule 5.6: reported as well, never decision-bearing")
    return acc


def rule56_items(D, g):
    """The rule 5.6 reporting items of one unit (E for the gate; the five missing items; continuity)."""
    E = AR.per_config(D, g["avg"], "illusory", "control180")
    E336 = AR.per_config(D, g["336"], "illusory", "control180")
    E448 = AR.per_config(D, g["448"], "illusory", "control180")
    E90 = AR.per_config(D, g["avg"], "illusory", "control90")
    nU, dU = AR.ratio_parts(D, g["avg"], "U20")
    nA, dA = AR.ratio_parts(D, g["avg"], "A20")
    seedE = [float(AR.per_config(D, g[f"seed{s}"], "illusory", "control180").mean()) for s in (0, 1, 2)]
    return dict(E=X.lin(E, D), seed_point_estimates=seedE,
                E_336=X.lin(E336, D), E_448=X.lin(E448, D), E_vs_control90=X.lin(E90, D),
                U20=X.ratio(nU, dU, D), A20=X.ratio(nA, dA, D),
                E_continuity_percentile=AR.pct_summary(E, D.idx),
                published_percentile_for_gate=dict(E_336=AR.pct_summary(E336, D.idx), E_448=AR.pct_summary(E448, D.idx),
                                                   E_vs_control90=AR.pct_summary(E90, D.idx),
                                                   U20=AR.pct_ratio(nU, dU, D.idx), A20=AR.pct_ratio(nA, dA, D.idx)))


def item1_unit(D, g):
    """Rule 6.1 (A4.1) exactly as explore_saved.item1: explore_saved.a41_block per size and for the size average."""
    out = {f"px{z}": ES.a41_block(D, g[z]) for z in ("336", "448")}
    out["size_average"] = ES.a41_block(D, g["avg"])
    return out


def item3_unit(D, g):
    """Rule 6.3 (A4.3) exactly as explore_saved.item3 for one unit: profile, per-seed values, two-level bootstrap."""
    per = np.stack([AR.per_config(D, g[f"seed{s}"], "illusory", "control180") for s in (0, 1, 2)])
    return dict(profile=ES.a43_profile(D, g["avg"]),
                per_seed={str(s): ES.a43_seed(D, g[f"seed{s}"]) for s in (0, 1, 2)},
                two_level_seed_bootstrap=ES.two_level(per, D.idx))


def item5_unit(D, g):
    """Rule 6.5 (A4.5) exactly as explore_saved.item5 for one unit (f_core30, illusory minus control180, WCU); the
    published percentile summary only for gate G4."""
    v = g["f_core30"]
    return dict(core_share=ES.item5(D, {"unit": v})["unit"],
                published_percentile_for_gate=AR.pct_summary(AR.per_config(D, v, "illusory", "control180"), D.idx))


def count_groups(recs):
    """Groups of item 3's per-seed interval count, per feature set and pooled (implementation choice (g))."""
    ok = {k: r for k, r in recs.items() if not r.get("not_computable")}
    groups = {}
    for name in FSETS + ("all_four_sets",):
        ks = [k for k in ok if name == "all_four_sets" or k.split("|")[0] == name]
        g = dict(valid=[k for k in ks if ok[k]["accuracy"]["status"] == "valid"],
                 below_criterion=[k for k in ks if ok[k]["accuracy"]["status"] == BELOW],
                 reference_R4=[k for k in ks if k.endswith("|R4")],
                 overall=ks)
        if name == "all_four_sets":
            g["new_cells_only"] = [k for k in ks if k not in REG_R4_KEYS]
        groups[name] = g
    return groups


def seed_counts(groups, recs, ref_sign):
    """Per-seed interval counts as explore_saved.item3 (WCU and published percentile; point signs alongside)."""
    out = {}
    for name, gs in groups.items():
        out[name] = {}
        for gname, keys in gs.items():
            rec = dict(n_seed_runs=3 * len(keys), n_point_R4_sign=0, n_wcu_excl0_R4_side=0, n_percentile_excl0_R4_side=0,
                       not_counted_wcu=[], not_counted_percentile=[], cells=list(keys))
            for k in keys:
                sg = ref_sign[k.split("|")[0]]
                for s in ("0", "1", "2"):
                    r = recs[k]["item3"]["per_seed"][s]
                    rec["n_point_R4_sign"] += int(np.sign(r["E"]["mean"]) == sg)
                    if ES.same_side(r["E"]["ci"], sg):
                        rec["n_wcu_excl0_R4_side"] += 1
                    else:
                        rec["not_counted_wcu"].append(dict(cell=k, seed=int(s), mean=r["E"]["mean"], ci_wcu=r["E"]["ci"]))
                    if ES.same_side(r["E_percentile"]["ci"], sg):
                        rec["n_percentile_excl0_R4_side"] += 1
                    else:
                        rec["not_counted_percentile"].append(dict(cell=k, seed=int(s), mean=r["E_percentile"]["mean"],
                                                                  ci_percentile=r["E_percentile"]["ci"]))
            out[name][gname] = rec
    return out


def other_items(recs):
    """Rule Section 6 items that would apply to the new cells beyond items 1 and 3, with run-time evidence."""
    n_maps = 0
    for d in PATHS["maps"]:
        for _, _, fs_ in os.walk(d):
            n_maps += len(fs_)
    cache_found = sorted(f for f in (os.listdir(PATHS["cache"]) if os.path.isdir(PATHS["cache"]) else [])
                         if any(f.startswith(s + "_") for s in FSETS))
    stim_cache = os.path.join(PATHS["cache"], "stim")
    stim_found = sorted(f for f in (os.listdir(stim_cache) if os.path.isdir(stim_cache) else []) if any(s in f for s in FSETS))
    rigid_found = sorted(f for f in (os.listdir(PATHS["rigid_heads"]) if os.path.isdir(PATHS["rigid_heads"]) else [])
                         if any(f"_{s}_" in f for s in FSETS) and not any(k.replace("|", "_") in f for k in REG_R4_KEYS))
    maps_txt = (f"the G3 and A14 runs saved no maps (V3_MAPS=none in gpu/run_gpu_re.sh; {n_maps} files under "
                "gpu/g3/maps and gpu/a14/maps)")
    regen = "computable only after re-inferring maps on the CPU from the saved checkpoints (a new scoring step with its own pipeline check)"
    return [
        dict(item=2, name="A4.2 ratio E90 / E180", status="not computed (not listed for the new cells)",
             reason="rule 6.2 lists the 20 swap cells, IIE, MSDB and the 15 nulls; E against control90 of the new cells "
                    "is the rule 5.6 item and is reported in Section 1"),
        dict(item=4, name="A4.4 log mass", status="not computable from saved outputs",
             reason=f"needs per-display maps (masses in the notch sectors and the inducer ink); {maps_txt}; {regen}"),
        dict(item=6, name="A4.6 calibration robustness", status="not computable from saved outputs",
             reason=f"needs logits re-inferred heads-only from a float16 stim_v3 feature cache or a backbone pass: no feature "
                    f"cache for the new feature sets (v2/readout/cache files found: {len(cache_found) + len(stim_found)}) "
                    f"and {maps_txt}; {regen}"),
        dict(item=7, name="A6 second part: absolute CC, NSS and information gain with the centre bias off",
             status="not computable from saved outputs",
             reason=f"heads-only from cached MIT1003 features: no <fset>_mit200.npy cache for the new feature sets "
                    f"(found: {len(cache_found)})"),
        dict(item=7, name="A6 first part: validity margin curve", status="not computed (described for the swap cells)",
             reason="rule 6.7 and rule 1.1 item 4 describe the curve for the 20 swap cells; the stored MIT1003 CCs of "
                    "the new cells would allow it, but no margin curve is defined for them"),
        dict(item=10, name="A12 Kavsek-style AOIs", status="not computable from saved outputs",
             reason=f"needs per-display maps (masses in the whole contour area P); {maps_txt}; {regen}"),
        dict(item=11, name="A13 region variant", status="not computable from saved outputs",
             reason=f"needs per-display maps (masses in P and in D minus P); {maps_txt}; {regen}"),
        dict(item=15, name="rigid rotation outside P5: E_rig, L_rig, A* and s*", status="not computable from saved outputs",
             reason=f"the G3 and A14 heads were not scored on stim_rigid (scores_rigid/heads/scores files for them: "
                    f"{len(rigid_found)}); rule 5.5 includes them only 'if they are scored on these displays later'"),
    ]


def compute_units(D, plain):
    """Per-unit records. First the accuracy record of every unit from its MIT1003 files alone (validity before any
    Kanizsa statistic, as rule 5.6 orders it for P6), then the rule 5.6 items and items 1, 3 and 5. A unit missing a
    stim_v3 or MIT1003 file, or whose set's R4 misses one, is outside P6 too and is "not computable from saved outputs"
    (reason recorded). A unit that P6 counts but that lacks a stim_alt file stops the run (GateError): leaving it out
    would change the Holm family that gate G3 compares with P6 (implementation choice (j))."""
    recs, missing_units = {}, []
    miss = {(fs, ar): {st: [os.path.relpath(p, X.ROOT) for p in ps if not os.path.exists(p)]
                       for st, ps in needed_files(fs, ar).items()} for fs, ar, _ in units()}
    core = {k: m["stim_v3"] + m["mit"] for k, m in miss.items()}             # the six files P6 needs
    r4_missing = {fs: core[(fs, "R4")] for fs in FSETS if core[(fs, "R4")]}
    alt_only = [f"{fs}|{ar}: " + ", ".join(m["stim_alt"]) for (fs, ar), m in miss.items()
                if m["stim_alt"] and not core[(fs, ar)] and fs not in r4_missing]
    if alt_only:
        _fail("precondition (stim_alt files of units counted by P6)",
              "a unit inside the P6 family lacks stim_alt files, so its Holm family would differ from P6's: " +
              "; ".join(alt_only))
    for fs, ar, role in units():
        key = f"{fs}|{ar}"
        if core[(fs, ar)] or fs in r4_missing:
            own, parts = core[(fs, ar)] + miss[(fs, ar)]["stim_alt"], []
            if own:
                parts.append("missing saved outputs: " + ", ".join(own))
            if fs in r4_missing and ar != "R4":
                parts.append("the R4 reference of this set is not computable (missing: " + ", ".join(r4_missing[fs]) + ")")
            reason = "; ".join(parts)
            recs[key] = dict(role=role, not_computable=f"not computable from saved outputs ({reason})")
            missing_units.append(dict(unit=key, reason=reason))
            continue
        recs[key] = dict(role=role, source=os.path.relpath(src_dir(fs, ar), X.ROOT))
    ccs = {k: mit_ccs(*k.split("|")) for k, r in recs.items() if not r.get("not_computable")}
    for key in ccs:
        fs, ar = key.split("|")
        recs[key]["accuracy"] = accuracy_record(fs, ar, ccs[key], float(np.mean(ccs[f"{fs}|R4"])), plain)
    for key in ccs:
        g = unit_getters(*key.split("|"))
        recs[key].update(rule_5_6=rule56_items(D, g), item1=item1_unit(D, g), item3=item3_unit(D, g),
                         item5=item5_unit(D, g))
        X.log(f"computed {key}")
    return recs, missing_units


def run(D):
    """Gates, the rule 5.6 items and items 1 and 3; returns the result dict or raises GateError (nothing written)."""
    stored_es = AR._load(PATHS["explore_saved"])
    gates = dict(G1_code=gate_code(stored_es))
    if not gates["G1_code"]["passed"]:
        _fail("G1 (reused code unchanged)", json.dumps(gates["G1_code"], default=str))
    stored_primary = AR._load(PATHS["primary"])
    g2, p6new = gate_primary(D, stored_primary)
    gates["G2_primary_P6"] = g2
    if not g2["passed"]:
        _fail("G2 (out/primary_P6.json reproduced)", json.dumps(g2["first_mismatches"], default=str))
    X.log(f"G2 passed: {g2['n_numeric']} numeric leaves, max |difference| {g2['max_abs_diff']}")
    st6 = stored_primary["result"]["P6"]

    plain = plain_thresholds()
    recs, missing_units = compute_units(D, plain)
    ok = {k: r for k, r in recs.items() if not r.get("not_computable")}
    g3 = []
    try:
        for key, r in ok.items():
            fs, ar = key.split("|")
            g3 += [dict(unit=key, **row) for row in gate_unit(fs, ar, r, st6[fs])]
        g3 += [dict(unit="set", **row) for row in gate_sets(recs, st6)]
    except KeyError as e:                                             # a stored P6 field is missing ("not run" set)
        _fail("G3 (per-unit values against out/primary_P6.json)", f"stored field missing: {e!r}")
    g3 += [dict(unit="rule", **row) for row in gate_rule_fixed(recs, plain)]
    bad = [x for x in g3 if not x["ok"]]
    gates["G3_per_unit"] = dict(n_checks=len(g3), n_failed=len(bad), failures=bad[:20], tolerance=TOL, passed=not bad,
                                rows=g3)
    if bad:
        _fail("G3 (per-unit values against out/primary_P6.json and rule 5.6)", json.dumps(bad[:5], default=str))
    X.log(f"G3 passed: {len(g3)} checks")
    rfs = dict(g3=AR._load(PATHS["rf_g3"]), a14=AR._load(PATHS["rf_a14"]), reg=AR._load(PATHS["rf_reg"]))
    gates["G4_continuity_published_method"] = gate_continuity(ok, rfs)
    if not gates["G4_continuity_published_method"]["passed"]:
        _fail("G4 (published-method continuity values, 4 decimals)",
              json.dumps(gates["G4_continuity_published_method"]["failures"][:5], default=str))
    X.log("G4 passed")
    gates["G5_registers_vs_explore_saved"] = gate_explore_saved(stored_es, ok)
    if not gates["G5_registers_vs_explore_saved"]["passed"]:
        _fail("G5 (registers R4 items 1 and 3 equal explore_saved.json)",
              json.dumps({k: v.get("first_mismatches", v.get("note")) for k, v in
                          gates["G5_registers_vs_explore_saved"]["units"].items()}, default=str))
    gates["internal_rule56_equals_item1"] = gate_internal(ok)
    if not gates["internal_rule56_equals_item1"]["passed"]:
        _fail("internal (rule 5.6 items equal item 1)",
              json.dumps(gates["internal_rule56_equals_item1"]["failures"], default=str))
    X.log("G5 and internal checks passed")

    # P6 fields (reproduced in G2) attached per cell: Holm p, keeps-sign, opposite-sign interval, R4 reference
    p6 = p6new["P6"]
    per_set = {}
    for fs in FSETS:
        ps = p6[fs]
        per_set[fs] = dict(label=SET_LABEL[fs], status=ps["status"], R4_sign=ps["R4_sign"],
                           R4_reference_defined=ps["R4_reference_defined"], n_valid=ps.get("n_valid"),
                           n_keep=ps.get("n_keep"), complete=ps.get("complete"), cells={})
        for ar in AR.ARCHS:
            key = f"{fs}|{ar}"
            r = recs[key]
            if r.get("not_computable"):
                per_set[fs]["cells"][ar] = dict(role=r["role"], not_computable=r["not_computable"])
                continue
            c = ps["cells"].get(ar, {}) if ar != "R4" else {}
            per_set[fs]["cells"][ar] = dict(
                role=r["role"], source=r["source"], accuracy=r["accuracy"], E_wcu=r["rule_5_6"]["E"],
                p_holm=c.get("p_holm"), keeps_sign=c.get("keeps_sign"), opposite_sign_interval=c.get("opposite_sign_interval"),
                seed_point_estimates=r["rule_5_6"]["seed_point_estimates"],
                E_336=r["rule_5_6"]["E_336"], E_448=r["rule_5_6"]["E_448"], E_vs_control90=r["rule_5_6"]["E_vs_control90"],
                U20=r["rule_5_6"]["U20"], A20=r["rule_5_6"]["A20"],
                L20=r["item1"]["size_average"]["L20"], L_alt20=r["item3"]["profile"]["20"]["L_alt"],
                E_continuity_percentile=r["rule_5_6"]["E_continuity_percentile"],
                published_percentile_for_gate=r["rule_5_6"]["published_percentile_for_gate"])
    ref_sign = {fs: int(p6[fs]["R4_sign"]) for fs in FSETS}
    others = other_items(recs)
    groups = count_groups(recs)
    item3 = dict(profiles={k: r["item3"]["profile"] for k, r in ok.items()},
                 per_seed={k: r["item3"]["per_seed"] for k, r in ok.items()},
                 two_level_seed_bootstrap={k: r["item3"]["two_level_seed_bootstrap"] for k, r in ok.items()},
                 per_seed_counts=seed_counts(groups, ok, ref_sign),
                 count_definition="a seed's interval counts if it excludes 0 on the side of R4's sign on the same "
                                  "features (R4's seed-averaged E; registers R4 of 17 November for dinov2reg); seeds "
                                  "size-averaged; WCU = rule interval, percentile = published two-stage interval")
    return dict(
        definitions=dict(E="r(illusory) - r(control180)", E90="r(illusory) - r(control90)", L="r(illusory) - r(mis theta)",
                         L_alt="r(illusory) - r(alt theta)", U="E(mis theta) / E(illusory)", A="E(alt theta) / E(illusory)",
                         D="(r(illusory) - r(alt theta)) / (r(illusory) - r(mis theta))", measure="r_inside (rule 2.2)",
                         core_share="f_core30 = M(core30) / M(display), illusory minus control180 (item 5)",
                         units="18 new cells (G3 8, A14 10 incl. the two A14 R4 references) and 2 registers R4 references"),
        units=[dict(unit=f"{fs}|{ar}", role=role) for fs, ar, role in units()],
        gates=gates, rule_5_6=per_set, plain_threshold_recomputed=plain,
        item1_A4_1_per_size={k: r["item1"] for k, r in ok.items()},
        item3_A4_3_profiles=item3,
        item5_A4_5_core_share={k: r["item5"] for k, r in ok.items()},
        not_computable=dict(units=missing_units, other_section6_items=[x for x in others
                                                                       if x["status"].startswith("not computable")]),
        not_computed_here=[x for x in others if not x["status"].startswith("not computable")])


# ====================================================================== markdown
f4, ci4 = ES.f4, ES.ci4


def yn(b):
    return "n/a" if b is None else ("yes" if b else "no")


def passed(b):
    return "passed" if b else "FAILED"


def rval(x, nd=4):
    """A ratio value, signed like the interval bounds."""
    return "n/a" if x is None else f"{x:+.{nd}f}"


def rtxt(r):
    """Ratio cell: value [BCa] with the flag; 'not reportable' without a value (rule 4.3)."""
    if not isinstance(r, dict):
        return str(r)
    if r.get("reported") is False:
        return "not computed (uniform-loss interval includes 0)"
    if r.get("value") is None:
        return "n/a"
    if r.get("reportable") is False:
        return "not reportable (denominator interval includes 0)"
    return f"{rval(r['value'])} {ci4(r.get('ci_bca'))}" + (" (unstable)" if r.get("flag") == "unstable" else "")


def rshort(r):
    if r.get("reported") is False:
        return "nc"
    if r.get("value") is None:
        return "n/a"
    if r.get("reportable") is False:
        return "nr"
    return rval(r["value"], 3) + ("u" if r.get("flag") else "")


def fieller_txt(r):
    """Fieller interval of a single ratio (rule 4.2); not printed where the ratio is not reportable (rule 4.3)."""
    if r.get("reportable") is False:
        return "nr"
    if r.get("value") is None:
        return "n/a"
    f = r.get("fieller") or {}
    return ci4(f["ci"]) if (f.get("bounded") and f.get("ci")) else "unbounded"


def seed_ratio(x, E):
    """Per-seed U or A point: printed only when that seed's E WCU interval excludes 0 (rule 4.3), else 'nr'."""
    return rval(x, 3) if X.excl0(E["ci"]) else "nr"


def ltxt(r):
    return f"{f4(r['mean'])} {ci4(r['ci'])}"


def kname(key):
    """A unit key 'fset|arch' as a markdown table cell (a pipe would split the cell)."""
    return key.replace("|", " ")


def ptxt(p):
    return "n/a" if p is None else f"{p:.4g}"


def num_txt(x, fmt):
    return "n/a" if x is None else format(x, fmt)


def write_md(res, path, prov, wall):
    L = []
    w = L.append
    w("# P6 cells: rule 5.6 reporting items and rule Section 6 items 1, 3 and 5 (`report_p6_cells.py`)\n")
    w("**EXPLORATORY AND REPORTING ONLY.** Nothing here is decision-bearing. P6 and the title decision are those of "
      "`out/primary_P6.json` (reproduced, gate G2). Linear quantities: mean and 95% WCU interval (12 clusters); ratios: "
      "value, BCa interval and flag; continuity: the published two-stage percentile interval. All values unrounded in "
      "`P6_reporting.json`; 4 decimals here.\n")
    w(f"- Code SHA-256: `explore/report_p6_cells.py` {prov['code']['explore/report_p6_cells.py']}; `explore/excommon.py` "
      f"{prov['code']['explore/excommon.py']}; `explore/explore_saved.py` {prov['code']['explore/explore_saved.py']}.")
    w(f"- Rule SHA-256 (`REANALYSIS_RULE.md`): {prov['rule_sha256']}. Public commit (`PUBLIC_PUSH.txt`): {prov['public_commit']}.")
    w(f"- Run {res['run']['start_utc']} to {res['run']['end_utc']} ({wall:.1f} s compute). require_public: "
      f"{res['run']['hashed_files_verified']} hashed files and every stimulus verified.\n")
    g = res["gates"]
    w("## 0. Reproduction gates (all passed; any failure would have stopped the run before writing)\n")
    w(f"- G1 reused code unchanged (explore_saved.py and excommon.py equal the hashes in explore_saved.json): "
      f"{passed(g['G1_code']['passed'])}.")
    g2 = g["G2_primary_P6"]
    w(f"- G2 out/primary_P6.json ({g2['stored_created_utc']}) reproduced in full (P3, P6, title): {g2['n_leaves']} leaves, "
      f"{g2['n_numeric']} numeric, max |difference| {g2['max_abs_diff']}, tolerance {g2['tolerance']}: "
      f"{passed(g2['passed'])}.")
    g3 = g["G3_per_unit"]
    w(f"- G3 per unit through this script's loader, against primary_P6.json at {g3['tolerance']}: CC mean, threshold, "
      f"validity, status, E mean, WCU interval, WCR p, se, t and cluster-mean skewness (R4 rows: CC mean, sign, reference "
      f"defined); per set Holm p, keeps-sign and opposite-sign flags and n_valid; the rule-fixed registers R4 means, "
      f"thresholds and plain thresholds exact. primary_P6.json stores neither per-seed CCs nor seed point estimates: the "
      f"seed point estimates enter G3 only through their mean equalling E, and both are checked one by one only at 4 "
      f"decimals in G4 (mit_cc_seed, E_seed_mean). {g3['n_checks']} checks, {g3['n_failed']} failed.")
    g4 = g["G4_continuity_published_method"]
    w("- G4 published-method continuity values (gpu/g3, gpu/a14 and nn/registers results_full.json, 4 decimals): " +
      "; ".join(f"{k} {v['n_agree']} of {v['n']}" for k, v in g4["categories"].items()) + ".")
    g5 = g["G5_registers_vs_explore_saved"]["units"]
    w("- G5 registers R4 items 1, 3 and 5 against explore_saved.json: " +
      "; ".join(f"{kname(k)} {v.get('n_numeric')} numeric leaves, max |difference| {v.get('max_abs_diff')}"
                for k, v in g5.items()) + ".")
    gi = g["internal_rule56_equals_item1"]
    w(f"- Internal: rule 5.6 per-size E, U20 and A20 equal the item 1 values ({gi['n_compared']} records, max |difference| "
      f"{gi['max_abs_diff']}).\n")

    w("## 1. Rule 5.6 items per feature set\n")
    w("CC per seed and mean: held-out MIT1003 CC; validity by `analyze_re.accuracy_valid`. E = illusory minus control180 "
      "(size average); Holm p within the set over its valid non-R4 cells. U20 and A20: BCa; `unstable` = denominator "
      "|t| < 4; `not reportable` = denominator WCU interval includes 0 (rule 4.3). Each set has three tables: the rule "
      "5.6 items; seed point estimates, WCR p, the P6 flags, the cluster-mean skewness of E (rule 4.1, diagnostic) and "
      "the plain-threshold comparison; and each ratio next to the difference it summarises with its Fieller interval "
      "(rules 4.2 and 4.3).\n")
    for fs, ps in res["rule_5_6"].items():
        w(f"### {fs} ({ps['label']})\n")
        w(f"R4 reference sign {ps['R4_sign']:+d}, reference defined: {yn(ps['R4_reference_defined'])}; valid non-R4 "
          f"cells {ps['n_valid']}; keeping the sign after Holm {ps['n_keep']}.\n")
        w("| cell | validity | CC per seed | mean CC | threshold | E [WCU] | Holm p | E 336 [WCU] | E 448 [WCU] | "
          "E vs control90 [WCU] | U20 [BCa] | A20 [BCa] | E [published percentile] |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ar, c in ps["cells"].items():
            if c.get("not_computable"):
                w(f"| {ar} | {c['not_computable']} |" + " |" * 11)
                continue
            a = c["accuracy"]
            val = ("reference (R4)" if ar == "R4" else ("valid" if a["valid"] else "below criterion (flag: reported, not counted)"))
            thr = "n/a (reference)" if ar == "R4" else f"{a['threshold']:.4f}"
            holm = "n/a (reference)" if ar == "R4" else (ptxt(c["p_holm"]) if a["valid"] else "n/a (below criterion)")
            w(f"| {ar} | {val} | {' / '.join(f'{x:.4f}' for x in a['cc_seeds'])} | {a['cc_mean']:.4f} | {thr} | "
              f"{ltxt(c['E_wcu'])} | {holm} | {ltxt(c['E_336'])} | {ltxt(c['E_448'])} | {ltxt(c['E_vs_control90'])} | "
              f"{rtxt(c['U20'])} | {rtxt(c['A20'])} | {ci4(c['E_continuity_percentile']['ci'])} |")
        w("")
        w("| cell | seed point estimates 0 / 1 / 2 | WCR p | skewness of the cluster means of E | keeps R4's sign | "
          "interval of the opposite sign | CC against the plain DINOv2-B R4 threshold (never decision-bearing) |")
        w("|---|---|---|---|---|---|---|")
        for ar, c in ps["cells"].items():
            if c.get("not_computable"):
                continue
            a = c["accuracy"]
            pt = a.get("plain_R4_threshold_check")
            ptx = "n/a" if pt is None else (f"{a['cc_mean_4dp']:.4f} against {pt['plain_threshold']:.4f} "
                                            f"({pt['plain_feature_set']}): {'at or above' if pt['at_or_above'] else 'below'}")
            keeps = "n/a" if ar == "R4" else ("n/a (below criterion)" if not a["valid"] else yn(c["keeps_sign"]))
            opp = "n/a" if ar == "R4" else (yn(c["opposite_sign_interval"]) +
                                            ("" if a["valid"] else " (below criterion; not counted)"))
            w(f"| {ar} | {' / '.join(f4(x) for x in c['seed_point_estimates'])} | {ptxt(c['E_wcu']['p'])} | "
              f"{num_txt(c['E_wcu'].get('skew_cluster_means'), '+.3f')} | {keeps} | {opp} | {ptx} |")
        w("")
        w("| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller | abs(t) of E |")
        w("|---|---|---|---|---|---|---|---|")
        for ar, c in ps["cells"].items():
            if c.get("not_computable"):
                continue
            w(f"| {ar} | {ltxt(c['L20'])} | {rtxt(c['U20'])} | {fieller_txt(c['U20'])} | {ltxt(c['L_alt20'])} | "
              f"{rtxt(c['A20'])} | {fieller_txt(c['A20'])} | {num_txt(c['U20'].get('t_den'), '.2f')} |")
        w("")

    w("## 2. Rule Section 6 item 1 (A4.1 per size), exploratory\n")
    w("| cell | size | E [WCU], p | L20 [WCU] | U20 [BCa] | A20 [BCa] | D20 [BCa] |")
    w("|---|---|---|---|---|---|---|")
    for key, rec in res["item1_A4_1_per_size"].items():
        for s in ("px336", "px448"):
            b = rec[s]
            w(f"| {kname(key)} | {s[2:]} | {ltxt(b['E'])}, p {ptxt(b['E']['p'])} | {ltxt(b['L20'])} | {rtxt(b['U20'])} | "
              f"{rtxt(b['A20'])} | {rtxt(b['D20'])} |")
    w("")

    r3 = res["item3_A4_3_profiles"]
    w("## 3. Rule Section 6 item 3 (A4.3), exploratory\n")
    w("Per-seed interval counts (seed intervals excluding 0 on the side of R4's sign on the same features; seeds "
      "size-averaged):\n")
    w("| feature set | group | seed runs | point estimate with R4's sign | WCU interval on R4's side | published percentile interval on R4's side |")
    w("|---|---|---|---|---|---|")
    for name, gs in r3["per_seed_counts"].items():
        for gname, c in gs.items():
            w(f"| {name} | {gname} | {c['n_seed_runs']} | {c['n_point_R4_sign']} | {c['n_wcu_excl0_R4_side']} | "
              f"{c['n_percentile_excl0_R4_side']} |")
    allc = r3["per_seed_counts"]["all_four_sets"]["overall"]
    for x in allc["not_counted_wcu"]:
        w(f"- not counted under WCU: {kname(x['cell'])} seed {x['seed']}, {f4(x['mean'])} {ci4(x['ci_wcu'])}")
    for x in allc["not_counted_percentile"]:
        w(f"- not counted under the percentile interval: {kname(x['cell'])} seed {x['seed']}, {f4(x['mean'])} {ci4(x['ci_percentile'])}")
    w("\nProfiles (size average, seeds averaged): L_uni, U, A, D at 10 / 20 / 30 degrees; slope of E per unit support "
      "ratio (two-stage percentile); two-level seed bootstrap interval of E (percentile).\n")
    w("| cell | L_uni 10 / 20 / 30 | U 10 / 20 / 30 | A 10 / 20 / 30 | D 10 / 20 / 30 | slope of E [pct] | E two-level [pct] |")
    w("|---|---|---|---|---|---|---|")
    for key, p in r3["profiles"].items():
        lu = " / ".join(f4(p[str(t)]["L_uni"]["mean"]) + ("*" if X.excl0(p[str(t)]["L_uni"]["ci"]) else "") for t in THETAS)
        u = " / ".join(rshort(p[str(t)]["U"]) for t in THETAS)
        a = " / ".join(rshort(p[str(t)]["A"]) for t in THETAS)
        d = " / ".join(rshort(p[str(t)]["D"]) for t in THETAS)
        sl, tl = p["slope_E"], r3["two_level_seed_bootstrap"][key]
        w(f"| {kname(key)} | {lu} | {u} | {a} | {d} | {f4(sl['slope_per_unit_sr'])} {ci4(sl['ci'])} | {ci4(tl['ci_two_level'])} "
          f"({tl['position_two_level']}{', changed' if tl['changed'] else ''}) |")
    w("\n`*` WCU interval excludes 0; `u` ratio unstable; `nr` not reportable (denominator interval includes 0; rule 4.3); "
      "`nc` D not computed (uniform-loss interval includes 0). Intervals of U, A and D are in the JSON.\n")
    w("Per-seed values (seeds size-averaged): E with its WCU interval and WCR p, E with the published percentile interval, "
      "L_uni(20) with its WCU interval, U20 and A20 as points (`nr` where that seed's E WCU interval includes 0; rule "
      "4.3; the points are in the JSON).\n")
    w("| cell | seed | E [WCU], p | E [published percentile] | L_uni 20 [WCU] | U20 | A20 |")
    w("|---|---|---|---|---|---|---|")
    for key, ss in r3["per_seed"].items():
        for s, r in ss.items():
            w(f"| {kname(key)} | {s} | {ltxt(r['E'])}, p {ptxt(r['E']['p'])} | {ci4(r['E_percentile']['ci'])} | "
              f"{ltxt(r['20']['L_uni'])} | {seed_ratio(r['20']['U'], r['E'])} | {seed_ratio(r['20']['A'], r['E'])} |")
    ntl = sum(1 for v in r3["two_level_seed_bootstrap"].values() if v["changed"])
    w(f"\nTwo-level seed bootstrap: the interval of E changes position relative to 0 in {ntl} of "
      f"{len(r3['two_level_seed_bootstrap'])} units.\n")

    w("## 4. Rule Section 6 item 5 (A4.5 core share f_core30, illusory minus control180), exploratory\n")
    w("| cell | f_core30 effect [WCU] | WCR p | skewness of the cluster means | published percentile interval |")
    w("|---|---|---|---|---|")
    for key, r in res["item5_A4_5_core_share"].items():
        cs = r["core_share"]
        w(f"| {kname(key)} | {ltxt(cs)} | {ptxt(cs['p'])} | {num_txt(cs.get('skew_cluster_means'), '+.3f')} | "
          f"{ci4(r['published_percentile_for_gate']['ci'])} |")
    w("")

    nc = res["not_computable"]
    w("## 5. Not computable from saved outputs\n")
    if nc["units"]:
        for x in nc["units"]:
            w(f"- {kname(x['unit'])}: {x['reason']}.")
    else:
        w("- Units: none. Every score and MIT1003 file of the 20 units exists; every rule 5.6 item and every part of items "
          "1, 3 and 5 was computed for every unit.")
    for x in nc["other_section6_items"]:
        w(f"- Rule 6 item {x['item']} ({x['name']}): {x['status']}; {x['reason']}.")
    w("\n## 6. Not computed by this script\n")
    for x in res["not_computed_here"]:
        w(f"- Rule 6 item {x['item']} ({x['name']}): {x['status']}; {x['reason']}.")
    w("")
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(path + ".tmp", path)
    print("wrote", path, flush=True)


# ====================================================================== outputs and main
def provenance():
    pub = os.path.join(X.RE, "PUBLIC_PUSH.txt")
    code = {os.path.relpath(p, X.RE): X.sha256_file(p) for p in
            (os.path.abspath(__file__), os.path.join(X.HERE, "excommon.py"), os.path.join(X.HERE, "explore_saved.py"))}
    return dict(code=code, rule_sha256=X.sha256_file(os.path.join(X.RE, "REANALYSIS_RULE.md")),
                public_commit=open(pub).read().split()[0] if os.path.exists(pub) else None)


def existing_outputs():
    """Output files that already exist (implementation choice (k): never overwritten)."""
    return [q for q in (os.path.join(X.RESULTS, OUT_JSON), os.path.join(X.RESULTS, OUT_MD)) if os.path.exists(q)]


def write_outputs(res, wall):
    ex = existing_outputs()
    if ex:
        raise FileExistsError("refusing to overwrite (move them to results/explore/_previous/<UTC>_P6_reporting/ "
                              "first): " + ", ".join(ex))
    inputs = ["gpu/g3/scores/{stim_v3,stim_alt,mit}_dinov2reg_<L>_<arch>_s<seed>.json",
              "gpu/a14/scores/{stim_v3,stim_alt,mit}_dinov2ms_<L>_<arch>_s<seed>.json",
              "nn/registers/full/scores (registers R4)", "v3/gpu/full/scores (P3 swap cells; plain R4 MIT1003 CCs)",
              "out/primary_P6.json", "results/explore/explore_saved.json", "gpu/g3/results_full.json",
              "gpu/a14/results_full.json", "nn/registers/full/results_full.json"]
    p = X.write_json(OUT_JSON, res, os.path.abspath(__file__), inputs=inputs,
                     notes="exploratory and reporting only; P6 and the title decision are reproduced, never changed",
                     extra_code=[os.path.join(X.HERE, "explore_saved.py")])
    write_md(X._clean(res), os.path.join(X.RESULTS, OUT_MD), provenance(), wall)
    return p


def main():
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_hashed = X.require_public()                                     # rule Section 8.1, before anything else
    X.log(f"require_public: {n_hashed} hashed files verified")
    ex = existing_outputs()
    if ex:                                                            # choice (k): before any computation
        X.log("outputs exist, nothing computed or written; move them to results/explore/_previous/<UTC>_P6_reporting/ "
              "first: " + ", ".join(ex))
        sys.exit(4)
    D = X.design()
    try:
        res = run(D)
    except GateError as e:
        X.log(f"GATE FAILED, nothing written: {e}")
        sys.exit(3)
    wall = time.time() - t0
    res["run"] = dict(start_utc=start, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), compute_s=wall,
                      hashed_files_verified=n_hashed)
    write_outputs(res, wall)


if __name__ == "__main__":
    main()
