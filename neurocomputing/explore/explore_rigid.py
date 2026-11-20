"""
explore_rigid.py: RUN_PLAN.md step R8, rigid-rotation profiles of every readout (REANALYSIS_RULE.md Section 6, item 15,
and the exploratory parts of Section 5.5). Exploratory: unadjusted, never decision-bearing, never used for a headline
claim (rule Section 6, header).

Written 18 November 2026, about 19:00 UTC: after the rule was fixed (11:29:31Z) and made public (PUBLIC_PUSH.txt), and
AFTER R7 had been run and read (out/primary_P5.json, 18:37 UTC: DeepGaze IIE and MSDB both P5 category 3). The
quantities are those the rule fixed; the implementation choices the rule leaves open are listed below and were made at
writing time, with the P5 result known (DEVIATIONS.md D7).

Rule text (quoted verbatim):
  Section 6, item 15: "Rigid rotation, everything outside P5: theta = 10, 30 (A*, C*, s* with the matched controls) and
  45 (E_rig, L_rig) for IIE and MSDB; E_rig, L_rig, A* and s* for the swap heads, sweep readouts and registers heads at
  every theta, with the nulls at their own values; the secondary quantities of Section 5.5."
  Section 5.5: "theta = 10 and 30: the same quantities, exploratory. theta = 45: only E_rig(45) and L_rig(45) (there is
  no uniform 45 degree set and no matched control), exploratory."
  Section 5.5: "Secondary quantities, reported for every model and null and never decision-bearing: the drafted
  alignment loss (RI - M) - (RC - C), the control-free RI - M, and the control's change under rigid rotation RC - C."
  Section 6, first paragraph: "Intervals: WCU (12 clusters; 6 for shape-specific; 4 for item 18) for linear
  quantities, BCa for ratios with the stability flag, ..."

Definitions (rule Section 5.5; per configuration i and angle theta; I = r(illusory), C = r(control180), M = r(mis theta),
MC = r(mis theta_control180), RI = r(rig theta illusory), RC = r(rig theta control180), r = r_inside):
  E(0) = I - C;  E_rig(theta) = RI - RC;  E'_mis(theta) = M - MC;
  L_uni(theta) = E(0) - E'_mis(theta);  L_rig(theta) = E(0) - E_rig(theta);
  A*(theta) = L_uni - L_rig = E_rig - E'_mis ("align_effect");  C*(theta) = L_uni - 2 L_rig ("half_criterion");
  s*(theta) = L_rig / L_uni ("rigid_share", ratio of means);
  secondary: A_drafted = (RI - M) - (RC - C), A_control_free = RI - M, control_rigid_change = RC - C.
  theta = 45: E_rig(45) = RI45 - RC45 and L_rig(45) = E(0) - E_rig(45) only.
The JSON uses the engine's key names (align_effect, half_criterion, rigid_share) so that every block can be compared
with out/primary_P5.json key for key.

Implementation (engine: analyze_re.py, imported read only through excommon; no statistic is new):
  - Units and loading.
    DeepGaze IIE, DeepGaze MSDB and the 15 local nulls: exactly as analyze_re.step_primary for P5 (published stim_v3 and
    stim_alt per-image scores merged with scores_rigid/<tag>.json and scores_rigid/nulls.json).
    Swap (20 cells, 60 heads) and registers (2 cells, 6 heads): analyze_re.head_scores on the published stim_v3 files
    (v3/gpu/full/scores, nn/registers/full/scores) and on scores_rigid/heads/scores/stim_rigid_<tag>_s<seed>.json,
    seeds then sizes averaged per display, merged by display name.
    Sweep (18 readouts, single seed): published stim_v3 per-image r_inside (excommon.sweep_published_file) and
    scores_rigid/<bb>_L<L>_s<size>.json, the 336 and 448 px values averaged per display.
    Every quantity is a linear function of per-display values (s* is a ratio of means of linear quantities), so averaging
    per display first equals averaging per configuration first (rule Section 2.2).
  - IIE and MSDB at theta = 10 and 30: analyze_re.p5(D, models, nulls, theta) as is (A* and the 15 model-minus-null
    differences with the IUT indicator and count, C*, s* with BCa, stability flag, denominator check and Fieller, L_uni,
    L_rig, the secondary quantities), plus E_rig(theta) (WCU). The engine's category at these angles is renamed
    "category_analogue" (exploratory, not a P5 category); the abstract-rule keys (both_category_1,
    abstract_leads_with_alternation) are dropped, because the abstract rule is defined at theta = 20 only.
  - IIE and MSDB at theta = 20 is P5 itself (decision-bearing). It is recomputed with the same call only to (a) check
    that this script's loading reproduces out/primary_P5.json (every shared numeric leaf; the check is written to the
    JSON) and (b) show the theta = 20 point of the profile, labelled with its role; E_rig(20) (WCU) is added.
  - The 15 nulls at their own values, theta = 10, 20, 30: E_rig, L_uni, L_rig, A*, C*, s*, secondary. C* is kept for the
    nulls for continuity with results/primary_P5_partial.json (nulls_own_values, theta = 20), which this script
    reproduces as a second check.
  - Readouts (swap, sweep, registers), theta = 10, 20, 30: E_rig, L_rig, A*, s* (item 15), L_uni (the denominator of s*,
    with its WCU interval and WCR p) and the three secondary quantities ("every model" in Section 5.5 is read as
    including the exploratory readouts that Section 5.5 lists under "Models"). C* is not computed for readouts (item 15
    does not list it).
  - theta = 45, every unit: E_rig(45) and L_rig(45) only. E(0) once per unit.
  - Linear quantities: analyze_re.wild_cluster through excommon.lin (WCU 95% interval and WCR p; 12 clusters, 9,999
    Webb draws, RandomState(2026); skewness of the cluster means as the engine reports it).
  - s*: analyze_re.bca on the v2 two-stage indices (10,000 resamples, RandomState(2026)); stability flag (|t| of L_uni
    below 4 = unstable); reportable only if L_uni's WCU interval excludes 0; Fieller alongside. The same computation as
    analyze_re.p5's rigid_share, plus a "flag" field.
  - Swap cells: the accuracy-criterion flag of rule Section 2.7 (analyze_re.accuracy_valid; mean of the three seed CCs
    in v3/gpu/full/scores/mit_<tag>_s<seed>.json against R4 of the same feature set), reported next to the cell, never
    used to drop it. Registers R4 and sweep readouts carry no flag (no non-R4 cell; single seed).
  - Pipeline checks: the rigid scores were written by the hashed scoring code after its own checks (rule Section 3.2);
    the check lines of those runs are copied from the run logs into the JSON. Completeness: a unit whose score file or
    any needed display is missing is reported as "not computable" with the reason (no substitute input).
Usage (CPU only, numpy and scipy, no torch):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
      $PY -B -I explore/explore_rigid.py [--md-only]
Outputs: results/explore/explore_rigid.json; results/explore/explore_rigid.md (short summary, generated from the JSON;
--md-only regenerates it from the JSON without recomputing anything).
Tests: explore/test_explore_rigid.py (synthetic inputs only).
"""
import argparse, hashlib, json, math, os, re, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
THETAS = (10, 20, 30)
T45 = 45
MODELS = ("DeepGaze IIE", "DeepGaze MSDB")
RIG_HEADS = os.path.join(X.SCORES_RIGID, "heads", "scores")
P5_OUT = os.path.join(X.RE, "out", "primary_P5.json")
P5_PARTIAL = os.path.join(X.RE, "results", "primary_P5_partial.json")
OUT_NAME = "explore_rigid.json"
MD_PATH = os.path.join(X.RESULTS, "explore_rigid.md")
MSDB_MD5 = "51db3832b635ebc15523d3e3e8a9bc14"                            # scores_rigid/DeepGaze_MSDB.json, 18:36 UTC
ROLE20 = ("P5 (decision-bearing; out/primary_P5.json). Recomputed here with the same engine call only for the profile "
          "and the reproduction check; nothing here changes P5.")
NOTE_CAT = ("category_analogue: analyze_re.p5_category applied at this angle; exploratory, not a P5 category, never "
            "decision-bearing (the rule defines categories at theta = 20 only)")
SYMBOLS = dict(E0="E(0) = I - C", E_rig="E_rig(theta) = RI - RC", L_uni="L_uni(theta) = E(0) - E'_mis(theta)",
               L_rig="L_rig(theta) = E(0) - E_rig(theta)", align_effect="A*(theta) = E_rig - E'_mis = L_uni - L_rig",
               half_criterion="C*(theta) = L_uni - 2 L_rig", rigid_share="s*(theta) = L_rig / L_uni (ratio of means)",
               A_drafted="(RI - M) - (RC - C)", A_control_free="RI - M", control_rigid_change="RC - C")
LOG_FILES = [("R1 local nulls", "results/run_logs/R1_local_nulls.log"), ("R2 DeepGaze IIE", "results/run_logs/R2_iie.log"),
             ("R6 DeepGaze MSDB", "results/msdb_rigid/run.log"), ("R3 sweep dinov2", "results/run_logs/R3_sweep_all.log"),
             ("R3 sweep clip (D1)", "results/run_logs/R3b_sweep_clip.log"),
             ("R3 sweep siglip (D1)", "results/run_logs/R3c_sweep_siglip.log"),
             ("R4a heads dinov2", "results/run_logs/R4a_check_dinov2.log"),
             ("R4a heads siglip", "results/run_logs/R4a_check_siglip.log"),
             ("R4a heads iie", "results/run_logs/R4a_check_iie.log"),
             ("R4a heads dinov2reg", "results/run_logs/R4a_check_dinov2reg.log")]
LOG_PAT = re.compile(r"pipeline check|reproduction check|max_abs_diff|exit \d+ wall|final: 660")


# ====================================================================== loading (read only)
def load_models_and_nulls(rig_dir=X.SCORES_RIGID):
    """DeepGaze IIE, MSDB and the 15 nulls exactly as analyze_re.step_primary (P5): published stim_v3 and stim_alt scores
    merged with the rigid scores. A model or null whose rigid file is missing is None (reported as not computable)."""
    models, why = {}, {}
    for name in MODELS:
        tag = name.replace(" ", "_")
        fp = os.path.join(rig_dir, tag + ".json")
        if os.path.exists(fp):
            models[name] = AR.getter({**AR.survey_scores(tag), **AR._load(fp)})
        else:
            models[name], why[name] = None, f"not computable: {os.path.basename(fp)} missing"
    N = AR.null_scores()
    fpn = os.path.join(rig_dir, "nulls.json")
    NR = AR._load(fpn) if os.path.exists(fpn) else {}
    nulls = {}
    for k in AR.NULLS:
        if k in NR:
            nulls[k] = AR.getter({**N[k], **NR[k]})
        else:
            nulls[k], why[k] = None, "not computable: no rigid scores in nulls.json"
    return models, nulls, why


def load_head_unit(fs, ar, pub_dir, rig_dir=RIG_HEADS, seeds=(0, 1, 2)):
    """(getter, None) for a readout head cell (swap or registers): published stim_v3 per-image r_inside and the rigid
    scores, each through analyze_re.head_scores (seeds, then sizes averaged per display); (None, reason) if a file is
    missing."""
    tag = f"{fs}_{ar}"
    need = [os.path.join(pub_dir, f"stim_v3_{tag}_s{s}.json") for s in seeds] + \
           [os.path.join(rig_dir, f"stim_rigid_{tag}_s{s}.json") for s in seeds]
    miss = [os.path.basename(p) for p in need if not os.path.exists(p)]
    if miss:
        return None, "not computable: score files missing: " + ", ".join(miss)
    pub = AR.head_scores(pub_dir, tag, seeds=seeds, sets=("stim_v3",))
    rig = AR.head_scores(rig_dir, tag, seeds=seeds, sets=("stim_rigid",))
    assert not (set(pub) & set(rig)), "display names overlap between stim_v3 and stim_rigid"
    return AR.getter({**pub, **rig}), None


def load_sweep_unit(bb, L, pub_file=X.sweep_published_file, rig_dir=X.SCORES_RIGID, sizes=X.SIZES):
    """(getter, None) for a sweep readout (single seed): published stim_v3 per-image r_inside and the rigid scores
    ({name: {measure: value}}), the values of the two sizes averaged per display; (None, reason) if a file is missing.
    Only display names present at every size are kept (a needed display missing at one size makes the unit not
    computable through the KeyError of analyze_re.per_config)."""
    acc = {}
    for s in sizes:
        pf, rf = pub_file(bb, L, s, "stim_v3"), os.path.join(rig_dir, f"{bb}_L{L}_s{s}.json")
        if pf is None or not os.path.exists(pf) or not os.path.exists(rf):
            return None, f"not computable: published or rigid score file missing for {bb}_L{L}_s{s}"
        pub, rig = AR._load(pf), AR._load(rf)
        assert not (set(pub) & set(rig)), "display names overlap between stim_v3 and stim_rigid"
        for nm, rec in {**pub, **rig}.items():
            acc.setdefault(nm, []).append(rec[AR.MEAS] if isinstance(rec, dict) else rec)
    return AR.getter({nm: float(np.mean(v)) for nm, v in acc.items() if len(v) == len(sizes)}), None


def swap_validity(fs, ar, cc_dir=AR.SWAP_SCORES):
    """Accuracy criterion of rule Section 2.7 (analyze_re.accuracy_valid), reported only."""
    def ccm(a):
        return float(np.mean([AR._load(os.path.join(cc_dir, f"mit_{fs}_{a}_s{s}.json"))["CC"] for s in (0, 1, 2)]))
    try:
        c, r4 = ccm(ar), ccm("R4")
    except (FileNotFoundError, KeyError) as e:
        return dict(valid=None, note=f"accuracy flag not available ({type(e).__name__})")
    return dict(cc_mean=c, cc_R4_mean=r4, valid=AR.accuracy_valid(c, r4),
                note="rule Section 2.7 criterion, reported next to the cell; not used to drop it")


# ====================================================================== quantities (engine of analyze_re)
def comp_arrays(D, val, theta):
    """Per-configuration (K, P) arrays at theta in (10, 20, 30): analyze_re.p5_parts plus E(0), E_rig and C*."""
    L_uni, L_rig, Aal, sec = AR.p5_parts(D, val, theta)
    E0 = AR.per_config(D, val, "illusory", "control180")
    E_rig = AR.per_config(D, val, f"rig{theta}_illusory", f"rig{theta}_control180")
    return dict(E0=E0, E_rig=E_rig, L_uni=L_uni, L_rig=L_rig, align_effect=Aal, half_criterion=L_uni - 2 * L_rig,
                secondary=sec)


def arrays45(D, val):
    """theta = 45: E_rig(45) = RI45 - RC45 and L_rig(45) = E(0) - E_rig(45) only (rule Section 5.5)."""
    E0 = AR.per_config(D, val, "illusory", "control180")
    E_rig = AR.per_config(D, val, f"rig{T45}_illusory", f"rig{T45}_control180")
    return dict(E_rig=E_rig, L_rig=E0 - E_rig)


def share_flag(stable, excl):
    if not excl:
        return "not reportable (denominator interval includes 0)"
    return None if stable else "unstable"


def rigid_share(L_rig, L_uni, D):
    """s* = mean(L_rig) / mean(L_uni): the computation of analyze_re.p5's rigid_share (BCa, |t| of L_uni, denominator
    WCU check, Fieller), plus a flag."""
    if not (np.all(np.isfinite(L_rig)) and np.all(np.isfinite(L_uni))):
        return dict(value=None, flag="not computable: configuration values missing or not finite")
    share = AR.bca(lambda a, b: a / b, [L_rig, L_uni], D.idx)
    tden = AR.t_den(L_uni.ravel(), D.g)
    wl = AR.wild_cluster(L_uni.ravel(), D.g)
    excl = bool(wl["ci"][0] > 0 or wl["ci"][1] < 0)
    out = dict(share, t_den=tden, stable=bool(tden >= AR.T_STABLE), denominator_excludes_zero=excl,
               fieller=AR.fieller(L_rig, L_uni, D.g))
    out["flag"] = share_flag(out["stable"], excl)
    return out


def own_values(D, val, theta, with_C):
    """A unit at its own values at theta in (10, 20, 30): E_rig, L_uni, L_rig, A*, s*, secondary (and C* if with_C)."""
    a = comp_arrays(D, val, theta)
    out = dict(E_rig=X.lin(a["E_rig"], D), L_uni=X.lin(a["L_uni"], D), L_rig=X.lin(a["L_rig"], D),
               align_effect=X.lin(a["align_effect"], D), rigid_share=rigid_share(a["L_rig"], a["L_uni"], D),
               secondary={k: X.lin(x, D) for k, x in a["secondary"].items()})
    if with_C:
        out["half_criterion"] = X.lin(a["half_criterion"], D)
    return out


def own45(D, val):
    a = arrays45(D, val)
    return dict(E_rig=X.lin(a["E_rig"], D), L_rig=X.lin(a["L_rig"], D))


def models_block(D, models, nulls, theta):
    """IIE and MSDB at theta (analyze_re.p5 as is). Returns (block, dropped abstract keys, the engine's raw output per
    model before any key is added or renamed, for the reproduction check)."""
    res = AR.p5(D, models, nulls, theta)
    dropped = {k: res.pop(k) for k in ("both_category_1", "abstract_leads_with_alternation")}
    raw = {m: plain(res[m]) for m in models}
    for m in models:
        r = res[m]
        if not r.get("computable"):
            continue
        r["E_rig"] = X.lin(AR.per_config(D, models[m], f"rig{theta}_illusory", f"rig{theta}_control180"), D)
        rs = r["rigid_share"]                                           # the engine's s* carries no flag field
        rs["flag"] = share_flag(rs["stable"], rs["denominator_excludes_zero"])
        if theta == 20:
            r["role"] = ROLE20
        else:
            r["category_analogue"] = r.pop("category")
            r["note"] = NOTE_CAT
    return res, dropped, raw


# ====================================================================== checks
def _conv(o):
    """As analyze_re.jdump's converter (numpy scalars and arrays to Python; non-finite floats kept)."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def plain(o):
    """A detached JSON-equivalent copy, written the way out/primary_P5.json was written (so the two compare 1:1)."""
    return json.loads(json.dumps(o, default=_conv))


def _num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def leaf_diff(a, b):
    """Compare two nested results on their shared keys: (max |difference| of numeric leaves, number of numeric leaves
    compared, list of mismatched non-numeric leaves or non-finite mismatches)."""
    mx, n, bad = 0.0, 0, []

    def walk(x, y, path):
        nonlocal mx, n
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) & set(y), key=str):
                walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                bad.append(path + " (length)")
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif _num(x) and _num(y):
            fx, fy = float(x), float(y)
            if math.isfinite(fx) and math.isfinite(fy):
                mx = max(mx, abs(fx - fy)); n += 1
            elif not (fx == fy or (math.isnan(fx) and math.isnan(fy))):
                bad.append(path)
        elif x != y:
            bad.append(path)
    walk(a, b, "")
    return dict(max_abs_diff=mx, n_numeric_compared=n, mismatched=bad[:20], n_mismatched=len(bad))


def log_lines(re_dir=X.RE):
    out = {}
    for label, rel in LOG_FILES:
        p = os.path.join(re_dir, rel)
        if not os.path.exists(p):
            out[label] = dict(file=rel, lines=["log file missing"])
            continue
        lines = [ln.rstrip("\n") for ln in open(p, errors="replace") if LOG_PAT.search(ln)]
        out[label] = dict(file=rel, lines=lines[-6:])
    return out


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ====================================================================== assembly
def compute_all(D, models, nulls, readouts, why=None, thetas=THETAS):
    """models: {name: getter or None}; nulls: {name: getter or None}; readouts: {unit: dict(group, val, reason, extra)}.
    Returns (units, checks_raw) where checks_raw holds the engine's raw theta = 20 output for the reproduction checks."""
    why = why or {}
    units, raw20, dropped20 = {}, {}, None
    nulls_ok = all(v is not None for v in nulls.values())
    # DeepGaze IIE and MSDB
    for m in models:
        units[m] = dict(group="deepgaze", status="ok" if models[m] is not None else why.get(m, "not computable"),
                        theta={})
        if models[m] is not None:
            units[m]["E0"] = X.lin(AR.per_config(D, models[m], "illusory", "control180"), D)
    if nulls_ok:
        for th in thetas:
            blk, dropped, raw = models_block(D, models, nulls, th)
            if th == 20:
                raw20, dropped20 = raw, dropped
            for m in models:
                if models[m] is not None:
                    units[m]["theta"][str(th)] = blk[m]
    else:
        for m in models:
            if models[m] is not None:
                units[m]["status"] = "partly computable: nulls missing, so the model-minus-null differences are absent"
                for th in thetas:
                    units[m]["theta"][str(th)] = own_values(D, models[m], th, with_C=True)
    for m in models:
        if models[m] is not None:
            units[m]["theta"][str(T45)] = own45(D, models[m])
    # the 15 nulls at their own values
    for k, v in nulls.items():
        rec = dict(group="nulls", status="ok" if v is not None else why.get(k, "not computable"), theta={})
        if v is not None:
            rec["E0"] = X.lin(AR.per_config(D, v, "illusory", "control180"), D)
            for th in thetas:
                rec["theta"][str(th)] = own_values(D, v, th, with_C=True)
            rec["theta"][str(T45)] = own45(D, v)
        units[k] = rec
    # readouts
    for u, spec in readouts.items():
        rec = dict(group=spec["group"], **(spec.get("extra") or {}), theta={})
        if spec["val"] is None:
            rec["status"] = spec.get("reason") or "not computable"
            units[u] = rec
            continue
        try:
            rec["E0"] = X.lin(AR.per_config(D, spec["val"], "illusory", "control180"), D)
            for th in thetas:
                rec["theta"][str(th)] = own_values(D, spec["val"], th, with_C=False)
            rec["theta"][str(T45)] = own45(D, spec["val"])
            rec["status"] = "ok"
        except (KeyError, AssertionError) as e:
            rec = dict(group=spec["group"], **(spec.get("extra") or {}), theta={},
                       status=f"not computable: missing display or score ({type(e).__name__}: {str(e)[:120]})")
        units[u] = rec
    return units, dict(raw20=raw20, dropped20=dropped20)


def real_readouts():
    """{unit: dict(group, val, reason, extra)} for the 20 swap cells, the 18 sweep readouts and the 2 registers cells."""
    out = {}
    for fs in X.SWAP_FSETS:
        for ar in X.ARCHS:
            try:
                val, why = load_head_unit(fs, ar, X.SWAP_SCORES)
            except AssertionError as e:
                val, why = None, f"not computable: {str(e)[:160]}"
            out[f"{fs}|{ar}"] = dict(group="swap", val=val, reason=why, extra=dict(accuracy=swap_validity(fs, ar)))
    for bb, Ls in X.SWEEP_LAYERS.items():
        for L in Ls:
            try:
                val, why = load_sweep_unit(bb, L)
            except AssertionError as e:
                val, why = None, f"not computable: {str(e)[:160]}"
            out[f"{bb}_L{L}"] = dict(group="sweep", val=val, reason=why, extra=dict(seeds="single seed (0)"))
    for fs in X.REG_FSETS:
        try:
            val, why = load_head_unit(fs, "R4", X.REG_SCORES)
        except AssertionError as e:
            val, why = None, f"not computable: {str(e)[:160]}"
        out[f"{fs}|R4"] = dict(group="registers", val=val, reason=why)
    return out


def input_md5s():
    """md5 of every rigid score file read (provenance)."""
    fs = [os.path.join(X.SCORES_RIGID, f) for f in ("DeepGaze_IIE.json", "DeepGaze_MSDB.json", "nulls.json")]
    fs += [os.path.join(X.SCORES_RIGID, f"{bb}_L{L}_s{s}.json") for bb, Ls in X.SWEEP_LAYERS.items() for L in Ls
           for s in X.SIZES]
    fs += [os.path.join(RIG_HEADS, f"stim_rigid_{fs_}_{ar}_s{s}.json") for fs_ in X.SWAP_FSETS for ar in X.ARCHS
           for s in (0, 1, 2)]
    fs += [os.path.join(RIG_HEADS, f"stim_rigid_{fs_}_R4_s{s}.json") for fs_ in X.REG_FSETS for s in (0, 1, 2)]
    return {os.path.relpath(p, X.RE): (md5(p) if os.path.exists(p) else None) for p in fs}


# ====================================================================== markdown summary (from the JSON)
def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def fp(p):
    return "n/a" if p is None else f"{p:.4g}"


def lin_cell(r):
    if not isinstance(r, dict) or r.get("mean") is None:
        return "n/a"
    return f"{f4(r['mean'])} [{f4(r['ci'][0])}, {f4(r['ci'][1])}], p {fp(r.get('p'))}"


def lin_short(r):
    if not isinstance(r, dict) or r.get("mean") is None:
        return "n/a"
    s = f"{f4(r['mean'])} [{f4(r['ci'][0])}, {f4(r['ci'][1])}]"
    return s + (" *" if X.excl0(r["ci"]) else "")


def dn(u):
    """Unit name for a markdown table cell (a literal '|' would split the cell)."""
    return u.replace("|", " ")


def share_cell(s):
    if not isinstance(s, dict) or s.get("value") is None:
        return "n/a"
    ci = s.get("ci") or [None, None]
    flag = s["flag"] if "flag" in s else share_flag(s.get("stable"), s.get("denominator_excludes_zero"))
    return f"{s['value']:.4f} [{f4(ci[0])}, {f4(ci[1])}]" + (f" ({flag})" if flag else "")


def write_md(rec, path=MD_PATH):
    R = rec["result"]
    U = R["units"]
    L = []
    w = L.append
    w("# R8: rigid-rotation profiles of every readout (exploratory)")
    w("")
    w(f"- Rule Section 6, item 15 (and the exploratory parts of Section 5.5); `RUN_PLAN.md` R8. Exploratory: unadjusted, "
      f"never decision-bearing, never used for a headline claim. The decision-bearing P5 (theta = 20, DeepGaze IIE and "
      f"MSDB; both category 3) is `out/primary_P5.json` and nothing here changes it.")
    w(f"- Script `explore/explore_rigid.py` (sha256 {rec['code'].get('explore/explore_rigid.py', '?')[:16]}...), engine "
      f"`analyze_re.py` imported read only; run {R['run']['started_utc']} to {R['run']['finished_utc']} "
      f"({R['run']['wall_s']:.1f} s compute), CPU only, numpy and scipy only. JSON: `results/explore/explore_rigid.json` "
      f"(created {rec['created_utc']}); values below are printed at 4 decimals, unrounded values are in the JSON.")
    ck = R["checks"]
    p5c, nc = ck["P5_reproduction"], ck["nulls_theta20_vs_P5_partial"]
    w(f"- Reproduction checks: IIE and MSDB at theta = 20 against `out/primary_P5.json`: max |difference| "
      f"{p5c['max_abs_diff']:.3g} over {p5c['n_numeric_compared']} numeric leaves, {p5c['n_mismatched']} non-numeric "
      f"mismatches; the 15 nulls at theta = 20 against `results/primary_P5_partial.json` (nulls_own_values): max "
      f"|difference| {nc['max_abs_diff']:.3g} over {nc['n_numeric_compared']} leaves, {nc['n_mismatched']} mismatches. "
      f"`scores_rigid/DeepGaze_MSDB.json` md5 {ck['msdb_md5']} (expected {MSDB_MD5}: "
      f"{'match' if ck['msdb_md5_ok'] else 'MISMATCH'}). Units computable: {ck['n_units_ok']} of {ck['n_units']}.")
    w("- Notation: A* = E_rig - E'_mis (alignment effect at matched edge orientation), C* = L_uni - 2 L_rig, "
      "s* = L_rig / L_uni; intervals: WCU 95% (12 clusters) with the WCR p for linear quantities, BCa for s* with the "
      "stability flag (|t| of L_uni < 4) and the denominator check. In the compact tables `*` marks a WCU interval that "
      "excludes 0. theta = 45 has E_rig and L_rig only (no uniform 45 degree set, no matched control).")
    w("")
    w("## 1. DeepGaze IIE and DeepGaze MSDB")
    w("")
    for m in MODELS:
        u = U.get(m, {})
        w(f"### {m}" + ("" if u.get("status") == "ok" else f" ({u.get('status')})"))
        w("")
        if u.get("status") != "ok":
            continue
        w(f"E(0) = I - C: {lin_cell(u.get('E0'))}")
        w("")
        w("| theta | role | E_rig | L_uni | L_rig | A* | C* | s* [BCa] (flag) | A* minus each null: separated of 15 (IUT) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for th in ("10", "20", "30"):
            t = u["theta"].get(th, {})
            role = "P5 (decision-bearing)" if th == "20" else "exploratory"
            cat = t.get("category") if th == "20" else t.get("category_analogue")
            sep = f"{t.get('n_separated')} of 15 ({'met' if t.get('iut_met') else 'not met'})" + \
                  (f"; category {cat}" if th == "20" else f"; analogue {cat}")
            w(f"| {th} | {role} | {lin_cell(t.get('E_rig'))} | {lin_cell(t.get('L_uni'))} | {lin_cell(t.get('L_rig'))} | "
              f"{lin_cell(t.get('align_effect'))} | {lin_cell(t.get('half_criterion'))} | "
              f"{share_cell(t.get('rigid_share'))} | {sep} |")
        t = u["theta"].get("45", {})
        w(f"| 45 | exploratory | {lin_cell(t.get('E_rig'))} | (none) | {lin_cell(t.get('L_rig'))} | (none) | (none) | "
          f"(none) | (none) |")
        w("")
        w("| theta | (RI - M) - (RC - C) | RI - M | RC - C |")
        w("|---|---|---|---|")
        for th in ("10", "20", "30"):
            s = u["theta"].get(th, {}).get("secondary", {})
            w(f"| {th} | {lin_cell(s.get('A_drafted'))} | {lin_cell(s.get('A_control_free'))} | "
              f"{lin_cell(s.get('control_rigid_change'))} |")
        w("")
        nn = []
        for th in ("10", "30"):
            vs = u["theta"].get(th, {}).get("vs_null", {})
            miss = [k for k, r in vs.items() if not (r.get("p") is not None and r["p"] < AR.ALPHA and r["mean"] > 0)]
            nn.append(f"theta {th}: not separated from {', '.join(miss) if miss else 'none'}")
        w("Nulls not separated by A* (model minus null, WCR p < .05 and positive): " + "; ".join(nn) + ".")
        w("")
    w("## 2. The 15 local nulls at their own values")
    w("")
    w("| null | A*(10) | A*(20) | A*(30) | L_rig(10) | L_rig(20) | L_rig(30) | L_rig(45) |")
    w("|---|---|---|---|---|---|---|---|")
    for k in AR.NULLS:
        u = U.get(k, {})
        if u.get("status") != "ok":
            w(f"| {k} | {u.get('status')} |||||||")
            continue
        t = u["theta"]
        w(f"| {k} | " + " | ".join(lin_short(t[th].get("align_effect")) for th in ("10", "20", "30")) + " | " +
          " | ".join(lin_short(t[th].get("L_rig")) for th in ("10", "20", "30", "45")) + " |")
    w("")
    w("## 3. Readouts (swap heads, sweep readouts, registers heads)")
    w("")
    rd = [u for u in U if U[u]["group"] in ("swap", "sweep", "registers")]
    w("A* and s* (s* at theta = 20; the other angles are in the JSON):")
    w("")
    w("| readout | group | accuracy flag | A*(10) | A*(20) | A*(30) | s*(20) [BCa] (flag) |")
    w("|---|---|---|---|---|---|---|")
    for u in rd:
        r = U[u]
        acc = r.get("accuracy", {}).get("valid") if r["group"] == "swap" else None
        accs = {True: "valid", False: "below criterion", None: ""}[acc]
        if r.get("status") != "ok":
            w(f"| {dn(u)} | {r['group']} | {accs} | {r.get('status')} ||||")
            continue
        t = r["theta"]
        w(f"| {dn(u)} | {r['group']} | {accs} | " + " | ".join(lin_short(t[th].get("align_effect")) for th in ("10", "20", "30"))
          + f" | {share_cell(t['20'].get('rigid_share'))} |")
    w("")
    w("L_rig by angle (E_rig(45) in the JSON):")
    w("")
    w("| readout | L_rig(10) | L_rig(20) | L_rig(30) | L_rig(45) |")
    w("|---|---|---|---|---|")
    for u in rd:
        r = U[u]
        if r.get("status") != "ok":
            continue
        t = r["theta"]
        w(f"| {dn(u)} | " + " | ".join(lin_short(t[th].get("L_rig")) for th in ("10", "20", "30", "45")) + " |")
    w("")
    cnt = R.get("counts", {})
    if cnt:
        w("Counts over the computable readouts (descriptive only; unadjusted intervals):")
        w("")
        w("| theta | readouts | A* interval above 0 | A* interval below 0 | L_rig interval above 0 | L_rig interval below 0 | s* flagged |")
        w("|---|---|---|---|---|---|---|")
        for th in ("10", "20", "30", "45"):
            c = cnt.get(th)
            if c:
                w(f"| {th} | {c['n']} | {c.get('A_pos', '(none)')} | {c.get('A_neg', '(none)')} | {c['L_pos']} | "
                  f"{c['L_neg']} | {c.get('s_flagged', '(none)')} |")
        w("")
    w("## 4. Notes")
    w("")
    for n in R.get("notes_md", []):
        w(f"- {n}")
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(path + ".tmp", path)
    print("wrote", path, flush=True)
    return path


def counts(units):
    out = {}
    rd = [u for u in units.values() if u["group"] in ("swap", "sweep", "registers") and u.get("status") == "ok"]
    for th in ("10", "20", "30", "45"):
        c = dict(n=len(rd), L_pos=0, L_neg=0)
        if th != "45":
            c.update(A_pos=0, A_neg=0, s_flagged=0)
        for u in rd:
            t = u["theta"][th]
            ci = t["L_rig"]["ci"]
            c["L_pos"] += int(ci[0] is not None and ci[0] > 0); c["L_neg"] += int(ci[1] is not None and ci[1] < 0)
            if th != "45":
                ca = t["align_effect"]["ci"]
                c["A_pos"] += int(ca[0] is not None and ca[0] > 0); c["A_neg"] += int(ca[1] is not None and ca[1] < 0)
                c["s_flagged"] += int(t["rigid_share"].get("flag") is not None)
        out[th] = c
    return out


NOTES_MD = [
    "Choices made at writing time (rule leaves them open; DEVIATIONS.md D7): C* is reported for the nulls (continuity "
    "with `results/primary_P5_partial.json`) and for IIE and MSDB, not for the readouts (item 15 does not list it); "
    "L_uni is reported for every unit as the denominator of s*; the secondary quantities are reported for every unit; "
    "the swap accuracy flag (rule 2.7) is shown, never used to drop a cell; at theta = 10 and 30 the engine's category "
    "function is shown as `category_analogue` (not a P5 category).",
    "Readouts see the display squashed to 336 or 448 px, so a display rotation by theta is not a rotation of their input "
    "(rule Section 2.6); this holds equally for every angle and control.",
    "The sweep readouts are single seed. Ratios s* carry their flag; a flagged s* is never counted (rule 4.3).",
    "Nothing here is decision-bearing; the P5 categories and the abstract rule are those of `out/primary_P5.json`.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-only", action="store_true", help="regenerate the markdown summary from the JSON only")
    a = ap.parse_args()
    X.require_public()                                                # rule Section 8.1, before anything is read
    if a.md_only:
        write_md(json.load(open(os.path.join(X.RESULTS, OUT_NAME))))
        return
    t0, started = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    D = X.design()
    X.log("loading DeepGaze IIE, MSDB and the 15 nulls (as analyze_re.step_primary P5)")
    models, nulls, why = load_models_and_nulls()
    X.log("loading the 20 swap cells, 18 sweep readouts and 2 registers cells")
    readouts = real_readouts()
    X.log("computing (theta 10, 20, 30, 45)")
    units, raw = compute_all(D, models, nulls, readouts, why)
    stored = json.load(open(P5_OUT))["result"]["P5"]
    p5c = leaf_diff({m: raw["raw20"][m] for m in MODELS}, {m: stored[m] for m in MODELS}) if raw["raw20"] else \
        dict(max_abs_diff=None, n_numeric_compared=0, mismatched=["not computed"], n_mismatched=1)
    if raw["dropped20"] is not None:
        dk = leaf_diff(raw["dropped20"], {k: stored[k] for k in raw["dropped20"]})
        p5c["abstract_keys_equal"] = dk["n_mismatched"] == 0
    part = json.load(open(P5_PARTIAL))["nulls_own_values"]
    nc = leaf_diff({k: units[k]["theta"]["20"] for k in AR.NULLS if units[k].get("status") == "ok"},
                   {k: part[k] for k in AR.NULLS})
    mmd5 = md5(os.path.join(X.SCORES_RIGID, "DeepGaze_MSDB.json"))
    checks = dict(P5_reproduction=p5c, nulls_theta20_vs_P5_partial=nc, msdb_md5=mmd5, msdb_md5_ok=mmd5 == MSDB_MD5,
                  n_units=len(units), n_units_ok=sum(u.get("status") == "ok" for u in units.values()),
                  not_computable={u: r["status"] for u, r in units.items() if r.get("status") != "ok"})
    wall = time.time() - t0
    res = dict(definition=SYMBOLS, thetas=list(THETAS), theta45=T45,
               design=dict(configurations=60, clusters=12, wild_draws=AR.B_WILD, bca_resamples=AR.B_PAIRS, seed=AR.SEED),
               run=dict(started_utc=started, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=wall),
               checks=checks, pipeline_check_logs=log_lines(), inputs_md5=input_md5s(), counts=counts(units),
               units=units, notes_md=NOTES_MD)
    p = X.write_json(OUT_NAME, res, os.path.abspath(__file__),
                     inputs=["v3/scores/raw_v3maps and raw (DeepGaze, sweep primary blocks)", "v3/scores/raw_nulls.json",
                             "nn/results/raw_v3maps (other sweep blocks)", "v3/gpu/full/scores (swap stim_v3, mit CC)",
                             "nn/registers/full/scores (registers stim_v3)", "neuro/reanalysis/scores_rigid (all)",
                             "out/primary_P5.json and results/primary_P5_partial.json (checks only)"],
                     notes="R8 (RUN_PLAN.md), rule Section 6 item 15; exploratory, never decision-bearing; see the "
                           "script header for every implementation choice",
                     extra_code=[os.path.join(X.RE, "analyze_re.py")])
    X.log(f"P5 reproduction: max |diff| {p5c['max_abs_diff']}, {p5c['n_mismatched']} mismatches; nulls vs P5 partial: "
          f"max |diff| {nc['max_abs_diff']}, {nc['n_mismatched']} mismatches; units ok {checks['n_units_ok']} of "
          f"{checks['n_units']}; compute {wall:.1f} s")
    write_md(json.load(open(p)))


if __name__ == "__main__":
    main()
