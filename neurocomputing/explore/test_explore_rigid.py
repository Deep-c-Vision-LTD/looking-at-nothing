"""
Unit tests of explore/explore_rigid.py (RUN_PLAN.md R8) on synthetic inputs only. No score, map, model or result file of
the project is read: analyze_re._load is wrapped so that any read outside this test's temporary folder fails, and the
published loaders (survey_scores, null_scores) are replaced by synthetic ones where a loader is tested. No torch import
(asserted). One process, 2 threads, seconds.
Writes only its own output file (--out, default explore/tests_explore_rigid_output.txt) and a few small synthetic JSON
files in a new scratch folder neuro/_trash/test_explore_rigid_scratch_<UTC time>/ (inside neuro/, never deleted by the
test; nothing is written outside neuro/).
Usage: OMP_NUM_THREADS=2 $PY -B -I explore/test_explore_rigid.py --out explore/tests_explore_rigid_output.txt
Written 18 November 2026.
"""
import argparse, ast, json, math, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402
import explore_rigid as ER                                                # noqa: E402

AR = X.AR
RES, LINES = [], []
SCRATCH = os.path.join(os.path.dirname(X.RE), "_trash",
                       "test_explore_rigid_scratch_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
os.makedirs(SCRATCH, exist_ok=False)
_ORIG_LOAD = AR._load


def _guarded_load(p):
    if not os.path.abspath(p).startswith(SCRATCH + os.sep):
        raise RuntimeError(f"test read outside its scratch folder: {p}")
    return _ORIG_LOAD(p)


AR._load = _guarded_load                                                  # no project score file can be read


def out(s):
    LINES.append(s); print(s, flush=True)


def check(name, ok, detail=""):
    RES.append(bool(ok))
    out(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""))


class SynthDesign:
    """12 clusters x 5 positions with synthetic configuration names; two-stage indices with fewer resamples (speed)."""

    def __init__(self, K=12, P=5, B=2000, seed=2026):
        self.K, self.P = K, P
        self.cells = [[f"c{k:02d}p{p}" for p in range(P)] for k in range(K)]
        self.configs = [c for cell in self.cells for c in cell]
        self.g = np.repeat(np.arange(K), P)
        self.sr = np.array([0.3, 0.5, 0.7] * (K // 3), float)
        self.idx = AR.two_stage_idx(K, P, B, seed)

    def arr(self, f):
        return np.array([[f(c) for c in cell] for cell in self.cells], float)


def synth(cfgs, rng, al=0.03, o_in=0.01, o_out=0.004, g_in=-0.005, g_out=0.002, noise=0.01):
    """Scores built from the decomposition of rule Section 5.5 (per configuration and theta):
    I = b_in + al; C = b_out; M = b_in + o_in; MC = b_out + o_out; RI = b_in + al + o_in + g_in; RC = b_out + o_out + g_out
    (o and g scale with theta / 20). Returns (scores {name: value}, components {cfg: dict})."""
    sc, comp = {}, {}
    for c in cfgs:
        b_in, b_out = 0.6 + noise * rng.randn(), 0.55 + noise * rng.randn()
        a = al + noise * rng.randn()
        sc[f"{c}_illusory"], sc[f"{c}_control180"] = b_in + a, b_out
        cc = dict(b_in=b_in, b_out=b_out, al=a)
        for th in (10, 20, 30, 45):
            f = th / 20
            oi, oo = f * o_in + noise * rng.randn(), f * o_out + noise * rng.randn()
            gi, go = f * g_in + noise * rng.randn(), f * g_out + noise * rng.randn()
            if th != 45:
                sc[f"{c}_mis{th}"], sc[f"{c}_mis{th}_control180"] = b_in + oi, b_out + oo
            sc[f"{c}_rig{th}_illusory"], sc[f"{c}_rig{th}_control180"] = b_in + a + oi + gi, b_out + oo + go
            cc[th] = dict(o_in=oi, o_out=oo, g_in=gi, g_out=go)
        comp[c] = cc
    return sc, comp


# ---------------------------------------------------------------------------------------------- imports and guard
def test_imports_and_guard():
    out("--- imports, no torch, require_public first")
    check("importing explore_rigid imports no torch", "torch" not in sys.modules)
    tree = ast.parse(open(os.path.join(HERE, "explore_rigid.py")).read())
    mainblk = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0].body
    pos = [i for i, st in enumerate(mainblk) if "require_public()" in ast.unparse(st)]
    before = [ast.unparse(st) for st in mainblk[:pos[0]]] if pos else ["<none>"]
    ok = bool(pos) and all(("ap" in b and ("argparse" in b or "ap." in b)) for b in before)
    check("require_public() is the first action of main (only argument parsing before it)", ok,
          f"statement {pos[0] if pos else None}")
    src = open(os.path.join(HERE, "explore_rigid.py")).read()
    check("the script refers to the engine only through excommon (analyze_re imported read only)",
          "import analyze_re" not in src and "AR = X.AR" in src)


# ---------------------------------------------------------------------------------------------- algebra
def test_algebra(D, rng):
    out("--- per-configuration quantities against the decomposition of rule Section 5.5")
    sc, comp = synth(D.configs, rng)
    val = AR.getter(sc)
    for th in (10, 20, 30):
        a = ER.comp_arrays(D, val, th)

        def tru(f):
            return D.arr(lambda c: f(comp[c], comp[c][th]))
        E0 = tru(lambda k, t: k["b_in"] + k["al"] - k["b_out"])
        Erig = tru(lambda k, t: k["b_in"] - k["b_out"] + k["al"] + t["o_in"] - t["o_out"] + t["g_in"] - t["g_out"])
        Emis = tru(lambda k, t: k["b_in"] - k["b_out"] + t["o_in"] - t["o_out"])
        Astar = tru(lambda k, t: k["al"] + t["g_in"] - t["g_out"])
        drafted = tru(lambda k, t: k["al"] + t["g_in"] - t["g_out"] - t["o_out"])
        cfree = tru(lambda k, t: k["al"] + t["g_in"])
        rc = tru(lambda k, t: t["o_out"] + t["g_out"])
        errs = dict(E0=np.abs(a["E0"] - E0).max(), E_rig=np.abs(a["E_rig"] - Erig).max(),
                    L_uni=np.abs(a["L_uni"] - (E0 - Emis)).max(), L_rig=np.abs(a["L_rig"] - (E0 - Erig)).max(),
                    A=np.abs(a["align_effect"] - Astar).max(),
                    C=np.abs(a["half_criterion"] - ((E0 - Emis) - 2 * (E0 - Erig))).max(),
                    drafted=np.abs(a["secondary"]["A_drafted"] - drafted).max(),
                    control_free=np.abs(a["secondary"]["A_control_free"] - cfree).max(),
                    control_change=np.abs(a["secondary"]["control_rigid_change"] - rc).max())
        check(f"theta {th}: E(0), E_rig, L_uni, L_rig, A* = a + (g_in - g_out), C*, the three secondary quantities",
              max(errs.values()) < 1e-12, f"max err {max(errs.values()):.2e}")
    a45 = ER.arrays45(D, val)
    E0 = D.arr(lambda c: comp[c]["b_in"] + comp[c]["al"] - comp[c]["b_out"])
    Er = D.arr(lambda c: comp[c]["b_in"] - comp[c]["b_out"] + comp[c]["al"] + comp[c][45]["o_in"] - comp[c][45]["o_out"]
               + comp[c][45]["g_in"] - comp[c][45]["g_out"])
    check("theta 45: only E_rig and L_rig, E_rig = RI45 - RC45, L_rig = E(0) - E_rig",
          set(a45) == {"E_rig", "L_rig"} and np.abs(a45["E_rig"] - Er).max() < 1e-12
          and np.abs(a45["L_rig"] - (E0 - Er)).max() < 1e-12)
    o45 = ER.own45(D, val)
    check("own45 reports E_rig(45) and L_rig(45) with WCU intervals and nothing else",
          set(o45) == {"E_rig", "L_rig"} and all(len(o45[k]["ci"]) == 2 for k in o45))
    r = X.lin(a45["L_rig"], D)
    w = AR.wild_cluster(a45["L_rig"].ravel(), D.g)
    check("linear quantities are analyze_re.wild_cluster (WCU interval, WCR p) exactly",
          r["mean"] == w["mean"] and r["ci"] == w["ci"] and r["p"] == w["p"])


# ---------------------------------------------------------------------------------------------- engine identity
def nulls_partial_style(D, v):
    """The nulls_own_values construction of results/scripts/p5_partial.py (copied), theta 20."""
    A = AR
    L_uni, L_rig, Aal, sec = A.p5_parts(D, v, 20)
    share = A.bca(lambda a, b: a / b, [L_rig, L_uni], D.idx)
    wl = A.wild_cluster(L_uni.ravel(), D.g)
    tden = A.t_den(L_uni.ravel(), D.g)
    return dict(L_uni=wl, L_rig=A.wild_cluster(L_rig.ravel(), D.g), align_effect=A.wild_cluster(Aal.ravel(), D.g),
                half_criterion=A.wild_cluster((L_uni - 2 * L_rig).ravel(), D.g),
                rigid_share=dict(share, t_den=tden, stable=bool(tden >= A.T_STABLE),
                                 denominator_excludes_zero=bool(wl["ci"][0] > 0 or wl["ci"][1] < 0),
                                 fieller=A.fieller(L_rig, L_uni, D.g)),
                secondary={s: A.wild_cluster(x.ravel(), D.g) for s, x in sec.items()})


def synth_units(D, rng):
    models = {m: AR.getter(synth(D.configs, rng, al=0.04)[0]) for m in ER.MODELS}
    nulls = {k: AR.getter(synth(D.configs, rng, al=0.0, g_in=0.0, g_out=0.0)[0]) for k in AR.NULLS}
    readouts = {"fs_a|R4": dict(group="swap", val=AR.getter(synth(D.configs, rng)[0]),
                                extra=dict(accuracy=dict(valid=True))),
                "fs_a|R1": dict(group="swap", val=None, reason="not computable: score files missing: x.json",
                                extra=dict(accuracy=dict(valid=False))),
                "bb_L4": dict(group="sweep", val=AR.getter(synth(D.configs, rng)[0])),
                "reg_L12|R4": dict(group="registers", val=AR.getter(synth(D.configs, rng)[0]))}
    # a readout with a missing rigid display: must be reported as not computable, not silently filled
    sc = synth(D.configs, rng)[0]
    del sc[f"{D.configs[3]}_rig30_control180"]
    readouts["bb_L8"] = dict(group="sweep", val=AR.getter(sc))
    return models, nulls, readouts


def test_engine_identity(D, rng):
    out("--- identity with the engine (analyze_re.p5) and with the P5-partial null construction")
    models, nulls, readouts = synth_units(D, rng)
    units, raw = ER.compute_all(D, models, nulls, readouts)
    for th in (10, 20, 30):
        ref = AR.p5(D, models, nulls, th)
        for m in ER.MODELS:
            blk = units[m]["theta"][str(th)]
            d = ER.leaf_diff({k: v for k, v in blk.items() if k not in ("E_rig", "role", "note", "category_analogue")},
                             ER.plain(ref[m]))
            check(f"theta {th}, {m}: every engine leaf identical to analyze_re.p5", d["max_abs_diff"] == 0.0
                  and d["n_mismatched"] == 0 and d["n_numeric_compared"] > 100, f"{d['n_numeric_compared']} leaves")
            if th == 20:
                check(f"theta 20, {m}: category kept as P5's, role label added", blk.get("category") == ref[m]["category"]
                      and "decision-bearing" in blk.get("role", ""))
            else:
                check(f"theta {th}, {m}: category renamed category_analogue (exploratory), abstract keys absent",
                      "category" not in blk and blk.get("category_analogue") == ref[m]["category"]
                      and "both_category_1" not in units[m]["theta"][str(th)])
            e = AR.per_config(D, models[m], f"rig{th}_illusory", f"rig{th}_control180")
            check(f"theta {th}, {m}: E_rig added with its WCU interval", blk["E_rig"]["mean"] == float(e.mean()))
            rs = blk["rigid_share"]
            check(f"theta {th}, {m}: s* flag added from the engine's stable and denominator check",
                  rs["flag"] == ER.share_flag(rs["stable"], rs["denominator_excludes_zero"]))
    d = ER.leaf_diff(raw["raw20"], {m: ER.plain(AR.p5(D, models, nulls, 20)[m]) for m in ER.MODELS})
    check("raw theta-20 output kept for the reproduction check equals analyze_re.p5", d["max_abs_diff"] == 0.0
          and d["n_mismatched"] == 0)
    check("abstract-rule keys of theta 20 kept for the check", set(raw["dropped20"]) ==
          {"both_category_1", "abstract_leads_with_alternation"})
    dn = ER.leaf_diff({k: units[k]["theta"]["20"] for k in AR.NULLS},
                      {k: ER.plain(nulls_partial_style(D, nulls[k])) for k in AR.NULLS})
    check("nulls at theta 20 reproduce the nulls_own_values construction of p5_partial.py exactly",
          dn["max_abs_diff"] == 0.0 and dn["n_mismatched"] == 0 and dn["n_numeric_compared"] > 15 * 50,
          f"{dn['n_numeric_compared']} leaves")
    nb = units[AR.NULLS[0]]["theta"]
    check("nulls: theta 10/20/30 carry E_rig, L_uni, L_rig, A*, C*, s*, secondary; theta 45 E_rig and L_rig only",
          all(set(nb[t]) == {"E_rig", "L_uni", "L_rig", "align_effect", "half_criterion", "rigid_share", "secondary"}
              for t in ("10", "20", "30")) and set(nb["45"]) == {"E_rig", "L_rig"})
    rb = units["fs_a|R4"]["theta"]
    check("readouts: E_rig, L_uni, L_rig, A*, s*, secondary at 10/20/30, no C*; theta 45 E_rig and L_rig only",
          all(set(rb[t]) == {"E_rig", "L_uni", "L_rig", "align_effect", "rigid_share", "secondary"}
              for t in ("10", "20", "30")) and set(rb["45"]) == {"E_rig", "L_rig"})
    check("a readout with missing score files is not computable with its reason (no substitute)",
          units["fs_a|R1"]["status"].startswith("not computable: score files missing")
          and units["fs_a|R1"]["theta"] == {} and units["fs_a|R1"]["accuracy"]["valid"] is False)
    check("a readout with one missing rigid display is not computable (KeyError), no partial values",
          units["bb_L8"]["status"].startswith("not computable: missing display") and units["bb_L8"]["theta"] == {},
          units["bb_L8"]["status"][:90])
    check("groups and E(0) recorded for every computable unit",
          all("E0" in units[u] for u in units if units[u].get("status") == "ok")
          and units["reg_L12|R4"]["group"] == "registers")
    c = ER.counts(units)
    check("descriptive counts over computable readouts (3 here) at every angle, A* and s* not at 45",
          all(c[t]["n"] == 3 for t in c) and "A_pos" not in c["45"] and "s_flagged" in c["20"])
    # a missing model is not computable and does not stop the rest
    m2 = dict(models); m2["DeepGaze MSDB"] = None
    u2, r2 = ER.compute_all(D, m2, nulls, {}, why={"DeepGaze MSDB": "not computable: DeepGaze_MSDB.json missing"})
    check("a model without rigid scores: not computable, the other model still computed",
          u2["DeepGaze MSDB"]["status"].startswith("not computable") and u2["DeepGaze MSDB"]["theta"] == {}
          and set(u2["DeepGaze IIE"]["theta"]) == {"10", "20", "30", "45"})
    return units


# ---------------------------------------------------------------------------------------------- s* flag
def test_share(D, rng):
    out("--- rigid share s*: BCa, stability flag, denominator check, Fieller")
    L_uni = D.arr(lambda c: 0.05 + 0.005 * rng.randn())
    L_rig = D.arr(lambda c: 0.02 + 0.005 * rng.randn())
    s = ER.rigid_share(L_rig, L_uni, D)
    check("strong denominator: value = ratio of means, stable, reportable, no flag, Fieller bounded",
          abs(s["value"] - L_rig.mean() / L_uni.mean()) < 1e-12 and s["stable"] and s["denominator_excludes_zero"]
          and s["flag"] is None and s["fieller"]["bounded"])
    L_uni0 = D.arr(lambda c: 0.002 * rng.randn())
    s0 = ER.rigid_share(L_rig, L_uni0, D)
    check("denominator around 0: flagged not reportable", s0["flag"] == "not reportable (denominator interval includes 0)")
    check("flag logic: excludes 0 but |t| < 4 gives 'unstable'; both fine gives None",
          ER.share_flag(False, True) == "unstable" and ER.share_flag(True, True) is None
          and ER.share_flag(True, False).startswith("not reportable"))
    bad = L_uni.copy(); bad[0, 0] = np.nan
    check("non-finite configuration values: not computable, no interval", ER.rigid_share(L_rig, bad, D)["value"] is None)


# ---------------------------------------------------------------------------------------------- loaders
def wj(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"))


def test_loaders(D, rng):
    out("--- loaders (temporary files only)")
    T = SCRATCH
    pub, rig = os.path.join(T, "pub"), os.path.join(T, "rig")
    cfg = D.configs[0]
    # heads: {size: {measure: {name: value}}}, seeds 0 and 1 (2 x 2 cells)
    for s in (0, 1):
        wj(os.path.join(pub, f"stim_v3_fsX_R2_s{s}.json"),
           {z: {"r_inside": {f"{cfg}_illusory": 0.1 * (s + 1) + (z == "448"), f"{cfg}_mis20": 0.5}} for z in ("336", "448")})
        wj(os.path.join(rig, f"stim_rigid_fsX_R2_s{s}.json"),
           {z: {"r_inside": {f"{cfg}_rig20_illusory": 0.2 * s + (z == "336")}} for z in ("336", "448")})
    val, why = ER.load_head_unit("fsX", "R2", pub, rig, seeds=(0, 1))
    check("heads: seeds then sizes averaged per display, stim_v3 and stim_rigid merged",
          why is None and abs(val(cfg, "illusory") - (0.15 + 0.5)) < 1e-12 and abs(val(cfg, "rig20_illusory") - 0.6) < 1e-12
          and val(cfg, "mis20") == 0.5)
    v2, why2 = ER.load_head_unit("fsX", "R2", pub, rig, seeds=(0, 1, 2))
    check("heads: a missing seed file gives (None, reason)", v2 is None and "stim_v3_fsX_R2_s2.json" in why2, why2)
    wj(os.path.join(rig, "stim_rigid_fsY_R2_s0.json"), {"336": {"r_inside": {f"{cfg}_illusory": 1.0}},
                                                        "448": {"r_inside": {f"{cfg}_illusory": 1.0}}})
    wj(os.path.join(pub, "stim_v3_fsY_R2_s0.json"), {"336": {"r_inside": {f"{cfg}_illusory": 0.0}},
                                                      "448": {"r_inside": {f"{cfg}_illusory": 0.0}}})
    try:
        ER.load_head_unit("fsY", "R2", pub, rig, seeds=(0,)); ok = False
    except AssertionError:
        ok = True
    check("heads: a display name in both stim_v3 and stim_rigid is refused (no silent overwrite)", ok)
    # sweep: {name: {measure: value}}, two sizes
    for s in (336, 448):
        wj(os.path.join(pub, f"bbZ_L4_s{s}.json"), {f"{cfg}_illusory": {"r_inside": s / 1000.0, "f_ink": 9.0}})
        wj(os.path.join(rig, f"bbZ_L4_s{s}.json"), {f"{cfg}_rig45_control180": {"r_inside": 1.0 if s == 336 else 0.0}})
    pubf = lambda bb, L, s, st: os.path.join(pub, f"{bb}_L{L}_s{s}.json")
    vs, whys = ER.load_sweep_unit("bbZ", 4, pub_file=pubf, rig_dir=rig)
    check("sweep: the two sizes averaged per display (single seed), published and rigid merged",
          whys is None and abs(vs(cfg, "illusory") - 0.392) < 1e-12 and abs(vs(cfg, "rig45_control180") - 0.5) < 1e-12)
    vn, whyn = ER.load_sweep_unit("bbZ", 6, pub_file=pubf, rig_dir=rig)
    check("sweep: a missing file gives (None, reason)", vn is None and "bbZ_L6" in whyn)
    # DeepGaze and nulls as analyze_re.step_primary, with synthetic published loaders
    sv, nv = AR.survey_scores, AR.null_scores
    try:
        AR.survey_scores = lambda tag: {f"{cfg}_illusory": {"r_inside": 0.7}, f"{cfg}_mis20": {"r_inside": 0.6}}
        AR.null_scores = lambda: {k: {f"{cfg}_illusory": {"r_inside": 0.3}} for k in AR.NULLS}
        dg = os.path.join(T, "dg")
        wj(os.path.join(dg, "DeepGaze_IIE.json"), {f"{cfg}_rig20_illusory": {"r_inside": 0.65}})
        wj(os.path.join(dg, "nulls.json"), {k: {f"{cfg}_rig20_illusory": {"r_inside": 0.25}} for k in AR.NULLS})
        models, nulls, why = ER.load_models_and_nulls(dg)
        check("DeepGaze: published and rigid scores merged as step_primary; MSDB without a rigid file is None",
              models["DeepGaze IIE"](cfg, "illusory") == 0.7 and models["DeepGaze IIE"](cfg, "rig20_illusory") == 0.65
              and models["DeepGaze MSDB"] is None and "DeepGaze_MSDB.json missing" in why["DeepGaze MSDB"])
        check("nulls: published and rigid merged for all 15", len(nulls) == 15 and
              all(nulls[k](cfg, "illusory") == 0.3 and nulls[k](cfg, "rig20_illusory") == 0.25 for k in AR.NULLS))
    finally:
        AR.survey_scores, AR.null_scores = sv, nv
    # accuracy flag with the published rounding
    cc = os.path.join(T, "cc")
    for a, c in (("R4", 0.7000), ("R1", 0.6701), ("R2", 0.6699)):
        for s in (0, 1, 2):
            wj(os.path.join(cc, f"mit_fsC_{a}_s{s}.json"), {"CC": c})
    f1, f2, f4 = ER.swap_validity("fsC", "R1", cc), ER.swap_validity("fsC", "R2", cc), ER.swap_validity("fsC", "R4", cc)
    check("accuracy flag as analyze_re.accuracy_valid (0.03 margin, published rounding); R4 valid",
          f1["valid"] is True and f2["valid"] is False and f4["valid"] is True)
    check("accuracy flag not available when a CC file is missing (no crash)", ER.swap_validity("fsQ", "R1", cc)["valid"] is None)
    try:
        AR._load(os.path.join(X.RE, "out", "primary_P5.json")); ok = False
    except RuntimeError:
        ok = True
    check("the read guard refuses any project file (no real score or result is read by these tests)", ok)


# ---------------------------------------------------------------------------------------------- checks and output
def test_leaf_diff():
    out("--- reproduction-check comparison")
    a = dict(x=1.0, y=[1.0, 2.0], z=dict(b=True, s="3", n=None, i=float("inf")), only_a=5.0)
    b = dict(x=1.0, y=[1.0, 2.0], z=dict(b=True, s="3", n=None, i=float("inf")), only_b=7.0)
    d = ER.leaf_diff(a, b)
    check("identical shared leaves: 0 difference, 0 mismatches (keys present in one only are ignored)",
          d["max_abs_diff"] == 0.0 and d["n_mismatched"] == 0 and d["n_numeric_compared"] == 3)
    b2 = json.loads(json.dumps(b)); b2["y"][1] = 2.5; b2["z"]["b"] = False
    d2 = ER.leaf_diff(a, b2)
    check("a numeric difference and a boolean mismatch are both found", abs(d2["max_abs_diff"] - 0.5) < 1e-15
          and d2["n_mismatched"] == 1 and "/z/b" in d2["mismatched"][0])
    p = ER.plain(dict(a=np.float64(1.5), b=np.bool_(True), c=(1, 2), d=np.array([1.0, np.inf])))
    check("plain(): numpy to Python, tuples to lists, infinity kept (as analyze_re.jdump writes it)",
          p == dict(a=1.5, b=True, c=[1, 2], d=[1.0, float("inf")]))


def test_markdown(D, units):
    out("--- markdown summary from a synthetic JSON record")
    res = dict(units=units, run=dict(started_utc="t0", finished_utc="t1", wall_s=1.0),
               checks=dict(P5_reproduction=dict(max_abs_diff=0.0, n_numeric_compared=10, n_mismatched=0),
                           nulls_theta20_vs_P5_partial=dict(max_abs_diff=0.0, n_numeric_compared=10, n_mismatched=0),
                           msdb_md5=ER.MSDB_MD5, msdb_md5_ok=True, n_units=len(units),
                           n_units_ok=sum(u.get("status") == "ok" for u in units.values())),
               counts=ER.counts(units), notes_md=ER.NOTES_MD)
    rec = json.loads(json.dumps(X._clean(dict(created_utc="t", code={"explore/explore_rigid.py": "0" * 64}, result=res))))
    p = os.path.join(SCRATCH, "md", "x.md")
    os.makedirs(os.path.dirname(p))
    ER.write_md(rec, p)
    txt = open(p).read()
    check("markdown written, labelled exploratory, with the IIE and MSDB tables, the nulls and the readouts",
          "exploratory" in txt and "### DeepGaze IIE" in txt and "### DeepGaze MSDB" in txt and "corner_s8" in txt
          and "| fs_a R4 |" in txt and "| 45 |" in txt and "P5 (decision-bearing)" in txt)
    check("markdown table cells never contain a unit name with a literal '|' (it would split the row)",
          "fs_a|R4" not in txt and "reg_L12|R4" not in txt)
    rows = [ln for ln in txt.splitlines() if ln.startswith("| 10 | exploratory |")]
    want = [ER.share_flag(units[m]["theta"]["10"]["rigid_share"]["stable"],
                          units[m]["theta"]["10"]["rigid_share"]["denominator_excludes_zero"]) for m in ER.MODELS]
    check("markdown shows the s* flag of IIE and MSDB (engine blocks) derived from stable and the denominator check",
          len(rows) == 2 and all((f is None and "(unstable)" not in r and "(not reportable" not in r) or
                                 (f is not None and f"({f})" in r) for r, f in zip(rows, want)), want)
    check("markdown shows a not-computable readout with its reason", "not computable: score files missing" in txt)
    check("markdown marks intervals that exclude 0 with '*' only when they do",
          ER.lin_short(dict(mean=0.1, ci=[0.05, 0.2])).endswith("*") and not
          ER.lin_short(dict(mean=0.1, ci=[-0.05, 0.2])).endswith("*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "tests_explore_rigid_output.txt"))
    a = ap.parse_args()
    t0 = time.time()
    out(f"test_explore_rigid.py, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, synthetic inputs only, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    rng = np.random.RandomState(7)
    D = SynthDesign()
    test_imports_and_guard()
    test_algebra(D, rng)
    units = test_engine_identity(D, rng)
    test_share(D, rng)
    test_loaders(D, rng)
    test_leaf_diff()
    test_markdown(D, units)
    check("no torch imported by the whole test run", "torch" not in sys.modules)
    out(f"{len(RES)} checks, {RES.count(False)} failed, {time.time() - t0:.1f}s")
    with open(a.out, "w") as f:
        f.write("\n".join(LINES) + "\n")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
