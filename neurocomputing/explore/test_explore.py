"""
Unit tests of the exploratory scripts in neuro/reanalysis/explore on synthetic inputs (no score, map or model is read;
the only project files read are stimulus metadata and AOIs, the stimulus generator and nn/analyze_nn.py as text).
Light by design: one process, 2 threads, seconds; no torch import (asserted). Writes nothing except its own output
(explore/tests_explore_output.txt, written by the caller with tee).
Usage: OMP_NUM_THREADS=2 $PY -B explore/test_explore.py | tee explore/tests_explore_output.txt
"""
import ast, math, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""), flush=True)


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


# ---------------------------------------------------------------------------------------------- imports and guard
def test_imports_and_guard():
    print("--- imports, no torch, require_public first")
    mods = ["a44_logmass", "a46_calibration", "a11_scale", "a12_kavsek", "a13_region", "a3_reg_tokens_extract",
            "a3_reg_tokens_analyze", "a15a_token_norms", "s13_block12_nulls", "item7_corrected", "a6_margin_curve",
            "mapsrc"]
    for m in mods:
        __import__(m)
    check("importing every explore module imports no torch (heavy code is lazy)", "torch" not in sys.modules,
          f"{len(mods)} modules")
    for f in mods[:-1]:
        tree = ast.parse(open(os.path.join(HERE, f + ".py")).read())
        if f == "a3_reg_tokens_extract":                                 # guarded in the __main__ block
            mainblk = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)][0].body
        else:
            mainblk = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0].body
        pos = [i for i, st in enumerate(mainblk) if "require_public()" in ast.unparse(st)]
        before = [ast.unparse(st) for st in mainblk[:pos[0]]] if pos else ["<none>"]
        ok = bool(pos) and all(("ap" in b and ("argparse" in b or "ap." in b)) for b in before)
        check(f"{f}: require_public() is the first action of main (only argument parsing before it)", ok,
              f"statement {pos[0] if pos else None}")


# ---------------------------------------------------------------------------------------------- bookkeeping
def test_collector_check():
    print("--- Collector and Check")
    C = X.Collector()
    for seed in (0, 1, 2):
        for s in (336, 448):
            C.add("u", seed, s, "a", {"v": seed + (s == 448)})
            C.add("u", seed, s, "b", {"v": 10.0})
    pd = C.per_display("u", "v")
    check("per_display averages over seeds and sizes", abs(pd["a"] - 1.5) < 1e-12 and pd["b"] == 10.0, pd)
    check("per_display within one size averages over seeds", abs(C.per_display("u", "v", 448)["a"] - 2.0) < 1e-12)
    check("sizes() lists the unit's sizes", C.sizes("u", "v") == [336, 448])
    C.add("u", 0, 336, "c", {"v": 1.0})
    try:
        C.per_display("u", "v"); ok = False
    except AssertionError:
        ok = True
    check("an incomplete (seed, size) set raises instead of silently changing the averaging", ok)
    K = X.Check()
    K.add("swap", "u1", 0.50000, 0.50005); K.add("swap", "u2", 0.5, 0.5003); K.add("swap", "u3", 0.5, None)
    check("Check: a unit within 1e-4 passes, one at 3e-4 fails, the group fails", K.unit_ok("swap", "u1")
          and not K.unit_ok("swap", "u2") and not K.group_ok("swap"), K.report()["swap"])
    check("Check: a unit with no published value is not 'ok'", not K.unit_ok("swap", "u3"))
    o = X._clean({"a": float("nan"), "b": np.float32(1.5), "c": np.array([1, 2]), "d": (np.bool_(True),)})
    check("_clean: NaN -> None, numpy -> Python", o == {"a": None, "b": 1.5, "c": [1, 2], "d": [True]}, o)


# ---------------------------------------------------------------------------------------------- maps
def test_maps_rank_q16():
    print("--- rank equalisation, q16, r")
    r = X.rank_equalise(np.array([[1.0, 2.0], [2.0, 3.0]]))
    check("rank equalisation: rank / N with ties averaged", np.allclose(r, [[0.25, 0.625], [0.625, 1.0]]), r.tolist())
    m = np.full((X.H, X.W), 3.0)
    ins, ind = np.arange(0, 1000), np.arange(5000, 7000)
    check("r of a constant map = area ratio", abs(X.r_from(m, ins, ind) - 1000 / 3000) < 1e-12)
    rng = np.random.RandomState(0)
    a = rng.rand(X.H, X.W).astype(np.float32) * 7
    q = X.q16(a)
    check("q16 normalises by the maximum in float16", q.dtype == np.float16 and float(q.max()) == 1.0)
    check("safe_log: log of a zero mass is NaN", math.isnan(X.safe_log(0.0)) and abs(X.safe_log(math.e) - 1) < 1e-12)


def test_pub_readout_and_engine():
    print("--- mapsrc published-form map and Engine.emit")
    import cv2
    import mapsrc as M
    rng = np.random.RandomState(1)
    sig = (1 / (1 + np.exp(-rng.randn(336, 336) * 3))).astype(np.float32)
    pub = M._pub_readout("swap", sig)
    man = np.clip(X.q16(cv2.resize(sig.astype(np.float16).astype(np.float32), (X.W, X.H))).astype(np.float32)
                  .astype(np.float64), 0, None)
    check("swap/registers form = q16(resize(float16(sigmoid))) (eval_full.score_map)", np.array_equal(pub, man))
    pub2 = M._pub_readout("sweep", sig)
    man2 = X.clip64(X.q16(cv2.resize(sig, (X.W, X.H))))
    check("sweep form = q16(resize(sigmoid)) (score_nn / score_rigid)", np.array_equal(pub2, man2))
    disp = dict(stimset="stim_v3", name="cfg_illusory", cfg="cfg", cond="illusory", inside=np.arange(100),
                ind=np.arange(1000, 1300))
    r_true = X.r_from(pub, disp["inside"], disp["ind"])
    saved = M.published_r
    M.published_r = lambda g, u, sd, s, st: {"cfg_illusory": r_true + 5e-5}
    try:
        eng = M.Engine()
        eng.emit("swap", "dinov2_L12|R4", 0, 336, disp, pub, dict(logit=None, sig=sig, size=336),
                 lambda ctx, p, raw: {"x": 2.0})
        ok1 = eng.K.unit_ok("swap", "dinov2_L12|R4") and eng.C.per_display("dinov2_L12|R4", "x") == {"cfg_illusory": 2.0}
        M.published_r = lambda g, u, sd, s, st: {"cfg_illusory": r_true + 5e-4}
        eng2 = M.Engine()
        eng2.emit("swap", "dinov2_L12|R4", 0, 336, disp, pub, {}, lambda ctx, p, raw: {"x": 1.0})
        ok2 = not eng2.K.unit_ok("swap", "dinov2_L12|R4")
    finally:
        M.published_r = saved
    check("Engine.emit stores fn's values and checks r_inside against the published value (5e-5 passes)", ok1)
    check("a 5e-4 difference fails the 1e-4 readout tolerance", ok2)
    ng = M.null_maps(np.full((X.H, X.W), 255, np.float32), 255.0)
    check("null_maps returns the 15 nulls", sorted(ng) == sorted(AR.NULLS), len(ng))


# ---------------------------------------------------------------------------------------------- A4.4
def test_a44():
    print("--- A4.4 log mass")
    import a44_logmass as A
    D = SynthDesign()
    m = np.ones((X.H, X.W)); m[:10] = 3.0
    REG = {("c00p0", "illusory"): dict(inside=np.arange(0, 1024), notch=np.arange(20480, 21504), ink=np.array([], int))}
    out = A.make_fn(REG)(dict(disp=dict(cfg="c00p0", cond="illusory")), m, {})
    tot = m.sum()
    check("log m_R = log(mass in R / total mass)", abs(out["logm_inside"] - math.log(3 * 1024 / tot)) < 1e-12
          and abs(out["logm_notch"] - math.log(1024 / tot)) < 1e-12, out)
    check("an empty or zero-mass region gives NaN (unit's region not computable)", math.isnan(out["logm_ink"]))
    C = X.Collector()
    rng = np.random.RandomState(2)
    true = {}
    for c in D.configs:
        d = 0.2 + 0.05 * rng.randn()
        true[c] = d
        for seed in (0, 1, 2):
            for s in (336, 448):
                b = rng.randn()
                C.add("u", seed, s, f"{c}_illusory", {"logm_inside": b + d + (0.01 if s == 448 else -0.01)})
                C.add("u", seed, s, f"{c}_control180", {"logm_inside": b})
    res = A.contrast(C, "u", D, "logm_inside")
    want = np.mean([np.mean([true[c]]) for c in D.configs])
    # the per-display noise b differs per (seed, size) but cancels only within a cell; compare with the exact average
    exact = D.arr(lambda c: np.mean([C.v["u"]["logm_inside"][f"{c}_illusory"][k] for k in C.v["u"]["logm_inside"][f"{c}_illusory"]])
                  - np.mean([C.v["u"]["logm_inside"][f"{c}_control180"][k] for k in C.v["u"]["logm_inside"][f"{c}_control180"]])).mean()
    check("contrast = mean over configurations of the seed- and size-averaged log difference",
          abs(res["mean"] - exact) < 1e-12 and abs(res["mean"] - want) < 1e-12, f"{res['mean']:.6f}")
    check("contrast carries a WCU interval around the mean and a WCR p", res["ci"][0] < res["mean"] < res["ci"][1]
          and res["p"] < 0.05)


def test_regions_on_generated_stimulus():
    print("--- stim_v3 regions on the generator's rendering (phi = 0)")
    RR = X.render_rigid()
    cfg = "square_sr50_dark_p2"
    p = X.kz_params()[cfg]
    t = time.time()
    out = {}
    for cond in ("illusory", "control180"):
        arr = np.array(RR.render(p, cond, 0))
        out[cond] = X.v3_regions(cfg, cond, arr=arr)
    rg = out["illusory"]
    check("notch and ink are disjoint; notch lies inside the inducer discs", not (rg["notch"] & rg["ink"]).any()
          and not (rg["notch"] & ~rg["ind"]).any())
    check("inside and inducer discs are disjoint", not (rg["inside"] & rg["ind"]).any())
    share_in = (out["illusory"]["notch"] & rg["polygon"]).sum() / out["illusory"]["notch"].sum()
    share_out = (out["control180"]["notch"] & rg["polygon"]).sum() / out["control180"]["notch"].sum()
    check("illusory notches point into the polygon, control180 notches out of it", share_in > 0.9 and share_out < 0.1,
          f"{share_in:.3f} / {share_out:.3f}")
    lost = (rg["inside"] & ~rg["polygon"]).sum() / rg["inside"].sum()
    check("the exact polygon covers the inside AOI up to the PIL half-pixel shift (< 2%)", lost < 0.02, f"{lost:.4f}")
    stray = max((out[c]["ink"] & ~rg["ind"]).sum() / out[c]["ink"].sum() for c in out)
    check("ink present and (up to antialiased edge pixels, < 1%) inside the inducer discs", rg["ink"].sum() > 0
          and stray < 0.01, f"{stray:.4f}, {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------------------------- A4.6
def test_a46():
    print("--- A4.6 calibration")
    import cv2
    import a46_calibration as A
    ins, ind = np.arange(0, 2000), np.arange(10000, 13000)
    sal = np.full((X.H, X.W), 0.5)
    r = A.deepgaze_r(sal, ins, ind)
    check("constant density: every transform gives the area ratio", all(abs(v - 0.4) < 1e-12 for v in r.values()), r)
    rng = np.random.RandomState(3)
    sal = rng.rand(X.H, X.W) + 0.01
    r = A.deepgaze_r(sal, ins, ind)
    check("DeepGaze T = 1 reproduces r of the density", abs(r["T1"] - X.r_from(sal, ins, ind)) < 1e-12)
    z = rng.randn(336, 336) * 2                                       # float64, so the shift below is exact
    sig = (1 / (1 + np.exp(-z))).astype(np.float32)
    a = A.readout_r(z, sig, ins, ind)
    b = A.readout_r(z + 3.0, sig, ins, ind)
    check("readout exp(logit/T) is invariant to a constant logit shift (max subtracted)",
          all(abs(a[k] - b[k]) < 1e-9 for k in a if k.startswith("T")))
    direct = X.r_from(cv2.resize(np.exp(z.astype(np.float64) / 1.5), (X.W, X.H)), ins, ind)
    check("readout T = 1.5: exp(z / T) at s x s, then bilinear resize", abs(a["T1.5"] - direct) < 1e-9)
    check("transform list: MSI-Net rank only; others rank and five T",
          A.transforms_for("msinet") == ["pub", "rank"] and len(A.transforms_for("sweep")) == 7)
    per = {k: {"E": {"mean": -0.1}} for k in A.transforms_for("swap")}
    lab, keeps = A.criterion(-0.11, per, "swap")
    per["T2"]["E"]["mean"] = 0.01
    lab2, _ = A.criterion(-0.11, per, "swap")
    check("criterion: robust when every transform keeps the published sign, sensitive when one flips",
          lab == "calibration-robust" and lab2 == "calibration-sensitive")
    D = SynthDesign()
    pd = {}
    for c in D.configs:
        e = 0.1 + 0.01 * rng.randn()
        for cond, v in (("illusory", 0.5 + e), ("control180", 0.5), ("control90", 0.48), ("mis20", 0.5 + 0.6 * e),
                        ("alt20a", 0.5 + 0.8 * e), ("alt20b", 0.5 + 0.8 * e)):
            pd[f"{c}_{cond}"] = v
    st = A.stats_for(pd, D, True)
    check("stats_for: E, U20 = 0.6, A20 = 0.8 (ratios of means)", abs(st["U20"]["value"] - 0.6) < 1e-9
          and abs(st["A20"]["value"] - 0.8) < 1e-9 and st["E"]["ci"][0] > 0, (st["U20"]["value"], st["A20"]["value"]))


# ---------------------------------------------------------------------------------------------- A12, A13
def test_kavsek_region():
    print("--- A12 Kavsek-style AOIs and A13 region variant")
    import a12_kavsek as K12
    yy, xx = np.mgrid[0:X.H, 0:X.W]
    P = (xx >= 400) & (xx < 600) & (yy >= 300) & (yy < 500)
    Dm = np.zeros_like(P)
    for cx, cy in ((400, 300), (600, 300), (400, 500), (600, 500)):
        Dm |= (xx - cx) ** 2 + (yy - cy) ** 2 <= 50 ** 2
    ix = X.pd_index_from(P, Dm)
    U = np.ones((X.H, X.W))
    mu = X.pd_masses(U, ix)
    g = X.uniform_gain(len(ix["P"]), len(ix["DnP"]), len(ix["PnD"]), len(ix["D"]))
    check("uniform map: E_K equals the area gain exactly (E_K minus gain = 0)",
          abs((X.r_K(mu, "illusory") - X.r_K(mu, "control180")) - g) < 1e-12, f"gain {g:.4f}")
    check("uniform map: r_N = |P| / (|P| + |D minus P|), the same AOIs in both conditions (E_N = 0)",
          abs(X.r_N(mu) - len(ix["P"]) / (len(ix["P"]) + len(ix["DnP"]))) < 1e-12)
    rng = np.random.RandomState(4)
    s = rng.rand(X.H, X.W).astype(np.float32)
    sys.path.insert(0, os.path.join(X.RE, "scoring"))
    import common_rigid as CR
    a = dict(polygon=P, ind=Dm, notch_illusory=np.zeros_like(P), ink_illusory=np.zeros_like(P),
             notch_control180=np.zeros_like(P), ink_control180=np.zeros_like(P))
    e_i = CR.measures_extra(s, a, dict(cond="rig20_illusory", base_cond="illusory"))
    e_c = CR.measures_extra(s, a, dict(cond="rig20_control180", base_cond="control180"))
    m = X.pd_masses(X.clip64(s), ix)
    check("r_K and r_N equal common_rigid.measures_extra (the stim_rigid scoring pass)",
          abs(e_i["r_K"] - X.r_K(m, "illusory")) < 1e-12 and abs(e_c["r_K"] - X.r_K(m, "control180")) < 1e-12
          and abs(e_i["r_N"] - X.r_N(m)) < 1e-12 and abs(e_c["r_N"] - X.r_N(m)) < 1e-12)
    D = SynthDesign()
    PD = {c: ix for c in D.configs}
    G0 = K12.gains_v3(PD, D)
    check("gains_v3: per-configuration gain from the pixel counts", np.allclose(G0, g))
    out = K12.make_fn(PD)(dict(disp=dict(cfg="c00p0", cond="control180")), X.clip64(s), {})
    check("A12 per-display fn returns r_K of the condition", abs(out["r_K"] - X.r_K(m, "control180")) < 1e-12)


def test_slope():
    print("--- support-ratio slope (published two-stage percentile method)")
    D = SynthDesign()
    v = D.arr(lambda c: 0.1 + 0.25 * D.sr[int(c[1:3])])
    s = X.slope_pct(v, D)
    check("exact linear effect: slope 0.25 per unit support ratio, degenerate interval", abs(s["slope_per_unit_sr"] - 0.25)
          < 1e-12 and abs(s["ci"][0] - 0.25) < 1e-9 and abs(s["ci"][1] - 0.25) < 1e-9, s["slope_per_unit_sr"])
    rng = np.random.RandomState(5)
    v2 = v + 0.02 * rng.randn(*v.shape)
    s2 = X.slope_pct(v2, D)
    check("noisy effect: the interval contains the point estimate", s2["ci"][0] < s2["slope_per_unit_sr"] < s2["ci"][1])


# ---------------------------------------------------------------------------------------------- A11
def test_a11():
    print("--- A11 NNLS mixture fit")
    import cv2
    import a11_scale as A
    g = np.full((X.H, X.W), 255, np.float32)
    for cx, cy in ((300, 250), (520, 250), (300, 470), (520, 470)):
        cv2.circle(g, (cx, cy), 45, 0, -1)
    sig = (4, 16, 40)
    RG = A.regressors(g, 255.0, sigmas=sig)
    ink = np.abs(g - 255.0) / 255.0
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    full = 2.0 * cv2.GaussianBlur(har, (0, 0), 16) / cv2.GaussianBlur(har, (0, 0), 16).max() + \
        0.5 * cv2.GaussianBlur(ink, (0, 0), 16) + 0.1
    y = A.block_mean(full / full.mean())
    fits = {s: [A.fit(y, *RG[s]) + ("stim_v3",)] for s in sig}
    r2 = {s: 1 - fits[s][0][0] / fits[s][0][1] for s in sig}
    check("a map built from sigma-16 regressors is fitted exactly at sigma 16 (R2 = 1)", r2[16] > 1 - 1e-9, r2)
    pr = A.profile(fits, sigmas=sig)
    check("profile: best sigma 16, pooled R2 profile and per-set profile", pr["best_sigma"] == 16
          and pr["by_stimset"]["stim_v3"]["best_sigma"] == 16, pr["pooled_R2_profile"])
    check("pooled R2 = 1 - sum SSE / sum SST", abs(A.pooled([(1.0, 4.0), (1.0, 6.0)]) - 0.8) < 1e-12)
    check("block mean grid is 192 x 256", A.block_mean(np.zeros((X.H, X.W))).shape == (192, 256))


# ---------------------------------------------------------------------------------------------- S13
def test_s13():
    print("--- S13 block 12 minus ink and edge nulls")
    import s13_block12_nulls as S
    D = SynthDesign()
    rng = np.random.RandomState(6)

    def mk(E, U, noise):
        tab = {}
        for c in D.configs:
            e = E + noise * rng.randn()
            tab[(c, "control180")] = 0.5
            tab[(c, "illusory")] = 0.5 + e
            tab[(c, "mis20")] = 0.5 + U * e
        return lambda c, cond: tab[(c, cond)]
    b12, nk = mk(-0.10, 0.7, 0.01), mk(-0.08, 0.8, 0.005)
    out = S.s13(D, b12, {"ink_s8": nk})
    r = out["vs_null"]["ink_s8"]
    check("U20 difference: point = U20(block 12) - U20(null)", abs(r["U20_difference"]["diff"] - (0.7 - 0.8)) < 1e-9,
          r["U20_difference"]["diff"])
    check("both denominators screened (|t| >= 4, interval excludes 0) and BCa interval around the point",
          r["U20_difference"]["stable"] and r["U20_difference"]["denominators_exclude_zero"]
          and r["U20_difference"]["ci_bca"][0] < r["U20_difference"]["diff"] < r["U20_difference"]["ci_bca"][1])
    E1 = D.arr(lambda c: b12(c, "illusory") - b12(c, "control180")).mean()
    E2 = D.arr(lambda c: nk(c, "illusory") - nk(c, "control180")).mean()
    want = -(E1 - 0.7 * E1) * 1 + (E2 - 0.8 * E2)                    # sign(E) = -1 for both: -L1 - (-L2)
    check("sign-adjusted loss difference uses each model's own sign(E)", abs(r["sign_adjusted_L20_difference"]["mean"]
                                                                            - want) < 1e-9,
          f"{r['sign_adjusted_L20_difference']['mean']:.5f} vs {want:.5f}")
    check("signs recorded (-1, -1)", out["block12"]["sign"] == -1.0 and r["null_sign"] == -1.0)


# ---------------------------------------------------------------------------------------------- item 7
def test_item7():
    print("--- secondary item 7, corrected sign")
    import item7_corrected as I
    src = open(os.path.join(X.ROOT, "nn", "analyze_nn.py")).read()
    me = open(os.path.join(HERE, "item7_corrected.py")).read()
    tree = ast.parse(me)
    for fn in ("parts_for", "spearman", "perm_p", "boot_rho", "ols_r2", "adj", "assoc"):
        node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn][0]
        seg = ast.get_source_segment(me, node)
        check(f"{fn} is a verbatim copy of nn/analyze_nn.py", seg in src, f"{len(seg)} chars")
    sp, sb = I.l20_sign(round(-0.0000288, 4), -0.0000288, "stored")
    check("stored logic: the rounded E of DINOv2-B L2 (-0.0) gives point sign +1", sp == 1.0)
    check("stored logic: and a bootstrap row of zeros (sign -0.0)", float(np.abs(sb * np.ones(5)).sum()) == 0.0)
    sp2, sb2 = I.l20_sign(round(-0.0000288, 4), -0.0000288, "corrected")
    check("corrected logic: sign of the unrounded E (-1) for point and bootstrap", sp2 == -1.0 and sb2 == -1.0)
    rng = np.random.RandomState(7)
    x = rng.uniform(230, 330, 18); y = rng.randn(18) * 0.01
    a = I.assoc([f"r{i}" for i in range(18)], x, y, rng.randn(18, 300) * 0.01 + y[:, None],
                ["a"] * 6 + ["b"] * 6 + ["c"] * 6)
    check("assoc runs: rho, 100,000-permutation p, bootstrap CI, within-network T", -1 <= a["rho"] <= 1
          and a["perm_kind"] == "100000 random permutations" and len(a["rho_ci"]) == 2 and a["T"] is not None)


# ---------------------------------------------------------------------------------------------- A6
def test_a6():
    print("--- A6 margin curve")
    import a6_margin_curve as A
    check("13 margins 0 to 0.06 in steps of 0.005", A.MARGINS[0] == 0.0 and A.MARGINS[-1] == 0.06 and len(A.MARGINS) == 13)
    cells = {("A", "R4"): dict(cc=0.7255, E=-0.11, ci=[-0.14, -0.08]),
             ("A", "R1"): dict(cc=0.7235, E=-0.13, ci=[-0.16, -0.10]),
             ("A", "R2"): dict(cc=0.6950, E=-0.05, ci=[-0.10, 0.01]),
             ("B", "R4"): dict(cc=0.7400, E=0.09, ci=[0.07, 0.11]),
             ("B", "R1"): dict(cc=0.6980, E=0.05, ci=[0.03, 0.07]),
             ("B", "R2"): dict(cc=0.7350, E=-0.01, ci=[-0.03, 0.01])}
    rows, ex = A.curve(cells)
    r = {x["margin"]: x for x in rows}
    check("margin 0: no non-R4 cell is valid", r[0.0]["n_valid"] == 0)
    check("margin 0.005: two valid, point agreement 1 of 2, same-side interval 1 of 2",
          r[0.005]["n_valid"] == 2 and r[0.005]["n_sign_agree_point"] == 1 and r[0.005]["n_same_side_interval_excl0"] == 1,
          r[0.005])
    check("margin 0.06: all four valid", r[0.06]["n_valid"] == 4)
    check("exit margins follow the published rounding (gaps 0.002, 0.005, 0.0305, 0.042)",
          ex["A|R1"]["smallest_grid_margin_valid"] == 0.005 and ex["B|R2"]["smallest_grid_margin_valid"] == 0.005
          and ex["A|R2"]["smallest_grid_margin_valid"] == 0.035 and ex["B|R1"]["smallest_grid_margin_valid"] == 0.045,
          {k: v["smallest_grid_margin_valid"] for k, v in ex.items()})
    check("0.03 validity as analyze_re.accuracy_valid", ex["A|R2"]["valid_at_0p03"] is False and ex["A|R1"]["valid_at_0p03"])


# ---------------------------------------------------------------------------------------------- A3 (iv)
def test_a3():
    print("--- A3 (iv) registers token analysis on synthetic token components")
    import a3_reg_tokens_extract as EX
    import a3_reg_tokens_analyze as A
    rng = np.random.RandomState(8)
    hs = [rng.randn(7, 5 + 16, 8) for _ in range(3)]
    names, tests = list(EX.K_IMGS), list(EX.K_TESTS)
    dot, nt, nrc = EX.token_components(hs, names, tests, drop=5)
    f = hs[1][:, 5:, :]; d = f - f[0:1]; drc = d[names.index("realC")]
    check("token components: 5 tokens dropped, dot / nt / nrc as run_vit.py",
          dot.shape == (3, 5, 16) and np.allclose(nrc[1], (drc * drc).sum(-1), rtol=1e-6)
          and np.allclose(dot[1, 0], (d[names.index("illusory")] * drc).sum(-1), rtol=1e-6, atol=1e-6))
    D = X.design()
    meta = AR.configs_meta()
    yy, xx = np.mgrid[0:X.H, 0:X.W]
    inside = (xx >= 412) & (xx < 612) & (yy >= 284) & (yy < 484)
    ind = np.zeros_like(inside)
    for cx, cy in ((412, 284), (612, 284), (412, 484), (612, 484)):
        ind |= (xx - cx) ** 2 + (yy - cy) ** 2 <= 60 ** 2
    inside &= ~ind
    aoi = dict(inside=inside, ind=ind)
    CFGS = [dict(config=c, illusion="kanizsa", cell=meta[c][:3], pos=meta[c][3], aoi="AOI") for c in D.configs]
    store = {}
    for S in X.SIZES:
        G, P = S // 14, 14
        fp = EX.footprints(G, G, P, P, X.H / S, X.W / S)
        inf, far = EX.region_masks(fp, aoi, EX.figure_dist(aoi))
        inf, far = inf.ravel(), far.ravel()
        geo = A.token_geo(dict(config="geo", aoi="AOI"), G, P, S, load=lambda p: aoi)
        out_tok = geo[1]
        for k, c in enumerate(D.configs):
            nrc_ = np.ones((13, G * G), np.float32)
            dot_ = np.zeros((13, 5, G * G), np.float32)
            e_int = 0.5 + 0.02 * rng.randn()
            dot_[:, 0, inf] = e_int                                   # illusory test, interior tokens
            dot_[:, 0, out_tok] = 0.2                                 # illusory test, outside tokens
            dot_[:, 1, inf] = 0.1                                     # control90 "test"
            dot_[:, 3, inf] = 0.5 * e_int                             # mis20 test, interior
            dot_[:, 3, out_tok] = 0.1                                 # mis20 test, outside
            store[os.path.join(A.TOK, f"{A.NET}_s{S}", c + ".npz")] = dict(
                dot=dot_, nt=np.ones_like(dot_), nrc=nrc_, same=np.ones((6, G * G), bool), inside_full=inf, far=far,
                imgs=np.array(EX.K_IMGS), tests=np.array(EX.K_TESTS), grid=np.array([G, P, S]))
    load = lambda p: aoi if p == "AOI" else store[p]
    A._geo.clear()
    t = time.time()
    res = A.analyse(CFGS, load=load)
    blk = res["i_prespecified"]["size_average"]["12"]
    check("(i) interior about 0.5 against nulls 0.2 (background) and 0.1 (control90): figure-specific",
          abs(blk["interior"]["mean"] - 0.5) < 0.01 and blk["figure_specific"] and blk["figure_specific_published_rounding"],
          f"{blk['interior']['mean']:.4f}")
    nr = res["ii_near_exterior"]["12"]
    check("(ii) near-exterior set: interior 0.5 against ext_near 0.2, passes", abs(nr["ext_near"]["mean"] - 0.2) < 1e-6
          and nr["passes_near_exterior"], f"ext_near {nr['ext_near']['mean']:.4f}")
    ex = res["iii_paired_excess"]["12"]
    check("(iii) paired excess about 0.3 with a WCU interval; retention under mis20 about (0.25 - 0.1) / 0.3",
          abs(ex["excess"]["mean"] - 0.3) < 0.01 and ex["excess"]["ci"][0] > 0
          and abs(ex["retention_mis20"]["value"] - (0.25 - 0.1) / 0.3) < 0.03,
          f"{ex['excess']['mean']:.4f}, {ex['retention_mis20']['value']:.4f}, {time.time() - t:.1f}s")
    full_in, full_out, mind = A.token_geo(dict(config="geo", aoi="AOI"), 24, 14, 336, load=lambda p: aoi)
    check("ext_near tokens lie wholly outside the figure", not (full_out & full_in).any() and full_out.sum() > 0)


# ---------------------------------------------------------------------------------------------- A15(a)
def test_a15a():
    print("--- A15(a) token norms and the Critic fix 18 condition")
    import a15a_token_norms as A
    n = np.ones((2, 100)); n[0, 5] = 3.0
    st = A.norm_stats(n)
    check("norm stats: max/median, share and any above twice the median", st["max_over_median"] == [3.0, 1.0]
          and st["share_gt2xmedian"] == [0.01, 0.0] and st["any_gt2xmedian"] == [True, False], st)
    rec = lambda any_k: {f"{s}|kanizsa": dict(per_image=dict(any_gt2xmedian=[any_k and s == 448, False])) for s in X.SIZES}
    res = {"clip": {4: rec(False), 24: rec(True)}, "siglip": {24: rec(False)}}
    c = A.condition(res)
    check("condition flags exactly the blocks with a high-norm token on a Kanizsa display",
          c["flagged_blocks"] == ["clip block 24"] and c["test_time_registers_required"], c["flagged_blocks"])
    disp = A.rf_displays()
    check("receptive-field display set: 24 displays, position = cell index mod 5",
          len(disp) == 24 and os.path.basename(disp[0]).endswith("_p0_illusory.png")
          and os.path.basename(disp[2]).split("_")[3] == "p1", os.path.basename(disp[2]))


# ---------------------------------------------------------------------------------------------- rigid layout
def test_rigid_layout():
    print("--- stim_rigid score layouts")
    d = {"336": {"f_inside": {"a": 0.1, "b": 0.2}, "r_K": {"a": 0.5, "b": 0.6}}}
    lay = X.heads_layout(d, 336)
    check("heads layout {size: {measure: {name}}} -> {name: {measure}}", lay == {"a": {"f_inside": 0.1, "r_K": 0.5},
                                                                                 "b": {"f_inside": 0.2, "r_K": 0.6}})
    D = SynthDesign()
    sc = {}
    for seed in (0, 1, 2):
        for s in (336, 448):
            rec = {}
            for c in D.configs:
                rec[X.rig_name(c, 20, "illusory")] = {"f_inside": 0.2 + 0.01 * seed}
                rec[X.rig_name(c, 20, "control180")] = {"f_inside": 0.1}
            sc[(seed, s)] = rec
    pc = X.rigid_per_config("swap", "x|R4", lambda a, b: math.log(a["f_inside"]) - math.log(b["f_inside"]), D,
                            phis=(20,), sc=sc)
    want = np.mean([math.log(0.2 + 0.01 * k) - math.log(0.1) for k in (0, 1, 2)])
    check("rigid_per_config averages fn over the (seed, size) cells per configuration", np.allclose(pc[20], want))
    pc2 = X.rigid_per_config("swap", "x|R4", lambda a, b: 1.0, D, phis=(10,), sc=sc)
    check("a missing rigid display gives NaN for its configuration (no silent substitute)", np.isnan(pc2[10]).all())


def test_metal_guard():
    print("--- Metal guard (read-only ioreg query)")
    r = X.metal_guard()
    check("this test process holds no Metal client (or ioreg is unavailable)", r in (True, None), r)


def main():
    t0 = time.time()
    print(f"test_explore.py; numpy {np.__version__}; threads {os.environ['OMP_NUM_THREADS']}; started "
          f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    for f in (test_imports_and_guard, test_collector_check, test_maps_rank_q16, test_pub_readout_and_engine, test_a44,
              test_regions_on_generated_stimulus, test_a46, test_kavsek_region, test_slope, test_a11, test_s13, test_item7,
              test_a6, test_a3, test_a15a, test_rigid_layout, test_metal_guard):
        f()
    check("no torch imported by the whole test run", "torch" not in sys.modules)
    print(f"{len(RES)} checks, {RES.count(False)} failed, {time.time() - t0:.1f}s")
    sys.exit(1 if not all(RES) else 0)


if __name__ == "__main__":
    main()
