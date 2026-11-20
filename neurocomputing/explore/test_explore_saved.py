"""
Unit tests of explore/explore_saved.py (RUN_PLAN.md S3) on synthetic inputs only. No score, map, model or result file of
the project is read: analyze_re._load is wrapped so that any read outside this test's scratch folder fails, and the
score folders of excommon are pointed at the scratch folder. No torch import (asserted). One process, 2 threads.
Writes only its output file (--out) and small synthetic JSON files in a new scratch folder
neuro/_trash/test_explore_saved_scratch_<UTC time>/ (inside neuro/, never deleted by the test).
Usage: OMP_NUM_THREADS=2 $PY -B -I explore/test_explore_saved.py --out explore/tests_explore_saved_output.txt
Written 18 November 2026.
"""
import argparse, ast, inspect, json, math, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402
import explore_saved as ES                                                # noqa: E402

AR = X.AR
RES, LINES = [], []
SCRATCH = os.path.join(os.path.dirname(X.RE), "_trash",
                       "test_explore_saved_scratch_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
os.makedirs(SCRATCH, exist_ok=False)
_ORIG_LOAD = AR._load


def _guarded_load(p):
    if not os.path.abspath(p).startswith(SCRATCH + os.sep):
        raise RuntimeError(f"test read outside its scratch folder: {p}")
    return _ORIG_LOAD(p)


AR._load = _guarded_load
X.SWAP_SCORES = os.path.join(SCRATCH, "swap")
X.REG_SCORES = os.path.join(SCRATCH, "reg")
os.makedirs(X.SWAP_SCORES); os.makedirs(X.REG_SCORES)


def out(s):
    LINES.append(s); print(s, flush=True)


def check(name, ok, detail=""):
    RES.append(bool(ok))
    out(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""))


def synth_meta():
    return {f"{sh}_sr{int(sr * 100)}_{pol}_p{pos}": (sh, sr, pol, pos)
            for sh in ("square", "triangle") for sr in (0.3, 0.5, 0.7) for pol in ("dark", "light") for pos in range(5)}


META = synth_meta()
D = AR.Design(META)
CONDS = ["illusory", "control180", "control90", "mis10", "mis20", "mis30"]
ALT = [f"alt{t}{ph}" for t in (10, 20, 30) for ph in "ab"]


def synth_scores(rng, E=0.1, u=(0.8, 0.6, 0.4), a=(0.95, 0.85, 0.7), e90=0.07, noise=0.01, base=0.5):
    """{name: r} with E(illusory) about E, E(mis theta) about u * E, E(alt theta) about a * E (phases a, b split),
    E(control90) about e90."""
    sc = {}
    for c, (sh, sr, pol, pos) in META.items():
        b = base + noise * rng.randn()
        sc[f"{c}_control180"] = b
        sc[f"{c}_illusory"] = b + E + noise * rng.randn()
        sc[f"{c}_control90"] = sc[f"{c}_illusory"] - e90 + noise * rng.randn()
        for i, t in enumerate((10, 20, 30)):
            sc[f"{c}_mis{t}"] = b + u[i] * E + noise * rng.randn()
            m = b + a[i] * E + noise * rng.randn()
            d = 0.01 * rng.randn()
            sc[f"{c}_alt{t}a"], sc[f"{c}_alt{t}b"] = m + d, m - d
    return sc


def write_heads(folder, tag, per_seed_size):
    """per_seed_size: {(seed, size): {name: r}} -> stim_v3_ and stim_alt_ files {size: {measure: {name: v}}}."""
    for st in ("stim_v3", "stim_alt"):
        for seed in (0, 1, 2):
            d = {}
            for z in ("336", "448"):
                sc = per_seed_size[(seed, z)]
                names = [n for n in sc if ("_alt" in n) == (st == "stim_alt")]
                d[z] = {"r_inside": {n: sc[n] for n in names}, "f_core30": {n: 0.5 * sc[n] for n in names},
                        "inside": {n: sc[n] + 1e-3 for n in names}}
            json.dump(d, open(os.path.join(folder, f"{st}_{tag}_s{seed}.json"), "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "tests_explore_saved_output.txt"))
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(11)
    out(f"test_explore_saved.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; scratch {os.path.relpath(SCRATCH, X.RE)}")

    # 1. no torch, guard first in main
    check("no torch imported", "torch" not in sys.modules)
    src = inspect.getsource(ES.main)
    check("main calls require_public before reading any score",
          src.index("require_public") < min(src.index(k) for k in ("configs_meta", "load_json", "cell_getter")))
    tree = ast.parse(open(os.path.join(HERE, "explore_saved.py")).read())
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    check("explore_saved imports no torch", "torch" not in imported)

    # 2. two_level equals the cbb/check_seed_bootstrap.py code (copied below) on synthetic per-seed arrays
    B = 2000
    idx = AR.two_stage_idx(D.K, D.P, B, 2026)
    per = 0.1 + 0.05 * rng.randn(3, D.K, D.P)
    r = ES.two_level(per, idx, B=B)
    ci_, pi_ = idx
    si = np.random.RandomState(2027).randint(0, 3, (B, 3))
    plan = per.mean(0)[ci_[:, :, None], pi_].mean((1, 2))
    two = np.stack([per[si[:, j]][np.arange(B)[:, None, None], ci_[:, :, None], pi_] for j in range(3)]).mean(0).mean((1, 2))
    cp, ct = np.percentile(plan, [2.5, 97.5]), np.percentile(two, [2.5, 97.5])
    check("two_level plan interval equals the cbb code", np.allclose(r["ci_plan"], cp, atol=0, rtol=0))
    check("two_level two-level interval equals the cbb code", np.allclose(r["ci_two_level"], ct, atol=0, rtol=0))
    same = np.repeat(per[:1], 3, 0)
    r2 = ES.two_level(same, idx, B=B)
    check("two_level with identical seeds: both intervals equal", np.allclose(r2["ci_plan"], r2["ci_two_level"]))
    wide = per.copy(); wide[1] += 0.2; wide[2] -= 0.2
    r3 = ES.two_level(wide, idx, B=B)
    check("two_level widens with seed spread",
          (r3["ci_two_level"][1] - r3["ci_two_level"][0]) > (r3["ci_plan"][1] - r3["ci_plan"][0]))

    # 3. A4.1 block on synthetic scores with known structure
    sc = synth_scores(rng, noise=0.002)
    v = AR.getter(sc)
    b = ES.a41_block(D, v)
    E_true = np.mean([sc[f"{c}_illusory"] - sc[f"{c}_control180"] for c in META])
    check("A4.1 E equals the mean of illusory minus control180", abs(b["E"]["mean"] - E_true) < 1e-15)
    U_true = np.mean([sc[f"{c}_mis20"] - sc[f"{c}_control180"] for c in META]) / E_true
    check("A4.1 U20 is the ratio of means", abs(b["U20"]["value"] - U_true) < 1e-12)
    A_true = np.mean([0.5 * (sc[f"{c}_alt20a"] + sc[f"{c}_alt20b"]) - sc[f"{c}_control180"] for c in META]) / E_true
    check("A4.1 A20 uses the mean of phases a and b", abs(b["A20"]["value"] - A_true) < 1e-12)
    L_true = np.mean([sc[f"{c}_illusory"] - sc[f"{c}_mis20"] for c in META])
    check("A4.1 L20 equals E(illusory) - E(mis20)", abs(b["L20"]["mean"] - L_true) < 1e-15 and
          abs(b["L20"]["mean"] - E_true * (1 - U_true)) < 1e-12)
    Dr = np.mean([sc[f"{c}_illusory"] - 0.5 * (sc[f"{c}_alt20a"] + sc[f"{c}_alt20b"]) for c in META]) / L_true
    check("A4.1 D20 reported and equal to the drop ratio when L20 excludes 0",
          b["D20"].get("reported") is True and abs(b["D20"]["value"] - Dr) < 1e-12)
    check("A4.1 U20 has BCa, Fieller and the stability flag", all(k in b["U20"] for k in ("ci_bca", "fieller", "stable", "t_den")))

    # 4. D(theta) not computed when the uniform-loss interval includes 0
    sc0 = synth_scores(rng, u=(1.0, 1.0, 1.0), noise=0.01)
    p0 = ES.a43_profile(D, AR.getter(sc0))
    nd = [t for t in (10, 20, 30) if X.excl0(p0[str(t)]["L_uni"]["ci"])]
    check("A4.3 D only where the uniform-loss interval excludes 0",
          all((p0[str(t)]["D"].get("reported") is True) == (t in nd) for t in (10, 20, 30)), f"L excl0 at {nd}")
    check("A4.3 D record names the reason when not computed",
          all("reason" in p0[str(t)]["D"] for t in (10, 20, 30) if t not in nd))
    p1 = ES.a43_profile(D, v)
    check("A4.3 L_uni = E(illusory) - E(mis) and U point equals ratio of means at every angle",
          all(abs(p1[str(t)]["L_uni"]["mean"] - (p1["E"]["mean"] - p1[str(t)]["U"]["value"] * p1["E"]["mean"])) < 1e-12
              for t in (10, 20, 30)))
    check("A4.3 slope has the two-stage percentile interval", "ci" in p1["slope_E"] and "slope_per_unit_sr" in p1["slope_E"])

    # 5. A4.2 ratio
    r42 = ES.a42_block(D, v)
    E90_true = np.mean([sc[f"{c}_illusory"] - sc[f"{c}_control90"] for c in META])
    check("A4.2 E90 against control90", abs(r42["E90"]["mean"] - E90_true) < 1e-15)
    check("A4.2 ratio E90 / E180 is the ratio of means",
          abs(r42["ratio_E90_over_E180"]["value"] - E90_true / E_true) < 1e-12)

    # 6. per-size and per-seed loaders on synthetic head files
    pss = {}
    for seed in (0, 1, 2):
        for z, off in (("336", 0.0), ("448", 0.02)):
            pss[(seed, z)] = {k: x + off + 0.001 * seed for k, x in sc.items()}
    write_heads(X.SWAP_SCORES, "dinov2_L12_R2", pss)
    write_heads(X.REG_SCORES, "dinov2reg_L12_R4", pss)
    nm = next(iter(META)) + "_illusory"
    g336 = ES.cell_getter("dinov2_L12", "R2", sizes=("336",))
    g448 = ES.cell_getter("dinov2_L12", "R2", sizes=("448",))
    gavg = ES.cell_getter("dinov2_L12", "R2")
    gs1 = ES.cell_getter("dinov2_L12", "R2", seeds=(1,))
    c0 = next(iter(META))
    check("per-size getter: 336 px = seed mean at 336", abs(g336(c0, "illusory") - (sc[nm] + 0.001)) < 1e-12)
    check("per-size getter: 448 px = seed mean at 448", abs(g448(c0, "illusory") - (sc[nm] + 0.021)) < 1e-12)
    check("size average = mean of the two sizes", abs(gavg(c0, "illusory") - (sc[nm] + 0.011)) < 1e-12)
    check("per-seed getter averages the two sizes of one seed", abs(gs1(c0, "illusory") - (sc[nm] + 0.011)) < 1e-12)
    check("alt phases read from the stim_alt file", abs(gavg(c0, "alt20") - (0.5 * (sc[c0 + "_alt20a"] + sc[c0 + "_alt20b"]) + 0.011)) < 1e-12)
    greg = ES.cell_getter("dinov2reg_L12", "R4")
    check("registers cells read from the registers folder", abs(greg(c0, "illusory") - gavg(c0, "illusory")) < 1e-15)
    gf = ES.cell_getter("dinov2_L12", "R2", meas="f_core30")
    check("measure f_core30 selected", abs(gf(c0, "illusory") - 0.5 * (sc[nm] + 0.011)) < 1e-12)

    # 7. sweep getter (two sizes averaged), with the published-file lookup redirected to the scratch folder
    sw = {s: {k: {"r_inside": x + (0.0 if s == 336 else 0.04), "f_core30": x} for k, x in sc.items()} for s in (336, 448)}
    for s in (336, 448):
        for st in ("stim_v3", "stim_alt"):
            json.dump({k: vv for k, vv in sw[s].items() if ("_alt" in k) == (st == "stim_alt")},
                      open(os.path.join(SCRATCH, f"sweep_{s}_{st}.json"), "w"))
    orig = X.sweep_published_file
    X.sweep_published_file = lambda bb, L, s, st: os.path.join(SCRATCH, f"sweep_{s}_{st}.json")
    try:
        gsw = ES.sweep_getter("dinov2", 12)
        check("sweep getter averages 336 and 448", abs(gsw(c0, "illusory") - (sc[nm] + 0.02)) < 1e-12)
    finally:
        X.sweep_published_file = orig

    # 8. A10: support-ratio sub-designs, enumeration, smallest p
    Dsr = {sr: ES.sr_design(META, sr) for sr in ES.SRS}
    check("A10 sub-design: 4 clusters x 5 positions at each support ratio", all(d.K == 4 and d.P == 5 for d in Dsr.values()))
    check("A10 clusters are shape x polarity", sorted(str(k[0]) + k[2] for k in Dsr[0.5].keys) ==
          sorted(["squaredark", "squarelight", "triangledark", "trianglelight"]))
    u = ES.a10_unit(Dsr, D, v)
    check("A10 WCU enumerates 1,296 weight vectors", all(u[str(s)]["draws"] == 1296 and u[str(s)]["enumerated"] for s in ES.SRS))
    check("A10 smallest attainable p = 6 / 1,296 = 0.00463",
          abs(u["0.3"]["p_min"] - 6 / 1296) < 1e-15 and abs(ES.P_MIN_4 - 0.0046296) < 1e-6)
    check("A10 mean of the three support-ratio means = overall E (balanced)",
          abs(np.mean([u[str(s)]["mean"] for s in ES.SRS]) - u["all_12_clusters"]["mean"]) < 1e-15)
    strong = {k: (x + (0.3 if k.endswith("_illusory") else 0.0)) for k, x in sc.items()}
    us = ES.a10_unit(Dsr, D, AR.getter(strong))
    check("A10 strong effect reaches p near the enumeration floor", all(us[str(s)]["p"] <= 2 * 3 ** 4 / 6 ** 4 + 1e-12 for s in ES.SRS),
          f"p {[round(us[str(s)]['p'], 5) for s in ES.SRS]}")
    check("A10 label", all(u[str(s)]["label"] == "4 clusters; approximate" for s in ES.SRS))
    check("A10 flags a p below the enumeration floor (engine tie tolerance at very large |t|)",
          all(x[str(s)]["p_below_enumeration_floor"] == (x[str(s)]["p"] < 6 / 1296 - 1e-12) for s in ES.SRS for x in (u, us)))
    um = ES.a10_unit(Dsr, D, AR.getter(synth_scores(np.random.RandomState(5), noise=0.05)))
    check("A10 moderate effect: p not below the floor", not any(um[str(s)]["p_below_enumeration_floor"] for s in ES.SRS),
          f"p {[round(um[str(s)]['p'], 5) for s in ES.SRS]}")

    # 9. per-seed count helper and comparison helper
    check("same_side: positive sign needs lower bound > 0", ES.same_side([0.01, 0.2], +1) and not ES.same_side([-0.01, 0.2], +1))
    check("same_side: negative sign needs upper bound < 0", ES.same_side([-0.2, -0.01], -1) and not ES.same_side([0.01, 0.2], -1))
    obj = dict(a=1.0, b=[1, 2.5, dict(c=np.float64(3.0))], d="x", e=None)
    cmp_ok = ES.compare(obj, X._clean(obj))
    check("compare: identical objects", cmp_ok["identical"] and cmp_ok["n_numeric"] == 4)
    bad = X._clean(obj); bad["b"][2]["c"] = 3.0 + 1e-9; bad["d"] = "y"
    cmp_bad = ES.compare(obj, bad)
    check("compare: numeric and string mismatches found", cmp_bad["n_mismatch"] == 2 and abs(cmp_bad["max_abs_diff"] - 1e-9) < 1e-15)
    check("close4 tolerance", ES.close4(0.12344, 0.1234) and not ES.close4(0.12346, 0.1234))

    # 10. ratio difference for item 17 on synthetic data
    sc2 = synth_scores(rng, u=(0.7, 0.5, 0.3), noise=0.002)
    n1, d1 = AR.ratio_parts(D, v, "U20"); n2, d2 = AR.ratio_parts(D, AR.getter(sc2), "U20")
    rd = X.ratio_diff(n1, d1, n2, d2, D)
    check("item 17 ratio difference = U20(a) - U20(b)", abs(rd["diff"] - (n1.mean() / d1.mean() - n2.mean() / d2.mean())) < 1e-12)
    check("item 17 ratio difference carries flags and BCa p", all(k in rd for k in ("flag", "stable", "p_bca", "ci_bca")))

    # 11. a43_seed record
    s0 = ES.a43_seed(D, v)
    check("per-seed record: E with WCU and the published percentile interval",
          "ci" in s0["E"] and "ci" in s0["E_percentile"] and abs(s0["E"]["mean"] - s0["E_percentile"]["mean"]) < 1e-15)

    n_fail = RES.count(False)
    out(f"# {len(RES)} checks, {n_fail} failed, {time.time() - t0:.1f}s")
    open(a.out, "w").write("\n".join(LINES) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
