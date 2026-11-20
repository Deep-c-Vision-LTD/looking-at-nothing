"""
Unit tests of explore/report_p6_cells.py on synthetic inputs only. No score, map, model or result file of the project is
read: analyze_re._load is wrapped so that any read outside this test's scratch folder fails, and every data path of the
script (G3, A14, registers and swap score folders, out/primary_P6.json, explore_saved.json, the three results_full.json
files, the map, cache and stim_rigid folders, the output folder) is pointed at the scratch folder. The stored files the
gates compare against are built here: primary_P6.json with analyze_re.p3, p6 and title_decision on test-built inputs,
explore_saved.json with the explore_saved functions, and the results_full.json files with copies of the published
percentile code (v2/readout/analyze.boot and summarise, gpu/analyze_gpu.ratio and its stim_alt block). Code files and
REANALYSIS_RULE.md / PUBLIC_PUSH.txt are read for their hashes only. No torch import (asserted). One process, 2 threads.
Writes only its output file (--out) and synthetic files in a new scratch folder
neuro/_trash/test_report_p6_cells_scratch_<UTC time>/ (inside neuro/, never deleted by the test).
Usage: OMP_NUM_THREADS=2 $PY -B -I explore/test_report_p6_cells.py --out explore/tests_report_p6_cells_output.txt
Written 20 November 2026; revised the same day after an independent code review (item 5, rule 4.3 for per-seed ratios,
never-overwrite guard, stim_alt precondition, ratio context, skewness, yes or no flags, signed ratio values).
"""
import argparse, ast, copy, hashlib, inspect, json, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402
import explore_saved as ES                                                # noqa: E402
import report_p6_cells as RP                                              # noqa: E402

AR = X.AR
RES, LINES = [], []
SCRATCH = os.path.join(os.path.dirname(X.RE), "_trash",
                       "test_report_p6_cells_scratch_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
os.makedirs(SCRATCH, exist_ok=False)
_ORIG_LOAD = AR._load


def _guarded_load(p):
    if not os.path.abspath(p).startswith(SCRATCH + os.sep):
        raise RuntimeError(f"test read outside its scratch folder: {p}")
    return _ORIG_LOAD(p)


AR._load = _guarded_load                                                  # no project data file can be read
S = lambda *a: os.path.join(SCRATCH, *a)                                  # noqa: E731
X.SWAP_SCORES = AR.SWAP_SCORES = S("swap")
X.REG_SCORES = S("reg")
RP.G3_SCORES, RP.A14_SCORES = S("g3"), S("a14")
X.RESULTS = S("results")
RP.PATHS = dict(primary=S("primary_P6.json"), explore_saved=S("explore_saved.json"), rf_g3=S("rf_g3.json"),
                rf_a14=S("rf_a14.json"), rf_reg=S("rf_reg.json"), maps=(S("maps_g3"), S("maps_a14")), cache=S("cache"),
                rigid_heads=S("rigid"))
for d in ("swap", "reg", "g3", "a14", "results", "maps_g3", "maps_a14"):
    os.makedirs(S(d))


def out(s):
    LINES.append(s); print(s, flush=True)


def check(name, ok, detail=""):
    RES.append(bool(ok))
    out(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""))


# ====================================================================== synthetic design and scores
def synth_meta():
    return {f"{sh}_sr{int(sr * 100)}_{pol}_p{pos}": (sh, sr, pol, pos)
            for sh in ("square", "triangle") for sr in (0.3, 0.5, 0.7) for pol in ("dark", "light") for pos in range(5)}


META = synth_meta()
D = AR.Design(META)
SYN = {}                                                                  # (fs, ar) -> {seed: {size: {name: r}}}


def synth_scores(rng, E, u=(0.8, 0.6, 0.4), a=(0.95, 0.85, 0.7), e90=0.07, noise=0.01, base=0.5):
    """{name: r} with E(illusory) about E, E(mis theta) about u * E, E(alt theta) about a * E (phases a, b),
    E(control90) about e90 * sign(E)."""
    sc = {}
    for c in META:
        b = base + noise * rng.randn()
        sc[f"{c}_control180"] = b
        sc[f"{c}_illusory"] = b + E + noise * rng.randn()
        sc[f"{c}_control90"] = sc[f"{c}_illusory"] - e90 * np.sign(E if E != 0 else 1) + noise * rng.randn()
        for i, t in enumerate((10, 20, 30)):
            sc[f"{c}_mis{t}"] = b + u[i] * E + noise * rng.randn()
            m = b + a[i] * E + noise * rng.randn()
            dd = 0.01 * rng.randn()
            sc[f"{c}_alt{t}a"], sc[f"{c}_alt{t}b"] = m + dd, m - dd
    return sc


def write_unit(folder, fs, ar, E, ccs, rng, noise=0.01):
    """Score files {size: {measure: {name: v}}} for stim_v3 and stim_alt, seeds 0, 1, 2, and mit files {'CC': cc}."""
    tag = f"{fs}_{ar}"
    SYN[(fs, ar)] = {}
    for seed in (0, 1, 2):
        per = {z: synth_scores(rng, E + (0.004 if z == "448" else 0.0) + 0.003 * (seed - 1), noise=noise)
               for z in ("336", "448")}
        SYN[(fs, ar)][seed] = per
        for st in ("stim_v3", "stim_alt"):
            d = {}
            for z in ("336", "448"):
                names = [n for n in per[z] if ("_alt" in n) == (st == "stim_alt")]
                d[z] = {"r_inside": {n: per[z][n] for n in names}, "f_core30": {n: 0.5 * per[z][n] for n in names},
                        "inside": {n: per[z][n] + 1e-3 for n in names}}
            json.dump(d, open(os.path.join(folder, f"{st}_{tag}_s{seed}.json"), "w"))
        if ccs is not None:
            json.dump({"CC": ccs[seed], "tag": tag}, open(os.path.join(folder, f"mit_{tag}_s{seed}.json"), "w"))


# new cells: (E, CCs); registers R4 and plain R4 CCs give the rule-fixed thresholds
NEW = {
    "dinov2reg_L6": {"R4": (0.11, [0.7377] * 3), "R1n": (0.06, [0.6990, 0.7000, 0.7010]), "R1": (0.04, [0.69] * 3),
                     "R2": (0.10, [0.7220, 0.7225, 0.7230]), "R3": (0.09, [0.7233] * 3)},
    "dinov2reg_L12": {"R4": (0.047, [0.7280, 0.7282, 0.7284]), "R1n": (0.0, [0.7078] * 3), "R1": (-0.035, [0.7055] * 3),
                      "R2": (0.025, [0.7223] * 3), "R3": (0.02, [0.7264] * 3)},
    "dinov2ms_L7L11": {"R4": (-0.022, [0.7450, 0.7451, 0.7452]), "R1n": (0.012, [0.7279] * 3), "R1": (0.02, [0.7238] * 3),
                       "R2": (-0.019, [0.7366] * 3), "R3": (-0.045, [0.7358] * 3)},
    "dinov2ms_L11": {"R4": (-0.095, [0.7300] * 3), "R1n": (-0.053, [0.7147] * 3), "R1": (-0.03, [0.6950] * 3),
                     "R2": (-0.08, [0.7252] * 3), "R3": (-0.09, [0.7271] * 3)},
}
NOISE = {("dinov2reg_L12", "R1n"): 0.03}


def build_scores():
    rng = np.random.RandomState(20)
    for fs, cells in NEW.items():
        for ar, (E, ccs) in cells.items():
            write_unit(RP.src_dir(fs, ar), fs, ar, E, ccs, rng, noise=NOISE.get((fs, ar), 0.01))
    for f in AR.SWAP_FSETS:                                               # published swap cells (P3 in gate G2)
        for a in AR.ARCHS:
            E = -0.12 if f == "dinov2_L12" else 0.09
            ccs = {("dinov2_L12", "R4"): [0.7255] * 3, ("dinov2_L6", "R4"): [0.7422] * 3}.get((f, a))
            write_unit(X.SWAP_SCORES, f, a, E, ccs, rng)


# ====================================================================== stored files (independent constructions)
def jdump_like(obj, p):
    """As analyze_re.jdump (numpy to Python)."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(dict(created_utc="2026-11-20T00:00:00Z", result=obj), open(p, "w"), indent=1, default=conv)


def build_primary():
    """primary_P6.json as analyze_re.step_primary would write it, from inputs built here with explicit folders."""
    folders = {"dinov2reg_L6": S("g3"), "dinov2reg_L12": S("g3"), "dinov2ms_L7L11": S("a14"), "dinov2ms_L11": S("a14")}
    newsets, cc = {}, {}
    for fs in RP.FSETS:
        heads, ccs = {}, {}
        for ar in AR.ARCHS:
            src = S("reg") if (fs.startswith("dinov2reg") and ar == "R4") else folders[fs]
            heads[ar] = AR.getter(AR.head_scores(src, f"{fs}_{ar}", sets=("stim_v3",)))
            ccs[ar] = float(np.mean([json.load(open(os.path.join(src, f"mit_{fs}_{ar}_s{s}.json")))["CC"] for s in (0, 1, 2)]))
        newsets[fs], cc[fs] = heads, ccs
    p6 = AR.p6(D, newsets, cc)
    heads = AR.load_swap()
    p3 = AR.p3(D, {k: heads[k] for k in AR.P3_CELLS}, {f: heads[(f, "R4")] for f in AR.SWAP_FSETS})
    p3["wording"] = AR.wording_p3(p3)
    jdump_like(dict(P3=p3, P6=p6, title=AR.title_decision(p6, p3)), RP.PATHS["primary"])


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_explore_saved():
    """explore_saved.json for the registers R4 units, computed with the explore_saved functions (item1, item3)."""
    i1, prof, per, tl, i5 = {}, {}, {}, {}, {}
    for fs in ("dinov2reg_L12", "dinov2reg_L6"):
        key = f"{fs}|R4"
        g = dict(avg=ES.cell_getter(fs, "R4"), **{z: ES.cell_getter(fs, "R4", sizes=(z,)) for z in ("336", "448")})
        i1[key] = {f"px{z}": ES.a41_block(D, g[z]) for z in ("336", "448")}
        i1[key]["size_average"] = ES.a41_block(D, g["avg"])
        prof[key] = ES.a43_profile(D, g["avg"])
        per[key] = {str(s): ES.a43_seed(D, ES.cell_getter(fs, "R4", seeds=(s,))) for s in (0, 1, 2)}
        st = np.stack([AR.per_config(D, ES.cell_getter(fs, "R4", seeds=(s,)), "illusory", "control180") for s in (0, 1, 2)])
        tl[key] = ES.two_level(st, D.idx)
        i5.update(ES.item5(D, {key: ES.cell_getter(fs, "R4", meas="f_core30")}))
    rec = dict(code={"explore/explore_saved.py": sha(os.path.join(HERE, "explore_saved.py")),
                     "explore/excommon.py": sha(os.path.join(HERE, "excommon.py"))},
               result=dict(item1_A4_1_per_size=i1, item3_A4_3_profiles=dict(profiles=prof, per_seed=per,
                                                                           two_level_seed_bootstrap=tl),
                           item5_A4_5_core_share=i5))
    json.dump(X._clean(rec), open(RP.PATHS["explore_saved"], "w"))


# copies of the published percentile code (v2/readout/analyze.py boot, summarise, design; gpu/analyze_gpu.py ratio)
def pub_idx(K, P):
    rng = np.random.RandomState(2026)
    return rng.randint(0, K, (10000, K)), rng.randint(0, P, (10000, K, P))


def pub_boot(vals):
    ci, pi = pub_idx(*vals.shape)
    bm = vals[ci[:, :, None], pi].mean((1, 2))
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(mean=round(float(vals.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)])


def pub_ratio(num, den):
    ci, pi = pub_idx(*num.shape)
    bn, bd = num[ci[:, :, None], pi].mean((1, 2)), den[ci[:, :, None], pi].mean((1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    return dict(value=round(float(num.mean() / den.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)])


def pub_cells():
    cells = {}
    for c, (sh, sr, pol, pos) in META.items():
        cells.setdefault((sh, sr, pol), []).append(c)
    return [sorted(v) for _, v in sorted(cells.items(), key=lambda kv: str(kv[0]))]


CELLS = pub_cells()


def pub_effects(per, test, ctrl, size):
    sizes = ("336", "448") if size == "avg" else (size,)
    return np.mean([[[per[s][f"{c}_{test}"] - per[s][f"{c}_{ctrl}"] for c in cell] for cell in CELLS] for s in sizes], 0)


def pub_summarise(fs, ar, test, ctrl, size, scale=1.0):
    per = [scale * pub_effects(SYN[(fs, ar)][sd], test, ctrl, size) for sd in (0, 1, 2)]
    r = pub_boot(np.mean(per, 0))
    r["seed_means"] = [round(float(e.mean()), 4) for e in per]
    r["seed_cis"] = [pub_boot(e)["ci"] for e in per]
    return r


def pub_alt(fs, ar):
    def m(cond):
        def one(c, sd):
            per = SYN[(fs, ar)][sd]
            if cond.startswith("alt"):
                return np.mean([per[z][f"{c}_{cond}{ph}"] for z in ("336", "448") for ph in "ab"])
            return np.mean([per[z][f"{c}_{cond}"] for z in ("336", "448")])
        return np.array([[np.mean([one(c, sd) for sd in (0, 1, 2)]) for c in cell] for cell in CELLS])
    ill, c180 = m("illusory"), m("control180")
    r = dict(effect_illusory=pub_boot(ill - c180))
    for t in (10, 20, 30):
        mis, alt = m(f"mis{t}"), m(f"alt{t}")
        r[f"ratio_uni{t}"] = pub_ratio(mis - c180, ill - c180)
        r[f"ratio_alt{t}"] = pub_ratio(alt - c180, ill - c180)
        r[f"drop_ratio{t}"] = pub_ratio(ill - alt, ill - mis)
    return dict(r_inside=r)


def build_results_full():
    files = dict(rf_g3={}, rf_a14={}, rf_reg={})
    for fs, cells in NEW.items():
        for ar, (E, ccs) in cells.items():
            name = "rf_reg" if (fs.startswith("dinov2reg") and ar == "R4") else ("rf_g3" if fs.startswith("dinov2reg") else "rf_a14")
            rf = files[name]
            for k in ("mit", "stim_v3", "stim_alt"):
                rf.setdefault(k, {}).setdefault(fs, {})
            rf["mit"][fs][ar] = dict(seeds={str(s): round(ccs[s], 4) for s in (0, 1, 2)}, mean=round(float(np.mean(ccs)), 4), n=3)
            kz = {"illusory-control180": {f"{z}|r_inside": pub_summarise(fs, ar, "illusory", "control180", z)
                                          for z in ("avg", "336", "448")},
                  "illusory-control90": {"avg|r_inside": pub_summarise(fs, ar, "illusory", "control90", "avg")}}
            kz["illusory-control180"]["avg|f_core30"] = pub_summarise(fs, ar, "illusory", "control180", "avg", 0.5)
            rf["stim_v3"][fs][ar] = dict(kanizsa=kz)
            rf["stim_alt"][fs][ar] = pub_alt(fs, ar)
    for rf in files.values():                                             # analyze_gpu validity (R4 in the same file)
        for fs, rec in rf["mit"].items():
            r4 = rec.get("R4")
            for ar, v in rec.items():
                if r4:
                    v["threshold"] = round(r4["mean"] - 0.03, 4); v["valid"] = bool(v["mean"] >= v["threshold"])
                else:
                    v["valid"] = None
    for name, rf in files.items():
        json.dump(rf, open(RP.PATHS[name], "w"))


def load_json(p):
    return json.load(open(p))


# ====================================================================== tests
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "tests_report_p6_cells_output.txt"))
    a = ap.parse_args()
    t0 = time.time()
    out(f"test_report_p6_cells.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; scratch "
        f"{os.path.relpath(SCRATCH, X.RE)}")

    # 1. static properties
    check("no torch imported", "torch" not in sys.modules)
    src = open(os.path.join(HERE, "report_p6_cells.py")).read()
    tree = ast.parse(src)
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    check("report_p6_cells imports no torch", "torch" not in imported and "torch" not in src.replace("No torch", ""))
    msrc = inspect.getsource(RP.main)
    check("main calls require_public before anything else",
          msrc.index("require_public") < min(msrc.index(k) for k in ("X.design", "run(", "write_outputs")))
    check("statistics come from analyze_re through excommon and explore_saved (no new interval code)",
          all(f"def {f}(" not in src for f in ("wild_cluster", "bca", "pct_summary", "holm", "fieller", "two_stage_idx")))
    check("items 1 and 3 use the explore_saved functions", RP.ES is ES and "ES.a41_block" in src and
          "ES.a43_profile" in src and "ES.a43_seed" in src and "ES.two_level" in src)
    doc = RP.__doc__
    check("header lists implementation choices (a) to (l)", all(f"  ({c}) " in doc for c in "abcdefghijkl"))
    check("only a missing stored field (KeyError) is reported as one", "except (KeyError, TypeError)" not in src and
          "except KeyError" in src)
    check("main refuses to run when an output exists, before any computation",
          msrc.index("require_public") < msrc.index("existing_outputs") < msrc.index("X.design"))
    check("item 5 uses explore_saved.item5", "ES.item5(" in src)
    check("no em-dash or en-dash in the script", "\u2014" not in src and "\u2013" not in src)
    check("20 units: 18 new cells and 2 registers R4 references",
          len(RP.units()) == 20 and sum(1 for _, _, r in RP.units() if "context only" in r) == 2)

    # 2. source folders as analyze_re.step_primary
    check("registers R4 read from the registers folder", RP.src_dir("dinov2reg_L12", "R4") == X.REG_SCORES)
    check("G3 cells read from gpu/g3", RP.src_dir("dinov2reg_L6", "R1") == RP.G3_SCORES)
    check("A14 cells (R4 included) read from gpu/a14", RP.src_dir("dinov2ms_L11", "R4") == RP.A14_SCORES and
          RP.src_dir("dinov2ms_L7L11", "R1n") == RP.A14_SCORES)

    build_scores()
    build_primary()
    build_explore_saved()
    build_results_full()
    out(f"synthetic inputs written ({time.time() - t0:.1f}s)")

    # 3. getters
    c0 = next(iter(META))
    per = SYN[("dinov2ms_L11", "R2")]
    g = RP.unit_getters("dinov2ms_L11", "R2")
    v336 = np.mean([per[s]["336"][f"{c0}_illusory"] for s in (0, 1, 2)])
    v448 = np.mean([per[s]["448"][f"{c0}_illusory"] for s in (0, 1, 2)])
    check("per-size getter = seed mean at that size", abs(g["336"](c0, "illusory") - v336) < 1e-12 and
          abs(g["448"](c0, "illusory") - v448) < 1e-12)
    check("size average = mean of the two per-size seed means", abs(g["avg"](c0, "illusory") - 0.5 * (v336 + v448)) < 1e-12)
    check("per-seed getter averages the two sizes of one seed",
          abs(g["seed1"](c0, "illusory") - 0.5 * (per[1]["336"][f"{c0}_illusory"] + per[1]["448"][f"{c0}_illusory"])) < 1e-12)
    altv = np.mean([per[s][z][f"{c0}_alt20{ph}"] for s in (0, 1, 2) for z in ("336", "448") for ph in "ab"])
    check("alt20 = phase mean read from the stim_alt files", abs(g["avg"](c0, "alt20") - altv) < 1e-12)

    # 4. the five rule 5.6 items on one unit
    r = RP.rule56_items(D, g)
    e336 = np.mean([np.mean([per[s]["336"][f"{c}_illusory"] - per[s]["336"][f"{c}_control180"] for s in (0, 1, 2)]) for c in META])
    check("per-size E (336) = mean of the per-size differences", abs(r["E_336"]["mean"] - e336) < 1e-12)
    check("per-size E carries the WCU interval and the WCR p (12 clusters)",
          r["E_336"]["G"] == 12 and len(r["E_448"]["ci"]) == 2 and "p" in r["E_448"])
    E = AR.per_config(D, g["avg"], "illusory", "control180")
    E90 = AR.per_config(D, g["avg"], "illusory", "control90")
    check("E against control90 = mean of illusory minus control90", abs(r["E_vs_control90"]["mean"] - E90.mean()) < 1e-15)
    nU = AR.per_config(D, g["avg"], "mis20", "control180")
    nA = AR.per_config(D, g["avg"], "alt20", "control180")
    check("U20 and A20 are ratios of means", abs(r["U20"]["value"] - nU.mean() / E.mean()) < 1e-12 and
          abs(r["A20"]["value"] - nA.mean() / E.mean()) < 1e-12)
    bca_ref = AR.bca(lambda x, y: x / y, [nU, E], D.idx)
    check("U20 BCa interval = analyze_re.bca as in analyze_re.p4 ratio_readout_bca", r["U20"]["ci_bca"] == bca_ref["ci"])
    check("U20 and A20 carry Fieller, |t| of the denominator, stability and reportability",
          all(k in r[m] for m in ("U20", "A20") for k in ("fieller", "t_den", "stable", "reportable")))
    check("continuity interval = analyze_re.pct_summary on E", r["E_continuity_percentile"] == AR.pct_summary(E, D.idx))
    pb = pub_boot(E)
    check("continuity interval reproduces the published v2 boot (4 decimals)",
          all(abs(x - y) <= 5.001e-5 for x, y in zip(r["E_continuity_percentile"]["ci"], pb["ci"])))
    pr = pub_ratio(nU, E)
    check("published-percentile U20 for the gate reproduces analyze_gpu.ratio (4 decimals)",
          all(abs(x - y) <= 5.001e-5 for x, y in zip(r["published_percentile_for_gate"]["U20"]["ci"], pr["ci"])))
    check("seed point estimates average to E", abs(np.mean(r["seed_point_estimates"]) - r["E"]["mean"]) < 1e-12)
    g0 = RP.unit_getters("dinov2reg_L12", "R1n")
    r0 = RP.rule56_items(D, g0)
    check("ratio not reportable exactly when the E interval includes 0",
          r0["U20"]["reportable"] == X.excl0(r0["E"]["ci"]) and r0["A20"]["reportable"] == X.excl0(r0["E"]["ci"]),
          f"E {r0['E']['mean']:+.4f} {r0['E']['ci']}")
    check("synthetic near-zero cell is not reportable (exercises the markdown rule)", r0["U20"]["reportable"] is False)

    # 5. items 1 and 3 are the explore_saved computations
    i1 = RP.item1_unit(D, g)
    check("item 1 per size = explore_saved.a41_block", X._clean(i1["px448"]) == X._clean(ES.a41_block(D, g["448"])))
    i3 = RP.item3_unit(D, g)
    check("item 3 profile = explore_saved.a43_profile", X._clean(i3["profile"]) == X._clean(ES.a43_profile(D, g["avg"])))
    st = np.stack([AR.per_config(D, g[f"seed{s}"], "illusory", "control180") for s in (0, 1, 2)])
    check("item 3 two-level bootstrap = explore_saved.two_level", X._clean(i3["two_level_seed_bootstrap"]) ==
          X._clean(ES.two_level(st, D.idx)))
    check("item 3 per-seed record = explore_saved.a43_seed", X._clean(i3["per_seed"]["2"]) == X._clean(ES.a43_seed(D, g["seed2"])))

    # 6. comparison helper
    obj = dict(a=1.0, b=[1, 2.5, dict(c=3.0)], s="x", t=True, n=None)
    check("gate_compare: identical objects pass", RP.gate_compare(obj, copy.deepcopy(obj))["passed"])
    o2 = copy.deepcopy(obj); o2["b"][2]["c"] = 3.0 + 1e-13
    check("gate_compare: 1e-13 within the 1e-12 tolerance", RP.gate_compare(obj, o2)["passed"])
    o3 = copy.deepcopy(obj); o3["b"][2]["c"] = 3.0 + 1e-9
    check("gate_compare: 1e-9 fails", not RP.gate_compare(obj, o3)["passed"])
    o4 = copy.deepcopy(obj); o4["t"] = False; o4["s"] = "y"; del o4["a"]
    c4 = RP.gate_compare(obj, o4)
    check("gate_compare: bool, string and missing-key mismatches found", c4["n_mismatch"] == 3)
    check("gate_compare: equal infinities pass, inf against a number fails",
          RP.gate_compare(dict(x=float("inf")), dict(x=float("inf")))["passed"] and
          not RP.gate_compare(dict(x=float("inf")), dict(x=1.0))["passed"])

    # 7. full run on synthetic inputs
    t1 = time.time()
    res = RP.run(D)
    out(f"run() on synthetic inputs: {time.time() - t1:.1f}s")
    gt = res["gates"]
    check("G1 passes (reused code hashes)", gt["G1_code"]["passed"])
    check("G2 passes: primary_P6.json reproduced leaf by leaf",
          gt["G2_primary_P6"]["passed"] and gt["G2_primary_P6"]["max_abs_diff"] == 0.0,
          f"{gt['G2_primary_P6']['n_numeric']} numeric leaves")
    check("G3 passes", gt["G3_per_unit"]["passed"], f"{gt['G3_per_unit']['n_checks']} checks")
    names = {row["check"] for row in gt["G3_per_unit"]["rows"]}
    check("G3 covers CC, threshold, validity, E, interval, p, Holm, flags, seeds and rule-fixed values",
          all(any(k in n for n in names) for k in
              ("CC mean", "threshold", "validity", "E mean", "E WCU ci[0]", "E WCU ci[1]", "E p", "Holm p", "keeps sign",
               "opposite-sign interval", "seed point estimates", "R4 mean (4 decimals) as fixed in rule 5.6",
               "plain dinov2_L12 R4 threshold", "plain dinov2_L6 R4 threshold", "R4 sign", "n_valid",
               "E skew_cluster_means")))
    check("G3 checks every unit", {row["unit"] for row in gt["G3_per_unit"]["rows"]} >=
          {f"{fs}|{ar}" for fs, ar, _ in RP.units()})
    g4 = gt["G4_continuity_published_method"]
    check("G4 passes against the published-method copies", g4["passed"],
          "; ".join(f"{k} {v['n_agree']}/{v['n']}" for k, v in g4["categories"].items()))
    check("G4 covers every continuity category",
          set(g4["categories"]) >= {"mit_cc_seed", "mit_cc_mean", "validity_flag_a14", "E_mean", "E_percentile_ci",
                                     "E_seed_mean", "E_seed_percentile_ci", "E_per_size", "E_per_size_percentile_ci",
                                     "E_vs_control90", "E_vs_control90_percentile_ci", "profile_U_A", "U20_A20_value",
                                     "U20_A20_percentile_ci", "item5_f_core30", "item5_f_core30_percentile_ci"})
    i5 = res["item5_A4_5_core_share"]
    gf = RP.getter("dinov2ms_L11", "R2", meas="f_core30", sets=("stim_v3",))
    check("item 5: 20 units, f_core30 effect = explore_saved.item5 (half the r_inside E in the synthetic data)",
          len(i5) == 20 and X._clean(i5["dinov2ms_L11|R2"]["core_share"]) == X._clean(ES.item5(D, {"u": gf})["u"]) and
          abs(i5["dinov2ms_L11|R2"]["core_share"]["mean"] - 0.5 * res["rule_5_6"]["dinov2ms_L11"]["cells"]["R2"]["E_wcu"]["mean"]) < 1e-12)
    check("G5 passes (registers R4 items 1 and 3 equal explore_saved.json)", gt["G5_registers_vs_explore_saved"]["passed"])
    check("internal: rule 5.6 items equal item 1", gt["internal_rule56_equals_item1"]["passed"])
    rs = res["rule_5_6"]
    check("every set has 5 rows (R4 included)", all(len(rs[fs]["cells"]) == 5 for fs in RP.FSETS))
    c = rs["dinov2reg_L6"]["cells"]
    check("invalid cells reported with the flag, never dropped",
          c["R1n"]["accuracy"]["status"] == RP.BELOW and c["R1"]["accuracy"]["status"] == RP.BELOW and
          c["R1n"]["E_336"]["mean"] is not None and c["R1n"]["p_holm"] is None)
    check("a14 below-criterion cell flagged (dinov2ms_L11 R1)", rs["dinov2ms_L11"]["cells"]["R1"]["accuracy"]["valid"] is False)
    p6s = load_json(RP.PATHS["primary"])["result"]["P6"]
    check("Holm p, keeps-sign and opposite-sign flags are those of primary_P6.json",
          all(rs[fs]["cells"][a]["p_holm"] == p6s[fs]["cells"][a].get("p_holm") and
              rs[fs]["cells"][a]["keeps_sign"] == p6s[fs]["cells"][a].get("keeps_sign") and
              rs[fs]["cells"][a]["opposite_sign_interval"] == p6s[fs]["cells"][a].get("opposite_sign_interval")
              for fs in RP.FSETS for a in AR.NEW_ARCHS))
    pt = c["R2"]["accuracy"]["plain_R4_threshold_check"]
    check("registers cells carry the plain-threshold comparison (never decision-bearing)",
          pt["plain_threshold"] == 0.7122 and "never decision-bearing" in pt["note"] and
          "plain_R4_threshold_check" not in rs["dinov2ms_L11"]["cells"]["R2"]["accuracy"])
    check("R4 rows: no threshold, validity or Holm p", rs["dinov2ms_L11"]["cells"]["R4"]["accuracy"]["threshold"] is None and
          rs["dinov2ms_L11"]["cells"]["R4"]["p_holm"] is None)
    cnt = res["item3_A4_3_profiles"]["per_seed_counts"]
    allc = cnt["all_four_sets"]
    check("per-seed counts: 20 units x 3 seeds overall, 18 x 3 new cells only",
          allc["overall"]["n_seed_runs"] == 60 and allc["new_cells_only"]["n_seed_runs"] == 54 and
          allc["reference_R4"]["n_seed_runs"] == 12)
    nvalid = sum(1 for fs in RP.FSETS for a in AR.NEW_ARCHS if rs[fs]["cells"][a]["accuracy"]["valid"])
    check("per-seed counts: valid group = valid non-R4 cells x 3", allc["valid"]["n_seed_runs"] == 3 * nvalid,
          f"{nvalid} valid cells")
    check("per-seed counts never exceed the number of seed runs",
          all(v[k] <= v["n_seed_runs"] for gs in cnt.values() for v in gs.values()
              for k in ("n_point_R4_sign", "n_wcu_excl0_R4_side", "n_percentile_excl0_R4_side")))
    check("not computable: no unit (all files present)", res["not_computable"]["units"] == [])
    oi = {x["item"]: x for x in res["not_computable"]["other_section6_items"]}
    check("not computable list names items 4, 6, 7, 10, 11, 15 with reasons", sorted(oi) == [4, 6, 7, 10, 11, 15] and
          all(x["reason"] for x in oi.values()) and "maps" in oi[4]["reason"] and "stim_rigid" in oi[15]["reason"])
    nh = {x["item"]: x for x in res["not_computed_here"]}
    check("items 2 and 7 (first part) listed as not computed here; item 5 is computed", sorted(nh) == [2, 7])
    rc = res["rule_5_6"]["dinov2ms_L11"]["cells"]["R2"]
    check("rule 5.6 cells carry L20 and L_alt20 next to U20 and A20",
          abs(rc["L20"]["mean"] - res["item1_A4_1_per_size"]["dinov2ms_L11|R2"]["size_average"]["L20"]["mean"]) == 0 and
          abs(rc["L_alt20"]["mean"] - res["item3_A4_3_profiles"]["profiles"]["dinov2ms_L11|R2"]["20"]["L_alt"]["mean"]) == 0)

    # 8. counting logic on a constructed case
    fake = {}
    for k, (m, ci, pc) in {"dinov2ms_L11|R1": (-0.02, [-0.03, -0.01], [-0.03, -0.012]),
                           "dinov2ms_L11|R2": (0.01, [-0.01, 0.03], [0.001, 0.02])}.items():
        fake[k] = {"item3": {"per_seed": {s: {"E": {"mean": m, "ci": ci}, "E_percentile": {"mean": m, "ci": pc}}
                                          for s in ("0", "1", "2")}}}
    sc = RP.seed_counts({"x": {"g": list(fake)}}, fake, {"dinov2ms_L11": -1})["x"]["g"]
    check("count logic: point signs, WCU and percentile sides", sc["n_point_R4_sign"] == 3 and sc["n_wcu_excl0_R4_side"] == 3
          and sc["n_percentile_excl0_R4_side"] == 3 and len(sc["not_counted_wcu"]) == 3, json.dumps(
              {k: sc[k] for k in ("n_point_R4_sign", "n_wcu_excl0_R4_side", "n_percentile_excl0_R4_side")}))

    # 9. gate failures (each stops the run; nothing written)
    stored = load_json(RP.PATHS["primary"])
    bad = copy.deepcopy(stored); bad["result"]["P6"]["dinov2ms_L11"]["cells"]["R2"]["mean"] += 1e-9
    check("G2 fails on a 1e-9 change of a stored E", not RP.gate_primary(D, bad)[0]["passed"])
    ok_ = copy.deepcopy(stored); ok_["result"]["P6"]["dinov2ms_L11"]["cells"]["R2"]["mean"] += 1e-14
    check("G2 passes a 1e-14 change (tolerance 1e-12)", RP.gate_primary(D, ok_)[0]["passed"])
    bad2 = copy.deepcopy(stored); bad2["result"]["title"]["narrowed"] = not bad2["result"]["title"]["narrowed"]
    check("G2 fails on a changed title flag", not RP.gate_primary(D, bad2)[0]["passed"])
    os.replace(RP.PATHS["primary"], RP.PATHS["primary"] + ".orig")
    json.dump(bad, open(RP.PATHS["primary"], "w"))
    before = sorted(os.listdir(X.RESULTS))
    try:
        RP.run(D)
        raised = False
    except RP.GateError as e:
        raised = "G2" in str(e)
    os.replace(RP.PATHS["primary"] + ".orig", RP.PATHS["primary"])
    check("run() raises GateError (G2) and writes nothing", raised and sorted(os.listdir(X.RESULTS)) == before)
    recs = _recs_from(res)
    st6 = copy.deepcopy(stored["result"]["P6"]); st6["dinov2reg_L12"]["cells"]["R3"]["cc"] += 1e-9
    rows = RP.gate_unit("dinov2reg_L12", "R3", recs["dinov2reg_L12|R3"], st6["dinov2reg_L12"])
    check("G3 fails on a 1e-9 change of a stored CC", any(not x["ok"] and x["check"] == "CC mean" for x in rows))
    st6b = copy.deepcopy(stored["result"]["P6"]); st6b["dinov2ms_L7L11"]["cells"]["R2"]["p_holm"] += 1e-6
    check("G3 set-level fails on a changed Holm p", any(not x["ok"] for x in RP.gate_sets(recs, st6b)))
    keep = copy.deepcopy(RP.RULE_REG)
    RP.RULE_REG["dinov2reg_L12"]["r4"] = 0.7283
    check("G3 fails when the rule-fixed registers R4 mean differs",
          any(not x["ok"] for x in RP.gate_rule_fixed(recs, res["plain_threshold_recomputed"])))
    RP.RULE_REG.clear(); RP.RULE_REG.update(keep)
    check("G3 rule-fixed checks pass again after restoring", all(x["ok"] for x in RP.gate_rule_fixed(recs, res["plain_threshold_recomputed"])))
    rfs = dict(g3=load_json(RP.PATHS["rf_g3"]), a14=load_json(RP.PATHS["rf_a14"]), reg=load_json(RP.PATHS["rf_reg"]))
    rfb = copy.deepcopy(rfs); rfb["a14"]["stim_v3"]["dinov2ms_L11"]["R3"]["kanizsa"]["illusory-control90"]["avg|r_inside"]["ci"][0] += 2e-4
    check("G4 fails on a 2e-4 change of a stored percentile bound", not RP.gate_continuity(recs, rfb)["passed"])
    rfc = copy.deepcopy(rfs); del rfc["g3"]["stim_alt"]["dinov2reg_L6"]["R2"]["r_inside"]["ratio_alt20"]
    check("G4 fails on a missing stored value", not RP.gate_continuity(recs, rfc)["passed"])
    rfd = copy.deepcopy(rfs); rfd["a14"]["mit"]["dinov2ms_L11"]["R1"]["valid"] = True
    check("G4 fails on a changed A14 validity flag", not RP.gate_continuity(recs, rfd)["passed"])
    rfe = copy.deepcopy(rfs); rfe["g3"]["stim_v3"]["dinov2reg_L12"]["R2"]["kanizsa"]["illusory-control180"]["avg|f_core30"]["mean"] += 2e-4
    check("G4 fails on a 2e-4 change of a stored f_core30 effect", not RP.gate_continuity(recs, rfe)["passed"])
    es = load_json(RP.PATHS["explore_saved"])
    esd = copy.deepcopy(es); esd["result"]["item5_A4_5_core_share"]["dinov2reg_L12|R4"]["mean"] += 1e-9
    check("G5 fails on a 1e-9 change of a stored item 5 value", not RP.gate_explore_saved(esd, recs)["passed"])
    esb = copy.deepcopy(es); esb["result"]["item1_A4_1_per_size"]["dinov2reg_L6|R4"]["px336"]["E"]["mean"] += 1e-9
    check("G5 fails on a 1e-9 change of a stored explore_saved value", not RP.gate_explore_saved(esb, recs)["passed"])
    esc = copy.deepcopy(es); esc["code"]["explore/excommon.py"] = "0" * 64
    check("G1 fails when a reused file's hash differs", not RP.gate_code(esc)["passed"])

    # 10. not computable paths (one set only, to keep the test short)
    def moved(paths, fn):
        for q in paths:
            os.replace(q, q + ".moved")
        try:
            return fn()
        finally:
            for q in paths:
                os.replace(q + ".moved", q)

    def cu():
        return RP.compute_units(D, res["plain_threshold_recomputed"])

    keep_f = RP.FSETS
    RP.FSETS = ("dinov2ms_L11",)
    A = lambda f: os.path.join(RP.A14_SCORES, f)                          # noqa: E731
    recs2, miss2 = moved([A("stim_v3_dinov2ms_L11_R3_s1.json")], cu)
    check("a unit missing a stim_v3 file (outside P6 too) is not computable, with the file named",
          [m["unit"] for m in miss2] == ["dinov2ms_L11|R3"] and "stim_v3_dinov2ms_L11_R3_s1.json" in miss2[0]["reason"]
          and recs2["dinov2ms_L11|R3"]["not_computable"].startswith("not computable from saved outputs"))
    check("the other units of the set are still computed",
          all("rule_5_6" in recs2[f"dinov2ms_L11|{a}"] and "item5" in recs2[f"dinov2ms_L11|{a}"] for a in ("R4", "R1n", "R1", "R2")))
    try:
        moved([A("stim_alt_dinov2ms_L11_R3_s1.json")], cu)
        alt_stop = False
    except RP.GateError as e:
        alt_stop = "precondition" in str(e) and "stim_alt_dinov2ms_L11_R3_s1.json" in str(e)
    check("a unit inside the P6 family lacking only a stim_alt file stops the run (GateError naming the file)", alt_stop)
    recs3, miss3 = moved([A("mit_dinov2ms_L11_R4_s0.json")], cu)
    check("a missing R4 makes every unit of its set not computable, each with a reason",
          len(miss3) == 5 and all("R4" in m["reason"] or "mit_dinov2ms_L11_R4_s0.json" in m["reason"] for m in miss3))
    recs4, miss4 = moved([A("mit_dinov2ms_L11_R4_s0.json"), A("stim_alt_dinov2ms_L11_R2_s0.json")], cu)
    check("a stim_alt gap in a set whose R4 is missing (set outside P6) is reported, not a stop",
          len(miss4) == 5 and any(m["unit"] == "dinov2ms_L11|R2" and "stim_alt_dinov2ms_L11_R2_s0.json" in m["reason"]
                                  and "R4 reference" in m["reason"] for m in miss4))
    RP.FSETS = keep_f
    before = sorted(os.listdir(X.RESULTS))
    try:
        moved([os.path.join(RP.G3_SCORES, "stim_alt_dinov2reg_L6_R2_s2.json")], lambda: RP.run(D))
        run_stop = False
    except RP.GateError as e:
        run_stop = "precondition" in str(e)
    check("run() with a stim_alt gap inside the P6 family raises GateError after G2 and writes nothing",
          run_stop and sorted(os.listdir(X.RESULTS)) == before)

    # 11. outputs
    res["run"] = dict(start_utc="2026-11-20T00:00:00Z", end_utc="2026-11-20T00:00:10Z", compute_s=10.0,
                      hashed_files_verified=33)
    check("no output exists before the first write", RP.existing_outputs() == [])
    pj = RP.write_outputs(res, 10.0)
    rec = load_json(pj)
    check("JSON written to the output folder with code, rule and commit provenance",
          os.path.dirname(pj) == X.RESULTS and set(rec["code"]) == {"explore/report_p6_cells.py", "explore/excommon.py",
                                                                    "explore/explore_saved.py"} and
          len(rec["rule_sha256"]) == 64 and rec["public_commit"] and rec["label"].startswith("exploratory"))
    mdp = os.path.join(X.RESULTS, RP.OUT_MD)
    md = open(mdp).read()
    snap = (open(pj, "rb").read(), open(mdp, "rb").read())
    try:
        RP.write_outputs(res, 10.0)
        refused = False
    except FileExistsError:
        refused = True
    check("a second write is refused and leaves both outputs byte-identical",
          refused and (open(pj, "rb").read(), open(mdp, "rb").read()) == snap and len(RP.existing_outputs()) == 2)
    check("markdown embeds the code SHA-256, rule SHA-256 and public commit",
          rec["code"]["explore/report_p6_cells.py"] in md and rec["rule_sha256"] in md and rec["public_commit"] in md)
    check("markdown has no em-dash or en-dash", "\u2014" not in md and "\u2013" not in md)
    sec1 = md.split("## 2.")[0]
    rows56 = [ln for ln in sec1.splitlines() if ln.startswith("| R") and " / " in ln and "[" in ln]
    check("markdown rule 5.6 tables: 20 rows with the 13 columns", len(rows56) == 20 and
          all(ln.count(" | ") == 12 for ln in rows56), f"{len(rows56)} rows")
    check("markdown flags below-criterion cells", md.count("below criterion (flag: reported, not counted)") == 3)
    check("markdown prints not reportable ratios without a value",
          "not reportable (denominator interval includes 0)" in md and "nr" in md)
    check("markdown: every set has the ratio-context table (L20, L_alt20, Fieller) and the skewness column",
          sec1.count("| cell | L20 [WCU] | U20 [BCa] | U20 Fieller | L_alt20 [WCU] | A20 [BCa] | A20 Fieller |") == 4 and
          sec1.count("skewness of the cluster means of E") == 4)
    ctx = [ln for ln in sec1.splitlines() if ln.startswith("| R") and ln.count(" | ") == 7]
    check("markdown ratio-context rows: 20, Fieller printed only where reportable",
          len(ctx) == 20 and all(("nr" in ln) == ("not reportable" in ln) for ln in ctx), f"{len(ctx)} rows")
    seedrows = [ln for ln in md.split("## 3.")[1].splitlines() if ln.startswith("| dinov2reg_L12 R1n | ")
                and ln.split(" | ")[1] in ("0", "1", "2")]
    check("markdown per-seed U20 and A20 are nr where the seed's E interval includes 0 (synthetic dinov2reg_L12 R1n)",
          len(seedrows) == 3 and all(ln.rstrip().endswith("| nr | nr |") for ln in seedrows))
    seedok = [ln for ln in md.split("## 3.")[1].splitlines() if ln.startswith("| dinov2reg_L6 R4 | ")
              and ln.split(" | ")[1] in ("0", "1", "2")]
    check("markdown per-seed U20 and A20 printed signed where reportable",
          len(seedok) == 3 and all(ln.split(" | ")[-2].startswith("+") for ln in seedok))
    check("markdown prints flags as yes or no, never Python booleans", "True" not in md and "False" not in md)
    check("markdown ratio values are signed like their intervals",
          RP.rtxt(dict(value=0.5, ci_bca=[0.4, 0.6], reportable=True)).startswith("+0.5000 [+0.4000, +0.6000]"))
    check("markdown item 5 section with 20 rows", "## 4. Rule Section 6 item 5" in md and
          len([ln for ln in md.split("## 4.")[1].split("## 5.")[0].splitlines() if ln.startswith("| dinov2")]) == 20)
    check("markdown lists the not computable items", "## 5. Not computable from saved outputs" in md and
          "Rule 6 item 4 (A4.4 log mass)" in md and "## 6. Not computed by this script" in md)
    check("markdown table cells carry no unit key with a pipe", "|R1" not in md and "|R4" not in md)

    n_fail = RES.count(False)
    out(f"# {len(RES)} checks, {n_fail} failed, {time.time() - t0:.1f}s")
    open(a.out, "w").write("\n".join(LINES) + "\n")
    sys.exit(1 if n_fail else 0)


def _recs_from(res):
    """Per-unit records in the form run() builds internally, rebuilt from the result for the gate unit tests."""
    recs = {}
    for fs, ps in res["rule_5_6"].items():
        for ar, c in ps["cells"].items():
            key = f"{fs}|{ar}"
            recs[key] = dict(accuracy=c["accuracy"],
                             rule_5_6=dict(E=c["E_wcu"], seed_point_estimates=c["seed_point_estimates"],
                                           E_336=c["E_336"], E_448=c["E_448"], E_vs_control90=c["E_vs_control90"],
                                           U20=c["U20"], A20=c["A20"], E_continuity_percentile=c["E_continuity_percentile"],
                                           published_percentile_for_gate=c["published_percentile_for_gate"]),
                             item1=res["item1_A4_1_per_size"][key], item5=res["item5_A4_5_core_share"][key],
                             item3=dict(profile=res["item3_A4_3_profiles"]["profiles"][key],
                                        per_seed=res["item3_A4_3_profiles"]["per_seed"][key],
                                        two_level_seed_bootstrap=res["item3_A4_3_profiles"]["two_level_seed_bootstrap"][key]))
    return recs


if __name__ == "__main__":
    main()
