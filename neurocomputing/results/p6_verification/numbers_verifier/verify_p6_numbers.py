"""Independent recomputation of P6 (and P3 point estimates) from raw per-image score files.

Read-only on the project. Builds its own design, own score averaging, own CC means and thresholds, own Holm,
and an independent wild cluster bootstrap implementation; additionally calls analyze_re.wild_cluster (imported
read-only by file location) on its own per-configuration vectors to confirm the stored intervals and p-values.
Writes only to the scratch directory given as argv[1].
"""
import importlib.util, itertools, json, math, os, sys
import numpy as np

OUTDIR = sys.argv[1]
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
JSON_P6 = os.path.join(RE, "out/primary_P6.json")
REG_R4 = os.path.join(ROOT, "nn/registers/full/scores")
SWAP = os.path.join(ROOT, "v3/gpu/full/scores")
MEAS = "r_inside"
SIZES = ("336", "448")
SEEDS = (0, 1, 2)
ARCHS = ("R1n", "R1", "R2", "R3", "R4")
FSETS = {"dinov2reg_L6": os.path.join(RE, "gpu/g3/scores"), "dinov2reg_L12": os.path.join(RE, "gpu/g3/scores"),
         "dinov2ms_L7L11": os.path.join(RE, "gpu/a14/scores"), "dinov2ms_L11": os.path.join(RE, "gpu/a14/scores")}


def src_dir(fs, ar):
    return REG_R4 if (fs.startswith("dinov2reg") and ar == "R4") else FSETS[fs]


# ---------------------------------------------------------------- design (own construction)
items = json.load(open(os.path.join(ROOT, "v2/stim_v3/items.json")))
ill = [it for it in items if it["illusion"] == "kanizsa" and it["cond"] == "illusory"]
cl = {}
for it in ill:
    cl.setdefault((it["shape"], it["sr"], it["pol"]), []).append((it["pos"], it["config"]))
keys = sorted(cl, key=str)
CONFIGS = [c for k in keys for _, c in sorted(cl[k])]
GROUPS = np.repeat(np.arange(len(keys)), 5)
assert len(CONFIGS) == 60 and len(keys) == 12


def load(p):
    with open(p) as f:
        return json.load(f)


def per_seed_size_E(fs, ar):
    """dict (seed, size) -> np.array(60) of r_inside(illusory) - r_inside(control180)."""
    out = {}
    for s in SEEDS:
        d = load(os.path.join(src_dir(fs, ar), f"stim_v3_{fs}_{ar}_s{s}.json"))
        for z in SIZES:
            m = d[z][MEAS]
            out[(s, z)] = np.array([m[f"{c}_illusory"] - m[f"{c}_control180"] for c in CONFIGS], float)
    return out


def per_config_vector(fs, ar):
    """Seed-average per configuration, then size-average (equivalent to the mean of the 6 seed x size values of
    each image, then the difference; computed here as mean of image values first, exactly as head_scores)."""
    vals = {}
    for s in SEEDS:
        d = load(os.path.join(src_dir(fs, ar), f"stim_v3_{fs}_{ar}_s{s}.json"))
        for z in SIZES:
            m = d[z][MEAS]
            for c in CONFIGS:
                for cond in ("illusory", "control180"):
                    vals.setdefault(f"{c}_{cond}", []).append(m[f"{c}_{cond}"])
    img = {k: float(np.mean(v)) for k, v in vals.items()}
    return np.array([img[f"{c}_illusory"] - img[f"{c}_control180"] for c in CONFIGS], float)


def cc_seeds(fs, ar):
    out = []
    for s in SEEDS:
        m = load(os.path.join(src_dir(fs, ar), f"mit_{fs}_{ar}_s{s}.json"))
        assert m["fset"] == fs and m["arch"] == ar and m["seed"] == s, (fs, ar, s)
        out.append(dict(CC=float(m["CC"]), n=m.get("n"), cc_img_mean=float(np.mean(m["cc_per_image"])),
                        n_img=len(m["cc_per_image"])))
    return out


# ---------------------------------------------------------------- independent wild cluster bootstrap (own code)
W6 = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def my_wcb(d, g, B=9999, seed=2026, alpha=0.05):
    d = np.asarray(d, float); G = int(g.max()) + 1; N = d.size
    nG = np.array([np.sum(g == j) for j in range(G)], float)

    def crse(x):
        e = x - x.mean()
        S = np.array([e[g == j].sum() for j in range(G)])
        return math.sqrt(G / (G - 1) * np.sum(S ** 2)) / N
    mu = d.mean(); se = crse(d); t = mu / se
    W = W6[np.random.RandomState(seed).randint(0, 6, (B, G))]
    # restricted (H0: mu = 0): d* = 0 + v_g * d_i
    Ssum0 = np.array([d[g == j].sum() for j in range(G)])
    A0 = W * Ssum0[None, :]
    m0 = A0.sum(1) / N
    S0 = A0 - nG[None, :] * m0[:, None]
    se0 = np.sqrt(G / (G - 1) * (S0 ** 2).sum(1)) / N
    t0 = m0 / se0
    k_up = int(np.sum(t0 >= t - 1e-12)); k_lo = int(np.sum(t0 <= t + 1e-12))
    p = min(1.0, 2 * (1 + min(k_up, k_lo)) / (B + 1))
    # unrestricted: d* = mu + v_g * (d_i - mu)
    Ssum1 = np.array([(d - mu)[g == j].sum() for j in range(G)])
    A1 = W * Ssum1[None, :]
    m1 = A1.sum(1) / N
    S1 = A1 - nG[None, :] * m1[:, None]
    se1 = np.sqrt(G / (G - 1) * (S1 ** 2).sum(1)) / N
    tu = m1 / se1
    qlo, qhi = np.quantile(tu, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(mu), se=float(se), t=float(t), p=float(p), ci=[float(mu - qhi * se), float(mu - qlo * se)])


def my_holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps, kind="mergesort")
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- import analyze_re read-only (no sys.path change)
spec = importlib.util.spec_from_file_location("analyze_re_ro", os.path.join(RE, "analyze_re.py"))
AR = importlib.util.module_from_spec(spec); spec.loader.exec_module(AR)

J = load(JSON_P6)
P6 = J["result"]["P6"]
report = dict(created_utc=J["created_utc"], cells=[], mismatches=[], fsets={})
TOL = 1e-9


def chk(label, a, b, tol=TOL):
    ok = (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= tol)
    if not ok:
        report["mismatches"].append(f"{label}: recomputed {a} vs json {b}")
    return ok


for fs in FSETS:
    js = P6[fs]
    # R4 reference
    cc4 = cc_seeds(fs, "R4"); cc4m = float(np.mean([x["CC"] for x in cc4]))
    thr = round(round(cc4m, 4) - 0.03, 4)
    v4 = per_config_vector(fs, "R4")
    w4_mine = my_wcb(v4, GROUPS); w4_ar = AR.wild_cluster(v4, GROUPS)
    chk(f"{fs} R4 cc mean", cc4m, js["R4_cc"])
    chk(f"{fs} R4 E", v4.mean(), js["R4"]["mean"])
    chk(f"{fs} R4 se", w4_ar["se"], js["R4"]["se"])
    for i in (0, 1):
        chk(f"{fs} R4 ci[{i}] (analyze_re.wild_cluster on own vector)", w4_ar["ci"][i], js["R4"]["ci"][i])
        chk(f"{fs} R4 ci[{i}] (own WCB)", w4_mine["ci"][i], js["R4"]["ci"][i], 1e-9)
    chk(f"{fs} R4 p (ar)", w4_ar["p"], js["R4"]["p"]); chk(f"{fs} R4 p (own)", w4_mine["p"], js["R4"]["p"])
    sgn = int(np.sign(v4.mean())); ref_ok = w4_mine["p"] < 0.05
    if sgn != js["R4_sign"] or ref_ok != js["R4_reference_defined"]:
        report["mismatches"].append(f"{fs} R4 sign/reference: {sgn},{ref_ok} vs {js['R4_sign']},{js['R4_reference_defined']}")
    pse4 = per_seed_size_E(fs, "R4")
    r4cell = dict(fset=fs, arch="R4", cc_seeds=[x["CC"] for x in cc4], cc_mean=cc4m, threshold=thr, valid=None,
                  E_recomputed=float(v4.mean()), E_json=js["R4"]["mean"], ci_own=w4_mine["ci"], ci_ar=w4_ar["ci"],
                  ci_json=js["R4"]["ci"], p_own=w4_mine["p"], p_json=js["R4"]["p"],
                  seed_point_estimates=[float(np.mean([pse4[(s, z)].mean() for z in SIZES])) for s in SEEDS],
                  size_E={z: float(np.mean([pse4[(s, z)].mean() for s in SEEDS])) for z in SIZES},
                  seed_size_E={f"s{s}_{z}": float(pse4[(s, z)].mean()) for s in SEEDS for z in SIZES},
                  cc_img_mean=[x["cc_img_mean"] for x in cc4], n_img=[x["n_img"] for x in cc4])
    report["cells"].append(r4cell)
    valid_arch, pvals = [], []
    cells_here = {}
    for ar in ("R1n", "R1", "R2", "R3"):
        jc = js["cells"][ar]
        ccs = cc_seeds(fs, ar); ccm = float(np.mean([x["CC"] for x in ccs]))
        valid = round(ccm, 4) >= thr
        v = per_config_vector(fs, ar)
        wm = my_wcb(v, GROUPS); wa = AR.wild_cluster(v, GROUPS)
        pse = per_seed_size_E(fs, ar)
        seed_pe = [float(np.mean([pse[(s, z)].mean() for z in SIZES])) for s in SEEDS]
        # check that per-seed size-averaged values average to the pooled estimate (balanced design)
        chk(f"{fs} {ar} seed-avg consistency", float(np.mean(seed_pe)), float(v.mean()), 1e-12)
        chk(f"{fs} {ar} cc mean", ccm, jc["cc"]); chk(f"{fs} {ar} threshold", thr, jc["threshold"], 0)
        if valid != jc["valid"]:
            report["mismatches"].append(f"{fs} {ar} validity {valid} vs {jc['valid']}")
        chk(f"{fs} {ar} E", v.mean(), jc["mean"]); chk(f"{fs} {ar} se", wa["se"], jc["se"])
        for i in (0, 1):
            chk(f"{fs} {ar} ci[{i}] (ar)", wa["ci"][i], jc["ci"][i]); chk(f"{fs} {ar} ci[{i}] (own)", wm["ci"][i], jc["ci"][i])
        chk(f"{fs} {ar} p (ar)", wa["p"], jc["p"]); chk(f"{fs} {ar} p (own)", wm["p"], jc["p"])
        opp = bool((sgn > 0 and wm["ci"][1] < 0) or (sgn < 0 and wm["ci"][0] > 0)) if ref_ok else bool(wm["ci"][0] > 0 or wm["ci"][1] < 0)
        if opp != jc["opposite_sign_interval"]:
            report["mismatches"].append(f"{fs} {ar} opposite {opp} vs {jc['opposite_sign_interval']}")
        cell = dict(fset=fs, arch=ar, cc_seeds=[x["CC"] for x in ccs], cc_mean=ccm, threshold=thr, valid=bool(valid),
                    E_recomputed=float(v.mean()), E_json=jc["mean"], ci_own=wm["ci"], ci_ar=wa["ci"], ci_json=jc["ci"],
                    p_own=wm["p"], p_json=jc["p"], opposite=opp, seed_point_estimates=seed_pe,
                    seed_point_signs=[int(np.sign(x)) for x in seed_pe],
                    size_E={z: float(np.mean([pse[(s, z)].mean() for s in SEEDS])) for z in SIZES},
                    seed_size_E={f"s{s}_{z}": float(pse[(s, z)].mean()) for s in SEEDS for z in SIZES},
                    cc_img_mean=[x["cc_img_mean"] for x in ccs], n_img=[x["n_img"] for x in ccs],
                    E_minus_R4=float(v.mean() - v4.mean()))
        # exploratory: per-seed WCU intervals and paired readout-minus-R4 WCU
        cell["per_seed_wcu"] = {}
        for s in SEEDS:
            vs = 0.5 * (pse[(s, "336")] + pse[(s, "448")])
            w = my_wcb(vs, GROUPS); cell["per_seed_wcu"][f"s{s}"] = dict(mean=w["mean"], ci=w["ci"], p=w["p"])
        for z in SIZES:
            vz = np.mean([pse[(s, z)] for s in SEEDS], axis=0)
            w = my_wcb(vz, GROUPS); cell.setdefault("per_size_wcu", {})[z] = dict(mean=w["mean"], ci=w["ci"], p=w["p"])
        wd = my_wcb(v - v4, GROUPS); cell["minus_R4_wcu"] = dict(mean=wd["mean"], ci=wd["ci"], p=wd["p"])
        # exploratory: by shape and by cluster
        sh = np.array([k[0] for k in keys])[GROUPS]
        cell["by_shape"] = {s_: float(v[sh == s_].mean()) for s_ in ("square", "triangle")}
        cell["cluster_means"] = {str(keys[j]): float(v[GROUPS == j].mean()) for j in range(12)}
        cell["n_configs_positive"] = int(np.sum(v > 0))
        cells_here[ar] = cell
        if valid:
            valid_arch.append(ar); pvals.append(wm["p"])
    if valid_arch:
        adj = my_holm(pvals)
        for ar, pa in zip(valid_arch, adj):
            c = cells_here[ar]; jc = js["cells"][ar]
            c["p_holm"] = float(pa)
            chk(f"{fs} {ar} p_holm", pa, jc["p_holm"])
            keep = bool(ref_ok and pa < 0.05 and np.sign(c["E_recomputed"]) == sgn)
            c["keeps_sign"] = keep
            if keep != jc["keeps_sign"]:
                report["mismatches"].append(f"{fs} {ar} keeps_sign {keep} vs {jc['keeps_sign']}")
    nk = sum(bool(cells_here[a].get("keeps_sign")) for a in valid_arch)
    if len(valid_arch) != js["n_valid"] or nk != js["n_keep"]:
        report["mismatches"].append(f"{fs} n_valid/n_keep {len(valid_arch)},{nk} vs {js['n_valid']},{js['n_keep']}")
    report["fsets"][fs] = dict(R4_sign=sgn, R4_ref=ref_ok, n_valid=len(valid_arch), n_keep=nk, threshold=thr,
                               R4_cc_mean=cc4m, R4_size_E=r4cell["size_E"], R4_seed_pe=r4cell["seed_point_estimates"])
    report["cells"].extend(cells_here[a] for a in ("R1n", "R1", "R2", "R3"))

# ---------------------------------------------------------------- P3 recomputation (point estimates + intervals)
P3J = J["result"]["P3"]
P3_CELLS = [("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
            ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"), ("siglip_L24", "R3"),
            ("iie_shapenetc", "R2"), ("iie_shapenetc", "R3")]


def swap_vec(f, a):
    vals = {}
    for s in SEEDS:
        d = load(os.path.join(SWAP, f"stim_v3_{f}_{a}_s{s}.json"))
        for z in SIZES:
            m = d[z][MEAS]
            for c in CONFIGS:
                for cond in ("illusory", "control180"):
                    vals.setdefault(f"{c}_{cond}", []).append(m[f"{c}_{cond}"])
    img = {k: float(np.mean(v)) for k, v in vals.items()}
    return np.array([img[f"{c}_illusory"] - img[f"{c}_control180"] for c in CONFIGS], float)


p3 = {}
refs = {}
for f in sorted(set(x[0] for x in P3_CELLS)):
    r = my_wcb(swap_vec(f, "R4"), GROUPS); refs[f] = r
    chk(f"P3 ref {f} mean", r["mean"], P3J["reference"][f]["mean"]); chk(f"P3 ref {f} p", r["p"], P3J["reference"][f]["p"])
ps = []
for f, a in P3_CELLS:
    v = swap_vec(f, a); r = my_wcb(v, GROUPS); jc = P3J["cells"][f"{f}|{a}"]
    chk(f"P3 {f} {a} mean", r["mean"], jc["mean"]); chk(f"P3 {f} {a} p", r["p"], jc["p"])
    for i in (0, 1):
        chk(f"P3 {f} {a} ci[{i}]", r["ci"][i], jc["ci"][i])
    p3[(f, a)] = r; ps.append(r["p"])
adj = my_holm(ps); nkeep = 0; anyopp = False
for (f, a), pa in zip(P3_CELLS, adj):
    r = p3[(f, a)]; sg = int(np.sign(refs[f]["mean"])); rd = refs[f]["p"] < 0.05
    keep = bool(pa < 0.05 and rd and np.sign(r["mean"]) == sg); nkeep += keep
    opp = bool((sg > 0 and r["ci"][1] < 0) or (sg < 0 and r["ci"][0] > 0)) if rd else bool(r["ci"][0] > 0 or r["ci"][1] < 0)
    anyopp |= opp
    chk(f"P3 {f} {a} p_holm", pa, P3J["cells"][f"{f}|{a}"]["p_holm"])
report["P3"] = dict(n=len(P3_CELLS), n_keep=nkeep, any_opposite=anyopp, json_n_keep=P3J["n_keep"], json_any_opposite=P3J["any_opposite"])
if nkeep != P3J["n_keep"] or anyopp != P3J["any_opposite"]:
    report["mismatches"].append(f"P3 n_keep/any_opposite {nkeep},{anyopp} vs {P3J['n_keep']},{P3J['any_opposite']}")

# ---------------------------------------------------------------- title decision (own logic)
reasons, opposite = [], []
for fs in ("dinov2reg_L6", "dinov2reg_L12", "dinov2ms_L7L11"):
    F = report["fsets"][fs]
    if not F["R4_ref"]:
        reasons.append(f"(b) {fs}")
    if F["n_valid"] < 2:
        reasons.append(f"(c0) {fs}")
    for c in report["cells"]:
        if c["fset"] == fs and c["arch"] != "R4" and c["valid"] and not c.get("keeps_sign"):
            reasons.append(f"(c) {fs} {c['arch']}")
if nkeep < len(P3_CELLS):
    reasons.append("(c1)")
for c in report["cells"]:
    if c["arch"] != "R4" and c["valid"] and c["opposite"]:
        opposite.append(f"(d) {c['fset']} {c['arch']}")
if anyopp:
    opposite.append("(d) P3")
title = None if opposite else ("T1" if not reasons else "TS")
report["title_recomputed"] = dict(title=title, narrowed=bool(opposite), reasons=reasons + opposite)
TJ = J["result"]["title"]
if (TJ["title"] is not None) != (title is not None) or TJ["narrowed"] != bool(opposite):
    report["mismatches"].append(f"title {title},{bool(opposite)} vs {TJ['title']},{TJ['narrowed']}")
jr = [r.split(":")[0] for r in TJ["reasons"]]
report["title_json_reasons"] = jr

os.makedirs(OUTDIR, exist_ok=True)
json.dump(report, open(os.path.join(OUTDIR, "verify_p6_out.json"), "w"), indent=1)
print("mismatches:", len(report["mismatches"]))
for m in report["mismatches"]:
    print("  ", m)
print("P3:", report["P3"])
print("title:", report["title_recomputed"])
print("json reasons:", jr)
for c in report["cells"]:
    print(f"{c['fset']:15s} {c['arch']:4s} cc {['%.5f' % x for x in c['cc_seeds']]} mean {c['cc_mean']:.6f} thr {c['threshold']} "
          f"valid {c['valid']} E {c['E_recomputed']:+.6f} (json {c['E_json']:+.6f}) diff {c['E_recomputed'] - c['E_json']:.1e} "
          f"ci {c['ci_own'][0]:+.4f},{c['ci_own'][1]:+.4f} p {c['p_own']:.4f} ph {c.get('p_holm')} seeds {['%+.4f' % x for x in c['seed_point_estimates']]} "
          f"size {c['size_E']['336']:+.4f}/{c['size_E']['448']:+.4f}")
