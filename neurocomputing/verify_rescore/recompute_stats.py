"""
Independent recomputation of P1 to P4 of neuro/reanalysis/REANALYSIS_RULE.md (Sections 4 and 5.1 to 5.4).

Written 18 November 2026 for the verification step. It does NOT import analyze_re.py or any project module: loaders,
design, wild cluster bootstrap (Webb six-point weights; CR1 se; WCR p with the null imposed; WCU percentile-t
interval; random draws for 12 clusters, full enumeration for 6), pairs cluster BCa, Holm and the decision logic are
written here from the rule text. Inputs are the raw per-image score files (read only):
  v2/stim_v3/items.json (design), v3/scores/raw_v3maps/DeepGaze_{IIE,MSDB}.json (stim_v3), v3/scores/raw/... (stim_alt),
  v3/scores/raw_nulls.json (15 nulls, both sets), v3/gpu/full/scores/stim_{v3,alt}_<fset>_<arch>_s<seed>.json (swap).
Output: neuro/verify_rescore/recompute_stats.json (nothing else is written).
Run: nice -n 10 env OMP_NUM_THREADS=4 $PY -B -I recompute_stats.py
"""
import hashlib, itertools, json, math, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                          # re-analysis folder (neurocomputing/ in the repository)
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root (as analyze_re.py)
OUT = os.path.join(HERE, "recompute_stats.json")

SEED, B_WILD, B_PAIRS = 2026, 9999, 10000
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
FSETS = ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")
ARCHS = ("R1n", "R1", "R2", "R3", "R4")
P3_CELLS = [("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
            ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"), ("siglip_L24", "R3"),
            ("iie_shapenetc", "R2"), ("iie_shapenetc", "R3")]
P4_CELLS = [("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
            ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"), ("iie_shapenetc", "R2"),
            ("iie_shapenetc", "R3")]


# ------------------------------------------------------------------ own guard (rule Section 8.1), read only
def guard():
    rec, pub = os.path.join(RE, "REANALYSIS_RULE.sha256"), os.path.join(RE, "PUBLIC_PUSH.txt")
    assert os.path.exists(rec) and os.path.exists(pub), "rule not fixed/public"
    n = 0
    for line in open(rec):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            h = hashlib.sha256(open(os.path.join(RE, p[1]), "rb").read()).hexdigest()
            assert h == p[0], f"hashed file changed: {p[1]}"
            n += 1
    return n


# ------------------------------------------------------------------ design (rule Section 2.1)
def make_design(shape=None):
    items = json.load(open(os.path.join(ROOT, "v2/stim_v3/items.json")))
    cells = {}
    for it in items:
        if it["illusion"] == "kanizsa" and it["cond"] == "illusory" and (shape is None or it["shape"] == shape):
            cells.setdefault((it["shape"], it["sr"], it["pol"]), []).append((it["pos"], it["config"]))
    keys = sorted(cells, key=str)
    grid = [[c for _, c in sorted(cells[k])] for k in keys]
    assert len({len(g) for g in grid}) == 1
    return keys, grid


def arr(grid, f):
    return np.array([[f(c) for c in row] for row in grid], float)


# ------------------------------------------------------------------ loaders (read only)
def jl(p):
    return json.load(open(p))


def model_scores(tag):
    d = dict(jl(os.path.join(ROOT, "v3/scores/raw_v3maps", tag + ".json")))
    alt = jl(os.path.join(ROOT, "v3/scores/raw", tag + ".json"))
    assert not set(d) & set(alt)
    d.update(alt)
    return {k: v["r_inside"] for k, v in d.items()}


def null_scores():
    N = jl(os.path.join(ROOT, "v3/scores/raw_nulls.json"))
    return {k: {n: v["r_inside"] for n, v in N[k].items()} for k in NULLS}


def head_scores(fset, arch):
    acc = {}
    for st in ("stim_v3", "stim_alt"):
        for s in (0, 1, 2):
            d = jl(os.path.join(ROOT, "v3/gpu/full/scores", f"{st}_{fset}_{arch}_s{s}.json"))
            for z in ("336", "448"):
                for k, v in d[z]["r_inside"].items():
                    acc.setdefault(k, []).append(v)
    assert all(len(v) == 6 for v in acc.values())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def val(sc, c, cond):
    if cond in ("alt10", "alt20", "alt30"):
        return 0.5 * (sc[f"{c}_{cond}a"] + sc[f"{c}_{cond}b"])
    return sc[f"{c}_{cond}"]


# ------------------------------------------------------------------ wild cluster bootstrap-t (rule Section 4.1)
def cr1_se(x):
    """x (..., G, P): CR1 cluster-robust se of the grand mean."""
    G, P = x.shape[-2], x.shape[-1]
    N = G * P
    res = x - x.mean(axis=(-2, -1), keepdims=True)
    S = res.sum(-1)
    return np.sqrt(G / (G - 1) * (S ** 2).sum(-1)) / N


_W = {}
def weights(G):
    if G not in _W:
        if G <= 6:
            idx = np.array(list(itertools.product(range(6), repeat=G)))
            _W[G] = (WEBB[idx], True)
        else:
            rs = np.random.RandomState(SEED)
            _W[G] = (WEBB[rs.randint(0, 6, size=(B_WILD, G))], False)
    return _W[G]


def wild(d, mu0=0.0):
    G, P = d.shape
    W, enum = weights(G)
    m = d.mean(); se = cr1_se(d); t = (m - mu0) / se
    # WCR: null imposed
    e0 = d - mu0
    ds = mu0 + W[:, :, None] * e0[None]
    ts = (ds.mean((1, 2)) - mu0) / cr1_se(ds)
    def pval(tol):
        ge, le = np.sum(ts >= t - tol), np.sum(ts <= t + tol)
        if enum:
            return min(1.0, 2 * min(ge, le) / len(ts))
        return min(1.0, 2 * (1 + min(ge, le)) / (len(ts) + 1))
    p = pval(0.0)                     # strict floating-point comparison
    # tie-aware: weight vectors that reproduce t exactly in exact arithmetic (constant vectors scale d, t is scale
    # free) can land 1 ulp either side of t in floating point; count them as ties (tolerance 1e-10 * max(1, |t|))
    p_ties = pval(1e-10 * max(1.0, abs(t)))
    # WCU: unrestricted residuals
    e1 = d - m
    du = m + W[:, :, None] * e1[None]
    tu = (du.mean((1, 2)) - m) / cr1_se(du)
    q_lo, q_hi = np.quantile(tu, [0.025, 0.975])
    ci = [m - q_hi * se, m - q_lo * se]
    skew = float(stats.skew(d.mean(1)))
    return dict(mean=float(m), se=float(se), t=float(t), p=float(p_ties), p_strict=float(p), ci=[float(ci[0]), float(ci[1])],
                G=G, N=G * P, enumerated=enum, n_vectors=len(ts), skew_cluster_means=skew)


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj.tolist()


# ------------------------------------------------------------------ pairs cluster BCa (rule Section 4.2)
_IDX = {}
def two_stage(G, P):
    if (G, P) not in _IDX:
        rs = np.random.RandomState(SEED)
        _IDX[(G, P)] = (rs.randint(0, G, (B_PAIRS, G)), rs.randint(0, P, (B_PAIRS, G, P)))
    return _IDX[(G, P)]


def boot_means(x, idx):
    ci, pi = idx
    return x[ci[:, :, None], pi].mean((1, 2))


def bca(stat_fn, arrays, alpha=0.05, theta0=0.0):
    G, P = arrays[0].shape
    idx = two_stage(G, P)
    est = stat_fn([a.mean() for a in arrays])
    bm = [boot_means(a, idx) for a in arrays]
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = stat_fn(bm)
    boot = boot[np.isfinite(boot)]
    z0 = stats.norm.ppf(np.mean(boot < est))
    jk = np.array([stat_fn([np.delete(a, k, axis=0).mean() for a in arrays]) for k in range(G)])
    dj = jk.mean() - jk
    acc = float((dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5))
    def level(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo_l, hi_l = level(stats.norm.ppf(alpha / 2)), level(stats.norm.ppf(1 - alpha / 2))
    ci = np.quantile(boot, [lo_l, hi_l])
    # p by inversion: the level at which theta0 is an endpoint
    q0 = np.mean(boot < theta0)
    n_below = int(np.sum(boot < theta0))
    def inv(q):
        zq = stats.norm.ppf(q) - z0
        u = zq / (1 + acc * zq)              # u = z0 + z_alpha
        za = u - z0
        return float(min(1.0, 2 * min(stats.norm.cdf(za), stats.norm.sf(za))))
    boundary = bool(q0 <= 0 or q0 >= 1)
    p_raw = 0.0 if boundary else inv(q0)     # tested value outside every resample: p below the resolution
    # alternative convention (not stated in the rule): q0 clipped to [0.5/B, 1 - 0.5/B] before inversion
    nb = len(boot)
    p_clip = inv(min(max(q0, 0.5 / nb), 1 - 0.5 / nb))
    p = p_raw
    return dict(est=float(est), ci=[float(ci[0]), float(ci[1])], p=p, p_clip=p_clip, z0=float(z0), acc=acc, n_boot=int(len(boot)),
                n_boot_below_theta0=n_below, p_at_boundary=boundary)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    n_hashed = guard()
    keys, grid = make_design(); ksq, gsq = make_design("square")
    assert len(grid) == 12 and len(grid[0]) == 5 and len(gsq) == 6
    models = {"DeepGaze IIE": model_scores("DeepGaze_IIE"), "DeepGaze MSDB": model_scores("DeepGaze_MSDB")}
    nulls = null_scores()
    out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), hashed_files_verified=n_hashed,
               design=dict(clusters=[str(k) for k in keys], square_clusters=[str(k) for k in ksq]))

    # P1 (rule 5.1): L20 = r(illusory) - r(mis20), model minus null, WCR 12 clusters, direction positive, IUT
    L = lambda sc, g: arr(g, lambda c: val(sc, c, "illusory") - val(sc, c, "mis20"))
    P1 = {}
    for mn, sc in models.items():
        Lm = L(sc, grid)
        comps = {k: wild(Lm - L(nulls[k], grid)) for k in NULLS}
        null_L = {k: float(L(nulls[k], grid).mean()) for k in NULLS}
        largest = max(null_L, key=null_L.get)
        met = all(r["p"] < 0.05 and r["mean"] > 0 for r in comps.values())
        P1[mn] = dict(L_model=wild(Lm), vs_null=comps, null_mean_L20=null_L, null_with_largest_loss=largest,
                      n_separated=sum(r["p"] < 0.05 and r["mean"] > 0 for r in comps.values()),
                      max_p=max(r["p"] for r in comps.values()), met=met,
                      met_strict=all(r["p_strict"] < 0.05 and r["mean"] > 0 for r in comps.values()))
    out["P1"] = P1

    # P2 (rule 5.2): square Delta20 = r(alt20) - r(mis20), model minus null, 6 clusters, enumeration, IUT
    Dl = lambda sc, g: arr(g, lambda c: val(sc, c, "alt20") - val(sc, c, "mis20"))
    P2 = {}
    for mn, sc in models.items():
        Dm = Dl(sc, gsq)
        comps = {k: wild(Dm - Dl(nulls[k], gsq)) for k in NULLS}
        met = all(r["p"] < 0.05 and r["mean"] > 0 for r in comps.values())
        weakest = max(comps, key=lambda k: comps[k]["p"])
        P2[mn] = dict(Delta_model=wild(Dm), vs_null=comps, weakest=weakest,
                      n_separated=sum(r["p"] < 0.05 and r["mean"] > 0 for r in comps.values()),
                      max_p=max(r["p"] for r in comps.values()), met=met,
                      met_strict=all(r["p_strict"] < 0.05 and r["mean"] > 0 for r in comps.values()))
    out["P2"] = P2

    # P3 (rule 5.3): E = r(illusory) - r(control180) per cell, WCR two-sided 12 clusters, Holm over 10
    H = {(f, a): head_scores(f, a) for f in FSETS for a in ARCHS}
    E = lambda sc: arr(grid, lambda c: val(sc, c, "illusory") - val(sc, c, "control180"))
    ref = {}
    for f in FSETS:
        r = wild(E(H[(f, "R4")]))
        r["defined"] = r["p"] < 0.05
        r["sign"] = int(np.sign(r["mean"]))
        ref[f] = r
    cells = {}
    for f, a in P3_CELLS:
        cells[f"{f}|{a}"] = wild(E(H[(f, a)]))
    adj = holm([cells[f"{f}|{a}"]["p"] for f, a in P3_CELLS])
    nkeep = 0
    for (f, a), pa in zip(P3_CELLS, adj):
        c = cells[f"{f}|{a}"]
        c["p_holm"] = pa
        c["keeps_sign"] = bool(ref[f]["defined"] and pa < 0.05 and np.sign(c["mean"]) == ref[f]["sign"])
        lo, hi = c["ci"]
        c["opposite_interval"] = bool((ref[f]["sign"] > 0 and hi < 0) or (ref[f]["sign"] < 0 and lo > 0)) \
            if ref[f]["defined"] else bool(lo > 0 or hi < 0)
        nkeep += c["keeps_sign"]
    adj_s = holm([cells[f"{f}|{a}"]["p_strict"] for f, a in P3_CELLS])
    for (f, a), ps in zip(P3_CELLS, adj_s):
        cells[f"{f}|{a}"]["p_holm_strict"] = ps
    out["P3"] = dict(reference=ref, cells=cells, n_keep=nkeep,
                     n_keep_strict=sum(bool(ref[f]["defined"] and cells[f"{f}|{a}"]["p_holm_strict"] < 0.05 and
                                            np.sign(cells[f"{f}|{a}"]["mean"]) == ref[f]["sign"]) for f, a in P3_CELLS),
                     n_opposite=sum(c["opposite_interval"] for c in cells.values()), met_all=nkeep == 10)

    # P4 (rule 5.4): U20 and A20 differences readout minus R4, paired BCa, Holm over 18
    def parts(sc):
        I = arr(grid, lambda c: val(sc, c, "illusory")); C = arr(grid, lambda c: val(sc, c, "control180"))
        M = arr(grid, lambda c: val(sc, c, "mis20")); A = arr(grid, lambda c: val(sc, c, "alt20"))
        return dict(den=I - C, U20=M - C, A20=A - C)
    con = {}
    for f, a in P4_CELLS:
        X, R = parts(H[(f, a)]), parts(H[(f, "R4")])
        for meas in ("U20", "A20"):
            fn = lambda m: m[0] / m[1] - m[2] / m[3]
            r = bca(fn, [X[meas], X["den"], R[meas], R["den"]])
            rx = bca(lambda m: m[0] / m[1], [X[meas], X["den"]], theta0=0.0)
            rr = bca(lambda m: m[0] / m[1], [R[meas], R["den"]], theta0=0.0)
            wx, wr = wild(X["den"]), wild(R["den"])
            r.update(ratio_readout=rx["est"], ratio_readout_bca=rx["ci"], ratio_R4=rr["est"], ratio_R4_bca=rr["ci"],
                     t_den_readout=wx["t"], t_den_R4=wr["t"], t_den_min=float(min(abs(wx["t"]), abs(wr["t"]))),
                     den_ci_readout=wx["ci"], den_ci_R4=wr["ci"],
                     denominators_exclude_zero=bool((wx["ci"][0] > 0 or wx["ci"][1] < 0) and (wr["ci"][0] > 0 or wr["ci"][1] < 0)))
            con[f"{f}|{a}|{meas}"] = r
    ks = list(con)
    adj = holm([con[k]["p"] for k in ks])
    adj_clip = holm([con[k]["p_clip"] for k in ks])
    for k, pa, pc in zip(ks, adj, adj_clip):
        c = con[k]
        c["p_holm"] = pa
        c["p_holm_clip"] = pc
        c["counted_clip"] = bool(pc < 0.05 and c["t_den_min"] >= 4 and c["denominators_exclude_zero"])
        c["screen_ok"] = bool(c["t_den_min"] >= 4 and c["denominators_exclude_zero"])
        c["counted"] = bool(pa < 0.05 and c["screen_ok"])
    out["P4"] = dict(contrasts=con, n_counted=sum(c["counted"] for c in con.values()),
                     n_counted_clip=sum(c["counted_clip"] for c in con.values()),
                     n_screen_fail=sum(not c["screen_ok"] for c in con.values()))
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
