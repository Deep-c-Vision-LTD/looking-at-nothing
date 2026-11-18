"""
independent_check.py (independent statistics reviewer, 18 Nov 2026).

Synthetic data only. No real score is read, no model is run, no torch is imported.
Run: nice -n 10 env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 <venv python> -I independent_check.py

Parts
  1. Webb six-point weights re-derived (moments) and the 6^G enumeration.
  2. An independent wild cluster bootstrap (explicit bootstrap samples, CR1 from the OLS sandwich written out with
     a cluster-indicator matrix) compared with analyze_re.wild_cluster on identical weight sets (must agree to 1e-9).
  3. Coverage / size on DGPs that differ from test_analyze_re.py: WCU percentile-t (the rule's interval), the interval
     obtained by inverting the WCR test (the interval that matches the rule's p-value), CR1 with t(G-1), and the
     two-stage percentile interval (published method). G = 12 (999 random Webb draws) and G = 6 (full enumeration).
  4. With G = 6 and enumeration: can a strong effect get p > 2 x 3^6 / 6^6 = 0.031 (rule Section 1.1 item 13 says
     "at most")? Constructed case plus a random search.
  5. Independent BCa (same two-stage indices) and BCa p inversion by bisection on alpha, against analyze_re.bca.
  6. Holm against an independent implementation.
  7. Title-rule edge cases with analyze_re.p6 / title_decision on synthetic cells.
"""
import itertools, math, os, sys, time
import numpy as np

np.seterr(all="ignore")
RE = "/Users/hamitsoyel/Downloads/illusion_research/neuro/reanalysis"
sys.path.insert(0, RE)
import analyze_re as A  # noqa: E402  (numpy / scipy only; require_public() runs only from its main())
from scipy import stats as sps  # noqa: E402

T0 = time.time()
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


SQ = math.sqrt
WEBB = np.array([-SQ(1.5), -1.0, -SQ(0.5), SQ(0.5), 1.0, SQ(1.5)])


# ------------------------------------------------------------------------------------------------ 1. weights
def part1():
    say("== 1. Webb six-point weights")
    m1, m2, m3, m4 = [(WEBB ** k).mean() for k in (1, 2, 3, 4)]
    say(f"E[v] = {m1:.2e}, E[v^2] = {m2:.12f}, E[v^3] = {m3:.2e}, E[v^4] = {m4:.6f} (7/6 = {7 / 6:.6f})")
    say("analyze_re.WEBB equals the re-derived set:", bool(np.allclose(np.sort(A.WEBB), WEBB, atol=0, rtol=0)))
    W, enum = A.webb_weights(6)
    Wm = np.array(list(itertools.product(WEBB, repeat=6)))
    same = enum and W.shape == Wm.shape and np.array_equal(np.unique(np.round(W, 12), axis=0),
                                                           np.unique(np.round(Wm, 12), axis=0))
    say(f"6^6 enumeration identical as a set: {same} ({W.shape[0]} vectors)")
    # sign-constant share
    sc = np.mean(np.all(Wm > 0, 1) | np.all(Wm < 0, 1))
    say(f"share of sign-constant vectors = {sc:.5f} (2 x 3^6 / 6^6 = {2 * 3 ** 6 / 6 ** 6:.5f}); "
        f"constant vectors (all v_g equal) = {np.mean(np.all(Wm == Wm[:, :1], 1)) * 46656:.0f}")


# ------------------------------------------------------------------------------------------------ 2. independent WCB
def indicator(g):
    G = g.max() + 1
    M = np.zeros((g.size, G)); M[np.arange(g.size), g] = 1.0
    return M


def cr1_sandwich(Y, M):
    """Y: (B, N) samples (rows); intercept-only OLS per row; CR1 = G/(G-1) * (N-1)/(N-K) * bread meat bread, K = 1."""
    B, N = Y.shape; G = M.shape[1]
    beta = Y.mean(1)                                  # (X'X)^-1 X'y with X = 1
    E = Y - beta[:, None]
    S = E @ M                                         # cluster score sums X_g' e_g
    meat = (S ** 2).sum(1)
    V = (G / (G - 1)) * ((N - 1) / (N - 1)) * meat / N ** 2   # bread = 1/N
    return beta, np.sqrt(V)


def my_wild(y, g, W, mu0=0.0, alpha=0.05):
    """Independent wild cluster bootstrap: explicit y* = centre + v_g * resid; t* from the sandwich."""
    M = indicator(g); N = y.size
    mu, se = cr1_sandwich(y[None, :], M); mu, se = mu[0], se[0]
    t = (mu - mu0) / se
    V = W[:, g]                                       # (B, N) weight of each observation's cluster
    # WCR
    Ys = mu0 + V * (y - mu0)[None, :]
    b, s = cr1_sandwich(Ys, M); tr = (b - mu0) / s
    p = 2 * min(np.mean(tr >= t - 1e-12), np.mean(tr <= t + 1e-12))
    # WCU
    Yu = mu + V * (y - mu)[None, :]
    b, s = cr1_sandwich(Yu, M); tu = (b - mu) / s
    tu = tu[np.isfinite(tu)]
    lo, hi = np.quantile(tu, [alpha / 2, 1 - alpha / 2])
    return dict(mean=mu, se=se, t=t, p=min(1.0, p), ci=(mu - hi * se, mu - lo * se), tr=tr)


def wcr_p(y, g, W, mu0, M=None, V=None):
    """WCR p only (used for test inversion)."""
    M = indicator(g) if M is None else M
    V = W[:, g] if V is None else V
    mu, se = cr1_sandwich(y[None, :], M); t = (mu[0] - mu0) / se[0]
    b, s = cr1_sandwich(mu0 + V * (y - mu0)[None, :], M); tr = (b - mu0) / s
    return min(1.0, 2 * min(np.mean(tr >= t - 1e-12), np.mean(tr <= t + 1e-12)))


def wcr_inverted_ci(y, g, W, alpha=0.05, iters=16):
    """Set of mu0 not rejected by the WCR test at alpha (bisection on each side of the estimate)."""
    M, V = indicator(g), W[:, g]
    mu, se = cr1_sandwich(y[None, :], M); mu, se = mu[0], se[0]
    _wcr = wcr_p
    wcr_p_ = lambda yy, gg, WW, m0: _wcr(yy, gg, WW, m0, M, V)
    out = []
    for sgn in (-1, 1):
        a, b = mu, mu + sgn * 12 * se                  # p(a) = 1 > alpha, p(b) < alpha
        if wcr_p_(y, g, W, b) >= alpha:
            out.append(b); continue
        for _ in range(iters):
            c = 0.5 * (a + b)
            if wcr_p_(y, g, W, c) >= alpha: a = c
            else: b = c
        out.append(0.5 * (a + b))
    return (min(out), max(out))


def part2():
    say("\n== 2. independent wild cluster bootstrap vs analyze_re.wild_cluster (identical weights)")
    rng = np.random.default_rng(7)
    worst = 0.0
    for G in (6, 12):
        W = A.webb_weights(G)[0]                       # same set analyze_re uses (enumeration or RandomState(2026))
        for k in range(4):
            P = 5; g = np.repeat(np.arange(G), P)
            y = 0.05 * k + rng.normal(0, 1, G)[g] * rng.uniform(0.3, 1.5) + rng.standard_t(4, G * P) * 0.4
            a = A.wild_cluster(y, g); m = my_wild(y, g, W)
            d = max(abs(a["mean"] - m["mean"]), abs(a["se"] - m["se"]), abs(a["t"] - m["t"]), abs(a["ci"][0] - m["ci"][0]),
                    abs(a["ci"][1] - m["ci"][1]))
            # analyze_re's random-draw p uses (1 + count) / (B + 1); compare like with like
            if a["enumerated"]:
                d = max(d, abs(a["p"] - m["p"]))
            else:
                k_up = int(np.sum(m["tr"] >= m["t"] - 1e-12)); k_lo = int(np.sum(m["tr"] <= m["t"] + 1e-12))
                d = max(d, abs(a["p"] - min(1.0, 2 * (1 + min(k_up, k_lo)) / (W.shape[0] + 1))))
            worst = max(worst, d)
    say(f"largest |difference| over 8 datasets (mean, se, t, p, WCU endpoints): {worst:.2e} -> "
        f"{'AGREE' if worst < 1e-9 else 'DISAGREE'}")


# ------------------------------------------------------------------------------------------------ 3. coverage
def dgp(rng, G, P, kind, mu):
    g = np.repeat(np.arange(G), P)
    if kind == "sym":           # symmetric, heteroskedastic clusters, Laplace noise
        sc = np.linspace(0.3, 1.2, G); rng.shuffle(sc)
        u = rng.normal(0, 1, G) * sc
        e = rng.laplace(0, 0.25, G * P)
    elif kind == "skew":        # centred lognormal cluster effects (right-skewed), Gaussian noise
        u = (rng.lognormal(0, 0.9, G) - math.exp(0.405)) * 0.7
        e = rng.normal(0, 0.3, G * P)
    elif kind == "lever":       # one cluster with 3x the spread of the others
        sc = np.ones(G); sc[rng.integers(G)] = 3.0
        u = rng.normal(0, 0.6, G) * sc
        e = rng.normal(0, 0.3, G * P)
    return mu + u[g] + e, g


def coverage(G, kind, nsim, seed, B=999, invert=True):
    rng = np.random.default_rng(seed)
    P, mu = 5, 0.2
    if G <= 6:
        W = np.array(list(itertools.product(WEBB, repeat=G)))
    else:
        W = WEBB[rng.integers(0, 6, (B, G))]           # my own draws (not RandomState(2026))
    idx = A.two_stage_idx(G, P, B=1000, seed=seed)
    tq = sps.t.ppf(0.975, G - 1)
    hit = dict(wcu=0, wcr_inv=0, t=0, pct=0); rej = 0; width = dict(wcu=[], wcr_inv=[], t=[])
    for _ in range(nsim):
        y, g = dgp(rng, G, P, kind, mu)
        r = my_wild(y, g, W, mu0=mu)
        hit["wcu"] += r["ci"][0] <= mu <= r["ci"][1]; width["wcu"].append(r["ci"][1] - r["ci"][0])
        rej += r["p"] < 0.05
        lo, hi = r["mean"] - tq * r["se"], r["mean"] + tq * r["se"]
        hit["t"] += lo <= mu <= hi; width["t"].append(hi - lo)
        if invert:
            c = wcr_inverted_ci(y, g, W)
            hit["wcr_inv"] += c[0] <= mu <= c[1]; width["wcr_inv"].append(c[1] - c[0])
        s = A.pct_summary(y.reshape(G, P), idx)
        hit["pct"] += s["ci"][0] <= mu <= s["ci"][1]
    cov = {k: v / nsim for k, v in hit.items()}
    w = {k: float(np.median(v)) if v else float("nan") for k, v in width.items()}
    return cov, rej / nsim, w


def part3():
    say("\n== 3. coverage of 95% intervals and size of the WCR test at .05 (synthetic; independent DGPs)")
    say("   columns: WCU percentile-t (rule) | WCR-inverted | CR1 t(G-1) | two-stage percentile (published) | WCR size; "
        "median widths WCU / WCR-inverted / t")
    for G, nsim in ((12, 400), (6, 150)):
        mcse = math.sqrt(0.95 * 0.05 / nsim)
        for kind in ("sym", "skew", "lever"):
            cov, size, w = coverage(G, kind, nsim, seed=100 + G + len(kind))
            say(f"   G = {G:2d} {kind:5s} n = {nsim}: {cov['wcu']:.3f} | {cov['wcr_inv']:.3f} | {cov['t']:.3f} | "
                f"{cov['pct']:.3f} | size {size:.3f}   widths {w['wcu']:.3f} / {w['wcr_inv']:.3f} / {w['t']:.3f}   "
                f"(MC SE of a coverage near .95: {mcse:.3f})")


# ------------------------------------------------------------------------------------------------ 4. p bound, G = 6
def part4():
    say("\n== 4. G = 6 enumeration: is p <= 2 x 3^6 / 6^6 = 0.0313 for every strong effect?")
    g = np.repeat(np.arange(6), 5)
    W = np.array(list(itertools.product(WEBB, repeat=6)))
    # constructed: heterogeneous positive cluster means, one cluster near zero
    for means in ([2, 5, 10, 15, 20, 0.05], [4, 6, 8, 10, 12, 0.0], [3, 9, 9, 9, 15, 0.0]):
        jitter = np.tile(np.linspace(-0.2, 0.2, 5), 6)
        y = np.repeat(np.array(means, float), 5) + jitter
        r = A.wild_cluster(y, g); m = my_wild(y, g, W)
        say(f"   cluster means {means}: t = {r['t']:.2f}, analyze_re p = {r['p']:.4f}, independent p = {m['p']:.4f}")
    rng = np.random.default_rng(11)
    best = (0, None)
    nstrong = 0
    for _ in range(3000):
        cm = rng.uniform(0, 1, 6) ** rng.uniform(0.3, 3) * rng.uniform(1, 20)
        if rng.uniform() < 0.5:
            cm[rng.integers(6)] = rng.uniform(-0.5, 0.5)
        y = np.repeat(cm, 5) + rng.normal(0, 0.3, 30)
        r = A.wild_cluster(y, g)
        if r["t"] > 5:
            nstrong += 1
            if r["p"] > best[0]:
                best = (r["p"], (np.round(cm, 2).tolist(), round(r["t"], 2)))
    say(f"   random search: {nstrong} datasets with t > 5; largest enumerated p = {best[0]:.4f} at cluster means "
        f"{best[1][0] if best[1] else None}, t = {best[1][1] if best[1] else None}")


# ------------------------------------------------------------------------------------------------ 5. BCa
def my_bca(stat, arrays, idx, alpha=0.05):
    ci_, pi_ = idx
    pt = stat(*[a.mean() for a in arrays])
    boots = np.array(stat(*[a[ci_[:, :, None], pi_].mean((1, 2)) for a in arrays]))
    boots = boots[np.isfinite(boots)]
    z0 = sps.norm.ppf((np.sum(boots < pt) + 0.5 * np.sum(boots == pt)) / boots.size)
    K = arrays[0].shape[0]
    jk = np.array([stat(*[np.delete(a, k, 0).mean() for a in arrays]) for k in range(K)])
    d = jk.mean() - jk
    acc = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5)

    def endpoint(al):
        z = sps.norm.ppf(al)
        return np.quantile(boots, sps.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))
    return pt, (endpoint(alpha / 2), endpoint(1 - alpha / 2)), z0, acc, endpoint


def part5():
    say("\n== 5. BCa: independent implementation vs analyze_re.bca (same two-stage indices), and p by bisection")
    rng = np.random.default_rng(3)
    G, P = 12, 5
    den = 1 + rng.normal(0, 0.3, (G, 1)) + rng.normal(0, 0.2, (G, P))
    num = 0.6 * den + (rng.lognormal(0, 0.8, (G, 1)) - 1.4) * 0.15 + rng.normal(0, 0.1, (G, P))
    idx = A.two_stage_idx(G, P, B=10000, seed=2026)
    stat = lambda a, b: a / b
    a = A.bca(stat, [num, den], idx, theta0=0.5)
    pt, ci, z0, acc, endpoint = my_bca(stat, [num, den], idx)
    say(f"   ratio {pt:.4f}: analyze_re {np.round(a['ci'], 6)} z0 {a['z0']:.5f} a {a['acc']:.5f}; "
        f"independent {np.round(ci, 6)} z0 {z0:.5f} a {acc:.5f}")
    # p for theta0 = 0.5 by bisection on alpha: the two-sided level at which 0.5 is an endpoint
    th = 0.5
    side_lo = th < pt
    lo_a, hi_a = 1e-6, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo_a + hi_a)
        e = endpoint(mid / 2) if side_lo else endpoint(1 - mid / 2)
        inside = (e <= th) if side_lo else (e >= th)   # th inside the (1 - mid) interval
        if inside: lo_a = mid
        else: hi_a = mid
    say(f"   BCa p for theta0 = 0.5: analyze_re {a['p']:.5f}; bisection on the interval {0.5 * (lo_a + hi_a):.5f}")


# ------------------------------------------------------------------------------------------------ 6. Holm
def my_holm(p):
    p = np.asarray(p); m = p.size; o = np.argsort(p)
    adj = np.empty(m); prev = 0
    for i, j in enumerate(o):
        prev = max(prev, min(1, (m - i) * p[j])); adj[j] = prev
    return adj


def part6():
    say("\n== 6. Holm")
    rng = np.random.default_rng(5)
    worst = 0
    for _ in range(2000):
        p = rng.uniform(0, 0.2, rng.integers(1, 20))
        worst = max(worst, np.max(np.abs(A.holm(p)[0] - my_holm(p))))
    say(f"   largest |difference| of adjusted p over 2,000 random families: {worst:.1e}")


# ------------------------------------------------------------------------------------------------ 7. title rule
def part7():
    say("\n== 7. title rule edge cases (analyze_re.p6 / title_decision on synthetic cells)")
    meta = {}
    for sh in ("square", "triangle"):
        for sr in (0.3, 0.5, 0.7):
            for pol in ("dark", "light"):
                for p in range(5):
                    meta[f"{sh}_sr{int(sr * 100)}_{pol}_p{p}"] = (sh, sr, pol, p)
    D = A.Design(meta)
    rng = np.random.RandomState(1)

    def head(eff):
        sc = {}
        for cell in D.cells:
            cu = rng.normal(0, 0.01)
            for c in cell:
                b = 0.3 + rng.normal(0, 0.01)
                sc[f"{c}_control180"] = b; sc[f"{c}_illusory"] = b + eff + cu + rng.normal(0, 0.005)
        return A.getter(sc)
    sets = {fs: {a: head(0.08) for a in A.ARCHS} for fs in ("dinov2reg_L6", "dinov2reg_L12", "dinov2ms_L7L11", "dinov2ms_L11")}
    cc = {fs: {"R4": 0.74, "R1n": 0.735, "R1": 0.735, "R2": 0.735, "R3": 0.735} for fs in sets}
    p3_ok = dict(any_opposite=False, n=10, n_keep=10)
    # (a) every non-R4 cell on a title-bearing set below the accuracy criterion
    cc_a = {k: dict(v) for k, v in cc.items()}
    cc_a["dinov2reg_L12"].update(R1n=0.70, R1=0.70, R2=0.70, R3=0.70)
    t = A.title_decision(A.p6(D, sets, cc_a), p3_ok)
    say(f"   (a) all four new readouts on dinov2reg_L12 below the criterion (0 valid comparisons): stronger = "
        f"{t['stronger']}; title = {t['title'][:58]}...")
    # (b) P3 keeps the sign in only 8 of 10 (no opposite interval), every new set passes
    t = A.title_decision(A.p6(D, sets, cc), dict(any_opposite=False, n=10, n_keep=8))
    say(f"   (b) P3 only 8 of 10 keep the sign, new sets all pass: stronger = {t['stronger']}")
    # (c) R4 of a new set missing (run incomplete) -> does p6 report 'not run' or crash?
    s_c = {k: dict(v) for k, v in sets.items()}; del s_c["dinov2ms_L7L11"]["R4"]
    try:
        A.p6(D, s_c, cc); say("   (c) missing R4: handled")
    except Exception as e:
        say(f"   (c) missing R4 on dinov2ms_L7L11: p6 raises {type(e).__name__}: {e}")
    # (d) a scored cell without its CC file
    cc_d = {k: dict(v) for k, v in cc.items()}; del cc_d["dinov2reg_L6"]["R2"]
    try:
        A.p6(D, sets, cc_d); say("   (d) missing CC: handled")
    except Exception as e:
        say(f"   (d) a cell scored but its CC missing: p6 raises {type(e).__name__}: {e}")
    # (e) R4 reference undefined (near zero) but a valid readout with a clear interval
    s_e = {k: dict(v) for k, v in sets.items()}
    s_e["dinov2ms_L11"] = {a: head(0.0) for a in A.ARCHS}; s_e["dinov2ms_L11"]["R2"] = head(-0.06)
    r6 = A.p6(D, s_e, cc)
    r = r6["dinov2ms_L11"]
    say(f"   (e) dinov2ms_L11 with R4 near zero (reference defined = {r['R4_reference_defined']}, R4 point sign "
        f"{r['R4_sign']:+d}) and R2 = -0.06: R2 opposite_sign_interval = {r['cells']['R2']['opposite_sign_interval']}; "
        f"title = {A.title_decision(r6, p3_ok)['title']}")


def main():
    say(f"independent_check.py; analyze_re.py sha256 {A.sha256(A.__file__)}; numpy {np.__version__}; "
        f"started {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    for f in (part1, part2, part4, part5, part6, part7, part3):
        f()
    say(f"\ndone in {time.time() - T0:.0f} s")
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "independent_check_output.txt"), "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
