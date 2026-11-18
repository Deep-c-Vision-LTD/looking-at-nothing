"""
Unit tests for analyze_re.py on synthetic data with known truth (no real score is read).
Revised 18 November 2026 after the rule reviews (REVIEW_RESPONSES.md): P5 in one currency with orientation-matched
controls (design finding 1), the title conditions (c0), (c1), (d) with an undefined reference, P6 with missing runs or
CCs, the published rounding of the accuracy criterion, the abstract rule of Section 5.5.
Run: nice -n 10 env OMP_NUM_THREADS=4 python test_analyze_re.py   (also works under pytest)
Coverage simulations use fixed seeds, so the printed numbers are reproducible.
"""
import math, os, sys, time
import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_re as A  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    return ok


def clustered(rng, G, P, mu, skew=False):
    """d_ig = mu + u_g + e_ig with heterogeneous cluster scales; skewed variant uses centred exponentials."""
    scale = np.linspace(0.5, 1.5, G)
    if skew:
        u = (rng.exponential(1.0, G) - 1.0) * scale * 0.6
        e = (rng.exponential(1.0, (G, P)) - 1.0) * 0.4
    else:
        u = rng.normal(0, 1, G) * scale * 0.6
        e = rng.standard_t(5, (G, P)) * 0.3
    return mu + u[:, None] + e


# ---------------------------------------------------------------------------------------------- basic pieces
def test_cluster_se():
    rng = np.random.RandomState(1)
    G, P = 12, 5
    d = clustered(rng, G, P, 0.2).ravel(); g = np.repeat(np.arange(G), P)
    e = d - d.mean(); S = np.array([e[g == k].sum() for k in range(G)])
    manual = math.sqrt(G / (G - 1) * (S ** 2).sum()) / d.size
    # equals the CR1 sandwich of OLS on an intercept: (X'X)^-1 X' Omega X (X'X)^-1 with the G/(G-1) factor
    check("cluster_se equals the CR1 sandwich of an intercept-only OLS", abs(A.cluster_se(d, g) - manual) < 1e-14,
          f"se {A.cluster_se(d, g):.6f}")


def test_webb_enumeration():
    W, enum = A.webb_weights(6)
    uniq = np.unique(W, axis=0).shape[0]
    freq = [np.sum(np.isclose(W[:, 0], v)) for v in A.WEBB]
    check("Webb enumeration for G = 6 has 6^6 = 46,656 distinct vectors", enum and W.shape == (46656, 6) and uniq == 46656,
          f"shape {W.shape}, distinct {uniq}")
    check("each Webb value appears equally often per cluster (7,776 times)", all(f == 7776 for f in freq), str(freq))
    check("Webb weights have mean 0 and variance 1", abs(A.WEBB.mean()) < 1e-15 and abs((A.WEBB ** 2).mean() - 1) < 1e-15)
    W12, enum12 = A.webb_weights(12)
    check("G = 12 uses 9,999 random draws (RandomState 2026)", (not enum12) and W12.shape == (9999, 12))
    W12b, _ = A.webb_weights(12)
    check("random draws are reproducible", np.array_equal(W12, W12b))


def test_wild_identity_and_pmin():
    rng = np.random.RandomState(2)
    G, P = 6, 5
    d = clustered(rng, G, P, 3.0).ravel(); g = np.repeat(np.arange(G), P)
    r = A.wild_cluster(d, g)
    check("G = 6 is enumerated and the lower bound on the two-sided p is 6 / 46,656",
          r["enumerated"] and abs(r["p_min"] - 6 / 46656) < 1e-15, f"p_min {r['p_min']:.3e}")
    # In practice a strong effect is bounded by the share of sign-constant weight vectors, 2 x 3^6 / 6^6 = 0.03125:
    # those vectors can give |t*| >= |t| when they make the weighted cluster sums more homogeneous than the data.
    check("a strong effect (t > 7) gets p < .05 and at most the sign-constant share 2 x 3^6 / 6^6 = 0.031",
          r["p"] < 0.05 and r["p"] <= 2 * 3 ** 6 / 6 ** 6 + 1e-12, f"p {r['p']:.4f}, t {r['t']:.2f}")
    homog = np.repeat(np.array([3.0, 3.1, 2.9, 3.05, 2.95, 3.0]), P) + np.tile(np.linspace(-0.01, 0.01, P), G)
    rh = A.wild_cluster(homog, g)
    check("with nearly identical cluster sums the p approaches the 6 / 46,656 lower bound (p < .005)", rh["p"] < 0.005,
          f"p {rh['p']:.5f}")
    # the identity weight vector reproduces t exactly under the restricted bootstrap with mu0 = 0
    n = np.bincount(g).astype(float); E0 = np.bincount(g, weights=d)
    m, se = A._wild_t(E0, n, d.size, G, np.ones((1, G)))
    check("identity weights reproduce the observed t under WCR", abs(m[0] / se[0] - r["t"]) < 1e-10)


def test_enum_vs_random():
    rng = np.random.RandomState(3)
    G, P = 6, 5
    d = clustered(rng, G, P, 0.25).ravel(); g = np.repeat(np.arange(G), P)
    r_enum = A.wild_cluster(d, g)
    r_rand = A.wild_cluster(d, g, B=200000, enum_max=0)
    check("exact enumeration and 200,000 random Webb draws give the same p (|diff| < 0.005)",
          abs(r_enum["p"] - r_rand["p"]) < 0.005, f"enum {r_enum['p']:.4f} random {r_rand['p']:.4f}")
    check("and the same interval (endpoints within 2% of its width)",
          max(abs(a - b) for a, b in zip(r_enum["ci"], r_rand["ci"])) < 0.02 * (r_enum["ci"][1] - r_enum["ci"][0]),
          f"enum {np.round(r_enum['ci'], 4)} random {np.round(r_rand['ci'], 4)}")


def test_two_stage_idx():
    ci, pi = A.two_stage_idx(12, 5, B=100)
    rng = np.random.RandomState(2026)
    ci2, pi2 = rng.randint(0, 12, (100, 12)), rng.randint(0, 5, (100, 12, 5))
    check("two-stage indices equal the v2 construction (RandomState(2026), clusters first)",
          np.array_equal(ci, ci2) and np.array_equal(pi, pi2))


# ---------------------------------------------------------------------------------------------- coverage
def coverage(G, P, nsim, skew, B=1999, seed=10):
    rng = np.random.RandomState(seed)
    mu = 0.3; g = np.repeat(np.arange(G), P)
    cov_wcu = rej_wcr = cov_pct = 0
    idx = A.two_stage_idx(G, P, B=1000, seed=seed)
    for _ in range(nsim):
        x = clustered(rng, G, P, mu, skew)
        r = A.wild_cluster(x.ravel(), g, mu0=mu, B=B)
        cov_wcu += r["ci"][0] <= mu <= r["ci"][1]
        rej_wcr += r["p"] < 0.05
        s = A.pct_summary(x, idx)
        cov_pct += s["ci"][0] <= mu <= s["ci"][1]
    return cov_wcu / nsim, rej_wcr / nsim, cov_pct / nsim


def test_coverage_g12():
    """Symmetric (normal cluster effects, t5 noise, cluster scales 0.5 to 1.5): the method must be close to nominal.
    Strongly skewed stress case (centred exponential cluster effects and noise): symmetric wild weights cannot
    correct skewness, so the criterion is only that it does better than the published two-stage percentile
    interval; the observed coverage is reported in the rule."""
    se = math.sqrt(0.95 * 0.05 / 1000)
    cw, rr, cp = coverage(12, 5, 1000, False)
    check("G = 12 (symmetric errors): WCU 95% interval coverage within [0.92, 0.98]",
          0.92 <= cw <= 0.98, f"coverage {cw:.3f} (MC SE {se:.3f}); two-stage percentile interval {cp:.3f}")
    check("G = 12 (symmetric errors): WCR test size at .05 within [0.02, 0.08]", 0.02 <= rr <= 0.08, f"rejection rate {rr:.3f}")
    cw, rr, cp = coverage(12, 5, 1000, True)
    check("G = 12 (strongly skewed stress case): WCU coverage at least that of the two-stage percentile interval and >= 0.87",
          cw >= cp and cw >= 0.87, f"coverage {cw:.3f} vs percentile {cp:.3f}; WCR size {rr:.3f} (reported, not a pass criterion)")


def test_coverage_g6():
    se = math.sqrt(0.95 * 0.05 / 600)
    cw, rr, cp = coverage(6, 5, 600, False, seed=11)
    check("G = 6, full enumeration (symmetric errors): WCU coverage within [0.91, 0.99]",
          0.91 <= cw <= 0.99, f"coverage {cw:.3f} (MC SE {se:.3f}); two-stage percentile interval {cp:.3f}")
    check("G = 6, full enumeration (symmetric errors): WCR size within [0.01, 0.09]", 0.01 <= rr <= 0.09, f"rejection rate {rr:.3f}")
    cw, rr, cp = coverage(6, 5, 600, True, seed=11)
    check("G = 6, full enumeration (strongly skewed stress case): WCU coverage at least that of the percentile interval and >= 0.86",
          cw >= cp and cw >= 0.86, f"coverage {cw:.3f} vs percentile {cp:.3f}; WCR size {rr:.3f} (reported, not a pass criterion)")


# ---------------------------------------------------------------------------------------------- BCa
def test_bca_symmetric():
    rng = np.random.RandomState(4)
    x = rng.normal(1.0, 1.0, (40, 1))
    idx = A.two_stage_idx(40, 1, B=20000, seed=5)
    b = A.bca(lambda m: m, [x], idx)
    p = A.pct_summary(x, idx)
    check("BCa equals the percentile interval when the statistic is a symmetric mean (|z0|, |a| small)",
          abs(b["z0"]) < 0.05 and abs(b["acc"]) < 0.02 and max(abs(u - v) for u, v in zip(b["ci"], p["ci"])) < 0.02,
          f"BCa {np.round(b['ci'], 4)} pct {np.round(p['ci'], 4)} z0 {b['z0']:.3f} a {b['acc']:.4f}")


def test_bca_vs_scipy():
    rng = np.random.RandomState(6)
    x = rng.lognormal(0, 0.8, 30)
    idx = A.two_stage_idx(30, 1, B=40000, seed=7)
    b = A.bca(lambda m: np.log(m), [x[:, None]], idx)
    s = sps.bootstrap((x,), lambda v, axis=-1: np.log(np.mean(v, axis=axis)), n_resamples=40000, method="BCa",
                      random_state=np.random.RandomState(8))
    lo, hi = s.confidence_interval
    w = hi - lo
    check("BCa matches scipy.stats.bootstrap(method='BCa') on i.i.d. skewed data (endpoints within 3% of width)",
          abs(b["ci"][0] - lo) < 0.03 * w and abs(b["ci"][1] - hi) < 0.03 * w,
          f"ours {np.round(b['ci'], 4)} scipy [{lo:.4f}, {hi:.4f}]")


def test_bca_p_inversion():
    rng = np.random.RandomState(9)
    G, P = 12, 5
    num = clustered(rng, G, P, 0.5, skew=True); den = clustered(rng, G, P, 1.0)
    idx = A.two_stage_idx(G, P, B=10000, seed=2026)
    stat = lambda a, b: a / b
    theta0 = 0.2
    b = A.bca(stat, [num, den], idx, theta0=theta0)
    b2 = A.bca(stat, [num, den], idx, alpha=b["p"])
    end = min(b2["ci"], key=lambda e: abs(e - theta0))
    w = b["ci"][1] - b["ci"][0]
    check("BCa p-value inverts the BCa interval: the (1 - p) interval has theta0 as an endpoint (within 1% of width)",
          abs(end - theta0) < 0.01 * w, f"p {b['p']:.4f}; endpoint {end:.4f} vs theta0 {theta0}")


def test_bca_ratio_coverage():
    rng = np.random.RandomState(12)
    G, P, nsim = 12, 5, 300
    true = 0.6
    hit = hit_f = 0
    for i in range(nsim):
        den = 1.0 + clustered(rng, G, P, 0.0) * 0.3
        num = true * den + clustered(rng, G, P, 0.0, skew=True) * 0.15
        idx = A.two_stage_idx(G, P, B=2000, seed=100 + i)
        b = A.bca(lambda a, c: a / c, [num, den], idx)
        hit += b["ci"][0] <= true <= b["ci"][1]
        f = A.fieller(num, den, np.repeat(np.arange(G), P))
        hit_f += f["bounded"] and f["ci"][0] <= true <= f["ci"][1]
    check("pairs cluster BCa interval for a ratio of means (G = 12): coverage within [0.90, 0.995]",
          0.90 <= hit / nsim <= 0.995, f"coverage {hit / nsim:.3f} over {nsim} datasets; Fieller {hit_f / nsim:.3f}")


# ---------------------------------------------------------------------------------------------- Holm, stability, Fieller
def test_holm():
    adj, rej = A.holm([0.01, 0.04, 0.03, 0.005])
    check("Holm adjusted p-values for [.01, .04, .03, .005] are [.03, .06, .06, .02]",
          np.allclose(adj, [0.03, 0.06, 0.06, 0.02]) and list(rej) == [True, False, False, True], f"{np.round(adj, 3)}")
    adj, rej = A.holm([0.2, 0.001])
    check("Holm with m = 2", np.allclose(adj, [0.2, 0.002]) and list(rej) == [False, True])
    # family-wise error under the global null with independent uniform p-values
    rng = np.random.RandomState(13)
    fwe = np.mean([A.holm(rng.uniform(size=10))[1].any() for _ in range(20000)])
    check("Holm controls the family-wise error at .05 (10 independent true nulls)", fwe <= 0.055, f"FWE {fwe:.4f}")


def test_stability():
    G, P = 12, 5; g = np.repeat(np.arange(G), P)
    rng = np.random.RandomState(14)
    base = clustered(rng, G, P, 0.0).ravel()
    base = base - base.mean()
    se = A.cluster_se(base, g)
    d3, d5 = base + 3 * se, base + 5 * se
    check("stability screen: |t| = 3 is unstable, |t| = 5 is stable",
          A.t_den(d3, g) < A.T_STABLE <= A.t_den(d5, g), f"t {A.t_den(d3, g):.2f}, {A.t_den(d5, g):.2f}")


def test_fieller():
    G, P = 12, 5; g = np.repeat(np.arange(G), P)
    rng = np.random.RandomState(15)
    den = 1.0 + clustered(rng, G, P, 0.0) * 0.2
    num = 0.5 * den + clustered(rng, G, P, 0.0) * 0.1
    f = A.fieller(num, den, g)
    check("Fieller interval is bounded and contains the point estimate for a well-determined denominator",
          f["bounded"] and f["ci"][0] < f["value"] < f["ci"][1], f"{f['value']:.3f} {np.round(f['ci'], 3)} t_den {f['t_den']:.1f}")
    den0 = clustered(rng, G, P, 0.0) * 0.5
    f0 = A.fieller(num, den0, g)
    check("Fieller interval is unbounded when the denominator is not distinguishable from zero", not f0["bounded"],
          f"t_den {f0['t_den']:.2f}")


# ---------------------------------------------------------------------------------------------- contrasts on a synthetic design
def synth_meta(npos=5):
    meta = {}
    for sh in ("square", "triangle"):
        for sr in (0.3, 0.5, 0.7):
            for pol in ("dark", "light"):
                for p in range(npos):
                    meta[f"{sh}_sr{int(sr * 100)}_{pol}_p{p}"] = (sh, sr, pol, p)
    return meta


def synth_scores(D, rng, eff=0.1, a=0.05, o_in=0.0, o_out=0.0, g_in=0.0, g_out=0.0, alt_keep=0.5, noise=0.01,
                 rig_noise=None):
    """{name: value} in the terms of the design review (finding 1): per configuration, inward baseline b_in = base + eff,
    outward baseline b_out = base; a = alignment effect lost when the inward inducers are misaligned; o_in, o_out =
    change produced by turning every notch edge by theta (inward, outward displays); g_in, g_out = change produced by
    rotating the configuration on the canvas. Displays:
      I = b_in, C = b_out, M (mis20) = b_in + o_in - a, MC (mis20_control180) = b_out + o_out,
      RI (rig20_illusory) = b_in + o_in + g_in, RC (rig20_control180) = b_out + o_out + g_out,
      alt20 = M + alt_keep * a (each phase).
    Then L_uni(contrast) = a - o_in + o_out, L_rig = -(o_in - o_out) - (g_in - g_out), A* = a + g_in - g_out, and the
    drafted (rejected) alignment loss (RI - M) - (RC - C) = a + g_in - g_out - o_out."""
    rn = noise / 4 if rig_noise is None else rig_noise
    sc = {}
    for cell_k, cell in zip(D.keys, D.cells):
        cu = rng.normal(0, noise)
        for c in cell:
            base = 0.3 + rng.normal(0, noise)
            ill = base + eff + cu + rng.normal(0, noise / 2)
            sc[f"{c}_illusory"] = ill; sc[f"{c}_control180"] = base
            mis = ill + o_in - a + rng.normal(0, noise / 2)
            sc[f"{c}_mis20"] = mis
            sc[f"{c}_mis20_control180"] = base + o_out + rng.normal(0, noise / 4)
            for ph in "ab":
                sc[f"{c}_alt20{ph}"] = mis + alt_keep * a + rng.normal(0, noise / 4)
            sc[f"{c}_rig20_control180"] = base + o_out + g_out + rng.normal(0, rn)
            sc[f"{c}_rig20_illusory"] = ill + o_in + g_in + rng.normal(0, rn)
    return sc


def null_like(D, rng):
    """A local null: large positive E, gains a little figure mass under misalignment, no alternation retention."""
    return synth_scores(D, rng, eff=0.2, a=-0.005, alt_keep=0.0, noise=0.004)


def test_p1_p2_p5_logic():
    meta = synth_meta(); D = A.Design(meta); Dsq = A.Design(meta, "square")
    rng = np.random.RandomState(16)
    nulls = {f"n{i}": A.getter(null_like(D, rng)) for i in range(15)}
    good = A.getter(synth_scores(D, rng, a=0.05))
    flat = A.getter(synth_scores(D, rng, a=-0.005, alt_keep=0.0))
    r1 = A.p1(D, {"M": good, "F": flat}, nulls)
    check("P1: a model that loses 0.05 figure mass passes the IUT against 15 nulls that gain mass", r1["M"]["met"],
          f"separated {r1['M']['n_separated']}/15")
    check("P1: a model that behaves like the nulls fails the IUT", not r1["F"]["met"], f"separated {r1['F']['n_separated']}/15")
    r2 = A.p2(Dsq, {"M": good, "F": flat}, nulls)
    any_cmp = next(iter(r2["M"]["vs_null"].values()))
    check("P2 runs on 6 square clusters with full enumeration and passes for alternation retention 0.5",
          r2["M"]["met"] and any_cmp["enumerated"] and any_cmp["G"] == 6, f"draws {any_cmp['draws']}")
    check("P2 fails for a model without alternation retention", not r2["F"]["met"])
    check("P2 wording is ratio-free (no 'cost', no ratio) and names the separated model",
          "cost" not in A.wording_p2(r2) and "0.55" not in A.wording_p2(r2) and "for M" in A.wording_p2(r2),
          A.wording_p2(r2)[:90])
    # P5 categories, end to end
    c1 = A.getter(synth_scores(D, rng, a=0.05))                          # rigid rotation removes nothing
    c2a = A.getter(synth_scores(D, rng, a=0.03, o_in=-0.04))             # rigid reproduces 0.04 of a 0.07 loss
    c3 = A.getter(synth_scores(D, rng, a=-0.01, o_in=-0.05))             # orientation alone; no alignment effect
    r5 = A.p5(D, {"c1": c1, "c2a": c2a, "c3": c3, "nc": None}, nulls)
    check("P5: rigid rotation removes none of the loss -> category 1", r5["c1"]["category"] == "1",
          f"s* {r5['c1']['rigid_share']['value']:.2f}, A* {r5['c1']['align_effect']['mean']:+.3f}")
    check("P5: rigid rotation reproduces more than half of the loss, alignment effect still beyond the nulls -> 2a",
          r5["c2a"]["category"] == "2a",
          f"s* {r5['c2a']['rigid_share']['value']:.2f}, iut {r5['c2a']['iut_met']}, C* p {r5['c2a']['half_criterion']['p']:.3f}")
    check("P5: an orientation effect without alignment effect -> category 3", r5["c3"]["category"] == "3",
          f"s* {r5['c3']['rigid_share']['value']:.2f}, separated {r5['c3']['n_separated']}/15")
    check("P5: a model without rigid scores is not computable and not in category 1",
          r5["nc"]["category"] is None and not r5["nc"]["computable"] and r5["abstract_leads_with_alternation"])
    check("P5: secondary quantities (drafted A, control-free A, control change) are reported, never categorised",
          set(r5["c1"]["secondary"]) == {"A_drafted", "A_control_free", "control_rigid_change"})


def test_p5_orientation_of_control():
    """Design review, finding 1: with no alignment effect but an orientation response of the outward control
    (o_out = 0.02), the drafted alignment loss is about -0.02 while A* is about 0; with a = 0.03, A* is about 0.03."""
    meta = synth_meta(); D = A.Design(meta)
    rng = np.random.RandomState(21)
    for a_true, want_star, want_draft in ((0.0, 0.0, -0.02), (0.03, 0.03, 0.01)):
        v = A.getter(synth_scores(D, rng, a=a_true, o_out=0.02, noise=0.004))
        L_uni, L_rig, Astar, sec = A.p5_parts(D, v)
        check(f"P5 estimand: a = {a_true}, o_out = 0.02 -> A* = {want_star:+.2f} and drafted A = {want_draft:+.2f} (within 0.004)",
              abs(Astar.mean() - want_star) < 0.004 and abs(sec["A_drafted"].mean() - want_draft) < 0.004,
              f"A* {Astar.mean():+.4f}, drafted {sec['A_drafted'].mean():+.4f}, L_uni {L_uni.mean():+.4f}, L_rig {L_rig.mean():+.4f}")
    v = A.getter(synth_scores(D, rng, a=0.0, o_in=0.01, o_out=0.01, g_in=0.02, g_out=0.005, noise=0.004))
    L_uni, L_rig, Astar, sec = A.p5_parts(D, v)
    check("P5 estimand: canvas effects enter A* as g_in - g_out (0.015), equal orientation responses cancel",
          abs(Astar.mean() - 0.015) < 0.004 and abs(L_uni.mean()) < 0.004, f"A* {Astar.mean():+.4f}, L_uni {L_uni.mean():+.4f}")


def test_p5_category_and_abstract():
    cat = A.p5_category
    check("P5 categories: (i)+(ii) -> 1; (i), not (ii), s* 0.6 -> 2a; s* 0.3 -> 2b; no loss -> 2b; not (i) -> 3",
          cat(True, True, 0.05, 0.1) == "1" and cat(True, False, 0.05, 0.6) == "2a" and cat(True, False, 0.05, 0.3) == "2b"
          and cat(True, False, -0.01, 0.9) == "2b" and cat(False, True, 0.05, 0.1) == "3")
    p1 = {"DeepGaze IIE": dict(met=True), "DeepGaze MSDB": dict(met=True), "wording": "x"}
    p2b = {"DeepGaze IIE": dict(met=True), "DeepGaze MSDB": dict(met=True), "wording": "x"}
    p2o = {"DeepGaze IIE": dict(met=True), "DeepGaze MSDB": dict(met=False), "wording": "x"}
    p2n = {"DeepGaze IIE": dict(met=False), "DeepGaze MSDB": dict(met=False), "wording": "x"}
    p5a = {"DeepGaze IIE": dict(category="1"), "DeepGaze MSDB": dict(category="1")}
    p5b = {"DeepGaze IIE": dict(category="1"), "DeepGaze MSDB": dict(category="2b")}
    r_a = A.abstract_rotation_rule(p1, p2b, p5a); r_b = A.abstract_rotation_rule(p1, p2b, p5b)
    r_o = A.abstract_rotation_rule(p1, p2o, p5b); r_n = A.abstract_rotation_rule(p1, p2n, p5b)
    check("abstract: both P1 met and both P5 category 1 -> uniform-rotation clause stays", r_a["uniform_models"] == ["DeepGaze IIE", "DeepGaze MSDB"]
          and r_a["highlight4"].startswith("Inducer rotation"), r_a["abstract"])
    check("abstract: a model outside category 1 -> lead with square alternation (both P2 met)", "square alternation" in r_b["abstract"]
          and r_b["highlight4"] == "On squares, alternating rotation separates DeepGaze IIE and MSDB from 15 local nulls", r_b["highlight4"])
    check("abstract: P2 met for one model only -> alternation for that model only", r_o["alternation_models"] == ["DeepGaze IIE"]
          and "IIE from" in r_o["highlight4"], r_o["highlight4"])
    check("abstract: neither P5 category 1 for both nor any P2 -> no rotation claim, no highlight 4", r_n["highlight4"] is None
          and r_n["abstract"] == "no rotation claim in the abstract")
    check("highlight 4 variants are at most 85 characters", all(r["highlight4_chars"] <= 85 for r in (r_a, r_b, r_o)),
          str([r["highlight4_chars"] for r in (r_a, r_b, r_o)]))


def test_p3_p4_p6_title():
    meta = synth_meta(); D = A.Design(meta)
    rng = np.random.RandomState(17)
    neg = lambda e: A.getter(synth_scores(D, rng, eff=e, a=0.3 * e))
    heads = {("f1", "R4"): neg(-0.1), ("f1", "R1"): neg(-0.08), ("f1", "R2"): neg(-0.002),
             ("f2", "R4"): neg(0.09), ("f2", "R2"): neg(0.07), ("f2", "R3"): neg(-0.05)}
    r3 = A.p3(D, {k: v for k, v in heads.items() if k[1] != "R4"}, {"f1": heads[("f1", "R4")], "f2": heads[("f2", "R4")]})
    c = r3["cells"]
    check("P3: same-sign cells with clear effects keep the sign after Holm", c["f1|R1"]["keeps_sign"] and c["f2|R2"]["keeps_sign"])
    check("P3: a near-zero cell is not counted and is not called a reversal",
          (not c["f1|R2"]["keeps_sign"]) and (not c["f1|R2"]["opposite_sign_interval"]))
    check("P3: an opposite-signed cell is flagged (title and claim narrowed)", c["f2|R3"]["opposite_sign_interval"] and r3["any_opposite"])
    check("WCR results carry the skewness of the cluster means (diagnostic)", "skew_cluster_means" in c["f1|R1"],
          f"skew {c['f1|R1']['skew_cluster_means']:+.2f}")
    # P4 with one unstable denominator
    h4 = {("f", "R4"): A.getter(synth_scores(D, rng, eff=0.1, a=0.03)),
          ("f", "R2"): A.getter(synth_scores(D, rng, eff=0.1, a=0.06)),
          ("f", "R3"): A.getter(synth_scores(D, rng, eff=0.004, a=0.0, noise=0.01))}
    r4 = A.p4(D, h4, contrasts=[("f", "R2", "U20"), ("f", "R3", "U20")])
    k2, k3 = r4["contrasts"]["f|R2|U20"], r4["contrasts"]["f|R3|U20"]
    check("P4: a stable contrast with a real profile difference survives Holm and is counted", k2["survives"] and k2["counted"],
          f"diff {k2['diff']:+.3f} {np.round(k2['ci'], 3)} p_holm {k2['p_holm']:.4f}")
    check("P4: a contrast with an unstable denominator is never counted", (not k3["stable"]) and (not k3["counted"]),
          f"t_den {k3['t_den_min']:.2f}")
    check("P4: Fieller intervals are computed for both ratios of a contrast", "fieller_R4" in k2 and "fieller_readout" in k2
          and k2["fieller_R4"]["bounded"])
    # P6 and the title decision
    def sets(e6=0.1, e12=0.05, ems=0.08, e11=-0.06):
        return {"dinov2reg_L6": {a_: neg(e6) for a_ in A.ARCHS}, "dinov2reg_L12": {a_: neg(e12) for a_ in A.ARCHS},
                "dinov2ms_L7L11": {a_: neg(ems) for a_ in A.ARCHS}, "dinov2ms_L11": {a_: neg(e11) for a_ in A.ARCHS}}
    newsets = sets()
    cc = {fs: {"R4": 0.74, "R1n": 0.70, "R1": 0.72, "R2": 0.73, "R3": 0.735} for fs in newsets}
    r6 = A.p6(D, newsets, cc)
    check("P6: the accuracy criterion is applied against R4 on the same new feature set (R1n at -0.04 is below)",
          r6["dinov2reg_L6"]["cells"]["R1n"]["status"] == "below criterion" and r6["dinov2reg_L6"]["cells"]["R1"]["status"] == "valid"
          and r6["dinov2reg_L6"]["n_valid"] == 3)
    t = A.title_decision(r6, r3)
    check("title: an opposite-signed valid interval (here in P3) rules out both titles (claim narrowed)",
          (not t["stronger"]) and t["narrowed"] and t["title"] is None, "; ".join(t["reasons"][:2]))
    r3ok = dict(r3, any_opposite=False, n_keep=r3["n"])
    t2 = A.title_decision(r6, r3ok)
    check("title: all sets complete, R4 defined, >= 2 valid, every valid cell keeps the sign, P3 in full -> stronger title",
          t2["stronger"] and t2["title"] == A.T1, t2["title"][:60])
    r3part = dict(r3, any_opposite=False, n_keep=r3["n"] - 1)
    tp = A.title_decision(r6, r3part)
    check("title (c1): P3 not met in full (one cell fails to exclude zero) -> scoped title, not narrowed",
          (not tp["stronger"]) and (not tp["narrowed"]) and any(x.startswith("(c1)") for x in tp["reasons"]), tp["reasons"][0])
    r6b = {k: v for k, v in r6.items() if k != "dinov2ms_L7L11"}
    tb = A.title_decision(r6b, r3ok)
    check("title (a): a missing A14 run forces the scoped title", (not tb["stronger"]) and tb["title"] == A.TS, tb["reasons"][0])
    r6c = {k: v for k, v in r6.items() if k != "dinov2ms_L11"}
    tc = A.title_decision(r6c, r3ok)
    check("title (a): a missing hs[11]-alone run also forces the scoped title (it can veto through (d))",
          (not tc["stronger"]) and tc["title"] == A.TS, tc["reasons"][0])
    cc_low = {fs: dict(v) for fs, v in cc.items()}
    cc_low["dinov2reg_L12"].update(R1n=0.70, R1=0.70, R2=0.70, R3=0.70)
    tv = A.title_decision(A.p6(D, newsets, cc_low), r3ok)
    check("title (c0): no valid non-R4 readout on a title-bearing set -> scoped title (no vacuous pass)",
          (not tv["stronger"]) and tv["title"] == A.TS and any(x.startswith("(c0)") for x in tv["reasons"]), tv["reasons"][0])
    cc_one = {fs: dict(v) for fs, v in cc.items()}
    cc_one["dinov2ms_L7L11"].update(R1n=0.70, R1=0.70, R2=0.70)
    to = A.title_decision(A.p6(D, newsets, cc_one), r3ok)
    check("title (c0): one valid non-R4 readout (fewer than 2) -> scoped title", (not to["stronger"]) and to["title"] == A.TS,
          to["reasons"][0])
    nz = dict(newsets); nz["dinov2reg_L12"] = dict(newsets["dinov2reg_L12"], R3=neg(0.0005))
    tz = A.title_decision(A.p6(D, nz, cc), r3ok)
    check("title (c): a valid registers cell whose interval includes zero forces the scoped title (not a narrowing)",
          (not tz["stronger"]) and (not tz["narrowed"]), "; ".join(tz["reasons"][:2]))
    no4 = {k: dict(v) for k, v in newsets.items()}; del no4["dinov2reg_L6"]["R4"]
    t4 = A.title_decision(A.p6(D, no4, cc), r3ok)
    check("P6: a missing R4 makes the set 'not run' (no exception) and forces the scoped title",
          A.p6(D, no4, cc)["dinov2reg_L6"]["status"] == "not run" and t4["title"] == A.TS, t4["reasons"][0])
    cc_nocc = {fs: dict(v) for fs, v in cc.items()}; del cc_nocc["dinov2ms_L7L11"]["R2"]
    r6n = A.p6(D, newsets, cc_nocc)
    tn = A.title_decision(r6n, r3ok)
    check("P6: a cell with scores but no held-out CC is 'not run' (no exception) and forces the scoped title",
          r6n["dinov2ms_L7L11"]["cells"]["R2"]["status"] == "not run" and tn["title"] == A.TS, tn["reasons"][0])
    for sign in (+1, -1):
        un = dict(newsets); un["dinov2ms_L11"] = dict(newsets["dinov2ms_L11"], R4=neg(sign * 0.0005), R2=neg(-0.06 * sign))
        tu = A.title_decision(A.p6(D, un, cc), r3ok)
        ru = A.p6(D, un, cc)["dinov2ms_L11"]
        check(f"title (d): R4 undefined on hs[11] (point estimate {sign:+d} x 0.0005) and a valid readout with a clear "
              "sign -> both titles ruled out", (not ru["R4_reference_defined"]) and tu["narrowed"] and tu["title"] is None,
              "; ".join(tu["reasons"][-1:]))
    bad = sets(); bad["dinov2reg_L12"]["R2"] = neg(-0.05)
    t3 = A.title_decision(A.p6(D, bad, cc), r3ok)
    check("title (d): an opposite-signed valid cell on the registers checkpoint rules out both titles", (not t3["stronger"]) and t3["narrowed"],
          "; ".join(t3["reasons"][:2]))
    try:
        A.title_decision(r6, None); ok = False
    except ValueError:
        ok = True
    check("title: the P3 result is mandatory", ok)


def test_accuracy_rule():
    check("accuracy criterion: CC equal to R4 - 0.03 is valid, below it is not",
          A.accuracy_valid(0.7122, 0.7422) and not A.accuracy_valid(0.7121, 0.7422))
    check("accuracy criterion uses the published rounding (cell 0.71216 -> 0.7122 against round(0.7422 - 0.03, 4) = 0.7122)",
          A.accuracy_valid(0.71216, 0.74224) and not (0.71216 >= 0.74224 - 0.03))
    check("published thresholds reproduced: registers block 12 R4 0.72824 -> 0.6982, block 6 R4 0.73773 -> 0.7077",
          A.accuracy_valid(0.6982, 0.72824) and not A.accuracy_valid(0.6981, 0.72824)
          and A.accuracy_valid(0.7077, 0.73773) and not A.accuracy_valid(0.7076, 0.73773))


def main():
    t0 = time.time()
    print(f"analyze_re.py sha256 {A.sha256(A.__file__)}")
    print(f"numpy {np.__version__}; started {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    for f in (test_cluster_se, test_webb_enumeration, test_wild_identity_and_pmin, test_enum_vs_random, test_two_stage_idx,
              test_holm, test_stability, test_fieller, test_accuracy_rule, test_bca_symmetric, test_bca_vs_scipy,
              test_bca_p_inversion, test_p1_p2_p5_logic, test_p5_orientation_of_control, test_p5_category_and_abstract,
              test_p3_p4_p6_title, test_coverage_g12, test_coverage_g6, test_bca_ratio_coverage):
        print(f"--- {f.__name__}", flush=True)
        f()
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed, {time.time() - t0:.0f} s")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
