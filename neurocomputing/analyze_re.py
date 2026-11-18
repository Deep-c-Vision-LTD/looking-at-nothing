"""
analyze_re.py: statistics engine and primary contrasts P1 to P6 of neuro/reanalysis/REANALYSIS_RULE.md.

Pure numpy / scipy. No torch, no model, no GPU. The functions in the first half are generic and unit-tested on
synthetic data in test_analyze_re.py; the second half loads the saved per-image scores and applies them.

Unit of analysis: one value per Kanizsa configuration (60 = 12 clusters x 5 positions; 30 = 6 x 5 for a shape-only
contrast), seeds averaged per configuration and the 336 and 448 px values averaged (size average) where a readout was
scored at both sizes. Clusters = shape x support ratio x polarity cells, ordered as v2/scores/analyze_v3.design
(sorted by str of the key); positions are observations within clusters.

Interval and test methods (rule Section 4):
  linear contrasts (a mean of per-configuration values d_i): wild cluster bootstrap with Webb six-point weights.
    test  (WCR): residuals under H0 (d_i - mu0), t* = (mean* - mu0) / se*, two-sided equal-tailed p.
    interval (WCU): unrestricted residuals, percentile-t interval [mu - q(1 - a/2) se, mu - q(a/2) se].
    se = CR1 cluster-robust standard error of the mean: sqrt(G / (G - 1) * sum_g (sum_{i in g} e_i)^2) / N.
    G <= 6: all 6^G weight vectors are enumerated (46,656 for G = 6); otherwise 9,999 random draws,
    numpy RandomState(2026). Two-sided equal-tailed p: for draws p = 2 (1 + min(#{t* >= t}, #{t* <= t})) / (B + 1);
    for enumeration p = 2 min(share of t* >= t, share of t* <= t). Decisions use the WCR p; the WCU interval is the
    reported interval estimate (rule Section 4.1). Diagnostic, no decision: the sample skewness of the cluster means.
  ratios of means and paired ratio differences: pairs cluster bootstrap (the two-stage v2 resample: clusters with
    replacement, positions with replacement within each drawn cluster; 10,000 resamples, RandomState(2026)) with the
    BCa adjustment (Efron 1987; bias from the bootstrap distribution, acceleration from the delete-one-cluster
    jackknife); p-value by inverting the BCa interval. Fieller intervals (cluster-robust covariance, t with G - 1 df)
    are computed alongside as a check.
  multiplicity: Holm (1979) within a stated family; intersection-union test (Berger and Hsu 1996) where a claim
    needs every component (no correction).
  stability: a ratio is "unstable" if |t| of its denominator (mean / CR1 se) is below 4; unstable ratios are never
    counted (rule Section 5.4).

Revised on 18 November 2026 after the two rule reviews (neuro/reanalysis/REVIEW_RESPONSES.md), before the rule was
fixed: P5 in one currency with orientation-matched controls (A*, C*, s*, categories 1, 2a, 2b, 3); the title rule with
conditions (a) to (d) of rule Section 5.6 including (c0) and (c1); P6 robust to missing runs; the accuracy criterion
with the published rounding; Fieller intervals for both ratios of a P4 contrast; denominator checks; the skewness
diagnostic; require_public() verifies every hashed file and every stimulus file.

Usage (every step, the sanity gate included, refuses to run before the rule and this file are public; see
require_public()):
  python analyze_re.py sanity                    reproduce published percentile intervals (gate; prints PASS/FAIL)
  python analyze_re.py primary --which P1,P2,P3,P4
  python analyze_re.py primary --which P5        after the whole-display rotation set has been scored
  python analyze_re.py primary --which P6        after the G3 and A14 GPU runs have been scored
Outputs: neuro/reanalysis/out/<step>.json
"""
import argparse, hashlib, itertools, json, math, os, sys, time
import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(HERE))   # project root
OUT = os.path.join(HERE, "out")
SEED = 2026
B_WILD = 9999
B_PAIRS = 10000
ENUM_MAX_G = 6
ALPHA = 0.05
T_STABLE = 4.0
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


# ====================================================================== generic statistics
def cluster_se(d, g):
    """CR1 cluster-robust standard error of the mean of d (groups g, labels 0..G-1)."""
    d = np.asarray(d, float); g = np.asarray(g)
    G = int(g.max()) + 1; N = d.size
    e = d - d.mean()
    S = np.bincount(g, weights=e, minlength=G)
    return math.sqrt(G / (G - 1) * float((S ** 2).sum())) / N


def webb_weights(G, B=B_WILD, seed=SEED, enum_max=ENUM_MAX_G):
    """(W, enumerated): W has shape (n_draws, G). Full enumeration of the 6^G Webb vectors if G <= enum_max."""
    if G <= enum_max:
        idx = np.array(list(itertools.product(range(6), repeat=G)), dtype=np.int64)
        return WEBB[idx], True
    rng = np.random.RandomState(seed)
    return WEBB[rng.randint(0, 6, (B, G))], False


def _wild_t(E, n, N, G, W):
    """t* numerators and se* for weight matrix W given cluster sums E of the residuals used to build d*."""
    A = W * E[None, :]                                   # cluster sums of v_g * e_i
    m = A.sum(1) / N                                     # mean* - centre
    S = A - n[None, :] * m[:, None]                      # cluster sums of bootstrap residuals
    se = np.sqrt(G / (G - 1) * (S ** 2).sum(1)) / N
    return m, se


def wild_cluster(d, g, mu0=0.0, alpha=ALPHA, B=B_WILD, seed=SEED, enum_max=ENUM_MAX_G):
    """Wild cluster bootstrap-t for the mean of d. Returns mean, se, t, two-sided WCR p, WCU percentile-t CI."""
    d = np.asarray(d, float); g = np.asarray(g, int)
    G = int(g.max()) + 1; N = d.size
    n = np.bincount(g, minlength=G).astype(float)
    mu = float(d.mean()); se = cluster_se(d, g)
    t = (mu - mu0) / se if se > 0 else (math.copysign(math.inf, mu - mu0) if mu != mu0 else 0.0)
    W, enum = webb_weights(G, B, seed, enum_max)
    # WCR: impose H0
    E0 = np.bincount(g, weights=d - mu0, minlength=G)
    m0, se0 = _wild_t(E0, n, N, G, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(se0 > 0, m0 / se0, np.where(m0 == 0, 0.0, np.sign(m0) * np.inf))
    up, lo = np.mean(t0 >= t - 1e-12), np.mean(t0 <= t + 1e-12)
    if enum:
        p = min(1.0, 2 * min(up, lo))
        p_min = 2 * 3 / W.shape[0]                       # the 3 constant positive (or negative) vectors reproduce t
    else:
        k_up, k_lo = int(np.sum(t0 >= t - 1e-12)), int(np.sum(t0 <= t + 1e-12))
        p = min(1.0, 2 * (1 + min(k_up, k_lo)) / (W.shape[0] + 1))
        p_min = 2 / (W.shape[0] + 1)
    # WCU: unrestricted residuals, percentile-t interval
    E1 = np.bincount(g, weights=d - mu, minlength=G)
    m1, se1 = _wild_t(E1, n, N, G, W)
    ok = se1 > 0
    tu = m1[ok] / se1[ok]
    qlo, qhi = np.quantile(tu, [alpha / 2, 1 - alpha / 2])
    ci = [mu - qhi * se, mu - qlo * se]
    cm = np.bincount(g, weights=d, minlength=G) / n            # cluster means (diagnostic only)
    skew = float(sps.skew(cm)) if np.ptp(cm) > 0 else 0.0
    return dict(mean=mu, se=se, t=t, p=float(p), p_min=float(p_min), ci=[float(ci[0]), float(ci[1])],
                draws=int(W.shape[0]), enumerated=bool(enum), G=G, N=N, skew_cluster_means=skew)


def two_stage_idx(K, P, B=B_PAIRS, seed=SEED):
    """The v2 cluster resample (v2/scores/analyze_v3.py): clusters with replacement, then positions with replacement
    within each drawn cluster slot; RandomState(seed), cluster draws first."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, K, (B, K)), rng.randint(0, P, (B, K, P))


def bmeans(v, idx):
    """Bootstrap means of a (K, P) array under two-stage indices."""
    ci, pi = idx
    return v[ci[:, :, None], pi].mean((1, 2))


def pct_summary(v, idx, alpha=ALPHA):
    """Published-style percentile interval and two-sided bootstrap p for the mean of a (K, P) array."""
    bm = bmeans(v, idx)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)))
    return dict(mean=float(v.mean()), ci=[float(lo), float(hi)], p=float(p))


def pct_ratio(num, den, idx, alpha=ALPHA):
    bn, bd = bmeans(num, idx), bmeans(den, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(value=float(num.mean() / den.mean()), ci=[float(lo), float(hi)])


def _jack_cluster_means(v):
    """Delete-one-cluster means of a (K, P) array (balanced)."""
    K = v.shape[0]
    tot = v.sum(); cs = v.sum(1)
    return (tot - cs) / ((K - 1) * v.shape[1])


def bca(stat, arrays, idx, alpha=ALPHA, theta0=None):
    """BCa interval for stat(*means) where means are the means of the (K, P) arrays.
    stat must accept numpy arrays (vectorised). Returns value, ci, z0, acc, p (two-sided, for theta0 if given)."""
    pt = float(stat(*[a.mean() for a in arrays]))
    boot = np.asarray(stat(*[bmeans(a, idx) for a in arrays]), float)
    boot = boot[np.isfinite(boot)]
    Bn = boot.size
    z0 = sps.norm.ppf((np.sum(boot < pt) + 0.5 * np.sum(boot == pt)) / Bn)
    jk = np.asarray(stat(*[_jack_cluster_means(a) for a in arrays]), float)
    dj = jk.mean() - jk
    den = 6.0 * (np.sum(dj ** 2)) ** 1.5
    acc = float(np.sum(dj ** 3) / den) if den > 0 else 0.0

    def adj(z):
        return sps.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))

    a1, a2 = adj(sps.norm.ppf(alpha / 2)), adj(sps.norm.ppf(1 - alpha / 2))
    lo, hi = np.quantile(boot, [a1, a2])
    out = dict(value=pt, ci=[float(lo), float(hi)], z0=float(z0), acc=acc, B=int(Bn), alpha_adj=[float(a1), float(a2)])
    if theta0 is not None:
        out["p"] = bca_p(boot, z0, acc, theta0)
        out["theta0"] = theta0
    return out


def bca_p(boot, z0, acc, theta0):
    """Two-sided p from inverting the BCa interval: the level at which theta0 is an endpoint."""
    Bn = boot.size
    q0 = (np.sum(boot < theta0) + 0.5 * np.sum(boot == theta0)) / Bn
    q0 = min(max(q0, 0.5 / Bn), 1 - 0.5 / Bn)
    zq = sps.norm.ppf(q0)
    w = (zq - z0) / (1 + acc * (zq - z0))
    a1 = sps.norm.cdf(w - z0)
    return float(min(1.0, 2 * min(a1, 1 - a1)))


def fieller(num, den, g, alpha=ALPHA):
    """Fieller interval for mean(num) / mean(den) with CR1 cluster-robust covariance and t(G - 1) critical value.
    Returns value, ci (None if unbounded), bounded flag, t of the denominator."""
    num = np.asarray(num, float).ravel(); den = np.asarray(den, float).ravel(); g = np.asarray(g, int).ravel()
    G = int(g.max()) + 1; N = num.size
    a, b = num.mean(), den.mean()
    Sa = np.bincount(g, weights=num - a, minlength=G); Sb = np.bincount(g, weights=den - b, minlength=G)
    c = G / (G - 1) / N ** 2
    Vaa, Vbb, Vab = c * (Sa ** 2).sum(), c * (Sb ** 2).sum(), c * (Sa * Sb).sum()
    tc = sps.t.ppf(1 - alpha / 2, G - 1)
    tb = b / math.sqrt(Vbb) if Vbb > 0 else math.inf
    A = b * b - tc * tc * Vbb; Bq = -2 * (a * b - tc * tc * Vab); Cq = a * a - tc * tc * Vaa
    disc = Bq * Bq - 4 * A * Cq
    if A <= 0 or disc < 0:
        return dict(value=a / b, ci=None, bounded=False, t_den=tb)
    r1, r2 = (-Bq - math.sqrt(disc)) / (2 * A), (-Bq + math.sqrt(disc)) / (2 * A)
    return dict(value=a / b, ci=[min(r1, r2), max(r1, r2)], bounded=True, t_den=tb)


def holm(pvals, alpha=ALPHA):
    """Holm step-down: adjusted p-values and reject flags, in the input order."""
    p = np.asarray(pvals, float); m = p.size
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj, adj < alpha


def t_den(den_flat, g):
    """|t| of a denominator mean (CR1 se) for the stability screen."""
    se = cluster_se(den_flat, g)
    return abs(float(np.mean(den_flat))) / se if se > 0 else math.inf


def directional(res, sign):
    """A WCR result supports the predicted direction (sign +1 or -1) at level ALPHA."""
    return bool(res["p"] < ALPHA and np.sign(res["mean"]) == sign)


def iut(results, sign=+1):
    """Intersection-union test: every component supports the predicted direction."""
    return all(directional(r, sign) for r in results)


# ====================================================================== design
class Design:
    """Clusters = (shape, sr, pol) sorted by str, positions 0..4 within; optional shape filter."""

    def __init__(self, configs_meta, shape=None):
        cells = {}
        for c, (sh, sr, pol, pos) in configs_meta.items():
            if shape is None or sh == shape:
                cells.setdefault((sh, sr, pol), []).append((pos, c))
        self.keys = sorted(cells, key=str)
        self.cells = [[c for _, c in sorted(cells[k])] for k in self.keys]
        self.K, self.P = len(self.cells), len(self.cells[0])
        assert all(len(c) == self.P for c in self.cells)
        self.configs = [c for cell in self.cells for c in cell]
        self.g = np.repeat(np.arange(self.K), self.P)
        self.sr = np.array([k[1] for k in self.keys], float)
        self.idx = two_stage_idx(self.K, self.P)

    def arr(self, f):
        """(K, P) array from a function config -> value."""
        return np.array([[f(c) for c in cell] for cell in self.cells], float)


def configs_meta():
    items = json.load(open(os.path.join(ROOT, "v2/stim_v3/items.json")))
    return {it["config"]: (it["shape"], it["sr"], it["pol"], it["pos"]) for it in items
            if it["illusion"] == "kanizsa" and it["cond"] == "illusory"}


# ====================================================================== score loaders (read only)
MEAS = "r_inside"


def _load(p):
    return json.load(open(p))


def survey_scores(tag):
    """DeepGaze IIE / MSDB (and the 9 primary sweep blocks): {name: {measure: value}} from v3/scores raw_v3maps (stim_v3)
    and raw (stim_alt); alt<theta> = mean of phases a and b."""
    d = dict(_load(os.path.join(ROOT, "v3/scores/raw_v3maps", tag + ".json")))
    d.update(_load(os.path.join(ROOT, "v3/scores/raw", tag + ".json")))
    return d


def null_scores():
    return _load(os.path.join(ROOT, "v3/scores/raw_nulls.json"))


def head_scores(score_dir, tag, seeds=(0, 1, 2), sets=("stim_v3", "stim_alt"), sizes=("336", "448"), meas=MEAS):
    """Readout heads (swap, registers, G3, A14): files <set>_<tag>_s<seed>.json = {size: {measure: {name: value}}}.
    Returns {name: value} averaged over seeds, then over the two sizes (size average, as published)."""
    acc = {}
    for st in sets:
        for s in seeds:
            p = os.path.join(score_dir, f"{st}_{tag}_s{s}.json")
            d = _load(p)
            for z in sizes:
                for k, v in d[z][meas].items():
                    acc.setdefault(k, []).append(v)
    n = len(seeds) * len(sizes)
    out = {}
    for k, v in acc.items():
        assert len(v) == n, (k, len(v), n)
        out[k] = float(np.mean(v))
    return out


def getter(scores, meas=MEAS):
    """value(config, cond) for either score layout ({name: {measure: v}} or {name: v}); alt<theta> = phase mean."""
    def val(c, cond):
        if cond.startswith("alt") and cond[-1] not in "ab":
            return 0.5 * (val(c, cond + "a") + val(c, cond + "b"))
        x = scores[f"{c}_{cond}"]
        return x[meas] if isinstance(x, dict) else x
    return val


# ====================================================================== primary contrasts
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
SWAP_FSETS = ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")
ARCHS = ("R1n", "R1", "R2", "R3", "R4")
P4_LIST = [  # fixed in REANALYSIS_RULE.md Section 5.4 from the known percentile intervals (|t| >= 4 for both
    # denominators); the two SigLIP-L block 24 R3 contrasts are excluded (denominator |t| about 3.0)
    (f, a, m) for f, a in (("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
                           ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"),
                           ("iie_shapenetc", "R2"), ("iie_shapenetc", "R3")) for m in ("U20", "A20")]
P3_CELLS = [("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
            ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"), ("siglip_L24", "R3"),
            ("iie_shapenetc", "R2"), ("iie_shapenetc", "R3")]


def per_config(D, val, test, ctrl=None):
    """(K, P) array of val(test) - val(ctrl) (or val(test) if ctrl is None)."""
    if ctrl is None:
        return D.arr(lambda c: val(c, test))
    return D.arr(lambda c: val(c, test) - val(c, ctrl))


def p1(D, models, nulls):
    """P1: L = r(illusory) - r(mis20), model minus each null; IUT over the 15 nulls, per model."""
    out = {}
    Ln = {k: per_config(D, v, "illusory", "mis20") for k, v in nulls.items()}
    for m, v in models.items():
        Lm = per_config(D, v, "illusory", "mis20")
        comp = {k: wild_cluster((Lm - Ln[k]).ravel(), D.g) for k in nulls}
        worst = max(nulls, key=lambda k: float(Ln[k].mean()))
        out[m] = dict(L_model=wild_cluster(Lm.ravel(), D.g), vs_null=comp, met=iut(comp.values(), +1),
                      null_with_largest_loss=worst, n_separated=sum(directional(r, +1) for r in comp.values()))
    return out


def p2(Dsq, models, nulls):
    """P2: squares only (6 clusters, full enumeration), Delta = r(alt20) - r(mis20), model minus each null; IUT."""
    out = {}
    Dn = {k: per_config(Dsq, v, "alt20", "mis20") for k, v in nulls.items()}
    for m, v in models.items():
        Dm = per_config(Dsq, v, "alt20", "mis20")
        comp = {k: wild_cluster((Dm - Dn[k]).ravel(), Dsq.g) for k in nulls}
        out[m] = dict(Delta_model=wild_cluster(Dm.ravel(), Dsq.g), vs_null=comp, met=iut(comp.values(), +1),
                      n_separated=sum(directional(r, +1) for r in comp.values()))
    return out


def opposite_interval(ci, ref_sign, ref_defined):
    """Condition (d) of rule Section 5.6: a valid cell has "an interval of the opposite sign" if its 95% WCU interval
    excludes 0 on the other side of R4's sign; if R4's reference sign is undefined (R4's WCR p >= .05), any valid cell
    whose interval excludes 0 counts, because that readout then sets a sign that R4 does not."""
    if not ref_defined:
        return bool(ci[0] > 0 or ci[1] < 0)
    return bool((ref_sign > 0 and ci[1] < 0) or (ref_sign < 0 and ci[0] > 0))


def p3(D, cells, ref):
    """P3: E = r(illusory) - r(control180) for each valid non-reference cell; sign of R4 on the same features;
    Holm over the family. cells: {(fset, arch): val}; ref: {fset: val of R4}. "Keeps the sign": Holm-adjusted WCR
    p < .05, point estimate on R4's side, R4's reference defined (R4's own WCR p < .05)."""
    res, keys, refres = {}, list(cells), {}
    for k in keys:
        fs = k[0]
        if fs not in refres:
            refres[fs] = wild_cluster(per_config(D, ref[fs], "illusory", "control180").ravel(), D.g)
        r = wild_cluster(per_config(D, cells[k], "illusory", "control180").ravel(), D.g)
        r["ref_sign"] = int(np.sign(refres[fs]["mean"]))
        r["ref_defined"] = bool(refres[fs]["p"] < ALPHA)
        res[k] = r
    adj, rej = holm([res[k]["p"] for k in keys])
    for k, a, j in zip(keys, adj, rej):
        res[k]["p_holm"] = float(a)
        res[k]["keeps_sign"] = bool(j and res[k]["ref_defined"] and np.sign(res[k]["mean"]) == res[k]["ref_sign"])
        res[k]["opposite_sign_interval"] = opposite_interval(res[k]["ci"], res[k]["ref_sign"], res[k]["ref_defined"])
    n_keep = sum(res[k]["keeps_sign"] for k in keys)
    return dict(cells={f"{k[0]}|{k[1]}": res[k] for k in keys}, n=len(keys), n_keep=n_keep,
                any_opposite=any(res[k]["opposite_sign_interval"] for k in keys),
                reference={fs: dict(mean=r["mean"], ci=r["ci"], p=r["p"]) for fs, r in refres.items()})


def ratio_parts(D, val, meas):
    """Numerator and denominator (K, P) arrays of U20 or A20 (effects against control180)."""
    den = per_config(D, val, "illusory", "control180")
    num = per_config(D, val, "mis20" if meas == "U20" else "alt20", "control180")
    return num, den


def p4(D, heads, contrasts=P4_LIST):
    """P4: paired BCa of ratio differences (readout minus R4), U20 and A20; Holm over the pre-listed family; stability
    screen of both denominators (|t| < 4 flags the contrast; flagged contrasts stay in the Holm family but are not
    counted)."""
    res = {}
    for fs, ar, meas in contrasts:
        n1, d1 = ratio_parts(D, heads[(fs, ar)], meas)
        n2, d2 = ratio_parts(D, heads[(fs, "R4")], meas)
        b = bca(lambda a, b_, c, d: a / b_ - c / d, [n1, d1, n2, d2], D.idx, theta0=0.0)
        tden = min(t_den(d1.ravel(), D.g), t_den(d2.ravel(), D.g))
        w1, w2 = wild_cluster(d1.ravel(), D.g), wild_cluster(d2.ravel(), D.g)
        reportable = bool(all(w["ci"][0] > 0 or w["ci"][1] < 0 for w in (w1, w2)))
        res[f"{fs}|{ar}|{meas}"] = dict(diff=b["value"], ci=b["ci"], p=b["p"], z0=b["z0"], acc=b["acc"],
                                        ratio_readout=n1.mean() / d1.mean(), ratio_R4=n2.mean() / d2.mean(),
                                        t_den_min=tden, stable=bool(tden >= T_STABLE),
                                        denominators_exclude_zero=reportable,
                                        fieller_readout=fieller(n1, d1, D.g), fieller_R4=fieller(n2, d2, D.g),
                                        ratio_readout_bca=bca(lambda a, b_: a / b_, [n1, d1], D.idx),
                                        ratio_R4_bca=bca(lambda a, b_: a / b_, [n2, d2], D.idx))
    keys = list(res)
    adj, rej = holm([res[k]["p"] for k in keys])
    for k, a, j in zip(keys, adj, rej):
        res[k]["p_holm"] = float(a); res[k]["survives"] = bool(j)
        res[k]["counted"] = bool(j and res[k]["stable"] and res[k]["denominators_exclude_zero"])
    return dict(contrasts=res, family=len(keys), n_counted=sum(res[k]["counted"] for k in keys))


def p5_parts(D, val, theta=20):
    """Per-configuration P5 quantities (rule Section 5.5), every term a contrast against a control with the same edge
    orientation: E(0) = I - C; E_rig = RI - RC (whole display rotated by theta); E'_mis = M - MC (stim_v3 uniform
    rotation and its orientation-matched control mis<theta>_control180).
      L_uni = E(0) - E'_mis     loss under uniform rotation, in the contrast currency
      L_rig = E(0) - E_rig      loss from turning edges and figure together, alignment kept
      A     = L_uni - L_rig = E_rig - E'_mis   alignment effect at matched edge orientation theta
    Secondary (never decision-bearing): the drafted alignment loss (RI - M) - (RC - C), which keeps the control's
    orientation response; the control-free RI - M; the control's change under rigid rotation RC - C."""
    E_ill = per_config(D, val, "illusory", "control180")
    E_rig = per_config(D, val, f"rig{theta}_illusory", f"rig{theta}_control180")
    E_mis = per_config(D, val, f"mis{theta}", f"mis{theta}_control180")
    L_uni = E_ill - E_mis
    L_rig = E_ill - E_rig
    sec = dict(A_drafted=per_config(D, val, "illusory", f"mis{theta}") - L_rig,
               A_control_free=per_config(D, val, f"rig{theta}_illusory", f"mis{theta}"),
               control_rigid_change=per_config(D, val, f"rig{theta}_control180", "control180"))
    return L_uni, L_rig, E_rig - E_mis, sec


def p5_category(iut_met, half_met, L_uni_mean, share):
    """Rule Section 5.5: 1 "survives" ((i) and (ii)); 2a "shrinks" ((i), not (ii), L_uni > 0 and s* >= 0.5);
    2b "half criterion undecided" ((i), not (ii), otherwise); 3 "not separable" (not (i))."""
    if not iut_met:
        return "3"
    if half_met:
        return "1"
    return "2a" if (L_uni_mean > 0 and share >= 0.5) else "2b"


def p5(D, models, nulls, theta=20):
    """P5: alignment effect at matched edge orientation (A*), model minus each null (IUT), the half criterion
    C* = L_uni - 2 L_rig, the rigid share s* = L_rig / L_uni (BCa, stability flag and denominator check on L_uni) and
    the outcome category per model. A model whose scores are None (pipeline check failed) is "not computable" and is
    treated as not in category 1."""
    out = {}
    Dn = {k: p5_parts(D, v, theta)[2] for k, v in nulls.items()}
    for m, v in models.items():
        if v is None:
            out[m] = dict(category=None, computable=False, note="not computable: pipeline check failed or scores missing")
            continue
        L_uni, L_rig, Aal, sec = p5_parts(D, v, theta)
        comp = {k: wild_cluster((Aal - Dn[k]).ravel(), D.g) for k in nulls}
        C = wild_cluster((L_uni - 2 * L_rig).ravel(), D.g)
        share = bca(lambda a, b: a / b, [L_rig, L_uni], D.idx)
        tden = t_den(L_uni.ravel(), D.g)
        wl = wild_cluster(L_uni.ravel(), D.g)
        met = iut(comp.values(), +1)
        half_ok = bool(directional(C, +1))
        cat = p5_category(met, half_ok, float(L_uni.mean()), share["value"])
        out[m] = dict(computable=True, L_uni=wl, L_rig=wild_cluster(L_rig.ravel(), D.g),
                      align_effect=wild_cluster(Aal.ravel(), D.g), vs_null=comp, iut_met=met,
                      n_separated=sum(directional(r, +1) for r in comp.values()), half_criterion=C,
                      rigid_share=dict(share, t_den=tden, stable=bool(tden >= T_STABLE),
                                       denominator_excludes_zero=bool(wl["ci"][0] > 0 or wl["ci"][1] < 0),
                                       fieller=fieller(L_rig, L_uni, D.g)),
                      secondary={k: wild_cluster(x.ravel(), D.g) for k, x in sec.items()}, category=cat)
    out["both_category_1"] = all(out[m]["category"] == "1" for m in models)
    out["abstract_leads_with_alternation"] = not out["both_category_1"]
    return out


def abstract_rotation_rule(p1res, p2res, p5res):
    """Rule Section 5.5 (abstract and highlight 4). U = models with P1 met and P5 category 1; A = models with P2 met.
    Both in U: the uniform-rotation clause stays (both models), followed by the ratio-free alternation clause for the
    models in A. Otherwise the abstract leads with the ratio-free square alternation clause for the models in A, and
    carries no rotation claim if A is empty."""
    ms = [m for m in p1res if isinstance(p1res[m], dict) and "met" in p1res[m] and m in p2res]
    U = [m for m in ms if p1res[m].get("met") and (p5res.get(m) or {}).get("category") == "1"]
    A = [m for m in ms if p2res[m].get("met")]
    short = lambda xs: " and ".join(x.replace("DeepGaze ", "") for x in xs)
    if len(U) == len(ms):
        hl = "Inducer rotation separates DeepGaze IIE and MSDB from 15 local image nulls"
        lead = "uniform rotation (both models), then alternation for " + (short(A) if A else "no model")
    elif A:
        hl = f"On squares, alternating rotation separates DeepGaze {short(A)} from 15 local nulls"
        lead = "square alternation for " + short(A) + "; no uniform-rotation clause"
    else:
        hl, lead = None, "no rotation claim in the abstract"
    return dict(uniform_models=U, alternation_models=A, abstract=lead, highlight4=hl,
                highlight4_chars=len(hl) if hl else 0)


def accuracy_valid(cc_cell, cc_r4_same_features, margin=0.03):
    """Accuracy criterion for every cell, old or new, computed exactly as in the published analysis
    (v3/gpu/analyze_full.py): the mean of the three unrounded seed CCs (held-out MIT1003, 336 px, centre bias on),
    rounded to 4 decimals, at least round(round(R4 mean, 4) - 0.03, 4). Inputs are the unrounded seed means."""
    m = round(float(cc_cell), 4)
    thr = round(round(float(cc_r4_same_features), 4) - margin, 4)
    return bool(m >= thr)


NEW_ARCHS = ("R1n", "R1", "R2", "R3")
MIN_VALID = 2          # condition (c0): smallest number of valid non-R4 readouts on any feature set of the published swap


def p6(D, newsets, cc):
    """P6: for each new feature set: the R4 reference (WCR at .05), then the P3 test on every valid non-R4 cell, Holm
    within the feature set. newsets: {fset: {arch: val}} (an arch is present only if its three score files and its
    three MIT1003 files exist); cc: {fset: {arch: mean CC over seeds 0, 1, 2}}. A missing R4 makes the set "not run";
    a missing cell is "not run" (rule Section 5.6 (a)). Validity uses the held-out CC alone."""
    out = {}
    for fs, heads in newsets.items():
        if "R4" not in heads or "R4" not in cc.get(fs, {}):
            out[fs] = dict(status="not run", R4=None, R4_sign=0, R4_reference_defined=False, complete=False, n_valid=0,
                           cells={ar: dict(status="not run") for ar in NEW_ARCHS})
            continue
        r4 = wild_cluster(per_config(D, heads["R4"], "illusory", "control180").ravel(), D.g)
        ref_ok = bool(r4["p"] < ALPHA)
        sgn = int(np.sign(r4["mean"]))
        cells = {}
        for ar in NEW_ARCHS:
            if ar not in heads or ar not in cc.get(fs, {}):
                cells[ar] = dict(status="not run")
                continue
            valid = accuracy_valid(cc[fs][ar], cc[fs]["R4"])
            r = wild_cluster(per_config(D, heads[ar], "illusory", "control180").ravel(), D.g)
            r.update(valid=valid, status="valid" if valid else "below criterion", cc=cc[fs][ar],
                     threshold=round(round(cc[fs]["R4"], 4) - 0.03, 4))
            r["opposite_sign_interval"] = opposite_interval(r["ci"], sgn, ref_ok)
            cells[ar] = r
        vk = [a for a, r in cells.items() if r.get("status") == "valid"]
        if vk:
            adj, rej = holm([cells[a]["p"] for a in vk])
            for a, pa, j in zip(vk, adj, rej):
                cells[a]["p_holm"] = float(pa)
                cells[a]["keeps_sign"] = bool(ref_ok and j and np.sign(cells[a]["mean"]) == sgn)
        out[fs] = dict(status="run", R4=r4, R4_cc=cc[fs]["R4"], R4_sign=sgn, R4_reference_defined=ref_ok, cells=cells,
                       n_valid=len(vk), n_keep=sum(bool(cells[a].get("keeps_sign")) for a in vk),
                       complete=all(r.get("status") != "not run" for r in cells.values()))
    return out


TITLE_BEARING = ("dinov2reg_L6", "dinov2reg_L12", "dinov2ms_L7L11")
REQUIRED_COMPLETE = TITLE_BEARING + ("dinov2ms_L11",)
T1 = ("Looking at Nothing: The Checkpoint and Layer a Saliency Model Reads, Not Its Readout Architecture, "
      "Set the Sign of Its Kanizsa Contrast")
TS = ("Looking at Nothing: Under Five Readout Architectures, the Checkpoint and Layer a Saliency Model Reads "
      "Set the Sign of Its Kanizsa Contrast")


def title_decision(p6res, p3res):
    """Rule Section 5.6. T1 only if all hold:
      (a)  G3 and A14 completed and were scored for every cell of the three title-bearing feature sets and of
           dinov2ms_L11 (a missing run or cell fails (a));
      (b)  on each title-bearing set R4's reference sign is defined (R4's WCR p < .05);
      (c0) on each title-bearing set at least MIN_VALID = 2 non-R4 cells meet the accuracy criterion;
      (c)  on each title-bearing set every valid non-R4 cell keeps R4's sign (Holm-adjusted WCR p < .05 within the set,
           point estimate on R4's side);
      (c1) P3 is met in full (all 10 valid non-reference swap cells keep R4's sign after Holm);
      (d)  no valid cell on any new feature set (dinov2ms_L11 included) and no valid P3 cell has a 95% WCU interval of
           the opposite sign (opposite_interval(); with R4 undefined, any interval excluding zero counts).
    Otherwise the scoped title; if (d) fails, neither title (claim and title narrowed)."""
    if p3res is None:
        raise ValueError("title_decision needs the P3 result (rule Section 5.6 (c1))")
    reasons, opposite = [], []
    for fs in REQUIRED_COMPLETE:
        r = p6res.get(fs)
        if r is None or not r.get("complete"):
            reasons.append(f"(a) {fs}: not run or incomplete")
    for fs in TITLE_BEARING:
        r = p6res.get(fs)
        if r is None or r.get("status") == "not run":
            continue
        if not r["R4_reference_defined"]:
            reasons.append(f"(b) {fs}: R4 gave no reliable contrast (reference sign undefined)")
        if r["n_valid"] < MIN_VALID:
            reasons.append(f"(c0) {fs}: {r['n_valid']} of 4 non-R4 readouts met the accuracy criterion (need {MIN_VALID})")
        for a, c in r["cells"].items():
            if c.get("status") == "valid" and not c.get("keeps_sign"):
                reasons.append(f"(c) {fs} {a}: valid but does not keep R4's sign after Holm adjustment")
    if p3res["n_keep"] < p3res["n"]:
        reasons.append(f"(c1) P3: {p3res['n_keep']} of {p3res['n']} valid swap cells keep R4's sign after Holm adjustment")
    for fs, r in p6res.items():
        for a, c in (r.get("cells") or {}).items():
            if c.get("status") == "valid" and c.get("opposite_sign_interval"):
                opposite.append(f"(d) {fs} {a}: valid cell with an interval of the opposite sign")
    if p3res.get("any_opposite"):
        opposite.append("(d) P3: a valid swap cell has an interval of the opposite sign")
    reasons += opposite
    if opposite:
        return dict(title=None, stronger=False, narrowed=True, reasons=reasons,
                    note="neither T1 nor the scoped title: narrow the claim and title to the feature sets where no "
                         "valid readout has an opposite-signed interval")
    return dict(title=T1 if not reasons else TS, stronger=not reasons, narrowed=False, reasons=reasons)


# ====================================================================== wording (rule Section 7), filled from results
def f3(x):
    return f"{x:+.3f}"


def ci3(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def wording_p1(res):
    met = [m for m in res if res[m]["met"]]
    if len(met) == 2:
        parts = [f"{m.replace('DeepGaze ', '')} {f3(res[m]['vs_null'][res[m]['null_with_largest_loss']]['mean'])} "
                 f"{ci3(res[m]['vs_null'][res[m]['null_with_largest_loss']]['ci'])}" for m in res]
        return ("Unlike all 15 local nulls, both DeepGaze models lost figure mass under a uniform 20 degree inducer "
                f"rotation (differences against the null with the largest loss: {', '.join(parts)}), a manipulation "
                "that also turns the notch edges oblique.")
    if len(met) == 1:
        m = met[0]; o = [x for x in res if x != m][0]
        return (f"Unlike all 15 local nulls, {m} lost figure mass under a uniform 20 degree inducer rotation, a "
                "manipulation that also turns the notch edges oblique; under uniform rotation, "
                f"{o} was not separable from {15 - res[o]['n_separated']} of the 15 local nulls.")
    return " ".join(f"Under uniform rotation, {m} was not separable from {15 - res[m]['n_separated']} of the 15 "
                    "local nulls." for m in res)


def wording_p2(res):
    """Rule Section 7.1, P2 (ratio-free; stats review finding 2)."""
    met = [m for m in res if isinstance(res[m], dict) and res[m].get("met")]
    if met:
        parts = [f"{f3(res[m]['Delta_model']['mean'])} {ci3(res[m]['Delta_model']['ci'])} for {m.replace('DeepGaze ', '')}"
                 for m in met]
        s = ("On squares, alternating the rotation direction, which keeps the axis mirror symmetries, kept more mass "
             f"inside the figure than uniform rotation (difference {' and '.join(parts)}), more than for every local "
             "null, for which the two rotations left the same figure mass (differences 0.0000 to +0.0003).")
    else:
        s = ""
    for m in res:
        if isinstance(res[m], dict) and not res[m].get("met"):
            s += (f" On six square clusters the alternation test did not separate {m} from "
                  f"{15 - res[m]['n_separated']} of the 15 local nulls.")
    return s.strip()


def wording_p3(res):
    if res["any_opposite"]:
        return "TITLE AND CLAIM NARROWED: at least one valid readout has an interval of the opposite sign (Section 7.3)."
    if res["n_keep"] == res["n"]:
        return ("No readout that met the accuracy criterion changed the sign set by the checkpoint and layer read "
                f"({res['n']} of {res['n']} non-reference cells after Holm adjustment; 30 of 30 seed point estimates "
                "had that sign).")
    return (f"{res['n_keep']} of {res['n']} kept the sign with intervals excluding zero after Holm adjustment and none "
            "had an interval of the opposite sign.")


# ====================================================================== public-dependency guard and outputs
def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def require_public(re_dir=HERE):
    """Refuse to compute any new statistic unless (rule Section 8.1): REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt exist;
    PUBLIC_PUSH.txt starts with the public commit hash; every file hashed in REANALYSIS_RULE.sha256 is unchanged; and
    every stimulus file matches its md5 in stim_rigid/manifest.json (same logic as scoring/common_rigid.py)."""
    rec, pub = os.path.join(re_dir, "REANALYSIS_RULE.sha256"), os.path.join(re_dir, "PUBLIC_PUSH.txt")
    if not (os.path.exists(rec) and os.path.exists(pub)):
        sys.exit("refusing: REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt must exist (rule and code public first)")
    tok = open(pub).read().split()
    if not tok or len(tok[0]) < 7 or any(c not in "0123456789abcdef" for c in tok[0]):
        sys.exit("refusing: PUBLIC_PUSH.txt must start with the public commit hash")
    n = 0
    for line in open(rec):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            fp = os.path.join(re_dir, parts[1])
            if not os.path.exists(fp) or sha256(fp) != parts[0]:
                sys.exit(f"refusing: {parts[1]} is missing or differs from the hash recorded in REANALYSIS_RULE.sha256")
            n += 1
    if n == 0:
        sys.exit("refusing: REANALYSIS_RULE.sha256 lists no file")
    man = json.load(open(os.path.join(re_dir, "stim_rigid", "manifest.json")))
    for f, m in man["files"].items():
        fp = os.path.join(re_dir, "stim_rigid", f)
        if not os.path.exists(fp) or hashlib.md5(open(fp, "rb").read()).hexdigest() != m:
            sys.exit(f"refusing: stimulus file {f} differs from stim_rigid/manifest.json")
    return n


def jdump(obj, name):
    os.makedirs(OUT, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    p = os.path.join(OUT, name)
    json.dump(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), result=obj), open(p, "w"),
              indent=1, default=conv)
    print("wrote", p)


# ====================================================================== steps
SWAP_SCORES = os.path.join(ROOT, "v3/gpu/full/scores")


def load_swap():
    return {(f, a): getter(head_scores(SWAP_SCORES, f"{f}_{a}")) for f in SWAP_FSETS for a in ARCHS}


def load_survey_and_nulls():
    models = {"DeepGaze IIE": getter(survey_scores("DeepGaze_IIE")), "DeepGaze MSDB": getter(survey_scores("DeepGaze_MSDB"))}
    N = null_scores()
    nulls = {k: getter(N[k]) for k in NULLS}
    return models, nulls


def step_sanity():
    """Gate (rule Section 3.1): the copied loaders and the v2 percentile bootstrap reproduce published values to printed
    precision. Computes only published statistics (percentile intervals, v2 indices)."""
    meta = configs_meta(); D = Design(meta); Dsq = Design(meta, "square")
    V3R = _load(os.path.join(ROOT, "v3/scores/results.json"))["models"]
    FULL = _load(os.path.join(ROOT, "v3/gpu/full/results_full.json"))
    models, nulls = load_survey_and_nulls()
    fails, checks = [], 0

    def close(a, b, tol=6e-5):
        return abs(a - b) <= tol

    for name, v in list(models.items()) + [("null " + k, v) for k, v in nulls.items()]:
        ref = V3R[name]["results"]["r_inside"]
        s = pct_summary(per_config(D, v, "illusory", "mis20"), D.idx)
        r = ref["all"]["drop_uni"]["20"]; checks += 1
        if not (close(s["mean"], r["mean"]) and close(s["ci"][0], r["ci"][0]) and close(s["ci"][1], r["ci"][1])):
            fails.append((name, "L20", s, r))
        s = pct_summary(per_config(Dsq, v, "alt20", "mis20"), Dsq.idx)
        r = ref["square"]["alt_minus_mis"]["20"]; checks += 1
        if not (close(s["mean"], r["mean"]) and close(s["ci"][0], r["ci"][0]) and close(s["ci"][1], r["ci"][1])):
            fails.append((name, "sq Delta20", s, r))
    heads = load_swap()
    for (f, a), v in heads.items():
        s = pct_summary(per_config(D, v, "illusory", "control180"), D.idx)
        r = FULL["stim_v3"][f][a]["kanizsa"]["illusory-control180"]["avg|r_inside"]; checks += 1
        if not (close(s["mean"], r["mean"]) and close(s["ci"][0], r["ci"][0]) and close(s["ci"][1], r["ci"][1])):
            fails.append((f, a, "E", s, r))
        for meas, key in (("U20", "ratio_uni20"), ("A20", "ratio_alt20")):
            num, den = ratio_parts(D, v, meas)
            s = pct_ratio(num, den, D.idx); r = FULL["stim_alt"][f][a]["r_inside"][key]; checks += 1
            if not (close(s["value"], r["value"]) and close(s["ci"][0], r["ci"][0]) and close(s["ci"][1], r["ci"][1])):
                fails.append((f, a, meas, s, r))
    res = dict(checks=checks, failed=len(fails), failures=fails[:20], passed=not fails)
    print(f"sanity gate: {checks} checks, {len(fails)} failed -> {'PASS' if not fails else 'FAIL'}")
    return res


def step_primary(which):
    meta = configs_meta(); D = Design(meta); Dsq = Design(meta, "square")
    if "P5" in which:                                          # the abstract rule of Section 5.5 needs P1 and P2
        which = set(which) | {"P1", "P2"}
    res = {}
    if {"P1", "P2"} & which:
        models, nulls = load_survey_and_nulls()
        if "P1" in which:
            res["P1"] = p1(D, models, nulls); res["P1"]["wording"] = wording_p1(res["P1"])
        if "P2" in which:
            res["P2"] = p2(Dsq, models, nulls); res["P2"]["wording"] = wording_p2(res["P2"])
    if {"P3", "P4"} & which:
        heads = load_swap()
        if "P3" in which:
            res["P3"] = p3(D, {k: heads[k] for k in P3_CELLS}, {f: heads[(f, "R4")] for f in SWAP_FSETS})
            res["P3"]["wording"] = wording_p3(res["P3"])
        if "P4" in which:
            res["P4"] = p4(D, heads)
    if "P5" in which:
        rig = os.path.join(HERE, "scores_rigid")
        models = {}
        for name in ("DeepGaze IIE", "DeepGaze MSDB"):
            tag = name.replace(" ", "_")
            fp = os.path.join(rig, tag + ".json")
            # a model whose rigid scores are missing (for example because its pipeline check failed) is not computable
            models[name] = getter({**survey_scores(tag), **_load(fp)}) if os.path.exists(fp) else None
        N, NR = null_scores(), _load(os.path.join(rig, "nulls.json"))
        nulls = {k: getter({**N[k], **NR[k]}) for k in NULLS}
        res["P5"] = p5(D, models, nulls)
        if "P1" in res and "P2" in res:
            res["abstract_rotation"] = abstract_rotation_rule(res["P1"], res["P2"], res["P5"])
    if "P6" in which:
        if "P3" not in res:                                    # the title rule needs P3 (condition (c1) and (d))
            heads = load_swap()
            res["P3"] = p3(D, {k: heads[k] for k in P3_CELLS}, {f: heads[(f, "R4")] for f in SWAP_FSETS})
            res["P3"]["wording"] = wording_p3(res["P3"])
        newsets, cc = {}, {}
        for fs, d in (("dinov2reg_L6", "gpu/g3"), ("dinov2reg_L12", "gpu/g3"), ("dinov2ms_L7L11", "gpu/a14"),
                      ("dinov2ms_L11", "gpu/a14")):
            sd = os.path.join(HERE, d, "scores")
            heads, ccs = {}, {}
            for ar in ARCHS:
                src = sd
                if fs.startswith("dinov2reg") and ar == "R4":           # registers R4 exists already (17 Nov)
                    src = os.path.join(ROOT, "nn/registers/full/scores")
                need = [os.path.join(src, f"stim_v3_{fs}_{ar}_s{s}.json") for s in (0, 1, 2)] + \
                       [os.path.join(src, f"mit_{fs}_{ar}_s{s}.json") for s in (0, 1, 2)]
                if not all(os.path.exists(x) for x in need):           # a cell counts only with all six files
                    continue
                heads[ar] = getter(head_scores(src, f"{fs}_{ar}", sets=("stim_v3",)))
                ccs[ar] = float(np.mean([_load(os.path.join(src, f"mit_{fs}_{ar}_s{s}.json"))["CC"] for s in (0, 1, 2)]))
            newsets[fs], cc[fs] = heads, ccs
        res["P6"] = p6(D, newsets, cc)
        res["title"] = title_decision(res["P6"], res["P3"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sanity", "primary"])
    ap.add_argument("--which", default="P1,P2,P3,P4")
    a = ap.parse_args()
    require_public()                                   # rule Section 8: nothing runs on real scores before the push
    if a.step == "sanity":
        jdump(step_sanity(), "sanity.json")
        return
    which = set(a.which.split(","))
    jdump(step_primary(which), "primary_" + "_".join(sorted(which)) + ".json")


if __name__ == "__main__":
    main()
