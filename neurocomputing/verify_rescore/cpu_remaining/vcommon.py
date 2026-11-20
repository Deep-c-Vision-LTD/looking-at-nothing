"""
vcommon.py: shared helpers of the independent verification of S3, S4, S5 and the test-time registers (19 November 2026).

Written from REANALYSIS_RULE.md Sections 2 and 4 only; imports no project module (analyze_re is imported by each script
only for require_public(), rule Section 8.1). numpy only. Nothing here writes a file.
  - design: 60 Kanizsa configurations of v2/stim_v3/items.json, 12 clusters (shape, sr, pol) sorted by str(key),
    positions sorted (rule 2.1); sub-designs for item 18 (4 clusters = shape x polarity within one support ratio).
  - wild cluster bootstrap-t (rule 4.1): Webb six-point weights; CR1 se; WCR p with the null imposed (12 clusters: 9,999
    random vectors from RandomState(2026) drawn as WEBB[randint(0, 6, (B, G))]; <= 6 clusters: all 6^G vectors);
    WCU percentile-t interval with unrestricted residuals.
  - v2 two-stage percentile bootstrap (published method, rule 4.5): clusters with replacement then positions within
    the drawn cluster, RandomState(2026), 10,000 resamples, 2.5 and 97.5 percentiles.
"""
import itertools, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(os.path.dirname(HERE))         # re-analysis folder (neurocomputing/ in the repository)
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root (as analyze_re.py)
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
SEED = 2026


def guard():
    import sys
    if RE not in sys.path:
        sys.path.insert(0, RE)
    import analyze_re
    return analyze_re.require_public(RE)


def jl(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ design
def design(sr=None):
    """grid [K][P] of configuration names; keys (shape, sr, pol) or (shape, pol) for a support-ratio sub-design."""
    items = jl(os.path.join(ROOT, "v2", "stim_v3", "items.json"))
    cl = {}
    for it in items:
        if it["illusion"] != "kanizsa" or it["cond"] != "illusory":
            continue
        if sr is not None and abs(it["sr"] - sr) > 1e-9:
            continue
        key = (it["shape"], it["sr"], it["pol"]) if sr is None else (it["shape"], it["pol"])
        cl.setdefault(key, []).append((it["pos"], it["config"]))
    keys = sorted(cl, key=str)
    grid = [[c for _, c in sorted(cl[k])] for k in keys]
    assert len({len(g) for g in grid}) == 1
    return keys, grid


def per_config(grid, f):
    return np.array([[f(c) for c in row] for row in grid], float)


# ------------------------------------------------------------------ wild cluster bootstrap (rule 4.1)
def cr1_se(x):
    G, P = x.shape[-2], x.shape[-1]
    res = x - x.mean(axis=(-2, -1), keepdims=True)
    return np.sqrt(G / (G - 1) * (res.sum(-1) ** 2).sum(-1)) / (G * P)


_W = {}


def weights(G, B=9999):
    if (G, B) not in _W:
        if G <= 6:
            idx = np.array(list(itertools.product(range(6), repeat=G)))
            _W[(G, B)] = (WEBB[idx], True)
        else:
            rs = np.random.RandomState(SEED)
            _W[(G, B)] = (WEBB[rs.randint(0, 6, size=(B, G))], False)
    return _W[(G, B)]


def wild(d, mu0=0.0, chunk=None):
    """d: (G, P) per-configuration values (equal cluster sizes). Returns mean, se, t, WCR p, WCU ci."""
    d = np.asarray(d, float)
    G, P = d.shape
    W, enum = weights(G)
    m = d.mean(); se = cr1_se(d); t = (m - mu0) / se
    tol = 1e-10 * max(1.0, abs(t))
    ts, tu = [], []
    step = chunk or W.shape[0]
    for a in range(0, W.shape[0], step):
        w = W[a:a + step]
        ds = mu0 + w[:, :, None] * (d - mu0)[None]
        ts.append((ds.mean((1, 2)) - mu0) / cr1_se(ds))
        du = m + w[:, :, None] * (d - m)[None]
        tu.append((du.mean((1, 2)) - m) / cr1_se(du))
    ts, tu = np.concatenate(ts), np.concatenate(tu)
    ge, le = int(np.sum(ts >= t - tol)), int(np.sum(ts <= t + tol))
    p = min(1.0, 2 * min(ge, le) / len(ts)) if enum else min(1.0, 2 * (1 + min(ge, le)) / (len(ts) + 1))
    q_lo, q_hi = np.quantile(tu, [0.025, 0.975])
    return dict(mean=float(m), se=float(se), t=float(t), p=float(p), ci=[float(m - q_hi * se), float(m - q_lo * se)],
                G=G, enumerated=bool(enum), n_vectors=int(len(ts)))


def wild_iid(v):
    """every observation its own cluster (G = N), as A6 (S4) does; 9,999 Webb draws from RandomState(2026)."""
    v = np.asarray(v, float)
    return wild(v.reshape(-1, 1), chunk=500)


# ------------------------------------------------------------------ v2 two-stage percentile bootstrap (rule 4.5)
_I = {}


def two_stage(K, P, B=10000):
    if (K, P) not in _I:
        rs = np.random.RandomState(SEED)
        _I[(K, P)] = (rs.randint(0, K, (B, K)), rs.randint(0, P, (B, K, P)))
    return _I[(K, P)]


def pct_ci(d):
    d = np.asarray(d, float)
    ci, pi = two_stage(*d.shape)
    bm = d[ci[:, :, None], pi].mean((1, 2))
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------ comparison helper
def cmp(mine, stored):
    """max |difference| between two nested structures of numbers (same keys / lengths)."""
    if isinstance(mine, dict):
        return max([cmp(mine[k], stored[k]) for k in mine] or [0.0])
    if isinstance(mine, (list, tuple)):
        return max([cmp(a, b) for a, b in zip(mine, stored)] or [0.0])
    return abs(float(mine) - float(stored))
