"""
Independent recomputation of P5 (theta = 20), the P5 categories, the abstract / highlight-4 rule and the
"P1 leaves the contributions" clause of neuro/reanalysis/REANALYSIS_RULE.md (Sections 4, 5.1, 5.2, 5.5, 7.1, 7.2).

Written 18 November 2026 for the P5 verification step, as an extension of verify_rescore/recompute_stats.py (whose
design, loaders, wild cluster bootstrap and pairs cluster BCa are copied here unchanged). It does NOT import
analyze_re.py or any project module. The guard require_public() below is an independent implementation of rule
Section 8.1 (both files exist, PUBLIC_PUSH.txt starts with a commit hash, every file hashed in REANALYSIS_RULE.sha256
unchanged, every stim_rigid file matches its md5 in stim_rigid/manifest.json) and runs before any score is read.

Inputs (raw per-image score files, read only):
  v2/stim_v3/items.json (design; clusters sorted by str of (shape, sr, pol), positions within)
  v3/scores/raw_v3maps/DeepGaze_{IIE,MSDB}.json (stim_v3: I, C, M), v3/scores/raw/DeepGaze_{IIE,MSDB}.json (stim_alt)
  v3/scores/raw_nulls.json (15 nulls, stim_v3 and stim_alt)
  neuro/reanalysis/scores_rigid/DeepGaze_{IIE,MSDB}.json, scores_rigid/nulls.json (RI, RC, MC)
Measure r_inside throughout. Output: neuro/verify_rescore/recompute_p5.json (nothing else is written).
Run: nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -B -I recompute_p5.py
"""
import hashlib, itertools, json, math, os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                          # re-analysis folder (neurocomputing/ in the repository)
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root (as analyze_re.py)
OUT = os.path.join(HERE, "recompute_p5.json")
MSDB_MD5 = "51db3832b635ebc15523d3e3e8a9bc14"       # stated by the task for scores_rigid/DeepGaze_MSDB.json

SEED, B_WILD, B_PAIRS = 2026, 9999, 10000
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
MODELS = ("DeepGaze IIE", "DeepGaze MSDB")
THETA = 20


# ------------------------------------------------------------------ guard (rule Section 8.1), independent copy
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def require_public():
    rec, pub = os.path.join(RE, "REANALYSIS_RULE.sha256"), os.path.join(RE, "PUBLIC_PUSH.txt")
    if not (os.path.exists(rec) and os.path.exists(pub)):
        sys.exit("refusing: REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt must exist")
    tok = open(pub).read().split()
    if not tok or len(tok[0]) < 7 or any(c not in "0123456789abcdef" for c in tok[0]):
        sys.exit("refusing: PUBLIC_PUSH.txt must start with the public commit hash")
    n = 0
    for line in open(rec):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64 and all(c in "0123456789abcdef" for c in p[0]):
            fp = os.path.join(RE, p[1])
            if not os.path.exists(fp) or hashlib.sha256(open(fp, "rb").read()).hexdigest() != p[0]:
                sys.exit(f"refusing: hashed file changed or missing: {p[1]}")
            n += 1
    if n == 0:
        sys.exit("refusing: no hashed file listed")
    man = json.load(open(os.path.join(RE, "stim_rigid", "manifest.json")))
    m = 0
    for f, h in man["files"].items():
        fp = os.path.join(RE, "stim_rigid", f)
        if not os.path.exists(fp) or md5(fp) != h:
            sys.exit(f"refusing: stimulus file {f} differs from stim_rigid/manifest.json")
        m += 1
    return dict(hashed_files_verified=n, stimulus_files_verified=m, public_commit=tok[0])


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


def merge_disjoint(*ds):
    out = {}
    for d in ds:
        assert not set(out) & set(d), "overlapping display names between score files"
        out.update(d)
    return out


def model_scores(tag):
    v3 = jl(os.path.join(ROOT, "v3/scores/raw_v3maps", tag + ".json"))
    alt = jl(os.path.join(ROOT, "v3/scores/raw", tag + ".json"))
    rig = jl(os.path.join(RE, "scores_rigid", tag + ".json"))
    assert len(rig) == 660, (tag, len(rig))
    d = merge_disjoint(v3, alt, rig)
    return {k: v["r_inside"] for k, v in d.items()}


def null_scores():
    N = jl(os.path.join(ROOT, "v3/scores/raw_nulls.json"))
    NR = jl(os.path.join(RE, "scores_rigid", "nulls.json"))
    out = {}
    for k in NULLS:
        assert len(NR[k]) == 660, (k, len(NR[k]))
        out[k] = {n: v["r_inside"] for n, v in merge_disjoint(N[k], NR[k]).items()}
    return out


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
    e0 = d - mu0                                         # WCR: null imposed
    ds = mu0 + W[:, :, None] * e0[None]
    ts = (ds.mean((1, 2)) - mu0) / cr1_se(ds)
    def pval(tol):
        ge, le = np.sum(ts >= t - tol), np.sum(ts <= t + tol)
        if enum:
            return min(1.0, 2 * min(ge, le) / len(ts))
        return min(1.0, 2 * (1 + min(ge, le)) / (len(ts) + 1))
    p = pval(0.0)                                        # strict floating-point comparison
    p_ties = pval(1e-10 * max(1.0, abs(t)))              # ties counted (see VERIFY.md discrepancy 2)
    e1 = d - m                                           # WCU: unrestricted residuals
    du = m + W[:, :, None] * e1[None]
    tu = (du.mean((1, 2)) - m) / cr1_se(du)
    q_lo, q_hi = np.quantile(tu, [0.025, 0.975])
    ci = [m - q_hi * se, m - q_lo * se]
    skew = float(stats.skew(d.mean(1)))
    # smallest attainable p: with enumeration the 3 constant positive Webb vectors reproduce t exactly, so
    # #{t* >= t} >= 3 and p >= 6 / 6^G (0.00013 for G = 6, as rule Section 4.1 states); with draws 2 / (B + 1)
    p_min = (2 * 3 / len(ts)) if enum else 2 / (len(ts) + 1)
    return dict(mean=float(m), se=float(se), t=float(t), p=float(p_ties), p_strict=float(p), p_min=float(p_min),
                ci=[float(ci[0]), float(ci[1])], G=G, N=G * P, enumerated=enum, n_vectors=len(ts),
                skew_cluster_means=skew)


def directional_pos(r):
    return bool(r["p"] < 0.05 and r["mean"] > 0)


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


def bca_ratio(num, den, alpha=0.05):
    G, P = num.shape
    idx = two_stage(G, P)
    est = num.mean() / den.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = boot_means(num, idx) / boot_means(den, idx)
    n_nonfinite = int(np.sum(~np.isfinite(boot)))
    boot = boot[np.isfinite(boot)]
    z0 = stats.norm.ppf(np.mean(boot < est))
    jk = np.array([np.delete(num, k, axis=0).mean() / np.delete(den, k, axis=0).mean() for k in range(G)])
    dj = jk.mean() - jk
    acc = float((dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5))
    def level(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo_l, hi_l = level(stats.norm.ppf(alpha / 2)), level(stats.norm.ppf(1 - alpha / 2))
    ci = np.quantile(boot, [lo_l, hi_l])
    return dict(value=float(est), ci=[float(ci[0]), float(ci[1])], z0=float(z0), acc=acc, B=int(len(boot)),
                n_nonfinite=n_nonfinite, alpha_adj=[float(lo_l), float(hi_l)])


def fieller(num, den, alpha=0.05):
    """Fieller interval for mean(num) / mean(den): CR1 covariance of the two means, t with G - 1 df."""
    G, P = num.shape
    N = G * P
    a, b = num.mean(), den.mean()
    Sa, Sb = (num - a).sum(1), (den - b).sum(1)
    k = G / (G - 1) / N ** 2
    Vaa, Vbb, Vab = k * (Sa ** 2).sum(), k * (Sb ** 2).sum(), k * (Sa * Sb).sum()
    t = stats.t.ppf(1 - alpha / 2, G - 1)
    A2 = b ** 2 - t ** 2 * Vbb
    B1 = a * b - t ** 2 * Vab
    C0 = a ** 2 - t ** 2 * Vaa
    disc = B1 ** 2 - A2 * C0
    if A2 > 0 and disc >= 0:
        r = math.sqrt(disc)
        return dict(value=float(a / b), ci=[float((B1 - r) / A2), float((B1 + r) / A2)], bounded=True,
                    t_crit=float(t), t_den=float(b / math.sqrt(Vbb)))
    return dict(value=float(a / b), ci=None, bounded=False, t_crit=float(t), t_den=float(b / math.sqrt(Vbb)))


# ------------------------------------------------------------------ P5 estimands (rule Section 5.5)
def p5_parts(sc, grid, th=THETA):
    I = arr(grid, lambda c: val(sc, c, "illusory"))
    C = arr(grid, lambda c: val(sc, c, "control180"))
    M = arr(grid, lambda c: val(sc, c, f"mis{th}"))
    RI = arr(grid, lambda c: val(sc, c, f"rig{th}_illusory"))
    RC = arr(grid, lambda c: val(sc, c, f"rig{th}_control180"))
    MC = arr(grid, lambda c: val(sc, c, f"mis{th}_control180"))
    E0, Erig, Emis = I - C, RI - RC, M - MC
    Luni = E0 - Emis
    Lrig = E0 - Erig
    Astar = Luni - Lrig
    Cstar = Luni - 2 * Lrig
    ident = dict(Astar_eq_Erig_minus_Emis=float(np.max(np.abs(Astar - (Erig - Emis)))),
                 Cstar_eq_Astar_minus_Lrig=float(np.max(np.abs(Cstar - (Astar - Lrig)))))
    sec = dict(A_drafted=(RI - M) - (RC - C), A_control_free=RI - M, control_rigid_change=RC - C)
    return dict(E0=E0, Erig=Erig, Emis=Emis, Luni=Luni, Lrig=Lrig, Astar=Astar, Cstar=Cstar, sec=sec, ident=ident)


def category(iut_met, half_met, luni_mean, share):
    if not iut_met:
        return "3"
    if half_met:
        return "1"
    if luni_mean > 0 and share >= 0.5:
        return "2a"
    return "2b"


HL_BOTH_U = "Inducer rotation separates DeepGaze IIE and MSDB from 15 local image nulls"
HL_ALT = {("DeepGaze IIE", "DeepGaze MSDB"): "On squares, alternating rotation separates DeepGaze IIE and MSDB from 15 local nulls",
          ("DeepGaze IIE",): "On squares, alternating rotation separates DeepGaze IIE from 15 local nulls",
          ("DeepGaze MSDB",): "On squares, alternating rotation separates DeepGaze MSDB from 15 local nulls"}
HL_CHARS_RULE = {"both_U": 74, ("DeepGaze IIE", "DeepGaze MSDB"): 84, ("DeepGaze IIE",): 75, ("DeepGaze MSDB",): 76}


def f4(x):
    return f"{x:+.4f}"


def ci4(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def f3(x):
    return f"{x:+.3f}"


def ci3(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def cat3_sentence(model, r, fx, fc):
    k = 15 - r["n_separated"]
    return (f"Rotating the whole display reproduced {fx(r['L_rig']['mean'])} {fc(r['L_rig']['ci'])} of the "
            f"uniform-rotation loss ({fx(r['L_uni']['mean'])} {fc(r['L_uni']['ci'])}), and the part that depends on "
            f"inducer alignment ({fx(r['align_effect']['mean'])} {fc(r['align_effect']['ci'])}) did not separate "
            f"{model} from {k} of the 15 local nulls, so this control could not confirm that the uniform-rotation "
            "loss depends on alignment.")


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    guard = require_public()                              # rule Section 8.1, before any score is read
    inputs = {p: md5(os.path.join(ROOT, p)) for p in (
        "v2/stim_v3/items.json", "v3/scores/raw_v3maps/DeepGaze_IIE.json", "v3/scores/raw_v3maps/DeepGaze_MSDB.json",
        "v3/scores/raw/DeepGaze_IIE.json", "v3/scores/raw/DeepGaze_MSDB.json", "v3/scores/raw_nulls.json",
        "neuro/reanalysis/scores_rigid/DeepGaze_IIE.json", "neuro/reanalysis/scores_rigid/DeepGaze_MSDB.json",
        "neuro/reanalysis/scores_rigid/nulls.json")}
    assert inputs["neuro/reanalysis/scores_rigid/DeepGaze_MSDB.json"] == MSDB_MD5
    keys, grid = make_design(); ksq, gsq = make_design("square")
    assert len(grid) == 12 and len(grid[0]) == 5 and len(gsq) == 6
    models = {"DeepGaze IIE": model_scores("DeepGaze_IIE"), "DeepGaze MSDB": model_scores("DeepGaze_MSDB")}
    nulls = null_scores()
    out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), guard=guard, input_md5=inputs,
               design=dict(clusters=[str(k) for k in keys], square_clusters=[str(k) for k in ksq],
                           first_config_per_cluster=[g[0] for g in grid]))

    # P1 (rule 5.1), copied from recompute_stats.py
    L = lambda sc, g: arr(g, lambda c: val(sc, c, "illusory") - val(sc, c, "mis20"))
    P1 = {}
    for mn, sc in models.items():
        Lm = L(sc, grid)
        comps = {k: wild(Lm - L(nulls[k], grid)) for k in NULLS}
        null_L = {k: float(L(nulls[k], grid).mean()) for k in NULLS}
        largest = max(null_L, key=null_L.get)
        P1[mn] = dict(L_model=wild(Lm), vs_null=comps, null_with_largest_loss=largest,
                      n_separated=sum(directional_pos(r) for r in comps.values()),
                      met=all(directional_pos(r) for r in comps.values()),
                      met_strict=all(r["p_strict"] < 0.05 and r["mean"] > 0 for r in comps.values()))
    out["P1"] = P1

    # P2 (rule 5.2), copied from recompute_stats.py
    Dl = lambda sc, g: arr(g, lambda c: val(sc, c, "alt20") - val(sc, c, "mis20"))
    P2 = {}
    for mn, sc in models.items():
        Dm = Dl(sc, gsq)
        comps = {k: wild(Dm - Dl(nulls[k], gsq)) for k in NULLS}
        P2[mn] = dict(Delta_model=wild(Dm), vs_null=comps,
                      n_separated=sum(directional_pos(r) for r in comps.values()),
                      met=all(directional_pos(r) for r in comps.values()),
                      met_strict=all(r["p_strict"] < 0.05 and r["mean"] > 0 for r in comps.values()))
    P2["null_square_Delta20_means"] = {k: float(Dl(nulls[k], gsq).mean()) for k in NULLS}   # for "0.0000 to +0.0003"
    out["P2"] = P2

    # P5 (rule 5.5), theta = 20
    NP = {k: p5_parts(nulls[k], grid) for k in NULLS}
    P5 = {}
    for mn, sc in models.items():
        X = p5_parts(sc, grid)
        comps = {k: wild(X["Astar"] - NP[k]["Astar"]) for k in NULLS}
        iut_met = all(directional_pos(r) for r in comps.values())
        Lu, Lr, Al, Ch = wild(X["Luni"]), wild(X["Lrig"]), wild(X["Astar"]), wild(X["Cstar"])
        half_met = directional_pos(Ch)
        share = bca_ratio(X["Lrig"], X["Luni"])
        share.update(t_den=Lu["t"], stable=bool(abs(Lu["t"]) >= 4.0),
                     denominator_excludes_zero=bool(Lu["ci"][0] > 0 or Lu["ci"][1] < 0),
                     fieller=fieller(X["Lrig"], X["Luni"]))
        share["reportable"] = share["denominator_excludes_zero"]
        cat = category(iut_met, half_met, Lu["mean"], share["value"])
        P5[mn] = dict(computable=True, L_uni=Lu, L_rig=Lr, align_effect=Al, vs_null=comps, iut_met=iut_met,
                      iut_met_strict=all(r["p_strict"] < 0.05 and r["mean"] > 0 for r in comps.values()),
                      n_separated=sum(directional_pos(r) for r in comps.values()),
                      not_separated=[k for k in NULLS if not directional_pos(comps[k])],
                      half_criterion=Ch, half_met=half_met, rigid_share=share,
                      secondary={k: wild(v) for k, v in X["sec"].items()}, category=cat,
                      identities_max_abs=X["ident"],
                      E0=wild(X["E0"]), E_rig=wild(X["Erig"]), E_mis_matched=wild(X["Emis"]))
    P5["nulls_own_values"] = {k: dict(L_uni=float(NP[k]["Luni"].mean()), L_rig=float(NP[k]["Lrig"].mean()),
                                      A_star=float(NP[k]["Astar"].mean()), C_star=float(NP[k]["Cstar"].mean()))
                              for k in NULLS}
    P5["both_category_1"] = all(P5[m]["category"] == "1" for m in MODELS)
    out["P5"] = P5

    # abstract / highlight-4 rule (rule 5.5, last block) and the contributions clause
    U = [m for m in MODELS if P1[m]["met"] and P5[m]["category"] == "1"]
    A = [m for m in MODELS if P2[m]["met"]]
    if len(U) == 2:
        branch, hl = "both models in U: uniform-rotation clause (both models) then alternation clause for A", HL_BOTH_U
        hl_rule_chars = HL_CHARS_RULE["both_U"]
    elif A:
        branch = "A not empty: abstract leads with the ratio-free square alternation clause for A; no uniform-rotation clause"
        hl = HL_ALT[tuple(A)]; hl_rule_chars = HL_CHARS_RULE[tuple(A)]
    else:
        branch, hl, hl_rule_chars = "A empty: no rotation claim; highlight 4 replaced or dropped", None, None
    both_cat3 = all(P5[m]["category"] == "3" for m in MODELS)
    out["abstract_rotation"] = dict(
        uniform_models=U, alternation_models=A, branch=branch, highlight4=hl,
        highlight4_chars=None if hl is None else len(hl), highlight4_chars_stated_in_rule=hl_rule_chars,
        highlight_char_counts_all={"both_U": len(HL_BOTH_U), **{"+".join(k): len(v) for k, v in HL_ALT.items()}},
        rule72_drop_uniform_abstract_clause=any(P5[m]["category"] != "1" for m in MODELS),
        P1_leaves_contributions=both_cat3,
        p2_abstract_clause_values=[f3(P2[m]["Delta_model"]["mean"]) for m in A])
    out["wording"] = dict(
        category3_4dp={m: cat3_sentence(m, P5[m], f4, ci4) for m in MODELS if P5[m]["category"] == "3"},
        category3_3dp={m: cat3_sentence(m, P5[m], f3, ci3) for m in MODELS if P5[m]["category"] == "3"})
    assert "torch" not in sys.modules
    out["torch_imported"] = "torch" in sys.modules
    out["wall_s"] = round(time.time() - t0, 1)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    json.dump(out, open(OUT, "w"), indent=1, default=conv)
    print(f"wrote {OUT} ({out['wall_s']} s)")
    for m in MODELS:
        r = P5[m]
        print(m, "cat", r["category"], "sep", r["n_separated"], "Luni", f4(r["L_uni"]["mean"]), ci4(r["L_uni"]["ci"]),
              "Lrig", f4(r["L_rig"]["mean"]), "A*", f4(r["align_effect"]["mean"]), "C*", f4(r["half_criterion"]["mean"]),
              "p", r["half_criterion"]["p"])
    print(out["abstract_rotation"])


if __name__ == "__main__":
    main()
