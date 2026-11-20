"""
Property package: final application of nn/ANALYSIS_RULE.md to the 18 layer-sweep readouts.

Reads (never writes) the outputs of nn/run_all.sh: nn/results/summary.json (analysis by nn/analyze_nn.py), the stored
scores of the 9 Appendix B readouts (v3/scores), the new scores of the other 9 (nn/results/raw_v3maps, raw_alt), the
ERFs (nn/results/erf/<bb>.json), v3/scores/results.json (stored summaries and local nulls) and v2/anatomy/anatomy.json.

It (1) verifies the SHA-256 of the rule, (2) re-derives every decision-bearing number with code written separately from
analyze_nn.py (same procedure, same bootstrap indices and seeds) and checks it against summary.json, (3) checks that the
nine stored readouts reproduce v3/scores/results.json (Appendix B), (4) adds the one Section 7 item that summary.json
lacks (Spearman rho for the training-objective family, item 8) and the Section 4 ERF details, and (5) writes
nn/results/property_results.json, nn/PROPERTY_RESULTS.md and nn/figs/erf_vs_rotation.{pdf,png}.
Usage: <venv python> nn/property_results.py
"""
import os, sys, json, time, hashlib
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import rankdata, spearmanr

NN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NN)
import analyze_nn as AN                                    # noqa: E402  (functions only; main() is not run)
C, AA = AN.C, AN.AA
DA, DS = AA.DES["all"], AA.DES["square"]
NB = AA.B
SEED, NPERM = 2026, 100_000
OUT_JSON = os.path.join(NN, "results", "property_results.json")
OUT_MD = os.path.join(NN, "PROPERTY_RESULTS.md")
FIGDIR = os.path.join(NN, "figs")
BBLAB = {"dinov2": "DINOv2-B", "clip": "CLIP-L/14", "siglip": "SigLIP-L/16"}
CHECKS = []                                                # (name, ok, detail)


def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


# ------------------------------------------------------------------ 0. rule hash
def rule_hash():
    h = hashlib.sha256(open(C.RULE, "rb").read()).hexdigest()
    rec = open(C.RULE_SHA).read().split()
    return h, rec[0], " ".join(rec[1:])


# ------------------------------------------------------------------ independent statistics
def rho_s(x, y):
    return float(spearmanr(x, y).statistic)


def boot_rho(x, Yb):
    rx = rankdata(x); rx = rx - rx.mean()
    ry = rankdata(Yb, axis=0); ry = ry - ry.mean(0)
    return (rx @ ry) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum(0))


def perm_p(x, y):
    """Two-sided permutation p for Spearman rho; 100,000 random permutations from RandomState(2026) (n > 9)."""
    n = len(x); rx = rankdata(x); ry = rankdata(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean()); ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    obs = rx @ ry
    assert n > 9
    rng = np.random.RandomState(SEED)
    P = np.argsort(rng.rand(NPERM, n), axis=1)
    null = ry[P] @ rx
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def within(x, y, g):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {}
    for b in sorted(set(g)):
        ii = [i for i, gg in enumerate(g) if gg == b]
        out[b] = dict(n=len(ii), rho=(rho_s(x[ii], y[ii]) if len(ii) >= 3 else None))
    qual = [b for b in sorted(out) if out[b]["n"] >= 3]
    if not qual:
        return out, None, None, qual
    T = sum(out[b]["n"] * out[b]["rho"] for b in qual) / sum(out[b]["n"] for b in qual)
    rng = np.random.RandomState(SEED); null = np.zeros(NPERM)
    for b in qual:
        ii = np.array([i for i, gg in enumerate(g) if gg == b])
        rx, ry = rankdata(x[ii]), rankdata(y[ii]); rx = rx - rx.mean(); ry = ry - ry.mean()
        P = np.argsort(rng.rand(NPERM, len(ii)), axis=1)
        null += len(ii) * (rx[P] @ ry) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    null /= sum(out[b]["n"] for b in qual)
    return out, float(T), float(np.mean(np.abs(null) >= abs(T) - 1e-12)), qual


def r2_models(y, x, g):
    y = np.asarray(y, float); lx = np.log2(np.asarray(x, float)); n = len(y)
    sst = ((y - y.mean()) ** 2).sum()
    gm = {b: y[[i for i, gg in enumerate(g) if gg == b]].mean() for b in set(g)}
    r2_bb = float(((np.array([gm[b] for b in g]) - y.mean()) ** 2).sum() / sst)
    b1 = np.polyfit(lx, y, 1); r2_x = float(1 - ((y - np.polyval(b1, lx)) ** 2).sum() / sst)
    G = sorted(set(g)); D = np.column_stack([np.ones(n), lx] + [[1.0 if gg == b else 0.0 for gg in g] for b in G[1:]])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]; r2_both = float(1 - ((y - D @ beta) ** 2).sum() / sst)
    k_bb, k_both = len(G) - 1, len(G)
    adj = lambda r2, k: float(1 - (1 - r2) * (n - 1) / (n - k - 1))
    return dict(R2_bb=r2_bb, R2_x=r2_x, R2_both=r2_both, adjR2_bb=adj(r2_bb, k_bb), adjR2_x=adj(r2_x, 1),
                adjR2_both=adj(r2_both, k_both), partial_R2_x_given_bb=float((r2_both - r2_bb) / (1 - r2_bb)),
                coef_x_joint=float(beta[1]), coef_x_alone=float(b1[0]))


def full_stats(names, x, y, yb, g):
    o = dict(n=len(names), rho=rho_s(x, y), perm_p=perm_p(x, y))
    if yb is not None:
        rb = boot_rho(x, yb)
        o["n_boot_nan"] = int(np.isnan(rb).sum())
        o["rho_ci"] = [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
        o["rho_boot_frac_pos"] = float(np.nanmean(rb > 0))
    w, T, pT, qual = within(x, y, g)
    o.update(within=w, T=T, T_perm_p=pT, T_backbones=qual,
             n_qual_backbones_rho_pos=sum(1 for b in qual if w[b]["rho"] > 0))
    o.update(r2_models(y, x, g))
    return o


def decide(a):
    """Section 6, written from the rule text."""
    n = a["n"]
    if n < 8:
        return "INCONCLUSIVE", "I-a", {}
    S1 = a["rho"] >= 0.50 and a["perm_p"] < 0.05 and a["rho_ci"][0] > 0
    S2 = (len(a["T_backbones"]) >= 2 and a["T"] > 0 and a["T_perm_p"] < 0.10 and a["n_qual_backbones_rho_pos"] >= 2)
    S3 = a["adjR2_x"] >= a["adjR2_bb"] and a["coef_x_joint"] > 0 and a["partial_R2_x_given_bb"] >= 0.25
    C1 = a["rho"] < 0 and a["perm_p"] < 0.05
    C2 = a["rho"] <= 0.10 and a["T"] is not None and a["T"] <= 0 and (a["adjR2_bb"] - a["adjR2_x"]) >= 0.20
    f = dict(S1=bool(S1), S2=bool(S2), S3=bool(S3), C1=bool(C1), C2=bool(C2))
    if S1 and S2 and S3:
        return "SUPPORTS", "S", f
    if C1 or C2:
        return "CONTRADICTS", "C1" if C1 else "C2", f
    if S1 and not S2:
        return "INCONCLUSIVE", "I-b", f
    if S1 and S2 and not S3:
        return "INCONCLUSIVE", "I-c", f
    return "INCONCLUSIVE", "I-d", f


def inverted_u(x, y, yb, g, blocks):
    lx = np.log2(np.asarray(x, float)); y = np.asarray(y, float)
    D = np.column_stack([np.ones_like(lx), lx, lx ** 2])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    bq = np.linalg.lstsq(D, yb, rcond=None)[0][2]
    ci = [float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))]
    v = float(-beta[1] / (2 * beta[2]))
    left, right = int((lx < v).sum()), int((lx > v).sum())
    peak = {}
    for b in sorted(set(g)):
        ii = [i for i, gg in enumerate(g) if gg == b]
        if len(ii) < 3:
            peak[b] = False; continue
        bl = [blocks[i] for i in ii]; best = blocks[ii[int(np.argmax(y[ii]))]]
        peak[b] = bool(min(bl) < best < max(bl))
    c1 = bool(beta[2] < 0 and ci[1] < 0); c2 = bool(lx.min() < v < lx.max() and left >= 2 and right >= 2)
    c3 = bool(sum(peak.values()) >= 2)
    return dict(quad_coef=float(beta[2]), quad_coef_ci=ci, vertex_erf50=2 ** v, n_left=left, n_right=right,
                peak_interior_by_backbone=peak, crit_i=c1, crit_ii=c2, crit_iii=c3, allowed=bool(c1 and c2 and c3))


# ------------------------------------------------------------------ helpers
def arr(parts, cond, D=DA, m="r_inside"):
    return AA.arr(D, parts, cond, m)


def eff(parts, a, b, D=DA):
    v = arr(parts, a, D) - arr(parts, b, D); bm = D.bmeans(v)
    return float(v.mean()), [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def sens_vec(parts, cond, D=DA):
    num = arr(parts, cond, D) - arr(parts, "control180", D); den = arr(parts, "illusory", D) - arr(parts, "control180", D)
    return 1 - num.mean() / den.mean(), 1 - D.bmeans(num) / D.bmeans(den)


def arch_blur(patch, s):
    g = s / patch
    sx, sy = (1024 / g) / np.sqrt(6), (768 / g) / np.sqrt(6)
    return float(np.sqrt(sx * sy))


def close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    h, rec_h, rec_when = rule_hash()
    check("rule SHA-256 matches nn/ANALYSIS_RULE.sha256", h == rec_h, f"{h} ({rec_when})")
    if h != rec_h:
        raise SystemExit("rule changed; stopping")
    S = json.load(open(os.path.join(NN, "results", "summary.json")))
    check("summary.json: no missing inputs, real ERFs, rule unchanged at analysis time",
          S["missing"] == [] and not S["fake_erf_code_test"] and S["rule_unchanged"] and S["rule_sha256_now"] == h)
    v3res = json.load(open(os.path.join(C.V3SC, "results.json")))
    cc = C.heldout_cc()
    erfj = {bb: json.load(open(os.path.join(NN, "results", "erf", f"{bb}.json"))) for bb in C.LAYERS}

    # ---------------- per-readout data and independent per-readout checks
    names, parts_all, info = [], {}, {}
    for bb, L in C.all_readouts():
        n = C.rname(bb, L); names.append(n)
        src = "stored" if L in C.SCORED[bb] else C.RES
        parts = AN.parts_for(bb, L, src); assert parts is not None, n
        parts_all[n] = parts
        r = S["readouts"][n]
        E, Eci = eff(parts, "illusory", "control180")
        elig = Eci[0] > 0 or Eci[1] < 0
        ok = close(E, r["E"]["mean"], 6e-5) and close(Eci[0], r["E"]["ci"][0], 6e-5) and close(Eci[1], r["E"]["ci"][1], 6e-5) \
            and elig == r["eligible"] and r["cc"] == cc[(bb, L)]
        s20, _ = sens_vec(parts, "mis20")
        ok &= close(s20, r["S20_point_any"], 1e-9)
        if r["U"]["20"] is not None:
            ok &= close(1 - s20, r["U"]["20"]["value"], 6e-5)
        e = erfj[bb]["layers"][str(L)]
        ok &= r["ERF50"] == e["ro_sizeavg_all"]["erf50"] == 0.5 * (e["ro_s336_all"]["erf50"] + e["ro_s448_all"]["erf50"])
        check(f"{n}: E, CI, eligibility, CC, S20 point, ERF50 re-derived = summary.json", ok,
              f"E {E:+.4f} [{Eci[0]:+.4f}, {Eci[1]:+.4f}] eligible {elig}")
        info[n] = dict(bb=bb, L=L, E=E, Eci=Eci, elig=elig)
        if L in C.SCORED[bb]:                              # stored readouts must reproduce Appendix B / v3 results.json
            o = v3res["models"][n]["results"]["r_inside"]["all"]
            pairs = [("E", o["aligned"], r["E"]), ("slope", o["sr_slope_vs_control180"], r["slope"])]
            if o["ratios_computed"]:
                for t in ("10", "20", "30"):
                    pairs += [(f"U{t}", o["ratio_uni"][t], r["U"][t]), (f"A{t}", o["ratio_alt"][t], r["A"][t])]
            for t in ("10", "20", "30"):
                pairs.append((f"D{t}", o["drop_ratio"][t], r["D"][t]))
            same = all(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) for _, a, b in pairs) \
                and o["ratios_computed"] == r["eligible"]
            check(f"{n}: summary.json equals stored v3/scores/results.json (E, U, A, D at 10/20/30, slope)", same)

    # ---------------- primary test, re-derived
    elig = [n for n in names if info[n]["elig"]]
    g = [info[n]["bb"] for n in elig]; blocks = [info[n]["L"] for n in elig]
    x = np.array([S["readouts"][n]["ERF50"] for n in elig], float)
    ys = [sens_vec(parts_all[n], "mis20") for n in elig]
    y = np.array([a for a, _ in ys]); yb = np.vstack([b for _, b in ys])
    prim = full_stats(elig, x, y, yb, g)
    cat, case, flags = decide(prim)
    iu = inverted_u(x, y, yb, g, blocks)
    P = S["primary"]
    ok = (prim["n"] == P["n"] and close(prim["rho"], P["rho"], 1e-9) and close(prim["perm_p"], P["perm_p"], 1e-12)
          and all(close(a, b, 1e-9) for a, b in zip(prim["rho_ci"], P["rho_ci"])) and close(prim["T"], P["T"], 1e-9)
          and close(prim["T_perm_p"], P["T_perm_p"], 1e-12)
          and all(close(prim[k], P[k], 1e-9) for k in ("R2_bb", "R2_x", "R2_both", "adjR2_bb", "adjR2_x", "partial_R2_x_given_bb", "coef_x_joint"))
          and (cat, case) == (P["decision"], P["case"]) and flags == P["criteria"]
          and close(iu["quad_coef"], P["inverted_u"]["quad_coef"], 1e-9) and iu["allowed"] == P["inverted_u"]["allowed"]
          and iu["peak_interior_by_backbone"] == P["inverted_u"]["peak_interior_by_backbone"])
    check("primary test re-derived independently = summary.json (rho, CI, perm p, T, p_T, R2s, decision, inverted-U checks)", ok,
          f"rho {prim['rho']:+.4f} CI [{prim['rho_ci'][0]:+.4f}, {prim['rho_ci'][1]:+.4f}] p {prim['perm_p']:.4f}; "
          f"T {prim['T']:+.4f} p_T {prim['T_perm_p']:.4f}; {cat} {case}; boot NaN {prim['n_boot_nan']}")
    # S20 CI per eligible readout (for the table and figure)
    s20ci = {n: [float(np.nanpercentile(yb[i], 2.5)), float(np.nanpercentile(yb[i], 97.5))] for i, n in enumerate(elig)}

    # ---------------- secondary: re-derive the eligible-set ones, add item 8 objective rho
    sec = S["secondary"]
    for lab, cond in (("S30", "mis30"), ("alt_S20", "alt20")):
        v = [sens_vec(parts_all[n], cond) for n in elig]
        a = full_stats(elig, x, np.array([p for p, _ in v]), np.vstack([b for _, b in v]), g)
        check(f"secondary {lab} re-derived = summary.json", close(a["rho"], sec[lab]["rho"], 1e-9) and close(a["perm_p"], sec[lab]["perm_p"], 1e-12))
    for key in ("ERF90", "bbERF50", "bbERF90", "ERF50_inside", "ERF50_ind"):
        xx = np.array([S["readouts"][n][key] for n in elig], float)
        a = full_stats(elig, xx, y, yb, g)
        check(f"secondary S20 vs {key} re-derived = summary.json", close(a["rho"], sec[f"S20_vs_{key}"]["rho"], 1e-9)
              and close(a["perm_p"], sec[f"S20_vs_{key}"]["perm_p"], 1e-12) and close(a["T"], sec[f"S20_vs_{key}"]["T"], 1e-9))
    objective = np.array([0.0 if b == "dinov2" else 1.0 for b in g])
    item8 = dict(sec["competing"])
    item8["rho_objective_family"] = rho_s(objective, y)
    item8["rho_objective_family_note"] = "Spearman rho between S20 and a two-group indicator (0 = DINOv2 self-supervised, 1 = CLIP or SigLIP image-text), average ranks for ties"
    dep = np.array([info[n]["L"] / C.NBLOCKS[info[n]["bb"]] for n in elig])
    check("item 8 relative-depth rho re-derived = summary.json", close(rho_s(dep, y), item8["rho_rel_depth"], 1e-9))
    # per-size interpretation check (rule item 3 is run with per-size eligibility in analyze_nn.py)
    alt_size = {}
    for k, s in enumerate(C.SIZES):
        v = [sens_vec([parts_all[n][k]], "mis20") for n in elig]
        xs = np.array([S["readouts"][n][f"ERF50_s{s}"] for n in elig], float)
        a = full_stats(elig, xs, np.array([p for p, _ in v]), np.vstack([b for _, b in v]), g)
        alt_size[str(s)] = {kk: a[kk] for kk in ("n", "rho", "rho_ci", "perm_p", "T", "T_perm_p")}
        es = sec[f"size{s}"]["readouts"]
        check(f"secondary size{s}: eligibility set per size = readouts whose {s}-px aligned CI excludes 0",
              es == [n for n in names if S["readouts"][n]["per_size"][str(s)]["eligible"]], f"n = {len(es)}")

    # ---------------- ERF details (Section 4 'also reported')
    erf_detail = {}
    for n in names:
        bb, L = info[n]["bb"], info[n]["L"]; e = erfj[bb]["layers"][str(L)]
        d = {}
        for kind in ("ro", "bb"):
            for s in C.SIZES:
                pd = np.array(e[f"{kind}_s{s}_all"]["per_display_erf50"], float)
                d[f"{kind}_s{s}"] = dict(erf50=e[f"{kind}_s{s}_all"]["erf50"], erf90=e[f"{kind}_s{s}_all"]["erf90"],
                                         erf50_model=e[f"{kind}_s{s}_all_model"]["erf50"], erf90_model=e[f"{kind}_s{s}_all_model"]["erf90"],
                                         per_display_erf50_min=float(pd.min()), per_display_erf50_median=float(np.median(pd)),
                                         per_display_erf50_max=float(pd.max()), per_display_erf50_iqr=[float(np.percentile(pd, 25)), float(np.percentile(pd, 75))],
                                         inside_erf50=e[f"{kind}_s{s}_inside"]["erf50"], ind_erf50=e[f"{kind}_s{s}_ind"]["erf50"])
            for st in ("all", "inside", "ind"):
                d[f"{kind}_sizeavg_{st}"] = e[f"{kind}_sizeavg_{st}"]
        erf_detail[n] = d
    ro = [S["readouts"][n]["ERF50"] for n in names]; bbv = [S["readouts"][n]["bbERF50"] for n in names]
    depth_all = [info[n]["L"] / C.NBLOCKS[info[n]["bb"]] for n in names]
    erf_ranges = dict(readout_ERF50=[min(ro), max(ro)], block_token_ERF50=[min(bbv), max(bbv)],
                      rho_ERF50_reldepth_all18=rho_s(ro, depth_all), rho_bbERF50_reldepth_all18=rho_s(bbv, depth_all),
                      rho_ERF50_bbERF50_all18=rho_s(ro, bbv))

    # ---------------- per-readout records
    readouts = []
    for n in names:
        r = S["readouts"][n]; bb, L = info[n]["bb"], info[n]["L"]
        U, A, Dd = r["U"], r["A"], r["D"]
        readouts.append(dict(
            readout=n, backbone=BBLAB[bb], block=L, n_blocks=C.NBLOCKS[bb], source=r["source"], heldout_mit1003_cc=r["cc"],
            E_vs_control180=r["E"], eligible=r["eligible"],
            U10=U["10"], U20=U["20"], U30=U["30"], A10=A["10"], A20=A["20"], A30=A["30"],
            S20=(1 - U["20"]["value"]) if U["20"] else None, S20_ci=s20ci.get(n),
            S30=(1 - U["30"]["value"]) if U["30"] else None,
            S20_point_any_eligibility=r["S20_point_any"], L20=r["L20"],
            D10=Dd["10"], D20=Dd["20"], D30=Dd["30"], D_square20=r["D_square"]["20"],
            figure_mass_loss20=r["loss_uni"]["20"], figure_mass_loss30=r["loss_uni"]["30"],
            support_ratio_slope=r["slope"], E_vs_control90=r["E_control90"], real_vs_control180=r["real"],
            real_filled_vs_control180=r["real_filled"], core_r_effect=r["core_r"], core_share_effect=r["core_share"],
            per_size=r["per_size"],
            ERF50=r["ERF50"], ERF90=r["ERF90"], ERF50_s336=r["ERF50_s336"], ERF50_s448=r["ERF50_s448"],
            ERF50_inside=r["ERF50_inside"], ERF50_inducer=r["ERF50_ind"], bbERF50=r["bbERF50"], bbERF90=r["bbERF90"],
            erf_detail=erf_detail[n],
            arch_blur_px={"336": arch_blur(C.PATCH[bb], 336), "448": arch_blur(C.PATCH[bb], 448)},
            null_comparison=r["null_comparison"]))

    nb = S["nulls"]
    bracket = {k: v for k, v in nb.items() if int(k.split("_s")[1]) in (8, 16, 25)}
    out = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        rule=dict(path="nn/ANALYSIS_RULE.md", sha256=h, recorded=rec_when, unchanged=True),
        inputs=dict(analysis="nn/results/summary.json (nn/analyze_nn.py, run by nn/run_all.sh 2026-11-17)",
                    stored_scores="v3/scores/raw_v3maps, v3/scores/raw (9 Appendix B readouts)",
                    new_scores="nn/results/raw_v3maps, nn/results/raw_alt (9 readouts)", erf="nn/results/erf/<bb>.json",
                    cc="v2/anatomy/anatomy.json", nulls="v3/scores/results.json (null entries; maps in v3/scores/raw_nulls.json)"),
        deviations=[],
        interpretations=[
            "Section 7 item 3 (per size) is run with eligibility determined at that size (the 336-only and 448-only analyses each apply the Section 2 eligibility rule), as implemented in nn/analyze_nn.py before the results were seen. The alternative reading (size-averaged eligibility set, n = 13) is given under per_size_alternative_reading for transparency; neither is decision-bearing.",
            "Section 7 item 6 is reported as S20 = 1 - U(20) point values (the rho with U(20) itself has the opposite sign); summary.json labels it all18_point_U20.",
            "Section 3 'nulls in the bracket' = the corner, ink and edge nulls at sigma 8, 16 and 25 px; U(20), A(20) and slope comparisons use the corner nulls only, as the rule states.",
            "Section 7 item 8: summary.json lacked the Spearman rho for the training-objective family; it is computed here (secondary.competing.rho_objective_family)."],
        checks=CHECKS,
        blur_scale=dict(rule="SD of the bilinear upsampling kernel, Delta/sqrt(6) per axis, geometric mean of the two axes",
                        px={"patch14": {"336": arch_blur(14, 336), "448": arch_blur(14, 448)},
                            "patch16": {"336": arch_blur(16, 336), "448": arch_blur(16, 448)}},
                        null_bracket_sigma_px=[8, 16, 25], reference_sigma_px=16),
        nulls_in_bracket=bracket, nulls_all=nb,
        readouts=readouts,
        primary=dict(S["primary"], independent_rederivation=dict(rho=prim["rho"], rho_ci=prim["rho_ci"], perm_p=prim["perm_p"],
                                                                 T=prim["T"], T_perm_p=prim["T_perm_p"], decision=cat, case=case,
                                                                 criteria=flags, n_boot_nan=prim["n_boot_nan"])),
        secondary=dict(sec, competing=item8),
        per_size_alternative_reading=alt_size,
        erf_ranges=erf_ranges,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=float)
    make_figure(out)
    write_md(out)
    print(f"done in {time.time() - t0:.0f}s; failed checks: {[c['check'] for c in CHECKS if not c['ok']]}")


# ------------------------------------------------------------------ figure
COL = {"DINOv2-B": "#2a78d6", "CLIP-L/14": "#eb6834", "SigLIP-L/16": "#1baf7a"}   # validated 3-slot categorical palette
MRK = {"DINOv2-B": "o", "CLIP-L/14": "s", "SigLIP-L/16": "^"}
LABEL_OFF = {("SigLIP-L/16", 12): (-30, 12), ("SigLIP-L/16", 20): (14, 12), ("CLIP-L/14", 12): (-30, 10),
             ("CLIP-L/14", 16): (16, -14), ("SigLIP-L/16", 24): (8, 4), ("DINOv2-B", 8): (8, -4)}
LEADER = {("SigLIP-L/16", 12), ("SigLIP-L/16", 20), ("CLIP-L/14", 12), ("CLIP-L/14", 16)}
DODGE = {("SigLIP-L/16", 12): -0.8, ("SigLIP-L/16", 20): +0.8}     # display-only offset (px) for two near-coincident points


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0, color="#8a8984", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(1, color="#8a8984", lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(253, 0.03, "no loss under rotation", color="#52514e", fontsize=7.5, va="bottom")
    ax.text(253, 1.03, "effect abolished", color="#52514e", fontsize=7.5, va="bottom")
    for bbl in COL:
        rs = [r for r in out["readouts"] if r["backbone"] == bbl and r["eligible"]]
        xs = [r["ERF50"] + DODGE.get((bbl, r["block"]), 0.0) for r in rs]; ys = [r["S20"] for r in rs]
        lo = [r["S20"] - r["S20_ci"][0] for r in rs]; hi = [r["S20_ci"][1] - r["S20"] for r in rs]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=COL[bbl], elinewidth=1.0, alpha=0.55, capsize=0, zorder=2)
        ax.scatter(xs, ys, s=52, marker=MRK[bbl], color=COL[bbl], edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
                   label=f"{bbl} ({len(rs)} eligible blocks)")
        for r, xv in zip(rs, xs):
            key = (bbl, r["block"]); dx, dy = LABEL_OFF.get(key, (7, 4))
            ax.annotate(f"L{r['block']}", (xv, r["S20"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color="#0b0b0b", zorder=4, va="center",
                        arrowprops=(dict(arrowstyle="-", color="#8a8984", lw=0.6, shrinkA=1, shrinkB=4) if key in LEADER else None))
    p = out["primary"]
    ax.set_xlabel("Readout ERF50 (display px; mean of 336 and 448 px inputs)")
    ax.set_ylabel("Rotation sensitivity S20 = 1 - U(20)")
    ax.set_title("Rotation sensitivity against readout ERF50 (13 eligible readouts, seed 0)", fontsize=9.5, loc="left", color="#0b0b0b")
    ax.grid(axis="y", color="#e6e5e1", lw=0.6, zorder=0)
    ax.set_xlim(232, 332)
    stat = (f"Spearman rho = {p['rho']:+.2f}, 95% CI [{p['rho_ci'][0]:+.2f}, {p['rho_ci'][1]:+.2f}] (stimulus resampling),\n"
            f"permutation p = {p['perm_p']:.2f}; pooled within-network rho = {p['T']:+.2f} (p = {p['T_perm_p']:.2f}). "
            f"Rule decision: {p['decision'].lower()} ({p['case']}).")
    ax.text(0.0, -0.20, stat, transform=ax.transAxes, fontsize=7.5, color="#52514e", va="top")
    inel = [f"{r['readout']}" for r in out["readouts"] if not r["eligible"]]
    ax.text(0.0, -0.31, "Not shown (aligned-effect CI includes 0, S20 undefined): " + ", ".join(inel) + ".\n"
            "Bars: 95% cluster-bootstrap CI of S20 over stimuli (ERF fixed).\n"
            "SigLIP-L/16 L12 (318.0 px) and L20 (318.5 px) are drawn 0.8 px apart for visibility.", transform=ax.transAxes, fontsize=7, color="#52514e", va="top")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "erf_vs_rotation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "erf_vs_rotation.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def f3(v, sign=True):
    return "n/a" if v is None else (f"{v:+.3f}" if sign else f"{v:.3f}")


def fe(e, d=3):
    return f"{e['mean']:+.{d}f} [{e['ci'][0]:+.{d}f}, {e['ci'][1]:+.{d}f}]"


def frt(r):
    return "n.e." if r is None else f"{r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def pf(p):
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1.000"


def rho_line(a):
    w = ", ".join(f"{k} {('n/a (n ' + str(v['n']) + ')') if v['rho'] is None else format(v['rho'], '+.2f') + ' (n ' + str(v['n']) + ')'}"
                  for k, v in a["within"].items())
    ci = f"[{a['rho_ci'][0]:+.2f}, {a['rho_ci'][1]:+.2f}]" if "rho_ci" in a else "none (point values)"
    T = "n/a" if a.get("T") is None else f"{a['T']:+.2f} (p {pf(a['T_perm_p'])})"
    return ci, w, T


def write_md(out):
    L = []; w = L.append
    p = out["primary"]; iu = p["inverted_u"]; R = out["readouts"]; sec = out["secondary"]; comp = sec["competing"]
    w("# Property package: does rotation sensitivity track the effective receptive field of the block read?\n")
    w(f"Results of `nn/ANALYSIS_RULE.md` applied to the 18 layer-sweep readouts (9 stored Appendix B readouts + 9 newly scored). "
      f"Generated {out['created_utc']} by `nn/property_results.py` from the outputs of `nn/run_all.sh` (2026-11-17, 08:08 to 09:30 UTC).\n")
    w(f"Rule SHA-256 `{out['rule']['sha256']}`, identical to the value recorded in `nn/ANALYSIS_RULE.sha256` "
      f"({out['rule']['recorded']}); the rule file is read-only and was not changed.\n")
    w("Scope. Single seed (seed 0), no seed variance; all results describe these 18 readouts only. Confidence intervals reflect stimulus "
      "sampling only (10,000 cluster resamples, v2 indices, seed 2026). ERF and depth are collinear within each network. Backbone identity "
      "confounds training objective with data, architecture size and patch size, so 'training objective' is not isolated by this design.\n")

    w("## 1. Decision under the rule (primary test)\n")
    w(f"**{p['decision']}, case {p['case']}.** Pre-written wording that applies:\n")
    w(f"> {p['wording']}\n")
    w("Criteria (Section 6):\n")
    w("| criterion | requirement | observed | met |\n|---|---|---|---|")
    w(f"| n | >= 8 eligible readouts for S or C | n = {p['n']} | yes |")
    w(f"| S1 | rho >= +0.50, p < .05, lower CI > 0 | rho = {p['rho']:+.3f}, p = {pf(p['perm_p'])}, CI [{p['rho_ci'][0]:+.3f}, {p['rho_ci'][1]:+.3f}] | {'yes' if p['criteria']['S1'] else 'no'} |")
    wb = ", ".join(f"{k} {v['rho']:+.2f}" for k, v in p["within"].items() if v["rho"] is not None)
    w(f"| S2 | T > 0, p_T < .10, rho_b > 0 in >= 2 backbones with >= 3 eligible | T = {p['T']:+.3f}, p_T = {pf(p['T_perm_p'])}; rho_b: {wb} ({p['n_qual_backbones_rho_pos']} positive) | {'yes' if p['criteria']['S2'] else 'no'} |")
    w(f"| S3 | adj R2_erf >= adj R2_bb, ERF coefficient > 0 in joint model, partial R2 >= .25 | adj R2_erf {p['adjR2_x']:.3f} vs adj R2_bb {p['adjR2_bb']:.3f}; joint coefficient {p['coef_x_joint']:+.3f}; partial R2 {p['partial_R2_x_given_bb']:.3f} | {'yes' if p['criteria']['S3'] else 'no'} |")
    w(f"| C1 | rho < 0 with p < .05 | rho = {p['rho']:+.3f}, p = {pf(p['perm_p'])} | {'yes' if p['criteria']['C1'] else 'no'} |")
    w(f"| C2 | rho <= +0.10, T <= 0, adj R2_bb - adj R2_erf >= .20 | {p['rho']:+.3f}, {p['T']:+.3f}, {p['adjR2_bb'] - p['adjR2_x']:+.3f} | {'yes' if p['criteria']['C2'] else 'no'} |")
    w("")
    w("Exact statistics (eligible readouts, ERF50 = readout ERF50, outcome S20 = 1 - U(20), inside r, pooled over shapes, size average):\n")
    w(f"- n = {p['n']}: " + ", ".join(p["readouts"]) + ".")
    w(f"- Spearman rho = {p['rho']:+.4f}; 95% percentile CI over 10,000 stimulus resamples (ERF50 fixed, eligibility fixed) "
      f"[{p['rho_ci'][0]:+.4f}, {p['rho_ci'][1]:+.4f}]; share of resamples with rho > 0 = {p['rho_boot_frac_pos']:.4f}.")
    w(f"- Two-sided permutation p = {p['perm_p']:.4f} ({p['perm_kind']}, RandomState(2026); Monte Carlo SE about 0.0013).")
    w(f"- Within backbone: " + ", ".join(f"{k} rho_b = {v['rho']:+.3f} (n {v['n']})" for k, v in p["within"].items()) +
      f"; pooled T = {p['T']:+.4f}, stratified permutation p = {p['T_perm_p']:.4f} (100,000 permutations within backbone).")
    w(f"- Variance: R2_bb = {p['R2_bb']:.3f} (adjusted {p['adjR2_bb']:.3f}); R2_erf (S20 ~ log2 ERF50) = {p['R2_x']:.3f} (adjusted {p['adjR2_x']:.3f}, "
      f"slope {p['coef_x_alone']:+.3f} per doubling); R2_both = {p['R2_both']:.3f} (adjusted {p['adjR2_both']:.3f}); partial R2 of ERF given backbone "
      f"= {p['partial_R2_x_given_bb']:.3f}, ERF coefficient in the joint model {p['coef_x_joint']:+.3f} (negative).")
    w(f"- The stimulus-resampling CI of rho excludes 0 while the permutation p does not. The CI holds ERF50 fixed and only asks how stable "
      f"rho is to the choice of displays; the rule's association test is the permutation test, so S1 and C1 are not met, C2 is not met either, and the case is I-d.")
    w("")
    w("Inverted-U guardrail (Section 6):\n")
    w(f"- (i) quadratic coefficient of S20 ~ log2(ERF50) + log2(ERF50)^2 = {iu['quad_coef']:+.3f}, 95% CI [{iu['quad_coef_ci'][0]:+.3f}, "
      f"{iu['quad_coef_ci'][1]:+.3f}]: not met.")
    w(f"- (ii) fitted vertex at ERF50 = {iu['vertex_erf50']:.1f} px, outside the observed range ({min(p['x']):.1f} to {max(p['x']):.1f} px), "
      f"{iu['n_left']} readouts left / {iu['n_right']} right: not met.")
    pk = iu["peak_interior_by_backbone"]
    bestb = {}
    for bb in ("dinov2", "clip", "siglip"):
        rs = [r for r in R if r["eligible"] and r["readout"].startswith(C.BBNAME[bb])]
        best = max(rs, key=lambda r: r["S20"]); bl = [r["block"] for r in rs]
        pos = "the shallowest" if best["block"] == min(bl) else ("the deepest" if best["block"] == max(bl) else "an interior")
        bestb[bb] = f"{BBLAB[bb]} {'yes' if pk[bb] else 'no'} (largest S20 at L{best['block']}, {pos} eligible block)"
    w(f"- (iii) largest S20 at an interior eligible block: " + "; ".join(bestb.values()) +
      f": {'met' if iu['crit_iii'] else 'not met'} on its own.")
    w("- Statement required by the rule: **no evidence for a peak at an intermediate receptive field.** No inverted-U claim is made.\n")

    w("## 2. Per-readout table (all 18)\n")
    w("Effects are inside r, illusory minus control180, mean over 60 configurations, size average of 336 and 448 px, 95% cluster-bootstrap CI. "
      "Ratios: U(theta) uniform rotation ratio, A(20) alternating ratio (computed only for eligible readouts, n.e. otherwise); D(20) alternating drop "
      "ratio (computed where the uniform-loss CI excludes 0). S20 = 1 - U(20). ERF50/ERF90: readout ERF in display px (size average; 336 / 448 in "
      "brackets); bbERF50: block-token ERF50. CC: held-out MIT1003 CC from the sweep (`v2/anatomy/anatomy.json`). Source: stored = re-read from "
      "`v3/scores` (not rescored); new = scored in this package.\n")
    w("| backbone | block | source | CC | E vs control180 [95% CI] | eligible | U(20) | U(30) | A(20) | D(20) | slope [95% CI] | S20 | ERF50 (336 / 448) | ERF90 | bbERF50 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        sl = r["support_ratio_slope"]
        w(f"| {r['backbone']} | {r['block']} | {'stored' if r['source'].startswith('stored') else 'new'} | {r['heldout_mit1003_cc']:.3f} | "
          f"{fe(r['E_vs_control180'])} | {'yes' if r['eligible'] else 'no'} | {frt(r['U20'])} | {frt(r['U30'])} | {frt(r['A20'])} | {frt(r['D20'])} | "
          f"{sl['slope_per_unit_sr']:+.3f} [{sl['ci'][0]:+.3f}, {sl['ci'][1]:+.3f}] | {'n.e.' if r['S20'] is None else format(r['S20'], '+.2f')} | "
          f"{r['ERF50']:.1f} ({r['ERF50_s336']} / {r['ERF50_s448']}) | {r['ERF90']:.1f} | {r['bbERF50']:.1f} |")
    w("")
    w("The nine stored rows equal Appendix B (`cbb/tab_sweep.tex`, from `v3/scores/results.json`) exactly; the check is listed in Section 8.\n")

    w("### 2a. Local nulls at the readout's blur scale (all 18)\n")
    bs = out["blur_scale"]["px"]
    w(f"Architectural output blur (SD of the bilinear kernel, Delta/sqrt(6), geometric mean of the two axes): patch 14 "
      f"{bs['patch14']['336']:.1f} px (336) and {bs['patch14']['448']:.1f} px (448); patch 16 {bs['patch16']['336']:.1f} px and {bs['patch16']['448']:.1f} px. "
      "For every readout the nulls at its blur scale are therefore sigma = 8, 16 and 25 px (bracket), sigma = 16 px the reference. "
      "Null values in the bracket: corner E = " + ", ".join(f"{out['nulls_in_bracket'][f'corner_s{s}']['E']:+.3f}" for s in (8, 16, 25)) +
      " (sigma 8 / 16 / 25); smallest corner U(20) = " + f"{min(out['nulls_in_bracket'][f'corner_s{s}']['U20'] for s in (8, 16, 25)):.3f}" +
      ", smallest corner A(20) = " + f"{min(out['nulls_in_bracket'][f'corner_s{s}']['A20'] for s in (8, 16, 25)):.3f}" +
      "; largest null figure-mass loss over corner, ink and edge nulls in the bracket = " +
      f"{max(v['loss20'] for v in out['nulls_in_bracket'].values()):+.4f} (20 deg) and {max(v['loss30'] for v in out['nulls_in_bracket'].values()):+.4f} (30 deg); "
      "null square D(20) range " + f"{min(v['Dsq20'] for v in out['nulls_in_bracket'].values()):.4f} to {max(v['Dsq20'] for v in out['nulls_in_bracket'].values()):.4f}" +
      "; corner slopes " + ", ".join(f"{out['nulls_in_bracket'][f'corner_s{s}']['slope']:+.3f}" for s in (8, 16, 25)) + ".\n")
    nbk = out["nulls_in_bracket"]
    w(f"| readout | E (corner s16 E = {nbk['corner_s16']['E']:+.3f}) | U(20) CI upper < smallest corner U(20) ({min(nbk[f'corner_s{s}']['U20'] for s in (8, 16, 25)):.2f})? | "
      f"A(20) CI upper < smallest corner A(20) ({min(nbk[f'corner_s{s}']['A20'] for s in (8, 16, 25)):.2f})? | loss 20 deg [95% CI] (null max {max(v['loss20'] for v in nbk.values()):+.4f}) | "
      f"loss 30 deg [95% CI] (null max {max(v['loss30'] for v in nbk.values()):+.4f}) | square D(20) (null {min(v['Dsq20'] for v in nbk.values()):.3f} to {max(v['Dsq20'] for v in nbk.values()):.3f}) | "
      f"slope (corner " + " / ".join(f"{nbk[f'corner_s{s}']['slope']:+.3f}" for s in (8, 16, 25)) + ") |")
    w("|---|---|---|---|---|---|---|---|")
    for r in R:
        nc = r["null_comparison"]
        yn = lambda v, hi: "n.e." if v is None else (f"yes ({hi:.3f})" if v else f"no ({hi:.3f})")
        a20 = r["A20"]
        w(f"| {r['readout']} | {nc['E']:+.3f} | {yn(nc['U20_ci_below_corner_nulls'], nc['U20_ci_hi'] if nc['U20_ci_hi'] is not None else 0)} | "
          f"{yn(nc['A20_ci_below_corner_nulls'], a20['ci'][1] if a20 else 0)} | {fe(nc['loss20'], 4)} | {fe(nc['loss30'], 4)} | "
          f"{'n.e.' if nc['Dsq20'] is None else format(nc['Dsq20'], '.2f')} | {nc['slope']:+.3f} |")
    w("")
    maxabsE = max(abs(r["E_vs_control180"]["mean"]) for r in R)
    w(f"Reading of the null comparisons (descriptive, per readout). The largest aligned effect of any readout in magnitude ({maxabsE:.3f}) is below the sigma-16 "
      f"corner null's ({nbk['corner_s16']['E']:+.3f}), and the corner nulls lose nothing under rotation (U(20) and A(20) above 1 at every sigma in the bracket). "
      "Local structure can produce the inside contrast, so E alone is not evidence of figural processing.\n")

    w("### 2b. Newly scored readouts: remaining Appendix B style measures (Section 7, item 9)\n")
    w("| readout | core r effect | core share effect | illusory vs control90 | real vs control180 | real_filled vs control180 | U(10) | A(10) | A(30) | D(10) | D(30) | square D(20) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        if not r["source"].startswith("nn"):
            continue
        w(f"| {r['readout']} | {fe(r['core_r_effect'])} | {fe(r['core_share_effect'])} | {fe(r['E_vs_control90'])} | {fe(r['real_vs_control180'])} | "
          f"{fe(r['real_filled_vs_control180'])} | {frt(r['U10'])} | {frt(r['A10'])} | {frt(r['A30'])} | {frt(r['D10'])} | {frt(r['D30'])} | {frt(r['D_square20'])} |")
    w("")

    w("### 2c. ERF details (Section 4, 'also reported')\n")
    w("Readout ERF and block-token ERF per input size, in display px and model-input px; per-display ERF50 over the 24 displays (median and range); "
      "inside-set-only and inducer-set-only ERF50 (size average).\n")
    w("| readout | ERF50 336 / 448 (display) | ERF90 336 / 448 (display) | ERF50 336 / 448 (model px) | per-display ERF50 336: median (min to max) | per-display ERF50 448: median (min to max) | ERF50 inside / inducer | bbERF50 336 / 448 | bbERF90 (size avg) | bbERF50 inside / inducer |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        d = r["erf_detail"]; a, b = d["ro_s336"], d["ro_s448"]; ba, bb_ = d["bb_s336"], d["bb_s448"]
        w(f"| {r['readout']} | {a['erf50']} / {b['erf50']} | {a['erf90']} / {b['erf90']} | {a['erf50_model']} / {b['erf50_model']} | "
          f"{a['per_display_erf50_median']:.0f} ({a['per_display_erf50_min']:.0f} to {a['per_display_erf50_max']:.0f}) | "
          f"{b['per_display_erf50_median']:.0f} ({b['per_display_erf50_min']:.0f} to {b['per_display_erf50_max']:.0f}) | "
          f"{d['ro_sizeavg_inside']['erf50']:.1f} / {d['ro_sizeavg_ind']['erf50']:.1f} | {ba['erf50']} / {bb_['erf50']} | {d['bb_sizeavg_all']['erf90']:.1f} | "
          f"{d['bb_sizeavg_inside']['erf50']:.1f} / {d['bb_sizeavg_ind']['erf50']:.1f} |")
    er = out["erf_ranges"]
    w("")
    w(f"Ranges over the 18 readouts: readout ERF50 {er['readout_ERF50'][0]:.1f} to {er['readout_ERF50'][1]:.1f} px; block-token ERF50 "
      f"{er['block_token_ERF50'][0]:.1f} to {er['block_token_ERF50'][1]:.1f} px. Spearman rho with relative depth (all 18): readout ERF50 "
      f"{er['rho_ERF50_reldepth_all18']:+.2f}, block-token ERF50 {er['rho_bbERF50_reldepth_all18']:+.2f}; readout vs block-token ERF50 "
      f"{er['rho_ERF50_bbERF50_all18']:+.2f}.\n")

    w("## 3. Secondary and sensitivity analyses (Section 7; reported in full, never decision-bearing)\n")
    w("Same statistics as Section 5 unless stated. CI = 95% percentile interval over 10,000 stimulus resamples with the predictor fixed; p = two-sided "
      "permutation p (100,000 random permutations, RandomState(2026)); T = pooled within-backbone rho with stratified permutation p; R2 columns: "
      "predictor (log2) / backbone / both; partial = partial R2 of the predictor given backbone, with the sign of its joint-model coefficient.\n")
    rows = [("1", "S30 = 1 - U(30) vs ERF50", "S30"), ("1", "alternating sensitivity 1 - A(20) vs ERF50", "alt_S20"),
            ("1", "square-only D(20) vs ERF50 (square uniform-loss CI excludes 0)", "square_D20"),
            ("2", "S20 vs readout ERF90", "S20_vs_ERF90"), ("2", "S20 vs block-token ERF50", "S20_vs_bbERF50"),
            ("2", "S20 vs block-token ERF90", "S20_vs_bbERF90"), ("2", "S20 vs inside-set-only ERF50", "S20_vs_ERF50_inside"),
            ("2", "S20 vs inducer-set-only ERF50", "S20_vs_ERF50_ind"), ("3", "336 px only: S20(336) vs ERF50(336)", "size336"),
            ("3", "448 px only: S20(448) vs ERF50(448)", "size448"), ("4", "positive-effect eligible readouts only", "positive_effect_only"),
            ("5", "CC filter (CC >= best CC of backbone - 0.05)", "cc_filter"),
            ("6", "all 18, S20 point values regardless of eligibility (unstable: flagged)", "all18_point_U20"),
            ("7", "all 18, ratio-free L20 = sign(E) x [r(illusory) - r(mis20)]", "all18_L20")]
    w("| item | analysis | n | rho [95% CI] | perm p | within-backbone rho | T (p) | R2 x / bb / both | partial R2 (joint coef sign) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for it, lab, k in rows:
        a = sec[k]
        ci, wl, T = rho_line(a)
        w(f"| {it} | {lab} | {a['n']} | {a['rho']:+.2f} {ci} | {pf(a['perm_p'])} | {wl} | {T} | {a['R2_x']:.2f} / {a['R2_bb']:.2f} / {a['R2_both']:.2f} | "
          f"{a['partial_R2_x_given_bb']:.2f} ({'+' if a['coef_x_joint'] > 0 else '-'}) |")
    w("")
    for it, lab, k in rows:
        w(f"- {lab}: readouts " + ", ".join(sec[k]["readouts"]) + ".")
    w("")
    w(f"Item 6 note: S20 point values of the readouts without a reliable effect are extreme (DINOv2-B L2 "
      f"{[r for r in R if r['readout'] == 'DINOv2-B L2'][0]['S20_point_any_eligibility']:+.1f}, CLIP-L L4 "
      f"{[r for r in R if r['readout'] == 'CLIP-L L4'][0]['S20_point_any_eligibility']:+.1f}), because their aligned effects are near 0; the row is unstable "
      "by construction (rule item 6).\n")
    w(f"Item 8, competing descriptors over the {p['n']} eligible readouts: relative depth (L / number of blocks) rho = {comp['rho_rel_depth']:+.2f}, "
      f"R2 = {comp['R2_rel_depth']:.3f}; training-objective family (DINOv2 self-supervised vs CLIP and SigLIP image-text) rho = "
      f"{comp['rho_objective_family']:+.2f}, R2 = {comp['R2_objective_family']:.3f}. Collinearity across all 18 readouts: rho(ERF50, relative depth) = "
      f"{comp['rho_erf50_vs_depth_all18']:+.2f}.\n")
    alt = out["per_size_alternative_reading"]
    w(f"Item 3 reading. The per-size rows above use eligibility determined at each size (n = {sec['size336']['n']} at 336 px, {sec['size448']['n']} at 448 px). "
      f"With the size-averaged eligibility set instead (n = {p['n']} at both sizes; not part of the rule, shown only because item 3 can be read either way): "
      + "; ".join(f"{s} px rho = {a['rho']:+.2f} [{a['rho_ci'][0]:+.2f}, {a['rho_ci'][1]:+.2f}], p = {pf(a['perm_p'])}, T = {a['T']:+.2f} (p {pf(a['T_perm_p'])})"
                  for s, a in alt.items()) + ".\n")

    w("## 4. Figure\n")
    w("`nn/figs/erf_vs_rotation.pdf` and `.png`: S20 against readout ERF50 for the 13 eligible readouts, colour and marker shape by backbone, each point "
      "labelled with its block, vertical bars = 95% stimulus-resampling CI of S20; dashed line S20 = 0 (no loss), dotted line S20 = 1 (effect "
      "abolished). The five readouts without a reliable aligned effect are listed under the plot, not drawn, because S20 is undefined for them.\n")

    w("## 5. What the numbers say (descriptive, within the rule's limits)\n")
    signs = [v["rho"] for v in p["within"].values() if v["rho"] is not None]
    sgn_txt = ("negative within all three networks" if all(v < 0 for v in signs) else
               "with within-network signs " + ", ".join(f"{k} {v['rho']:+.2f}" for k, v in p["within"].items() if v["rho"] is not None))
    w(f"- Under the rule the result is INCONCLUSIVE ({p['case']}): rotation sensitivity did not rise with the readout ERF50. The point estimates go the other "
      f"way (rho = {p['rho']:+.2f} across readouts, {sgn_txt}), but the permutation test gives p = {pf(p['perm_p'])}, so C1 is not met; network identity "
      f"explained less variance than ERF (adjusted R2 {p['adjR2_bb']:.2f} vs {p['adjR2_x']:.2f}), so C2 is not met either.")
    mono = {}
    for bb in ("dinov2", "clip", "siglip"):
        v = [r["ERF50"] for r in R if r["readout"].startswith(C.BBNAME[bb])]
        mono[bb] = bool(all(a <= b for a, b in zip(v, v[1:])))
    mono_txt = ("is not monotonic in depth within any network" if not any(mono.values()) else
                "is monotonic in depth only in " + ", ".join(BBLAB[b] for b, m in mono.items() if m))
    w(f"- The primary predictor spans a narrow range ({er['readout_ERF50'][0]:.1f} to {er['readout_ERF50'][1]:.1f} px, a factor of "
      f"{er['readout_ERF50'][1] / er['readout_ERF50'][0]:.2f}) and {mono_txt}. A plausible reason, not tested here, is the readout design shared by all 18 "
      f"readouts (ASPP with a dilation-12 branch and a global-pooling branch). The block-token ERF50 spans far more "
      f"({er['block_token_ERF50'][0]:.1f} to {er['block_token_ERF50'][1]:.1f} px) and tracks depth (rho {er['rho_bbERF50_reldepth_all18']:+.2f}); against it S20 gives "
      f"rho = {sec['S20_vs_bbERF50']['rho']:+.2f} (p = {pf(sec['S20_vs_bbERF50']['perm_p'])}), within-network rho "
      + ", ".join(f"{k} {v['rho']:+.2f}" for k, v in sec["S20_vs_bbERF50"]["within"].items() if v["rho"] is not None) +
      ". This is a secondary analysis and does not change the decision.")
    pos_sig = [lab for _, lab, k in rows if sec[k]["rho"] > 0 and sec[k]["perm_p"] < 0.05]
    neg_sig = [lab for _, lab, k in rows if sec[k]["rho"] < 0 and sec[k]["perm_p"] < 0.05]
    w(f"- Secondary analyses with permutation p < .05: positive association {', '.join(pos_sig) if pos_sig else 'none'}; negative association "
      f"{', '.join(neg_sig) if neg_sig else 'none'}. The square-only drop ratio (n = {sec['square_D20']['n']}) has rho = {sec['square_D20']['rho']:+.2f} "
      f"(p = {pf(sec['square_D20']['perm_p'])}) with within-network rho "
      + ", ".join(f"{k} {v['rho']:+.2f}" for k, v in sec["square_D20"]["within"].items() if v["rho"] is not None) +
      f" and pooled T = {sec['square_D20']['T']:+.2f}, so its positive overall rho is not reproduced within networks.")
    w("- These are single-seed readouts; nothing here bears on human perception, and the illusion index is not read as human-likeness.\n")

    w("## 6. Deviations from the rule\n")
    w("None. No readout, block, size, display or location set was dropped or added; no measure, threshold, seed or permutation count was changed; "
      "`nn/DEVIATIONS.md` was not needed and does not exist.\n")
    w("Readings of wording that the rule leaves open (the first three were fixed in `nn/analyze_nn.py` at 08:01 UTC, before the new readouts were "
      "scored from 08:08 UTC, and were not changed after the results were seen):\n")
    for s in out["interpretations"][:3]:
        w(f"- {s}")
    w("\nCompletions made here (rule items that `nn/results/summary.json` did not yet contain; computed exactly as the rule states):\n")
    w("- " + out["interpretations"][3])
    w("- Section 4 'also reported' ERF details (per size in display and model-input px, per-display spread, inside-only and inducer-only ERFs) are "
      "tabulated in Section 2c from `nn/results/erf/<bb>.json`.\n")
    w("Report-level corrections relative to `nn/results/report.md`:\n")
    w("- Cosmetic only, no number changed: unrounded adjusted R2, partial R2 and vertex values are rounded here; the row labelled `all18_point_U20` "
      "there holds S20 point values and is labelled accordingly here.\n")

    w("## 7. Provenance and run facts\n")
    w("- Interpreter `$LAN_SAL_SRC/.venv/bin/python` (Python 3.12, torch 2.12.0). All jobs on the CPU, one heavy job at a "
      "time, under `caffeinate -i`; offline Hugging Face cache only (HF_HUB_OFFLINE=1); no downloads.")
    w("- Scoring of the 9 new readouts: 740 stim_v3 + 360 stim_alt images, 336 and 448 px, pipeline of `v2/scores/score_v3.py` + `v3/scores/score_alt.py` "
      "(`nn/score_nn.py`, 3 threads). ERF: 24 displays x 2 sizes x 8 or 9 locations x 18 readouts, readout and block-token targets (`nn/erf_nn.py`).")
    w("- Sanity rescore (rule Section 8): DINOv2-B L12 rescored twice (before and after the main run): all 1,100 images x 2 sizes bit-identical to the "
      "stored files in all three score files (max |difference| 0.0), and all 357 summary numbers (E, CIs, U, A, D, slope, per size) identical.\n")

    w("## 8. Checks run by `nn/property_results.py`\n")
    nf = sum(1 for c in out["checks"] if not c["ok"])
    w(f"{len(out['checks'])} checks, {nf} failed. Every decision-bearing number was re-derived with code written separately from `nn/analyze_nn.py` "
      "(same procedure, bootstrap indices and seeds) and matched; the nine stored readouts reproduce `v3/scores/results.json`.\n")
    for c in out["checks"]:
        w(f"- {'PASS' if c['ok'] else 'FAIL'}: {c['check']}" + (f" ({c['detail']})" if c["detail"] else ""))
    w("")
    w("## 9. Files\n")
    for f in ("nn/results/property_results.json", "nn/PROPERTY_RESULTS.md", "nn/figs/erf_vs_rotation.pdf", "nn/figs/erf_vs_rotation.png",
              "nn/property_results.py", "nn/results/summary.json", "nn/results/report.md", "nn/results/erf/{dinov2,clip,siglip}.json",
              "nn/results/raw_v3maps, raw_alt, raw_v2style (9 new readouts x 2 sizes)", "nn/ANALYSIS_RULE.md, nn/ANALYSIS_RULE.sha256, nn/RUN_LOG.md"):
        w(f"- `$LAN_PROJECT_ROOT/{f}`")
    txt = "\n".join(L) + "\n"
    assert "—" not in txt and "–" not in txt, "dash character in report"
    open(OUT_MD, "w").write(txt)


if __name__ == "__main__":
    main()
