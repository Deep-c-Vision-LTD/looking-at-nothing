"""
Analysis of nn/ANALYSIS_RULE.md (rotation sensitivity vs effective receptive field over the 18 layer-sweep readouts).

Inputs (read only): stored scores of the 9 scored readouts (v3/scores/raw_v3maps, v3/scores/raw), new scores of the 9
unscored readouts (nn/results/raw_v3maps, nn/results/raw_alt), ERFs (nn/results/erf/<bb>.json), held-out CC
(v2/anatomy/anatomy.json), local nulls (v3/scores/results.json, null entries). Measures, effects, ratios and bootstrap
come from v3/scores/analyze_alt (imported unchanged; v2 resample indices, 10,000 resamples, seed 2026).
Outputs: nn/results/summary.json and nn/results/report.md.
Usage: python analyze_nn.py [--allow-missing]   (--allow-missing: analyse whatever is present; for dry runs only)
Code test only: NN_FAKE_ERF=1 replaces every ERF by a seeded random number and NN_ANALYSIS_OUT=<dir inside nn/> redirects
the outputs; such a run carries no information and is never a result.
"""
import os, sys, json, time, itertools
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nn as C                                    # noqa: E402
CA, AA = C.import_v3()
DA, DS = AA.DES["all"], AA.DES["square"]
B = AA.B
NPERM, SEED = 100_000, 2026
OUTDIR = os.path.abspath(os.environ.get("NN_ANALYSIS_OUT", C.RES))
assert OUTDIR.startswith(C.NN), OUTDIR
FAKE = os.environ.get("NN_FAKE_ERF") == "1"


# ------------------------------------------------------------------ data
def parts_for(bb, L, source):
    """source 'stored' = v3/scores; otherwise a results directory with raw_v3maps/ and raw_alt/."""
    parts = []
    for s in C.SIZES:
        t = C.tag(bb, L, s)
        if source == "stored":
            v3, al = C.load_json(os.path.join(C.V3SC, "raw_v3maps", t + ".json")), C.load_json(os.path.join(C.V3SC, "raw", t + ".json"))
        else:
            v3, al = C.load_json(os.path.join(source, "raw_v3maps", t + ".json")), C.load_json(os.path.join(source, "raw_alt", t + ".json"))
        if v3 is None or al is None or len(v3) != 480 or len(al) != 360:
            return None
        parts.append(AA.with_alt({**v3, **al}))
    return parts


def X(parts, cond, m="r_inside", D=DA):
    return AA.arr(D, parts, cond, m)


def ratio_pt(parts, num_cond, D=DA, m="r_inside", den=("illusory", "control180"), num_ref="control180"):
    a = X(parts, num_cond, m, D) - X(parts, num_ref, m, D); b = X(parts, den[0], m, D) - X(parts, den[1], m, D)
    return float(a.mean() / b.mean()), D.bmeans(a) / D.bmeans(b)


def sens(parts, cond, D=DA):
    """1 - ratio(cond vs control180 / illusory vs control180): point value and bootstrap vector."""
    v, bv = ratio_pt(parts, cond, D)
    return 1 - v, 1 - bv


def drop_ratio(parts, theta, D):
    num = X(parts, "illusory", D=D) - X(parts, f"alt{theta}", D=D); den = X(parts, "illusory", D=D) - X(parts, f"mis{theta}", D=D)
    return float(num.mean() / den.mean()), D.bmeans(num) / D.bmeans(den)


def load_erf():
    out = {}
    if FAKE:                                              # code test only: synthetic ERFs, no information
        rng = np.random.RandomState(12345)
        for bb, L in C.all_readouts():
            v = {k: float(rng.uniform(20, 300)) for k in ("a", "b", "c", "d", "e", "f", "g")}
            out[(bb, L)] = {f"{kind}_sizeavg_{st}": dict(erf50=v["a"] + i, erf90=v["b"] + i)
                            for i, (kind, st) in enumerate((k, t) for k in ("ro", "bb") for t in ("all", "inside", "ind"))}
            out[(bb, L)]["ro_s336_all"] = dict(erf50=int(v["c"])); out[(bb, L)]["ro_s448_all"] = dict(erf50=int(v["d"]))
        return out
    for bb in C.LAYERS:
        p = os.path.join(C.RES, "erf", f"{bb}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            for L, rec in d["layers"].items():
                out[(bb, int(L))] = rec
    return out


# ------------------------------------------------------------------ statistics
def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y):
    n = len(x); rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum()); ry = (ry - ry.mean()) / np.sqrt(((ry - ry.mean()) ** 2).sum())
    obs = float(rx @ ry)
    if n <= 9:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64); kind = f"exact ({len(perms)} permutations)"
    else:
        rng = np.random.RandomState(SEED); perms = np.argsort(rng.rand(NPERM, n), 1); kind = f"{NPERM} random permutations"
    null = ry[perms] @ rx
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), kind


def boot_rho(x, Yb):
    """x: (n,), Yb: (n, B) bootstrap values; Spearman rho per resample."""
    rx = rankdata(x); rx = rx - rx.mean()
    ry = np.apply_along_axis(rankdata, 0, Yb); ry = ry - ry.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx @ ry) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum(0))
    return r


def ols_r2(y, cols):
    Xm = np.column_stack([np.ones(len(y))] + cols) if cols else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ beta; sst = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / sst) if sst > 0 else float("nan"), beta


def adj(r2, n, k):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else None


def assoc(names, x, y, yb=None, groups=None):
    """Section 5 statistics for predictor x and outcome y over readouts 'names' (lists aligned)."""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(names)
    out = dict(n=n, readouts=names, x=[round(v, 3) for v in x], y=[round(v, 4) for v in y])
    if n < 3:
        out["note"] = "fewer than 3 readouts"; return out
    out["rho"] = spearman(x, y)
    out["perm_p"], out["perm_kind"] = perm_p(x, y)
    if yb is not None:
        rb = boot_rho(x, yb)
        out["rho_ci"] = [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
        out["rho_boot_frac_pos"] = float(np.nanmean(rb > 0))
    groups = groups if groups is not None else [nm.split()[0] for nm in names]
    G = sorted(set(groups)); qual = [g for g in G if groups.count(g) >= 3]
    out["within"] = {}
    for g in G:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out["within"][g] = dict(n=len(idx), rho=(spearman(x[idx], y[idx]) if len(idx) >= 3 else None))
    if qual:
        idxs = [np.array([i for i, gg in enumerate(groups) if gg == g]) for g in qual]
        T = sum(len(ii) * spearman(x[ii], y[ii]) for ii in idxs) / sum(len(ii) for ii in idxs)
        rng = np.random.RandomState(SEED); null = np.zeros(NPERM)
        for ii in idxs:                                   # within-backbone permutation of x (ranks are permutation-equivariant)
            rx, ry = rankdata(x[ii]), rankdata(y[ii]); rx = rx - rx.mean(); ry = ry - ry.mean()
            den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
            perms = np.argsort(rng.rand(NPERM, len(ii)), 1)
            null += len(ii) * ((rx[perms] @ ry) / den if den > 0 else 0.0)
        null /= sum(len(ii) for ii in idxs)
        out["T"] = float(T); out["T_perm_p"] = float(np.mean(np.abs(null) >= abs(T) - 1e-12)); out["T_backbones"] = qual
        out["n_qual_backbones_rho_pos"] = int(sum(1 for g in qual if out["within"][g]["rho"] is not None and out["within"][g]["rho"] > 0))
    else:
        out["T"] = None; out["T_perm_p"] = None; out["T_backbones"] = []; out["n_qual_backbones_rho_pos"] = 0
    lx = np.log2(x)
    dummies = [np.array([1.0 if gg == g else 0.0 for gg in groups]) for g in G[1:]]
    r2_bb, _ = ols_r2(y, dummies); r2_erf, b1 = ols_r2(y, [lx]); r2_both, b2 = ols_r2(y, [lx] + dummies)
    out["R2_bb"], out["R2_x"], out["R2_both"] = r2_bb, r2_erf, r2_both
    out["adjR2_bb"], out["adjR2_x"], out["adjR2_both"] = adj(r2_bb, n, len(dummies)), adj(r2_erf, n, 1), adj(r2_both, n, 1 + len(dummies))
    out["partial_R2_x_given_bb"] = float((r2_both - r2_bb) / (1 - r2_bb)) if r2_bb < 1 else None
    out["coef_x_joint"] = float(b2[1]); out["coef_x_alone"] = float(b1[1])
    return out


# ------------------------------------------------------------------ decision (Section 6)
def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}"


def p3(v):
    return "n/a" if v is None else ("< .001" if v < 0.001 else f"{v:.3f}".lstrip("0") if v < 1 else "1.000")


def decide(a):
    n = a["n"]
    if n < 8:
        return "INCONCLUSIVE", "I-a", (f"Only {n} of the 18 readouts had an aligned effect whose interval excluded zero, too few to test whether "
                                       "rotation sensitivity follows the receptive field; we report the values descriptively."), {}
    rho, lo, p, T, pT = a["rho"], a["rho_ci"][0], a["perm_p"], a["T"], a["T_perm_p"]
    S1 = rho >= 0.50 and p < 0.05 and lo > 0
    S2 = (T is not None and T > 0 and pT is not None and pT < 0.10 and len(a["T_backbones"]) >= 2 and a["n_qual_backbones_rho_pos"] >= 2)
    ae, ab = a["adjR2_x"], a["adjR2_bb"]
    S3 = (ae is not None and ab is not None and ae >= ab and a["coef_x_joint"] > 0 and a["partial_R2_x_given_bb"] is not None
          and a["partial_R2_x_given_bb"] >= 0.25)
    C1 = rho < 0 and p < 0.05
    C2 = (rho <= 0.10 and (T is None or T <= 0) and ae is not None and ab is not None and ab - ae >= 0.20)
    flags = dict(S1=bool(S1), S2=bool(S2), S3=bool(S3), C1=bool(C1), C2=bool(C2))
    ci = f"[{a['rho_ci'][0]:+.2f}, {a['rho_ci'][1]:+.2f}]"
    Ts = "n/a" if T is None else f"{T:+.2f}"
    if S1 and S2 and S3:
        txt = (f"Across the {n} layer-sweep readouts with a reliable aligned effect, loss of the Kanizsa effect under a 20 deg inducer rotation "
               f"increased with the effective receptive field of the readout (Spearman rho = {rho:+.2f}, 95% CI {ci}, permutation p = {p3(p)}), "
               f"within networks (pooled within-network rho = {Ts}, p = {p3(pT)}) as well as across them, and the receptive field accounted for "
               f"at least as much variance as network identity (adjusted R2 {ae:.2f} vs {ab:.2f}). In this single-seed sweep, rotation sensitivity "
               "therefore follows the spatial extent over which the readout integrates more closely than the network it reads, as the "
               "receptive-field account predicts. Within each network receptive field and depth rise together, so these data do not separate "
               "receptive field from depth.")
        return "SUPPORTS", "S", txt, flags
    if C1 or C2:
        txt = (f"Loss of the Kanizsa effect under a 20 deg inducer rotation did not increase with the effective receptive field of the readout "
               f"(Spearman rho = {rho:+.2f}, 95% CI {ci}, permutation p = {p3(p)}; pooled within-network rho = {Ts}); network identity "
               f"accounted for more of its variance (adjusted R2 {ab if ab is not None else float('nan'):.2f} vs {ae if ae is not None else float('nan'):.2f}). "
               "The layer sweep therefore does not support the proposal that rotation sensitivity follows the receptive field of the block read "
               "rather than the network.")
        if C1:
            txt += " Rotation sensitivity was larger for readouts with smaller receptive fields."
        return "CONTRADICTS", "C1" if C1 else "C2", txt, flags
    if S1 and not S2:
        return "INCONCLUSIVE", "I-b", (f"Rotation sensitivity rose with receptive field across readouts (rho = {rho:+.2f}) but not consistently within "
                                       f"networks (pooled within-network rho = {Ts}, p = {p3(pT)}), so differences between networks, which confound "
                                       "training objective, data and architecture, could carry the association."), flags
    if S1 and S2 and not S3:
        return "INCONCLUSIVE", "I-c", ("Rotation sensitivity rose with receptive field across and within networks, but the receptive field did not "
                                       f"explain more of its variance than network identity (adjusted R2 {ae:.2f} vs {ab:.2f})."), flags
    return "INCONCLUSIVE", "I-d", (f"The layer sweep neither supports nor rules out the receptive-field account (rho = {rho:+.2f}, 95% CI {ci}, "
                                   f"p = {p3(p)}; pooled within-network rho = {Ts}; adjusted R2 for receptive field "
                                   f"{'n/a' if ae is None else f'{ae:.2f}'} and for network identity {'n/a' if ab is None else f'{ab:.2f}'})."), flags


def inverted_u(names, x, y, yb, groups, blocks):
    lx = np.log2(np.asarray(x, float)); y = np.asarray(y, float)
    Xm = np.column_stack([np.ones_like(lx), lx, lx ** 2])
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    bb_ = np.linalg.lstsq(Xm, yb, rcond=None)[0]                     # (3, B)
    c_ci = [float(np.percentile(bb_[2], 2.5)), float(np.percentile(bb_[2], 97.5))]
    vertex = float(-beta[1] / (2 * beta[2])) if beta[2] != 0 else None
    left = int(np.sum(lx < vertex)) if vertex is not None else 0; right = int(np.sum(lx > vertex)) if vertex is not None else 0
    c1 = beta[2] < 0 and c_ci[1] < 0
    c2 = vertex is not None and lx.min() < vertex < lx.max() and left >= 2 and right >= 2
    interior = {}
    for g in sorted(set(groups)):
        ii = [i for i, gg in enumerate(groups) if gg == g]
        if len(ii) < 3:
            interior[g] = False; continue
        bl = [blocks[i] for i in ii]; best = ii[int(np.argmax(y[ii]))]
        interior[g] = bool(min(bl) < blocks[best] < max(bl))
    c3 = sum(interior.values()) >= 2
    return dict(quad_coef=float(beta[2]), quad_coef_ci=c_ci, vertex_log2_erf50=vertex, vertex_erf50=(2 ** vertex if vertex is not None else None),
                n_left=left, n_right=right, peak_interior_by_backbone=interior, crit_i=bool(c1), crit_ii=bool(c2), crit_iii=bool(c3),
                allowed=bool(c1 and c2 and c3),
                wording=("an inverted-U statement is allowed by Section 6" if (c1 and c2 and c3)
                         else "no evidence for a peak at an intermediate receptive field"))


# ------------------------------------------------------------------ per-readout records
NULLS_BRACKET = (8, 16, 25)


def readout_record(bb, L, parts, cc):
    res = AA.analyse(parts)
    o = res["r_inside"]["all"]
    rec = dict(backbone=bb, block=L, rel_depth=L / C.NBLOCKS[bb], cc=cc, source=None,
               E=o["aligned"], eligible=bool(AA.excl0(o["aligned"])),
               U={t: (o["ratio_uni"][t] if o["ratios_computed"] else None) for t in (10, 20, 30)},
               A={t: (o["ratio_alt"][t] if o["ratios_computed"] else None) for t in (10, 20, 30)},
               D={t: o["drop_ratio"][t] for t in (10, 20, 30)},
               D_square={t: res["r_inside"]["square"]["drop_ratio"][t] for t in (10, 20, 30)},
               loss_uni={t: o["drop_uni"][t] for t in (10, 20, 30)},
               slope=o["sr_slope_vs_control180"],
               E_control90=o["effects_vs_control90"]["illusory"],
               real=o["effects_vs_control180"]["real"], real_filled=o["effects_vs_control180"]["real_filled"],
               core_r=res["r_core30"]["all"]["aligned"], core_share=res["f_core30"]["all"]["aligned"])
    rec["S20_point_any"] = sens(parts, "mis20")[0]                # point value regardless of eligibility (secondary 6)
    sgn = np.sign(o["aligned"]["mean"]) or 1.0
    l20 = X(parts, "illusory") - X(parts, "mis20")
    rec["L20"] = float(sgn * l20.mean())
    return rec, res


def null_table(v3res):
    out = {}
    for kind in ("corner", "ink", "edge"):
        for s in (8, 16, 25, 40, 60):
            o = v3res["models"][f"null {kind}_s{s}"]["results"]["r_inside"]
            out[f"{kind}_s{s}"] = dict(E=o["all"]["aligned"]["mean"], ratios_computed=o["all"]["ratios_computed"],
                                       U20=(o["all"]["ratio_uni"]["20"]["value"] if o["all"]["ratios_computed"] else None),
                                       A20=(o["all"]["ratio_alt"]["20"]["value"] if o["all"]["ratios_computed"] else None),
                                       loss20=o["all"]["drop_uni"]["20"]["mean"], loss30=o["all"]["drop_uni"]["30"]["mean"],
                                       Dsq20=(o["square"]["drop_ratio"]["20"]["value"] if o["square"]["drop_ratio"]["20"] else None),
                                       slope=o["all"]["sr_slope_vs_control180"]["slope_per_unit_sr"])
    return out


def null_compare(rec, nt):
    br = [f"corner_s{s}" for s in NULLS_BRACKET]; allb = [f"{k}_s{s}" for k in ("corner", "ink", "edge") for s in NULLS_BRACKET]
    minU = min(nt[k]["U20"] for k in br if nt[k]["U20"] is not None); minA = min(nt[k]["A20"] for k in br if nt[k]["A20"] is not None)
    U20, A20 = rec["U"][20], rec["A"][20]
    dsq = [nt[k]["Dsq20"] for k in allb if nt[k]["Dsq20"] is not None]
    return dict(E=rec["E"]["mean"], corner16_E=nt["corner_s16"]["E"], corner_bracket_E=[nt[k]["E"] for k in br],
                U20=None if U20 is None else U20["value"], U20_ci_hi=None if U20 is None else U20["ci"][1], min_corner_U20_bracket=minU,
                U20_ci_below_corner_nulls=None if U20 is None else bool(U20["ci"][1] < minU),
                A20=None if A20 is None else A20["value"], min_corner_A20_bracket=minA,
                A20_ci_below_corner_nulls=None if A20 is None else bool(A20["ci"][1] < minA),
                loss20=rec["loss_uni"][20], loss30=rec["loss_uni"][30],
                max_null_loss20_bracket=max(nt[k]["loss20"] for k in allb), max_null_loss30_bracket=max(nt[k]["loss30"] for k in allb),
                Dsq20=None if rec["D_square"][20] is None else rec["D_square"][20]["value"],
                null_Dsq20_range=[min(dsq), max(dsq)] if dsq else None,
                slope=rec["slope"]["slope_per_unit_sr"], slope_ci=rec["slope"]["ci"],
                corner_slopes_bracket=[nt[k]["slope"] for k in br])


# ------------------------------------------------------------------ main
def main(allow_missing=False):
    ok, h, recd = C.rule_ok()
    t0 = time.time()
    cc = C.heldout_cc(); erf = load_erf()
    v3res = json.load(open(os.path.join(C.V3SC, "results.json")))
    nt = null_table(v3res)
    recs, parts_all, missing = {}, {}, []
    for bb, L in C.all_readouts():
        src = "stored" if L in C.SCORED[bb] else C.RES
        parts = parts_for(bb, L, src)
        if parts is None or (bb, L) not in erf:
            missing.append(C.rname(bb, L) + (" (scores)" if parts is None else " (ERF)"))
            if parts is None:
                continue
        rec, res = readout_record(bb, L, parts, cc[(bb, L)])
        rec["source"] = "stored v3/scores" if src == "stored" else "nn/results (new)"
        if (bb, L) in erf:
            e = erf[(bb, L)]
            rec["ERF50"] = e["ro_sizeavg_all"]["erf50"]; rec["ERF90"] = e["ro_sizeavg_all"]["erf90"]
            rec["ERF50_inside"] = e["ro_sizeavg_inside"]["erf50"]; rec["ERF50_ind"] = e["ro_sizeavg_ind"]["erf50"]
            rec["bbERF50"] = e["bb_sizeavg_all"]["erf50"]; rec["bbERF90"] = e["bb_sizeavg_all"]["erf90"]
            rec["ERF50_s336"] = e["ro_s336_all"]["erf50"]; rec["ERF50_s448"] = e["ro_s448_all"]["erf50"]
        rec["per_size"] = {}
        for k, s in enumerate(C.SIZES):
            rs = AA.analyse([parts[k]])["r_inside"]["all"]
            rec["per_size"][str(s)] = dict(E=rs["aligned"], eligible=bool(AA.excl0(rs["aligned"])),
                                           U20=(rs["ratio_uni"][20] if rs["ratios_computed"] else None))
        rec["null_comparison"] = null_compare(rec, nt)
        recs[C.rname(bb, L)] = rec; parts_all[C.rname(bb, L)] = parts
        print(f"{C.rname(bb, L)}: E {rec['E']['mean']:+.4f} eligible {rec['eligible']} U20 "
              f"{None if rec['U'][20] is None else rec['U'][20]['value']} ERF50 {rec.get('ERF50')}", flush=True)
    if missing and not allow_missing:
        raise SystemExit(f"missing inputs: {missing}")
    have = [n for n in recs if "ERF50" in recs[n]]
    grp = lambda names: [recs[n]["backbone"] for n in names]

    def S_vec(names, cond):
        pts, bs = zip(*[sens(parts_all[n], cond) for n in names]); return list(pts), np.vstack(bs)

    out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), rule_sha256_now=h, rule_sha256_recorded=recd, rule_unchanged=ok,
               missing=missing, readouts=recs, nulls=nt)
    # primary
    elig = [n for n in have if recs[n]["eligible"]]
    prim = None
    if len(elig) >= 3:
        y, yb = S_vec(elig, "mis20")
        prim = assoc(elig, [recs[n]["ERF50"] for n in elig], y, yb, grp(elig))
        cat, case, txt, flags = decide(prim)
        prim.update(decision=cat, case=case, wording=txt, criteria=flags)
        prim["inverted_u"] = inverted_u(elig, [recs[n]["ERF50"] for n in elig], y, yb, grp(elig), [recs[n]["block"] for n in elig])
    else:
        prim = dict(n=len(elig), decision="INCONCLUSIVE", case="I-a",
                    wording=f"Only {len(elig)} of the 18 readouts had an aligned effect whose interval excluded zero, too few to test whether "
                            "rotation sensitivity follows the receptive field; we report the values descriptively.")
    out["primary"] = prim
    # secondary
    sec = {}
    if len(elig) >= 3:
        for lab, cond in (("S30", "mis30"), ("alt_S20", "alt20")):
            y, yb = S_vec(elig, cond); sec[lab] = assoc(elig, [recs[n]["ERF50"] for n in elig], y, yb, grp(elig))
        sq = [n for n in have if recs[n]["D_square"][20] is not None]
        if len(sq) >= 3:
            pts, bs = zip(*[drop_ratio(parts_all[n], 20, DS) for n in sq])
            sec["square_D20"] = assoc(sq, [recs[n]["ERF50"] for n in sq], list(pts), np.vstack(bs), grp(sq))
        y, yb = S_vec(elig, "mis20")
        for key in ("ERF90", "bbERF50", "bbERF90", "ERF50_inside", "ERF50_ind"):
            sec[f"S20_vs_{key}"] = assoc(elig, [recs[n][key] for n in elig], y, yb, grp(elig))
        pos = [n for n in elig if recs[n]["E"]["mean"] > 0]
        if len(pos) >= 3:
            y2, yb2 = S_vec(pos, "mis20"); sec["positive_effect_only"] = assoc(pos, [recs[n]["ERF50"] for n in pos], y2, yb2, grp(pos))
        best = {bb: max(recs[n]["cc"] for n in have if recs[n]["backbone"] == bb) for bb in C.LAYERS}
        ccf = [n for n in elig if recs[n]["cc"] >= best[recs[n]["backbone"]] - 0.05]
        if len(ccf) >= 3:
            y3, yb3 = S_vec(ccf, "mis20"); sec["cc_filter"] = assoc(ccf, [recs[n]["ERF50"] for n in ccf], y3, yb3, grp(ccf))
        # competing descriptors over eligible readouts
        yv = np.array(y)
        dep = [recs[n]["rel_depth"] for n in elig]
        objg = [C.OBJECTIVE[recs[n]["backbone"]] for n in elig]
        r2_dep, _ = ols_r2(yv, [np.array(dep)])
        r2_obj, _ = ols_r2(yv, [np.array([1.0 if o == "image-text" else 0.0 for o in objg])])
        sec["competing"] = dict(rho_rel_depth=spearman(dep, yv), R2_rel_depth=r2_dep, R2_objective_family=r2_obj,
                                rho_erf50_vs_depth_all18=spearman([recs[n]["ERF50"] for n in have], [recs[n]["rel_depth"] for n in have]))
    for s in C.SIZES:
        es = [n for n in have if recs[n]["per_size"][str(s)]["eligible"]]
        if len(es) >= 3:
            pts, bs = zip(*[sens([parts_all[n][C.SIZES.index(s)]], "mis20") for n in es])
            sec[f"size{s}"] = assoc(es, [recs[n][f"ERF50_s{s}"] for n in es], list(pts), np.vstack(bs), grp(es))
    if len(have) >= 3:
        sec["all18_point_U20"] = assoc(have, [recs[n]["ERF50"] for n in have], [recs[n]["S20_point_any"] for n in have], None, grp(have))
        sec["all18_point_U20"]["flag"] = "includes readouts whose aligned CI includes 0; ratios of near-zero effects are unstable"
        Lb = np.vstack([np.sign(recs[n]["E"]["mean"]) * DA.bmeans(X(parts_all[n], "illusory") - X(parts_all[n], "mis20")) for n in have])
        sec["all18_L20"] = assoc(have, [recs[n]["ERF50"] for n in have], [recs[n]["L20"] for n in have], Lb, grp(have))
    out["secondary"] = sec
    out["seconds"] = round(time.time() - t0, 1)
    out["fake_erf_code_test"] = FAKE
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1, default=float)
    write_report(out)
    print("decision:", prim.get("decision"), prim.get("case"), "|", prim.get("wording"))


def fci(e):
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def fr(r):
    return "n.e." if r is None else f"{r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def write_report(out):
    L = []; w = L.append
    w("# Rotation sensitivity vs effective receptive field: results under nn/ANALYSIS_RULE.md\n")
    if out.get("fake_erf_code_test"):
        w("**CODE TEST WITH SYNTHETIC ERF VALUES: NOT A RESULT.**\n")
    w(f"Generated by nn/analyze_nn.py on {out['created']}. Rule SHA-256 now {out['rule_sha256_now']} "
      f"({'unchanged' if out['rule_unchanged'] else 'CHANGED since recording'}). Single seed (0); descriptive; CIs reflect stimulus "
      "sampling only. ERF and depth are collinear within each network.\n")
    if out["missing"]:
        w(f"**Missing inputs: {', '.join(out['missing'])}.**\n")
    p = out["primary"]
    w("## Decision (primary test)\n")
    w(f"**{p['decision']}** (case {p.get('case')}). {p['wording']}\n")
    if "rho" in p:
        w(f"n = {p['n']} eligible readouts; rho = {p['rho']:+.3f}, 95% CI [{p['rho_ci'][0]:+.3f}, {p['rho_ci'][1]:+.3f}] "
          f"(share of resamples with rho > 0: {p['rho_boot_frac_pos']:.3f}), permutation p = {p['perm_p']:.4f} ({p['perm_kind']}); "
          f"within-backbone rho: " + ", ".join(f"{g} {f2(v['rho'])} (n {v['n']})" for g, v in p["within"].items()) +
          f"; T = {f2(p['T'])}, stratified p = {p3(p['T_perm_p'])}; R2 backbone {p['R2_bb']:.3f} (adj {p['adjR2_bb']}), "
          f"R2 log2 ERF50 {p['R2_x']:.3f} (adj {p['adjR2_x']}), R2 both {p['R2_both']:.3f}, partial R2 of ERF | backbone "
          f"{p['partial_R2_x_given_bb']}, ERF coefficient in joint model {p['coef_x_joint']:+.3f}. Criteria: {p.get('criteria')}\n")
        iu = p["inverted_u"]
        w(f"Inverted U: quadratic coefficient {iu['quad_coef']:+.3f} [{iu['quad_coef_ci'][0]:+.3f}, {iu['quad_coef_ci'][1]:+.3f}], vertex "
          f"ERF50 {iu['vertex_erf50']}, readouts left/right {iu['n_left']}/{iu['n_right']}, interior peak by backbone "
          f"{iu['peak_interior_by_backbone']}: {iu['wording']}.\n")
    w("## All 18 readouts\n")
    w("| readout | source | CC | E [95% CI] | eligible | U(20) | A(20) | D(20) | slope | ERF50 px (336 / 448) | ERF90 | block-token ERF50 | S20 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in out["readouts"].items():
        e50 = r.get("ERF50"); s20 = (1 - r["U"][20]["value"]) if r["U"][20] else None
        w(f"| {n} | {r['source']} | {r['cc']:.3f} | {fci(r['E'])} | {'yes' if r['eligible'] else 'no'} | {fr(r['U'][20])} | {fr(r['A'][20])} | "
          f"{fr(r['D'][20])} | {r['slope']['slope_per_unit_sr']:+.3f} | "
          f"{'n/a' if e50 is None else f'{e50:.1f} ({r['ERF50_s336']} / {r['ERF50_s448']})'} | {r.get('ERF90', 'n/a')} | {r.get('bbERF50', 'n/a')} | "
          f"{'n.e.' if s20 is None else f'{s20:+.2f}'} |")
    w("")
    w("## Newly scored readouts: further measures and null comparisons (nulls at sigma 8 / 16 / 25 px)\n")
    w("| readout | core r | core share | vs control90 | real | real filled | U(10) / U(30) | A(10) / A(30) | corner s16 E | U20 CI below corner nulls | loss20 vs max null | sq D(20) vs nulls | slope vs corner slopes |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in out["readouts"].items():
        if not r["source"].startswith("nn"):
            continue
        nc = r["null_comparison"]
        w(f"| {n} | {fci(r['core_r'])} | {fci(r['core_share'])} | {fci(r['E_control90'])} | {fci(r['real'])} | {fci(r['real_filled'])} | "
          f"{fr(r['U'][10])} / {fr(r['U'][30])} | {fr(r['A'][10])} / {fr(r['A'][30])} | {nc['corner16_E']:+.3f} | {nc['U20_ci_below_corner_nulls']} "
          f"(min {nc['min_corner_U20_bracket']:.2f}) | {fci(nc['loss20'])} vs {nc['max_null_loss20_bracket']:+.3f} | "
          f"{'n.e.' if nc['Dsq20'] is None else format(nc['Dsq20'], '.2f')} vs {nc['null_Dsq20_range']} | {nc['slope']:+.3f} vs "
          + ", ".join(f"{v:+.3f}" for v in nc["corner_slopes_bracket"]) + " |")
    w("")
    w("## Secondary and sensitivity analyses (not decision-bearing)\n")
    w("| analysis | n | rho [95% CI] | perm p | within-backbone rho | T (p) | R2 x / backbone / both | partial R2 |")
    w("|---|---|---|---|---|---|---|---|")
    for k, a in out["secondary"].items():
        if k == "competing" or "rho" not in a:
            continue
        ci = f"[{a['rho_ci'][0]:+.2f}, {a['rho_ci'][1]:+.2f}]" if "rho_ci" in a else ""
        w(f"| {k} | {a['n']} | {a['rho']:+.2f} {ci} | {a['perm_p']:.4f} | " + ", ".join(f"{g} {f2(v['rho'])}" for g, v in a["within"].items())
          + f" | {f2(a['T'])} ({p3(a['T_perm_p'])}) | {a['R2_x']:.2f} / {a['R2_bb']:.2f} / {a['R2_both']:.2f} | {a['partial_R2_x_given_bb']} |")
    if "competing" in out["secondary"]:
        c = out["secondary"]["competing"]
        w(f"\nCompeting descriptors (eligible readouts): rho(relative depth, S20) {c['rho_rel_depth']:+.2f}, R2 relative depth {c['R2_rel_depth']:.2f}, "
          f"R2 objective family {c['R2_objective_family']:.2f}; rho(ERF50, relative depth) over all readouts {c['rho_erf50_vs_depth_all18']:+.2f}.")
    open(os.path.join(OUTDIR, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(allow_missing="--allow-missing" in sys.argv)
