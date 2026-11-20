"""
neuro/reanalysis/explore/ttr/ttr_analyze.py: analysis of the test-time-registers re-scoring (REANALYSIS_RULE.md Section 6,
item 12; DESIGN.md Section 4.4). EXPLORATORY: unadjusted, labelled, never decision-bearing; it cannot change P3, P4, P5
or the title. Numpy and scipy only (statistics from the fixed engine analyze_re through excommon). Written 18 November
2026 after P1 to P5, R8 and every exploratory output had been seen, before any TTR score existed.

Definitions (fixed here, before any TTR score was read):
  units     10 sweep readouts clip_L12, clip_L16, clip_L20, clip_L24, siglip_L4, siglip_L8, siglip_L12, siglip_L16,
            siglip_L20, siglip_L24 (seed 0, size average) and 5 swap cells siglip_L24|R1n ... |R4 (seeds averaged per
            display, then sizes; per-seed values kept for the seed point-estimate signs).
  arms      plain (no hooks), reg0 (register only; stim_v3 and MIT1003), ttr (register + shift).
  per unit and arm, linear quantities (WCU interval, WCR p, 12 clusters): E = r(illusory) - r(control180),
            E90 = r(illusory) - r(control90), E_real = r(real) - r(control180) (positive control), E at 336 and at 448
            px, L10, L20, L30 = r(illusory) - r(mis theta); ratios (BCa, Fieller, stability flag |t| >= 4, reportable
            only if the denominator's WCU interval excludes 0): U10, U20, U30 = E(mis theta) / E(illusory); with stim_alt
            (plain, ttr): A20 = E(alt20) / E(illusory) (alt20 = mean of phases a and b) and the square
            Delta20 = r(alt20) - r(mis20) (6 clusters, enumeration).
  paired    ttr - plain and reg0 - plain per configuration: dE, dE90, dL20 (WCU); dU20 and (ttr) dA20 (paired BCa,
            excommon.ratio_diff, both denominators screened).
  sign      E_ttr against E_plain: "interval on the same side" / "interval includes 0" (never called a reversal, rule
            7.1) / "interval on the opposite side"; the same for reg0.
  swap      P3-like (unadjusted): each non-R4 cell's E_ttr against R4_ttr's sign (WCR p < .05 on R4's side), and its
            seed point-estimate signs; P4-like: (R2 - R4) U20 and A20 under ttr (and plain, for comparison), paired
            BCa as analyze_re.p4; the R3 pair is excluded, as in rule 5.4.
  accuracy  held-out MIT1003 CC per unit and arm (swap: mean of the three seed CCs); dCC; flag "accuracy not retained
            under TTR" if CC_ttr < CC_plain - 0.03; the swap criterion analogue analyze_re.accuracy_valid(CC_ttr cell,
            CC_ttr R4) is shown and never used to drop a unit.
  tokens    gate K statistics (A15(a) 24 + 24 images) per flagged block and arm; on the 60 stim_v3 illusory and 60
            control180 displays the token-weighted location shares (inside, inducer, elsewhere) of the tokens above
            twice the median, per flagged block, arm and size; ratio median and 90th percentile over the 480 displays.
  check     plain-arm E of every unit against E computed by the same code from the published per-image values.
Outputs: results/explore/ttr_results.json (values unrounded, non-finite as null) and results/explore/ttr_results.md
(4 decimals), labelled exploratory, provenance as excommon.write_json.
Usage: nice -n 10 env OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $PY -B -I explore/ttr/ttr_analyze.py
"""
import json, math, os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import ttr_common as C                                                    # noqa: E402

X = C.X
AR = X.AR
UNITS = C.SWEEP_UNITS + C.SWAP_UNITS
V3_NEEDED = ("illusory", "control180", "control90", "real", "mis10", "mis20", "mis30")


# ====================================================================== loading
def unit_files(unit):
    if "|" in unit:
        return [(seed, unit.replace("|", "_")) for seed in C.SWAP_SEEDS]
    return [(0, unit)]


def load_unit(arm, unit, stimset, score_dir=None):
    """{name: value} averaged over seeds (per size), then over sizes; plus per-size and per-seed dicts; None if a file is
    missing."""
    d = score_dir or os.path.join(C.SCORES_TTR, arm)
    per = {}                                                              # (seed, size) -> {name: v}
    for seed, tag in unit_files(unit):
        p = os.path.join(d, f"{stimset}_{tag}_s{seed}.json")
        if not os.path.exists(p):
            return None
        j = json.load(open(p))
        for sz in ("336", "448"):
            per[(seed, sz)] = j[sz]["r_inside"]
    names = set.intersection(*[set(v) for v in per.values()])
    seeds = sorted({k[0] for k in per})
    by_size = {sz: {n: float(np.mean([per[(sd, sz)][n] for sd in seeds])) for n in names} for sz in ("336", "448")}
    avg = {n: 0.5 * (by_size["336"][n] + by_size["448"][n]) for n in names}
    by_seed = {sd: {n: 0.5 * (per[(sd, "336")][n] + per[(sd, "448")][n]) for n in names} for sd in seeds}
    return dict(avg=avg, size=by_size, seed=by_seed)


def load_published(unit, stimset):
    """Same layout from the published per-image files (for the plain-arm reproduction check)."""
    per = {}
    for seed, _ in unit_files(unit):
        for sz in ("336", "448"):
            if "|" in unit:
                fs, ar = unit.split("|")
                j = X.load_json(os.path.join(X.SWAP_SCORES, f"{stimset}_{fs}_{ar}_s{seed}.json"))
                per[(seed, sz)] = j[sz]["r_inside"]
            else:
                bb, L = unit.split("_L")
                j = X.load_json(X.sweep_published_file(bb, int(L), int(sz), stimset))
                per[(seed, sz)] = {k: v["r_inside"] for k, v in j.items()}
    seeds = sorted({k[0] for k in per})
    names = set.intersection(*[set(v) for v in per.values()])
    by_size = {sz: {n: float(np.mean([per[(sd, sz)][n] for sd in seeds])) for n in names} for sz in ("336", "448")}
    return {n: 0.5 * (by_size["336"][n] + by_size["448"][n]) for n in names}


def getter(sc):
    def val(c, cond):
        if cond.startswith("alt") and cond[-1] not in "ab":
            return 0.5 * (sc[f"{c}_{cond}a"] + sc[f"{c}_{cond}b"])
        return sc[f"{c}_{cond}"]
    return val


def pc(D, val, a, b=None):
    return D.arr(lambda c: val(c, a) - (val(c, b) if b else 0.0))


# ====================================================================== statistics per unit
def lin(arr, D):
    return X.lin(arr.ravel(), D)


def side(ci, ref_sign):
    if ci[0] is None:
        return None
    if ci[0] <= 0 <= ci[1]:
        return "interval includes 0"
    s = 1 if ci[0] > 0 else -1
    return "interval on the same side" if s == ref_sign else "interval on the opposite side"


def unit_stats(val, D, Dsq, val_alt=None):
    out = dict(E=lin(pc(D, val, "illusory", "control180"), D), E90=lin(pc(D, val, "illusory", "control90"), D),
               E_real=lin(pc(D, val, "real", "control180"), D))
    for th in (10, 20, 30):
        out[f"L{th}"] = lin(pc(D, val, "illusory", f"mis{th}"), D)
        out[f"U{th}"] = X.ratio(pc(D, val, f"mis{th}", "control180"), pc(D, val, "illusory", "control180"), D)
    if val_alt is not None:
        out["A20"] = X.ratio(pc(D, val_alt, "alt20", "control180"), pc(D, val_alt, "illusory", "control180"), D)
        out["square_Delta20"] = lin(pc(Dsq, val_alt, "alt20", "mis20"), Dsq)
    return out


def paired(val_a, val_p, D, alt=False):
    out = {}
    for nm, (x, y) in dict(dE=("illusory", "control180"), dE90=("illusory", "control90"), dL20=("illusory", "mis20"),
                           dE_real=("real", "control180")).items():
        out[nm] = lin(pc(D, val_a, x, y) - pc(D, val_p, x, y), D)
    out["dU20"] = X.ratio_diff(pc(D, val_a, "mis20", "control180"), pc(D, val_a, "illusory", "control180"),
                               pc(D, val_p, "mis20", "control180"), pc(D, val_p, "illusory", "control180"), D)
    if alt:
        out["dA20"] = X.ratio_diff(pc(D, val_a, "alt20", "control180"), pc(D, val_a, "illusory", "control180"),
                                   pc(D, val_p, "alt20", "control180"), pc(D, val_p, "illusory", "control180"), D)
    return out


def merged(arm, unit):
    v3 = load_unit(arm, unit, "stim_v3")
    if v3 is None:
        return None, None
    alt = load_unit(arm, unit, "stim_alt") if arm != "reg0" else None
    full = dict(v3["avg"])
    if alt is not None:
        full.update(alt["avg"])
    return dict(v3=v3, alt=alt, full=full), (alt is not None)


def mit_cc(arm, unit):
    ccs = []
    for seed, tag in unit_files(unit):
        p = os.path.join(C.SCORES_TTR, arm, f"mit_{tag}_s{seed}.json")
        if not os.path.exists(p):
            return None
        ccs.append(json.load(open(p))["CC"])
    return dict(seed_cc=ccs, CC=float(np.mean(ccs)))


# ====================================================================== tokens
def token_outcome():
    out = {}
    for bb in ("clip", "siglip"):
        out[bb] = {}
        for arm in C.ARMS:
            p = os.path.join(C.SCORES_TTR, "norms", f"{arm}_stim_v3_{bb}.json")
            if not os.path.exists(p):
                continue
            disp = json.load(open(p))["displays"]
            for L in C.FLAGGED[bb]:
                for sz in ("336", "448"):
                    ratios = [disp[n][sz][str(L)]["ratio"] for n in disp]
                    rec = dict(ratio_median_480=float(np.median(ratios)), ratio_p90_480=float(np.percentile(ratios, 90)),
                               share_mean_480=float(np.mean([disp[n][sz][str(L)]["share"] for n in disp])))
                    for cond in ("illusory", "control180"):
                        locs = [disp[n][sz][str(L)]["loc"] for n in disp if n.endswith("_" + cond)]
                        tot = sum(l["n"] for l in locs)
                        rec[cond] = dict(n_tokens=int(tot), n_displays=len(locs),
                                         **{k: (sum(l[k] * l["n"] for l in locs if l["n"]) / tot if tot else None)
                                            for k in ("inside", "inducer", "elsewhere")})
                    out[bb].setdefault(str(L), {}).setdefault(arm, {})[sz] = rec
    return out


# ====================================================================== main
def wall_times():
    p = os.path.join(C.RE, "results", "run_logs", "wall_times.tsv")
    rows = []
    if os.path.exists(p):
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if f and f[0].startswith("TTR_"):
                rows.append(dict(label=f[0], start=f[1], end=f[2], wall_s=int(f[3]), exit=int(f[4])))
    return rows


def main():
    C.require_public()                                                    # rule Section 8.1
    t0 = time.time()
    D, Dsq = X.design(), X.design("square")
    res = dict(units={}, swap={}, accuracy={}, reproduction={}, notes=[])
    vals = {}
    for u in UNITS:
        r = {}
        for arm in C.ARMS:
            m, has_alt = merged(arm, u)
            if m is None:
                r[arm] = dict(note="not computable: score files missing")
                continue
            vals[(u, arm)] = m
            val = getter(m["full"])
            r[arm] = unit_stats(val, D, Dsq, val if has_alt else None)
            r[arm]["E_by_size"] = {sz: lin(pc(D, getter(m["v3"]["size"][sz]), "illusory", "control180"), D)
                                   for sz in ("336", "448")}
        for arm in ("ttr", "reg0"):
            if (u, arm) in vals and (u, "plain") in vals:
                va, vp = getter(vals[(u, arm)]["full"]), getter(vals[(u, "plain")]["full"])
                alt = vals[(u, arm)]["alt"] is not None and vals[(u, "plain")]["alt"] is not None
                r[f"{arm}_minus_plain"] = paired(va, vp, D, alt)
                ep = r["plain"]["E"]["mean"]
                r[f"sign_status_{arm}"] = side(r[arm]["E"]["ci"], int(np.sign(ep)))
        res["units"][u] = r
        if (u, "plain") in vals:
            pub = load_published(u, "stim_v3")
            pub.update(load_published(u, "stim_alt"))
            vp, vq = getter(vals[(u, "plain")]["full"]), getter(pub)
            e1 = pc(D, vp, "illusory", "control180").mean(); e2 = pc(D, vq, "illusory", "control180").mean()
            md = max(abs(vals[(u, "plain")]["full"][n] - pub[n]) for n in vals[(u, "plain")]["full"])
            res["reproduction"][u] = dict(E_plain=float(e1), E_published=float(e2), abs_diff_E=float(abs(e1 - e2)),
                                          max_abs_diff_per_display_avg=float(md))
        acc = {arm: mit_cc(arm, u) for arm in C.ARMS}
        rec = dict(acc)
        if acc["plain"] and acc["ttr"]:
            rec["dCC_ttr"] = acc["ttr"]["CC"] - acc["plain"]["CC"]
            rec["accuracy_not_retained_under_TTR"] = bool(acc["ttr"]["CC"] < acc["plain"]["CC"] - 0.03)
        if acc["plain"] and acc["reg0"]:
            rec["dCC_reg0"] = acc["reg0"]["CC"] - acc["plain"]["CC"]
        res["accuracy"][u] = rec
        C.log(f"{u} done ({time.time() - t0:.0f}s)")

    # ---- swap analogues
    r4 = "siglip_L24|R4"
    for arm in ("plain", "ttr"):
        if (r4, arm) not in vals:
            continue
        ref = res["units"][r4][arm]["E"]
        ref_sign, ref_def = int(np.sign(ref["mean"])), bool(ref["p"] < 0.05)
        p3 = {}
        for ar in ("R1n", "R1", "R2", "R3"):
            u = f"siglip_L24|{ar}"
            if (u, arm) not in vals:
                continue
            e = res["units"][u][arm]["E"]
            seeds = vals[(u, arm)]["v3"]["seed"]
            seed_est = {str(sd): float(pc(D, getter(v), "illusory", "control180").mean()) for sd, v in seeds.items()}
            p3[ar] = dict(E=e, keeps_R4_sign_unadjusted=bool(ref_def and e["p"] < 0.05 and np.sign(e["mean"]) == ref_sign),
                          opposite_sign_interval=AR.opposite_interval(e["ci"], ref_sign, ref_def),
                          seed_point_estimates=seed_est,
                          n_seeds_with_R4_sign=int(sum(np.sign(v) == ref_sign for v in seed_est.values())))
        res["swap"][f"P3_like_{arm}"] = dict(R4=ref, R4_sign=ref_sign, R4_defined=ref_def, cells=p3)
        p4 = {}
        u2 = "siglip_L24|R2"
        if (u2, arm) in vals:
            v2, v4 = getter(vals[(u2, arm)]["full"]), getter(vals[(r4, arm)]["full"])
            for meas, test in (("U20", "mis20"), ("A20", "alt20")):
                p4[meas] = X.ratio_diff(pc(D, v2, test, "control180"), pc(D, v2, "illusory", "control180"),
                                        pc(D, v4, test, "control180"), pc(D, v4, "illusory", "control180"), D)
        res["swap"][f"P4_like_{arm}"] = p4
        cc = {ar: (res["accuracy"][f"siglip_L24|{ar}"].get(arm) or {}).get("CC") for ar in C.SWAP_ARCHS}
        if all(v is not None for v in cc.values()):
            res["swap"][f"accuracy_criterion_analogue_{arm}"] = {
                ar: AR.accuracy_valid(cc[ar], cc["R4"]) for ar in C.SWAP_ARCHS}

    # ---- sweep sign pattern under ttr
    res["sweep_sign_pattern"] = {u: dict(plain=res["units"][u]["plain"]["E"]["mean"] if "E" in res["units"][u]["plain"] else None,
                                         ttr=res["units"][u].get("ttr", {}).get("E", {}).get("mean"),
                                         sign_status=res["units"][u].get("sign_status_ttr"))
                                 for u in C.SWEEP_UNITS}
    # ---- tokens, selection, gates
    res["tokens_stim_v3"] = token_outcome()
    sel = {bb: X.load_json(os.path.join(C.RES_TTR, f"TTR_select_{bb}.json")) for bb in ("clip", "siglip")}
    res["selection"] = {bb: (None if s is None else {k: s["result"][k] for k in (
        "emergence_block", "highest_layer", "detection_block", "tau", "evaluated_flagged_blocks",
        "flagged_before_emergence", "chosen", "n_neurons", "neuron_dict", "not_suppressed_on_selection_images",
        "items_without_outliers", "outliers_per_item", "jaccard_outliers_vs_gt2", "position_constancy",
        "synthetic_check", "median_ratio_by_size", "timings")}) for bb, s in sel.items()}
    gk = X.load_json(os.path.join(C.RES_TTR, "TTR_gateK.json"))
    res["gate_K"] = None if gk is None else gk["result"]["gate_K"]
    res["gate_F_P"] = {}
    for bb in ("clip", "siglip"):
        g = X.load_json(os.path.join(C.RES_TTR, f"TTR_score_{bb}_plain.json"))
        if g:
            res["gate_F_P"][bb] = dict(gate_F=g["result"]["gate_F"], gate_P={k: v for k, v in g["result"]["gate_P"].items()
                                                                             if k not in ("units",)},
                                       exact_formula_check=g["result"]["exact_formula_check"])
    res["selection_sha256"] = {bb: (X.sha256_file(os.path.join(C.RES_TTR, f"TTR_select_{bb}.json"))
                                    if sel[bb] else None) for bb in ("clip", "siglip")}
    res["wall_times"] = wall_times()
    res["analysis_seconds"] = time.time() - t0
    X.write_json("ttr_results.json", res, os.path.abspath(__file__),
                 inputs=["scores_ttr/{plain,reg0,ttr}", "scores_ttr/norms", "results/explore/ttr/*.json",
                         "published per-image scores (plain reproduction check)"],
                 extra_code=[os.path.join(C.HERE, "ttr_common.py")])
    write_md(res)


# ====================================================================== markdown
def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.4f}"


def ci4(c):
    if not c or c[0] is None:
        return "[n/a]"
    star = "*" if (c[0] > 0 or c[1] < 0) else ""
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]{star}"


def linfmt(r):
    if not r or r.get("mean") is None:
        return "n/a"
    return f"{f4(r['mean'])} {ci4(r['ci'])}"


def ratfmt(r):
    if not r or r.get("value") is None:
        return "n/a"
    fl = r.get("flag")
    return f"{r['value']:.4f} {ci4(r.get('ci_bca'))}" + (f" ({fl})" if fl else "")


def rdfmt(r):
    if not r or r.get("diff") is None:
        return "n/a"
    fl = r.get("flag")
    return f"{f4(r['diff'])} {ci4(r.get('ci_bca'))}" + (f" ({fl})" if fl else "")


def summary_md(res):
    """Section 0: the main numbers in sentences (all exploratory; computed from res, nothing typed by hand)."""
    L = ["## 0. Summary (exploratory)", ""]
    a = L.append
    sel = res["selection"]
    for bb, s in sel.items():
        if s is None:
            continue
        a(f"- {bb}: {s['n_neurons']} register neurons in layers {sorted(int(l) for l in s['neuron_dict'])} "
          f"({s['chosen']['branch']} branch, k = {s['chosen']['k']}; emergence block {s['emergence_block']}, so the "
          f"primary branch searched layers 0 to {s['highest_layer']} and no k up to 300 met the criterion there; the "
          f"fallback searched layers 0 to 23).")
    gk = res.get("gate_K") or {}
    for bb, g in gk.items():
        ev = g["evaluated_flagged_blocks"]
        pl = [g["summary"][f"kanizsa|{s}|plain|{b}"]["p90"] for b in ev for s in C.SIZES]
        tt = [g["summary"][f"kanizsa|{s}|ttr|{b}"]["p90"] for b in ev for s in C.SIZES]
        r0 = [g["summary"][f"kanizsa|{s}|reg0|{b}"]["p90"] for b in ev for s in C.SIZES]
        a(f"- {bb} token norms on the 24 A15(a) Kanizsa displays, blocks {ev} at 336 and 448 px (90th percentile of "
          f"max/median): plain {min(pl):.2f} to {max(pl):.2f}; register only {min(r0):.2f} to {max(r0):.2f}; ttr "
          f"{min(tt):.2f} to {max(tt):.2f}. Strict gate K (<= 2.0) failing blocks: "
          f"{g['criterion']['kanizsa']['blocks_failing'] or 'none'} (labelled \"not suppressed on the displays\"); "
          f"on the 24 MIT1003 images: {g['criterion']['mit1003']['blocks_failing'] or 'none'} failing.")
        before = [b for b, lab in g["labels"].items() if lab.startswith("flagged, before")]
        if before:
            p = g["summary"]["kanizsa|336|plain|" + before[0]]["p90"]; t = g["summary"]["kanizsa|336|ttr|" + before[0]]["p90"]
            a(f"  - {bb} block {', '.join(before)}: flagged before the emergence block; unchanged by TTR ({p:.2f} -> {t:.2f} "
              "at 336 px), as no register neuron lies below it.")
    a("- On every stim_v3 display and flagged block, the plain tokens above twice the median sit outside both the "
      "inside AOI and the inducer AOI (background), for illusory and control180 displays alike (Section 5).")
    a("- Contrast E = r(illusory) - r(control180), readouts frozen (WCU 95% intervals; `*` excludes 0):")
    for u, r in res["units"].items():
        if "E" not in r.get("ttr", {}):
            a(f"  - {u}: ttr not computable")
            continue
        a(f"  - {u}: plain {linfmt(r['plain']['E'])}; ttr {linfmt(r['ttr']['E'])}; ttr - plain "
          f"{linfmt(r['ttr_minus_plain']['dE'])}; register only - plain {linfmt(r.get('reg0_minus_plain', {}).get('dE'))}; "
          f"{r.get('sign_status_ttr')} (relative to the sign of the plain point estimate).")
    p3 = res["swap"].get("P3_like_ttr")
    if p3:
        keep = [ar for ar, c in p3["cells"].items() if c["keeps_R4_sign_unadjusted"]]
        opp = [ar for ar, c in p3["cells"].items() if c["opposite_sign_interval"]]
        seeds = sum(c["n_seeds_with_R4_sign"] for c in p3["cells"].values())
        a(f"- SigLIP-L block 24 swap under ttr (P3-like, unadjusted): {len(keep)} of {len(p3['cells'])} non-R4 cells keep "
          f"R4's sign ({', '.join(keep)}); opposite-sign intervals: {opp or 'none'}; {seeds} of "
          f"{3 * len(p3['cells'])} seed point estimates have R4's sign.")
    for arm in ("plain", "ttr"):
        p4 = res["swap"].get(f"P4_like_{arm}", {})
        if p4:
            a(f"- P4-like (R2 - R4), {arm}: U20 {rdfmt(p4['U20'])}; A20 {rdfmt(p4['A20'])}.")
    d = [r["dCC_ttr"] for r in res["accuracy"].values() if r.get("dCC_ttr") is not None]
    nflag = sum(bool(r.get("accuracy_not_retained_under_TTR")) for r in res["accuracy"].values())
    if d:
        a(f"- Held-out MIT1003 CC, ttr - plain: {min(d):+.4f} to {max(d):+.4f} over {len(d)} units; {nflag} flagged "
          "(drop > 0.03).")
    u = "siglip_L24|R4"
    s = sel.get("siglip")
    if s and "E" in res["units"][u].get("ttr", {}) and gk.get("siglip"):
        g = gk["siglip"]
        pl = g["summary"]["kanizsa|336|plain|24"]["p90"]; tt = g["summary"]["kanizsa|336|ttr|24"]["p90"]
        r = res["units"][u]
        a("")
        a("Sentence template (DESIGN.md Section 4.4) filled for SigLIP-L block 24 R4: \"With test-time registers (Jiang "
          f"et al. 2025; {s['n_neurons']} register neurons in layers {min(int(l) for l in s['neuron_dict'])} to "
          f"{max(int(l) for l in s['neuron_dict'])}, chosen on 100 MIT1003 training images), the high-norm tokens at "
          f"SigLIP-L block 24 fell from {pl:.2f} to {tt:.2f} times the median (90th percentile, 336 px); with the readouts "
          f"unchanged, R4 gave {linfmt(r['ttr']['E'])} (same run without registers {linfmt(r['plain']['E'])}; difference "
          f"{linfmt(r['ttr_minus_plain']['dE'])}); exploratory.\"")
    a("")
    return L


def write_md(res):
    L = []
    a = L.append
    a("# Test-time registers on the flagged CLIP-L and SigLIP-L blocks (exploratory)")
    a("")
    a(f"- Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `explore/ttr/ttr_analyze.py` from "
      "`scores_ttr/` and `results/explore/ttr/`. **Exploratory** (REANALYSIS_RULE.md Section 6, item 12): unadjusted, "
      "never decision-bearing; P3, P4, P5 and the title are unchanged. Design: `explore/ttr/DESIGN.md`; deviations: "
      "`DEVIATIONS.md` D8.")
    a("- Readouts frozen (flag F1). Intervals: WCU (12 clusters; 6 for the square Delta20) for linear quantities, BCa "
      "with the stability flag for ratios and ratio differences; `*` marks an interval that excludes 0. 4 decimals; "
      "unrounded values in `ttr_results.json`.")
    a("")
    L.extend(summary_md(res))
    a("## 1. Selection and gates")
    a("")
    a("| backbone | e (h) | detection block d | tau 336 / 448 | chosen | neurons (per layer) | not suppressed on selection images | synthetic check |")
    a("|---|---|---|---|---|---|---|---|")
    for bb, s in res["selection"].items():
        if s is None:
            a(f"| {bb} | not run | | | | | | |")
            continue
        per = ", ".join(f"{l}: {len(v)}" for l, v in s["neuron_dict"].items())
        a(f"| {bb} | {s['emergence_block']} ({s['highest_layer']}) | {s['detection_block']} | "
          f"{s['tau']['336']:.2f} / {s['tau']['448']:.2f} | {s['chosen']['branch']} k = {s['chosen']['k']} "
          f"({'passed' if s['chosen']['passed'] else 'criterion not met'}) | {s['n_neurons']} ({per}) | "
          f"{s['not_suppressed_on_selection_images'] or 'none'} | {'pass' if s['synthetic_check']['passed'] else 'FAIL'} |")
    a("")
    for bb, g in res["gate_F_P"].items():
        gp = g["gate_P"]
        sweep_mit = [v["abs_diff"] for k, v in gp.get("mit", {}).items() if "abs_diff" in v]
        swap_txt = ("" if not gp.get("n_mit_values_compared") else
                    f"; swap MIT1003 per-image CC ({gp['n_mit_values_compared']} values, published on MPS to 5 decimals) "
                    f"max |diff| {gp['max_abs_diff_mit_per_image']:.3g} (tolerance 1e-4)")
        a(f"- {bb}: gate F {'pass' if g['gate_F']['passed'] else 'FAIL'} (max |diff| 0 required); gate P "
          f"{'pass' if gp['passed'] else 'FAIL'}: {gp.get('n_display_values_compared')} per-image r_inside values "
          f"(stim_v3 and stim_alt, both sizes), max |diff| {gp.get('max_abs_diff_display'):.3g} (tolerance 1e-4); sweep "
          f"MIT1003 mean CC of {len(sweep_mit)} readouts against the 4-decimal published means, max |diff| "
          f"{max(sweep_mit):.3g} (tolerance 1.5e-4){swap_txt}; exact-formula check max |diff| "
          f"{g['exact_formula_check']['max_abs_diff']:.3g} on {g['exact_formula_check']['n']} values.")
    a("")
    if res["gate_K"]:
        a("### Gate K: token norms on the A15(a) images (90th percentile of max/median over 24 images; plain -> ttr)")
        a("")
        a("| backbone | block | label | Kanizsa 336 | Kanizsa 448 | MIT1003 336 | MIT1003 448 | share > 2x median, Kanizsa 336 (plain -> ttr) |")
        a("|---|---|---|---|---|---|---|---|")
        for bb, g in res["gate_K"].items():
            for b in C.FLAGGED[bb]:
                cells = []
                for sn in ("kanizsa", "mit1003"):
                    for s in C.SIZES:
                        pl = g["summary"][f"{sn}|{s}|plain|{b}"]; tt = g["summary"][f"{sn}|{s}|ttr|{b}"]
                        cells.append(f"{pl['p90']:.2f} -> {tt['p90']:.2f}")
                pl = g["summary"][f"kanizsa|336|plain|{b}"]; tt = g["summary"][f"kanizsa|336|ttr|{b}"]
                a(f"| {bb} | {b} | {g['labels'][str(b)]} | " + " | ".join(cells) +
                  f" | {pl['share_mean']:.4f} -> {tt['share_mean']:.4f} |")
        a("")
        for bb, g in res["gate_K"].items():
            a(f"- {bb}: gate K on the displays {'passed' if g['criterion']['kanizsa']['passed'] else 'NOT passed'} "
              f"(failing blocks {g['criterion']['kanizsa']['blocks_failing'] or 'none'}); on MIT1003 "
              f"{'passed' if g['criterion']['mit1003']['passed'] else 'NOT passed'} (failing "
              f"{g['criterion']['mit1003']['blocks_failing'] or 'none'}); plain reproduces A15(a) to "
              f"{g['plain_reproduces_A15a_max_abs_diff']:.2g}.")
        a("")
    a("## 2. Contrast with and without test-time registers (frozen readouts)")
    a("")
    a("| unit | E plain | E ttr | ttr - plain | sign status | E reg0 | reg0 - plain | E_real ttr |")
    a("|---|---|---|---|---|---|---|---|")
    for u, r in res["units"].items():
        g = lambda arm, k: r.get(arm, {}).get(k)
        a(f"| {u.replace('|', ' ')} | {linfmt(g('plain', 'E'))} | {linfmt(g('ttr', 'E'))} | "
          f"{linfmt(r.get('ttr_minus_plain', {}).get('dE'))} | {r.get('sign_status_ttr')} | {linfmt(g('reg0', 'E'))} | "
          f"{linfmt(r.get('reg0_minus_plain', {}).get('dE'))} | {linfmt(g('ttr', 'E_real'))} |")
    a("")
    a("### 90 degree control and rotation profile")
    a("")
    a("| unit | arm | E90 | L10 | L20 | L30 | U10 | U20 | U30 | A20 | square Delta20 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for u, r in res["units"].items():
        for arm in C.ARMS:
            x = r.get(arm, {})
            if "E" not in x:
                continue
            a(f"| {u.replace('|', ' ')} | {arm} | {linfmt(x['E90'])} | {linfmt(x['L10'])} | {linfmt(x['L20'])} | "
              f"{linfmt(x['L30'])} | {ratfmt(x['U10'])} | {ratfmt(x['U20'])} | {ratfmt(x['U30'])} | "
              f"{ratfmt(x.get('A20'))} | {linfmt(x.get('square_Delta20'))} |")
    a("")
    a("### Paired changes (arm minus plain, per configuration)")
    a("")
    a("| unit | arm | dE90 | dL20 | dE_real | dU20 | dA20 |")
    a("|---|---|---|---|---|---|---|")
    for u, r in res["units"].items():
        for arm in ("ttr", "reg0"):
            x = r.get(f"{arm}_minus_plain")
            if not x:
                continue
            a(f"| {u.replace('|', ' ')} | {arm} | {linfmt(x['dE90'])} | {linfmt(x['dL20'])} | {linfmt(x['dE_real'])} | "
              f"{rdfmt(x['dU20'])} | {rdfmt(x.get('dA20'))} |")
    a("")
    a("## 3. SigLIP-L block 24 readout swap under test-time registers (P3-like and P4-like, unadjusted)")
    a("")
    for arm in ("plain", "ttr"):
        p3 = res["swap"].get(f"P3_like_{arm}")
        if not p3:
            continue
        a(f"- {arm}: R4 E {linfmt(p3['R4'])} (sign {p3['R4_sign']:+d}, defined {p3['R4_defined']}).")
        for ar, c in p3["cells"].items():
            a(f"  - {ar}: E {linfmt(c['E'])}, WCR p {c['E']['p']:.4g}; keeps R4's sign (unadjusted) "
              f"{c['keeps_R4_sign_unadjusted']}; opposite-sign interval {c['opposite_sign_interval']}; seeds with R4's "
              f"sign {c['n_seeds_with_R4_sign']} of 3 ({', '.join(f4(v) for v in c['seed_point_estimates'].values())}).")
        p4 = res["swap"].get(f"P4_like_{arm}", {})
        for meas, v in p4.items():
            a(f"  - P4-like (R2 - R4) {meas}: {rdfmt(v)}, BCa p {v['p_bca']:.4g}; ratios R2 {v['ratio_1']:.4f}, R4 {v['ratio_2']:.4f}.")
        acc = res["swap"].get(f"accuracy_criterion_analogue_{arm}")
        if acc:
            a(f"  - accuracy criterion analogue (shown, never used to drop a cell): {acc}")
    a("")
    a("## 4. Held-out MIT1003 CC (336 px, centre bias on)")
    a("")
    a("| unit | plain | reg0 | ttr | ttr - plain | flag (drop > 0.03) |")
    a("|---|---|---|---|---|---|")
    for u, r in res["accuracy"].items():
        cc = lambda arm: "n/a" if not r.get(arm) else f"{r[arm]['CC']:.4f}"
        a(f"| {u.replace('|', ' ')} | {cc('plain')} | {cc('reg0')} | {cc('ttr')} | {f4(r.get('dCC_ttr'))} | "
          f"{'accuracy not retained under TTR' if r.get('accuracy_not_retained_under_TTR') else 'no'} |")
    a("")
    a("## 5. Where the high-norm tokens sit on the stim_v3 displays (token-weighted shares, 60 illusory / 60 control180)")
    a("")
    a("| backbone | block | arm | size | ratio median / p90 (480) | illusory: inside / inducer / elsewhere (n) | control180: inside / inducer / elsewhere (n) |")
    a("|---|---|---|---|---|---|---|")
    fs = lambda v: "n/a" if v is None else f"{v:.3f}"
    for bb, blocks in res["tokens_stim_v3"].items():
        for b, arms in blocks.items():
            for arm, szs in arms.items():
                for sz, rec in szs.items():
                    il, co = rec["illusory"], rec["control180"]
                    a(f"| {bb} | {b} | {arm} | {sz} | {rec['ratio_median_480']:.2f} / {rec['ratio_p90_480']:.2f} | "
                      f"{fs(il['inside'])} / {fs(il['inducer'])} / {fs(il['elsewhere'])} ({il['n_tokens']}) | "
                      f"{fs(co['inside'])} / {fs(co['inducer'])} / {fs(co['elsewhere'])} ({co['n_tokens']}) |")
    a("")
    a("## 6. Plain-arm reproduction of the published E")
    a("")
    mx = max([v["abs_diff_E"] for v in res["reproduction"].values()], default=None)
    a(f"- Largest |E_plain - E_published| over {len(res['reproduction'])} units: {mx if mx is None else format(mx, '.3g')}.")
    a("")
    a("## 7. Wall times (results/run_logs/wall_times.tsv)")
    a("")
    a("| step | start (UTC) | wall s | exit |")
    a("|---|---|---|---|")
    for w in res["wall_times"]:
        a(f"| {w['label']} | {w['start']} | {w['wall_s']} | {w['exit']} |")
    a(f"| total of listed steps | | {sum(w['wall_s'] for w in res['wall_times'])} | |")
    a("")
    p = os.path.join(X.RESULTS, "ttr_results.md")
    with open(p + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(p + ".tmp", p)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
