"""
Task 1b analysis: uniform versus alternating misalignment, inside AOI and core AOI, for DeepGaze IIE, DeepGaze MSDB,
the nine primary sweep readouts (336 and 448 px averaged per configuration; each size also separately) and the 15
local nulls (ink, edge, corner at sigma 8, 16, 25, 40, 60).

Inputs: raw_v3maps/<tag>.json (stim_v3 conditions, from the saved v2 maps), raw/<tag>.json (stim_alt, scored here),
raw_nulls.json (local nulls on both sets).
alt<theta> = mean of phases a and b per configuration and measure.

Statistics: cluster bootstrap, 10,000 resamples over shape x support ratio x polarity cells with positions resampled
within cells. For the pooled design the resample indices are those of v2/scores/analyze_v3.py (same seed 2026, same
cell order; imported), so pooled numbers are directly comparable with v2. For the square-only and triangle-only splits
the same procedure is run on their 6 cells (seed 2026). 95% percentile CIs; two-sided bootstrap p (floor "p < .001").
  effect(test, ctrl)    = mean over configurations of m(test) - m(ctrl)
  ratio_uni(theta)      = effect(mis theta, control180) / effect(illusory, control180)
  ratio_alt(theta)      = effect(alt theta, control180) / effect(illusory, control180)
  diff(theta)           = ratio_alt - ratio_uni = (m(alt) - m(mis)) / effect(illusory, control180)
  drop_ratio(theta)     = (m(illusory) - m(alt)) / (m(illusory) - m(mis)): the alternating drop as a fraction of the
                          uniform drop (1 = same reduction, 0 = no reduction under alternation); control-free.
  ratios use the bootstrap ratio of resampled means; they are reported only where the denominator CI excludes 0.
  SR slope              = OLS slope of the per-configuration effect(illusory, control180) on support ratio (per unit
                          support ratio; the design spans 0.3 to 0.7), bootstrapped with the same resamples.
Writes v3/scores/results.json and v3/scores/table.md.
"""
import json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scoring"))
import common_alt as CA                                        # noqa: E402  (scoring/common_alt.py)
sys.path.insert(0, _HERE)
import analyze_v3 as A2                                         # noqa: E402  (v2 bootstrap indices; no side effects)
LP = CA.LP
SC = os.path.join(CA.V3, "scores")                              # <results>/v3/scores

B, SEED = A2.B, A2.SEED
THETAS = (10, 20, 30)
MEAS = ("r_inside", "r_core30", "f_core30", "r_core45", "f_core45", "r_eroded_all")
PRIMARY_MEAS = ("r_inside", "r_core30", "f_core30")
MEAS_LABEL = {"r_inside": "inside r", "r_core30": "core r (D 30)", "f_core30": "core share of map mass (D 30)",
              "r_core45": "core r (D 45)", "f_core45": "core share (D 45)", "r_eroded_all": "inside_eroded_all r"}
CONDS = list(CA.V3_CONDS) + [f"alt{t}" for t in THETAS] + list(CA.ALT_CONDS)
META = {}
for it in CA.ALT_ITEMS:
    META[it["config"]] = (it["shape"], it["sr"], it["pol"], it["pos"])


# ------------------------------------------------------------------ designs
class Design:
    def __init__(self, shape=None):
        cells = {}
        for c, (sh, sr, pol, pos) in META.items():
            if shape is None or sh == shape:
                cells.setdefault((sh, sr, pol), []).append((pos, c))
        keys = sorted(cells, key=str)
        self.keys = keys
        self.cells = [[c for _, c in sorted(cells[k])] for k in keys]
        self.sr = np.array([k[1] for k in keys], float)
        K, P = len(self.cells), len(self.cells[0])
        if shape is None:
            assert self.cells == A2.DESIGN["kanizsa"][1], "cell order differs from v2/scores/analyze_v3"
            self.idx = A2.IDX["kanizsa"]
        else:
            rng = np.random.RandomState(SEED)
            self.idx = (rng.randint(0, K, (B, K)), rng.randint(0, P, (B, K, P)))

    def bmeans(self, v):
        ci, pi = self.idx
        return v[ci[:, :, None], pi].mean((1, 2))


DES = {"all": Design(), "square": Design("square"), "triangle": Design("triangle")}


def pstr(p):
    return "p < .001" if p < 0.001 else ("p = " + f"{p:.3f}".lstrip("0"))


def summ(D, v):
    bm = D.bmeans(v); lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)))
    return dict(mean=round(float(v.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)], p=float(p), p_str=pstr(p))


def excl0(s):
    return s["ci"][0] > 0 or s["ci"][1] < 0


def ratio(D, num, den):
    bn, bd = D.bmeans(num), D.bmeans(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    return dict(value=round(float(num.mean() / den.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                frac_boot_den_same_sign=round(float(np.mean(np.sign(bd) == np.sign(den.mean()))), 4))


def slope(D, v):
    ci, pi = D.idx
    E = v[ci[:, :, None], pi]
    X = np.broadcast_to(D.sr[ci][:, :, None], E.shape)
    xm, em = X.mean((1, 2), keepdims=True), E.mean((1, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = ((X - xm) * (E - em)).sum((1, 2)) / ((X - xm) ** 2).sum((1, 2))
    x = np.broadcast_to(D.sr[:, None], v.shape)
    pt = float(((x - x.mean()) * (v - v.mean())).sum() / ((x - x.mean()) ** 2).sum())
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    p = min(1.0, 2 * min(np.nanmean(b <= 0), np.nanmean(b >= 0)))
    by = {str(s): round(float(v[D.sr == s].mean()), 4) for s in sorted(set(D.sr))}
    return dict(slope_per_unit_sr=round(pt, 4), ci=[round(float(lo), 4), round(float(hi), 4)], p=float(p), p_str=pstr(p),
                mean_by_sr=by)


# ------------------------------------------------------------------ data
def load_json(p):
    return LP.load_json(p) if LP.json_exists(p) else None


def with_alt(rec):
    rec = dict(rec)
    for c in CA.CONFIGS:
        for t in THETAS:
            a, b = rec[f"{c}_alt{t}a"], rec[f"{c}_alt{t}b"]
            rec[f"{c}_alt{t}"] = {m: 0.5 * (a[m] + b[m]) for m in MEAS}
    return rec


def tag_scores(tag):
    v3 = load_json(os.path.join(SC, "raw_v3maps", tag + ".json")); al = load_json(os.path.join(SC, "raw", tag + ".json"))
    if v3 is None or al is None or len(v3) != 480 or len(al) != 360:
        return None
    return with_alt({**v3, **al})


def arr(D, parts, cond, m):
    return np.mean([[[sc[f"{c}_{cond}"][m] for c in cell] for cell in D.cells] for sc in parts], 0)


# ------------------------------------------------------------------ per model
def analyse(parts):
    out = {}
    for m in MEAS:
        out[m] = {}
        for sub, D in DES.items():
            X = {c: arr(D, parts, c, m) for c in CONDS}
            o = dict(levels={c: summ(D, X[c]) for c in CONDS},
                     effects_vs_control180={c: summ(D, X[c] - X["control180"]) for c in CONDS if c != "control180"},
                     effects_vs_control90={c: summ(D, X[c] - X["control90"]) for c in CONDS if c != "control90"},
                     alt_minus_mis={t: summ(D, X[f"alt{t}"] - X[f"mis{t}"]) for t in THETAS})
            al = X["illusory"] - X["control180"]
            o["aligned"] = o["effects_vs_control180"]["illusory"]
            o["ratios_computed"] = excl0(o["aligned"])
            if o["ratios_computed"]:
                o["ratio_uni"] = {t: ratio(D, X[f"mis{t}"] - X["control180"], al) for t in THETAS}
                o["ratio_alt"] = {t: ratio(D, X[f"alt{t}"] - X["control180"], al) for t in THETAS}
                o["ratio_alt_phase"] = {f"{t}{ph}": ratio(D, X[f"alt{t}{ph}"] - X["control180"], al) for t in THETAS for ph in "ab"}
                o["ratio_diff_alt_minus_uni"] = {t: ratio(D, X[f"alt{t}"] - X[f"mis{t}"], al) for t in THETAS}
                al90 = X["illusory"] - X["control90"]
                if excl0(summ(D, al90)):
                    o["ratio_uni_vs_control90"] = {t: ratio(D, X[f"mis{t}"] - X["control90"], al90) for t in THETAS}
                    o["ratio_alt_vs_control90"] = {t: ratio(D, X[f"alt{t}"] - X["control90"], al90) for t in THETAS}
            o["drop_uni"] = {t: summ(D, X["illusory"] - X[f"mis{t}"]) for t in THETAS}
            o["drop_alt"] = {t: summ(D, X["illusory"] - X[f"alt{t}"]) for t in THETAS}
            o["drop_ratio"] = {t: (ratio(D, X["illusory"] - X[f"alt{t}"], X["illusory"] - X[f"mis{t}"])
                                   if excl0(o["drop_uni"][t]) else None) for t in THETAS}
            o["sr_slope_vs_control180"] = slope(D, al)
            o["sr_slope_vs_control90"] = slope(D, X["illusory"] - X["control90"])
            out[m][sub] = o
    return out


def main():
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), bootstrap=B, seed=SEED,
               conventions=__doc__, measures=MEAS_LABEL, models={}, missing=[], checks={})
    MODELS = [("DeepGaze IIE", ["DeepGaze_IIE"], "model"), ("DeepGaze MSDB", ["DeepGaze_MSDB"], "model")]
    for bb, bbn, Ls in (("dinov2", "DINOv2-B", (2, 8, 12)), ("clip", "CLIP-L", (4, 12, 24)), ("siglip", "SigLIP-L", (4, 8, 24))):
        for L in Ls:
            MODELS.append((f"{bbn} L{L}", [f"{bb}_L{L}_s336", f"{bb}_L{L}_s448"], "layer readout"))
    for name, tags, kind in MODELS:
        parts = [tag_scores(t) for t in tags]
        if any(p is None for p in parts):
            res["missing"].append(name); print("missing", name, flush=True); continue
        rec = dict(kind=kind, tags=tags, results=analyse(parts))
        if kind == "layer readout":
            rec["per_size"] = {t[-3:]: analyse([p]) for t, p in zip(tags, parts)}
        res["models"][name] = rec
        print("done", name, flush=True)
    nulls = load_json(os.path.join(SC, "raw_nulls.json"))
    if nulls is None:
        res["missing"].append("local nulls")
    else:
        for nm in sorted(nulls, key=lambda s: (s.split("_")[0], int(s.split("_s")[1]))):
            res["models"][f"null {nm}"] = dict(kind="local null", tags=[nm], results=analyse([with_alt(nulls[nm])]))
        print("done nulls", flush=True)
    # checks against v2
    ck = {}
    for tag in ["DeepGaze_IIE", "DeepGaze_MSDB"]:
        al = load_json(os.path.join(SC, "raw", tag + ".json"))
        if al:
            ck[f"{tag} stim_alt |r_inside(float16) - r_inside(float32)| max"] = max(abs(v["r_inside"] - v["r_inside_f32"]) for v in al.values())
    p = os.path.join(CA.V3, "logs", "measure_v3maps.log")
    if os.path.exists(p):
        for line in open(p):
            ck.setdefault("measure_v3maps", []).append(line.strip())
    p = os.path.join(CA.V3, "logs", "local_nulls_alt.log")
    if os.path.exists(p):
        ck["local_nulls"] = [l.strip() for l in open(p)]
    p = os.path.join(CA.V3, "logs", "score_alt.log")
    if os.path.exists(p):
        ck["pipeline_checks"] = [l.strip() for l in open(p) if "pipeline check" in l]
    p = os.path.join(CA.V3, "logs", "saved_map_check.log")
    if os.path.exists(p):
        ck["saved_map_check (12 stim_alt images per tag, measures recomputed from the saved float16 maps)"] = [l.strip() for l in open(p) if l.strip()]
    res["checks"] = ck
    json.dump(res, open(os.path.join(SC, "results.json"), "w"), indent=1)
    write_table(res)
    print("missing:", res["missing"])


# ------------------------------------------------------------------ table.md
def f3(e):
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def fr(r):
    return "n/a" if r is None else f"{r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"


def write_table(res):
    L = []; w = L.append
    M = res["models"]
    w("# Uniform versus alternating misalignment, inside and core AOI (v3/scores)\n")
    w(f"Generated by analysis/analyze_alt.py on {res['created']}. Conventions in results.json ('conventions').\n")
    w("Measures. r = map mass in a figure AOI / (that mass + mass on the inducer-disc AOI), map clipped at 0. "
      "Inside AOI = figure polygon minus the inducer discs (the published measure). Core AOI (analysis/core_aoi.py) = "
      "inside-AOI pixels whose Euclidean distance to the inducer-disc AOI exceeds 30 px; core r = M(core) / (M(core) + "
      "M(discs)); core share = M(core) / M(whole image). Effects are means over the 60 configurations (square 30, "
      "triangle 30) with 95% cluster-bootstrap CIs (10,000 resamples over shape x support ratio x polarity cells, "
      "positions resampled within cells). Layer readouts: per-configuration average of 336 and 448 px. "
      "alt theta = mean of the two alternation phases. ratio = effect(misaligned) / effect(illusory), both against "
      "control180 (shown only when the aligned CI excludes 0). drop ratio = (m(illusory) - m(alt)) / (m(illusory) - "
      "m(mis)): the alternating reduction as a fraction of the uniform reduction. SR slope = change of the "
      "illusory - control180 effect per unit support ratio.\n")
    if res["missing"]:
        w(f"**Missing: {', '.join(res['missing'])}.**\n")
    # key question
    w("## Key question: DeepGaze IIE and MSDB, uniform versus alternating rotation\n")
    for name in ("DeepGaze IIE", "DeepGaze MSDB"):
        if name not in M:
            continue
        w(f"### {name}\n")
        w("| measure | subset | aligned effect | uniform ratio 10 / 20 / 30 | alternating ratio 10 / 20 / 30 | alt - uni ratio 20 | drop ratio 10 / 20 / 30 |")
        w("|---|---|---|---|---|---|---|")
        for m in PRIMARY_MEAS:
            for sub in ("all", "square", "triangle"):
                o = M[name]["results"][m][sub]
                if o["ratios_computed"]:
                    u = " / ".join(fr(o["ratio_uni"][t]) for t in THETAS); a = " / ".join(fr(o["ratio_alt"][t]) for t in THETAS)
                    d = fr(o["ratio_diff_alt_minus_uni"][20])
                else:
                    u = a = d = "n/a"
                dr = " / ".join(fr(o["drop_ratio"][t]) for t in THETAS)
                w(f"| {MEAS_LABEL[m]} | {sub} | {f3(o['aligned'])} | {u} | {a} | {d} | {dr} |")
        w("")
    # local-null expectation for the drop ratio
    w("### Local-null expectation for the drop ratio\n")
    w("Range (min to max of the point estimates) over the local nulls whose uniform drop CI excludes 0. A purely local "
      "mechanism that follows inner-sector ink, edges or corners predicts these values.\n")
    w("| measure | subset | drop ratio 10 | drop ratio 20 | drop ratio 30 |\n|---|---|---|---|---|")
    for m in PRIMARY_MEAS:
        for sub in ("all", "square", "triangle"):
            cells = []
            for t in THETAS:
                v = [rec["results"][m][sub]["drop_ratio"][t]["value"] for n, rec in M.items()
                     if rec["kind"] == "local null" and rec["results"][m][sub]["drop_ratio"][t] is not None]
                cells.append(f"{min(v):.2f} to {max(v):.2f} (n = {len(v)})" if v else "n/a")
            w(f"| {MEAS_LABEL[m]} | {sub} | " + " | ".join(cells) + " |")
    w("")
    # are the local nulls near zero on the core measures?
    w("## Local nulls on the core measures: largest absolute effect against control180\n")
    w("Inducer-only conditions = illusory, control90, mis10/20/30, alt10/20/30 (the core AOI holds no inducer ink in any "
      "of them; pixel-verified). real and real_filled put outline or fill ink into the core, so they are listed apart. "
      "For scale: the DeepGaze IIE and MSDB illusory effects are in the first rows.\n")
    ind_only = ["illusory", "control90", "mis10", "mis20", "mis30", "alt10", "alt20", "alt30"]
    w("| model | core r: max over inducer-only | core r: real, real_filled | core share: max over inducer-only | core share: real, real_filled |\n|---|---|---|---|---|")
    for name, rec in M.items():
        if rec["kind"] == "layer readout":
            continue
        row = []
        for m in ("r_core30", "f_core30"):
            e = rec["results"][m]["all"]["effects_vs_control180"]
            c = max(ind_only, key=lambda k: abs(e[k]["mean"]))
            row.append(f"{e[c]['mean']:+.3f} ({c})")
            row.append(f"{e['real']['mean']:+.3f}, {e['real_filled']['mean']:+.3f}")
        w(f"| {name} | " + " | ".join(row) + " |")
    w("")
    # summary per measure
    for m in PRIMARY_MEAS:
        w(f"## All models and nulls: {MEAS_LABEL[m]} (pooled over shapes)\n")
        w("| model | illusory - control180 | p | illusory - control90 | uniform ratio 20 | alternating ratio 20 | alt - uni ratio 20 | drop ratio 20 | SR slope (per unit SR) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for name, rec in M.items():
            o = rec["results"][m]["all"]
            ok = o["ratios_computed"]
            w(f"| {name} | {f3(o['aligned'])} | {o['aligned']['p_str']} | {f3(o['effects_vs_control90']['illusory'])} | "
              f"{fr(o['ratio_uni'][20]) if ok else 'n/a'} | {fr(o['ratio_alt'][20]) if ok else 'n/a'} | "
              f"{fr(o['ratio_diff_alt_minus_uni'][20]) if ok else 'n/a'} | {fr(o['drop_ratio'][20])} | "
              f"{o['sr_slope_vs_control180']['slope_per_unit_sr']:+.3f} [{o['sr_slope_vs_control180']['ci'][0]:+.3f}, {o['sr_slope_vs_control180']['ci'][1]:+.3f}] |")
        w("")
    # ratios at all angles, by shape, primary measures
    for m in PRIMARY_MEAS:
        w(f"## Ratios at 10, 20, 30 deg by shape: {MEAS_LABEL[m]}\n")
        w("| model | subset | uniform 10 | uniform 20 | uniform 30 | alternating 10 | alternating 20 | alternating 30 |")
        w("|---|---|---|---|---|---|---|---|")
        for name, rec in M.items():
            for sub in ("all", "square", "triangle"):
                o = rec["results"][m][sub]
                if not o["ratios_computed"]:
                    w(f"| {name} | {sub} | aligned {f3(o['aligned'])}: n/a | | | | | |"); continue
                w(f"| {name} | {sub} | " + " | ".join(fr(o["ratio_uni"][t]) for t in THETAS) + " | "
                  + " | ".join(fr(o["ratio_alt"][t]) for t in THETAS) + " |")
        w("")
    # effects of every condition vs control180 (core measures), models and nulls
    for m in ("r_core30", "f_core30", "r_inside"):
        w(f"## Effects of every condition against control180: {MEAS_LABEL[m]} (pooled)\n")
        cs = ["illusory", "control90", "real", "real_filled", "mis10", "mis20", "mis30", "alt10", "alt20", "alt30"]
        w("| model | " + " | ".join(cs) + " | max abs |\n|---|" + "---|" * (len(cs) + 1))
        for name, rec in M.items():
            e = rec["results"][m]["all"]["effects_vs_control180"]
            mx = max(abs(e[c]["mean"]) for c in cs)
            w(f"| {name} | " + " | ".join(f3(e[c]) for c in cs) + f" | {mx:.3f} |")
        w("")
    # SR slopes
    w("## Support-ratio dependence of the aligned effect (illusory - control180)\n")
    w("| model | measure | subset | slope per unit SR [95% CI] | p | mean at SR 0.3 / 0.5 / 0.7 |\n|---|---|---|---|---|---|")
    for name, rec in M.items():
        for m in PRIMARY_MEAS:
            for sub in ("all", "square", "triangle"):
                s = rec["results"][m][sub]["sr_slope_vs_control180"]
                w(f"| {name} | {MEAS_LABEL[m]} | {sub} | {s['slope_per_unit_sr']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['p_str']} | "
                  + " / ".join(f"{v:+.3f}" for v in s["mean_by_sr"].values()) + " |")
    w("")
    # per size
    w("## Layer readouts at each input size (pooled)\n")
    w("| readout | size | measure | aligned | uniform ratio 20 | alternating ratio 20 | drop ratio 20 |\n|---|---|---|---|---|---|---|")
    for name, rec in M.items():
        if rec["kind"] != "layer readout":
            continue
        for s, an in rec["per_size"].items():
            for m in PRIMARY_MEAS:
                o = an[m]["all"]; ok = o["ratios_computed"]
                w(f"| {name} | {s} | {MEAS_LABEL[m]} | {f3(o['aligned'])} | {fr(o['ratio_uni'][20]) if ok else 'n/a'} | "
                  f"{fr(o['ratio_alt'][20]) if ok else 'n/a'} | {fr(o['drop_ratio'][20])} |")
    w("")
    w("## Sensitivity: core AOI at 45 px and inside_eroded_all (pooled)\n")
    w("| model | measure | aligned | uniform ratio 20 | alternating ratio 20 | drop ratio 20 |\n|---|---|---|---|---|---|")
    for name, rec in M.items():
        for m in ("r_core45", "f_core45", "r_eroded_all"):
            o = rec["results"][m]["all"]; ok = o["ratios_computed"]
            w(f"| {name} | {MEAS_LABEL[m]} | {f3(o['aligned'])} | {fr(o['ratio_uni'][20]) if ok else 'n/a'} | "
              f"{fr(o['ratio_alt'][20]) if ok else 'n/a'} | {fr(o['drop_ratio'][20])} |")
    w("")
    w("## Checks\n")
    for k, v in res["checks"].items():
        if isinstance(v, list):
            for x in v:
                w(f"- {k}: {x}")
        else:
            w(f"- {k}: {v:.2e}")
    open(os.path.join(SC, "table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
