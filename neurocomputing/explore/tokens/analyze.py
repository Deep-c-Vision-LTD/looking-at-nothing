"""
neuro/reanalysis/explore/tokens copy of v2/encoding/analyze.py (RUN_PLAN.md S5; REANALYSIS_RULE.md Section 6 item 8 (i)).
Changes, nothing else: outputs (results.json, per_config_indices.npz, figures, report.md) go to
neuro/reanalysis/A3_tokens/published/ instead of v2/encoding/; common_enc is imported read only from v2/encoding;
require_public() (analyze_re, rule Section 8.1) is called at the top when run as a script; the matplotlib cache is kept inside neuro/.
Original docstring follows.
Section C statistics, rules and plots (v2/ANALYSIS_PLAN_v2.md section C; choices in v2/DEVIATIONS.md C1 to C14).
Reads tokens/<net>_s<S>/*.npz and tokens/cnn_<mode>/*.npz, the sweep readouts' per-image values
(sweep/<bb>/sweep_results.json, ehr_results.json); writes results.json, report.md and fig_*.png.
Usage: python analyze.py
"""
import json, os, sys, time, warnings
# ---- neuro/reanalysis copy (RUN_PLAN.md S5, 18 November 2026): output folder and import paths redirected ----
import os as _os, sys as _sys
_sys.dont_write_bytecode = True
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_RE = _os.path.dirname(_os.path.dirname(_HERE))                                   # neuro/reanalysis
_ROOT0 = _os.environ.get("LAN_PROJECT_ROOT") or _os.path.dirname(_os.path.dirname(_RE))
if _os.path.join(_ROOT0, "v2", "encoding") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_ROOT0, "v2", "encoding"))                   # common_enc.py (read only)
if _HERE in _sys.path:
    _sys.path.remove(_HERE)
_sys.path.insert(0, _HERE)                                                        # the copied plots.py, report.py
OUTDIR = _os.path.join(_RE, "A3_tokens", "published")
_os.makedirs(OUTDIR, exist_ok=True)
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(OUTDIR, "_mplconfig"))     # matplotlib cache inside neuro/
if __name__ == "__main__":                                                        # rule Section 8.1, before any read
    _sys.path.insert(1, _RE)
    import analyze_re as _AR
    _AR.require_public(_RE)
# ---- end of the copy header ----
from common_enc import *  # noqa: F401,F403
from common_enc import (ENC, TOK, ROOT, SIZES, K_TESTS, E_TESTS, PRIMARY, SWEEP_LAYERS, VIT, Boot, configs, cells,
                        pstr, B)
import numpy as np
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore", category=RuntimeWarning)
NETS = tuple(os.environ.get("ENC_NETS", "dinov2,clip,siglip").split(","))   # ENC_NETS: partial test runs only
NET_NAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}
TESTS = K_TESTS                                     # Ehrenstein uses test index 0 (illusory) only
TI = {t: i for i, t in enumerate(TESTS)}
CFGS = configs()
NCFG = len(CFGS)
KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
EI = [i for i, c in enumerate(CFGS) if c["illusion"] == "ehrenstein"]
BOOT = {"kanizsa": Boot(cells(CFGS, "kanizsa")), "ehrenstein": Boot(cells(CFGS, "ehrenstein"))}
SETS = ("int", "bg", "int_pertest")


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def summ(ill, vals):
    s = BOOT[ill].summary(vals)
    return dict(mean=r4(s["mean"]), ci=[r4(s["ci"][0]), r4(s["ci"][1])], p=s["p_str"], n=s["n"])


# ------------------------------------------------------------------ per-configuration indices
def per_config_vit(net, S):
    """E, cos: {set: [NCFG, L, T]}; counts: {set: [NCFG]}; diagnostics."""
    d = os.path.join(TOK, f"{net}_s{S}")
    nL = VIT[net]["n_hs"]; T = len(TESTS)
    E = {s: np.full((NCFG, nL, T), np.nan) for s in SETS}; C = {s: np.full((NCFG, nL, T), np.nan) for s in SETS}
    cnt = {s: np.zeros(NCFG, int) for s in ("int", "bg")}; cnt_pt = np.zeros((NCFG, T), int)
    maxshare = np.full((NCFG, nL), np.nan); l0 = 0.0
    for ci, cf in enumerate(CFGS):
        z = np.load(os.path.join(d, cf["config"] + ".npz"))
        imgs = list(z["imgs"]); tests = list(z["tests"]); same = z["same"]; ins = z["inside_full"]; far = z["far"]
        row = {c: imgs.index(c) - 1 for c in imgs[1:]}                 # rows of `same` (identity with the control)
        allsame = same.all(0)
        masks = {"int": allsame & ins, "bg": allsame & far}
        cnt["int"][ci] = masks["int"].sum(); cnt["bg"][ci] = masks["bg"].sum()
        dot, nt, nrc = z["dot"].astype(np.float64), z["nt"].astype(np.float64), z["nrc"].astype(np.float64)
        if masks["int"].any():
            l0 = max(l0, float(np.abs(dot[0][:, masks["int"]]).max()), float(nrc[0][masks["int"]].max()))
        for s in SETS:
            for j, t in enumerate(tests):
                if s == "int_pertest":
                    need = [t] + [c for c in ("control90", "realC") if c in row]
                    m = np.all([same[row[c]] for c in dict.fromkeys(need)], 0) & ins
                    cnt_pt[ci, TI[t]] = m.sum()
                else:
                    m = masks[s]
                if not m.any():
                    continue
                D = dot[:, j][:, m].sum(-1); N = nt[:, j][:, m].sum(-1); R = nrc[:, m].sum(-1)
                ok = R > 0
                E[s][ci, ok, TI[t]] = D[ok] / R[ok]
                C[s][ci, ok, TI[t]] = D[ok] / np.sqrt(N[ok] * R[ok])
        if masks["int"].any():
            R = nrc[:, masks["int"]]
            tot = R.sum(-1); ok = tot > 0
            maxshare[ci, ok] = R.max(-1)[ok] / tot[ok]
    return dict(E=E, cos=C, cnt=cnt, cnt_pertest=cnt_pt, maxshare=maxshare, layer0_maxabs=l0)


def size_avg(a, b):
    return np.nanmean(np.stack([a, b]), 0)


# ------------------------------------------------------------------ readout effects (sweep)
def readout_effects(bb):
    """{L: {size: [NCFG] effect}} for the sweep layers; Kanizsa vs control (=control180), Ehrenstein vs control_end."""
    k = json.load(open(os.path.join(ROOT, "sweep", bb, "sweep_results.json")))
    e = json.load(open(os.path.join(ROOT, "sweep", bb, "ehr_results.json")))
    out = {}
    for L in SWEEP_LAYERS[bb]:
        out[L] = {}
        for s in SIZES:
            v = np.full(NCFG, np.nan)
            rk = k["kanizsa"][f"L{L}_s{s}"]["raw"]; re_ = e[f"L{L}_s{s}"]["raw"]
            for i, c in enumerate(CFGS):
                if c["illusion"] == "kanizsa":
                    v[i] = rk[c["config"] + "_illusory"] - rk[c["config"] + "_control"]
                else:
                    v[i] = re_[c["config"] + "_illusory"] - re_[c["config"] + "_control_end"]
            out[L][s] = v
        out[L]["avg"] = (out[L][336] + out[L][448]) / 2
    return out


# ------------------------------------------------------------------ rules
def layer_block(ill, E, C, L, test="illusory"):
    """Summaries and the figure-specific rule for one layer and one test."""
    t = TI[test]
    r = dict(interior=summ(ill, E["int"][:, L, t]), bg_null=summ(ill, E["bg"][:, L, t]),
             cos_interior=summ(ill, C["int"][:, L, t]), cos_bg_null=summ(ill, C["bg"][:, L, t]))
    if ill == "kanizsa":
        r["c90_null"] = summ(ill, E["int"][:, L, TI["control90"]])
        r["cos_c90_null"] = summ(ill, C["int"][:, L, TI["control90"]])
    def fs(a, nulls):
        if a["ci"][0] is None or any(n["ci"][1] is None for n in nulls):
            return None
        return bool(all(a["ci"][0] > n["ci"][1] for n in nulls))
    nulls = [r["bg_null"]] + ([r["c90_null"]] if ill == "kanizsa" else [])
    cnulls = [r["cos_bg_null"]] + ([r["cos_c90_null"]] if ill == "kanizsa" else [])
    r["figure_specific"] = fs(r["interior"], nulls)
    r["figure_specific_cos"] = fs(r["cos_interior"], cnulls)
    r["pertest_interior"] = summ(ill, E["int_pertest"][:, L, t])
    return r


def relatability(E, L, blk):
    """Kanizsa ratios of interior E (misaligned / aligned), only where aligned is figure-specific and above 0."""
    if not (blk["figure_specific"] and blk["interior"]["ci"][0] > 0):
        return None
    out = {}
    for th in (10, 20, 30):
        rr = BOOT["kanizsa"].ratio(E["int"][:, L, TI[f"mis{th}"]], E["int"][:, L, TI["illusory"]])
        out[f"ratio{th}"] = dict(ratio=r4(rr["ratio"]), ci=[r4(rr["ci"][0]), r4(rr["ci"][1])], n=rr["n"])
    q = out["ratio20"]
    out["label_plan"] = "figure code (relatability-tuned)" if q["ratio"] < 0.5 else "inward-notch code"
    out["label_B_thresholds"] = ("relatability-tuned" if (q["ratio"] <= 0.5 and q["ci"][1] < 0.75)
                                 else "inward-notch" if q["ci"][0] >= 0.5 else "intermediate")
    return out


# ------------------------------------------------------------------ Spearman
def spearman_boot(ill, x, y):
    """Across configurations, with the cluster bootstrap (rho recomputed on each resample)."""
    idx = KI if ill == "kanizsa" else EI
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~np.isnan(x[idx]) & ~np.isnan(y[idx])
    rho = spearmanr(x[idx][ok], y[idx][ok]).statistic
    bs = []
    for row in BOOT[ill].flat:
        xa, ya = x[row], y[row]; m = ~np.isnan(xa) & ~np.isnan(ya)
        if m.sum() > 3 and np.ptp(xa[m]) > 0 and np.ptp(ya[m]) > 0:
            bs.append(spearmanr(xa[m], ya[m]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(rho=r4(rho), ci=[r4(lo), r4(hi)], n=int(ok.sum()))


def spearman_layers(ill, Eavg, eff, layers):
    """Within one network across its sweep layers: layer means, bootstrap over configurations."""
    t = TI["illusory"]
    idx = KI if ill == "kanizsa" else EI
    mx = [np.nanmean(Eavg[idx, L, t]) for L in layers]; my = [np.nanmean(eff[L]["avg"][idx]) for L in layers]
    rho = spearmanr(mx, my).statistic
    bx = np.stack([BOOT[ill].dist(Eavg[:, L, t]) for L in layers], 1)
    by = np.stack([BOOT[ill].dist(eff[L]["avg"]) for L in layers], 1)
    rx, ry = rankdata(bx, axis=1), rankdata(by, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    bs = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    lo, hi = np.percentile(bs[np.isfinite(bs)], [2.5, 97.5])
    return dict(rho=r4(rho), ci=[r4(lo), r4(hi)], n_layers=len(layers), layers=list(layers),
                layer_mean_E=[r4(v) for v in mx], layer_mean_readout=[r4(v) for v in my])


# ------------------------------------------------------------------ CNN
def per_config_cnn(mode):
    d = os.path.join(TOK, f"cnn_{mode}")
    z0 = np.load(os.path.join(d, CFGS[0]["config"] + ".npz"))
    layers = [k.split("|")[1] for k in z0.files if k.startswith("nrc|")]
    T = len(TESTS)
    sets = ("rf_int", "rf_bg", "cell_int", "cell_bg")
    E = {s: np.full((NCFG, len(layers), T), np.nan) for s in sets}
    cnt = {s: np.zeros((NCFG, len(layers)), int) for s in sets}
    ndef = {s: np.zeros((NCFG, len(layers)), bool) for s in sets}
    maxabs = 0.0; maxnum = 0.0; geom = {}
    for ci, cf in enumerate(CFGS):
        z = np.load(os.path.join(d, cf["config"] + ".npz"))
        tests = list(z["tests"])
        for li, L in enumerate(layers):
            rc, cs, cin, cfar = (z[f"{n}|{L}"] for n in ("rf_clean", "cell_same", "cell_inside", "cell_far"))
            geom[L] = [int(v) for v in z[f"shape|{L}"]]
            maxabs = max(maxabs, float(z[f"maxabs_dtest_rfclean|{L}"]))
            M = {"rf_int": rc & cin, "rf_bg": rc & cfar, "cell_int": cs & cin, "cell_bg": cs & cfar}
            dot, nrc = z[f"dot|{L}"].astype(np.float64), z[f"nrc|{L}"].astype(np.float64)
            for s, m in M.items():
                cnt[s][ci, li] = m.sum()
                if not m.any():
                    continue
                R = nrc[m].sum()
                if s.startswith("rf"):
                    maxnum = max(maxnum, float(np.abs(dot[:, m].sum(-1)).max()))
                if R > 0:
                    ndef[s][ci, li] = True
                    for j, t in enumerate(tests):
                        E[s][ci, li, TI[t]] = dot[j, m].sum() / R
    return dict(layers=layers, E=E, cnt=cnt, defined=ndef, maxabs_dtest_rfclean=maxabs, max_abs_numerator_rf=maxnum,
                geom=geom)


def cnn_summary(mode):
    P = per_config_cnn(mode)
    out = dict(layers=P["layers"], geometry={L: dict(zip(("h", "w", "rf", "jump"), g)) for L, g in P["geom"].items()},
               max_abs_activation_difference_at_rf_clean_units=P["maxabs_dtest_rfclean"],
               max_abs_index_numerator_at_rf_clean_sets=P["max_abs_numerator_rf"], per_layer={})
    for li, L in enumerate(P["layers"]):
        rec = {}
        for ill, idx in (("kanizsa", KI), ("ehrenstein", EI)):
            r = {}
            for s in ("rf_int", "rf_bg", "cell_int", "cell_bg"):
                c = P["cnt"][s][idx, li]
                r[f"n_units_{s}"] = dict(min=int(c.min()), median=float(np.median(c)), max=int(c.max()),
                                         configs_with_units=int((c > 0).sum()),
                                         configs_defined=int(P["defined"][s][idx, li].sum()))
            r["plan_interior"] = summ(ill, P["E"]["rf_int"][:, li, 0])
            r["plan_bg_null"] = summ(ill, P["E"]["rf_bg"][:, li, 0])
            r["cell_interior"] = summ(ill, P["E"]["cell_int"][:, li, 0])
            r["cell_bg_null"] = summ(ill, P["E"]["cell_bg"][:, li, 0])
            nulls = [r["plan_bg_null"]]; cnulls = [r["cell_bg_null"]]
            if ill == "kanizsa":
                r["plan_c90_null"] = summ(ill, P["E"]["rf_int"][:, li, TI["control90"]])
                r["cell_c90_null"] = summ(ill, P["E"]["cell_int"][:, li, TI["control90"]])
                r["cell_mis20"] = summ(ill, P["E"]["cell_int"][:, li, TI["mis20"]])
                if r["cell_interior"]["ci"][0] is not None and r["cell_interior"]["ci"][0] > 0:
                    rr = BOOT[ill].ratio(P["E"]["cell_int"][:, li, TI["mis20"]], P["E"]["cell_int"][:, li, TI["illusory"]])
                    r["cell_ratio20"] = dict(ratio=r4(rr["ratio"]), ci=[r4(rr["ci"][0]), r4(rr["ci"][1])], n=rr["n"])
                nulls.append(r["plan_c90_null"]); cnulls.append(r["cell_c90_null"])
            def fs(a, nl):
                if a["ci"][0] is None:
                    return "no defined configuration"
                if any(n["ci"][1] is None for n in nl):
                    return "null undefined"
                return bool(all(a["ci"][0] > n["ci"][1] for n in nl))
            r["figure_specific_plan"] = fs(r["plan_interior"], nulls)
            r["figure_specific_cell_descriptive"] = fs(r["cell_interior"], cnulls)
            rec[ill] = r
        out["per_layer"][L] = rec
    return out, P


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), bootstrap=B, plan_sha256_prefix="85a84c25",
               definitions="see v2/DEVIATIONS.md C1-C14", networks={}, cnn={}, spearman={}, encoded_not_prioritised={},
               title_rule={})
    keep = {}
    for net in NETS:
        nL = VIT[net]["n_hs"]
        PC = {S: per_config_vit(net, S) for S in SIZES}
        Eavg = {s: size_avg(PC[336]["E"][s], PC[448]["E"][s]) for s in SETS}
        Cavg = {s: size_avg(PC[336]["cos"][s], PC[448]["cos"][s]) for s in SETS}
        variants = {"avg": (Eavg, Cavg), "336": (PC[336]["E"], PC[336]["cos"]), "448": (PC[448]["E"], PC[448]["cos"])}
        keep[net] = dict(Eavg=Eavg, PC=PC)
        R = dict(n_layers=nL, tokens={}, layer0_check={}, kanizsa={}, ehrenstein={})
        for S in SIZES:
            p = PC[S]
            R["layer0_check"][str(S)] = dict(max_abs_component_at_interior_tokens=p["layer0_maxabs"])
            tk = {}
            for ill, idx in (("kanizsa", KI), ("ehrenstein", EI)):
                ci_, cb = p["cnt"]["int"][idx], p["cnt"]["bg"][idx]
                pt = p["cnt_pertest"][idx, 0]
                tk[ill] = dict(interior_common=dict(min=int(ci_.min()), median=float(np.median(ci_)), max=int(ci_.max()),
                                                    total=int(ci_.sum()), configs_with_none=int((ci_ == 0).sum()),
                                                    by_config={CFGS[i]["config"]: int(p["cnt"]["int"][i]) for i in idx}),
                               interior_pertest_illusory=dict(min=int(pt.min()), median=float(np.median(pt)), max=int(pt.max()),
                                                              total=int(pt.sum()), configs_with_none=int((pt == 0).sum())),
                               background=dict(min=int(cb.min()), median=float(np.median(cb)), max=int(cb.max()),
                                               configs_with_none=int((cb == 0).sum())))
                if ill == "kanizsa":
                    for t in ("mis10", "mis20", "mis30"):
                        q = p["cnt_pertest"][idx, TI[t]]
                        tk[ill][f"interior_pertest_{t}"] = dict(total=int(q.sum()), configs_with_none=int((q == 0).sum()))
                # token-domination diagnostic: share of the outline-direction energy from the single largest token
                ms = p["maxshare"][idx]
                tk[ill]["max_token_share_of_rc_energy_mean_by_layer"] = [r4(np.nanmean(ms[:, L])) for L in range(nL)]
            R["tokens"][str(S)] = tk
        for ill in ("kanizsa", "ehrenstein"):
            for vname, (E, C) in variants.items():
                layers = []
                for L in range(nL):
                    blk = layer_block(ill, E, C, L)
                    blk["layer"] = L; blk["depth"] = round(L / (nL - 1), 3)
                    if ill == "kanizsa":
                        blk["mis"] = {t: summ(ill, E["int"][:, L, TI[t]]) for t in ("mis10", "mis20", "mis30")}
                        blk["mis_figure_specific"] = {t: layer_block(ill, E, C, L, t)["figure_specific"] for t in ("mis10", "mis20", "mis30")}
                        blk["relatability"] = relatability(E, L, blk)
                    layers.append(blk)
                R[ill][vname] = layers
        res["networks"][net] = R
        print(f"{net} done {time.time() - t0:.0f}s", flush=True)

    # readouts: encoded but not prioritised; Spearman
    for net in NETS:
        eff = readout_effects(net); Eavg = keep[net]["Eavg"]["int"]
        enp = {}; sp = {"per_primary_layer_across_configs": {}, "across_sweep_layers": {}}
        for ill, idx in (("kanizsa", KI), ("ehrenstein", EI)):
            enp[ill] = {}
            for L in SWEEP_LAYERS[net]:
                ro = {k: summ(ill, eff[L][k]) for k in ("avg", 336, 448)}
                blk = res["networks"][net][ill]["avg"][L]
                not_above = ro["avg"]["ci"][0] <= 0; strict = ro["avg"]["ci"][1] < 0
                rel = blk.get("relatability")
                rel_ok = None if ill == "ehrenstein" else bool(rel and rel["ratio20"]["ratio"] < 0.5)
                claim = (bool(blk["figure_specific"]) and not_above and bool(rel_ok)) if ill == "kanizsa" else None
                enp[ill][f"L{L}"] = dict(readout_effect=ro, readout_ci_not_above_0=bool(not_above),
                                         readout_ci_below_0_strict=bool(strict), figure_specific=blk["figure_specific"],
                                         ratio20_below_half=rel_ok,
                                         encoded_not_prioritised=claim if ill == "kanizsa" else "not evaluable (no misaligned condition)",
                                         encoded_not_prioritised_strict=(claim and strict) if ill == "kanizsa" else None)
            for L in PRIMARY[net]:
                sp["per_primary_layer_across_configs"].setdefault(ill, {})[f"L{L}"] = spearman_boot(ill, Eavg[:, L, 0], eff[L]["avg"])
            sp["across_sweep_layers"][ill] = spearman_layers(ill, Eavg, eff, SWEEP_LAYERS[net])
        res["encoded_not_prioritised"][net] = enp; res["spearman"][net] = sp
        print(f"{net} readouts done {time.time() - t0:.0f}s", flush=True)

    # title rule: middle primary layer of each network (C11)
    mid = {"dinov2": 8, "clip": 12, "siglip": 8}
    tr = {}
    for net in NETS:
        blk = res["networks"][net]["kanizsa"]["avg"][mid[net]]
        rel = blk["relatability"]
        tr[net] = dict(layer=mid[net], figure_specific=blk["figure_specific"],
                       relatability_tuned_plan=bool(rel and rel["label_plan"].startswith("figure code")),
                       middle_band_layers={str(b["layer"]): dict(fs=b["figure_specific"],
                                                                 rel=(b["relatability"] or {}).get("label_plan"))
                                           for b in res["networks"][net]["kanizsa"]["avg"]
                                           if 1 / 3 - 1e-9 <= b["layer"] / (VIT[net]["n_hs"] - 1) <= 2 / 3 + 1e-9})
    tr["all_three"] = len(NETS) == 3 and all(tr[n]["figure_specific"] and tr[n]["relatability_tuned_plan"] for n in NETS)
    res["title_rule"] = tr

    # per-configuration indices (for later re-analysis without the token files)
    arrs = dict(configs=np.array([c["config"] for c in CFGS]), tests=np.array(TESTS))
    for net in NETS:
        PC = keep[net]["PC"]
        for s in SETS:
            for S in SIZES:
                arrs[f"{net}_s{S}_{s}_E"] = PC[S]["E"][s].astype(np.float32)
                arrs[f"{net}_s{S}_{s}_cos"] = PC[S]["cos"][s].astype(np.float32)
            arrs[f"{net}_avg_{s}_E"] = keep[net]["Eavg"][s].astype(np.float32)
    pc_path = os.path.join(OUTDIR, "per_config_indices.npz")                     # copy: redirected
    np.savez_compressed(pc_path, **arrs)
    cnnP = {}
    for mode in ("iie", "sq336", "sq448"):
        if os.path.exists(os.path.join(TOK, f"cnn_{mode}", "DONE")):
            res["cnn"][mode], cnnP[mode] = cnn_summary(mode)
    out = os.path.join(OUTDIR, "results.json")                                    # copy: redirected
    json.dump(res, open(out, "w"), indent=1)
    print(f"results.json written {time.time() - t0:.0f}s", flush=True)
    import plots
    plots.make_all(res, keep, cnnP)
    import report
    report.write(res)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
