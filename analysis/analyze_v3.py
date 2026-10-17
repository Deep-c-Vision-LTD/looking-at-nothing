"""
Survey statistics on stim_v3 (inside effects, misalignment ratios, illusion indices), from the per-image scores
written by scoring/score_v3.py.

Per model (and, for the layer readouts, per input size 336, 448 and their mean):
  effects r(test) - r(primary control) for every condition, three AOI variants (inside = the published measure,
  inside_eroded = 8 px from the figure contour, inside_eroded_all), with a cluster bootstrap (10,000 resamples over
  shape x support ratio x polarity cells, positions resampled within cells; Ehrenstein: radius x line count x polarity);
  mass decomposition (share of total map mass inside the figure and on the inducers; logit(r) = log I - log D);
  section B relatability ratios effect(mis theta) / effect(aligned) with labels; section E illusion indices
  effect / (r(real_filled) - r(control)) where the denominator CI is above 0; cross-check against the first-version
  results (legacy/results) on the images both sets share.
Writes <results>/v2/scores/results.json and table.md. The float16 map check needs the maps of score_v3.py; without
them it is recorded as null.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as LP                                                              # noqa: E402
V2 = str(LP.V2); SC = os.path.join(V2, "scores"); LEG = str(LP.LEGACY_RESULTS)
RAW = os.path.join(SC, "raw")
ITEMS = json.load(open(os.path.join(str(LP.STIM_V3), "items.json")))
B, SEED = 10000, 2026
VARS = ("r_inside", "r_eroded", "r_eroded_all")
VAR_LABEL = {"r_inside": "inside", "r_eroded": "inside_eroded", "r_eroded_all": "inside_eroded_all"}
HUMAN = {"kanizsa": (0.78, 0.82, 0.54), "ehrenstein": (0.88, 0.91, 0.74)}      # Kavsek, pooled: illusory, real, control
HUMAN_INDEX = {k: (v[0] - v[2]) / (v[1] - v[2]) for k, v in HUMAN.items()}      # 0.857, 0.824
CTRL = {"kanizsa": "control180", "ehrenstein": "control_end"}
TESTS = {"kanizsa": [("illusory", "control180"), ("control90", "control180"), ("real", "control180"),
                     ("real_filled", "control180"), ("realC", "control180"), ("mis10", "control180"),
                     ("mis20", "control180"), ("mis30", "control180"), ("illusory", "control90")],
         "ehrenstein": [("illusory", "control_end"), ("real", "control_end"), ("real_filled", "control_end"),
                        ("realC", "control_end")]}
CONDS = {"kanizsa": ("illusory", "control180", "control90", "real", "real_filled", "realC", "mis10", "mis20", "mis30"),
         "ehrenstein": ("illusory", "control_end", "real", "real_filled", "realC")}

FIXED = [("DeepGaze IIE", "DeepGaze_IIE"), ("DeepGaze MSDB", "DeepGaze_MSDB"), ("MSI-Net", "MSI-Net"),
         ("xSoyelNet-ASPP", "xSoyelNet-ASPP"), ("xSoyelNet-ML", "xSoyelNet-ML"),
         ("Spectral residual", "Spectral_residual"), ("Edge-energy null", "Edge-energy_null")]
LAYERS = [("dinov2", "DINOv2-B", (2, 8, 12)), ("clip", "CLIP-L", (4, 12, 24)), ("siglip", "SigLIP-L", (4, 8, 24))]
OLD_NAME = {"DeepGaze IIE": "DeepGaze_IIE", "DeepGaze MSDB": "DeepGaze_MSDB", "MSI-Net": "MSI-Net",
            "xSoyelNet-ASPP": "xSoyelNet-ASPP", "xSoyelNet-ML": "xSoyelNet-ML", "Spectral residual": "Spectral_residual",
            "Edge-energy null": "Edge-energy_null"}


def pstr(p):
    return "p < .001" if p < 0.001 else ("p = " + f"{p:.3f}".lstrip("0"))


# ------------------------------------------------------------------ design and bootstrap
def design(ill):
    cells = {}
    for it in ITEMS:
        if it["illusion"] == ill and it["cond"] == "illusory":
            key = (it["shape"], it["sr"], it["pol"]) if ill == "kanizsa" else (it["R"], it["n"], it["pol"])
            cells.setdefault(key, []).append((it["pos"], it["config"]))
    keys = sorted(cells, key=str)
    return keys, [[c for _, c in sorted(cells[k])] for k in keys]

DESIGN = {ill: design(ill) for ill in ("kanizsa", "ehrenstein")}
IDX = {}
for ill, (_, cells) in DESIGN.items():
    K, P = len(cells), len(cells[0])
    assert all(len(c) == P for c in cells)
    rng = np.random.RandomState(SEED)                     # same generator and seed as v2/readout/analyze.py (A6)
    IDX[ill] = (rng.randint(0, K, (B, K)), rng.randint(0, P, (B, K, P)))


def bmeans(ill, vals):
    ci, pi = IDX[ill]
    return vals[ci[:, :, None], pi].mean((1, 2))


def summ(ill, vals, pct=True):
    bm = bmeans(ill, vals)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)))
    out = dict(mean=round(float(vals.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)], p=float(p), p_str=pstr(p))
    if pct:
        out["pct_pos"] = round(float(np.mean(vals > 0)) * 100, 1)
    return out


def ratio(ill, num, den):
    bn, bd = bmeans(ill, num), bmeans(ill, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.percentile(br, [2.5, 97.5])
    return dict(value=round(float(num.mean() / den.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                frac_boot_den_le0=round(float(np.mean(bd <= 0)), 4))


# ------------------------------------------------------------------ scores
def load(tag):
    p = os.path.join(RAW, tag + ".json")
    if not LP.json_exists(p):
        return None
    d = LP.load_json(p)
    return d if len(d) == len(ITEMS) else None


def arr(scs, ill, cond, key, fn=None):
    """[K, P] array of a per-image quantity, averaged over the score dicts in scs (input sizes)."""
    cells = DESIGN[ill][1]
    f = fn or (lambda rec: rec[key])
    return np.mean([[[f(sc[f"{c}_{cond}"]) for c in cell] for cell in cells] for sc in scs], 0)


def eff(scs, ill, test, ctrl, key):
    return arr(scs, ill, test, key) - arr(scs, ill, ctrl, key)


def log_comp(k):
    def f(rec):
        v = rec[k]
        return np.log(v) if v > 0 else np.nan
    return f


def analyse_variant(scs):
    out = {}
    for ill in ("kanizsa", "ehrenstein"):
        o = out[ill] = {"effects": {}, "levels": {}, "mass": {}}
        for test, ctrl in TESTS[ill]:
            nm = f"{test}-{ctrl}"
            o["effects"][nm] = {VAR_LABEL[v]: summ(ill, eff(scs, ill, test, ctrl, v)) for v in VARS}
            # mass decomposition: share of total map mass in the inside AOI and the inducer AOI; logit(r) = log I - log D
            m = {}
            for k in ("f_inside", "f_eroded", "f_ind"):
                m[k] = summ(ill, eff(scs, ill, test, ctrl, k))
            li = arr(scs, ill, test, None, log_comp("f_inside")) - arr(scs, ill, ctrl, None, log_comp("f_inside"))
            ld = arr(scs, ill, test, None, log_comp("f_ind")) - arr(scs, ill, ctrl, None, log_comp("f_ind"))
            if np.isfinite(li).all() and np.isfinite(ld).all():
                m["dlog_inside"] = summ(ill, li); m["dlog_ind"] = summ(ill, ld)
                m["dlogit_r"] = summ(ill, li - ld)
                m["share_of_dlogit_from_inside"] = round(float(li.mean() / (li - ld).mean()), 3) if abs((li - ld).mean()) > 1e-9 else None
            else:
                m["log_note"] = "zero mass in a region for some image; log decomposition skipped"
            o["mass"][nm] = m
        for c in CONDS[ill]:
            o["levels"][c] = {k: summ(ill, arr(scs, ill, c, k), pct=False) for k in ("r_inside", "r_eroded", "f_inside", "f_eroded", "f_ind")}
        # section E: illusion index
        ctrl = CTRL[ill]
        o["index"] = {}
        for v in VARS:
            num = eff(scs, ill, "illusory", ctrl, v)
            rec = {}
            for den_c in ("real_filled", "real"):
                den = eff(scs, ill, den_c, ctrl, v)
                d = summ(ill, den)
                r = dict(denominator=d, computed=bool(d["ci"][0] > 0))
                if r["computed"]:
                    r.update(ratio(ill, num, den))
                    h = HUMAN_INDEX[ill]
                    r["vs_human"] = "below" if r["ci"][1] < h else ("above" if r["ci"][0] > h else "includes")
                rec[den_c] = r
            o["index"][VAR_LABEL[v]] = rec
        # section B: relatability ratios (Kanizsa only)
        if ill == "kanizsa":
            o["section_B"] = {}
            for v in VARS:
                al = eff(scs, ill, "illusory", "control180", v)
                a = summ(ill, al)
                rec = dict(aligned=a, eligible=bool(a["ci"][0] > 0))
                if rec["eligible"]:
                    for th in (10, 20, 30):
                        rec[f"ratio_{th}"] = ratio(ill, eff(scs, ill, f"mis{th}", "control180", v), al)
                    r20 = rec["ratio_20"]
                    if r20["value"] <= 0.5 and r20["ci"][1] < 0.75:
                        rec["label"] = "relatability-tuned"
                    elif r20["ci"][0] >= 0.5:
                        rec["label"] = "inward-notch cue"
                    else:
                        rec["label"] = "intermediate"
                else:
                    rec["label"] = "not eligible (aligned Kanizsa effect CI not above 0)"
                o["section_B"][VAR_LABEL[v]] = rec
    return out


# ------------------------------------------------------------------ cross-check with the published results
def crosscheck(model, scs_by_size):
    """Per-image identity on the images shared with stim_v2 / stim_ehr, and effect means on the same configurations."""
    out = {}
    kz_cfgs = [c for cell in DESIGN["kanizsa"][1] for c in cell]
    eh_cfgs = [c for cell in DESIGN["ehrenstein"][1] for c in cell]
    if model in OLD_NAME:
        old_k = json.load(open(os.path.join(LEG, "kanizsa", f"results_{OLD_NAME[model]}.json")))
        old_e = json.load(open(os.path.join(LEG, "ehrenstein", f"results_{OLD_NAME[model]}.json")))
        new = scs_by_size["native"]
        pairs_k = [(f"{c}_{a}", f"{c}_{b}") for c in kz_cfgs for a, b in (("illusory", "illusory"), ("control", "control180"),
                                                                           ("control90", "control90"), ("real", "real"))]
        pairs_e = [(f"{c}_{a}", f"{c}_{a}") for c in eh_cfgs for a in ("illusory", "real", "control_end")]
        dk = np.array([new[n]["r_inside"] - old_k[o] for o, n in pairs_k]); de = np.array([new[n]["r_inside"] - old_e[o] for o, n in pairs_e])
        out["per_image"] = dict(kanizsa=dict(n=len(dk), max_abs=float(np.abs(dk).max()), mean_abs=float(np.abs(dk).mean())),
                                ehrenstein=dict(n=len(de), max_abs=float(np.abs(de).max()), mean_abs=float(np.abs(de).mean())))
        st = json.load(open(os.path.join(LEG, "stats.json")))["survey"][model]
        ek = np.mean([old_k[f"{c}_illusory"] - old_k[f"{c}_control"] for c in kz_cfgs])
        ek90 = np.mean([old_k[f"{c}_illusory"] - old_k[f"{c}_control90"] for c in kz_cfgs])
        ee = np.mean([old_e[f"{c}_illusory"] - old_e[f"{c}_control_end"] for c in eh_cfgs])
        by = st["kanizsa"]["by_sr"]; byn = st["ehrenstein"]["by_n"]
        out["effects"] = dict(
            kanizsa_illusory_control180=dict(stim_v3=float(np.mean([new[f"{c}_illusory"]["r_inside"] - new[f"{c}_control180"]["r_inside"] for c in kz_cfgs])),
                                             stim_v2_same_configs=float(ek),
                                             stats_json_by_sr_mean_030_050_070=float(np.mean([m for s, m in zip(by["sr"], by["mean"]) if s in (0.3, 0.5, 0.7)])),
                                             stats_json_all_120_configs=st["kanizsa"]["primary"]["mean"]),
            kanizsa_illusory_control90=dict(stim_v3=float(np.mean([new[f"{c}_illusory"]["r_inside"] - new[f"{c}_control90"]["r_inside"] for c in kz_cfgs])),
                                            stim_v2_same_configs=float(ek90), stats_json_all_120_configs=st["kanizsa"]["secondary"]["mean"]),
            ehrenstein_illusory_control_end=dict(stim_v3=float(np.mean([new[f"{c}_illusory"]["r_inside"] - new[f"{c}_control_end"]["r_inside"] for c in eh_cfgs])),
                                                 stim_ehr_same_configs=float(ee),
                                                 stats_json_by_n_mean_8_16=float(np.mean([m for n, m in zip(byn["n"], byn["mean"]) if n in (8, 16)])),
                                                 stats_json_all_120_configs=st["ehrenstein"]["primary"]["mean"]))
    else:
        bb, L = model
        sw = json.load(open(os.path.join(LEG, "layers", bb, "sweep_results.json")))
        eh = json.load(open(os.path.join(LEG, "layers", bb, "ehr_results.json")))
        layers = {r["layer"]: r for r in json.load(open(os.path.join(LEG, "stats.json")))["layers"][bb]}
        for si, s in enumerate((336, 448)):
            new = scs_by_size[str(s)]
            ok = sw["kanizsa"][f"L{L}_s{s}"]["raw"]; oe = eh[f"L{L}_s{s}"]["raw"]
            pk = [(f"{c}_{a}", f"{c}_{b}") for c in kz_cfgs for a, b in (("illusory", "illusory"), ("control", "control180"), ("control90", "control90"))]
            pe = [(f"{c}_{a}", f"{c}_{a}") for c in eh_cfgs for a in ("illusory", "control_end")]
            dk = np.array([new[n]["r_inside"] - ok[o] for o, n in pk]); de = np.array([new[n]["r_inside"] - oe[o] for o, n in pe])
            out[str(s)] = dict(
                per_image=dict(kanizsa=dict(n=len(dk), max_abs=float(np.abs(dk).max()), mean_abs=float(np.abs(dk).mean())),
                               ehrenstein=dict(n=len(de), max_abs=float(np.abs(de).max()), mean_abs=float(np.abs(de).mean()))),
                kanizsa_illusory_control180=dict(
                    stim_v3=float(np.mean([new[f"{c}_illusory"]["r_inside"] - new[f"{c}_control180"]["r_inside"] for c in kz_cfgs])),
                    sweep_raw_same_images=float(np.mean([ok[f"{c}_illusory"] - ok[f"{c}_control"] for c in kz_cfgs])),
                    stats_json_kanizsa_by_size=layers[L]["kanizsa_by_size"][si]),
                ehrenstein_illusory_control_end=dict(
                    stim_v3=float(np.mean([new[f"{c}_illusory"]["r_inside"] - new[f"{c}_control_end"]["r_inside"] for c in eh_cfgs])),
                    sweep_raw_same_images=float(np.mean([oe[f"{c}_illusory"] - oe[f"{c}_control_end"] for c in eh_cfgs])),
                    stats_json_all_120_configs=layers[L]["ehrenstein_by_size"][si]))
    return out


def fmap16_check(tag, n=12):
    """r recomputed from the saved float16 maps vs the float32 values in the score file (a sample of images); None when
    the maps (not distributed) are absent."""
    sc = load(tag)
    rng = np.random.RandomState(0); its = [ITEMS[i] for i in rng.choice(len(ITEMS), n, replace=False)]
    if not all(os.path.exists(os.path.join(SC, "maps", tag, it["name"] + ".npz")) for it in its):
        return None
    d = []
    for it in its:
        m = np.load(os.path.join(SC, "maps", tag, it["name"] + ".npz"))
        s = np.clip(m["sal"].astype(np.float64) * float(m["scale"]), 0, None)
        a = np.load(os.path.join(str(LP.STIM_V3), "img", it["aoi"]))
        I, D = s[a["inside"]].sum(), s[a["ind"]].sum()
        d.append(abs(I / (I + D) - sc[it["name"]]["r_inside"]))
    return float(max(d))


# ------------------------------------------------------------------ main
def main():
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), bootstrap=B, seed=SEED,
               human_index=HUMAN_INDEX, cells={ill: [str(k) for k in DESIGN[ill][0]] for ill in DESIGN},
               models={}, missing=[])
    for name, tag in FIXED:
        sc = load(tag)
        if sc is None:
            res["missing"].append(name); continue
        rec = dict(kind="model", tags=[tag], variants={"native": analyse_variant([sc])})
        rec["crosscheck"] = crosscheck(name, {"native": sc})
        rec["float16_map_check_max_abs_r"] = fmap16_check(tag)
        res["models"][name] = rec
    for bb, bbn, Ls in LAYERS:
        for L in Ls:
            name = f"{bbn} L{L}"
            sc = {s: load(f"{bb}_L{L}_s{s}") for s in ("336", "448")}
            if any(v is None for v in sc.values()):
                res["missing"].append(name); continue
            rec = dict(kind="layer readout", tags=[f"{bb}_L{L}_s{s}" for s in sc],
                       variants={"avg": analyse_variant([sc["336"], sc["448"]]), "336": analyse_variant([sc["336"]]),
                                 "448": analyse_variant([sc["448"]])})
            rec["crosscheck"] = crosscheck((bb, L), sc)
            chk = [fmap16_check(t) for t in rec["tags"]]
            rec["float16_map_check_max_abs_r"] = None if None in chk else max(chk)
            res["models"][name] = rec
    json.dump(res, open(os.path.join(SC, "results.json"), "w"), indent=1)
    write_table(res)
    print("models:", list(res["models"]), "missing:", res["missing"])


# ------------------------------------------------------------------ table.md
def f3(e):
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def f16(x):
    return "n/a (maps absent)" if x is None else f"{x:.1e}"


def primary_variant(rec):
    return rec["variants"]["native"] if "native" in rec["variants"] else rec["variants"]["avg"]


def write_table(res):
    L = []; w = L.append
    w("# Sections B and E on stim_v3: model scores\n")
    w(f"Generated by analysis/analyze_v3.py on {res['created']} from v2/scores/raw/*.json (scoring/score_v3.py).\n")
    w("r = saliency mass inside the figure / (inside + inducer discs or line annulus). Effect = r(test) - r(control); "
      "mean over configurations (Kanizsa 60, Ehrenstein 40) with a 95% cluster-bootstrap CI (10,000 resamples over "
      "shape x support ratio x polarity cells, or radius x line count x polarity, positions resampled within cells). "
      "Layer readouts: per-configuration effect averaged over 336 and 448 px (each size separately in results.json and at the end). "
      "All 740 maps per model run on the CPU.\n")
    if res["missing"]:
        w(f"**Missing (not yet scored): {', '.join(res['missing'])}.**\n")
    M = res["models"]
    w("## Summary (primary: inside AOI; layer readouts: mean of 336 and 448 px)\n")
    w("| model | Kanizsa illusory - control180 | p | illusory - control90 | Ehrenstein illusory - control_end | p | "
      "section B ratio(20) | section B label | Kanizsa index (real_filled) | Ehrenstein index (real_filled) |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, rec in M.items():
        v = primary_variant(rec); k = v["kanizsa"]; e = v["ehrenstein"]
        b = k["section_B"]["inside"]
        r20 = f"{b['ratio_20']['value']:.2f} [{b['ratio_20']['ci'][0]:.2f}, {b['ratio_20']['ci'][1]:.2f}]" if b["eligible"] else "n/a"
        ix = []
        for ill in ("kanizsa", "ehrenstein"):
            rf = v[ill]["index"]["inside"]["real_filled"]
            ix.append(f"{rf['value']:.2f} [{rf['ci'][0]:.2f}, {rf['ci'][1]:.2f}], {rf['vs_human']}" if rf["computed"]
                      else f"not computed (denominator {f3(rf['denominator'])})")
        ki, ee = k["effects"]["illusory-control180"]["inside"], e["effects"]["illusory-control_end"]["inside"]
        w(f"| {name} | {f3(ki)} | {ki['p_str']} | {f3(k['effects']['illusory-control90']['inside'])} | {f3(ee)} | {ee['p_str']} | "
          f"{r20} | {b['label'].split(' (')[0]} | {ix[0]} | {ix[1]} |")
    w("")
    for ill, tests in (("kanizsa", TESTS["kanizsa"]), ("ehrenstein", TESTS["ehrenstein"])):
        for var in ("inside", "inside_eroded", "inside_eroded_all"):
            w(f"## {ill.capitalize()} effects, {var} AOI\n")
            hdr = [f"{a} - {b}" for a, b in tests]
            w("| model | " + " | ".join(hdr) + " |\n|---|" + "---|" * len(hdr))
            for name, rec in M.items():
                v = primary_variant(rec)[ill]["effects"]
                w(f"| {name} | " + " | ".join(f3(v[f'{a}-{b}'][var]) for a, b in tests) + " |")
            w("")
        w(f"## {ill.capitalize()}: percentage of configurations with a positive effect (inside AOI)\n")
        hdr = [f"{a} - {b}" for a, b in tests]
        w("| model | " + " | ".join(hdr) + " |\n|---|" + "---|" * len(hdr))
        for name, rec in M.items():
            v = primary_variant(rec)[ill]["effects"]
            w(f"| {name} | " + " | ".join(f"{v[f'{a}-{b}']['inside']['pct_pos']:.0f}" for a, b in tests) + " |")
        w("")
    w("## Mass decomposition of the primary effect\n")
    w("Share of total map mass in the inside AOI (I) and on the inducers (D), and the exact per-image decomposition "
      "logit(r) = log I - log D. A negative r effect with dlog I < 0 and dlog D near 0 means less mass on the figure "
      "interior; with dlog D > 0 it means more mass on the inducers. Condition levels of I and D (below) show whether "
      "the control's interior is raised relative to the other inducer arrangements.\n")
    for ill, nm in (("kanizsa", "illusory-control180"), ("ehrenstein", "illusory-control_end")):
        w(f"### {ill.capitalize()} ({nm})\n")
        w("| model | effect r | d share I | d share D | dlog I | dlog D | dlogit r | part from I |\n|---|---|---|---|---|---|---|---|")
        for name, rec in M.items():
            v = primary_variant(rec)[ill]; m = v["mass"][nm]
            if "dlog_inside" in m:
                w(f"| {name} | {f3(v['effects'][nm]['inside'])} | {f3(m['f_inside'])} | {f3(m['f_ind'])} | {f3(m['dlog_inside'])} | "
                  f"{f3(m['dlog_ind'])} | {f3(m['dlogit_r'])} | {m['share_of_dlogit_from_inside']} |")
            else:
                w(f"| {name} | {f3(v['effects'][nm]['inside'])} | {f3(m['f_inside'])} | {f3(m['f_ind'])} | n/a | n/a | n/a | n/a |")
        w("")
        w(f"Condition levels, share of total map mass inside (I) / on the inducers (D), x 100:\n")
        conds = CONDS[ill]
        w("| model | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds))
        for name, rec in M.items():
            lv = primary_variant(rec)[ill]["levels"]
            w(f"| {name} | " + " | ".join(f"{100 * lv[c]['f_inside']['mean']:.2f} / {100 * lv[c]['f_ind']['mean']:.2f}" for c in conds) + " |")
        w("")
        w("Condition levels of r (inside AOI):\n")
        w("| model | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds))
        for name, rec in M.items():
            lv = primary_variant(rec)[ill]["levels"]
            w(f"| {name} | " + " | ".join(f"{lv[c]['r_inside']['mean']:.3f}" for c in conds) + " |")
        w("")
    w("## Section B: configural completion or inward-notch cue\n")
    w("ratio(theta) = effect(mis theta) / effect(aligned), both against control180; only where the aligned effect CI is "
      "above 0. Relatability-tuned: ratio(20) <= 0.5 and CI upper < 0.75; inward-notch cue: CI lower of ratio(20) >= 0.5; "
      "otherwise intermediate.\n")
    for var in ("inside", "inside_eroded", "inside_eroded_all"):
        w(f"### {var} AOI\n")
        w("| model | size | aligned effect | ratio 10 | ratio 20 | ratio 30 | label |\n|---|---|---|---|---|---|---|")
        for name, rec in M.items():
            for vn, v in rec["variants"].items():
                b = v["kanizsa"]["section_B"][var]
                if b["eligible"]:
                    rs = [f"{b[f'ratio_{t}']['value']:.2f} [{b[f'ratio_{t}']['ci'][0]:.2f}, {b[f'ratio_{t}']['ci'][1]:.2f}]" for t in (10, 20, 30)]
                else:
                    rs = ["n/a"] * 3
                w(f"| {name} | {vn} | {f3(b['aligned'])} | " + " | ".join(rs) + f" | {b['label']} |")
        w("")
    w("## Section E: illusion index\n")
    w(f"index = effect(illusory) / (r(real_filled) - r(control)), computed only where the denominator CI is above 0. "
      f"Human (Kavsek, pooled over ages, dominated by children, 97 of 127): Kanizsa {HUMAN_INDEX['kanizsa']:.2f}, "
      f"Ehrenstein {HUMAN_INDEX['ehrenstein']:.2f}. The outline-real denominator (the published index) is given for continuity.\n")
    for var in ("inside", "inside_eroded"):
        w(f"### {var} AOI\n")
        w("| model | size | Kanizsa denominator (real_filled) | Kanizsa index | vs human | Kanizsa index, outline real | "
          "Ehrenstein denominator (real_filled) | Ehrenstein index | vs human | Ehrenstein index, outline real |\n|---|---|---|---|---|---|---|---|---|---|")
        for name, rec in M.items():
            for vn, v in rec["variants"].items():
                row = []
                for ill in ("kanizsa", "ehrenstein"):
                    ix = v[ill]["index"][var]
                    rf, ro = ix["real_filled"], ix["real"]
                    row.append(f3(rf["denominator"]))
                    row.append(f"{rf['value']:.2f} [{rf['ci'][0]:.2f}, {rf['ci'][1]:.2f}]" if rf["computed"] else "not computed")
                    row.append(rf.get("vs_human", "n/a"))
                    row.append(f"{ro['value']:.2f} [{ro['ci'][0]:.2f}, {ro['ci'][1]:.2f}]" if ro["computed"] else "not computed")
                w(f"| {name} | {vn} | " + " | ".join(row) + " |")
        w("")
    w("## Cross-check against the published results (identical images)\n")
    w("Per-image r on the images shared with stim_v2 / stim_ehr (Kanizsa illusory, control, control90, real; Ehrenstein "
      "illusory, real, control_end for the survey models; illusory and controls for the sweep readouts), and the mean "
      "illusory - control effect on the same configurations.\n")
    w("| model | size | per-image max abs diff, Kanizsa (n) | per-image max abs diff, Ehrenstein (n) | Kanizsa effect: stim_v3 / old results same configs / stats.json | "
      "Ehrenstein effect: stim_v3 / old results same configs / stats.json | float16 map check |\n|---|---|---|---|---|---|---|")
    for name, rec in M.items():
        cx = rec["crosscheck"]
        if "per_image" in cx:
            e = cx["effects"]; k = e["kanizsa_illusory_control180"]; h = e["ehrenstein_illusory_control_end"]
            w(f"| {name} | native | {cx['per_image']['kanizsa']['max_abs']:.2e} ({cx['per_image']['kanizsa']['n']}) | "
              f"{cx['per_image']['ehrenstein']['max_abs']:.2e} ({cx['per_image']['ehrenstein']['n']}) | "
              f"{k['stim_v3']:+.4f} / {k['stim_v2_same_configs']:+.4f} / {k['stats_json_by_sr_mean_030_050_070']:+.4f} (SR .3/.5/.7; all 120: {k['stats_json_all_120_configs']:+.4f}) | "
              f"{h['stim_v3']:+.4f} / {h['stim_ehr_same_configs']:+.4f} / {h['stats_json_by_n_mean_8_16']:+.4f} (n 8/16; all 120: {h['stats_json_all_120_configs']:+.4f}) | "
              f"{f16(rec['float16_map_check_max_abs_r'])} |")
        else:
            for s in ("336", "448"):
                c = cx[s]; k = c["kanizsa_illusory_control180"]; h = c["ehrenstein_illusory_control_end"]
                w(f"| {name} | {s} | {c['per_image']['kanizsa']['max_abs']:.2e} ({c['per_image']['kanizsa']['n']}) | "
                  f"{c['per_image']['ehrenstein']['max_abs']:.2e} ({c['per_image']['ehrenstein']['n']}) | "
                  f"{k['stim_v3']:+.4f} / {k['sweep_raw_same_images']:+.4f} / {k['stats_json_kanizsa_by_size']:+.4f} | "
                  f"{h['stim_v3']:+.4f} / {h['sweep_raw_same_images']:+.4f} / (all 120: {h['stats_json_all_120_configs']:+.4f}) | "
                  f"{f16(rec['float16_map_check_max_abs_r'])} |")
    w("")
    w("## Layer readouts at each input size (inside AOI)\n")
    w("| readout | size | Kanizsa illusory - control180 | mis20 - control180 | real_filled - control180 | illusory - control90 | "
      "Ehrenstein illusory - control_end | real_filled - control_end |\n|---|---|---|---|---|---|---|---|")
    for name, rec in M.items():
        if rec["kind"] != "layer readout":
            continue
        for vn in ("336", "448"):
            v = rec["variants"][vn]; k = v["kanizsa"]["effects"]; e = v["ehrenstein"]["effects"]
            w(f"| {name} | {vn} | {f3(k['illusory-control180']['inside'])} | {f3(k['mis20-control180']['inside'])} | "
              f"{f3(k['real_filled-control180']['inside'])} | {f3(k['illusory-control90']['inside'])} | "
              f"{f3(e['illusory-control_end']['inside'])} | {f3(e['real_filled-control_end']['inside'])} |")
    w("")
    open(os.path.join(SC, "table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
