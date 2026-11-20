"""neuro/reanalysis/explore/tokens copy of v2/encoding/report.py (RUN_PLAN.md S5): output folder redirected to
neuro/reanalysis/A3_tokens/published/; common_enc read only from v2/encoding. Original docstring follows.
Section C report (called by analyze.py): report.md from results.json; report_notes.md (hand-written summary) is
included at the top if present."""
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
# ---- end of the copy header ----

import os
from common_enc import ENC as _ENC, PRIMARY, SWEEP_LAYERS
ENC = OUTDIR                                         # copy: redirected (was ENC_FIGDIR or v2/encoding)
NAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}


def f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}"


def ci(r, d=3):
    if r is None or r.get("mean") is None:
        return "n/a"
    return f"{f(r['mean'], d)} [{f(r['ci'][0], d)}, {f(r['ci'][1], d)}]"


def yn(v):
    return "n/a" if v is None else ("**yes**" if v is True else ("no" if v is False else str(v)))


def fs_layers(blocks, key="figure_specific"):
    return [b["layer"] for b in blocks if b.get(key)]


def write(res):
    L = []
    w = L.append
    w("# Section C: readout-free encoding index (stim_v3)\n")
    w(f"Generated {res['created']} by v2/encoding/analyze.py from v2/encoding/tokens/. Plan: v2/ANALYSIS_PLAN_v2.md "
      f"(sha256 85a84c25...), section C; implementation choices: v2/DEVIATIONS.md C1 to C14. Bootstrap: "
      f"{res['bootstrap']:,} cluster resamples (cells = shape x support ratio x polarity for Kanizsa, radius x line count x "
      f"polarity for Ehrenstein; positions resampled within cells), percentile 95% CIs. "
      f"E = sum<f(test) - f(ctrl), f(realC) - f(ctrl)> / sum||f(realC) - f(ctrl)||^2 over the configuration's qualifying "
      f"tokens; statistic = mean over configurations; primary = per-configuration E averaged over 336 and 448 px.\n")
    notes = os.path.join(ENC, "report_notes.md")
    if os.path.exists(notes):
        w(open(notes).read().strip() + "\n")
    nets = list(res["networks"])

    w("## 1. Figure-specific layers at a glance (size-averaged)\n")
    w("| network | Kanizsa: figure-specific layers (E) | Kanizsa (cos) | Ehrenstein, background null only (E) |")
    w("|---|---|---|---|")
    for n in nets:
        R = res["networks"][n]
        w(f"| {NAME[n]} | {fs_layers(R['kanizsa']['avg']) or 'none'} | {fs_layers(R['kanizsa']['avg'], 'figure_specific_cos') or 'none'} "
          f"| {fs_layers(R['ehrenstein']['avg']) or 'none'} |")
    w("")
    w("Size sensitivity (layers figure-specific at 336 px / 448 px separately, Kanizsa E):\n")
    for n in nets:
        R = res["networks"][n]
        w(f"- {NAME[n]}: 336 px {fs_layers(R['kanizsa']['336']) or 'none'}; 448 px {fs_layers(R['kanizsa']['448']) or 'none'}; "
          f"Ehrenstein 336 px {fs_layers(R['ehrenstein']['336']) or 'none'}, 448 px {fs_layers(R['ehrenstein']['448']) or 'none'}")
    w("")

    w("## 2. Qualifying tokens\n")
    w("Interior = pixel-identical patch across all section C images of the configuration and footprint inside the "
      "'inside' AOI (one common set per configuration). 'Per-test' = the plan's literal set for the illusory test "
      "(identity across illusory, control180, control90, realC). Background = identical patch and footprint > 80 px "
      "from the figure. Configurations with no interior token are skipped (undefined) at every layer.\n")
    w("| network | size | Kanizsa interior per config min / median / max | Kanizsa configs with none | common vs per-test total | "
      "Kanizsa background median | Ehrenstein interior min / median / max | Ehrenstein configs with none | Ehrenstein background median |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in nets:
        for s, T in res["networks"][n]["tokens"].items():
            k, e = T["kanizsa"], T["ehrenstein"]
            ki, ei = k["interior_common"], e["interior_common"]
            w(f"| {NAME[n]} | {s} | {ki['min']} / {ki['median']:g} / {ki['max']} | {ki['configs_with_none']} | "
              f"{ki['total']} vs {k['interior_pertest_illusory']['total']} | {k['background']['median']:g} | "
              f"{ei['min']} / {ei['median']:g} / {ei['max']} | {ei['configs_with_none']} | {e['background']['median']:g} |")
    w("")
    w("Layer-0 check (embedding layer; pixel-identical patches must give exactly identical embeddings): maximum absolute "
      "difference component at interior tokens = " + ", ".join(
          f"{NAME[n]} {s} px: {v['max_abs_component_at_interior_tokens']:g}" for n in nets
          for s, v in res["networks"][n]["layer0_check"].items()) + ".\n")

    for n in nets:
        R = res["networks"][n]
        w(f"## 3. {NAME[n]}: Kanizsa, all layers (size-averaged; test = illusory vs control180)\n")
        w("FS = figure-specific (interior CI lower bound above the upper bounds of both nulls). ratio20 = E(mis20) / "
          "E(illusory), only where the aligned index is figure-specific and above 0. Primary layers in bold.\n")
        w("| layer | depth | interior E | null: control90 | null: background | FS (E) | FS (cos) | interior cos | mis20 interior E | ratio20 [CI] | plan label | B-threshold label |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in R["kanizsa"]["avg"][1:]:
            rl = b.get("relatability")
            lay = f"**{b['layer']}**" if b["layer"] in PRIMARY[n] else str(b["layer"])
            rtxt = f"{rl['ratio20']['ratio']:.2f} [{rl['ratio20']['ci'][0]:.2f}, {rl['ratio20']['ci'][1]:.2f}]" if rl else "n/a"
            w(f"| {lay} | {b['depth']:.2f} | {ci(b['interior'])} | {ci(b['c90_null'])} | {ci(b['bg_null'])} | {yn(b['figure_specific'])} | "
              f"{yn(b['figure_specific_cos'])} | {ci(b['cos_interior'])} | {ci(b['mis']['mis20'])} | {rtxt} | "
              f"{rl['label_plan'] if rl else 'n/a'} | {rl['label_B_thresholds'] if rl else 'n/a'} |")
        w("")
        w(f"{NAME[n]}: misaligned tests that are themselves figure-specific (vs control90 and their own background null): " +
          "; ".join(f"{t}: {[b['layer'] for b in R['kanizsa']['avg'] if b['mis_figure_specific'][t]] or 'none'}"
                    for t in ("mis10", "mis20", "mis30")) + ".\n")
        rel = [(b["layer"], b["relatability"]) for b in R["kanizsa"]["avg"] if b.get("relatability")]
        if rel:
            w(f"{NAME[n]} ratios at figure-specific layers (10 / 20 / 30 deg): " + "; ".join(
                f"L{l}: {r['ratio10']['ratio']:.2f} / {r['ratio20']['ratio']:.2f} / {r['ratio30']['ratio']:.2f}" for l, r in rel) + ".\n")
        w(f"## 4. {NAME[n]}: Ehrenstein, all layers (size-averaged; illusory vs control_end; background null only)\n")
        w("| layer | depth | interior E | null: background | FS (E) | interior cos | null cos | FS (cos) |")
        w("|---|---|---|---|---|---|---|---|")
        for b in R["ehrenstein"]["avg"][1:]:
            lay = f"**{b['layer']}**" if b["layer"] in PRIMARY[n] else str(b["layer"])
            w(f"| {lay} | {b['depth']:.2f} | {ci(b['interior'])} | {ci(b['bg_null'])} | {yn(b['figure_specific'])} | "
              f"{ci(b['cos_interior'])} | {ci(b['cos_bg_null'])} | {yn(b['figure_specific_cos'])} |")
        w("")
        w(f"Token-domination diagnostic, {NAME[n]} (mean share of the interior outline-direction energy carried by the "
          f"single largest token, primary layers): " + "; ".join(
              f"{s} px " + ", ".join(f"L{l} {R['tokens'][s][ill]['max_token_share_of_rc_energy_mean_by_layer'][l]:.2f} ({ill[:3]})"
                                     for ill in ("kanizsa", "ehrenstein") for l in PRIMARY[n]) for s in R["tokens"]) + ".\n")

    w("## 5. Encoded but not prioritised (sweep readouts)\n")
    w("Readout effect = r(illusory) - r(control180) of the sweep readout at that layer (stim_v2 per-image values, 336 and "
      "448 px averaged). Claim (Kanizsa) = figure-specific encoding at the layer, readout CI not above 0, and ratio20 < 0.5. "
      "'strict' additionally needs the readout CI entirely below 0. Ehrenstein: the relatability condition cannot be "
      "evaluated (no misaligned condition), so only the first two conditions are shown.\n")
    w("| network | layer | Kanizsa readout effect [CI] | FS | ratio20 < 0.5 | claim | strict | Ehrenstein readout effect [CI] | FS (bg null only) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in nets:
        enp = res["encoded_not_prioritised"].get(n)
        if not enp:
            continue
        for Lk, r in enp["kanizsa"].items():
            e = enp["ehrenstein"][Lk]
            w(f"| {NAME[n]} | {Lk} | {ci(r['readout_effect']['avg'])} | {yn(r['figure_specific'])} | {yn(r['ratio20_below_half'])} | "
              f"{yn(r['encoded_not_prioritised'])} | {yn(r['encoded_not_prioritised_strict'])} | {ci(e['readout_effect']['avg'])} | {yn(e['figure_specific'])} |")
    w("")

    w("## 6. Encoding index vs sweep readout effect (descriptive Spearman)\n")
    w("(a) At each primary layer, across configurations (size-averaged interior E(illusory) vs readout effect).\n")
    w("| network | layer | Kanizsa rho [CI] (n) | Ehrenstein rho [CI] (n) |")
    w("|---|---|---|---|")
    for n in nets:
        sp = res["spearman"].get(n)
        if not sp:
            continue
        for Lk in sp["per_primary_layer_across_configs"]["kanizsa"]:
            a = sp["per_primary_layer_across_configs"]["kanizsa"][Lk]; b = sp["per_primary_layer_across_configs"]["ehrenstein"][Lk]
            w(f"| {NAME[n]} | {Lk} | {f(a['rho'], 2)} [{f(a['ci'][0], 2)}, {f(a['ci'][1], 2)}] ({a['n']}) | "
              f"{f(b['rho'], 2)} [{f(b['ci'][0], 2)}, {f(b['ci'][1], 2)}] ({b['n']}) |")
    w("")
    w("(b) Within each network across its six sweep layers (layer means; bootstrap over configurations).\n")
    w("| network | Kanizsa rho [CI] | Ehrenstein rho [CI] | layers |")
    w("|---|---|---|---|")
    for n in nets:
        sp = res["spearman"].get(n)
        if not sp:
            continue
        a, b = sp["across_sweep_layers"]["kanizsa"], sp["across_sweep_layers"]["ehrenstein"]
        w(f"| {NAME[n]} | {f(a['rho'], 2)} [{f(a['ci'][0], 2)}, {f(a['ci'][1], 2)}] | {f(b['rho'], 2)} [{f(b['ci'][0], 2)}, {f(b['ci'][1], 2)}] | {a['layers']} |")
    w("")

    w("## 7. CNN analogue: DeepGaze IIE backbone 0 (ResNet-50 ShapeNetC)\n")
    for mode, C in res["cnn"].items():
        w(f"### input {mode}\n")
        w(f"Maximum absolute activation difference (test vs control180) at units whose receptive field is free of inducer "
          f"pixels: {C['max_abs_activation_difference_at_rf_clean_units']:g}; maximum |index numerator| over those unit sets: "
          f"{C['max_abs_index_numerator_at_rf_clean_sets']:g}.\n")
        w("| layer | rf / jump (px) | Kanizsa: configs with plan units / defined | plan E | Ehrenstein: configs with plan units / defined | "
          "secondary cell-variant Kanizsa E | null c90 | null bg | FS (descriptive) | cell ratio20 [CI] |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for Lk in C["layers"]:
            g = C["geometry"][Lk]; k = C["per_layer"][Lk]["kanizsa"]; e = C["per_layer"][Lk]["ehrenstein"]
            w(f"| {Lk} | {g['rf']} / {g['jump']} | {k['n_units_rf_int']['configs_with_units']} / {k['n_units_rf_int']['configs_defined']} | "
              f"{ci(k['plan_interior'])} | {e['n_units_rf_int']['configs_with_units']} / {e['n_units_rf_int']['configs_defined']} | "
              f"{ci(k['cell_interior'])} | {ci(k['cell_c90_null'])} | {ci(k['cell_bg_null'])} | {yn(k['figure_specific_cell_descriptive'])} | "
              f"{(str(round(k['cell_ratio20']['ratio'], 2)) + ' [' + str(round(k['cell_ratio20']['ci'][0], 2)) + ', ' + str(round(k['cell_ratio20']['ci'][1], 2)) + ']') if k.get('cell_ratio20') else 'n/a'} |")
        w("")

    w("## 8. Title rule\n")
    tr = res["title_rule"]
    for n in nets:
        t = tr[n]
        w(f"- {NAME[n]} L{t['layer']}: figure-specific {yn(t['figure_specific'])}, relatability-tuned (plan) {yn(t['relatability_tuned_plan'])}; "
          f"layers at depth 1/3 to 2/3: " + ", ".join(f"L{k} FS {yn(v['fs'])}{(' ' + v['rel']) if v['rel'] else ''}" for k, v in t["middle_band_layers"].items()))
    w(f"\nEncoding claim allowed in the title (figure-specific and relatability-tuned at the middle layer of all three networks): "
      f"{yn(tr['all_three'])}\n")
    w("## Figures\n")
    for n in nets:
        w(f"- fig_{n}.png: {NAME[n]} per-layer index, nulls, misalignment, Ehrenstein, cosine")
    w("- fig_sizes.png: 336 vs 448 px separately; fig_ratio.png: misaligned/aligned ratios")
    for mode in res["cnn"]:
        w(f"- fig_cnn_{mode}.png: CNN analogue")
    open(os.path.join(ENC, "report.md"), "w").write("\n".join(L) + "\n")
