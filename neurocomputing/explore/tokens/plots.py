"""neuro/reanalysis/explore/tokens copy of v2/encoding/plots.py (RUN_PLAN.md S5): output folder redirected to
neuro/reanalysis/A3_tokens/published/; common_enc read only from v2/encoding. Original docstring follows.
Section C figures (called by analyze.py). Static PNGs, light surface; colours from the dataviz reference palette."""
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common_enc import ENC as _ENC, PRIMARY
ENC = OUTDIR                                         # copy: redirected (was ENC_FIGDIR or v2/encoding)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"illusory": "#0d366b", "mis10": "#256abf", "mis20": "#3987e5", "mis30": "#86b6ef"}   # ordinal: misalignment
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlecolor": INK})


def _style(ax, net, nL):
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for L in PRIMARY.get(net, ()):
        ax.axvline(L, color=GRID, lw=6, zorder=0, alpha=0.9)
    ax.set_xlim(0.5, nL - 0.5)
    ax.set_xlabel("hidden state (layer)")


def _series(ax, layers, key, color, label, fs_key=None, sub=None, lw=2, ms=8, clip=None):
    """layers: list of per-layer result blocks; key: path inside a block."""
    xs, m, lo, hi, fsl = [], [], [], [], []
    for b in layers:
        r = b
        for k in key:
            r = r[k] if r is not None else None
        if r is None or r["mean"] is None:
            continue
        xs.append(b["layer"]); m.append(r["mean"]); lo.append(r["ci"][0]); hi.append(r["ci"][1])
        fsl.append(bool(b.get(fs_key)) if fs_key else False)
    if not xs:
        return []
    xs, m, lo, hi = map(np.array, (xs, m, lo, hi))
    if clip is not None:
        lo_c, hi_c = np.clip(lo, *clip), np.clip(hi, *clip)
    else:
        lo_c, hi_c = lo, hi
    ax.fill_between(xs, lo_c, hi_c, color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, m, color=color, lw=lw, label=label, zorder=3)
    if fs_key:
        f = np.array(fsl)
        ax.plot(xs[f], m[f], "o", color=color, ms=ms, zorder=4, mec=SURF, mew=1.5)
        ax.plot(xs[~f], m[~f], "o", color=SURF, ms=ms - 1, zorder=4, mec=color, mew=1.5)
    return list(lo) + list(hi) + list(m)


def _ylim(ax, vals, pad=0.08):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if v.size == 0:
        return None
    a, b = min(v.min(), 0), max(v.max(), 0)
    d = (b - a) or 1
    ax.set_ylim(a - pad * d, b + pad * d)
    return a - pad * d, b + pad * d


def fig_net(res, net):
    R = res["networks"][net]; nL = R["n_layers"]
    K = R["kanizsa"]["avg"][1:]; E = R["ehrenstein"]["avg"][1:]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
    # (a) Kanizsa E with nulls; y-range from interior and control90 CIs and the background-null means
    ax = axs[0, 0]
    rng = []
    for b in K:
        rng += [b["interior"]["ci"][0], b["interior"]["ci"][1], b["c90_null"]["ci"][0], b["c90_null"]["ci"][1], b["bg_null"]["mean"]]
    lim = _ylim(ax, rng)
    _series(ax, K, ("interior",), BLUE, "interior, illusory", "figure_specific", clip=lim)
    _series(ax, K, ("c90_null",), ORANGE, "null: control90 as test", clip=lim)
    _series(ax, K, ("bg_null",), AQUA, "null: background tokens", clip=lim)
    _ylim(ax, rng)
    _style(ax, net, nL); ax.set_ylabel("encoding index E"); ax.set_title("Kanizsa: interior vs nulls")
    ax.legend(loc="lower right", fontsize=8)
    # (b) misalignment
    ax = axs[0, 1]; rng = []
    rng += _series(ax, K, ("interior",), RAMP["illusory"], "aligned (illusory)")
    for t in ("mis10", "mis20", "mis30"):
        rng += _series(ax, K, ("mis", t), RAMP[t], f"misaligned {t[3:]} deg")
    _ylim(ax, rng); _style(ax, net, nL); ax.set_ylabel("interior encoding index E")
    ax.set_title("Kanizsa: aligned vs misaligned inducers"); ax.legend(loc="lower right", fontsize=8)
    # (c) Ehrenstein
    ax = axs[1, 0]; rng = []
    for b in E:
        rng += [b["interior"]["ci"][0], b["interior"]["ci"][1], b["bg_null"]["mean"]]
    lim = _ylim(ax, rng)
    _series(ax, E, ("interior",), BLUE, "interior, illusory", "figure_specific", clip=lim)
    _series(ax, E, ("bg_null",), AQUA, "null: background tokens", clip=lim)
    _ylim(ax, rng)
    _style(ax, net, nL); ax.set_ylabel("encoding index E"); ax.set_title("Ehrenstein (vs control_end): interior vs background")
    ax.legend(loc="lower right", fontsize=8)
    # (d) cosine
    ax = axs[1, 1]; rng = []
    rng += _series(ax, K, ("cos_interior",), BLUE, "interior, illusory", "figure_specific_cos")
    rng += _series(ax, K, ("cos_c90_null",), ORANGE, "null: control90 as test")
    rng += _series(ax, K, ("cos_bg_null",), AQUA, "null: background tokens")
    _ylim(ax, rng); _style(ax, net, nL); ax.set_ylabel("cosine"); ax.set_title("Kanizsa: cosine")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{NAME[net]}: readout-free encoding index per layer (mean over configurations, 336 and 448 px averaged, "
                 f"95% cluster-bootstrap CI)\nfilled dot = figure-specific (interior CI above both nulls' CIs); "
                 f"grey bands = primary layers", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(ENC, f"fig_{net}.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_sizes(res, nets):
    fig, axs = plt.subplots(len(nets), 2, figsize=(11, 3.2 * len(nets)), squeeze=False)
    for i, net in enumerate(nets):
        R = res["networks"][net]; nL = R["n_layers"]
        for j, s in enumerate(("336", "448")):
            ax = axs[i, j]; K = R["kanizsa"][s][1:]; rng = []
            for b in K:
                rng += [b["interior"]["ci"][0], b["interior"]["ci"][1], b["c90_null"]["ci"][0], b["c90_null"]["ci"][1], b["bg_null"]["mean"]]
            lim = _ylim(ax, rng)
            _series(ax, K, ("interior",), BLUE, "interior, illusory", "figure_specific", clip=lim)
            _series(ax, K, ("c90_null",), ORANGE, "null: control90", clip=lim)
            _series(ax, K, ("bg_null",), AQUA, "null: background", clip=lim)
            _ylim(ax, rng); _style(ax, net, nL); ax.set_title(f"{NAME[net]}, Kanizsa, {s} px")
            ax.set_ylabel("E")
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Input-size sensitivity: Kanizsa interior index and nulls at 336 and 448 px separately "
                 "(filled dot = figure-specific at that size)", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(ENC, "fig_sizes.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_ratio(res, nets):
    fig, axs = plt.subplots(1, len(nets), figsize=(4 * len(nets), 3.4), squeeze=False)
    any_ = False
    for i, net in enumerate(nets):
        ax = axs[0, i]; R = res["networks"][net]; nL = R["n_layers"]
        for t, c in (("ratio10", RAMP["mis10"]), ("ratio20", RAMP["mis20"]), ("ratio30", RAMP["mis30"])):
            xs, m, lo, hi = [], [], [], []
            for b in R["kanizsa"]["avg"]:
                rl = b.get("relatability")
                if rl:
                    xs.append(b["layer"] + {"ratio10": -0.18, "ratio20": 0, "ratio30": 0.18}[t])
                    m.append(rl[t]["ratio"]); lo.append(rl[t]["ci"][0]); hi.append(rl[t]["ci"][1])
            if xs:
                any_ = True
                m, lo, hi = map(np.array, (m, lo, hi))
                ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o", color=c, ms=5, lw=1.5, capsize=0,
                            label=f"{t[5:]} deg")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--"); ax.axhline(1, color=GRID, lw=0.8)
        _style(ax, net, nL); ax.set_title(NAME[net]); ax.set_ylabel("E(misaligned) / E(aligned)")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("Kanizsa interior index, misaligned / aligned, at figure-specific layers (95% CI; dashed: 0.5)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(ENC, "fig_ratio.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p if any_ else p


def fig_cnn(res, mode="iie"):
    C = res["cnn"].get(mode)
    if not C:
        return None
    layers = C["layers"]; x = np.arange(len(layers))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    for k, (ill, col) in enumerate((("kanizsa", BLUE), ("ehrenstein", ORANGE))):
        n = [C["per_layer"][L][ill]["n_units_rf_int"]["configs_defined"] for L in layers]
        ax.bar(x + (k - 0.5) * 0.38, n, 0.36, color=col, label=ill, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=70, fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("configurations with a defined index")
    ax.set_title("Plan units (receptive field free of inducers):\nconfigurations where the index is defined (E = 0 exactly in all)")
    ax.legend(fontsize=8)
    ax = axs[1]
    # y-range from the interior and control90 CIs; the background null is clipped to it (unstable where the outline
    # barely enters a far unit's receptive field; values beyond the range are marked with triangles)
    rng = []
    for key in ("cell_interior", "cell_c90_null"):
        for L in layers:
            rng += [v for v in C["per_layer"][L]["kanizsa"][key]["ci"] if v is not None]
    lim = (min(min(rng), 0) - 0.08 * (max(rng) - min(min(rng), 0)), max(rng) + 0.25 * (max(rng) - min(min(rng), 0)))
    for key, col, lab in (("cell_interior", BLUE, "interior, illusory"), ("cell_c90_null", ORANGE, "null: control90 as test"),
                          ("cell_bg_null", AQUA, "null: background units (clipped)")):
        m = np.array([C["per_layer"][L]["kanizsa"][key]["mean"] if C["per_layer"][L]["kanizsa"][key]["mean"] is not None else np.nan for L in layers])
        lo = np.array([C["per_layer"][L]["kanizsa"][key]["ci"][0] if C["per_layer"][L]["kanizsa"][key]["ci"][0] is not None else np.nan for L in layers])
        hi = np.array([C["per_layer"][L]["kanizsa"][key]["ci"][1] if C["per_layer"][L]["kanizsa"][key]["ci"][1] is not None else np.nan for L in layers])
        ax.fill_between(x, np.clip(lo, *lim), np.clip(hi, *lim), color=col, alpha=0.16, lw=0)
        ax.plot(x, np.clip(m, *lim), color=col, lw=2, label=lab, marker="o", ms=4)
        up, dn = m > lim[1], m < lim[0]
        if up.any():
            ax.plot(x[up], np.full(up.sum(), lim[1]), "^", color=col, ms=8, clip_on=False)
        if dn.any():
            ax.plot(x[dn], np.full(dn.sum(), lim[0]), "v", color=col, ms=8, clip_on=False)
    ax.set_ylim(*lim)
    ax.axhline(0, color=INK2, lw=0.8); ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=70, fontsize=7); ax.set_ylabel("E")
    ax.set_title("Secondary (descriptive): units with a pixel-identical stride cell,\nreceptive field unrestricted (Kanizsa)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"DeepGaze IIE backbone 0 (ResNet-50, ShapeNetC), input {mode}", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(ENC, f"fig_cnn_{mode}.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def make_all(res, keep, cnnP):
    nets = list(res["networks"])
    out = [fig_net(res, n) for n in nets] + [fig_sizes(res, nets), fig_ratio(res, nets)]
    for mode in res["cnn"]:
        out.append(fig_cnn(res, mode))
    print("figures:", out, flush=True)
    return out
