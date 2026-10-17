#!/usr/bin/env python3
"""
paper_numbers/make_figures.py: the four figures of the manuscript.

Sources (read only):
  <out>/numbers.tex                 values and CIs for Figures 2 and 3 (written by make_numbers.py, so every plotted
                                    number equals the number printed in the text and tables)
  v3/scores/results.json            figure-mass loss under uniform and alternating rotation (Figure 4)
  stimuli/stim_v3, stimuli/stim_alt stimulus images and AOIs (Figure 1)

Writes <out>/figs/fig1_stimuli.pdf, fig2_survey_nulls.pdf, fig3_rotation.pdf, fig4_readout_swap.pdf (+ .png previews);
<out> = paper_numbers/output, or LAN_PAPER_OUT.
Run: python paper_numbers/make_figures.py   (after make_numbers.py)
"""
import io
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as LP  # noqa: E402

ROOT = LP.RESULTS
NUM = Path(os.environ.get("LAN_PAPER_OUT") or Path(__file__).resolve().parent / "output")
OUT = NUM / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Categorical palette (light mode), fixed order; ink tokens for text and neutral marks.
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NULLGREY = "#b9b8b0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": BASE,
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
MM = 1 / 25.4
# Text width of the sn-jnl single-column layout: \textwidth = 372 pt = 131.2 mm (main.log). Figures are drawn at this
# width and included at width=\linewidth, i.e. at scale 1.0, so the lettering prints at the sizes set above (7.5 to 8 pt).
FULL = 372 / 72.27


# ------------------------------------------------------------------ numbers.tex access
def load_macros():
    txt = (NUM / "numbers.tex").read_text()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", txt, re.M):
        out[m.group(1)] = m.group(2)
    return out


MAC = load_macros()


def _num(s):
    s = s.replace("\\ensuremath{", "").replace("}", "").replace("$", "").strip()
    return float(s)


def val(name):
    v = MAC[name]
    try:
        return _num(v)
    except ValueError:
        return None


def ci(name):
    v = MAC.get(name + "CI")
    if v is None:
        return None
    try:
        a, b = v.strip("[]").split(",")
        return _num(a), _num(b)
    except ValueError:
        return None


def fit_width(fig, by="width", target=None, steps=18):
    """Resize the figure so that its tight bounding box (labels and legends included) is the text width, so that LaTeX
    includes it at scale 1.0 and the lettering prints at its nominal size. by='width' changes the figure width (plots);
    by='height' changes the height. Bisection: the tight width grows monotonically with the figure size."""
    target = (target or FULL) - 2 * 0.02            # savefig adds pad_inches on both sides
    W0, H0 = fig.get_size_inches()

    def tight(x):
        """Width in inches of the PDF that savefig(bbox_inches='tight') writes, minus the padding."""
        fig.set_size_inches((x, H0) if by == "width" else (W0, x))
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.02)
        m = re.search(rb"/MediaBox \[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", buf.getvalue())
        return (float(m.group(3)) - float(m.group(1))) / 72.0 - 2 * 0.02

    x0 = W0 if by == "width" else H0
    lo, hi = 0.3 * x0, 2.0 * x0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if tight(mid) > target:
            hi = mid
        else:
            lo = mid
    w = tight(lo)
    if abs(w - target) > 0.02:
        print(f"  warning: tight width {w + 0.04:.3f} in, target {target + 0.04:.3f} in")


def save(fig, stem, by="width", fit=True):
    if fit:
        fit_width(fig, by)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1: stimuli
def crop(img, cx, cy, half):
    return img.crop((cx - half, cy - half, cx + half, cy + half))


def fig1():
    rows = [("square_sr50_dark_p0", 175), ("triangle_sr50_light_p0", 190)]
    cols = [("illusory", "Illusory\n", "v2"), ("real", "Real\noutline", "v2"), ("real_filled", "Real\nfilled", "v2"),
            ("control180", "180°\ncontrol", "v2"), ("control90", "90°\ncontrol", "v2"),
            ("mis20", "Uniform\n20°", "v2"), ("alt20a", "Altern.\n20° (a)", "v3"),
            ("alt20b", "Altern.\n20° (b)", "v3"), ("aoi", "Regions\n(illusory)", "v2")]
    # explicit layout in inches: the image grid spans the full text width (square panels), titles above, legend below
    W = FULL - 2 * 0.02
    gap, top, bottom = 0.035, 0.34, 0.26
    axw = (W - (len(cols) - 1) * gap) / len(cols)
    H = top + len(rows) * axw + (len(rows) - 1) * gap + bottom
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((len(rows), len(cols)), dtype=object)
    for i in range(len(rows)):
        for j in range(len(cols)):
            x = j * (axw + gap); y = H - top - (i + 1) * axw - i * gap
            axes[i, j] = fig.add_axes([x / W, y / H, axw / W, axw / H])
    for i, (cfg, half) in enumerate(rows):
        base = Image.open(LP.STIM_V3 / "img" / f"{cfg}_illusory.png").convert("L")
        aoi = np.load(LP.STIM_V3 / "img" / f"{cfg}_illusory_aoi.npz")
        ins, ind = aoi["inside"].astype(bool), aoi["ind"].astype(bool)
        ys, xs = np.nonzero(ins | ind)
        cx, cy = int(round((xs.min() + xs.max()) / 2)), int(round((ys.min() + ys.max()) / 2))
        for j, (cond, title, src) in enumerate(cols):
            ax = axes[i, j]
            if cond == "aoi":
                g = np.asarray(base, dtype=float) / 255.0
                bg = np.median(g)
                core = ins & (distance_transform_edt(~ind) > 30)
                rgb = np.stack([0.85 + 0.15 * (g if bg > 0.5 else 1 - g)] * 3, -1)
                rgb = np.clip(rgb, 0, 1)
                def tint(mask, hexc, a):
                    c = np.array([int(hexc[k:k + 2], 16) / 255 for k in (1, 3, 5)])
                    rgb[mask] = (1 - a) * rgb[mask] + a * c
                tint(ind, S2, 0.55)
                tint(ins & ~core, S1, 0.30)
                tint(core, S1, 0.75)
                im = Image.fromarray((rgb * 255).astype(np.uint8))
            else:
                d = (LP.STIM_V3 if src == "v2" else LP.STIM_ALT) / "img"
                im = Image.open(d / f"{cfg}_{cond}.png").convert("RGB")
            ax.imshow(crop(im, cx, cy, half), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(BASE); sp.set_linewidth(0.6)
            if i == 0:
                ax.set_title(title, fontsize=7, pad=3, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S1, alpha=0.30), plt.Rectangle((0, 0), 1, 1, color=S1, alpha=0.75),
               plt.Rectangle((0, 0), 1, 1, color=S2, alpha=0.55)]
    fig.legend(handles, ["Inside region", "Core (> 30 px from the inducers)", "Inducer region"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    save(fig, "fig1_stimuli", fit=False)


# ------------------------------------------------------------------ Figure 2: survey and local nulls
def fig2():
    models = [("IIE", "DeepGaze IIE"), ("MSDB", "DeepGaze MSDB"), ("MSINet", "MSI-Net"),
              ("XSAspp", "xSoyelNet-ASPP"), ("XSMl", "xSoyelNet-ML"),
              ("SpecRes", "Spectral residual")]
    sig = [("Eight", 8), ("Sixteen", 16), ("TwentyFive", 25), ("Forty", 40), ("Sixty", 60)]
    types = [("Corner", "Corner", S1, "o"), ("Ink", "Ink", S2, "s"), ("Edge", "Edge energy", S3, "D")]
    rows = []  # (label, macro_inside, macro_core, colour, marker)
    for k, lab in models:
        rows.append((lab, f"Svy{k}Kz", f"Svy{k}KzCore", INK, "o"))
    rows.append(None)
    for t, lab, c, mk in types:
        for w, s in sig:
            rows.append((f"{lab}, σ = {s} px", f"Null{t}{w}Kz", f"Null{t}{w}KzCore", c, mk))
        rows.append(None)
    rows = rows[:-1]
    y, labels, ypos = 0.0, [], []
    pos = []
    for r in rows:
        if r is None:
            y += 0.7
            continue
        pos.append((y, r)); labels.append(r[0]); ypos.append(y); y += 1
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.7), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, which, title in [(axes[0], 1, "Inside region"), (axes[1], 2, "Core (> 30 px from inducers)")]:
        ax.axvline(0, color=BASE, lw=0.8, zorder=0)
        for yy, r in pos:
            name = r[which]
            if name not in MAC or val(name) is None:
                ax.text(0.003, yy, "not computed", va="center", ha="left", fontsize=7, color=MUTED)
                continue
            v, c = val(name), ci(name)
            if c is not None:
                ax.plot(c, [yy, yy], color=r[3], lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot([v], [yy], marker=r[4], ms=4.2, color=r[3], mec="white", mew=0.8, zorder=3, ls="none")
        # band: the two DeepGaze models' output blur scale corresponds to sigma 16 to 25 px (marked in the y labels)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Illusory minus 180° control" if which == 1 else "Illusory minus 180° control ")
    axes[1].set_xlabel("")
    fig.supxlabel("Effect: r(illusory) minus r(180° control)", fontsize=8, y=0.035)
    axes[0].set_xlabel("")
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(labels)
    for t in axes[0].get_yticklabels():
        if "16 px" in t.get_text() or "25 px" in t.get_text():
            t.set_fontweight("bold")
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.215, 0.27); axes[1].set_xlim(-0.215, 0.27)
    handles = [Line2D([], [], color=INK, marker="o", ls="none", ms=4.2, mec="white", label="Saliency model"),
               Line2D([], [], color=S1, marker="o", ls="none", ms=4.2, mec="white", label="Corner null"),
               Line2D([], [], color=S2, marker="s", ls="none", ms=4.2, mec="white", label="Ink null"),
               Line2D([], [], color=S3, marker="D", ls="none", ms=4.2, mec="white", label="Edge-energy null")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.06))
    save(fig, "fig2_survey_nulls")


# ------------------------------------------------------------------ Figure 3: readout swap
def fig3():
    fsets = [("DinoTwelve", "DINOv2-B\nblock 12"), ("DinoSix", "DINOv2-B\nblock 6"),
             ("SiglipTwentyFour", "SigLIP-L\nblock 24"), ("IIEFeat", "IIE CNN\n(ShapeNet-C)")]
    ros = [("RoneFixed", "R1n pointwise, fixed norm.", S1, "o"), ("Rone", "R1 pointwise, image-wide norm.", S2, "s"),
           ("Rtwo", "R2 ASPP, no global pooling", S3, "D"), ("Rthree", "R3 dilation-1 branch", S4, "^"),
           ("Rfour", "R4 full ASPP", S5, "v")]
    off = np.linspace(-0.3, 0.3, len(ros))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 4.6), sharex=True, gridspec_kw=dict(hspace=0.22))
    panels = [("Kz", "Illusory minus 180° control (inside region)"), ("RealFilled", "Real filled minus 180° control (inside region)")]
    for ax, (q, title) in zip(axes, panels):
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        for i, (fs, _) in enumerate(fsets):
            for k, (ro, _, c, mk) in enumerate(ros):
                base = f"RS{fs}{ro}"
                x = i + off[k]
                v = val(base + q) if (base + q) in MAC else None
                if v is None:
                    ax.text(x, 0.0, "pending", rotation=90, ha="center", va="bottom", fontsize=7, color=MUTED)
                    continue
                cc = ci(base + q)
                valid = MAC.get(base + "Valid", "valid") == "valid"
                prov = MAC.get(base + "Provisional", "no") == "yes"
                if cc is not None:
                    ax.plot([x, x], cc, color=c, lw=1.2, ls=(0, (1.2, 1.2)) if prov else "-",
                            solid_capstyle="round", zorder=2)
                ax.plot([x], [v], marker=mk, ms=4.6, color=c if valid else "white", mec=c, mew=1.1, ls="none",
                        zorder=3)
        ax.set_xticks(range(len(fsets))); ax.set_xticklabels([f[1] for f in fsets])
        ax.set_xlim(-0.55, len(fsets) - 0.45)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=7.5)
    axes[0].set_ylabel("Effect"); axes[1].set_ylabel("Effect")
    handles = [Line2D([], [], color=c, marker=mk, ls="none", ms=4.6, mec=c, label=lab) for _, lab, c, mk in ros]
    handles += [Line2D([], [], color=INK2, marker="o", ls="none", ms=4.6, mfc="white", mec=INK2,
                       label="below validity threshold")]
    if any(v == "yes" for k, v in MAC.items() if k.startswith("RS") and k.endswith("Provisional")):
        handles += [Line2D([], [], color=INK2, lw=1.2, ls=(0, (1.2, 1.2)), label="fewer than 3 seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig4_readout_swap")


# ------------------------------------------------------------------ Figure 4: figure-mass loss under rotation
def fig4():
    R = LP.load_json(ROOT / "v3/scores/results.json")["models"]
    th = [0, 10, 20, 30]
    subsets = [("all", "All (60)"), ("square", "Squares (30)"), ("triangle", "Triangles (30)")]
    nulls = [k for k in R if k.startswith("null ")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5), sharey=True, gridspec_kw=dict(wspace=0.10))

    def series(rec, sub, key):
        d = rec["results"]["r_inside"][sub][key]
        m = [0.0] + [d[str(t)]["mean"] for t in th[1:]]
        lo = [0.0] + [d[str(t)]["ci"][0] for t in th[1:]]
        hi = [0.0] + [d[str(t)]["ci"][1] for t in th[1:]]
        return np.array(m), np.array(lo), np.array(hi)

    for ax, (sub, title) in zip(axes, subsets):
        ax.axhline(0, color=BASE, lw=0.8, zorder=0)
        for n in nulls:
            for key, ls in (("drop_uni", "-"), ("drop_alt", (0, (3, 2)))):
                m, _, _ = series(R[n], sub, key)
                ax.plot(th, m, color=NULLGREY, lw=0.7, ls=ls, zorder=1)
        for name, c, mk in (("DeepGaze IIE", S1, "o"), ("DeepGaze MSDB", S2, "s")):
            for key, ls, fill in (("drop_uni", "-", True), ("drop_alt", (0, (3, 2)), False)):
                m, lo, hi = series(R[name], sub, key)
                ax.fill_between(th, lo, hi, color=c, alpha=0.10, lw=0, zorder=2)
                ax.plot(th, m, color=c, lw=1.6, ls=ls, zorder=3)
                ax.plot(th, m, marker=mk, ms=4.2, ls="none", color=c if fill else "white", mec=c, mew=1.0, zorder=4)
        ax.set_xticks(th); ax.set_xticklabels([f"{t}°" for t in th])
        ax.set_xlabel("Inducer rotation")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Figure-mass loss\nr(aligned) minus r(rotated)")
    handles = [Line2D([], [], color=S1, marker="o", ms=4.2, lw=1.6, label="DeepGaze IIE"),
               Line2D([], [], color=S2, marker="s", ms=4.2, lw=1.6, label="DeepGaze MSDB"),
               Line2D([], [], color=NULLGREY, lw=0.9, label="15 local nulls"),
               Line2D([], [], color=INK2, lw=1.2, label="uniform (filled markers)"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (3, 2)), label="alternating (open markers)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig3_rotation")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written to", OUT)
