"""
Paper figures (figures/*.png and .pdf).
  fig1_kanizsa_maps          Kanizsa square (SR 0.5) in four versions with model maps (runs the models; needs the
                             xSoyelNet-ASPP weights, see models.py) and the DINOv2 block-6 / block-12 layer readouts
  fig2_kanizsa_support_ratio illusory - control against support ratio for every model
  fig3_ehrenstein            Ehrenstein stimuli and every model's effect against control B (control_end)
  fig4_layers                per-layer illusion effect and MIT1003 CC for the three backbones
  fig5_layer_size_heatmaps   effect for every layer and input size
Usage: python make_figures.py [1 2 3 4 5]   (2-5 only read results/)
"""
import json, os, sys
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import HERE, STIM, RESULTS, rel_inside
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
D = STIM["kanizsa"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
BBS = (("dinov2", "DINOv2-B"), ("clip", "CLIP ViT-L/14"), ("siglip", "SigLIP-L"))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

def fig1():
    import torch, models as MD, layer_sweep as LS
    LS.setup("dinov2"); bb = LS.load_backbone(); R = LS.load_readouts(bb)
    def layer_map(L, rgb, s=560):
        with torch.no_grad():
            R[L].use_cb = False; f = LS.feats(bb, LS.prep(rgb, s)[None])
            return cv2.resize(torch.sigmoid(R[L](f[L], (s, s)))[0, 0].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
    models = [("DeepGaze IIE", MD.deepgaze_iie()), ("DeepGaze MSDB", MD.deepgaze_msdb()),
              ("xSoyelNet-ASPP\n(DINOv2 last layer)", MD.xsoyelnet_aspp()),
              ("DINOv2 block 6\n(560 px)", lambda x: layer_map(6, x)), ("DINOv2 block 12\n(560 px)", lambda x: layer_map(12, x))]
    conds = ("illusory", "real", "control", "control90")
    clab = {"illusory": "Illusory", "real": "Real", "control": "Control (180°)", "control90": "Control (90°)"}
    y0, y1, x0, x1 = 214, 554, 300, 724                                   # crop around the centred square
    fig, ax = plt.subplots(len(models) + 1, 4, figsize=(6.6, 1.32 * (len(models) + 1)))
    for j, c in enumerate(conds):
        name = f"square_sr50_dark_p0_{c}"
        rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", name + ".png")), cv2.COLOR_BGR2RGB)
        a = np.load(os.path.join(D, "img", name + "_aoi.npz"))
        ax[0, j].imshow(rgb[y0:y1, x0:x1]); ax[0, j].set_title(clab[c])
        for i, (mn, fn) in enumerate(models, 1):
            sal = fn(rgb).astype(np.float32); ri = rel_inside(sal, a["inside"], a["ind"])
            s = sal[y0:y1, x0:x1]; s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            ax[i, j].imshow(rgb[y0:y1, x0:x1].mean(2), cmap="gray", vmin=0, vmax=255)
            ax[i, j].imshow(s, cmap="inferno", alpha=0.75)
            ax[i, j].contour(a["inside"][y0:y1, x0:x1], levels=[0.5], colors="c", linewidths=0.5, linestyles="--")
            ax[i, j].text(6, 34, f"{ri:.2f}", color="w", fontsize=6.5, weight="bold", bbox=dict(fc="k", alpha=0.5, lw=0, pad=1))
            if j == 0: ax[i, j].set_ylabel(mn, fontsize=7)
    for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
    for s_ in ax.ravel():
        for sp in s_.spines.values(): sp.set_visible(False)
    save(fig, "fig1_kanizsa_maps")

SURVEY = [("DeepGaze IIE", "#1b9e77"), ("DeepGaze MSDB", "#66a61e"), ("xSoyelNet-ML", "#7570b3"), ("MSI-Net", "#e6ab02"),
          ("xSoyelNet-ASPP", "#d95f02"), ("Spectral residual", "#a6761d"), ("Edge-energy null", "#666666")]

def fig2():
    items = json.load(open(os.path.join(D, "items.json"))); srs = sorted({it["sr"] for it in items})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for k, ctrl in enumerate(("control90", "control")):
        for mn, col in SURVEY:
            r = json.load(open(os.path.join(RESULTS, "kanizsa", f"results_{mn.replace(' ', '_')}.json")))
            mu, se = [], []
            for sr in srs:
                d = np.array([r[it["name"]] - r[it["name"].replace("_illusory", "_" + ctrl)] for it in items
                              if it["cond"] == "illusory" and it["sr"] == sr])
                mu.append(d.mean()); se.append(d.std(ddof=1) / np.sqrt(len(d)))
            mu, se = np.array(mu), np.array(se)
            ls = ":" if mn in ("Spectral residual", "Edge-energy null") else "-"
            ax[k].plot(srs, mu, ls, color=col, marker="o", ms=2.5, lw=1.2, label=mn)
            ax[k].fill_between(srs, mu - se, mu + se, color=col, alpha=0.15, lw=0)
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set_xlabel("Support ratio"); ax[k].set_title(f"Illusory − {'90°' if ctrl == 'control90' else '180°'} control")
    ax[0].set_ylabel("Δ relative saliency inside figure")
    ax[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "fig2_kanizsa_support_ratio")

def load_sweep(bb):
    return json.load(open(os.path.join(RESULTS, "layers", bb, "sweep_results.json")))

def fig3():
    sizes = (336, 448, 560, 672)
    fig, ax = plt.subplots(2, 3, figsize=(6.6, 4.8), gridspec_kw=dict(hspace=0.42))
    for row, (illu, key, lab_c) in enumerate((("Kanizsa", "d180", "p180"), ("Ehrenstein", "d_end", "p_end"))):
        for a, (bb, lab) in zip(ax[row], BBS):
            r = load_sweep(bb); Ls = sorted(int(L) for L in r["mit1003"])
            src = r["kanizsa"] if row == 0 else json.load(open(os.path.join(RESULTS, "layers", bb, "ehr_results.json")))
            Z = np.array([[src[f"L{L}_s{s}"][key] for s in sizes] for L in Ls])
            P = np.array([[src[f"L{L}_s{s}"][lab_c] for s in sizes] for L in Ls])
            im = a.imshow(Z, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto", origin="lower")
            for i in range(len(Ls)):
                for j in range(len(sizes)):
                    a.text(j, i, f"{Z[i, j]:+.2f}\n{P[i, j] * 100:.0f}%", ha="center", va="center", fontsize=5.5,
                           color="w" if abs(Z[i, j]) > 0.09 else "k")
            a.set_xticks(range(len(sizes)), sizes); a.set_yticks(range(len(Ls)), Ls)
            if row == 1: a.set_xlabel("Input size (px)")
            a.set_title(f"{illu}: {lab}")
        ax[row, 0].set_ylabel("Layer")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("Illusory − primary control")
    save(fig, "fig5_layer_size_heatmaps")

def fig4():
    sizes = (336, 448, 560, 672)
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.2), sharey=True)
    for k, (a, (bb, lab)) in enumerate(zip(ax, BBS)):
        r = load_sweep(bb); Ls = sorted(int(L) for L in r["mit1003"]); n = 12 if bb == "dinov2" else 24
        dep = np.array(Ls) / n
        e = json.load(open(os.path.join(RESULTS, "layers", bb, "ehr_results.json")))
        e180 = [np.mean([r["kanizsa"][f"L{L}_s{s}"]["d180"] for s in sizes]) for L in Ls]
        e90 = [np.mean([e[f"L{L}_s{s}"]["d_end"] for s in sizes]) for L in Ls]
        cc = [r["mit1003"][str(L)]["CC"] for L in Ls]
        w = 0.035
        a.bar(dep - w / 2, e180, w, color="#4575b4", label="Kanizsa")
        a.bar(dep + w / 2, e90, w, color="#fc8d59", label="Ehrenstein")
        a.axhline(0, color="k", lw=0.6); a.set_title(lab); a.set_xlabel("Relative depth (layer / #layers)")
        a.set_xticks(dep, [f"{L}" for L in Ls]); a.set_xlim(0.08, 1.08)
        t = a.twinx(); t.plot(dep, cc, "k-o", ms=2.5, lw=1, label="MIT1003 CC")
        t.set_ylim(0.6, 0.78); t.spines["top"].set_visible(False)
        best = int(np.argmax(cc)); t.plot(dep[best], cc[best], "o", mfc="none", mec="r", ms=6)
        if k < 2: t.set_yticklabels([])
        else: t.set_ylabel("MIT1003 held-out CC")
    ax[0].set_ylabel("Illusion effect (mean over sizes)"); ax[0].set_xlabel("Layer")
    for a in ax: a.set_xlabel("Layer")
    h1, l1 = ax[0].get_legend_handles_labels(); ax[0].legend(h1 + [plt.Line2D([], [], color="k", marker="o", ms=2.5)],
                                                             l1 + ["MIT1003 CC (right axis)"], frameon=False, loc="lower left")
    save(fig, "fig4_layers")

def fig5():
    E = STIM["ehrenstein"]; items = json.load(open(os.path.join(E, "items.json")))
    fig = plt.figure(figsize=(6.6, 2.4)); gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.25, 4.2], hspace=0.25, wspace=0.05)
    lab = {"illusory": "Illusory", "real": "Real", "control_mid": "Control A\n(liberal)", "control_end": "Control B\n(conservative)"}
    for j, c in enumerate(("illusory", "real", "control_mid", "control_end")):
        for i, n in enumerate((8, 16)):
            a = fig.add_subplot(gs[i, j]); img = cv2.imread(os.path.join(E, "img", f"ehr_R100_n{n}_dark_p0_{c}.png"))[184:584, 312:712]
            a.imshow(img[..., ::-1]); a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
            if i == 0: a.set_title(lab[c], fontsize=6.5)
            if j == 0: a.set_ylabel(f"{n} lines", fontsize=6.5)
    a = fig.add_subplot(gs[:, 5]); names, mu, se, cols = [], [], [], []
    for mn, col in SURVEY:
        r = json.load(open(os.path.join(RESULTS, "ehrenstein", f"results_{mn.replace(' ', '_')}.json")))
        d = np.array([r[it["name"]] - r[it["name"].replace("_illusory", "_control_end")] for it in items if it["cond"] == "illusory"])
        names.append(mn); mu.append(d.mean()); se.append(d.std(ddof=1) / np.sqrt(len(d))); cols.append(col)
    y = np.arange(len(names))[::-1]
    a.barh(y, mu, xerr=se, color=cols, height=0.65, error_kw=dict(lw=0.6, capsize=1.5))
    a.set_yticks(y, names); a.axvline(0, color="k", lw=0.6); a.yaxis.tick_right(); a.spines["left"].set_visible(False)
    a.spines["right"].set_visible(True); a.set_xlabel("Illusory − conservative control"); a.set_title("Ehrenstein (all 120 configurations)")
    save(fig, "fig3_ehrenstein")

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig5, "4": fig4, "5": fig3}[w]()
