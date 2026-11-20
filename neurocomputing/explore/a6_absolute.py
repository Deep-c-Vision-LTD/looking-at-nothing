"""
a6_absolute.py: A6 absolute metrics with the centre bias off, REANALYSIS_RULE.md Section 6 item 7 (second part; RUN_PLAN.md
Step 1, S4). Exploratory. Written 18 November 2026 (about 23:55 UTC), after P1 to P5 and the other exploratory analyses
had been run and read (DEVIATIONS.md D10). CPU only (V3_DEVICE=cpu, the ab_objective stub, no Metal query); heads only on
the cached float16 MIT1003 features; no backbone is run and nothing is downloaded.

Rule text (Section 6, item 7, quoted verbatim):
  "A6 validity margin: valid-set composition and sign agreement among valid cells for margins 0 to 0.06 in steps of
  0.005 (known to be flat at 100%; Section 1.1, item 4); absolute CC, NSS and information gain (bits per fixation,
  baseline = log of the mean training map of `ab/maps.npy`) on held-out MIT1003 with the centre bias off, heads-only
  from the cached features."
RUN_PLAN S4: "A6 absolute metrics with the centre bias off: copy of `v3/gpu/eval_full.py` `run_mit()` with
  `use_cb = False`, heads-only on `v2/readout/cache/<fset>_mit200.npy` (`V3_DEVICE=cpu`, `gpu/recommon.py` import
  path) | 60 checkpoints, MIT1003 maps of the saliency project, `ab/maps.npy` (information-gain baseline)".
Task: "First reproduce one published absolute metric with the centre bias ON to the stated tolerance."

Copied code (v3/gpu/eval_full.py run_mit(), published readout-swap evaluation), per head:
    m = load_head(h); m.use_cb = True; cc, nss = [], []
    F32 = fcache[h["fset"]]                                   # float16 features of the 200 held-out images, 336 px
    for b in range(0, len(items), 16):
        sal = torch.sigmoid(m(F32[b:b + 16].float().to(DEVICE), (S, S)))[:, 0].cpu().numpy()
        for i in range(sal.shape[0]):
            gt = gts[b + i]; hh, ww = gt.shape
            s_ = cv2.resize(sal[i], (ww, hh))
            cc.append(CC(s_, gt)); nss.append(NSS(s_, fxs[b + i]))
  with gts = imread(gt, grey) / 255 and fxs = (imread(fixPts, grey) > 128). Changes here, nothing else:
  (1) features from the CPU float16 cache v2/readout/cache/<fset>_mit200.npy (written by v2/readout/cache_features.py
      from the same 200 images in the same order, mit_heldout()) instead of a backbone pass on the MPS device;
  (2) the loop runs twice per head, use_cb = True (reproduction gate) and use_cb = False (the rule's quantity);
  (3) information gain per image (below) next to CC and NSS; (4) checkpoints read from v3/gpu/full/ckpt (read only).

Implementation choices (fixed before any score was read):
  - Heads: the 60 swap checkpoints (4 feature sets x R1n, R1, R2, R3, R4 x seeds 0, 1, 2). The registers heads have no
    MIT1003 feature cache and are not part of item 7 as written ("heads-only from the cached features").
  - Gate (task, rule Section 3.2 tolerance for CPU re-inference): with the centre bias ON, dinov2_L12_R4_s0 first: the
    mean CC and mean NSS over the 200 images must equal the published values of v3/gpu/full/scores/mit_<tag>.json to
    1e-4, and every per-image CC must equal the published per-image CC (stored rounded to 5 decimals) to 1e-4. If this
    first head fails, the script stops before any centre-bias-off value is computed. Every other head is gated the
    same way; a head that fails has its centre-bias-off values withheld ("not computable").
  - Information gain (bits per fixation; Kummerer, Wallis and Bethge 2015): per image, the model map s_ (the sigmoid map
    resized to the image exactly as for CC and NSS) and the baseline b (the mean of the 12,803 training maps of
    ab/maps.npy, items with fixations as fullcommon.train_items, 336 x 336, mean in float64, resized to the image with
    cv2.resize, bilinear) are each given a floor of 1e-12 x their maximum (so the logarithm is finite) and normalised to
    sum 1 over the image; IG = mean over the fixated pixels (fxs, as for NSS) of log2 p_model - log2 p_baseline; then
    the mean over the 200 images. The baseline is image-independent, so IG measures what the readout adds to the
    average training map.
  - Aggregation (rule Section 2.2 analogue): per image the three seeds are averaged; the cell value is the mean over
    the 200 images, with a WCU interval and WCR p from analyze_re.wild_cluster with every image as its own cluster
    (G = 200, 9,999 Webb draws, RandomState(2026)). Also: per-seed means; the paired per-image difference from R4 on
    the same features (WCU); the paired difference centre bias off minus on (WCU). Unadjusted, exploratory.
Usage (after the rule is public): V3_DEVICE=cpu $PY -B -I explore/a6_absolute.py
Outputs: results/explore/A6_absolute.json and results/explore/A6_absolute.md
"""
import os, sys, time
sys.dont_write_bytecode = True
os.environ["V3_DEVICE"] = "cpu"                                           # CPU only; recommon then loads the stub
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
FIRST = "dinov2_L12_R4_s0"
TOL = 1e-4
FLOOR = 1e-12
CACHE = os.path.join(X.ROOT, "v2", "readout", "cache")
CKPT = os.path.join(X.ROOT, "v3", "gpu", "full", "ckpt")


# ====================================================================== metrics (numpy only; unit-tested)
def norm_log2(m, floor=FLOOR):
    """log2 of a non-negative map with a floor of floor x max, normalised to sum 1."""
    p = np.asarray(m, np.float64)
    p = p + floor * p.max()
    return np.log2(p / p.sum())


def info_gain(s_full, fix, base_log2):
    """Bits per fixation: mean over fixated pixels of log2 p_model - log2 p_baseline (both normalised maps)."""
    f = np.asarray(fix) > 0.5
    if not f.any():
        return float("nan")
    return float(np.mean(norm_log2(s_full)[f] - np.asarray(base_log2, np.float64)[f]))


def gate_compare(cc_img, nss_img, pub, tol=TOL):
    """Centre-bias-on reproduction: mean CC, mean NSS and per-image CC (published at 5 decimals) within tol."""
    cc_img, nss_img = np.asarray(cc_img, float), np.asarray(nss_img, float)
    pci = np.asarray(pub["cc_per_image"], float)
    d_img = float(np.max(np.abs(cc_img - pci))) if pci.size == cc_img.size else float("inf")
    d_cc, d_nss = abs(float(cc_img.mean()) - float(pub["CC"])), abs(float(nss_img.mean()) - float(pub["NSS"]))
    return dict(CC=float(cc_img.mean()), NSS=float(nss_img.mean()), CC_published=float(pub["CC"]),
                NSS_published=float(pub["NSS"]), abs_diff_CC=d_cc, abs_diff_NSS=d_nss, max_abs_diff_per_image_CC=d_img,
                n_images=int(cc_img.size), tolerance=tol, passed=bool(d_cc <= tol and d_nss <= tol and d_img <= tol),
                note="per-image published CC is rounded to 5 decimals (adds at most 5e-6)")


def summarise(v, g=None):
    """Mean over images with every image as its own cluster (analyze_re.wild_cluster)."""
    v = np.asarray(v, float)
    return AR.wild_cluster(v, np.arange(v.size) if g is None else g)


# ====================================================================== model part (torch; imported in main only)
def load_mit(RC):
    import cv2
    items = RC.mit_heldout()
    gts = [cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in items]
    fxs = [(cv2.imread(p["fix"], cv2.IMREAD_GRAYSCALE) > 128) for p in items]
    return items, gts, fxs


def mean_train_map(RC):
    keep, _ = RC.train_items()
    maps = RC.train_maps()
    acc = np.zeros(maps.shape[1:], np.float64)
    for b in range(0, len(keep), 512):
        acc += np.asarray(maps[keep[b:b + 512]], np.float64).sum(0)
    return acc / len(keep), len(keep), int(maps.shape[0])


def head_maps(m, F, S, use_cb, torch, bs=16):
    m.use_cb = use_cb
    out = []
    with torch.no_grad():
        for b in range(0, F.shape[0], bs):
            x = torch.from_numpy(np.asarray(F[b:b + bs], dtype=np.float16)).float()
            out.append(torch.sigmoid(m(x, (S, S)))[:, 0].cpu().numpy())
    return np.concatenate(out)


def score_head(sal, gts, fxs, base, RC):
    import cv2
    cc, nss, ig = [], [], []
    for i in range(sal.shape[0]):
        gt = gts[i]; hh, ww = gt.shape
        s_ = cv2.resize(sal[i], (ww, hh))
        cc.append(RC.CC(s_, gt)); nss.append(RC.NSS(s_, fxs[i])); ig.append(info_gain(s_, fxs[i], base[i]))
    return np.array(cc), np.array(nss), np.array(ig)


def run_heads(RC, torch, gts, fxs, base, log):
    """Per head: centre bias on (gate) and off. Returns {tag: dict}; stops if the first head fails its gate."""
    import cv2  # noqa: F401
    tags = [f"{fs}_{ar}_s{s}" for fs in AR.SWAP_FSETS for ar in AR.ARCHS for s in (0, 1, 2)]
    tags.remove(FIRST); tags.insert(0, FIRST)
    feats, res = {}, {}
    for k, tag in enumerate(tags):
        fs = next(f for f in AR.SWAP_FSETS if tag.startswith(f + "_"))
        ar, seed = tag[len(fs) + 1:].split("_s")
        if fs not in feats:
            feats = {fs: np.load(os.path.join(CACHE, f"{fs}_mit200.npy"), mmap_mode="r")}
            assert feats[fs].shape[0] == len(gts), feats[fs].shape
        t0 = time.time()
        m = RC.make_head(ar, RC.FSETS[fs]["cin"])
        m.load_state_dict(torch.load(os.path.join(CKPT, tag + ".pt"), map_location="cpu"))
        m = m.to("cpu").eval()
        if k == 0:
            X.metal_guard()
        pub = AR._load(os.path.join(X.SWAP_SCORES, f"mit_{tag}.json"))
        on = score_head(head_maps(m, feats[fs], RC.S, True, torch), gts, fxs, base, RC)
        gate = gate_compare(on[0], on[1], pub)
        rec = dict(fset=fs, arch=ar, seed=int(seed), gate=gate, cb_on=dict(CC=on[0], NSS=on[1], IG=on[2]))
        if tag == FIRST and not gate["passed"]:
            log(f"GATE FAILED on {tag}: {gate}")
            return res, dict(first_head=tag, passed=False, gate=gate)
        if gate["passed"]:
            off = score_head(head_maps(m, feats[fs], RC.S, False, torch), gts, fxs, base, RC)
            rec["cb_off"] = dict(CC=off[0], NSS=off[1], IG=off[2])
        res[tag] = rec
        log(f"{k + 1}/60 {tag} gate {'pass' if gate['passed'] else 'FAIL'} (dCC {gate['abs_diff_CC']:.2e}, dNSS "
            f"{gate['abs_diff_NSS']:.2e}, max per-image {gate['max_abs_diff_per_image_CC']:.2e})"
            + (f"; cb off CC {rec['cb_off']['CC'].mean():.4f} NSS {rec['cb_off']['NSS'].mean():.4f} IG "
               f"{rec['cb_off']['IG'].mean():.4f}" if "cb_off" in rec else "") + f" {time.time() - t0:.1f}s")
    return res, dict(first_head=FIRST, passed=res[FIRST]["gate"]["passed"], gate=res[FIRST]["gate"])


def cell_stats(res):
    out = {}
    for fs in AR.SWAP_FSETS:
        for ar in AR.ARCHS:
            tags = [f"{fs}_{ar}_s{s}" for s in (0, 1, 2)]
            if not all(t in res and "cb_off" in res[t] for t in tags):
                out[f"{fs}|{ar}"] = "not computable (a seed failed its gate)"
                continue
            rec = {}
            for cb in ("cb_off", "cb_on"):
                rec[cb] = {}
                for met in ("CC", "NSS", "IG"):
                    per = np.stack([res[t][cb][met] for t in tags])
                    rec[cb][met] = dict(summarise(per.mean(0)), seed_means=[float(x) for x in per.mean(1)])
            rec["off_minus_on"] = {met: summarise(np.mean([res[t]["cb_off"][met] - res[t]["cb_on"][met] for t in tags], 0))
                                   for met in ("CC", "NSS", "IG")}
            out[f"{fs}|{ar}"] = rec
    for fs in AR.SWAP_FSETS:
        r4 = [f"{fs}_R4_s{s}" for s in (0, 1, 2)]
        for ar in AR.ARCHS:
            if ar == "R4" or not isinstance(out[f"{fs}|{ar}"], dict) or not isinstance(out[f"{fs}|R4"], dict):
                continue
            tags = [f"{fs}_{ar}_s{s}" for s in (0, 1, 2)]
            out[f"{fs}|{ar}"]["minus_R4_cb_off"] = {
                met: summarise(np.mean([res[t]["cb_off"][met] for t in tags], 0) -
                               np.mean([res[t]["cb_off"][met] for t in r4], 0)) for met in ("CC", "NSS", "IG")}
    return out


# ====================================================================== markdown
def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def lt(r):
    return f"{r['mean']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"


def dt(r):
    return f"{f4(r['mean'])} [{f4(r['ci'][0])}, {f4(r['ci'][1])}], p {r['p']:.4g}"


def write_md(res, path):
    L = []
    w = L.append
    w("# A6 absolute metrics with the centre bias off (`a6_absolute.py`, RUN_PLAN S4)\n")
    w("**EXPLORATORY** (REANALYSIS_RULE.md Section 6 item 7, second part; unadjusted, never decision-bearing). Held-out "
      "MIT1003 (last 200 images), 336 px, heads only on the cached float16 features, CPU. Values: mean over the 200 "
      "images of the seed-averaged per-image metric, 95% WCU interval with every image as its own cluster (G = 200, "
      "9,999 Webb draws). IG in bits per fixation against the mean training map of ab/maps.npy.\n")
    g = res["gate_first_head"]
    w(f"- Run {res['run']['start_utc']} to {res['run']['end_utc']} ({res['run']['compute_s']:.0f} s). require_public: "
      f"{res['run']['hashed_files_verified']} hashed files verified.")
    w(f"- Gate (centre bias ON, {g['first_head']} first): mean CC {g['gate']['CC']:.6f} against published "
      f"{g['gate']['CC_published']:.6f} (|difference| {g['gate']['abs_diff_CC']:.2e}); mean NSS {g['gate']['NSS']:.6f} "
      f"against {g['gate']['NSS_published']:.6f} ({g['gate']['abs_diff_NSS']:.2e}); largest per-image CC difference "
      f"{g['gate']['max_abs_diff_per_image_CC']:.2e}; tolerance {g['gate']['tolerance']}: "
      f"{'PASS' if g['passed'] else 'FAIL'}.")
    ga = res["gate_all"]
    w(f"- All heads: {ga['n_passed']} of {ga['n']} pass; largest |mean CC difference| {ga['max_abs_diff_CC']:.2e}, "
      f"|mean NSS difference| {ga['max_abs_diff_NSS']:.2e}, per-image CC {ga['max_abs_diff_per_image_CC']:.2e}.")
    w(f"- Baseline: mean of {res['baseline']['n_maps']} training maps (336 x 336); min/max of the mean map "
      f"{res['baseline']['min']:.4g} / {res['baseline']['max']:.4g}; floor {FLOOR} x max.\n")
    w("## Centre bias off (rule quantity)\n")
    w("| cell | CC | NSS | IG (bits/fix) | CC minus R4 | IG minus R4 |")
    w("|---|---|---|---|---|---|")
    for k, r in res["cells"].items():
        if not isinstance(r, dict):
            w(f"| {k} | {r} | | | | |")
            continue
        o = r["cb_off"]; d = r.get("minus_R4_cb_off")
        w(f"| {k} | {lt(o['CC'])} | {lt(o['NSS'])} | {lt(o['IG'])} | {dt(d['CC']) if d else 'reference'} | "
          f"{dt(d['IG']) if d else 'reference'} |")
    w("\n## Centre bias on (context; CC and NSS reproduce the published values) and the change when it is switched off\n")
    w("| cell | CC on | NSS on | IG on | CC off minus on | NSS off minus on | IG off minus on |")
    w("|---|---|---|---|---|---|---|")
    for k, r in res["cells"].items():
        if not isinstance(r, dict):
            continue
        o, d = r["cb_on"], r["off_minus_on"]
        w(f"| {k} | {lt(o['CC'])} | {lt(o['NSS'])} | {lt(o['IG'])} | {dt(d['CC'])} | {dt(d['NSS'])} | {dt(d['IG'])} |")
    s = res["summary"]
    w(f"\n- Centre bias off lowers CC in {s['n_cc_lower_off']} of {s['n_cells']} cells and IG in {s['n_ig_lower_off']} of "
      f"{s['n_cells']}; IG with the centre bias off is above 0 (WCU lower bound) in {s['n_ig_off_above0']} of {s['n_cells']}.")
    w(f"- Accuracy ranking with the centre bias off: the cell order by CC agrees with the centre-bias-on order in "
      f"Spearman rho {s['spearman_cc_on_off']:.4f} over the 20 cells (descriptive).")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path, flush=True)


# ====================================================================== main
def main():
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_hashed = X.require_public()                                     # rule Section 8.1, before anything is read
    sys.path.insert(0, os.path.join(X.RE, "gpu"))
    import recommon as RC                                             # stub ab_objective (V3_DEVICE=cpu), read only
    import torch
    assert RC.DEVICE == "cpu", RC.DEVICE
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    X.metal_guard()
    X.log("loading MIT1003 held-out maps and the training-map baseline")
    items, gts, fxs = load_mit(RC)
    assert len(items) == 200
    mm, n_keep, n_all = mean_train_map(RC)
    import cv2
    base = [norm_log2(cv2.resize(mm.astype(np.float32), (g.shape[1], g.shape[0]))).astype(np.float32) for g in gts]
    X.log(f"baseline from {n_keep} of {n_all} training maps; min {mm.min():.4g} max {mm.max():.4g}")
    res, first = run_heads(RC, torch, gts, fxs, base, X.log)
    out = dict(gate_first_head=first)
    if not first["passed"]:
        out["status"] = "stopped: the first head failed the centre-bias-on reproduction gate; nothing else computed"
    else:
        gates = [r["gate"] for r in res.values()]
        out["gate_all"] = dict(n=len(gates), n_passed=sum(g["passed"] for g in gates),
                               max_abs_diff_CC=max(g["abs_diff_CC"] for g in gates),
                               max_abs_diff_NSS=max(g["abs_diff_NSS"] for g in gates),
                               max_abs_diff_per_image_CC=max(g["max_abs_diff_per_image_CC"] for g in gates),
                               failed=[t for t, r in res.items() if not r["gate"]["passed"]])
        out["per_head"] = {t: dict(gate=r["gate"],
                                   cb_on={k: float(np.mean(v)) for k, v in r["cb_on"].items()},
                                   cb_off={k: float(np.mean(v)) for k, v in r["cb_off"].items()} if "cb_off" in r else None)
                           for t, r in res.items()}
        out["cells"] = cell_stats(res)
        cells = [r for r in out["cells"].values() if isinstance(r, dict)]
        from scipy.stats import spearmanr
        out["summary"] = dict(n_cells=len(cells),
                              n_cc_lower_off=sum(r["off_minus_on"]["CC"]["mean"] < 0 for r in cells),
                              n_ig_lower_off=sum(r["off_minus_on"]["IG"]["mean"] < 0 for r in cells),
                              n_ig_off_above0=sum(r["cb_off"]["IG"]["ci"][0] > 0 for r in cells),
                              spearman_cc_on_off=float(spearmanr([r["cb_on"]["CC"]["mean"] for r in cells],
                                                                 [r["cb_off"]["CC"]["mean"] for r in cells]).statistic))
        out["status"] = "complete"
    out["baseline"] = dict(n_maps=n_keep, n_in_file=n_all, min=float(mm.min()), max=float(mm.max()), floor=FLOOR,
                           definition="mean of ab/maps.npy over fullcommon.train_items() (uint8 maps, mean in float64), "
                                      "resized per image (cv2 bilinear), floor x max added, normalised; log2")
    out["definitions"] = dict(IG="mean over fixated pixels of log2 p_model - log2 p_baseline (bits per fixation)",
                              interval="analyze_re.wild_cluster with every image as its own cluster (G = 200)",
                              heads="60 swap checkpoints, v3/gpu/full/ckpt; features v2/readout/cache/<fset>_mit200.npy")
    out["run"] = dict(start_utc=start, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), compute_s=time.time() - t0,
                      hashed_files_verified=n_hashed, torch=torch.__version__, threads=torch.get_num_threads())
    X.write_json("A6_absolute.json", out, os.path.abspath(__file__),
                 inputs=["v3/gpu/full/ckpt/<tag>.pt", "v2/readout/cache/<fset>_mit200.npy", "ab/maps.npy", "ab/meta.json",
                         "MIT1003 (saliency_research/data/mit1003) last 200", "v3/gpu/full/scores/mit_<tag>.json"],
                 extra_code=[os.path.join(X.RE, "gpu", "recommon.py")])
    if out["status"] == "complete":
        write_md(X._clean(out), os.path.join(X.RESULTS, "A6_absolute.md"))
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
