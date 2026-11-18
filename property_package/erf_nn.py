"""
Gradient-based effective receptive field (ERF) of the sweep readouts and of the blocks they read
(nn/ANALYSIS_RULE.md, Section 4). CPU only (NN_THREADS threads, default 6); no training, no new data.

Image set: 24 displays = 12 configurations (one per shape x support ratio x polarity cell, cells in the order of
v3/scores/analyze_alt.Design().keys, position = cell index mod 5) x {illusory, control180}, from v2/stim_v3.
Locations per display (display px, from v3/stim_alt/aoi/<config>_aoi.npz): inside-AOI centroid + centroids of its four
quadrants (inside set, 5) and the centroid of every 8-connected component of the inducer-disc AOI (inducer set, 3 or 4).
Readout ERF: target = pre-sigmoid output of sweep.Readout (use_cb False) at the s x s output pixel nearest to the
location (round half up); backbone ERF: target = q . hidden_states[L][token containing the location], q ~ N(0, I) with
torch.Generator().manual_seed(1000003 * display_index + 1009 * location_index + L).
G(p) = sum_c |d target / d x_c(p)| / std_c on the s x s model input; w = G / sum G; radial CDF of w over Euclidean
distance in display px (model-input pixel centre and target centre mapped to 1024 x 768) for r = 0..1280, and over
distance in model-input px for r = 0..639. CDFs averaged over the locations of a display, then over displays.
Outputs: nn/results/erf/<bb>.json (ERF50/ERF90 per block, size, kind, location set; per-display ERF50) and
nn/results/erf/<bb>_cdfs.npz (per-display CDFs). Resumable per display (nn/results/erf/<bb>_partial.npz).
Usage: python erf_nn.py <dinov2|clip|siglip> [n_displays (default 24; smaller only for timing tests)] [out_dir]
"""
import os, sys, time, json
sys.dont_write_bytecode = True
_T = os.environ.get("NN_THREADS", "6")
for k, v in (("OMP_NUM_THREADS", _T), ("MKL_NUM_THREADS", _T), ("VECLIB_MAXIMUM_THREADS", _T),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[k] = v
import numpy as np, cv2, torch
from scipy import ndimage
torch.set_num_threads(int(_T))
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nn as C                                   # noqa: E402
CA, AA = C.import_v3()
import ab_objective as AB                               # noqa: E402
AB.DEV = "cpu"

BB = sys.argv[1]
NDISP = int(sys.argv[2]) if len(sys.argv) > 2 else 24
OUT = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(C.RES, "erf")
assert OUT.startswith(C.NN), OUT
os.makedirs(OUT, exist_ok=True); os.makedirs(C.LOGS, exist_ok=True)
LOG = os.path.join(C.LOGS, f"erf_{BB}.log")
RD, RM = 1281, 640                                      # CDF grid: display px 0..1280, model px 0..639
W, H = 1024, 768
P = C.PATCH[BB]


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def display_set():
    keys = AA.DES["all"].keys
    assert len(keys) == 12
    out = []
    for i, (sh, sr, pol) in enumerate(keys):
        cfg = f"{sh}_sr{int(round(sr * 100))}_{pol}_p{i % 5}"
        assert cfg in CA.CONFIGS, cfg
        for cond in ("illusory", "control180"):
            out.append((cfg, cond))
    return out


def locations(cfg):
    a = CA.aoi(cfg)
    ys, xs = np.nonzero(a["inside"]); cx, cy = xs.mean(), ys.mean()
    inside = [(cx, cy)]
    for qx in (False, True):
        for qy in (False, True):
            m = ((xs >= cx) == qx) & ((ys >= cy) == qy)
            assert m.sum() > 0, cfg
            inside.append((xs[m].mean(), ys[m].mean()))
    lab, n = ndimage.label(a["ind"], structure=np.ones((3, 3), bool))
    shape = cfg.split("_")[0]
    assert n == {"square": 4, "triangle": 3}[shape], (cfg, n)
    ind = []
    for c in range(1, n + 1):
        yy, xx = np.nonzero(lab == c); ind.append((xx.mean(), yy.mean()))
    return inside, ind


def cdf(w, d, n):
    idx = np.clip(np.ceil(d - 1e-9).astype(np.int64), 0, n - 1)
    return np.cumsum(np.bincount(idx.ravel(), weights=w.ravel(), minlength=n)[:n])


def main():
    argv = sys.argv; sys.argv = ["sweep.py", BB]
    import sweep
    sys.argv = argv
    assert sweep.DEV == "cpu" and sweep.BB == BB
    bb = sweep.load_backbone(); cin = bb.config.hidden_size
    for p in bb.parameters():
        p.requires_grad_(False)
    R = {}
    for L in C.LAYERS[BB]:
        m = sweep.Readout(cin)
        m.load_state_dict(torch.load(os.path.join(sweep.WORK, f"readout_L{L}.pt"), map_location="cpu")); m.eval(); m.use_cb = False
        for p in m.parameters():
            p.requires_grad_(False)
        R[L] = m
    std = torch.from_numpy(sweep.STD.astype(np.float32))
    cls = sweep.C["cls"]
    disps = display_set()[:NDISP]
    log(f"start {BB}: {len(disps)} displays, layers {C.LAYERS[BB]}, sizes {C.SIZES}, threads {torch.get_num_threads()}")
    part = os.path.join(OUT, f"{BB}_partial.npz")
    # store[(L, kind, s, set)] -> per-display CDF arrays; kinds ro (readout) / bb (backbone token); sets all/inside/ind
    store, done = {}, []
    if os.path.exists(part) and NDISP == 24:
        z = np.load(part, allow_pickle=True)
        store, done = z["store"].item(), list(z["done"])
        log(f"resuming after {len(done)} displays")
    t0 = time.time()
    for n, (cfg, cond) in enumerate(disps):
        if n in done:
            continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(C.STIM_V3, "img", f"{cfg}_{cond}.png")), cv2.COLOR_BGR2RGB)
        assert rgb.shape == (H, W, 3)
        ins, ind = locations(cfg)
        locs = [(X, Y, "inside") for X, Y in ins] + [(X, Y, "ind") for X, Y in ind]
        for s in C.SIZES:
            g = s // P
            rr, cc = np.mgrid[0:s, 0:s]
            Xp, Yp = (cc + 0.5) * W / s - 0.5, (rr + 0.5) * H / s - 0.5
            x = sweep.prep(rgb, s)[None].clone().requires_grad_(True)
            hs = bb(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
            for L in C.LAYERS[BB]:
                t = hs[L][:, 1:, :] if cls else hs[L]
                assert t.shape[1] == g * g, (t.shape, g)
                f = t.transpose(1, 2).reshape(1, -1, g, g)
                zmap = R[L](f, (s, s))
                acc = {(kind, st, sp): [] for kind in ("ro", "bb") for st in ("inside", "ind") for sp in ("d", "m")}
                for k, (X, Y, st) in enumerate(locs):
                    u, v = (X + 0.5) * s / W - 0.5, (Y + 0.5) * s / H - 0.5
                    i, j = int(np.clip(np.floor(v + 0.5), 0, s - 1)), int(np.clip(np.floor(u + 0.5), 0, s - 1))
                    tr, tc = int(np.clip(np.floor(v / P), 0, g - 1)), int(np.clip(np.floor(u / P), 0, g - 1))
                    q = torch.randn(t.shape[-1], generator=torch.Generator().manual_seed(1000003 * n + 1009 * k + L))
                    targets = {
                        "ro": (zmap[0, 0, i, j], ((j + 0.5) * W / s - 0.5, (i + 0.5) * H / s - 0.5), (j, i)),
                        "bb": ((t[0, tr * g + tc] * q).sum(), ((tc * P + P / 2) * W / s - 0.5, (tr * P + P / 2) * H / s - 0.5),
                               (tc * P + (P - 1) / 2, tr * P + (P - 1) / 2)),
                    }
                    for kind, (y, (Xc, Yc), (mc, mr)) in targets.items():
                        gr, = torch.autograd.grad(y, x, retain_graph=True)
                        G = (gr[0].abs() / std[:, None, None]).sum(0).numpy().astype(np.float64)
                        w = G / G.sum()
                        acc[(kind, st, "d")].append(cdf(w, np.hypot(Xp - Xc, Yp - Yc), RD))
                        acc[(kind, st, "m")].append(cdf(w, np.hypot(cc - mc, rr - mr), RM))
                for kind in ("ro", "bb"):
                    for sp in ("d", "m"):
                        a_in, a_ind = acc[(kind, "inside", sp)], acc[(kind, "ind", sp)]
                        for st, lst in (("all", a_in + a_ind), ("inside", a_in), ("ind", a_ind)):
                            store.setdefault((L, kind, s, st, sp), {})[n] = np.mean(lst, 0)
            del hs, zmap, x
        done.append(n)
        if NDISP == 24:
            np.savez(part, store=np.array(store, dtype=object), done=np.array(done))
        el = time.time() - t0
        nd = len([d for d in done])
        log(f"  display {n + 1}/{len(disps)} {cfg}_{cond}: {len(locs)} locations, {el:.0f}s elapsed")
    # summary
    def first(c, thr):
        return int(np.argmax(c >= thr)) if (c >= thr).any() else None
    res = dict(backbone=BB, hf=sweep.C["hf"], patch=P, displays=[f"{a}_{b}" for a, b in disps], sizes=list(C.SIZES),
               units="display px (1024 x 768) unless key ends in _model (model-input px)", layers={})
    arrs = {}
    for L in C.LAYERS[BB]:
        rec = {}
        for kind in ("ro", "bb"):
            for s in C.SIZES:
                for st in ("all", "inside", "ind"):
                    for sp in ("d", "m"):
                        per = store[(L, kind, s, st, sp)]
                        M = np.stack([per[k] for k in sorted(per)])
                        mean = M.mean(0)
                        key = f"{kind}_s{s}_{st}" + ("" if sp == "d" else "_model")
                        rec[key] = dict(erf50=first(mean, 0.5), erf90=first(mean, 0.9),
                                        per_display_erf50=[first(c, 0.5) for c in M])
                        arrs[f"L{L}_{key}"] = M.astype(np.float32)
            for st in ("all", "inside", "ind"):
                a, b = rec[f"{kind}_s336_{st}"], rec[f"{kind}_s448_{st}"]
                rec[f"{kind}_sizeavg_{st}"] = dict(erf50=0.5 * (a["erf50"] + b["erf50"]), erf90=0.5 * (a["erf90"] + b["erf90"]))
        res["layers"][str(L)] = rec
        log(f"{BB} L{L}: readout ERF50 {rec['ro_sizeavg_all']['erf50']:.1f} px (336 {rec['ro_s336_all']['erf50']}, 448 "
            f"{rec['ro_s448_all']['erf50']}), ERF90 {rec['ro_sizeavg_all']['erf90']:.1f}; block-token ERF50 "
            f"{rec['bb_sizeavg_all']['erf50']:.1f}, ERF90 {rec['bb_sizeavg_all']['erf90']:.1f}")
    res["seconds"] = round(time.time() - t0, 1)
    suffix = "" if NDISP == 24 else f"_test{NDISP}"
    json.dump(res, open(os.path.join(OUT, f"{BB}{suffix}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{BB}{suffix}_cdfs.npz"), **arrs)
    log(f"done {BB}: {res['seconds']}s")


if __name__ == "__main__":
    main()
