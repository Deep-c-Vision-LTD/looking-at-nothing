"""
neuro/reanalysis/explore/ttr/ttr_score.py: re-scoring of the frozen readouts that read the A15(a)-flagged blocks, with
and without test-time registers (Jiang et al. 2025). EXPLORATORY (REANALYSIS_RULE.md Section 6, item 12; DESIGN.md
Sections 4.3, 5 and 6). Written 18 November 2026 after P1 to P5, R8 and every exploratory output had been seen, before
any TTR norm or score existed.

Modes (one process each, one at a time; the task's run order is plain -> selection -> norms -> arms):
  plain <clip|siglip>   gate F (feature identity: the wrapper's plain features equal the stock sweep.feats and, for
                        SigLIP-L, fullcommon.feats bit for bit on 4 held-out MIT1003 images at both sizes, also after
                        hooks were installed and removed), then the plain arm on everything in scope, then gate P
                        (pipeline check: every published per-image value reproduced to 1e-4; sweep MIT1003 CC, published
                        only as 4-decimal means, to 1.5e-4, flag F7). Exit code 3 if any unit fails (stop and report).
  norms                 gate K for both backbones (before any TTR score is written): plain, reg0 and ttr token-norm
                        statistics of A15(a) on its 24 Kanizsa displays and 24 held-out MIT1003 images at 336 and 448
                        px, at the chosen setting; plain reproduces A15a_token_norms.json; criterion: 90th percentile of
                        max/median <= 2.0 at every flagged block b >= e; if it fails the default applies (proceed, and
                        label the blocks "not suppressed on the displays"; flag F5).
  arms <clip|siglip>    the ttr arm (stim_v3, MIT1003, stim_alt) and the reg0 arm (stim_v3, MIT1003); refuses unless
                        gate P passed for the backbone and gate K ran with the same TTR_select file (SHA-256 compared).
Scope (DESIGN.md Section 5, Tiers A, A2 and B; flag F6 default on): the 10 sweep readouts at the flagged blocks (CLIP-L
12, 16, 20, 24; SigLIP-L 4, 8, 12, 16, 20, 24; single seed, readout_L<L>.pt) and the 15 SigLIP-L block-24 swap heads
(R1n, R1, R2, R3, R4 x seeds 0, 1, 2; v3/gpu/full/ckpt), readouts frozen (flag F1); displays: the 480 stim_v3 Kanizsa
displays with published sweep values (60 configurations x illusory, control180, control90, real, real_filled, mis10,
mis20, mis30) and the 360 stim_alt displays, at 336 and 448 px, centre bias off; held-out MIT1003 200 at 336 px, centre
bias on (CC). One backbone pass per display and size serves every head (SigLIP-L: 6 sweep + 15 swap heads).
Maps: sweep q16(cv2.resize(float32 sigmoid)) (mapsrc published form); swap eval_full.score_map (float16 sigmoid ->
float32 -> resize -> q16); measure r_inside (common_alt.measures), computed on the AOI pixels (ttr_common.r_inside_fast;
checked for exact equality against the boolean-mask formula of common_alt.measures on the first displays). MIT1003 CC
as sweep.evaluate (sweep) and eval_full.run_mit (swap). No map is saved.
Per-display token diagnostic at every flagged block and arm: max/median, share and count above twice the median, the
register norm, and the shares of the tokens above twice the median whose patch centre falls inside the "inside" AOI,
inside the inducer AOI or elsewhere (scores_ttr/norms/).
Outputs: scores_ttr/<arm>/{stim_v3,stim_alt}_<tag>_s<seed>.json ({size: {"r_inside": {name: value}}}),
scores_ttr/<arm>/mit_<tag>_s<seed>.json, scores_ttr/norms/<arm>_<set>_<bb>.json, results/explore/ttr/
TTR_score_<bb>_<mode>.json and TTR_gateK.json (gates, timings, hashes).
CPU only: V3_DEVICE=cpu, the ab_objective stub (mapsrc.torch_env), torch threads 4, excommon.metal_guard() after every
model load; require_public() first.
Usage: nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       $PY -B -I explore/ttr/ttr_score.py plain siglip
"""
import argparse, json, os, sys, time, types
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("OMP_NUM_THREADS", "4")
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
os.environ["V3_DEVICE"] = "cpu"
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import ttr_common as C                                                    # noqa: E402

X = C.X
BATCH = 6
TOL = 1e-4
TOL_SWEEP_MIT_MEAN = 1.5e-4
N_EXACT_CHECK = 3                     # displays per unit and size checked against the boolean-mask formula


# ====================================================================== environment, models, heads
def env():
    import mapsrc                                                         # explore/mapsrc.py (torch_env: stub, CPU)
    T = mapsrc.torch_env()
    T["torch"].set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    T["mapsrc"] = mapsrc
    return T


def load_backbone_and_heads(T, bb, with_heads=True):
    torch, mapsrc = T["torch"], T["mapsrc"]
    sw = mapsrc.sweep_module(bb)
    model = sw.load_backbone()
    assert next(model.parameters()).device.type == "cpu"
    X.metal_guard()
    heads = dict(sweep={}, swap={})
    if with_heads:
        cin = model.config.hidden_size
        for L in C.FLAGGED[bb]:
            m = sw.Readout(cin)
            m.load_state_dict(torch.load(os.path.join(sw.WORK, f"readout_L{L}.pt"), map_location="cpu"))
            m.eval(); m.use_cb = False
            heads["sweep"][f"{bb}_L{L}"] = m
        if bb == "siglip":
            for u in C.SWAP_UNITS:
                fs, ar = u.split("|")
                for seed in C.SWAP_SEEDS:
                    ck = os.path.join(mapsrc.SWAP_CKPT, f"{fs}_{ar}_s{seed}.pt")
                    heads["swap"][(u, seed)] = mapsrc.load_head(ck, ar, fs)
        X.metal_guard()
    return sw, model, heads


def tag_of(unit):
    return unit.replace("|", "_")


# ====================================================================== one pass over images
def forward(T, sw, model, bb, arm, nd, rgbs, s, heads, want_norms=True):
    """One backbone pass for a batch at size s under an arm; returns sweep and swap sigmoid maps (numpy, s x s) and the
    token diagnostics at the flagged blocks."""
    torch = T["torch"]
    with torch.no_grad():
        x = torch.stack([sw.prep(r, s) for r in rgbs])
        with C.arm_hooks(model, bb, arm, nd):
            hs = C.hidden_states(model, x)
        out = dict(sweep={}, swap={}, norms={})
        for u, m in heads["sweep"].items():
            L = int(u.split("_L")[1])
            f = C.grid_feats(hs[L], bb, arm)
            out["sweep"][u] = torch.sigmoid(m(f, (s, s))[:, 0]).numpy()
        if heads["swap"]:
            f24 = C.grid_feats(hs[24], bb, arm).half().float()
            for key, m in heads["swap"].items():
                out["swap"][key] = torch.sigmoid(m(f24, (s, s))[:, 0]).numpy()
        if want_norms:
            for L in C.FLAGGED[bb]:
                out["norms"][L] = (C.patch_norms(hs[L], bb, arm), C.register_norm(hs[L], arm))
        del hs
    return out


def mit_forward(T, sw, model, bb, arm, nd, rgbs, heads):
    """MIT1003 at 336 px, centre bias on: sweep heads on float32 features, swap heads on float16-held features; the
    float32 sigmoid map (s x s) is returned (resized to the image by the caller, as sweep.evaluate / eval_full)."""
    torch = T["torch"]
    s = 336
    with torch.no_grad():
        x = torch.stack([sw.prep(r, s) for r in rgbs])
        with C.arm_hooks(model, bb, arm, nd):
            hs = C.hidden_states(model, x)
        out = dict(sweep={}, swap={})
        for u, m in heads["sweep"].items():
            L = int(u.split("_L")[1])
            m.use_cb = True
            out["sweep"][u] = torch.sigmoid(m(C.grid_feats(hs[L], bb, arm), (s, s)))[:, 0].numpy()
            m.use_cb = False
        if heads["swap"]:
            f24 = C.grid_feats(hs[24], bb, arm).half().float()
            for key, m in heads["swap"].items():
                m.use_cb = True
                out["swap"][key] = torch.sigmoid(m(f24, (s, s)))[:, 0].numpy()
                m.use_cb = False
    return out


class Store:
    """Per-display r_inside of every unit, the exact-formula check and the token diagnostics of one arm."""

    def __init__(self):
        self.r = {}            # (unit, seed) -> stimset -> size -> {name: r}
        self.norms = {}        # stimset -> name -> size -> block -> record
        self.exact = dict(n=0, max_abs_diff=0.0)
        self.mit = {}          # (unit, seed) -> [cc]

    def put(self, unit, seed, stimset, s, name, v):
        self.r.setdefault((unit, seed), {}).setdefault(stimset, {}).setdefault(str(s), {})[name] = v


def exact_r(full, disp):
    """common_alt.measures' r_inside with boolean masks (the published formula), for the exactness check."""
    q = C.q16(full)
    sm = np.clip(np.asarray(q, np.float32).astype(np.float64), 0, None)
    I, D = sm[disp["inside_mask"]].sum(), sm[disp["ind_mask"]].sum()
    return float(I / (I + D)) if I + D > 0 else float("nan")


def run_displays(T, sw, model, bb, arm, nd, heads, stimset, store, timings):
    import cv2
    disps = C.displays(stimset)
    t0 = time.time()
    nchk = {}
    for i in range(0, len(disps), BATCH):
        chunk = disps[i:i + BATCH]
        rgbs = [cv2.cvtColor(cv2.imread(d["path"]), cv2.COLOR_BGR2RGB) for d in chunk]
        for s in C.SIZES:
            o = forward(T, sw, model, bb, arm, nd, rgbs, s, heads)
            for u, sig in o["sweep"].items():
                for j, d in enumerate(chunk):
                    full = C.sweep_full(sig[j])
                    v = C.r_inside_fast(full, d["ins"], d["ind"])
                    k = (u, s)
                    if nchk.get(k, 0) < N_EXACT_CHECK:
                        store.exact["max_abs_diff"] = max(store.exact["max_abs_diff"], abs(v - exact_r(full, d)))
                        store.exact["n"] += 1; nchk[k] = nchk.get(k, 0) + 1
                    store.put(u, 0, stimset, s, d["name"], v)
            for (u, seed), sig in o["swap"].items():
                for j, d in enumerate(chunk):
                    full = C.swap_full(sig[j])
                    v = C.r_inside_fast(full, d["ins"], d["ind"])
                    k = (u, seed, s)
                    if nchk.get(k, 0) < N_EXACT_CHECK:
                        store.exact["max_abs_diff"] = max(store.exact["max_abs_diff"], abs(v - exact_r(full, d)))
                        store.exact["n"] += 1; nchk[k] = nchk.get(k, 0) + 1
                    store.put(u, seed, stimset, s, d["name"], v)
            for L, (nrm, reg) in o["norms"].items():
                rows = C.norm_rows(nrm)
                g = int(round(nrm.shape[1] ** 0.5))
                for j, d in enumerate(chunk):
                    rec = dict(ratio=float(rows["ratio"][j]), share=float(rows["share"][j]),
                               count=int(rows["count"][j]), max=float(rows["max"][j]), median=float(rows["median"][j]),
                               register_norm=None if reg is None else float(reg[j]),
                               loc=C.token_locations(rows["gt2"][j], g, d["inside_mask"], d["ind_mask"]))
                    store.norms.setdefault(stimset, {}).setdefault(d["name"], {}).setdefault(str(s), {})[str(L)] = rec
        if (i // BATCH) % 10 == 0:
            C.log(f"  {arm} {stimset} {i + len(chunk)}/{len(disps)} displays ({time.time() - t0:.0f}s)")
    timings[f"{arm}_{stimset}_s"] = time.time() - t0
    C.log(f"{arm} {stimset}: {len(disps)} displays in {timings[f'{arm}_{stimset}_s']:.0f}s")


def run_mit(T, sw, model, bb, arm, nd, heads, store, timings):
    import cv2
    from src.metrics import CC
    import ab_objective as AB
    from src.data import build_mit1003
    items = build_mit1003(f"{AB.SR}/mit1003")[-200:]
    assert len(items) == 200
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        rgbs = [cv2.cvtColor(cv2.imread(p["image"]), cv2.COLOR_BGR2RGB) for p in chunk]
        gts = [cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in chunk]
        o = mit_forward(T, sw, model, bb, arm, nd, rgbs, heads)
        for u, sig in o["sweep"].items():
            for j, gt in enumerate(chunk):
                h, w = rgbs[j].shape[:2]
                store.mit.setdefault((u, 0), []).append(CC(cv2.resize(sig[j], (w, h)), gts[j]))
        for (u, seed), sig in o["swap"].items():
            for j in range(len(chunk)):
                hh, ww = gts[j].shape
                store.mit.setdefault((u, seed), []).append(CC(cv2.resize(sig[j], (ww, hh)), gts[j]))
    timings[f"{arm}_mit_s"] = time.time() - t0
    C.log(f"{arm} MIT1003 held-out 200: {timings[f'{arm}_mit_s']:.0f}s")


def write_store(store, arm, bb, sel_sha=None):
    d = os.path.join(C.SCORES_TTR, arm)
    for (u, seed), sets in store.r.items():
        for st, per in sets.items():
            C.dump_scores({sz: {"r_inside": v} for sz, v in per.items()},
                          os.path.join(d, f"{st}_{tag_of(u)}_s{seed}.json"))
    for (u, seed), cc in store.mit.items():
        C.dump_scores(dict(unit=u, seed=seed, n=len(cc), CC=float(np.mean(cc)), cc_per_image=cc, arm=arm,
                           device="cpu", selection_sha256=sel_sha), os.path.join(d, f"mit_{tag_of(u)}_s{seed}.json"))
    for st, per in store.norms.items():
        C.dump_scores(dict(arm=arm, backbone=bb, stimset=st, selection_sha256=sel_sha, displays=per),
                      os.path.join(C.SCORES_TTR, "norms", f"{arm}_{st}_{bb}.json"))


# ====================================================================== published values (gate P)
def published(unit, seed, stimset, s):
    if "|" in unit:
        fs, ar = unit.split("|")
        j = X.load_json(os.path.join(X.SWAP_SCORES, f"{stimset}_{fs}_{ar}_s{seed}.json"))
        return None if j is None else dict(j[str(s)]["r_inside"])
    bb, L = unit.split("_L")
    p = X.sweep_published_file(bb, int(L), s, stimset)
    j = X.load_json(p) if p else None
    return None if j is None else {k: v["r_inside"] for k, v in j.items()}


def gate_p(store, bb):
    out = dict(tolerance=TOL, units={}, mit={})
    allok = True
    for (u, seed), sets in sorted(store.r.items(), key=str):
        for st, per in sets.items():
            for sz, vals in per.items():
                pub = published(u, seed, st, int(sz))
                key = f"{u}|s{seed}|{st}|{sz}"
                if pub is None:
                    out["units"][key] = dict(passed=False, note="published file missing"); allok = False
                    continue
                miss = [n for n in vals if n not in pub]
                diffs = [abs(vals[n] - pub[n]) for n in vals if n in pub]
                ok = (not miss) and len(diffs) == len(vals) and max(diffs) <= TOL
                out["units"][key] = dict(n=len(diffs), missing=len(miss), max_abs_diff=max(diffs) if diffs else None,
                                         passed=bool(ok))
                allok &= ok
    for (u, seed), cc in sorted(store.mit.items(), key=str):
        if "|" in u:
            fs, ar = u.split("|")
            j = X.load_json(os.path.join(X.SWAP_SCORES, f"mit_{fs}_{ar}_s{seed}.json"))
            pub = j["cc_per_image"]
            diffs = [abs(a - b) for a, b in zip(cc, pub)]
            ok = len(pub) == len(cc) and max(diffs) <= TOL
            out["mit"][f"{u}|s{seed}"] = dict(n=len(diffs), max_abs_diff=max(diffs), tolerance=TOL,
                                              mean_cc=float(np.mean(cc)), published_mean=j["CC"],
                                              published_device=j.get("device"), passed=bool(ok))
        else:
            bbn, L = u.split("_L")
            j = X.load_json(os.path.join(C.ROOT, "sweep", bbn, "sweep_results.json"))
            pubm = j["mit1003"][str(L)]["CC"]
            d = abs(float(np.mean(cc)) - pubm)
            ok = d <= TOL_SWEEP_MIT_MEAN
            out["mit"][f"{u}|s{seed}"] = dict(n=len(cc), mean_cc=float(np.mean(cc)), published_mean_4dp=pubm,
                                              abs_diff=d, tolerance=TOL_SWEEP_MIT_MEAN, passed=bool(ok))
        allok &= ok
    vals = [v for v in out["units"].values() if v.get("max_abs_diff") is not None]
    out["n_display_values_compared"] = int(sum(v["n"] for v in vals))
    out["max_abs_diff_display"] = max(v["max_abs_diff"] for v in vals) if vals else None
    out["n_mit_values_compared"] = int(sum(v["n"] for k, v in out["mit"].items() if "|R" in k))
    out["max_abs_diff_mit_per_image"] = max([v["max_abs_diff"] for k, v in out["mit"].items() if "max_abs_diff" in v],
                                            default=None)
    out["passed"] = bool(allok)
    out["failed"] = sorted([k for k, v in out["units"].items() if not v["passed"]] +
                           [k for k, v in out["mit"].items() if not v["passed"]])
    return out


# ====================================================================== gate F
def gate_f(T, sw, model, bb):
    import cv2
    torch, R = T["torch"], T["R"]
    import ab_objective as AB
    from src.data import build_mit1003
    items = build_mit1003(f"{AB.SR}/mit1003")[-200:][:4]
    rgbs = [cv2.cvtColor(cv2.imread(p["image"]), cv2.COLOR_BGR2RGB) for p in items]
    res = {}
    ok = True
    for s in C.SIZES:
        x = torch.stack([sw.prep(r, s) for r in rgbs])
        with torch.no_grad():
            stock = sw.feats(model, x)
            with C.arm_hooks(model, bb, "plain"):
                hs = C.hidden_states(model, x)
            mine = {L: C.grid_feats(hs[L], bb, "plain") for L in C.BB_CFG[bb]["layers"]}
            with C.arm_hooks(model, bb, "reg0"):
                C.hidden_states(model, x)
            with C.TTR(model, C.BB_CFG[bb]["cls"], {0: [0, 1, 2]}, 1):
                C.hidden_states(model, x)
            hs2 = C.hidden_states(model, x)
            again = {L: C.grid_feats(hs2[L], bb, "plain") for L in C.BB_CFG[bb]["layers"]}
            r = {}
            for L in C.BB_CFG[bb]["layers"]:
                r[f"sweep_L{L}"] = float((mine[L] - stock[L]).abs().max())
                r[f"after_hooks_removed_L{L}"] = float((again[L] - mine[L]).abs().max())
            if bb == "siglip":
                ex = types.SimpleNamespace(m=model, bb="siglip")
                fc = R.FC.feats(ex, x, s)["siglip_L24"]
                r["fullcommon_siglip_L24"] = float((fc - mine[24]).abs().max())
            res[str(s)] = r
            ok &= all(v == 0.0 for v in r.values())
    return dict(images=[os.path.basename(p["image"]) for p in items], max_abs_diff=res, passed=bool(ok))


# ====================================================================== gate K (norms on the A15(a) images)
def gate_k(T, sel, timings):
    import cv2
    torch = T["torch"]
    import a15a_token_norms as A15                                         # rf_displays() of A15(a) (read only)
    import ab_objective as AB
    from src.data import build_mit1003
    sets = dict(kanizsa=A15.rf_displays(), mit1003=[p["image"] for p in build_mit1003(f"{AB.SR}/mit1003")[-200:][:24]])
    a15 = X.load_json(os.path.join(C.RE, "results", "explore", "A15a_token_norms.json"))["result"]
    assert [os.path.basename(p) for p in sets["kanizsa"]] == a15["images"]["kanizsa"]
    assert [os.path.basename(p) for p in sets["mit1003"]] == a15["images"]["mit1003"]
    out = {}
    per_image = {}
    for bb in ("clip", "siglip"):
        t0 = time.time()
        S = sel[bb]
        nd = {int(k): v for k, v in S["rec"]["result"]["neuron_dict"].items()}
        eval_blocks = S["rec"]["result"]["evaluated_flagged_blocks"]
        sw, model, _ = load_backbone_and_heads(T, bb, with_heads=False)
        blocks = list(range(1, C.N_LAYERS + 1))
        res = {}
        repro = 0.0
        for sname, paths in sets.items():
            rgbs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
            aois = None
            if sname == "kanizsa":
                aois = []
                for p in paths:
                    cfg = os.path.basename(p).rsplit("_", 1)[0]
                    a = X.v3_aoi(cfg)
                    aois.append((a["inside"], a["ind"]))
            for s in C.SIZES:
                for arm in C.ARMS:
                    nrm = {b: [] for b in blocks}
                    reg = {b: [] for b in blocks}
                    with torch.no_grad():
                        for i in range(0, len(rgbs), 6):
                            x = torch.stack([sw.prep(r, s) for r in rgbs[i:i + 6]])
                            with C.arm_hooks(model, bb, arm, nd if arm == "ttr" else None):
                                hs = C.hidden_states(model, x)
                            for b in blocks:
                                nrm[b].append(C.patch_norms(hs[b], bb, arm))
                                if arm != "plain":
                                    reg[b].append(C.register_norm(hs[b], arm))
                    for b in blocks:
                        n = np.concatenate(nrm[b])
                        rows = C.norm_rows(n)
                        g = int(round(n.shape[1] ** 0.5))
                        rec = dict(C.summarise_ratios(rows["ratio"], rows["share"]),
                                   count_median=float(np.median(rows["count"])))
                        if arm != "plain":
                            rg = np.concatenate(reg[b])
                            rec["register_over_max_patch_median"] = float(np.median(rg / rows["max"]))
                        if aois is not None and b in C.FLAGGED[bb]:
                            locs = [C.token_locations(rows["gt2"][i], g, *aois[i]) for i in range(len(paths))]
                            tot = sum(l["n"] for l in locs)
                            rec["loc_pooled"] = {k: (sum(l[k] * l["n"] for l in locs if l["n"]) / tot if tot else None)
                                                 for k in ("inside", "inducer", "elsewhere")}
                            rec["loc_n_tokens"] = int(tot)
                        res[f"{sname}|{s}|{arm}|{b}"] = rec
                        per_image[f"{bb}|{sname}|{s}|{arm}|{b}"] = dict(ratio=rows["ratio"].tolist(),
                                                                         share=rows["share"].tolist(),
                                                                         count=rows["count"].tolist())
                        if arm == "plain" and b in C.BB_CFG[bb]["layers"]:
                            ref = a15["results"][bb][str(b)][f"{s}|{sname}"]["per_image"]
                            repro = max(repro, float(np.max(np.abs(rows["ratio"] - np.array(ref["max_over_median"])))),
                                        float(np.max(np.abs(rows["share"] - np.array(ref["share_gt2xmedian"])))))
            C.log(f"gate K {bb} {sname} done ({time.time() - t0:.0f}s)")
        crit = {}
        for sname in sets:
            fails = sorted({b for b in eval_blocks for s in C.SIZES if res[f"{sname}|{s}|ttr|{b}"]["p90"] > C.P90_MAX})
            crit[sname] = dict(passed=not fails, blocks_failing=fails,
                               p90_ttr={f"{b}|{s}": res[f"{sname}|{s}|ttr|{b}"]["p90"]
                                        for b in C.FLAGGED[bb] for s in C.SIZES},
                               p90_plain={f"{b}|{s}": res[f"{sname}|{s}|plain|{b}"]["p90"]
                                          for b in C.FLAGGED[bb] for s in C.SIZES})
        labels = {str(b): ("not suppressed on the displays" if b in crit["kanizsa"]["blocks_failing"] else
                           ("flagged, before the emergence block" if b not in eval_blocks else "suppressed"))
                  for b in C.FLAGGED[bb]}
        out[bb] = dict(selection_sha256=S["sha"], neuron_dict_n=int(sum(len(v) for v in nd.values())),
                       evaluated_flagged_blocks=eval_blocks, summary=res, criterion=crit, labels=labels,
                       plain_reproduces_A15a_max_abs_diff=repro)
        timings[f"gateK_{bb}_s"] = time.time() - t0
        del model
    C.dump_scores(per_image, os.path.join(C.SCORES_TTR, "norms", "gateK_per_image.json"))
    return out


# ====================================================================== main
def load_selection(bb):
    p = os.path.join(C.RES_TTR, f"TTR_select_{bb}.json")
    rec = json.load(open(p))
    assert rec["result"]["synthetic_check"]["passed"], "selection's synthetic check failed: scoring refused"
    return dict(path=p, sha=C.sha256_file(p), rec=rec)


def main():
    C.require_public()                                                    # rule Section 8.1
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("plain", "norms", "arms"))
    ap.add_argument("bb", nargs="?", choices=("clip", "siglip"))
    ap.add_argument("--arms", default="ttr,reg0")
    ap.add_argument("--sets", default="stim_v3,mit,stim_alt")
    a = ap.parse_args()
    t_start = time.time()
    T = env()
    timings = {}
    script = os.path.abspath(__file__)
    extra = [os.path.join(C.EXPLORE, "mapsrc.py"), os.path.join(C.EXPLORE, "a15a_token_norms.py")]

    if a.mode == "norms":
        sel = {bb: load_selection(bb) for bb in ("clip", "siglip")}
        res = gate_k(T, sel, timings)
        timings["total_s"] = time.time() - t_start
        C.write_json_ttr("TTR_gateK.json", dict(gate_K=res, timings=timings,
                                                selection={bb: dict(path=os.path.relpath(s["path"], C.RE), sha256=s["sha"])
                                                           for bb, s in sel.items()}),
                         script, inputs=["A15(a) 24 Kanizsa displays", "MIT1003 held-out first 24",
                                         "results/explore/ttr/TTR_select_{clip,siglip}.json"], extra_code=extra)
        return

    bb = a.bb
    assert bb, "backbone required"
    sw, model, heads = load_backbone_and_heads(T, bb)
    sys.path.insert(0, os.path.join(C.ROOT, "v3", "scores"))

    if a.mode == "plain":
        gf = gate_f(T, sw, model, bb)
        C.log(f"gate F: {'PASS' if gf['passed'] else 'FAIL'} {gf['max_abs_diff']}")
        store = Store()
        if gf["passed"]:
            for st in ("stim_v3", "stim_alt"):
                run_displays(T, sw, model, bb, "plain", None, heads, st, store, timings)
                write_store(store, "plain", bb)                           # saved after every stage
            run_mit(T, sw, model, bb, "plain", None, heads, store, timings)
            write_store(store, "plain", bb)
            gp = gate_p(store, bb)
        else:
            gp = dict(passed=False, note="not run: gate F failed")
        timings["total_s"] = time.time() - t_start
        C.log(f"gate P: {'PASS' if gp['passed'] else 'FAIL'}; display values {gp.get('n_display_values_compared')} "
              f"max |diff| {gp.get('max_abs_diff_display')}; MIT per-image max |diff| {gp.get('max_abs_diff_mit_per_image')}"
              f"; failed {gp.get('failed')}")
        C.write_json_ttr(f"TTR_score_{bb}_plain.json",
                         dict(backbone=bb, gate_F=gf, gate_P=gp, exact_formula_check=store.exact, timings=timings,
                              units=dict(sweep=list(heads["sweep"]), swap=sorted({u for u, _ in heads["swap"]}))),
                         script, inputs=["v2/stim_v3 (480 Kanizsa displays)", "v3/stim_alt (360)", "MIT1003 held-out 200",
                                         "sweep/<bb>/readout_L<L>.pt", "v3/gpu/full/ckpt (SigLIP-L 24 heads)",
                                         "published per-image scores (gate P)"], extra_code=extra)
        if not (gf["passed"] and gp["passed"]):
            C.log("STOP: gate F or gate P failed")
            sys.exit(3)
        return

    # ---- arms
    plain_rec = json.load(open(os.path.join(C.RES_TTR, f"TTR_score_{bb}_plain.json")))
    assert plain_rec["result"]["gate_P"]["passed"] and plain_rec["result"]["gate_F"]["passed"], "gate F/P not passed"
    gk = json.load(open(os.path.join(C.RES_TTR, "TTR_gateK.json")))
    S = load_selection(bb)
    assert gk["result"]["selection"][bb]["sha256"] == S["sha"], "TTR_select changed after gate K: scoring refused"
    nd = {int(k): v for k, v in S["rec"]["result"]["neuron_dict"].items()}
    arms = a.arms.split(","); sets = a.sets.split(",")
    exact = {}
    for arm in arms:
        assert arm in ("ttr", "reg0"), arm
        store = Store()
        for st in sets:
            if arm == "reg0" and st == "stim_alt":
                continue                                                  # Tier A2: reg0 on stim_v3 and MIT1003 only
            if st == "mit":
                run_mit(T, sw, model, bb, arm, nd, heads, store, timings)
            else:
                run_displays(T, sw, model, bb, arm, nd, heads, st, store, timings)
            write_store(store, arm, bb, S["sha"])                         # saved after every stage
        exact[arm] = store.exact
    timings["total_s"] = time.time() - t_start
    C.write_json_ttr(f"TTR_score_{bb}_arms_{'_'.join(arms)}.json",
                     dict(backbone=bb, arms=arms, sets=sets, selection=dict(path=os.path.relpath(S["path"], C.RE),
                                                                           sha256=S["sha"]),
                          neuron_dict_n=int(sum(len(v) for v in nd.values())), exact_formula_check=exact,
                          timings=timings),
                     script, inputs=["as plain", "results/explore/ttr/TTR_select_<bb>.json",
                                     "results/explore/ttr/TTR_gateK.json"], extra_code=extra)


if __name__ == "__main__":
    main()
