"""
neuro/reanalysis/explore/ttr/ttr_select.py <clip|siglip>: register-neuron selection for test-time registers
(Jiang et al. 2025, Algorithm 1) on CLIP-L or SigLIP-L, on 100 MIT1003 TRAINING images; no Kanizsa display is read.
EXPLORATORY (REANALYSIS_RULE.md Section 6, item 12; DESIGN.md Section 4.2). Written 18 November 2026 after P1 to P5,
R8 and every exploratory output had been seen, before any TTR norm or score existed.

Fixed choices (DESIGN.md Section 4.2; flag F2: ours, not the authors'):
  data      build_mit1003(<SR>/mit1003)[:-200] (803 training images), RandomState(2026).choice(803, 100, no
            replacement), sorted; disjoint from the 200 held-out images (asserted); a read guard refuses any display
            folder. Each image at 336 and 448 px with the readouts' preprocessing (sweep.prep, verbatim): 200 items.
  step 1    plain pass: per block 1..24 and item, patch-token L2 norms (CLS excluded); ratio max/median, share above
            twice the median; per block and size the median and 90th percentile over the 100 images.
  step 2    emergence block e = first block whose median ratio exceeds 4 at BOTH sizes; highest layer h = e - 1
            (0-based layer whose output is block e). Flagged blocks below e are reported as "flagged, before the
            emergence block" and left out of the suppression criterion.
  step 3    detection block d = block with the largest median ratio over the 200 items pooled; tau_s (per size) = mean
            + 3 SD of all patch-token norms at d over the 100 images; outliers of an item = its patch tokens with norm
            > tau_s (CLS excluded); items without outliers are skipped and counted. Diagnostic: Jaccard overlap of each
            item's outlier set at d with its tokens above twice the median at every flagged block.
  step 4    second plain pass with fc2 forward pre-hooks on all 24 layers (the post-nonlinearity activations): per
            item with outliers, mean |activation| over the outlier tokens, times the sparsity mask (act < 0.5 on at
            least 50% of the item's tokens); summed, then averaged over the items with outliers. Ranked for layers <= h
            (primary branch) and <= 23 (fallback branch); top 300 stored.
  step 5    k grid (5, 10, 20, 30, 45, 60, 80, 100, 150, 200, 300), ascending: for each k the ttr arm (top-k neurons)
            on the 200 items; criterion: at every flagged block b >= e and both sizes, the 90th percentile over the 100
            images of max/median (patches only) <= 2.0; the smallest passing k is taken (the sweep stops there). If no k
            passes, the fallback branch (h = 23) is swept the same way. If neither passes: the (branch, k) passing the
            most (block, size) pairs, ties to the primary branch, then the smaller k; failing blocks carry the label
            "high-norm tokens not suppressed". The register norm at d against the largest patch norm is reported.
  step 6    synthetic check (implementation only): uniform 1024 x 768 canvases at grey 255, 128, 0 and one seeded
            smooth-noise image (no inducer); plain and ttr at the chosen setting; condition: on every canvas and size,
            ttr lowers max/median at every flagged block b >= e and the register's norm at d exceeds the largest patch
            norm. If it fails the script exits with code 2 and scoring must not start.
Output: results/explore/ttr/TTR_select_<bb>.json (exploratory; provenance as excommon.write_json).
CPU only: torch threads 4, device cpu explicitly, excommon.metal_guard() after the model load; require_public() first.
Usage: nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       $PY -B -I explore/ttr/ttr_select.py clip
"""
import os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("OMP_NUM_THREADS", "4")
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import ttr_common as C                                                    # noqa: E402

X = C.X
BATCH = 6


def load_model(bb):
    import torch
    from transformers import CLIPVisionModel, SiglipVisionModel
    m = (CLIPVisionModel if bb == "clip" else SiglipVisionModel).from_pretrained(C.BB_CFG[bb]["hf"]).to("cpu").eval()
    assert next(m.parameters()).device.type == "cpu"
    X.metal_guard()
    return m


def make_prep(bb):
    import cv2, torch
    MEAN = np.array(C.BB_CFG[bb]["mean"], np.float32); STD = np.array(C.BB_CFG[bb]["std"], np.float32)

    def prep(rgb, s):                                                     # verbatim from sweep.prep
        x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))
    return prep


def read_rgb_guarded(p):
    import cv2
    C.assert_not_display(p)
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def norm_pass(model, bb, rgbs, s, arm, nd=None, blocks=range(1, C.N_LAYERS + 1), want_reg=None):
    """Patch norms {block: (n, T) float64} (and register norms at want_reg) for a list of RGB images at size s."""
    import torch
    prep = make_prep(bb)
    out = {b: [] for b in blocks}
    reg = []
    with torch.no_grad(), C.arm_hooks(model, bb, arm, nd):
        for i in range(0, len(rgbs), BATCH):
            x = torch.stack([prep(r, s) for r in rgbs[i:i + BATCH]])
            hs = C.hidden_states(model, x)
            for b in blocks:
                out[b].append(C.patch_norms(hs[b], bb, arm))
            if want_reg is not None and arm != "plain":
                reg.append(C.register_norm(hs[want_reg], arm))
    out = {b: np.concatenate(v) for b, v in out.items()}
    return out, (np.concatenate(reg) if reg else None)


def score_pass(model, bb, rgbs, s, outliers):
    """Second plain pass: per item with outliers, masked mean |act| at the outlier tokens, summed per (layer, neuron)."""
    import torch
    _, layers = C.model_parts(model)
    off = 1 if C.BB_CFG[bb]["cls"] else 0
    ssum = np.zeros((C.N_LAYERS, C.MLP_WIDTH))
    msum = np.zeros((C.N_LAYERS, C.MLP_WIDTH))
    state = {}
    handles = []
    for l, lay in enumerate(layers):
        def hook(mod, args, l=l):
            a = args[0].detach().contiguous().numpy()                     # float32 (B, T, 4096)
            for j, o in enumerate(state["outl"]):
                if o is None or len(o) == 0:
                    continue
                sc, mk = C.item_scores(a[j], np.asarray(o) + off)
                ssum[l] += sc
                msum[l] += mk
        handles.append(lay.mlp.fc2.register_forward_pre_hook(hook))
    prep = make_prep(bb)
    try:
        with torch.no_grad():
            for i in range(0, len(rgbs), BATCH):
                state["outl"] = outliers[i:i + BATCH]
                if all(o is None or len(o) == 0 for o in state["outl"]):
                    continue
                x = torch.stack([prep(r, s) for r in rgbs[i:i + BATCH]])
                C.hidden_states(model, x)
    finally:
        for h in handles:
            h.remove()
    return ssum, msum


def synthetic_canvases():
    """Uniform 1024 x 768 canvases at grey 255, 128 and 0, and one seeded smooth-noise image (RandomState(2026) normal
    noise on a 12 x 16 grid, bicubic-upsampled, scaled to 0..255). No inducer is drawn."""
    import cv2
    out = {}
    for gv in (255, 128, 0):
        out[f"uniform_{gv}"] = np.full((C.H, C.W, 3), gv, np.uint8)
    z = np.random.RandomState(2026).randn(12, 16, 3).astype(np.float32)
    zz = cv2.resize(z, (C.W, C.H), interpolation=cv2.INTER_CUBIC)
    zz = (zz - zz.min()) / (zz.max() - zz.min())
    out["smooth_noise"] = (zz * 255).round().astype(np.uint8)
    return out


def main():
    C.require_public()                                                    # rule Section 8.1
    bb = sys.argv[1]
    assert bb in ("clip", "siglip"), bb
    t_start = time.time()
    import torch, cv2
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cv2.setNumThreads(1)
    stub = os.path.join(C.RE, "_stub")
    sys.path.insert(0, stub)
    import ab_objective as AB                                             # the stub: DEV = "cpu", SR from the environment
    assert AB.__file__.startswith(stub) and AB.DEV == "cpu", AB.__file__
    from src.data import build_mit1003
    sel, held = C.selection_images(build_mit1003(f"{AB.SR}/mit1003"))
    C.log(f"{bb}: {len(sel)} selection images (MIT1003 training split), disjoint from {len(held)} held-out")
    rgbs = [read_rgb_guarded(p) for p in sel]
    model = load_model(bb)
    flagged = C.FLAGGED[bb]
    blocks = list(range(1, C.N_LAYERS + 1))
    timings = {}

    # ---- step 1: plain norm profile
    t0 = time.time()
    plain = {s: norm_pass(model, bb, rgbs, s, "plain")[0] for s in C.SIZES}
    timings["step1_plain_profile_s"] = time.time() - t0
    rows = {s: {b: C.norm_rows(plain[s][b]) for b in blocks} for s in C.SIZES}
    profile = {f"{s}": {str(b): dict(C.summarise_ratios(rows[s][b]["ratio"], rows[s][b]["share"]),
                                     count_median=float(np.median(rows[s][b]["count"])),
                                     count_range=[int(rows[s][b]["count"].min()), int(rows[s][b]["count"].max())])
                        for b in blocks} for s in C.SIZES}
    med_by_size = {s: {b: float(np.median(rows[s][b]["ratio"])) for b in blocks} for s in C.SIZES}
    pooled = {b: float(np.median(np.concatenate([rows[s][b]["ratio"] for s in C.SIZES]))) for b in blocks}
    C.log(f"step 1 done ({timings['step1_plain_profile_s']:.0f}s); median ratio by block (336/448): " +
          " ".join(f"{b}:{med_by_size[336][b]:.2f}/{med_by_size[448][b]:.2f}" for b in blocks))

    # ---- step 2: emergence block and highest layer
    e = C.emergence_block(med_by_size)
    if e is None:
        e_note = "no block exceeds the cutoff at both sizes; e set to 1 (h = 0) and every flagged block is evaluated"
        e = 1
    else:
        e_note = None
    h = e - 1
    eval_blocks = [b for b in flagged if b >= e]
    before_e = [b for b in flagged if b < e]
    C.log(f"step 2: emergence block e = {e}, highest layer h = {h}; evaluated flagged blocks {eval_blocks}; "
          f"flagged before e: {before_e}")

    # ---- step 3: detection block, tau, outliers
    d = C.detection_block(pooled)
    tau = {}
    outliers = {}
    n_skipped = {}
    jac = {}
    for s in C.SIZES:
        n = plain[s][d]
        tau[s] = float(n.mean() + C.TAU_SD * n.std())
        outliers[s] = [np.flatnonzero(n[i] > tau[s]) for i in range(n.shape[0])]
        n_skipped[s] = int(sum(len(o) == 0 for o in outliers[s]))
        jac[str(s)] = {}
        for b in flagged:
            js = [C.jaccard(outliers[s][i].tolist(), np.flatnonzero(rows[s][b]["gt2"][i]).tolist())
                  for i in range(n.shape[0]) if len(outliers[s][i])]
            jac[str(s)][str(b)] = dict(mean=float(np.mean(js)) if js else None,
                                       median=float(np.median(js)) if js else None, n=len(js))
    n_out = {str(s): dict(median=float(np.median([len(o) for o in outliers[s]])),
                          range=[int(min(len(o) for o in outliers[s])), int(max(len(o) for o in outliers[s]))])
             for s in C.SIZES}
    C.log(f"step 3: detection block d = {d}; tau {tau}; items without outliers {n_skipped}; outliers per item {n_out}")
    constancy = {str(b): {str(s): C.position_constancy(rows[s][b]["gt2"]) for s in C.SIZES} for b in flagged}

    # ---- step 4: neuron scores
    t0 = time.time()
    ssum = np.zeros((C.N_LAYERS, C.MLP_WIDTH)); msum = np.zeros((C.N_LAYERS, C.MLP_WIDTH))
    for s in C.SIZES:
        a, m = score_pass(model, bb, rgbs, s, outliers[s])
        ssum += a; msum += m
    n_items = sum(len(o) > 0 for s in C.SIZES for o in outliers[s])
    timings["step4_score_pass_s"] = time.time() - t0
    rank = {"primary": C.rank_neurons(ssum, n_items, h), "fallback": C.rank_neurons(ssum, n_items, C.N_LAYERS - 1)}
    C.log(f"step 4 done ({timings['step4_score_pass_s']:.0f}s); {n_items} items with outliers; top 5 primary "
          f"{rank['primary'][:5]}")

    # ---- step 5: k sweep
    grid, grid_full, reg_vs_patch = {}, {}, {}
    t0 = time.time()
    chosen = None
    for br in ("primary", "fallback"):
        if br == "fallback" and h == C.N_LAYERS - 1:
            break
        grid[br], grid_full[br], reg_vs_patch[br] = {}, {}, {}
        for k in C.K_GRID:
            nd = C.neuron_dict_top(rank[br], k)
            tk = time.time()
            tab, full, rvp = {}, {}, {}
            for s in C.SIZES:
                nrm, rg = norm_pass(model, bb, rgbs, s, "ttr", nd, blocks, want_reg=d)
                for b in blocks:
                    rr = C.norm_rows(nrm[b])
                    full[f"{b}|{s}"] = dict(p90=float(np.percentile(rr["ratio"], 90)),
                                            median=float(np.median(rr["ratio"])), share_mean=float(rr["share"].mean()))
                    if b in flagged:
                        tab[(b, s)] = full[f"{b}|{s}"]["p90"]
                rvp[str(s)] = dict(register_over_max_patch_median=float(np.median(rg / nrm[d].max(1))),
                                   share_register_above_max_patch=float(np.mean(rg > nrm[d].max(1))))
            grid[br][k], grid_full[br][k], reg_vs_patch[br][k] = tab, full, rvp
            ok, npass = C.passes(tab, eval_blocks)
            C.log(f"step 5 {br} k={k} ({len(nd)} layers {sorted(nd)}): {npass}/{2 * len(eval_blocks)} pass; "
                  + " ".join(f"{b}/{s}:{tab[(b, s)]:.2f}" for b in flagged for s in C.SIZES)
                  + f" ({time.time() - tk:.0f}s)")
            if ok:
                chosen = dict(branch=br, k=k)
                break
        if chosen:
            break
    timings["step5_k_sweep_s"] = time.time() - t0
    setting = C.choose_setting(grid, eval_blocks)
    if chosen:
        assert setting["branch"] == chosen["branch"] and setting["k"] == chosen["k"], (setting, chosen)
    nd = C.neuron_dict_top(rank[setting["branch"]], setting["k"])
    final_tab = grid[setting["branch"]][setting["k"]]
    not_suppressed = sorted({b for (b, s), v in final_tab.items() if b in eval_blocks and v > C.P90_MAX})
    C.log(f"chosen: {setting}; neurons per layer {{{', '.join(f'{l}: {len(v)}' for l, v in nd.items())}}}; "
          f"not suppressed: {not_suppressed}")

    # ---- step 6: synthetic check
    t0 = time.time()
    synth = {}
    ok_all = True
    for nm, img in synthetic_canvases().items():
        synth[nm] = {}
        for s in C.SIZES:
            pn, _ = norm_pass(model, bb, [img], s, "plain", None, blocks)
            tn, rg = norm_pass(model, bb, [img], s, "ttr", nd, blocks, want_reg=d)
            rec = {}
            lowers = True
            for b in flagged:
                rp, rt = float(C.norm_rows(pn[b])["ratio"][0]), float(C.norm_rows(tn[b])["ratio"][0])
                rec[str(b)] = dict(plain=rp, ttr=rt, lowered=bool(rt < rp))
                if b in eval_blocks and not rt < rp:
                    lowers = False
            reg_ok = bool(rg[0] > tn[d].max())
            synth[nm][str(s)] = dict(blocks=rec, register_norm_at_d=float(rg[0]), max_patch_norm_at_d=float(tn[d].max()),
                                     register_exceeds_max_patch=reg_ok, ttr_lowers_every_evaluated_block=lowers)
            ok_all = ok_all and lowers and reg_ok
    timings["step6_synthetic_s"] = time.time() - t0
    timings["total_s"] = time.time() - t_start
    C.log(f"step 6 synthetic check: {'PASS' if ok_all else 'FAIL'}")

    res = dict(backbone=bb, hf=C.BB_CFG[bb]["hf"], flagged_blocks=list(flagged),
               selection_images=[os.path.basename(p) for p in sel], n_items=2 * len(sel),
               profile_plain=profile, median_ratio_by_size={str(s): {str(b): v for b, v in med_by_size[s].items()}
                                                            for s in C.SIZES},
               median_ratio_pooled={str(b): v for b, v in pooled.items()},
               emergence_block=e, emergence_note=e_note, highest_layer=h, evaluated_flagged_blocks=eval_blocks,
               flagged_before_emergence=before_e, detection_block=d, tau={str(s): v for s, v in tau.items()},
               items_without_outliers={str(s): v for s, v in n_skipped.items()}, outliers_per_item=n_out,
               n_items_with_outliers=n_items, jaccard_outliers_vs_gt2=jac, position_constancy=constancy,
               ranking={br: [list(r) for r in v] for br, v in rank.items()},
               grid={br: {str(k): v for k, v in g.items()} for br, g in grid_full.items()},
               register_vs_max_patch_at_d={br: {str(k): v for k, v in g.items()} for br, g in reg_vs_patch.items()},
               chosen=setting, neuron_dict={str(l): v for l, v in nd.items()},
               n_neurons=int(sum(len(v) for v in nd.values())),
               not_suppressed_on_selection_images=not_suppressed, synthetic_check=dict(passed=ok_all, canvases=synth),
               timings=timings,
               settings=dict(emergence_cutoff=C.EMERGENCE_CUTOFF, tau_sd=C.TAU_SD, sparse_thr=C.SPARSE_THR,
                             sparse_frac=C.SPARSE_FRAC, k_grid=list(C.K_GRID), p90_max=C.P90_MAX,
                             selection_seed=C.SEL_SEED, n_selection_images=C.SEL_N),
               versions=dict(torch=torch.__version__, transformers=__import__("transformers").__version__,
                             numpy=np.__version__))
    C.write_json_ttr(f"TTR_select_{bb}.json", res, os.path.abspath(__file__),
                     inputs=["MIT1003 training split (100 images, RandomState(2026))", f"HF cache: {C.BB_CFG[bb]['hf']}"])
    if not ok_all:
        C.log("synthetic check FAILED: scoring must not start (implementation fault presumed)")
        sys.exit(2)


if __name__ == "__main__":
    main()
