"""
neuro/reanalysis/explore/ttr/ttr_common.py: shared code of the test-time-registers analysis (REANALYSIS_RULE.md
Section 6, item 12; design in explore/ttr/DESIGN.md, written 18 November 2026 about 20:10 UTC). EXPLORATORY: unadjusted,
labelled, never decision-bearing; it cannot change P3, P4, P5 or the title.

Written 18 November 2026 (evening) after S1, S2 (P1 to P4), R1 to R8, P5 and every exploratory output had been seen,
before any TTR token norm, map or score existed. Choices are those of DESIGN.md (copied below where they live in code):

- Method (Jiang et al. 2025, Algorithm 2 and the released Hugging Face models, NOT the repository hook's single scalar
  sign_max; flag F3): one zero-initialised register token appended at the END of the sequence, after the positional
  embeddings (forward hook on `embeddings`), before CLIP's `pre_layrnorm` (so it enters layer 1 as that LayerNorm's
  bias; SigLIP has no pre-LayerNorm and it enters as zeros); no positional embedding, no attention mask. In every layer
  l listed in the neuron dict, a forward pre-hook on `encoder.layers[l].mlp.fc2` (its input is the post-nonlinearity
  activation: quick_gelu for CLIP-L, gelu_pytorch_tanh for SigLIP-L) works on a clone of the activations A (B, T, 4096):
  for each register neuron r, A[:, reg, r] = signed max over ALL tokens (CLS, patches, register) of the UNMODIFIED
  A[:, t, r], per image and per neuron; A[:, patch, r] = 0; CLIP's CLS keeps its value (index 0); other neurons
  untouched; scale 1. A shift without a register is refused (it would write into a patch).
- Arms: plain (no hooks), reg0 (register appended, no shift; control), ttr (register + shift).
- Readout features: plain = the stock path (CLIP hidden_states[L][:, 1:], SigLIP hidden_states[L]); reg0 / ttr =
  CLIP hidden_states[L][:, 1:-1], SigLIP hidden_states[L][:, :-1]; reshaped to g x g with the g * g == T assertion
  (sweep.feats' transpose / reshape / contiguous). Sweep heads take float32 features, swap heads .half().float()
  (eval_full holds features in float16 between backbone and head).
- Token-norm statistics: patch tokens only (CLS and register excluded), L2 norm per token, per image: ratio max /
  median, share and count of tokens above twice the median, their grid positions (A15(a) definitions).
- Read guard for neuron selection: refuses any path under v2/stim_v3, v3/stim_alt, neuro/reanalysis/stim_rigid,
  stim_v2, stim_ehr (no Kanizsa display is ever used to select neurons).
- Selection images: MIT1003 training split build_mit1003(...)[:-200] (803 images), numpy RandomState(2026).choice(803,
  100, replace=False), sorted; disjoint from the 200 held-out images (asserted).
- Fast r_inside: exactly common_alt.measures' r_inside (map q16 -> float32 -> float64 -> clip at 0 -> sums over the
  inside and inducer AOIs), computed only on the AOI pixels through flat indices (same elements in the same order,
  so the same pairwise sums; checked against common_alt.measures on the first displays of every real run).

CPU only: nothing here touches MPS or Metal; the scripts call excommon.metal_guard() after every model load.
No torch import at module level (the pure-numpy parts are unit-tested without it).
"""
import hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                       # explore/ttr
EXPLORE = os.path.dirname(HERE)
if EXPLORE not in sys.path:
    sys.path.insert(0, EXPLORE)
import excommon as X                                                      # noqa: E402  (no torch)

RE = X.RE                                                                 # neuro/reanalysis
NEURO = os.path.dirname(RE)
ROOT = X.ROOT
RES_TTR = os.path.join(RE, "results", "explore", "ttr")                   # selection, gate and timing records
SCORES_TTR = os.path.join(RE, "scores_ttr")                               # per-display scores of the three arms
W, H = 1024, 768
SIZES = (336, 448)
ARMS = ("plain", "reg0", "ttr")

# verbatim from sweep.py CFG (clip, siglip)
BB_CFG = {
    "clip": dict(hf="openai/clip-vit-large-patch14", layers=(4, 8, 12, 16, 20, 24), cls=True,
                 mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    "siglip": dict(hf="google/siglip-large-patch16-256", layers=(4, 8, 12, 16, 20, 24), cls=False,
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
N_LAYERS = 24
MLP_WIDTH = 4096
# A15(a) result (results/explore/A15a_token_norms.json, result.critic_fix_18), recomputed in DESIGN.md Section 1
FLAGGED = {"clip": (12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
SWEEP_UNITS = [f"{bb}_L{L}" for bb in ("clip", "siglip") for L in FLAGGED[bb]]
SWAP_ARCHS = ("R1n", "R1", "R2", "R3", "R4")
SWAP_UNITS = [f"siglip_L24|{a}" for a in SWAP_ARCHS]
SWAP_SEEDS = (0, 1, 2)

# selection constants (DESIGN.md Section 4.2; flag F2: ours, not the authors')
SEL_SEED = 2026
SEL_N = 100
EMERGENCE_CUTOFF = 4.0          # first block whose median max/median exceeds 4 at both sizes
TAU_SD = 3.0                    # outlier threshold: mean + 3 SD of all patch norms at the detection block (per size)
SPARSE_THR, SPARSE_FRAC = 0.5, 0.5   # sparsity filter (notebook): act < 0.5 on at least 50% of the item's tokens
K_GRID = (5, 10, 20, 30, 45, 60, 80, 100, 150, 200, 300)
P90_MAX = 2.0                   # suppression criterion: 90th percentile of max/median <= 2.0
TOP_STORE = 300

DISPLAY_DIRS = [os.path.join(ROOT, "v2", "stim_v3"), os.path.join(ROOT, "v3", "stim_alt"),
                os.path.join(RE, "stim_rigid"), os.path.join(ROOT, "stim_v2"), os.path.join(ROOT, "stim_ehr")]


# ====================================================================== guards and provenance
def require_public():
    """Rule Section 8.1 (analyze_re.require_public through excommon)."""
    return X.require_public()


def assert_not_display(path):
    """Read guard of the neuron selection: refuse any Kanizsa / Ehrenstein display folder."""
    rp = os.path.realpath(path)
    for d in DISPLAY_DIRS:
        dd = os.path.realpath(d)
        if rp == dd or rp.startswith(dd + os.sep):
            raise RuntimeError(f"read guard: refusing a display path during neuron selection: {path}")
    return path


def selection_indices(n_train, n=SEL_N, seed=SEL_SEED):
    """Sorted indices of the selection images within the MIT1003 training split (RandomState(2026), no replacement)."""
    return sorted(int(i) for i in np.random.RandomState(seed).choice(n_train, n, replace=False))


def selection_images(mit_all):
    """(selection paths, held-out paths) from build_mit1003's full list; asserts the split sizes and disjointness."""
    assert len(mit_all) == 1003, len(mit_all)
    train, held = mit_all[:-200], mit_all[-200:]
    idx = selection_indices(len(train))
    sel = [train[i]["image"] for i in idx]
    hs = {p["image"] for p in held}
    assert not (set(sel) & hs), "selection images overlap the held-out 200"
    assert len(set(sel)) == SEL_N
    for p in sel:
        assert_not_display(p)
    return sel, [p["image"] for p in held]


def sha256_file(p):
    return X.sha256_file(p)


def write_json_ttr(name, result, script, inputs=(), notes=None, extra_code=()):
    """results/explore/ttr/<name> with excommon's provenance (script, helper hashes, rule hash, public commit)."""
    os.makedirs(RES_TTR, exist_ok=True)
    return X.write_json(os.path.join("ttr", name), result, script, inputs=inputs, notes=notes,
                        extra_code=[os.path.abspath(__file__)] + list(extra_code))


def dump_scores(obj, path):
    """Score files: write to a temporary name, then rename (as eval_full.dump_atomic)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(X._clean(obj), f)
    os.replace(path + ".tmp", path)


def log(msg):
    X.log(msg)


# ====================================================================== hooks (torch imported lazily)
def model_parts(model):
    """(embeddings module, list of encoder layers) of a CLIPVisionModel / SiglipVisionModel, whichever nesting the
    installed transformers uses (5.12: model.embeddings, model.encoder.layers; older: model.vision_model.*)."""
    root = model.vision_model if hasattr(model, "vision_model") else model
    emb, enc = getattr(root, "embeddings", None), getattr(root, "encoder", None)
    assert emb is not None and enc is not None and hasattr(enc, "layers"), type(model)
    return emb, list(enc.layers)


class TTR:
    """Install (and remove on exit) the register and shift hooks on an already loaded vision model.
    neuron_dict: {layer (0-based int): [neuron indices]}; n_reg: 0 or 1; has_cls: True for CLIP.
    with n_reg = 0 and an empty dict nothing is installed (the plain arm)."""

    def __init__(self, model, has_cls, neuron_dict=None, n_reg=1):
        nd = {int(k): sorted(int(n) for n in v) for k, v in (neuron_dict or {}).items() if len(v)}
        assert n_reg in (0, 1), n_reg
        assert not (nd and n_reg == 0), "a shift without a register is not allowed (it would write into a patch)"
        self.model, self.has_cls, self.nd, self.n_reg = model, bool(has_cls), nd, n_reg
        self.handles = []

    def __enter__(self):
        import torch
        emb, layers = model_parts(self.model)
        if self.n_reg:
            n_reg = self.n_reg

            def add_register(mod, args, out):
                return torch.cat([out, out.new_zeros(out.shape[0], n_reg, out.shape[2])], 1)
            self.handles.append(emb.register_forward_hook(add_register))
        for l, neurons in self.nd.items():
            assert 0 <= l < len(layers), l
            idx = torch.tensor(neurons, dtype=torch.long)
            has_cls = self.has_cls

            def shift(mod, args, idx=idx, has_cls=has_cls):
                x = args[0]
                sub = x.index_select(2, idx)                       # (B, T, k) unmodified activations
                m = sub.amax(dim=1)                                # per image, per neuron, over all tokens (signed)
                y = x.clone()
                y[:, :, idx] = 0
                if has_cls:
                    y[:, 0, idx] = sub[:, 0, :]
                y[:, -1, idx] = m
                return (y,) + tuple(args[1:])
            self.handles.append(layers[l].mlp.fc2.register_forward_pre_hook(shift))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def arm_hooks(model, kind, arm, neuron_dict=None):
    """TTR context for an arm name."""
    has_cls = BB_CFG[kind]["cls"]
    if arm == "plain":
        return TTR(model, has_cls, None, 0)
    if arm == "reg0":
        return TTR(model, has_cls, None, 1)
    if arm == "ttr":
        assert neuron_dict, "ttr arm needs a neuron dict"
        return TTR(model, has_cls, neuron_dict, 1)
    raise ValueError(arm)


def hidden_states(model, x):
    """All hidden states of the vision model (block L = hidden_states[L], rule Section 2.5)."""
    return model(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states


def patch_slice(kind, arm, T):
    """(start, stop) of the patch tokens in a sequence of length T: CLS (CLIP) and register (reg0, ttr) excluded."""
    a = 1 if BB_CFG[kind]["cls"] else 0
    b = T - (0 if arm == "plain" else 1)
    return a, b


def grid_feats(t, kind, arm):
    """(B, T, D) hidden state -> (B, D, g, g) readout features, as sweep.feats (transpose, reshape, contiguous)."""
    a, b = patch_slice(kind, arm, t.shape[1])
    if arm == "plain":
        tt = t[:, 1:, :] if BB_CFG[kind]["cls"] else t                  # the stock expression (bit identity)
    else:
        tt = t[:, a:b, :]
    g = int(round(tt.shape[1] ** 0.5))
    assert g * g == tt.shape[1], (tt.shape, kind, arm)
    return tt.transpose(1, 2).reshape(tt.shape[0], -1, g, g).contiguous()


def patch_norms(t, kind, arm):
    """(B, T, D) hidden state -> (B, g*g) float64 patch-token L2 norms (CLS and register excluded)."""
    a, b = patch_slice(kind, arm, t.shape[1])
    n = t[:, a:b, :].norm(dim=-1).double().numpy()
    g = int(round(n.shape[1] ** 0.5))
    assert g * g == n.shape[1], n.shape
    return n


def register_norm(t, arm):
    """L2 norm of the register token (last), or None for the plain arm."""
    if arm == "plain":
        return None
    return t[:, -1, :].norm(dim=-1).double().numpy()


# ====================================================================== token-norm statistics (numpy)
def norm_rows(n):
    """n: (images, tokens) norms -> per image: ratio max/median, share and count of tokens > 2 x median, max, median,
    and the boolean mask of tokens > 2 x median (A15(a) definitions)."""
    n = np.asarray(n, np.float64)
    med = np.median(n, 1)
    gt2 = n > 2 * med[:, None]
    return dict(ratio=n.max(1) / med, share=gt2.mean(1), count=gt2.sum(1).astype(int), max=n.max(1), median=med,
                gt2=gt2)


def summarise_ratios(ratios, shares=None):
    r = np.asarray(ratios, float)
    out = dict(n=int(r.size), median=float(np.median(r)), p90=float(np.percentile(r, 90)), max=float(r.max()),
               min=float(r.min()), n_with_token_gt2=None)
    if shares is not None:
        s = np.asarray(shares, float)
        out.update(share_mean=float(s.mean()), share_median=float(np.median(s)), n_with_token_gt2=int((s > 0).sum()))
    return out


def position_constancy(gt2_masks):
    """Share of images whose set of positions above twice the median equals the most common set (F4 diagnostic)."""
    keys = [tuple(np.flatnonzero(m).tolist()) for m in gt2_masks]
    if not keys:
        return dict(share_modal=None, modal_positions=None, n=0)
    vals, counts = np.unique(np.array([",".join(map(str, k)) for k in keys]), return_counts=True)
    j = int(np.argmax(counts))
    return dict(share_modal=float(counts[j] / len(keys)), modal_positions=vals[j], n=len(keys),
                n_distinct_sets=int(len(vals)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


# ====================================================================== neuron selection (numpy, testable)
def item_scores(acts, outlier_tok, thr=SPARSE_THR, frac=SPARSE_FRAC, sparse=True):
    """One item, one layer. acts: (T, N) post-nonlinearity activations (all tokens); outlier_tok: token indices of the
    outliers. Returns (score (N,) = mean |act| at the outliers, times the sparsity mask; mask (N,) bool).
    Sparsity filter (notebook): a neuron counts only if act < thr on at least frac of the item's tokens."""
    a = np.asarray(acts)
    o = np.asarray(outlier_tok, int)
    s = np.abs(a[o].astype(np.float64)).mean(0) if o.size else np.zeros(a.shape[1])
    mask = (a < thr).mean(0) >= frac if sparse else np.ones(a.shape[1], bool)
    return s * mask, mask


def emergence_block(median_by_size, cutoff=EMERGENCE_CUTOFF):
    """median_by_size: {size: {block: median ratio over the images}} -> first block (1..24) whose median ratio exceeds
    the cutoff at both sizes, or None."""
    blocks = sorted(next(iter(median_by_size.values())))
    for b in blocks:
        if all(median_by_size[s][b] > cutoff for s in median_by_size):
            return b
    return None


def detection_block(median_pooled):
    """{block: median ratio over all items pooled} -> block with the largest median ratio (ties: the lower block)."""
    bs = sorted(median_pooled)
    return max(bs, key=lambda b: (median_pooled[b], -b))


def rank_neurons(score_sum, n_items, highest_layer, top=TOP_STORE):
    """score_sum: (24, N) masked scores summed over the items with outliers; returns [(layer, neuron, score)] for layers
    <= highest_layer, descending by score (ties: lower layer, then lower neuron), at most `top` entries."""
    s = np.asarray(score_sum, np.float64) / max(n_items, 1)
    L, N = s.shape
    cand = [(l, n) for l in range(min(highest_layer, L - 1) + 1) for n in range(N)]
    vals = np.array([s[l, n] for l, n in cand])
    order = np.lexsort((np.array([n for _, n in cand]), np.array([l for l, _ in cand]), -vals))
    return [(int(cand[i][0]), int(cand[i][1]), float(vals[i])) for i in order[:top]]


def neuron_dict_top(ranking, k):
    """{layer: [neurons]} from the first k entries of a ranking."""
    nd = {}
    for l, n, _ in ranking[:k]:
        nd.setdefault(int(l), []).append(int(n))
    return {l: sorted(v) for l, v in sorted(nd.items())}


def passes(p90_table, eval_blocks, sizes=SIZES, thr=P90_MAX):
    """p90_table: {(block, size): 90th percentile of max/median}; True if every (block, size) in scope is <= thr.
    Also returns the number of passing pairs."""
    pairs = [(b, s) for b in eval_blocks for s in sizes]
    ok = [p90_table[(b, s)] <= thr for b, s in pairs]
    return all(ok), int(sum(ok))


def choose_setting(grid, eval_blocks, sizes=SIZES, thr=P90_MAX):
    """grid: {branch: {k: p90_table}} with branches 'primary' and 'fallback' (fallback may be absent).
    Rule (DESIGN.md Section 4.2 step 5): the smallest k that passes in the primary branch; else the smallest k that
    passes in the fallback branch; else the (branch, k) with the most passing (block, size) pairs, ties to the primary
    branch, then the smaller k. Returns dict(branch, k, passed, n_pass, n_pairs)."""
    npairs = len(eval_blocks) * len(sizes)
    for br in ("primary", "fallback"):
        for k in sorted(grid.get(br, {})):
            ok, n = passes(grid[br][k], eval_blocks, sizes, thr)
            if ok:
                return dict(branch=br, k=int(k), passed=True, n_pass=n, n_pairs=npairs)
    best = None
    for bi, br in enumerate(("primary", "fallback")):
        for k in sorted(grid.get(br, {})):
            ok, n = passes(grid[br][k], eval_blocks, sizes, thr)
            key = (-n, bi, k)
            if best is None or key < best[0]:
                best = (key, br, k, n)
    if best is None:
        return dict(branch=None, k=None, passed=False, n_pass=0, n_pairs=npairs)
    return dict(branch=best[1], k=int(best[2]), passed=False, n_pass=best[3], n_pairs=npairs)


# ====================================================================== scoring helpers (numpy, cv2)
def q16(sal):
    """Verbatim common_alt.q16 (= excommon.q16): map / max|map| in float16."""
    return X.q16(sal)


def r_inside_fast(full, ins_idx, ind_idx):
    """common_alt.measures' r_inside of q16(full), from the AOI pixels only (flat indices in row-major order)."""
    q = q16(full).ravel()
    I = float(np.clip(q[ins_idx].astype(np.float32).astype(np.float64), 0, None).sum())
    D = float(np.clip(q[ind_idx].astype(np.float32).astype(np.float64), 0, None).sum())
    return I / (I + D) if I + D > 0 else float("nan")


def sweep_full(sig):
    """Published sweep map before q16: cv2.resize of the float32 sigmoid map to the 1024 x 768 canvas."""
    import cv2
    return cv2.resize(np.asarray(sig, np.float32), (W, H))


def swap_full(sig):
    """eval_full.score_map before q16: the float16 sigmoid map cast to float32, resized to the canvas."""
    import cv2
    return cv2.resize(np.asarray(sig, np.float32).astype(np.float16).astype(np.float32), (W, H))


def token_locations(gt2_mask, g, inside, ind):
    """Shares of the tokens above twice the median whose patch centre falls in the inside AOI, the inducer AOI or
    elsewhere. gt2_mask: (g*g,) bool in row-major grid order; centres mapped from the s x s grid to the 1024 x 768
    canvas: x = (j + 0.5) * 1024 / g, y = (i + 0.5) * 768 / g."""
    pos = np.flatnonzero(np.asarray(gt2_mask, bool))
    n = int(pos.size)
    if n == 0:
        return dict(n=0, inside=None, inducer=None, elsewhere=None)
    i, j = pos // g, pos % g
    xs = np.minimum(((j + 0.5) * W / g).astype(int), W - 1)
    ys = np.minimum(((i + 0.5) * H / g).astype(int), H - 1)
    a = inside[ys, xs]
    b = ind[ys, xs] & ~a
    return dict(n=n, inside=float(a.mean()), inducer=float(b.mean()), elsewhere=float((~a & ~b).mean()))


# ====================================================================== displays
V3_CONDS = ("illusory", "control180", "control90", "real", "real_filled", "mis10", "mis20", "mis30")
ALT_CONDS = tuple(f"alt{t}{ph}" for t in (10, 20, 30) for ph in ("a", "b"))


def displays(stimset):
    """[(name, path, cfg, inside flat idx, ind flat idx, inside mask, ind mask)] for the 480 stim_v3 Kanizsa displays
    (60 configurations x 8 conditions) or the 360 stim_alt displays, in Design configuration order; AOIs as mapsrc
    (stim_v3: the configuration's illusory AOI file, identical across conditions; stim_alt: aoi/<cfg>_aoi.npz)."""
    D = X.design()
    out = []
    for c in D.configs:
        a = X.v3_aoi(c) if stimset == "stim_v3" else X.alt_aoi(c)
        base = X.STIM_V3 if stimset == "stim_v3" else X.STIM_ALT
        conds = V3_CONDS if stimset == "stim_v3" else ALT_CONDS
        for cond in conds:
            nm = f"{c}_{cond}"
            out.append(dict(name=nm, path=os.path.join(base, "img", nm + ".png"), cfg=c, cond=cond,
                            ins=X.idx(a["inside"]), ind=X.idx(a["ind"]), inside_mask=a["inside"], ind_mask=a["ind"]))
    return out
