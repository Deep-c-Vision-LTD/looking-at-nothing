"""
Unit tests of explore/ttr (test-time registers; DESIGN.md Section 6, "Tests first") on SYNTHETIC inputs only: tiny
random-weight CLIPVisionModel / SiglipVisionModel built in memory from configs (no download, no weight file), random
images, synthetic activation and norm tables. No score, map, display, weight or result file of the project is read:
after require_public() (rule Section 8.1) a read guard wraps builtins.open, numpy.load, cv2.imread, PIL.Image.open and
torch.load and refuses any path inside the project root except this test's own output file.
CPU only (device cpu, 2 threads; MPS never touched). One process, seconds.
Checks (DESIGN.md Section 6): (i) no hooks = stock hidden states bit for bit, and hooks are removed on exit; (ii) the
register is appended and, for CLIP, enters layer 1 as pre_layrnorm.bias (zeros for SigLIP); (iii) the shift writes the
per-image, per-neuron max over all unmodified tokens into the register, zeros the patches and keeps CLIP's CLS;
(iv) a batch of 3 equals 3 single runs to 1e-6; (v) neurons in layer l leave hidden_states[0..l] identical to reg0;
(vi) feature slicing (g x g; CLS and register out; plain = the stock expression); (vii) SigLIP has no CLS (every
non-register token zeroed); (viii) the selection ranks a planted register neuron first and the sparsity filter removes
a dense neuron; (ix) emergence / detection blocks, the k-sweep, fallback and tie logic on synthetic tables; (x) the read
guard and the selection-image disjointness. Also: the fast r_inside equals the boolean-mask formula of
common_alt.measures; the swap map path; token locations; norm rows; every script calls require_public() first.
Usage: OMP_NUM_THREADS=2 $PY -B -I explore/ttr/test_ttr.py --out explore/ttr/tests_ttr_output.txt
Written 18 November 2026.
"""
import argparse, ast, builtins, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import ttr_common as C                                                    # noqa: E402

RES, LINES = [], []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "tests_ttr_output.txt"))
A = ap.parse_args()
OUT = os.path.abspath(A.out)


def out(s):
    LINES.append(s); print(s, flush=True)


def check(name, ok, detail=""):
    RES.append(bool(ok))
    out(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""))


# ---------------------------------------------------------------- guard (after require_public)
C.require_public()
ROOT = os.path.realpath(C.ROOT)
_orig = dict(open=builtins.open)


def _refuse(p):
    try:
        rp = os.path.realpath(os.fspath(p))
    except TypeError:
        return False
    return (rp == ROOT or rp.startswith(ROOT + os.sep)) and rp != OUT and rp != OUT + ".tmp"


def _guarded_open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and _refuse(file):
        raise RuntimeError(f"test read guard: refusing project file {file}")
    return _orig["open"](file, *a, **k)


builtins.open = _guarded_open
import cv2, torch                                                         # noqa: E402
from PIL import Image                                                     # noqa: E402
for mod, name in ((np, "load"), (cv2, "imread"), (Image, "open"), (torch, "load")):
    f = getattr(mod, name)

    def wrap(p, *a, _f=f, **k):
        if isinstance(p, (str, bytes, os.PathLike)) and _refuse(p):
            raise RuntimeError(f"test read guard: refusing project file {p}")
        return _f(p, *a, **k)
    setattr(mod, name, wrap)
torch.set_num_threads(2)
t_start = time.time()


def guard_refuses(fn):
    try:
        fn()
        return False
    except RuntimeError as e:
        return "read guard" in str(e)


# ---------------------------------------------------------------- tiny models
from transformers import CLIPVisionConfig, CLIPVisionModel, SiglipVisionConfig, SiglipVisionModel   # noqa: E402


def tiny(kind, seed=0):
    torch.manual_seed(seed)
    kw = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, image_size=28,
              patch_size=7)
    if kind == "clip":
        m = CLIPVisionModel(CLIPVisionConfig(hidden_act="quick_gelu", **kw)).eval()
        with torch.no_grad():
            m.pre_layrnorm.bias.normal_()                                 # non-trivial bias for check (ii)
    else:
        m = SiglipVisionModel(SiglipVisionConfig(**kw)).eval()
    assert next(m.parameters()).device.type == "cpu"
    return m


def hs_of(m, x):
    with torch.no_grad():
        return C.hidden_states(m, x)


x3 = torch.randn(3, 3, 42, 42, generator=torch.Generator().manual_seed(1))   # 6 x 6 grid
G = 6
out("=== test_ttr.py " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f"  torch {torch.__version__}")

for kind in ("clip", "siglip"):
    out(f"--- {kind}")
    m = tiny(kind)
    cls = C.BB_CFG[kind]["cls"]
    T0 = G * G + (1 if cls else 0)
    stock = hs_of(m, x3)
    # (i)
    with C.arm_hooks(m, kind, "plain") as t:
        nh = len(t.handles)
        plain = hs_of(m, x3)
    check(f"(i) {kind}: plain arm installs no hook and equals the stock hidden states bit for bit",
          nh == 0 and all(torch.equal(a, b) for a, b in zip(stock, plain)), f"{len(stock)} hidden states")
    with C.arm_hooks(m, kind, "ttr", {1: [0, 5]}):
        pass
    after = hs_of(m, x3)
    emb, layers = C.model_parts(m)
    nh_left = len(emb._forward_hooks) + sum(len(l.mlp.fc2._forward_pre_hooks) for l in layers)
    check(f"(i) {kind}: hooks removed on exit (no hook left, hidden states bit-identical again)",
          nh_left == 0 and all(torch.equal(a, b) for a, b in zip(stock, after)))
    try:
        C.TTR(m, cls, {1: [0]}, n_reg=0)
        refused = False
    except AssertionError:
        refused = True
    check(f"{kind}: a shift without a register is refused", refused)
    # (ii)
    with C.arm_hooks(m, kind, "reg0"):
        reg0 = hs_of(m, x3)
    ok_len = all(h.shape[1] == T0 + 1 for h in reg0)
    if cls:
        ok0 = torch.allclose(reg0[0][:, -1, :], m.pre_layrnorm.bias.expand(3, -1), atol=1e-6) and \
              not torch.allclose(m.pre_layrnorm.bias, torch.zeros_like(m.pre_layrnorm.bias))
        detail = "hidden_states[0][:, -1] = pre_layrnorm.bias (non-zero)"
    else:
        ok0 = torch.equal(reg0[0][:, -1, :], torch.zeros(3, 32))
        detail = "hidden_states[0][:, -1] = 0 (no pre-LayerNorm)"
    check(f"(ii) {kind}: register appended at the end (T + 1 tokens in every hidden state) and enters layer 1 as expected",
          ok_len and ok0, detail)
    check(f"(ii) {kind}: the register leaves hidden_states[0] of the image tokens unchanged (no positional change)",
          torch.equal(reg0[0][:, :-1, :], stock[0]))
    # (iii) and (vii)
    l, neurons = 2, [3, 17, 40]
    seen = {}
    h_before = layers[l].mlp.fc2.register_forward_pre_hook(lambda mod, a: seen.__setitem__("before", a[0].clone()),
                                                           prepend=True)
    with C.arm_hooks(m, kind, "ttr", {l: neurons}):
        h_after = layers[l].mlp.fc2.register_forward_pre_hook(lambda mod, a: seen.__setitem__("after", a[0].clone()))
        ttr = hs_of(m, x3)
        h_after.remove()
    h_before.remove()
    b, a = seen["before"], seen["after"]
    idx = torch.tensor(neurons)
    mx = b[:, :, idx].amax(1)
    first_patch = 1 if cls else 0
    ok_reg = torch.equal(a[:, -1, idx], mx)
    ok_zero = torch.equal(a[:, first_patch:-1, idx], torch.zeros_like(a[:, first_patch:-1, idx]))
    ok_cls = torch.equal(a[:, 0, idx], b[:, 0, idx]) if cls else True
    other = torch.ones(64, dtype=torch.bool); other[idx] = False
    ok_other = torch.equal(a[:, :, other], b[:, :, other])
    per_image = not torch.equal(mx[0], mx[1])
    check(f"(iii) {kind}: register gets the per-image, per-neuron signed max over all unmodified tokens (CLS, patches, "
          "register)", ok_reg and per_image, f"max over {b.shape[1]} tokens")
    check(f"(iii) {kind}: patches set to 0 for the register neurons, other neurons untouched", ok_zero and ok_other)
    if cls:
        check("(iii) clip: CLS keeps its value for the register neurons", ok_cls)
    else:
        check("(vii) siglip: no CLS; every non-register token is zeroed for the register neurons",
              torch.equal(a[:, :-1, idx], torch.zeros_like(a[:, :-1, idx])))
    # (v)
    same_upto = all(torch.equal(ttr[k], reg0[k]) for k in range(l + 1))
    differs_after = not torch.equal(ttr[l + 1], reg0[l + 1])
    check(f"(v) {kind}: neurons in layer {l} leave hidden_states[0..{l}] identical to reg0 and change hidden_states[{l + 1}]",
          same_upto and differs_after)
    # (iv)
    nd = {1: [0, 9], 2: [3, 17, 40]}
    with C.arm_hooks(m, kind, "ttr", nd):
        hb = hs_of(m, x3)
        singles = [hs_of(m, x3[i:i + 1]) for i in range(3)]
    md = max(float((hb[k][i] - singles[i][k][0]).abs().max()) for i in range(3) for k in range(len(hb)))
    check(f"(iv) {kind}: a batch of 3 equals 3 single runs to 1e-6 (ttr arm, two hooked layers)", md <= 1e-6,
          f"max |diff| {md:.2e}")
    # (vi)
    t = reg0[3]
    fg = C.grid_feats(t, kind, "reg0")
    ref = (t[:, 1:-1, :] if cls else t[:, :-1, :]).transpose(1, 2).reshape(3, -1, G, G).contiguous()
    fp = C.grid_feats(stock[3], kind, "plain")
    refp = (stock[3][:, 1:, :] if cls else stock[3]).transpose(1, 2).reshape(3, -1, G, G).contiguous()
    check(f"(vi) {kind}: reg0/ttr features are g x g with CLS and register removed; plain = the stock expression",
          fg.shape == (3, 32, G, G) and torch.equal(fg, ref) and torch.equal(fp, refp))
    pn = C.patch_norms(t, kind, "reg0")
    check(f"(vi) {kind}: patch norms exclude CLS and register ({G * G} tokens)",
          pn.shape == (3, G * G) and np.allclose(pn, (t[:, 1:-1] if cls else t[:, :-1]).norm(dim=-1).double().numpy()))
    try:
        C.grid_feats(t[:, :-2, :], kind, "reg0")
        bad = False
    except AssertionError:
        bad = True
    check(f"(vi) {kind}: a token count that is not a square grid is refused", bad)
    rn = C.register_norm(ttr[4], "ttr")
    check(f"{kind}: register norm reported for reg0/ttr, None for plain",
          rn.shape == (3,) and C.register_norm(stock[4], "plain") is None)

# ---------------------------------------------------------------- (viii) selection scores
out("--- selection")
rng = np.random.RandomState(0)
acts = rng.randn(50, 10).astype(np.float32) * 0.1 - 0.3                # mostly below 0.5
outl = [3, 7]
acts[outl, 4] = 10.0                                                    # planted register neuron (sparse)
acts[:, 6] = 5.0                                                        # dense neuron: high everywhere
acts[outl, 6] = 12.0
sc, mk = C.item_scores(acts, outl)
check("(viii) the planted register neuron has the largest score; the dense neuron is removed by the sparsity filter",
      int(np.argmax(sc)) == 4 and sc[6] == 0.0 and not mk[6] and mk[4], f"scores {np.round(sc[[4, 6]], 3).tolist()}")
sc2, _ = C.item_scores(acts, outl, sparse=False)
check("(viii) without the sparsity filter the dense neuron would rank first", int(np.argmax(sc2)) == 6)
ssum = np.zeros((24, 10)); ssum[5, 4] = 9.0; ssum[2, 1] = 4.0; ssum[12, 7] = 20.0; ssum[2, 3] = 4.0
rk = C.rank_neurons(ssum, 2, highest_layer=8)
check("(viii) rank_neurons keeps layers <= h, orders by score, ties to the lower neuron",
      rk[0][:2] == (5, 4) and rk[1][:2] == (2, 1) and rk[2][:2] == (2, 3) and all(r[0] <= 8 for r in rk)
      and abs(rk[0][2] - 4.5) < 1e-12)
rk23 = C.rank_neurons(ssum, 2, highest_layer=23)
check("(viii) the fallback branch (h = 23) admits the later layer", rk23[0][:2] == (12, 7))
check("(viii) neuron_dict_top groups the top k by layer", C.neuron_dict_top(rk, 3) == {2: [1, 3], 5: [4]})

# ---------------------------------------------------------------- (ix) blocks and k sweep
med = {336: {b: (1.2 if b < 9 else (4.5 if b == 9 else 12.0)) for b in range(1, 25)},
       448: {b: (1.2 if b < 10 else 12.0) for b in range(1, 25)}}
check("(ix) emergence block: first block above 4 at BOTH sizes", C.emergence_block(med) == 10)
check("(ix) emergence block: None if no block exceeds the cutoff", C.emergence_block({336: {1: 2.0}, 448: {1: 3.0}}) is None)
check("(ix) detection block: largest pooled median, ties to the lower block",
      C.detection_block({4: 2.0, 12: 9.0, 16: 27.0, 20: 27.0}) == 16)
ev = [12, 16]


def tab(v):
    return {(b, s): v.get((b, s), 1.5) for b in ev for s in (336, 448)}


g1 = {"primary": {5: tab({(16, 448): 3.0}), 10: tab({}), 20: tab({})}}
check("(ix) smallest passing k in the primary branch", C.choose_setting(g1, ev) ==
      dict(branch="primary", k=10, passed=True, n_pass=4, n_pairs=4))
g2 = {"primary": {5: tab({(16, 448): 3.0}), 10: tab({(16, 448): 2.5})}, "fallback": {5: tab({(12, 336): 2.2}), 45: tab({})}}
check("(ix) fallback branch used when no primary k passes", C.choose_setting(g2, ev)["branch"] == "fallback" and
      C.choose_setting(g2, ev)["k"] == 45)
g3 = {"primary": {5: tab({(16, 448): 3.0, (12, 448): 3.0}), 10: tab({(16, 448): 2.5})},
      "fallback": {5: tab({(12, 336): 2.2}), 45: tab({(12, 336): 2.1, (16, 336): 2.1})}}
s3 = C.choose_setting(g3, ev)
check("(ix) neither passes: most passing pairs, ties to the primary branch, then the smaller k",
      s3 == dict(branch="primary", k=10, passed=False, n_pass=3, n_pairs=4), str(s3))
check("(ix) the criterion is <= 2.0 (2.0 itself passes)", C.passes(tab({(12, 336): 2.0}), ev)[0])

# ---------------------------------------------------------------- (x) guard and disjointness
out("--- guards")
check("(x) assert_not_display refuses stim_v3, stim_alt, stim_rigid, stim_v2, stim_ehr paths",
      all(guard_refuses(lambda p=p: C.assert_not_display(p)) for p in
          [os.path.join(C.ROOT, "v2", "stim_v3", "img", "x.png"), os.path.join(C.ROOT, "v3", "stim_alt", "img", "y.png"),
           os.path.join(C.RE, "stim_rigid", "img", "z.png"), os.path.join(C.ROOT, "stim_v2", "img", "a.png"),
           os.path.join(C.ROOT, "stim_ehr", "img", "b.png")]))
ok_mit = True
try:
    C.assert_not_display("/data/mit1003/ALLSTIMULI/i1.jpeg")
except RuntimeError:
    ok_mit = False
check("(x) assert_not_display accepts an MIT1003 path", ok_mit)
fake = [dict(image=f"/data/mit1003/ALLSTIMULI/img{i:04d}.jpeg") for i in range(1003)]
sel, held = C.selection_images(fake)
idx = C.selection_indices(803)
check("(x) 100 selection images from the 803-image training split, sorted, RandomState(2026), disjoint from the "
      "held-out 200", len(sel) == 100 and idx == sorted(idx) and max(idx) < 803 and not set(sel) & set(held)
      and idx == C.selection_indices(803) and idx[:3] == sorted(np.random.RandomState(2026).choice(803, 100, False))[:3])
check("(x) the test read guard refuses project files (open, numpy.load, cv2.imread, torch.load)",
      guard_refuses(lambda: open(os.path.join(C.RE, "REANALYSIS_RULE.md")))
      and guard_refuses(lambda: np.load(os.path.join(C.RE, "x.npy")))
      and guard_refuses(lambda: cv2.imread(os.path.join(C.ROOT, "v2", "stim_v3", "img", "a.png")))
      and guard_refuses(lambda: torch.load(os.path.join(C.ROOT, "sweep", "clip", "readout_L12.pt"))))

# ---------------------------------------------------------------- scoring helpers
out("--- scoring helpers")
rs = np.random.RandomState(3)
ok_fast, md = True, 0.0
for trial in range(20):
    full = cv2.resize(rs.rand(24, 24).astype(np.float32) - (0.3 if trial % 4 == 0 else 0.0), (C.W, C.H))
    ins = np.zeros((C.H, C.W), bool); ins[200:500, 300:700] = True
    ind = np.zeros((C.H, C.W), bool); ind[150:250, 250:350] = True; ind[450:550, 650:750] = True; ins &= ~ind
    q = C.q16(full)
    sm = np.clip(np.asarray(q, np.float32).astype(np.float64), 0, None)
    ref = float(sm[ins].sum() / (sm[ins].sum() + sm[ind].sum()))
    v = C.r_inside_fast(full, np.flatnonzero(ins.ravel()), np.flatnonzero(ind.ravel()))
    md = max(md, abs(v - ref)); ok_fast &= (v == ref)
check("fast r_inside equals common_alt.measures' boolean-mask formula exactly (20 random maps, some negative)",
      ok_fast, f"max |diff| {md}")
sig = rs.rand(28, 28).astype(np.float32)
check("swap map path = eval_full.score_map (float16 sigmoid -> float32 -> resize)",
      np.array_equal(C.swap_full(sig), cv2.resize(sig.astype(np.float16).astype(np.float32), (C.W, C.H))))
check("sweep map path = mapsrc published form (float32 sigmoid -> resize)",
      np.array_equal(C.sweep_full(sig), cv2.resize(sig, (C.W, C.H))))
n = np.ones((2, 16)); n[0, 5] = 10.0; n[1, [1, 2]] = 3.0
rows = C.norm_rows(n)
check("norm rows: ratio max/median, share and count above twice the median",
      np.allclose(rows["ratio"], [10.0, 3.0]) and rows["count"].tolist() == [1, 2] and
      np.allclose(rows["share"], [1 / 16, 2 / 16]))
mask = np.zeros(16, bool); mask[[0, 5, 15]] = True
ins = np.zeros((C.H, C.W), bool); ins[: C.H // 2, : C.W // 2] = True
ind = np.zeros((C.H, C.W), bool); ind[C.H // 2:, C.W // 2:] = True
loc = C.token_locations(mask, 4, ins, ind)
check("token locations: patch centres mapped to the canvas (4 x 4 grid: token 0 inside, 15 inducer, 5 inside)",
      loc["n"] == 3 and abs(loc["inside"] - 2 / 3) < 1e-12 and abs(loc["inducer"] - 1 / 3) < 1e-12 and loc["elsewhere"] == 0)
pcst = C.position_constancy([mask, mask, ~mask])
check("position constancy: share of images with the modal set", abs(pcst["share_modal"] - 2 / 3) < 1e-12)
check("jaccard", C.jaccard([1, 2], [2, 3]) == 1 / 3 and C.jaccard([], []) == 1.0)

# ---------------------------------------------------------------- scripts
out("--- scripts")
for nm in ("ttr_select.py", "ttr_score.py", "ttr_analyze.py"):
    src = _orig["open"](os.path.join(HERE, nm)).read()
    tree = ast.parse(src)
    mains = [f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "main"]
    first = mains[0].body[0] if mains else None
    ok = (first is not None and isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
          and getattr(first.value.func, "attr", "") == "require_public")
    check(f"{nm}: main() calls require_public() before anything else", ok)
    code_only = "\n".join(l.split("#")[0] for l in src.splitlines())
    check(f"{nm}: no MPS device in code (no to('mps'), device='mps', torch.mps or torch.backends.mps)",
          not any(s in code_only for s in ('"mps"', "'mps'", "torch.mps", "backends.mps")))
check("no MPS tensor was created by the tests", not any(p.device.type == "mps" for p in tiny("clip").parameters()))

dt = time.time() - t_start
out(f"{len(RES)} checks, {sum(not r for r in RES)} failed, {dt:.1f}s")
with _orig["open"](OUT + ".tmp", "w") as f:
    f.write("\n".join(LINES) + "\n")
os.replace(OUT + ".tmp", OUT)
sys.exit(0 if all(RES) else 1)
