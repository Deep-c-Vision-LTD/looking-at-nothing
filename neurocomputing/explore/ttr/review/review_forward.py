"""
neuro/reanalysis/explore/ttr/review/review_forward.py: independent re-implementation of the test-time-registers
intervention for the review (explore/ttr/REVIEW.md, 18 November 2026). EXPLORATORY check; decides nothing; writes no
score file.

What is independent: the intervention is NOT taken from ttr_common.TTR. It is written here from the authors' released
OpenCLIP ViT-L/14 model code (Hugging Face amildravid4292/clip-vitl14-test-time-registers, transformer.py, read on the
web 18 November 2026): one zero token concatenated after the embeddings (so CLIP's pre-LayerNorm acts on it); in each
listed layer the MLP forward is replaced by fc1 -> activation -> a zero map for the register neurons, the register
position set to torch.max over all tokens of the unmodified activations, CLS (index 0, CLIP only) set back to its
original value -> fc2 (instead of ttr_common's fc2 forward pre-hook). Feature slicing, map resizing and r_inside are
also written here. Taken from the project (not under review): the backbone loader, the preprocessing (sweep.prep), the
readout heads (sweep.Readout, recommon.make_head via mapsrc.load_head), the AOI files, the CPU environment (mapsrc
torch_env: ab_objective stub, V3_DEVICE=cpu).
Displays are scored one at a time (batch 1; the scored run used batch 6), so agreement is expected to ~1e-6, not 0.
Checks: plain, reg0 and ttr r_inside of 3 stim_v3 displays at 336 and 448 px against scores_ttr/<arm>/ for SigLIP-L
(swap R4 seed 0 on block 24, sweep block 12) and CLIP-L (sweep blocks 16 and 24); the register-token norm at a flagged
block against scores_ttr/norms; and the selection image list re-derived from build_mit1003 and RandomState(2026).
CPU only: device cpu, metal_guard() after each model load, no MPS call anywhere. require_public() first.
Run (one process): nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    $PY -B -I explore/ttr/review/review_forward.py      ($PY: the project's Python 3.12 interpreter)
Output: explore/ttr/review/review_forward_output.json (and stdout).
"""
import json, os, sys
sys.dont_write_bytecode = True
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
sys.path.insert(0, RE)
import analyze_re                                                # noqa: E402  (only for the guard)
analyze_re.require_public(RE)                                    # rule Section 8.1, first
os.environ["V3_DEVICE"] = "cpu"
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
sys.path.insert(0, os.path.join(RE, "explore"))
import numpy as np                                               # noqa: E402
import mapsrc                                                    # noqa: E402  (CPU environment, loaders)

ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))
SC = os.path.join(RE, "scores_ttr")
OUTP = os.path.join(RE, "explore", "ttr", "review", "review_forward_output.json")
DISPLAYS = ["square_sr50_light_p2_illusory", "square_sr50_light_p2_control180", "triangle_sr70_dark_p4_illusory"]
LINES, OUT = [], {}


def say(s):
    print(s, flush=True)
    LINES.append(s)


class MyTTR:
    """Register (+ optional neuron shift) written from the released transformer.py; restored on exit."""

    def __init__(self, torch, model, has_cls, nd=None, reg=True):
        self.t, self.m, self.cls, self.nd, self.reg = torch, model, has_cls, (nd or {}), reg
        self.saved = []

    def __enter__(self):
        torch = self.t
        root = self.m.vision_model if hasattr(self.m, "vision_model") else self.m
        if self.reg:
            emb = root.embeddings
            f0 = emb.forward

            def emb_fwd(*a, **k):
                x = f0(*a, **k)
                return torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)], dim=1)
            emb.forward = emb_fwd
            self.saved.append(emb)
        for l, neurons in self.nd.items():
            mlp = root.encoder.layers[l].mlp
            nn_ = list(neurons)

            def mlp_fwd(h, mlp=mlp, nn_=nn_):
                a = mlp.activation_fn(mlp.fc1(h))
                orig = a.clone()
                new = torch.zeros((a.shape[0], a.shape[1], len(nn_)), dtype=a.dtype)
                new[:, -1:, :] = torch.max(orig[:, :, nn_], dim=1, keepdim=True).values
                if self.cls:
                    new[:, 0, :] = a[:, 0, nn_]
                a[:, :, nn_] = new
                return mlp.fc2(a)
            mlp.forward = mlp_fwd
            self.saved.append(mlp)
        return self

    def __exit__(self, *e):
        for mod in self.saved:
            del mod.forward                                       # back to the class method
        self.saved = []
        return False


def r_inside(sig, inside, ind, swap):
    import cv2
    s = np.asarray(sig, np.float32)
    if swap:
        s = s.astype(np.float16).astype(np.float32)
    full = cv2.resize(s, (1024, 768))
    q = (full / max(float(np.abs(full).max()), 1e-30)).astype(np.float16)
    v = np.clip(q.astype(np.float32).astype(np.float64), 0, None)
    I, D = v[inside].sum(), v[ind].sum()
    return float(I / (I + D))


def grid(t, has_cls, extra):
    a = 1 if has_cls else 0
    tt = t[:, a:t.shape[1] - extra, :]
    g = int(round(tt.shape[1] ** 0.5))
    assert g * g == tt.shape[1]
    return tt.transpose(1, 2).reshape(tt.shape[0], -1, g, g).contiguous()


def run_backbone(T, bb, units):
    import cv2
    torch = T["torch"]
    sw = mapsrc.sweep_module(bb)
    model = sw.load_backbone()
    assert next(model.parameters()).device.type == "cpu"
    mapsrc.X.metal_guard()
    has_cls = bb == "clip"
    sel = json.load(open(os.path.join(RE, "results", "explore", "ttr", f"TTR_select_{bb}.json")))["result"]
    nd = {int(k): v for k, v in sel["neuron_dict"].items()}
    heads = {}
    for u in units:
        if "|" in u:
            heads[u] = mapsrc.load_head(os.path.join(mapsrc.SWAP_CKPT, "siglip_L24_R4_s0.pt"), "R4", "siglip_L24")
        else:
            L = int(u.split("_L")[1])
            m = sw.Readout(model.config.hidden_size)
            m.load_state_dict(torch.load(os.path.join(sw.WORK, f"readout_L{L}.pt"), map_location="cpu"))
            m.eval(); m.use_cb = False
            heads[u] = m
    mapsrc.X.metal_guard()
    stored = {arm: {u: {} for u in units} for arm in ("plain", "reg0", "ttr")}
    for arm in stored:
        for u in units:
            tag = "siglip_L24_R4_s0" if "|" in u else f"{u}_s0"
            stored[arm][u] = json.load(open(os.path.join(SC, arm, f"stim_v3_{tag}.json")))
    norms_t = json.load(open(os.path.join(SC, "norms", f"ttr_stim_v3_{bb}.json")))["displays"]
    maxd = {arm: 0.0 for arm in stored}
    n = 0
    reg_cmp = []
    flag_block = 16 if bb == "clip" else 24
    for nm in DISPLAYS:
        cfg = nm.rsplit("_", 1)[0]
        a = np.load(os.path.join(ROOT, "v2", "stim_v3", "img", f"{cfg}_illusory_aoi.npz"))
        inside, ind = a["inside"].astype(bool), a["ind"].astype(bool)
        rgb = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "v2", "stim_v3", "img", nm + ".png")), cv2.COLOR_BGR2RGB)
        for s in (336, 448):
            x = sw.prep(rgb, s)[None]
            for arm in ("plain", "reg0", "ttr"):
                with torch.no_grad():
                    if arm == "plain":
                        hs = model(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
                    else:
                        with MyTTR(torch, model, has_cls, nd if arm == "ttr" else None, reg=True):
                            hs = model(pixel_values=x, interpolate_pos_encoding=True,
                                       output_hidden_states=True).hidden_states
                    extra = 0 if arm == "plain" else 1
                    for u, h in heads.items():
                        L = 24 if "|" in u else int(u.split("_L")[1])
                        f = grid(hs[L], has_cls, extra)
                        if "|" in u:
                            f = f.half().float()
                        sig = torch.sigmoid(h(f, (s, s)))[0, 0].numpy()
                        mine = r_inside(sig, inside, ind, "|" in u)
                        theirs = stored[arm][u][str(s)]["r_inside"][nm]
                        maxd[arm] = max(maxd[arm], abs(mine - theirs)); n += 1
                    if arm == "ttr":
                        rn = float(hs[flag_block][0, -1, :].norm())
                        reg_cmp.append((rn, norms_t[nm][str(s)][str(flag_block)]["register_norm"]))
                        pn = hs[flag_block][0, (1 if has_cls else 0):-1, :].norm(dim=-1)
                        OUT.setdefault(f"{bb}_ttr_block{flag_block}_max_over_median", []).append(
                            float(pn.max() / pn.median()))
    rd = max(abs(a - b) / b for a, b in reg_cmp)
    for arm, d in maxd.items():
        ok = d <= 1e-4
        say(f"{'MATCH' if ok else 'DIFF '}  {bb} {arm}: my implementation vs scores_ttr/{arm} on {len(DISPLAYS)} displays x 2 "
            f"sizes x {len(units)} units: max |r_inside diff| {d:.3g}")
        OUT[f"{bb}_{arm}_max_abs_diff"] = d
    say(f"{'MATCH' if rd <= 1e-4 else 'DIFF '}  {bb} register-token norm at block {flag_block} (ttr) vs scores_ttr/norms: "
        f"max relative diff {rd:.3g}; my values {[round(a, 2) for a, _ in reg_cmp]}")
    OUT[f"{bb}_register_norm_rel_diff"] = rd
    del model


def check_selection():
    import ab_objective as AB
    from src.data import build_mit1003
    allm = build_mit1003(f"{AB.SR}/mit1003")
    train, held = allm[:-200], allm[-200:]
    idx = sorted(np.random.RandomState(2026).choice(len(train), 100, replace=False).tolist())
    mine = [os.path.basename(train[i]["image"]) for i in idx]
    stored = json.load(open(os.path.join(RE, "results", "explore", "ttr", "TTR_select_clip.json")))["result"]["selection_images"]
    hs = {os.path.basename(p["image"]) for p in held}
    ok = mine == stored and not (set(mine) & hs) and len(allm) == 1003
    say(f"{'MATCH' if ok else 'DIFF '}  selection images re-derived (build_mit1003, 803 training, RandomState(2026)): equal to "
        f"TTR_select: {mine == stored}; overlap with held-out 200: {len(set(mine) & hs)}")
    OUT["selection_rederived_equal"] = mine == stored
    OUT["selection_overlap_heldout"] = len(set(mine) & hs)


def main():
    T = mapsrc.torch_env()
    say("review_forward.py: independent TTR implementation (from the released transformer.py) vs the stored scores")
    say(f"pid {os.getpid()}, torch {T['torch'].__version__}, device cpu")
    run_backbone(T, "siglip", ["siglip_L24|R4", "siglip_L12"])
    run_backbone(T, "clip", ["clip_L16", "clip_L24"])
    check_selection()
    with open(OUTP + ".tmp", "w") as f:
        json.dump(dict(lines=LINES, checks=OUT), f, indent=1)
    os.replace(OUTP + ".tmp", OUTP)
    say(f"wrote {OUTP}")


if __name__ == "__main__":
    main()
