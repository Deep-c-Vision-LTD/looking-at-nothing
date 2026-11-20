"""
Independent CPU re-scoring of 20 random stim_rigid displays (verification step, 18 November 2026).

Models: DeepGaze IIE (deepgaze_pytorch, pretrained, full-resolution input, uniform centre bias, exp of the log density)
and two published swap heads on DINOv2-B (v3/gpu/full/ckpt/dinov2_L12_R4_s0.pt and dinov2_L6_R2_s1.pt, read only) at
336 and 448 px. Written here, not taken from the scoring scripts: display selection, image reading, DeepGaze call, the
DINOv2 forward (Hugging Face AutoModel, hidden_states[L] without CLS, reshaped to the patch grid, held in float16), the
map post-processing (sigmoid map in float16 -> float32 -> bilinear resize to 1024 x 768; q16 = map / max|map| in
float16; clip at 0) and r_inside = M(inside) / (M(inside) + M(inducer discs)) on the display's own AOI file. Only the
head architecture class is taken from the published code (recommon.make_head = v3/gpu/fullcommon.make_head), since the
checkpoints are state dicts of that class.
CPU only: the ab_objective stub is put first on sys.path before anything else is imported (recommon asserts it), no
tensor or module is ever moved to another device, torch uses 4 threads. A watchdog stops the process after
PROBE_LIMIT_S seconds (default 540) so the probe stays under 10 minutes while the MSDB job runs.
Output: neuro/verify_rescore/rescore_probe.json (written after every display). Nothing else is written.
"""
import json, math, os, signal, sys, time
T0 = time.time()
LIMIT = int(os.environ.get("PROBE_LIMIT_S", "540"))
for k, v in (("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"), ("VECLIB_MAXIMUM_THREADS", "4"),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("V3_DEVICE", "cpu")):
    os.environ[k] = v
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                          # re-analysis folder (neurocomputing/ in the repository)
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root (as analyze_re.py)
PROBE_SET = os.environ.get("PROBE_SET", "all")          # "all": the 660 stim_rigid displays; "rigid": the 480 rig<phi> displays
OUT = os.path.join(HERE, "rescore_probe.json" if PROBE_SET == "all" else f"rescore_probe_{PROBE_SET}.json")
STIM = os.path.join(RE, "stim_rigid")


def _timeout(signum, frame):
    raise TimeoutError(f"probe watchdog: {LIMIT} s reached")
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(LIMIT)

sys.path.insert(0, os.path.join(RE, "_stub"))
import ab_objective as AB                       # noqa: E402  the CPU stub (no backend query)
assert AB.__file__.startswith(os.path.join(RE, "_stub")) and AB.DEV == "cpu", AB.__file__
import numpy as np, cv2, torch                  # noqa: E402
sys.path.insert(0, os.path.join(RE, "gpu"))
import recommon                                 # noqa: E402  (head architecture only; asserts the stub)
assert recommon.DEVICE == "cpu"
torch.set_num_threads(4)
cv2.setNumThreads(1)

H_, W_ = 768, 1024
HEADS = [("dinov2_L12", "R4", 0, 12), ("dinov2_L6", "R2", 1, 6)]
SIZES = (336, 448)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def rgb_of(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    assert bgr is not None and bgr.shape == (H_, W_, 3), path
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def r_inside_q16(sal, inside, ind):
    sal = np.asarray(sal, np.float32)
    mx = float(np.abs(sal).max())
    q = (sal / (mx if mx > 0 else 1.0)).astype(np.float16)
    s = np.clip(q.astype(np.float32).astype(np.float64), 0, None)
    a, b = s[inside].sum(), s[ind].sum()
    return float(a / (a + b))


def main():
    items = json.load(open(os.path.join(STIM, "items.json")))
    if PROBE_SET == "rigid":
        items = [it for it in items if it["cond"].startswith("rig")]
        assert len(items) == 480
    rs = np.random.RandomState(20261118 if PROBE_SET == "all" else 20261119)
    pick = sorted(rs.choice(len(items), 20, replace=False).tolist())
    sel = [items[i] for i in pick]
    pub_iie = json.load(open(os.path.join(RE, "scores_rigid", "DeepGaze_IIE.json")))
    pub_h = {f"{f}_{a}_s{s}": json.load(open(os.path.join(RE, "scores_rigid", "heads", "scores", f"stim_rigid_{f}_{a}_s{s}.json")))
             for f, a, s, _ in HEADS}
    res = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid=os.getpid(), probe_set=PROBE_SET,
               seed=20261118 if PROBE_SET == "all" else 20261119,
               selected=[it["name"] for it in sel], torch_threads=torch.get_num_threads(), rows=[], status="running")

    def dump():
        json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
    dump()

    # models (CPU)
    import deepgaze_pytorch
    iie = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    torch.set_num_threads(4)
    from transformers import AutoModel
    bb = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    heads = {}
    for f, a, s, L in HEADS:
        m = recommon.make_head(a, 768)
        m.load_state_dict(torch.load(os.path.join(ROOT, "v3/gpu/full/ckpt", f"{f}_{a}_s{s}.pt"), map_location="cpu"))
        m.eval(); m.use_cb = False
        heads[f"{f}_{a}_s{s}"] = (m, L)
    for mod in [iie, bb] + [m for m, _ in heads.values()]:
        assert all(p.device.type == "cpu" for p in mod.parameters())
    res["load_s"] = round(time.time() - T0, 1); dump()

    cb = torch.full((1, H_, W_), -math.log(H_ * W_))
    for it in sel:
        t1 = time.time()
        rgb = rgb_of(os.path.join(STIM, "img", it["name"] + ".png"))
        A = np.load(os.path.join(STIM, it["aoi"]))
        inside, ind = A["inside"].astype(bool), A["ind"].astype(bool)
        row = dict(name=it["name"], cond=it["cond"])
        with torch.no_grad():
            x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1).copy())[None]
            dens = torch.exp(iie(x, cb))[0].numpy()
            dens = np.squeeze(dens).astype(np.float32)
            assert dens.shape == (H_, W_)
            row["DeepGaze_IIE"] = dict(mine=r_inside_q16(dens, inside, ind), published=pub_iie[it["name"]]["r_inside"])
            for s in SIZES:
                xi = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
                xi = torch.from_numpy(((xi / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())[None]
                hs = bb(pixel_values=xi, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
                for tag, (m, L) in heads.items():
                    t = hs[L][:, 1:, :]
                    g = int(round(t.shape[1] ** 0.5)); assert g * g == t.shape[1]
                    feat = t.transpose(1, 2).reshape(1, -1, g, g).contiguous().half().float()
                    sal = torch.sigmoid(m(feat, (s, s)))[0, 0].numpy().astype(np.float16).astype(np.float32)
                    full = cv2.resize(sal, (W_, H_))
                    row[f"{tag}_px{s}"] = dict(mine=r_inside_q16(full, inside, ind),
                                               published=pub_h[tag][str(s)]["r_inside"][it["name"]])
        for k, v in row.items():
            if isinstance(v, dict):
                v["abs_diff"] = abs(v["mine"] - v["published"])
        row["seconds"] = round(time.time() - t1, 2)
        res["rows"].append(row); dump()
    diffs = [v["abs_diff"] for r in res["rows"] for v in r.values() if isinstance(v, dict)]
    res.update(status="done", n_values=len(diffs), max_abs_diff=max(diffs), n_above_1e4=sum(d > 1e-4 for d in diffs),
               wall_s=round(time.time() - T0, 1))
    dump()
    print(f"done: {len(diffs)} values, max |mine - published| {max(diffs):.3e}, wall {res['wall_s']} s")


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as e:
        r = json.load(open(OUT)) if os.path.exists(OUT) else {}
        r["status"] = f"stopped: {e}"
        json.dump(r, open(OUT, "w"), indent=1)
        print(r["status"]); sys.exit(3)
