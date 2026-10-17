"""
Per-image scores of every survey model and of the nine primary layer readouts on stim_v3 (740 images).

Models are run through the shared model code in models/, imported, not copied:
  edge, sr        run_v2.edge_null, run_v2.spectral_residual
  msinet          maps written by models/msinet_maps.py (TensorFlow environment; see run_msinet_v3.py), read as uint8
                  PNGs
  aspp, iie, msdb gate_kanizsa.load_models (xSoyelNet-ASPP 448 px letterbox, centre bias off; DeepGaze IIE full
                  resolution, uniform centre bias; DeepGaze MSDB half resolution, 17.5 px/deg, dataset=None)
  ml              run_v2.build(["ml"]) (ab_layers.Readout + scales.sal_at, blocks 6+10, TTA 280/336/392/448 x flip,
                  centre bias off)
  sweep:<bb>      sweep.py inference (sweep.prep, sweep.feats, sweep.Readout, sigmoid, cv2.resize to the image,
                  centre bias off) for the primary layers (DINOv2 2, 8, 12; CLIP 4, 12, 24; SigLIP 4, 8, 24) at
                  336 and 448 px
Everything runs on the CPU with 3 threads (ab_objective.DEV is overridden to "cpu" before any model code is imported).

Per image and model tag: r on the inside AOI (gate_kanizsa.rel_inside, the published measure), on inside_eroded and on
inside_eroded_all, plus the share of total map mass in inside, inside_eroded, inside_eroded_all and the inducer AOI.
Maps: <results>/v2/scores/maps/<tag>/<name>.npz with sal = map / max(map) as float16 and scale = max(map) (float64)
(not distributed; analysis/core_aoi.py and scoring/measure_v3maps.py read them).
Scores: <results>/v2/scores/raw/<tag>.json (distributed as <tag>.json.xz; a rerun starts from scratch and writes .json).
Usage: python score_v3.py <key> [<key> ...]; keys: edge sr msinet aspp ml iie msdb sweep:dinov2 sweep:clip sweep:siglip
Optional: SCORE_SHARD="i/n" scores only images with index % n == i into raw/<tag>.shard<i>of<n>.json;
python score_v3.py merge <tag> merges the shard files into raw/<tag>.json (shards moved to <work>/trash/score_shards/).
"""
import json, os, sys, time
for k, v in (("OMP_NUM_THREADS", "3"), ("MKL_NUM_THREADS", "3"), ("VECLIB_MAXIMUM_THREADS", "3"),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[k] = os.environ.get(k, v) if k.endswith("OFFLINE") else v      # offline by default; =0 to download
import numpy as np, cv2, torch
torch.set_num_threads(3)
cv2.setNumThreads(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
import ab_objective as AB                    # noqa: E402  (puts the repository root on sys.path)
AB.DEV = "cpu"                               # CPU only, as in the published run
import gate_kanizsa as gk                    # noqa: E402
import paths as LP                           # noqa: E402

SC = str(LP.V2 / "scores"); V2 = str(LP.V2)
STIM = str(LP.STIM_V3)
ITEMS = json.load(open(os.path.join(STIM, "items.json")))
MAPS, RAW, LOGS = os.path.join(SC, "maps"), os.path.join(SC, "raw"), str(LP.LOGS)
for d in (MAPS, RAW, LOGS):
    os.makedirs(d, exist_ok=True)
PRIMARY_LAYERS = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
SIZES = (336, 448)
NAMES = {"edge": "Edge-energy null", "sr": "Spectral residual", "msinet": "MSI-Net", "aspp": "xSoyelNet-ASPP",
         "ml": "xSoyelNet-ML", "iie": "DeepGaze IIE", "msdb": "DeepGaze MSDB"}

_aoi = {}
def aoi(it):
    """AOIs are identical across the conditions of a configuration (verified by verify_stim_v3.py); cache per config."""
    c = it["config"]
    if c not in _aoi:
        a = np.load(os.path.join(STIM, "img", it["aoi"]))
        _aoi.clear()
        _aoi[c] = {k: a[k] for k in ("inside", "ind", "inside_eroded", "inside_eroded_all")}
    return _aoi[c]


def measures(sal, a):
    s = np.clip(sal.astype(np.float64), 0, None); tot = s.sum()
    I, E, EA, D = (s[a[k]].sum() for k in ("inside", "inside_eroded", "inside_eroded_all", "ind"))
    rr = lambda x: float(x / (x + D)) if x + D > 0 else float("nan")
    out = dict(r_inside=gk.rel_inside(sal, a["inside"], a["ind"]), r_eroded=rr(E), r_eroded_all=rr(EA),
               f_inside=float(I / tot), f_eroded=float(E / tot), f_eroded_all=float(EA / tot), f_ind=float(D / tot))
    assert abs(out["r_inside"] - rr(I)) < 1e-12
    return out


def save_map(tag, name, sal):
    mx = float(np.abs(sal).max())
    np.savez_compressed(os.path.join(MAPS, tag, name + ".npz"),
                        sal=(sal / (mx if mx > 0 else 1.0)).astype(np.float16), scale=np.float64(mx))


def read_rgb(name):
    return cv2.cvtColor(cv2.imread(os.path.join(STIM, "img", name + ".png")), cv2.COLOR_BGR2RGB)


def log(msg):
    with open(os.path.join(LOGS, "scores_v3.log"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


SHARD = tuple(int(x) for x in os.environ["SCORE_SHARD"].split("/")) if os.environ.get("SCORE_SHARD") else None


def out_path(t):
    """Main score file, or a shard file when SCORE_SHARD="i/n" is set (images with ITEMS index % n == i)."""
    return os.path.join(RAW, t + (f".shard{SHARD[0]}of{SHARD[1]}" if SHARD else "") + ".json")


def run(tags_fn, tags, label):
    """tags_fn(rgb, it) -> {tag: map}. Resumable: images already scored for every tag are skipped."""
    main_res = {t: (json.load(open(p)) if os.path.exists(p := os.path.join(RAW, t + ".json")) else {}) for t in tags}
    res = {t: (json.load(open(p)) if SHARD and os.path.exists(p := out_path(t)) else main_res[t]) for t in tags}
    for t in tags:
        os.makedirs(os.path.join(MAPS, t), exist_ok=True)
    todo = [it for k, it in enumerate(ITEMS) if (SHARD is None or k % SHARD[1] == SHARD[0])
            and not all(it["name"] in res[t] or it["name"] in main_res[t] for t in tags)]
    label = label + (f" shard {SHARD[0]}/{SHARD[1]}" if SHARD else "")
    log(f"{label}: {len(todo)} of {len(ITEMS)} images to score, tags {tags}")
    t0 = time.time()
    for i, it in enumerate(todo):
        maps = tags_fn(read_rgb(it["name"]), it)
        a = aoi(it)
        for t, sal in maps.items():
            assert sal.shape == (768, 1024) and np.isfinite(sal).all(), (t, it["name"], sal.shape)
            save_map(t, it["name"], sal)
            res[t][it["name"]] = measures(sal, a)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            for t in tags:
                json.dump(res[t], open(out_path(t), "w"))
            el = time.time() - t0
            log(f"  {label} {i + 1}/{len(todo)} {el:.0f}s ({el / (i + 1):.2f} s/img)")
    for t in tags:
        json.dump(res[t], open(out_path(t), "w"))
    log(f"{label}: done, {time.time() - t0:.0f}s")


def merge(tag):
    """Merge shard score files into raw/<tag>.json; the shard files are then moved to <work>/trash/score_shards/."""
    import glob, shutil
    p = os.path.join(RAW, tag + ".json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    shards = sorted(glob.glob(os.path.join(RAW, tag + ".shard*of*.json")))
    for sp in shards:
        for k, v in json.load(open(sp)).items():
            if k in res:
                assert all(abs(res[k][m] - v[m]) < 1e-9 for m in v), (k, res[k], v)
            res[k] = v
    json.dump(res, open(p, "w"))
    trash = os.path.join(str(LP.WORK), "trash", "score_shards"); os.makedirs(trash, exist_ok=True)
    for sp in shards:
        shutil.move(sp, os.path.join(trash, os.path.basename(sp)))
    log(f"merge {tag}: {len(shards)} shard files, {len(res)} images in {p}")


def tag_of(key):
    return NAMES[key].replace(" ", "_")


def main(keys):
    for key in keys:
        if key in ("edge", "sr", "ml"):
            import run_v2
            fn = run_v2.build([key])[NAMES[key]]
            torch.set_num_threads(3)
            if key == "ml":                           # check that the model really sits on the CPU
                import scales
                assert scales.DEV == "cpu" and next(AB.backbone().parameters()).device.type == "cpu"
            run(lambda rgb, it: {tag_of(key): np.asarray(fn(rgb), np.float32)}, [tag_of(key)], NAMES[key])
        elif key in ("aspp", "iie", "msdb"):
            fn = gk.load_models([key])[NAMES[key]]
            torch.set_num_threads(3)                 # load_models sets cpu_count() threads for DeepGaze; undo it
            run(lambda rgb, it: {tag_of(key): np.asarray(fn(rgb), np.float32)}, [tag_of(key)], NAMES[key])
        elif key == "msinet":
            png = os.path.join(SC, "maps_msinet_png")
            missing = [it["name"] for it in ITEMS if not os.path.exists(os.path.join(png, it["name"] + ".png"))]
            assert not missing, f"{len(missing)} MSI-Net maps missing; run run_msinet_v3.py first"
            run(lambda rgb, it: {tag_of(key): cv2.imread(os.path.join(png, it["name"] + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)},
                [tag_of(key)], NAMES[key])
        elif key.startswith("sweep:"):
            bb_name = key.split(":")[1]
            argv = sys.argv; sys.argv = ["sweep.py", bb_name]
            import sweep                              # module-level BB = sys.argv[1]; DEV = AB.DEV = "cpu"
            sys.argv = argv
            assert sweep.DEV == "cpu" and sweep.BB == bb_name
            bb = sweep.load_backbone(); cin = bb.config.hidden_size
            assert next(bb.parameters()).device.type == "cpu"
            R = {}
            for L in PRIMARY_LAYERS[bb_name]:
                m = sweep.Readout(cin)
                m.load_state_dict(torch.load(os.path.join(sweep.WORK, f"readout_L{L}.pt"), map_location="cpu")); m.eval(); m.use_cb = False
                R[L] = m
            tags = [f"{bb_name}_L{L}_s{s}" for s in SIZES for L in R]

            @torch.no_grad()
            def fn(rgb, it):
                out = {}
                for s in SIZES:
                    fs = sweep.feats(bb, sweep.prep(rgb, s)[None])
                    for L, m in R.items():
                        m.use_cb = False
                        out[f"{bb_name}_L{L}_s{s}"] = cv2.resize(torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy(),
                                                                (rgb.shape[1], rgb.shape[0]))
                return out
            run(fn, tags, f"sweep {bb_name}")
        else:
            raise ValueError(key)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2])
    else:
        main(sys.argv[1:])
