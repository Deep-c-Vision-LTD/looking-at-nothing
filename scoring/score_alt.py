"""
Score the 360 stim_alt images with DeepGaze IIE, DeepGaze MSDB and the nine primary sweep readouts (336 and 448 px).

Model code is that of scoring/score_v3.py (copied from its main(), not changed): gate_kanizsa.load_models for IIE
(full resolution, uniform centre bias) and MSDB (half resolution, 17.5 px/deg, dataset=None); sweep.py inference
(sweep.prep, sweep.feats, sweep.Readout with the trained readout_L<L>.pt, sigmoid, cv2.resize to 1024 x 768, centre
bias off). Everything on the CPU with 3 threads (ab_objective.DEV forced to "cpu" before any model code is imported;
parameter devices asserted).

Before scoring, each run scores a few stim_v3 images and compares r_inside with <results>/v2/scores/raw/<tag>.json
(pipeline identity check; logged, not saved as maps).
Per image: measures of common_alt.measures on the float16-quantised map (common_alt.q16), plus r_inside on the float32
map. Scores: <results>/v3/scores/raw/<tag>.json. Maps (float16, sal / max, with scale = max; not distributed):
  DeepGaze IIE / MSDB: <results>/v3/scores/maps/<tag>/<name>.npz at 768 x 1024 (as v2);
  sweep readouts: <results>/v3/scores/maps/<tag>/<name>.npz at the readout's native s x s output (before the resize to the image;
  the resize is cv2.resize(map, (1024, 768)), INTER_LINEAR), to keep the disk use small.
Usage: python score_alt.py <key> [<key> ...]; keys: iie msdb sweep:dinov2 sweep:clip sweep:siglip
Optional SCORE_SHARD="i/n" (images with index % n == i, into raw/<tag>.shard<i>of<n>.json); python score_alt.py merge <tag>.
"""
import json, os, sys, time
for k, v in (("OMP_NUM_THREADS", "3"), ("MKL_NUM_THREADS", "3"), ("VECLIB_MAXIMUM_THREADS", "3"),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[k] = os.environ.get(k, v) if k.endswith("OFFLINE") else v      # offline by default; =0 to download
import numpy as np, cv2, torch
torch.set_num_threads(3)
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common_alt as CA                     # noqa: E402
V2, V3 = CA.V2, CA.V3
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "models"))
import ab_objective as AB                    # noqa: E402
AB.DEV = "cpu"                               # CPU only, as in the published run
import gate_kanizsa as gk                    # noqa: E402

SC = os.path.join(V3, "scores")
ITEMS = CA.ALT_ITEMS
MAPS, RAW, LOGS = os.path.join(SC, "maps"), os.path.join(SC, "raw"), os.path.join(V3, "logs")
for d in (MAPS, RAW, LOGS):
    os.makedirs(d, exist_ok=True)
PRIMARY_LAYERS = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
SIZES = (336, 448)
NAMES = {"iie": "DeepGaze IIE", "msdb": "DeepGaze MSDB"}
CHECK = ["square_sr50_dark_p0_illusory", "triangle_sr30_light_p2_mis20", "square_sr70_light_p4_control90",
         "triangle_sr70_dark_p1_control180"]
SHARD = tuple(int(x) for x in os.environ["SCORE_SHARD"].split("/")) if os.environ.get("SCORE_SHARD") else None


def log(msg):
    with open(os.path.join(LOGS, "score_alt.log"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def out_path(t):
    return os.path.join(RAW, t + (f".shard{SHARD[0]}of{SHARD[1]}" if SHARD else "") + ".json")


def check_pipeline(tags_fn, tags, label):
    """Score a few stim_v3 images and compare r_inside (float32 map) with the v2 score files."""
    v2raw = {t: CA.LP.load_json(os.path.join(V2, "scores", "raw", t + ".json")) for t in tags}
    worst = {t: 0.0 for t in tags}
    for nm in CHECK:
        a = np.load(os.path.join(CA.STIM_V3, "img", nm + "_aoi.npz"))
        maps = tags_fn(read_rgb(os.path.join(CA.STIM_V3, "img", nm + ".png")), None)
        for t, sal in maps.items():
            r = gk.rel_inside(sal, a["inside"], a["ind"])
            worst[t] = max(worst[t], abs(r - v2raw[t][nm]["r_inside"]))
    log(f"{label}: pipeline check on {len(CHECK)} stim_v3 images, max |r_inside - v2| per tag: "
        + ", ".join(f"{t} {v:.2e}" for t, v in worst.items()))
    assert max(worst.values()) < 1e-4, worst


def run(tags_fn, tags, label, native=False):
    """tags_fn(rgb, it) -> {tag: full-resolution map} or {tag: (full map, native map)} when native=True."""
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
        maps = tags_fn(read_rgb(os.path.join(CA.STIM_ALT, "img", it["name"] + ".png")), it)
        a = CA.aoi(it["config"])
        for t, m in maps.items():
            full, nat = (m if native else (m, m))
            assert full.shape == (768, 1024) and np.isfinite(full).all(), (t, it["name"], full.shape)
            mx = float(np.abs(nat).max())
            np.savez_compressed(os.path.join(MAPS, t, it["name"] + ".npz"),
                                sal=(nat / (mx if mx > 0 else 1.0)).astype(np.float16), scale=np.float64(mx))
            rec = CA.measures(CA.q16(full), a)
            rec["r_inside_f32"] = gk.rel_inside(full, a["inside"], a["ind"])
            res[t][it["name"]] = rec
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            for t in tags:
                json.dump(res[t], open(out_path(t), "w"))
            el = time.time() - t0
            log(f"  {label} {i + 1}/{len(todo)} {el:.0f}s ({el / (i + 1):.2f} s/img)")
    for t in tags:
        json.dump(res[t], open(out_path(t), "w"))
    log(f"{label}: done, {time.time() - t0:.0f}s")


def merge(tag):
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
    trash = os.path.join(str(CA.LP.WORK), "trash", "score_shards"); os.makedirs(trash, exist_ok=True)
    for sp in shards:
        shutil.move(sp, os.path.join(trash, os.path.basename(sp)))
    log(f"merge {tag}: {len(shards)} shard files, {len(res)} images in {p}")


def main(keys):
    for key in keys:
        if key in ("iie", "msdb"):
            fn = gk.load_models([key])[NAMES[key]]
            torch.set_num_threads(3)                 # load_models sets cpu_count() threads for DeepGaze; undo it
            tag = NAMES[key].replace(" ", "_")
            f = lambda rgb, it: {tag: np.asarray(fn(rgb), np.float32)}
            if SHARD is None or SHARD[0] == 0:
                check_pipeline(f, [tag], NAMES[key])
            run(f, [tag], NAMES[key])
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
                        nat = torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy()
                        out[f"{bb_name}_L{L}_s{s}"] = (cv2.resize(nat, (rgb.shape[1], rgb.shape[0])), nat)
                return out
            check_pipeline(lambda rgb, it: {t: v[0] for t, v in fn(rgb, it).items()}, tags, f"sweep {bb_name}")
            run(fn, tags, f"sweep {bb_name}", native=True)
        else:
            raise ValueError(key)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2])
    else:
        main(sys.argv[1:])
