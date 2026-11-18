"""
Score the 660 displays of neuro/reanalysis/stim_rigid (480 whole-display rotations, 180 orientation-matched controls) with DeepGaze IIE, DeepGaze MSDB and all 18
layer-sweep readouts (336 and 448 px). Adapted from v3/scores/score_alt.py (model code unchanged: gate_kanizsa
load_models for IIE and MSDB, sweep.py inference with the trained readout_L<L>.pt, sigmoid, cv2.resize to 1024 x 768,
centre bias off); changes:
  - the CPU-only stub of ab_objective (neuro/reanalysis/_stub, no Metal backend query) is imported first, so no
    script imported later can load the original;
  - stimuli and AOIs from stim_rigid (per configuration and angle); scores to neuro/reanalysis/scores_rigid/<tag>.json;
    no maps are saved (disk);
  - all six sweep blocks per backbone (the published job scored the nine primary blocks; nn/score_nn.py the others);
  - the pipeline check (before any rigid display is scored) compares r_inside on 4 stim_v3 displays with
    v2/scores/raw/<tag>.json (primary sweep blocks, IIE, MSDB) or nn/results/raw_v2style/<tag>.json (other blocks);
    it must agree to 1e-4 (REANALYSIS_RULE.md Section 3.2);
  - threads from OMP_NUM_THREADS (default 4); torch.set_num_threads reset after DeepGaze load_models.
Changes of 18 November 2026 (rule review, before the rule was fixed):
  - DeepGaze IIE and MSDB are loaded through the repository copy looking-at-nothing/models/gate_kanizsa.py (LAN_REPO
    overrides the location), whose load_models code for "iie" and "msdb" is identical to the project-root file used
    for the published scores; sys.path is restored after that import, so sweep.py and ab_objective resolve as before;
  - require_public() (common_rigid.py) runs before anything is scored: the rule must be fixed and public and every
    hashed file and stimulus unchanged;
  - the exploratory AOI variants of rule Section 6 (r_N, r_K, f_polygon, f_notch, f_ink; common_rigid.measures_extra)
    are stored with every display, so no map needs to be kept;
  - 660 displays: the 480 rigid rotations and the 180 orientation-matched controls mis<theta>_control180.
Usage (after REANALYSIS_RULE.md is public): nice -n 10 python score_rigid.py <key> [<key> ...]
  keys: iie msdb sweep:dinov2 sweep:clip sweep:siglip
Optional SCORE_SHARD="i/n" (images with index % n == i) and "merge <tag>".
"""
import json, os, sys, time
sys.dont_write_bytecode = True
_T = os.environ.get("OMP_NUM_THREADS", "4")
for k, v in (("OMP_NUM_THREADS", _T), ("MKL_NUM_THREADS", _T), ("VECLIB_MAXIMUM_THREADS", _T),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[k] = v
HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)
STUB = os.path.join(RE, "_stub")
sys.path.insert(0, STUB)
import ab_objective as AB                    # noqa: E402  (the stub: DEV = "cpu", no backend query)
assert AB.__file__.startswith(STUB) and AB.DEV == "cpu", AB.__file__
import numpy as np, cv2, torch               # noqa: E402
torch.set_num_threads(int(_T))
cv2.setNumThreads(1)
sys.path.insert(0, HERE)
import common_rigid as CR                    # noqa: E402
ROOT, V2 = CR.ROOT, CR.V2
LAN_REPO = os.environ.get("LAN_REPO") or os.path.join(ROOT, "looking-at-nothing")
_saved_path = list(sys.path)
sys.path.insert(0, os.path.join(LAN_REPO, "models"))
import gate_kanizsa as gk                    # noqa: E402  (repository copy; iie and msdb loaders as published)
sys.path[:] = _saved_path                    # gate_kanizsa inserts the repository folders; undo, so sweep.py is ROOT's
assert os.path.dirname(os.path.abspath(gk.__file__)) == os.path.join(os.path.abspath(LAN_REPO), "models"), gk.__file__
sys.path.insert(1, ROOT)                     # after the stub, so ab_objective stays the stub

ITEMS = CR.RIGID_ITEMS
RAW, LOGS = CR.SCORES, CR.LOGS
for d in (RAW, LOGS):
    os.makedirs(d, exist_ok=True)
LAYERS = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
PRIMARY = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
SIZES = (336, 448)
NAMES = {"iie": "DeepGaze IIE", "msdb": "DeepGaze MSDB"}
CHECK = ["square_sr50_dark_p0_illusory", "triangle_sr30_light_p2_mis20", "square_sr70_light_p4_control90",
         "triangle_sr70_dark_p1_control180"]
SHARD = tuple(int(x) for x in os.environ["SCORE_SHARD"].split("/")) if os.environ.get("SCORE_SHARD") else None


def log(msg):
    with open(os.path.join(LOGS, "score_rigid.log"), "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def out_path(t):
    return os.path.join(RAW, t + (f".shard{SHARD[0]}of{SHARD[1]}" if SHARD else "") + ".json")


def ref_scores(t):
    p = os.path.join(V2, "scores", "raw", t + ".json")
    if not os.path.exists(p):
        p = os.path.join(ROOT, "nn", "results", "raw_v2style", t + ".json")
    return json.load(open(p))


def check_pipeline(tags_fn, tags, label):
    """Score 4 stim_v3 displays and compare r_inside (float32 map) with the published score files (read only)."""
    ref = {t: ref_scores(t) for t in tags}
    worst = {t: 0.0 for t in tags}
    for nm in CHECK:
        a = np.load(os.path.join(CR.STIM_V3, "img", nm + "_aoi.npz"))
        maps = tags_fn(read_rgb(os.path.join(CR.STIM_V3, "img", nm + ".png")), None)
        for t, sal in maps.items():
            r = gk.rel_inside(sal, a["inside"], a["ind"])
            worst[t] = max(worst[t], abs(r - ref[t][nm]["r_inside"]))
    log(f"{label}: pipeline check on {len(CHECK)} stim_v3 images, max |r_inside - published| per tag: "
        + ", ".join(f"{t} {v:.2e}" for t, v in worst.items()))
    assert max(worst.values()) < 1e-4, worst


def run(tags_fn, tags, label):
    main_res = {t: (json.load(open(p)) if os.path.exists(p := os.path.join(RAW, t + ".json")) else {}) for t in tags}
    res = {t: (json.load(open(p)) if SHARD and os.path.exists(p := out_path(t)) else main_res[t]) for t in tags}
    todo = [it for k, it in enumerate(ITEMS) if (SHARD is None or k % SHARD[1] == SHARD[0])
            and not all(it["name"] in res[t] or it["name"] in main_res[t] for t in tags)]
    label = label + (f" shard {SHARD[0]}/{SHARD[1]}" if SHARD else "")
    log(f"{label}: {len(todo)} of {len(ITEMS)} images to score, tags {tags}")
    t0 = time.time()
    for i, it in enumerate(todo):
        maps = tags_fn(read_rgb(os.path.join(CR.STIM_RIGID, "img", it["name"] + ".png")), it)
        a = CR.aoi(it["config"])
        for t, full in maps.items():
            assert full.shape == (768, 1024) and np.isfinite(full).all(), (t, it["name"], full.shape)
            q = CR.q16(full)
            rec = CR.measures(q, a)
            rec.update(CR.measures_extra(q, a, it))
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
    trash = os.path.join(RE, "_trash", "score_shards"); os.makedirs(trash, exist_ok=True)
    for sp in shards:
        shutil.move(sp, os.path.join(trash, os.path.basename(sp)))
    log(f"merge {tag}: {len(shards)} shard files, {len(res)} images in {p}")


def main(keys):
    for key in keys:
        if key in ("iie", "msdb"):
            fn = gk.load_models([key])[NAMES[key]]
            torch.set_num_threads(int(_T))           # load_models sets cpu_count() threads for DeepGaze; undo it
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
            assert sweep.DEV == "cpu" and sweep.BB == bb_name and sweep.AB is AB
            bb = sweep.load_backbone(); cin = bb.config.hidden_size
            assert next(bb.parameters()).device.type == "cpu"
            R = {}
            for L in LAYERS[bb_name]:
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
                        out[f"{bb_name}_L{L}_s{s}"] = cv2.resize(nat, (rgb.shape[1], rgb.shape[0]))
                return out
            if SHARD is None or SHARD[0] == 0:
                check_pipeline(fn, tags, f"sweep {bb_name}")
            run(fn, tags, f"sweep {bb_name}")
        else:
            raise ValueError(key)


if __name__ == "__main__":
    CR.require_public()                      # REANALYSIS_RULE.md Section 8.1: nothing is scored before the push
    if sys.argv[1] == "merge":
        merge(sys.argv[2])
    else:
        main(sys.argv[1:])
