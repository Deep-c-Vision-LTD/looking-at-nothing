"""
Shapley decomposition: MSDB "DINOv2-only" and "CLIP-only" lesion variants.
The released MSDB has one saliency network trained on all 2,560 channels; no single-backbone weights exist. The
variants clamp the other backbone's channels of the scale-weighted feature tensor to a per-channel constant (their
mean over all 48 x 64 positions and all 120 images: 60 illusory + 60 control180), so the map depends on the remaining
backbone only. Inputs: cache/msdb_X/<image>.npy (float16, written by run_msdb.py). The unlesioned map is recomputed
from the same cache ("full_fp16") for comparability.
Maps saved: maps/msdb_lesion/<image>.npz (full_fp16, dino_only, clip_only; 384 x 512 density x 384*512, float16).
Usage: python run_lesion.py -> scores/lesion.json
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_d import *  # noqa
from run_msdb import MSDB, GROUPS, XD, to_map, h, w, torch  # noqa: E402

OUT = os.path.join(SCORES, "lesion.json")
MD = os.path.join(MAPS, "msdb_lesion"); os.makedirs(MD, exist_ok=True)


def main():
    cfgs = kanizsa_configs()
    names = [n for c in cfgs for n in (c["ill"], c["ctrl"])]
    acc = np.zeros(2560, np.float64)
    for n in names:
        acc += np.load(os.path.join(XD, n + ".npy")).astype(np.float64).mean((1, 2))
    cmean = torch.from_numpy((acc / len(names)).astype(np.float32))[None, :, None, None]
    clip_sl = GROUPS["CLIP"]; dino_sl = slice(GROUPS["DINOv2_b10"].start, GROUPS["DINOv2_b6"].stop)
    M = MSDB()
    res = {"n_images_for_clamp": len(names), "configs": {}}
    t0 = time.time()
    for c in cfgs:
        aoi = read_aoi(c["ill"])
        rec = {}
        for tag, name in (("I", c["ill"]), ("C", c["ctrl"])):
            X = torch.from_numpy(np.load(os.path.join(XD, name + ".npy")).astype(np.float32))[None]
            Xd = X.clone(); Xd[:, clip_sl] = cmean[:, clip_sl].expand(-1, -1, X.shape[2], X.shape[3])
            Xc = X.clone(); Xc[:, dino_sl] = cmean[:, dino_sl].expand(-1, -1, X.shape[2], X.shape[3])
            maps = {}
            for key, XX in (("full_fp16", X), ("dino_only", Xd), ("clip_only", Xc)):
                d, mp = to_map(M.head(XX)[0])
                rec[f"{key}_{tag}"] = masses(mp, aoi); maps[key] = (d * (h * w)).astype(np.float16)
            np.savez_compressed(os.path.join(MD, name + ".npz"), **maps)
        res["configs"][c["config"]] = rec
    save_json(res, OUT)
    print(f"LESION DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
