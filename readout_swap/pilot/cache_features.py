"""
Cache float16 features for the readout-swap pilot (CPU).
Usage: python cache_features.py <dinov2|siglip|iie> train|stim_v2|stim_v3|maps
  train   : the 3,000-image training subset and the 200 held-out MIT1003 images at 336 px
  stim_*  : every stimulus of the set at 336 and 448 px
  maps    : copy the subset's 336 x 336 fixation maps from the training cache maps.npy (uint8)
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from common import AB, CACHE, LOGS, BB_FSETS, FSETS, Extractor, read_rgb, train_subset, mit_heldout, stim_items, log

bb, what = sys.argv[1], sys.argv[2]
LOG = os.path.join(LOGS, f"readout_cache_{bb}_{what}.log")
BS = 8


def run(paths, tag, sizes):
    ex = None
    for s in sizes:
        outs = {fs: os.path.join(CACHE, tag.format(fs=fs, s=s)) for fs in BB_FSETS[bb]}
        if all(os.path.exists(p + ".done") for p in outs.values()):
            log(LOG, f"skip {tag} s{s} (done)"); continue
        ex = ex or Extractor(bb)
        probe = ex([read_rgb(paths[0])], s)
        mm = {fs: np.lib.format.open_memmap(p, "w+", np.float16, (len(paths),) + tuple(probe[fs].shape[1:]))
              for fs, p in outs.items()}
        t0 = time.time()
        for b in range(0, len(paths), BS):
            f = ex([read_rgb(p) for p in paths[b:b + BS]], s)
            for fs in mm:
                mm[fs][b:b + BS] = f[fs].numpy().astype(np.float16)
            if (b // BS) % 25 == 0:
                log(LOG, f"  {tag} s{s} {b + BS}/{len(paths)} {time.time() - t0:.0f}s")
        for fs, p in outs.items():
            mm[fs].flush(); del mm[fs]
            open(p + ".done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(LOG, f"done {tag} s{s} shape {probe[BB_FSETS[bb][0]].shape} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    log(LOG, f"=== start {bb} {what} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if what == "maps":
        sub = train_subset()
        maps = np.load(os.path.join(AB.WORK, "maps.npy"), mmap_mode="r")
        np.save(os.path.join(CACHE, "maps_train.npy"), np.asarray(maps[sub["idx"]]))
        log(LOG, f"maps_train.npy {len(sub['idx'])}")
    elif what == "train":
        sub = train_subset()
        run(sub["images"], "{fs}_train.npy", (336,))
        run([p["image"] for p in mit_heldout()], "{fs}_mit200.npy", (336,))
    else:
        items = stim_items(what)
        names = [it["name"] for it in items]
        json.dump(names, open(os.path.join(CACHE, "stim", f"{what}_names.json"), "w"))
        run([it["img"] for it in items], "stim/" + what + "_{fs}_s{s}.npy", (336, 448))
    log(LOG, f"=== end {bb} {what} {time.strftime('%Y-%m-%d %H:%M:%S')}")
