"""
Score the pilot readouts on a stimulus set from cached features (centre bias off, as models/sweep.py).
For every readout and input size (336, 448): sigmoid map at s x s -> saved as float16 (maps/<stimset>/...npz);
the float16 map is cast back to float32, resized to the 1024 x 768 canvas (cv2 bilinear, as sweep.py) and scored
r = mass(inside) / (mass(inside) + mass(inducers)) for every inside-AOI variant in the stimulus AOI file.
Usage: python evaluate.py <fset> <stimset> [tag ...] [--force]
  tags: <arch>_s<seed> (e.g. R4_s0) or R4_ssweep (the original sweep readout); default = every checkpoint present.
Output: scores/<stimset>_<fset>_<tag>.json = {"<px>": {"<aoi variant>": {name: r}}}
"""
import glob, json, os, sys, time
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, CKPT, MAPS, SCORES, LOGS, DEV, FSETS, load_readout, stim_items, gk, log)

SIZES = (336, 448)
W, H = 1024, 768


def aoi_index(items):
    """Flat pixel indices of each AOI, shared by all conditions of a configuration (verified)."""
    cache, out = {}, {}
    for it in items:
        key = it.get("config") or it["name"].rsplit("_", 1)[0]
        a = np.load(it["aoi"])
        if key not in cache:
            cache[key] = {k: np.flatnonzero(a[k].ravel()) for k in a.files}
        else:                                              # AOIs must be identical across a configuration
            for k in a.files:
                assert np.array_equal(cache[key][k], np.flatnonzero(a[k].ravel())), (it["name"], k)
        out[it["name"]] = cache[key]
    return out


def rel(sal_flat, ins, ind):
    a, b = float(sal_flat[ins].astype(np.float64).sum()), float(sal_flat[ind].astype(np.float64).sum())
    return a / (a + b) if a + b > 0 else float("nan")


@torch.no_grad()
def run(fset, stimset, tags, force=False):
    LOG = os.path.join(LOGS, f"readout_eval_{stimset}_{fset}.log")
    items = stim_items(stimset)
    names = json.load(open(os.path.join(CACHE, "stim", f"{stimset}_names.json")))
    assert names == [it["name"] for it in items]
    aoi = aoi_index(items)
    variants = [k for k in next(iter(aoi.values())) if k != "ind"]
    os.makedirs(os.path.join(MAPS, stimset), exist_ok=True)
    feats = {s: np.load(os.path.join(CACHE, "stim", f"{stimset}_{fset}_s{s}.npy"), mmap_mode="r") for s in SIZES}
    for tag in tags:
        out_p = os.path.join(SCORES, f"{stimset}_{fset}_{tag}.json")
        if os.path.exists(out_p) and not force:
            continue
        arch, seed = tag.split("_s")
        m = load_readout(fset, arch, seed if seed == "sweep" else int(seed)); m.use_cb = False
        res, t0 = {}, time.time()
        for s in SIZES:
            F = feats[s]; mp = np.zeros((len(names), s, s), np.float16)
            for b in range(0, len(names), 16):
                f = torch.from_numpy(np.asarray(F[b:b + 16], np.float32)).to(DEV)
                mp[b:b + 16] = torch.sigmoid(m(f, (s, s)))[:, 0].cpu().numpy().astype(np.float16)
            np.savez_compressed(os.path.join(MAPS, stimset, f"{fset}_{tag}_px{s}.npz"), maps=mp, names=np.array(names))
            r = {v: {} for v in variants}
            for i, n in enumerate(names):
                sal = cv2.resize(mp[i].astype(np.float32), (W, H)).ravel()
                for v in variants:
                    r[v][n] = rel(sal, aoi[n][v], aoi[n]["ind"])
            res[str(s)] = r
        json.dump(res, open(out_p, "w"))
        log(LOG, f"{stimset} {fset} {tag} scored {len(names)} images x {len(SIZES)} sizes {time.time() - t0:.0f}s")


if __name__ == "__main__":
    fset, stimset = sys.argv[1], sys.argv[2]
    tags = [a for a in sys.argv[3:] if not a.startswith("--")]
    if not tags:
        tags = sorted(os.path.basename(p)[len(fset) + 1:-3] for p in glob.glob(os.path.join(CKPT, f"{fset}_R*_s*.pt")))
    run(fset, stimset, tags, "--force" in sys.argv)
