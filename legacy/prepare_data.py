"""
Training set shared by every readout: SALICON train (10,000) + MIT1003 minus its last 200 (803) + CAT2000 train (2,000).
Writes work/meta.json (image paths, fixation counts) and work/maps.npy (fixation maps at 336 x 336, uint8).
Images without recorded fixations are skipped during training.  Usage: python prepare_data.py
"""
import glob, json, os
import numpy as np, cv2, scipy.io as sio
from PIL import Image
from common import HERE, SALICON, CAT2000, build_mit1003

S = 336
WORK = os.path.join(HERE, "work"); os.makedirs(WORK, exist_ok=True)

def items():
    out = []
    for img in sorted(glob.glob(f"{SALICON}/images/train/*.jpg")):
        st = os.path.splitext(os.path.basename(img))[0]
        mp, mt = f"{SALICON}/train/{st}.png", f"{SALICON}/train/{st}.mat"
        if os.path.exists(mp) and os.path.exists(mt):
            out.append(dict(src="salicon", image=img, map=mp, fix=mt))
    for p in build_mit1003()[:-200]:
        out.append(dict(src="mit1003", image=p["image"], map=p["gt"], fix=p["fix"]))
    for img in sorted(glob.glob(f"{CAT2000}/Stimuli/*/*.jpg")):
        cat, st = img.split("/")[-2], os.path.splitext(os.path.basename(img))[0]
        mp, fl = f"{CAT2000}/FIXATIONMAPS/{cat}/{st}.jpg", f"{CAT2000}/FIXATIONLOCS/{cat}/{st}.mat"
        if os.path.exists(mp) and os.path.exists(fl):
            out.append(dict(src="cat2000", image=img, map=mp, fix=fl))
    return out

def n_fixations(it):
    if it["src"] == "salicon":
        return int(sum(o[0][2].shape[0] for o in sio.loadmat(it["fix"])["gaze"] if o[0][2].size))
    if it["src"] == "mit1003":
        return int((np.array(Image.open(it["fix"]).convert("L")) > 128).sum())
    return int((sio.loadmat(it["fix"])["fixLocs"] > 0).sum())

if __name__ == "__main__":
    its = items(); n = len(its)
    print("items", n, {s: sum(i["src"] == s for i in its) for s in ("salicon", "mit1003", "cat2000")})
    maps = np.lib.format.open_memmap(f"{WORK}/maps.npy", "w+", np.uint8, (n, S, S))
    for i, it in enumerate(its):
        maps[i] = cv2.resize(cv2.imread(it["map"], cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_AREA)
    maps.flush()
    json.dump(dict(items=its, nfix=[n_fixations(it) for it in its]), open(f"{WORK}/meta.json", "w"))
