"""
Training set, frozen DINOv2 backbone and training loss shared by every readout trained in this study (layer sweep,
xSoyelNet-ML recipe, readout swap).

Training data: SALICON train (10,000) + MIT1003 sorted, minus its last 200 (803) + CAT2000 train (2,000) = 12,803
images; the last 200 MIT1003 images are the held-out set. `python ab_objective.py cache` writes the list of training
items with their fixation points (meta.json) and the fixation maps at 336 x 336 (maps.npy, uint8) to
paths.TRAIN_CACHE; every training script reads these two files.
Loss (loss_A): sigmoid output, KL + 2 (1 - CC) against the fixation map.

Data locations: SAL_DATA/mit1003, SAL_DATA/salicon and CAT2000 (see paths.py).
Usage: python ab_objective.py cache
"""
import glob, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F, scipy.io as sio
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import paths as LP                                # noqa: E402
from src.model_aspp import ASPPModule             # noqa: E402,F401
from src.data import build_mit1003                # noqa: E402
from src.metrics import CC, NSS, SIM, AUC_Judd    # noqa: E402,F401

WORK = str(LP.TRAIN_CACHE); os.makedirs(WORK, exist_ok=True)
S, G = 336, 24                                   # input size, token grid
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SR = str(LP.DATA)                                # <SR>/salicon, <SR>/mit1003
CAT = str(LP.CAT2000)


# ------------------------------------------------------------------ data
def items():
    out = []
    for img in sorted(glob.glob(f"{SR}/salicon/images/train/*.jpg")):
        st = os.path.splitext(os.path.basename(img))[0]
        mp, mt = f"{SR}/salicon/train/{st}.png", f"{SR}/salicon/train/{st}.mat"
        if os.path.exists(mp) and os.path.exists(mt):
            out.append(dict(src="salicon", image=img, map=mp, fix=mt))
    for p in build_mit1003(f"{SR}/mit1003")[:-200]:
        out.append(dict(src="mit1003", image=p["image"], map=p["gt"], fix=p["fix"]))
    for img in sorted(glob.glob(f"{CAT}/Stimuli/*/*.jpg")):
        cat, st = img.split("/")[-2], os.path.splitext(os.path.basename(img))[0]
        mp, fl = f"{CAT}/FIXATIONMAPS/{cat}/{st}.jpg", f"{CAT}/FIXATIONLOCS/{cat}/{st}.mat"
        if os.path.exists(mp) and os.path.exists(fl):
            out.append(dict(src="cat2000", image=img, map=mp, fix=fl))
    return out


def fix_points(it, h, w):
    """Fixations as (row, col) on the S x S grid."""
    if it["src"] == "salicon":
        g = sio.loadmat(it["fix"])["gaze"]
        pts = np.vstack([o[0][2] for o in g if o[0][2].size]) - 1          # (x, y), 1-indexed
        xs, ys = pts[:, 0], pts[:, 1]
    elif it["src"] == "mit1003":
        m = np.array(Image.open(it["fix"]).convert("L")) > 128; ys, xs = np.nonzero(m)
    else:
        m = sio.loadmat(it["fix"])["fixLocs"] > 0; ys, xs = np.nonzero(m)
    r = np.clip((ys + 0.5) * S / h, 0, S - 1).astype(np.int32)
    c = np.clip((xs + 0.5) * S / w, 0, S - 1).astype(np.int32)
    return np.stack([r, c], 1)


def prep(rgb):
    x = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))


_bb = None
def backbone():
    global _bb
    if _bb is None:
        from transformers import AutoModel
        _bb = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    return _bb


def cache():
    """meta.json (items and fixation points) and maps.npy (fixation maps at S x S, uint8) for every training item.
    The original run also cached last-layer DINOv2 features (feat.npy) for an experiment that is not part of the
    paper; no script here reads them, so they are not written."""
    its = items(); n = len(its); print("items", n, {s: sum(i["src"] == s for i in its) for s in ("salicon", "mit1003", "cat2000")})
    MAPS = np.lib.format.open_memmap(f"{WORK}/maps.npy", "w+", np.uint8, (n, S, S))
    fixes = []
    t0 = time.time()
    for b in range(0, n, 32):
        chunk = its[b:b + 32]
        for j, it in enumerate(chunk):
            rgb = cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
            mp = cv2.imread(it["map"], cv2.IMREAD_GRAYSCALE)
            MAPS[b + j] = cv2.resize(mp, (S, S), interpolation=cv2.INTER_AREA)
            fixes.append(fix_points(it, h, w).tolist())
        if (b // 32) % 40 == 0:
            print(f"  {b + len(chunk)}/{n}  {time.time() - t0:.0f}s", flush=True)
    MAPS.flush()
    json.dump(dict(items=its, fixes=fixes), open(f"{WORK}/meta.json", "w"))
    print("cached", flush=True)


# ------------------------------------------------------------------ loss
def loss_A(logits, gt):
    pred = torch.sigmoid(logits)
    pn = pred / (pred.sum((-2, -1), keepdim=True) + 1e-8); gn = gt / (gt.sum((-2, -1), keepdim=True) + 1e-8)
    kl = (gn * (gn + 1e-8).log() - gn * (pn + 1e-8).log()).sum((-2, -1)).mean()
    p, g = pred.flatten(1), gt.flatten(1)
    pm, gm = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    cc = (pm * gm).sum(1) / (pm.norm(dim=1) * gm.norm(dim=1) + 1e-8)
    return kl + 2.0 * (1 - cc.mean())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        cache()
    else:
        raise SystemExit("usage: python ab_objective.py cache")
