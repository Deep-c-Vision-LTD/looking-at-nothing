"""
Fix test: does the training objective decide whether a saliency readout "sees" illusory figures?

Two readouts, identical in everything except the loss:
  A (map regression): sigmoid output, KL + 2(1 - CC) against blurred fixation maps
    (the objective of xSoyelNet-ASPP / fv_aspp.pt).
  B (probabilistic):  log-softmax over pixels, negative log-likelihood of the observed fixations
    (the objective of the DeepGaze family).
Shared: frozen DINOv2-base at 336x336 (24x24 tokens), ASPP readout (ch 128) + learned centre bias,
training data SALICON train (10,000) + MIT1003 minus its last 200 (803) + CAT2000 (2,000), 8 epochs,
AdamW, cosine schedule, no checkpoint selection (final epoch for both).
Evaluation: MIT1003 last 200 (held out) and the Kanizsa/Ehrenstein gate stimuli with the centre bias
switched off, including a stricter Kanizsa control with inducers rotated 90 degrees.

Usage: python ab_objective.py cache | train A | train B | eval
"""
import glob, json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F, scipy.io as sio
from PIL import Image

# STUB (neuro/reanalysis/_stub, 18 November 2026): locations come from environment variables with defaults derived
# from this file's place in the project tree, instead of the absolute paths of the project-root file:
#   LAN_PROJECT_ROOT  project root (default: three folders above _stub/, i.e. the folder that holds neuro/)
#   LAN_SAL_SRC       saliency model code with src/ (default: <parent of the project root>/saliency_research)
#   SAL_DATA          SALICON and MIT1003 data (default: $LAN_SAL_SRC/data)
#   CAT2000           CAT2000 training set (default: <parent of the project root>/msi/static)
_ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_SRC = os.environ.get("LAN_SAL_SRC") or os.path.join(os.path.dirname(_ROOT), "saliency_research")
sys.path.insert(0, _SRC)
from src.model_aspp import ASPPModule
from src.data import build_mit1003
from src.metrics import CC, NSS, SIM, AUC_Judd

HERE = _ROOT                                     # STUB: project root (the real file lives there)
WORK = os.path.join(HERE, "ab"); os.makedirs(WORK, exist_ok=True)
S, G = 336, 24                                   # input size, token grid
DEV = "cpu"   # STUB: no Metal backend query at import (the GPU is reserved; neuro/reanalysis CPU runs)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SR = os.environ.get("SAL_DATA") or os.path.join(_SRC, "data")                          # STUB: was an absolute path
CAT = os.environ.get("CAT2000") or os.path.join(os.path.dirname(_ROOT), "msi", "static")   # STUB: was an absolute path

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

@torch.no_grad()
def features(batch):
    out = backbone()(pixel_values=batch.to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]
    return out.transpose(1, 2).reshape(out.shape[0], -1, G, G)

def cache():
    its = items(); n = len(its); print("items", n, {s: sum(i["src"] == s for i in its) for s in ("salicon", "mit1003", "cat2000")})
    F16 = np.lib.format.open_memmap(f"{WORK}/feat.npy", "w+", np.float16, (n, 768, G, G))
    MAPS = np.lib.format.open_memmap(f"{WORK}/maps.npy", "w+", np.uint8, (n, S, S))
    fixes = []
    t0 = time.time()
    for b in range(0, n, 32):
        chunk = its[b:b + 32]; xs = []
        for j, it in enumerate(chunk):
            rgb = cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
            xs.append(prep(rgb))
            mp = cv2.imread(it["map"], cv2.IMREAD_GRAYSCALE)
            MAPS[b + j] = cv2.resize(mp, (S, S), interpolation=cv2.INTER_AREA)
            fixes.append(fix_points(it, h, w).tolist())
        F16[b:b + len(chunk)] = features(torch.stack(xs)).cpu().numpy().astype(np.float16)
        if (b // 32) % 40 == 0:
            print(f"  {b + len(chunk)}/{n}  {time.time() - t0:.0f}s", flush=True)
    F16.flush(); MAPS.flush()
    json.dump(dict(items=its, fixes=fixes), open(f"{WORK}/meta.json", "w"))
    print("cached", flush=True)

# ------------------------------------------------------------------ model
class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp = ASPPModule(768, 128, (1, 3, 6, 12)); self.head = torch.nn.Conv2d(128, 1, 1)
        self.center_bias = torch.nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True
    def forward(self, f, out_hw=(S, S)):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)

def loss_A(logits, gt):
    pred = torch.sigmoid(logits)
    pn = pred / (pred.sum((-2, -1), keepdim=True) + 1e-8); gn = gt / (gt.sum((-2, -1), keepdim=True) + 1e-8)
    kl = (gn * (gn + 1e-8).log() - gn * (pn + 1e-8).log()).sum((-2, -1)).mean()
    p, g = pred.flatten(1), gt.flatten(1)
    pm, gm = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    cc = (pm * gm).sum(1) / (pm.norm(dim=1) * gm.norm(dim=1) + 1e-8)
    return kl + 2.0 * (1 - cc.mean())

def loss_B(logits, fixes):
    logp = F.log_softmax(logits.flatten(1), 1)                    # log density over S*S pixels
    nll = []
    for k, pts in enumerate(fixes):
        idx = torch.as_tensor(pts[:, 0] * S + pts[:, 1], device=logits.device)
        nll.append(-logp[k, idx].mean())
    return torch.stack(nll).mean()

def train(which, epochs=8, bs=32, lr=5e-4):
    meta = json.load(open(f"{WORK}/meta.json")); n = len(meta["items"])
    feat = np.load(f"{WORK}/feat.npy", mmap_mode="r"); maps = np.load(f"{WORK}/maps.npy", mmap_mode="r")
    fixes = [np.array(f, dtype=np.int64).reshape(-1, 2) for f in meta["fixes"]]
    keep = [i for i in range(n) if len(fixes[i]) > 0]
    torch.manual_seed(0); rng = np.random.RandomState(0)
    m = Readout().to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        m.train(); order = rng.permutation(keep); tot, t0 = 0.0, time.time()
        for b in range(0, len(order) - bs + 1, bs):
            idx = np.sort(order[b:b + bs])
            f = torch.from_numpy(np.asarray(feat[idx], np.float32)).to(DEV)
            logits = m(f)
            if which == "A":
                gt = torch.from_numpy(np.asarray(maps[idx], np.float32) / 255.0)[:, None].to(DEV)
                loss = loss_A(logits, gt)
            else:
                loss = loss_B(logits, [fixes[i] for i in idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        print(f"[{which}] epoch {ep} loss {tot / (len(order) // bs):.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save(m.state_dict(), f"{WORK}/readout_{which}.pt")

# ------------------------------------------------------------------ eval
def predict(m, rgb, which, use_cb):
    m.use_cb = use_cb
    h, w = rgb.shape[:2]
    with torch.no_grad():
        logits = m(features(prep(rgb)[None]))
        sal = (torch.sigmoid(logits) if which == "A" else torch.softmax(logits.flatten(1), 1).reshape(logits.shape))[0, 0]
    return cv2.resize(sal.cpu().numpy().astype(np.float32), (w, h))

def evaluate():
    sys.path.insert(0, HERE); import gate_kanizsa as gk
    gitems = json.load(open(os.path.join(HERE, "gate", "items.json")))
    extra = []                                                     # stricter Kanizsa control: inducers rotated 90 deg
    for it in gitems:
        if it["shape"] == "kanizsa" and it["cond"] == "control":
            extra.append(dict(it, cond="control90", name=it["name"].replace("_control", "_control90")))
    out = {}
    for which in ("A", "B"):
        m = Readout().to(DEV); m.load_state_dict(torch.load(f"{WORK}/readout_{which}.pt", map_location=DEV)); m.eval()
        # MIT1003 held-out 200
        r = {"CC": [], "NSS": [], "SIM": [], "AUC": []}
        for p in build_mit1003(f"{SR}/mit1003")[-200:]:
            rgb = cv2.cvtColor(cv2.imread(p["image"]), cv2.COLOR_BGR2RGB)
            sal = predict(m, rgb, which, True)
            gt = cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            fx = (cv2.imread(p["fix"], cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
            r["CC"].append(CC(sal, gt)); r["NSS"].append(NSS(sal, fx)); r["SIM"].append(SIM(sal, gt)); r["AUC"].append(AUC_Judd(sal, fx))
        out[f"{which}_mit1003_heldout"] = {k: round(float(np.mean(v)), 4) for k, v in r.items()}
        # illusion gate, centre bias off
        res = {}
        for it in gitems + extra:
            if it["cond"] == "control90":
                cx, cy = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)][it["pos"]]
                side = (150, 190, 230)[it["size"]]; fg, bg = ((0, 255) if it["pol"] == "dark" else (255, 0))
                img, ins, ind = kanizsa90(gk, cx, cy, side, fg, bg)
                rgb = np.array(img.convert("RGB"))
            else:
                rgb = cv2.cvtColor(cv2.imread(os.path.join(HERE, "gate", "stimuli", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
                a = np.load(os.path.join(HERE, "gate", "stimuli", it["name"] + "_aoi.npz")); ins, ind = a["inside"], a["ind"]
            res[it["name"]] = gk.rel_inside(predict(m, rgb, which, False), ins, ind)
        out[f"{which}_gate"] = res
        for sh, conds in (("kanizsa", ("real", "illusory", "control", "control90")), ("ehrenstein", ("real", "illusory", "control"))):
            means = {c: round(float(np.mean([res[i["name"]] for i in gitems + extra if i["shape"] == sh and i["cond"] == c])), 3) for c in conds}
            print(which, sh, means, flush=True)
        print(which, "MIT1003 held-out", out[f"{which}_mit1003_heldout"], flush=True)
    json.dump(out, open(f"{WORK}/ab_results.json", "w"), indent=1)

def kanizsa90(gk, cx, cy, side, fg, bg):
    """Kanizsa inducers each rotated 90 deg clockwise from the illusory orientation."""
    from PIL import ImageDraw
    W, H, SS = gk.W, gk.H, gk.SS
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    r = 0.3 * side; h = side / 2
    corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    for (x, y), a in zip(corners, [0, 90, 180, 270]):
        start = (a + 90) % 360
        d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], start + 90, start + 360, fill=fg)
    img = img.resize((W, H), Image.LANCZOS)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = np.zeros((H, W), bool)
    for x, y in corners:
        disc |= (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    sq = (np.abs(xx - cx) <= h) & (np.abs(yy - cy) <= h)
    return img, sq & ~disc, disc

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        cache()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        evaluate()
