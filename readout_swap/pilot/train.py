"""
Train one pilot readout on cached features (layer-sweep recipe) and score it on the 200 held-out MIT1003 images
(336 px, centre bias on, as models/sweep.py evaluate()).
Usage: python train.py <fset> <R1|R2|R3|R4> <seed> [--force] [--epochs N]  (N != 6: exploratory, tag <arch>x<N>ep)
Outputs (under <results>/v2/readout): ckpt/<fset>_<arch>_s<seed>.pt, scores/mit_<fset>_<arch>_s<seed>.json; log in <work>/logs
"""
import json, os, sys, time
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AB, CACHE, SCORES, LOGS, DEV, S, FSETS, make_readout, ckpt_path, mit_heldout, CC, NSS, log)


def train(fset, arch, seed, epochs=6, bs=32, lr=5e-4):
    tag = f"{fset}_{arch}_s{seed}"
    LOG = os.path.join(LOGS, f"readout_train_{tag}.log")
    feat = np.load(os.path.join(CACHE, f"{fset}_train.npy"), mmap_mode="r")
    maps = np.load(os.path.join(CACHE, "maps_train.npy"))
    n = len(feat); assert n == len(maps) == 3000
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = make_readout(arch, FSETS[fset]["cin"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log(LOG, f"=== {tag} start {time.strftime('%Y-%m-%d %H:%M:%S')} params {sum(p.numel() for p in m.parameters())}")
    for ep in range(epochs):
        m.train(); order = rng.permutation(n); nb = n // bs; tot, t0 = 0.0, time.time()
        for b in range(nb):
            idx = np.sort(order[b * bs:(b + 1) * bs])
            f = torch.from_numpy(np.asarray(feat[idx], np.float32)).to(DEV)
            gt = torch.from_numpy(maps[idx].astype(np.float32) / 255.0)[:, None].to(DEV)
            loss = AB.loss_A(m(f, (S, S)), gt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        log(LOG, f"{tag} epoch {ep} loss {tot / nb:.4f} {time.time() - t0:.0f}s")
    torch.save(m.state_dict(), ckpt_path(fset, arch, seed))
    return m, LOG


@torch.no_grad()
def score_mit(m, fset, tag, LOG):
    m.eval(); m.use_cb = True
    feat = np.load(os.path.join(CACHE, f"{fset}_mit200.npy"), mmap_mode="r")
    cc, nss = [], []
    for i, p in enumerate(mit_heldout()):
        gt = cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        h, w = gt.shape
        f = torch.from_numpy(np.asarray(feat[i:i + 1], np.float32)).to(DEV)
        sal = cv2.resize(torch.sigmoid(m(f, (S, S)))[0, 0].cpu().numpy(), (w, h))
        fx = (cv2.imread(p["fix"], cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
        cc.append(CC(sal, gt)); nss.append(NSS(sal, fx))
    rec = dict(fset=fset, tag=tag, CC=float(np.mean(cc)), NSS=float(np.mean(nss)), cc_per_image=[round(c, 5) for c in cc])
    json.dump(rec, open(os.path.join(SCORES, f"mit_{tag}.json"), "w"))
    log(LOG, f"{tag} MIT1003 held-out CC {rec['CC']:.4f} NSS {rec['NSS']:.4f}")
    return rec


if __name__ == "__main__":
    fset, arch, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 6
    if epochs != 6:                                   # exploratory, outside the matched recipe
        arch = f"{arch}x{epochs}ep"
    tag = f"{fset}_{arch}_s{seed}"
    if os.path.exists(ckpt_path(fset, arch, seed)) and os.path.exists(os.path.join(SCORES, f"mit_{tag}.json")) and "--force" not in sys.argv:
        print("exists", tag); sys.exit(0)
    lock = ckpt_path(fset, arch, seed)[:-3] + ".lock"           # lets several queues share one work list
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("claimed by another queue", tag); sys.exit(0)
    os.write(fd, f"started {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}\n".encode())
    m, LOG = train(fset, arch, seed, epochs=epochs)
    score_mit(m, fset, tag, LOG)
    os.write(fd, f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode()); os.close(fd)
