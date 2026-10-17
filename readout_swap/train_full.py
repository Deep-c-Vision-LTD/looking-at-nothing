"""
Full-data readout training (readout swap). One job = one backbone x one seed x a list of readouts, all heads
trained jointly from one frozen forward pass per batch (as models/sweep.py), features on the fly.
Recipe per head = the layer-sweep recipe (models/sweep.py): loss KL + 2(1 - CC) at 336 x 336, epochs 6, batch 32 (last incomplete batch
dropped), AdamW lr 5e-4 wd 1e-4, cosine (T_max = epochs), final epoch, no augmentation.
Seed s: torch.manual_seed(s) immediately before each head is built (so each head gets the initial weights a
single-model pilot run with seed s would get) and numpy RandomState(s) for the batch order (one permutation per epoch).
Resumable: full state saved after every epoch.
Usage: python train_full.py --bb dinov2 --seed 0 --archs R1n,R1,R4 [--out DIR] [--fsets a,b] [--epochs 6]
       [--n-train N (dry run)] [--bs 32] [--max-batches K (benchmark)]
Outputs: DIR/ckpt/<fset>_<arch>_s<seed>.pt (+ .json training record), DIR/done/train_<fset>_<arch>_s<seed>
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from fullcommon import (make_extractor, AB, DEVICE, S, FSETS, BB_FSETS, CALIB_N, CALIB_SEED, Extractor, Loader, prep_batch, feats,
                        make_head, calibrate_r1n, load_chstats, train_items, train_maps, paths, log, sync, tag_of, OUT_DEFAULT)

ap = argparse.ArgumentParser()
ap.add_argument("--bb", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--archs", required=True); ap.add_argument("--out", default=OUT_DEFAULT)
ap.add_argument("--fsets", default=None); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--n-train", type=int, default=None); ap.add_argument("--max-batches", type=int, default=None)
ap.add_argument("--stop-after-epoch", type=int, default=None)          # dry run only: simulate an interruption
a = ap.parse_args()
P = paths(a.out)
fsets = a.fsets.split(",") if a.fsets else list(BB_FSETS[a.bb])
archs = a.archs.split(",")
assert all(FSETS[fs]["bb"] == a.bb for fs in fsets)
JOB = f"{a.bb}_{'-'.join(archs)}_s{a.seed}"
LOG = os.path.join(P["logs"], f"train_{JOB}.log")
STATE = os.path.join(P["state"], f"{JOB}.pt")
heads_spec = [(fs, ar) for fs in fsets for ar in archs]
todo = [(fs, ar) for fs, ar in heads_spec if not os.path.exists(os.path.join(P["done"], f"train_{tag_of(fs, ar, a.seed)}"))]
if not todo:
    log(LOG, f"skip {JOB}: all heads done"); sys.exit(0)
heads_spec = todo

keep, imgs = train_items(a.n_train)
maps = train_maps()
n = len(keep); nb = n // a.bs
if a.max_batches:
    nb = min(nb, a.max_batches)
log(LOG, f"=== {JOB} on {DEVICE}: {n} training images, {nb} batches/epoch, heads {heads_spec}")
ex = make_extractor(a.bb)


class Head:
    pass


def build():
    hs = []
    for fs, ar in heads_spec:
        h = Head(); h.fset, h.arch, h.tag = fs, ar, tag_of(fs, ar, a.seed)
        torch.manual_seed(a.seed)
        h.m = make_head(ar, FSETS[fs]["cin"]).to(DEVICE)
        h.opt = torch.optim.AdamW(h.m.parameters(), lr=a.lr, weight_decay=1e-4)
        h.sched = torch.optim.lr_scheduler.CosineAnnealingLR(h.opt, T_max=a.epochs)
        h.hist = []
        hs.append(h)
    return hs


heads = build()
start_ep = 0
if os.path.exists(STATE):
    st = torch.load(STATE, map_location="cpu", weights_only=False)
    assert st["heads"] == [h.tag for h in heads] and st["n"] == n and st["bs"] == a.bs, "state does not match job"
    for h, sd in zip(heads, st["sd"]):
        h.m.load_state_dict(sd["m"]); h.opt.load_state_dict(sd["opt"]); h.sched.load_state_dict(sd["sched"]); h.hist = sd["hist"]
    start_ep = st["epoch"] + 1
    torch.set_rng_state(st["rng_cpu"])
    if DEVICE == "mps" and st.get("rng_mps") is not None:
        torch.mps.set_rng_state(st["rng_mps"])
    log(LOG, f"resumed from epoch {st['epoch']}")
else:
    r1n = [h for h in heads if h.arch == "R1n"]
    if r1n:                                                     # R1n: fixed normalisation statistics, set once
        ci = sorted(np.random.RandomState(CALIB_SEED).choice(n, min(CALIB_N, n), replace=False).tolist())
        cf = {fs: [] for fs in {h.fset for h in r1n}}
        for b in range(0, len(ci), 32):
            f = feats(ex, prep_batch(ex, [imgs[i] for i in ci[b:b + 32]], S), S)
            for fs in cf:
                cf[fs].append(f[fs].cpu())
        cf = {fs: torch.cat(v) for fs, v in cf.items()}
        for h in r1n:
            log(LOG, f"R1n calibration {h.tag}: input stats from {h.fset}_chstats.npz, hidden layers from {len(ci)} images")
            calibrate_r1n(h.m, cf[h.fset], load_chstats(P, h.fset), LOG)
        del cf
    torch.manual_seed(a.seed)                                   # dropout stream starts from the seed

rng = np.random.RandomState(a.seed)
for ep in range(start_ep):                                      # replay the batch-order stream on resume
    rng.permutation(n)

for ep in range(start_ep, a.epochs):
    order = rng.permutation(n)
    bidx = [np.sort(order[b * a.bs:(b + 1) * a.bs]) for b in range(nb)]

    def load(idx):
        x = prep_batch(ex, [imgs[i] for i in idx], S)
        gt = torch.from_numpy(np.asarray(maps[[keep[i] for i in idx]], np.float32) / 255.0)[:, None]
        return x, gt

    for h in heads:
        h.m.train(); h.tot = torch.zeros((), device=DEVICE)
    t0 = time.time()
    for b, (x, gt) in enumerate(Loader(None, bidx, fn=load)):
        f = feats(ex, x, S)
        gt = gt.to(DEVICE)
        for h in heads:
            loss = AB.loss_A(h.m(f[h.fset], (S, S)), gt)
            h.opt.zero_grad(set_to_none=True); loss.backward(); h.opt.step()
            h.tot += loss.detach()
        if b % 50 == 0 or b == nb - 1:
            sync(); log(LOG, f"  {JOB} ep {ep} batch {b + 1}/{nb} {time.time() - t0:.1f}s")
    for h in heads:
        h.sched.step()
        h.hist.append(dict(epoch=ep, loss=round(h.tot.item() / nb, 5)))
    sync(); dt = time.time() - t0
    log(LOG, f"{JOB} epoch {ep} " + " ".join(f"{h.fset}:{h.arch} {h.hist[-1]['loss']:.4f}" for h in heads) + f" {dt:.0f}s")
    st = dict(heads=[h.tag for h in heads], n=n, bs=a.bs, epoch=ep, rng_cpu=torch.get_rng_state(),
              rng_mps=torch.mps.get_rng_state() if DEVICE == "mps" else None,
              sd=[dict(m={k: v.cpu() for k, v in h.m.state_dict().items()}, opt=h.opt.state_dict(),
                       sched=h.sched.state_dict(), hist=h.hist) for h in heads])
    torch.save(st, STATE + ".tmp"); os.replace(STATE + ".tmp", STATE)
    if a.stop_after_epoch is not None and ep == a.stop_after_epoch:
        log(LOG, f"stopping after epoch {ep} (--stop-after-epoch)"); sys.exit(3)

for h in heads:
    p = os.path.join(P["ckpt"], f"{h.tag}.pt")
    torch.save({k: v.cpu() for k, v in h.m.state_dict().items()}, p)
    json.dump(dict(tag=h.tag, fset=h.fset, arch=h.arch, seed=a.seed, job=JOB, device=DEVICE, n_train=n, batches=nb,
                   epochs=a.epochs, bs=a.bs, lr=a.lr, hist=h.hist, finished=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(p[:-3] + ".json", "w"), indent=1)
    open(os.path.join(P["done"], f"train_{h.tag}"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
log(LOG, f"=== {JOB} done")
