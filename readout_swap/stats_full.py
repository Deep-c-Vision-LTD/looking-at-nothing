"""
R1n input statistics: per-channel mean and standard deviation of each feature set of one backbone over all training
images x all tokens (grid positions), 336 px, computed once (float32 per batch, combined in float64 with Chan's
parallel update, so no E[x^2] - E[x]^2 cancellation).
Usage: python stats_full.py --bb dinov2|siglip|iie [--out DIR] [--n-train N]
Output: DIR/stats/<fset>_chstats.npz (mean, std, n_tokens, n_images)
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from fullcommon import (make_extractor, DEVICE, S, BB_FSETS, Extractor, Loader, feats, train_items, paths, log, sync, OUT_DEFAULT)

ap = argparse.ArgumentParser()
ap.add_argument("--bb", required=True); ap.add_argument("--out", default=OUT_DEFAULT)
ap.add_argument("--n-train", type=int, default=None); ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
P = paths(a.out)
LOG = os.path.join(P["logs"], f"stats_{a.bb}.log")
fsets = BB_FSETS[a.bb]
outs = {fs: os.path.join(P["stats"], f"{fs}_chstats.npz") for fs in fsets}
if all(os.path.exists(p) for p in outs.values()):
    log(LOG, f"skip {a.bb}: stats exist"); sys.exit(0)
keep, imgs = train_items(a.n_train)
log(LOG, f"=== stats {a.bb} on {DEVICE}: {len(imgs)} training images, bs {a.bs}")
ex = make_extractor(a.bb)
acc = {fs: None for fs in fsets}                                  # (n, mean, M2) float64
batches = [imgs[b:b + a.bs] for b in range(0, len(imgs), a.bs)]
t0 = time.time()
with torch.no_grad():
    for i, x in enumerate(Loader(ex, batches, S)):
        f = feats(ex, x, S)
        for fs in fsets:
            t = f[fs].transpose(0, 1).reshape(f[fs].shape[1], -1)     # C x (B * tokens), float32 on device
            nb = t.shape[1]; mb = t.mean(1); m2b = ((t - mb[:, None]) ** 2).sum(1)
            nb, mb, m2b = float(nb), mb.cpu().double().numpy(), m2b.cpu().double().numpy()
            if acc[fs] is None:
                acc[fs] = (nb, mb, m2b)
            else:
                n, m, m2 = acc[fs]; d = mb - m; nt = n + nb
                acc[fs] = (nt, m + d * nb / nt, m2 + m2b + d * d * n * nb / nt)
        if i % 50 == 0:
            sync(); log(LOG, f"  {a.bb} batch {i + 1}/{len(batches)} {time.time() - t0:.0f}s")
for fs in fsets:
    n, m, m2 = acc[fs]
    sd = np.sqrt(m2 / n)
    np.savez(outs[fs], mean=m.astype(np.float32), std=sd.astype(np.float32), n_tokens=n, n_images=len(imgs))
    log(LOG, f"{fs}: C {len(m)} tokens {n:.0f} mean(|mu|) {np.abs(m).mean():.4f} mean(sd) {sd.mean():.4f} "
             f"min sd {sd.min():.2e} max sd {sd.max():.2f}")
log(LOG, f"=== stats {a.bb} done {time.time() - t0:.0f}s")
