"""
Standard saliency metrics (numpy). Implementations follow Bylinskii et al. 2019.
All take float arrays; `sal` is a predicted saliency map, `gt` a continuous
fixation-density map, `fix` a binary fixation map (1 at fixated pixels).
"""
import numpy as np


def _norm01(x):
    x = x.astype(np.float64)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)


def CC(sal, gt):
    a = (sal - sal.mean()); b = (gt - gt.mean())
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def NSS(sal, fix):
    s = sal.astype(np.float64)
    s = (s - s.mean()) / (s.std() + 1e-12)
    f = fix > 0.5
    return float(s[f].mean()) if f.sum() > 0 else np.nan


def SIM(sal, gt):
    s = sal / (sal.sum() + 1e-12)
    g = gt / (gt.sum() + 1e-12)
    return float(np.minimum(s, g).sum())


def AUC_Judd(sal, fix, n_thresh=100):
    s = _norm01(sal).ravel(); f = fix.ravel() > 0.5
    if f.sum() == 0:
        return np.nan
    pos = s[f]
    th = np.linspace(0, 1, n_thresh)
    tp = np.array([(pos >= t).mean() for t in th])
    fp = np.array([(s >= t).mean() for t in th])
    order = np.argsort(fp)
    return float(np.trapezoid(tp[order], fp[order]))
