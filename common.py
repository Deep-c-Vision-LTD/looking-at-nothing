"""Shared pieces: paths, the inside-share measure, the ASPP readout, fixation metrics and the MIT1003 split."""
import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# ---- EDIT ME: where the public datasets live (only needed for training and the MIT1003 evaluation) ----
DATA = os.environ.get("SAL_DATA", os.path.join(HERE, "data"))
MIT1003 = os.path.join(DATA, "mit1003")          # ALLSTIMULI/*.jpeg, ALLFIXATIONMAPS/*_fixMap.jpg, *_fixPts.jpg
SALICON = os.path.join(DATA, "salicon")          # images/train/*.jpg, train/*.png (maps), train/*.mat (gaze)
CAT2000 = os.path.join(DATA, "cat2000")          # Stimuli/<Category>/*.jpg, FIXATIONMAPS/..., FIXATIONLOCS/...
WEIGHTS = os.environ.get("SAL_WEIGHTS", os.path.join(HERE, "weights"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
STIM = {"kanizsa": os.path.join(HERE, "stimuli", "kanizsa"), "ehrenstein": os.path.join(HERE, "stimuli", "ehrenstein")}
RESULTS = os.path.join(HERE, "results")


def rel_inside(sal, ins, ind):
    """Share of (non-negative) saliency inside the figure relative to figure + inducers."""
    s = np.clip(sal.astype(np.float64), 0, None)
    a, b = s[ins].sum(), s[ind].sum()
    return float(a / (a + b)) if a + b > 0 else float("nan")


class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.GELU()))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.GELU())
        self.proj = nn.Sequential(nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.GELU(), nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.proj(torch.cat(outs, dim=1))


class Readout(nn.Module):
    """ASPP readout + learned 16x16 centre bias on a frozen token grid (the centre bias is switched off for illusions)."""
    def __init__(self, cin):
        super().__init__()
        self.aspp = ASPPModule(cin, 128, (1, 3, 6, 12)); self.head = nn.Conv2d(128, 1, 1)
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True

    def forward(self, f, out_hw):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)


def loss_map(logits, gt):
    """KL + 2 (1 - CC) between sigmoid(logits) and the fixation map."""
    pred = torch.sigmoid(logits)
    pn = pred / (pred.sum((-2, -1), keepdim=True) + 1e-8); gn = gt / (gt.sum((-2, -1), keepdim=True) + 1e-8)
    kl = (gn * (gn + 1e-8).log() - gn * (pn + 1e-8).log()).sum((-2, -1)).mean()
    p, g = pred.flatten(1), gt.flatten(1)
    pm, gm = p - p.mean(1, keepdim=True), g - g.mean(1, keepdim=True)
    cc = (pm * gm).sum(1) / (pm.norm(dim=1) * gm.norm(dim=1) + 1e-8)
    return kl + 2.0 * (1 - cc.mean())


# ---- metrics (Bylinskii et al., 2019) ----
def CC(sal, gt):
    a = sal - sal.mean(); b = gt - gt.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0

def NSS(sal, fix):
    s = sal.astype(np.float64); s = (s - s.mean()) / (s.std() + 1e-12); f = fix > 0.5
    return float(s[f].mean()) if f.sum() > 0 else np.nan

def AUC_Judd(sal, fix, n_thresh=100):
    x = sal.astype(np.float64); r = x.max() - x.min()
    s = ((x - x.min()) / r if r > 1e-12 else np.zeros_like(x)).ravel(); f = fix.ravel() > 0.5
    if f.sum() == 0:
        return np.nan
    pos = s[f]; th = np.linspace(0, 1, n_thresh)
    tp = np.array([(pos >= t).mean() for t in th]); fp = np.array([(s >= t).mean() for t in th])
    o = np.argsort(fp)
    return float(np.trapezoid(tp[o], fp[o]))


def build_mit1003(root=MIT1003):
    """Sorted MIT1003 list; the last 200 are held out, the first 803 are used for training."""
    stim, fmap = os.path.join(root, "ALLSTIMULI"), os.path.join(root, "ALLFIXATIONMAPS")
    pairs = []
    for img in sorted(glob(os.path.join(stim, "*.jpeg")) + glob(os.path.join(stim, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        gt, pts = os.path.join(fmap, stem + "_fixMap.jpg"), os.path.join(fmap, stem + "_fixPts.jpg")
        if os.path.exists(gt):
            pairs.append({"image": img, "gt": gt, "fix": pts if os.path.exists(pts) else None})
    return pairs
