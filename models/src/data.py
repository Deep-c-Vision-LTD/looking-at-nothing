"""MIT1003 pair list (sorted; the last 200 images are the held-out set used throughout)."""
import os
from glob import glob


def build_mit1003(root):
    """root/ALLSTIMULI/<name>.jpeg, root/ALLFIXATIONMAPS/<name>_fixMap.jpg (+ _fixPts.jpg)."""
    stim = os.path.join(root, "ALLSTIMULI")
    fmap = os.path.join(root, "ALLFIXATIONMAPS")
    pairs = []
    for img in sorted(glob(os.path.join(stim, "*.jpeg")) + glob(os.path.join(stim, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        gt = os.path.join(fmap, stem + "_fixMap.jpg")
        pts = os.path.join(fmap, stem + "_fixPts.jpg")
        if os.path.exists(gt):
            pairs.append({"image": img, "gt": gt, "fix": pts if os.path.exists(pts) else None})
    return pairs
