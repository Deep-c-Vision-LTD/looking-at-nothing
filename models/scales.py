"""
Inference of the middle-layer readouts at a chosen input size (used for xSoyelNet-ML's test-time averaging over
280/336/392/448 px and horizontal flips; see run_v2.build and decomposition/run_ml.py).
sal_at(m, idx, rgb, s, use_cb, flip): sigmoid map of readout m on the concatenated DINOv2 hidden_states[idx] of the
image squashed to s x s (cv2.INTER_AREA), resized back to the image (cv2 bilinear), un-flipped.
"""
import json, os, sys
import numpy as np, cv2, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_objective as AB, ab_layers as AL  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__)); DEV = AB.DEV
MEAN, STD = AB.MEAN, AB.STD


def prep(rgb, s):
    x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))[None]


@torch.no_grad()
def feats(x, idx):
    hs = AB.backbone()(pixel_values=x.to(DEV), interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
    out = []
    for i in idx:
        t = hs[i][:, 1:, :]; g = int(round(t.shape[1] ** 0.5))
        out.append(t.transpose(1, 2).reshape(1, -1, g, g).contiguous())
    return torch.cat(out, 1)


@torch.no_grad()
def sal_at(m, idx, rgb, s, use_cb, flip=False):
    img = rgb[:, ::-1].copy() if flip else rgb
    m.use_cb = use_cb
    out = torch.sigmoid(m(feats(prep(img, s), idx), out_hw=(s, s)))[0, 0].cpu().numpy()
    out = cv2.resize(out.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    return out[:, ::-1] if flip else out
