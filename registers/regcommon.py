"""
Registers control for the readout swap: shared definitions.

Everything comes read-only from the published readout-swap code in the project tree (v3/gpu/fullcommon.py, which
imports v2/readout/common.py, ab_objective.py and v3/scores/common_alt.py). This module only adds one backbone:

  dinov2reg      facebook/dinov2-with-registers-base (Darcet et al. 2024; 4 register tokens, 86.58 M parameters),
                 same ImageNet mean and std as facebook/dinov2-base
  dinov2reg_L12  hidden_states[12] (output of block 12, before the final layer norm), 768 channels
  dinov2reg_L6   hidden_states[6] (same forward pass), 768 channels

Token layout of the registers model: [CLS, register 1..4, patch tokens]. The plain pipeline drops CLS
(t[:, 1:, :]); here CLS and the 4 registers are dropped (t[:, 1 + 4:, :]) and the remaining patch tokens are
reshaped to the g x g grid exactly as the plain pipeline does. The plain backbones keep their published code path
(fullcommon.feats is used unchanged for them).

No file outside nn/registers is written: byte-code writing is disabled before any project module is imported.
"""
import os, sys
sys.dont_write_bytecode = True
for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[_k] = _v

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.dirname(HERE)
ROOT = os.path.dirname(NN)
GPU = os.path.join(ROOT, "v3", "gpu")
if GPU not in sys.path:
    sys.path.insert(0, GPU)
import fullcommon as FC                                   # noqa: E402  (v3/gpu/fullcommon.py, read-only)
import common as C2                                       # noqa: E402  (v2/readout/common.py, read-only)
import numpy as np, torch                                 # noqa: E402

OUT_DEFAULT = os.path.join(HERE, "full")
PLAIN_OUT = os.path.join(GPU, "full")                     # published readout-swap run (read-only)

# ------------------------------------------------------------------ the added backbone (shared dicts, so every
# imported project module sees it)
FC.BACKBONES["dinov2reg"] = dict(hf="facebook/dinov2-with-registers-base", cls=True, nreg=4,
                                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
FC.FSETS["dinov2reg_L12"] = dict(bb="dinov2reg", layer=12, cin=768)
FC.FSETS["dinov2reg_L6"] = dict(bb="dinov2reg", layer=6, cin=768)
FC.BB_FSETS["dinov2reg"] = ("dinov2reg_L12", "dinov2reg_L6")
assert C2.FSETS is FC.FSETS and C2.BACKBONES is FC.BACKBONES

# re-exported names (train_reg.py / eval_reg.py import these instead of fullcommon's)
from fullcommon import (AB, DEVICE, S, FSETS, BB_FSETS, BACKBONES, CALIB_N, CALIB_SEED, Loader, make_head,  # noqa: E402,F401
                        calibrate_r1n, load_chstats, train_items, train_maps, paths, log, sync, tag_of, read_rgb,
                        mit_heldout, CC, NSS, V3)


class RegExtractor(C2.Extractor):
    """C2.Extractor for the registers backbone: same preprocessing (INTER_AREA resize to s x s, ImageNet
    normalisation), model loaded with AutoModel from the local cache (Dinov2WithRegistersModel)."""

    def __init__(self, bb):
        assert bb == "dinov2reg"
        from transformers import AutoModel
        self.bb = bb
        self.m = AutoModel.from_pretrained(BACKBONES[bb]["hf"]).eval()
        assert self.m.config.num_register_tokens == BACKBONES[bb]["nreg"], self.m.config.num_register_tokens
        self.mean = np.array(BACKBONES[bb]["mean"], np.float32); self.std = np.array(BACKBONES[bb]["std"], np.float32)


def make_extractor(bb):
    """Plain backbones: fullcommon.make_extractor unchanged. Registers backbone: RegExtractor on DEVICE (no float64
    buffers to cast)."""
    if bb != "dinov2reg":
        return FC.make_extractor(bb)
    ex = RegExtractor(bb)
    ex.m = ex.m.to(DEVICE).eval()
    return ex


def prep_batch(ex, img_paths, s=S):
    return torch.stack([ex.prep(read_rgb(p), s) for p in img_paths])


@torch.no_grad()
def feats(ex, x, s):
    """Plain backbones: fullcommon.feats unchanged. Registers backbone: the same code path with CLS and the register
    tokens dropped before the reshape to the patch grid."""
    if ex.bb != "dinov2reg":
        return FC.feats(ex, x, s)
    x = x.to(DEVICE)
    o = ex.m(pixel_values=x, output_hidden_states=True)
    hs = o.hidden_states
    drop = 1 + BACKBONES[ex.bb]["nreg"]
    out = {}
    for fs in BB_FSETS[ex.bb]:
        t = hs[FSETS[fs]["layer"]][:, drop:, :]
        g = int(round(t.shape[1] ** 0.5))
        assert g * g == t.shape[1], t.shape
        out[fs] = t.transpose(1, 2).reshape(t.shape[0], -1, g, g).contiguous()
    return out

Extractor = C2.Extractor                                  # name imported (unused) by the copied scripts
