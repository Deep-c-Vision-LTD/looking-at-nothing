"""
neuro/reanalysis/gpu: shared definitions for the GPU runs G3 and A14 of REANALYSIS_RULE.md Section 5.6, and for CPU
scoring of published heads on new displays. Adapted from nn/registers/regcommon.py (itself built on the published
readout-swap code v3/gpu/fullcommon.py, v2/readout/common.py), all read only. Adds two backbones:

  dinov2reg        facebook/dinov2-with-registers-base (as regcommon; 4 register tokens; CLS + registers dropped)
    dinov2reg_L12  hidden_states[12], 768 channels
    dinov2reg_L6   hidden_states[6] (same forward pass), 768 channels
  dinov2ms         facebook/dinov2-base, the public checkpoint of the readout swap, with new feature sets (A14)
    dinov2ms_L7L11 hidden_states[7] and hidden_states[11] concatenated along channels (7 first), 1,536 channels: the
                   two DINOv2 tensors read by DeepGaze MSDB and by xSoyelNet-ML (blocks 6 and 10 in 0-based numbering)
    dinov2ms_L11   hidden_states[11] alone (same forward pass), 768 channels
Token handling for dinov2ms is the plain pipeline's (CLS dropped, interpolate_pos_encoding=True, reshape to the grid).

Device: V3_DEVICE (default "cpu"). With V3_DEVICE=cpu the CPU-only stub of ab_objective (../_stub, no Metal backend
query) is imported before any project module, so nothing queries the GPU. GPU runs (V3_DEVICE=mps) only after the
GPU-idle check of run_gpu_re.sh and the author's go-ahead (REANALYSIS_RULE.md Section 8).
No file outside neuro/ is written: byte-code writing is disabled before any project module is imported.
Added 18 November 2026 (rule review, before the rule was fixed): require_public() and measures_extra(), defined at the
end of this file; every script of this folder calls require_public() before it trains, scores or analyses.
"""
import os, sys
sys.dont_write_bytecode = True
for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[_k] = _v

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))           # project root
SRC = os.environ.get("LAN_SAL_SRC") or os.path.join(os.path.dirname(ROOT), "saliency_research")   # model code (src/)
if os.environ.get("V3_DEVICE", "cpu") == "cpu":
    for p in (SRC, ROOT, os.path.join(RE, "_stub")):        # stub ends up first; v2/readout/common.py finds SRC and ROOT
        if p in sys.path:                                   # already present, so it does not re-insert them ahead
            sys.path.remove(p)
        sys.path.insert(0, p)
    import ab_objective as _AB                              # noqa: E402
    assert _AB.__file__.startswith(os.path.join(RE, "_stub")) and _AB.DEV == "cpu", _AB.__file__
GPU = os.path.join(ROOT, "v3", "gpu")
if GPU not in sys.path:
    sys.path.insert(1, GPU)
import fullcommon as FC                                   # noqa: E402  (v3/gpu/fullcommon.py, read only)
import common as C2                                       # noqa: E402  (v2/readout/common.py, read only)
import numpy as np, torch                                 # noqa: E402

PLAIN_OUT = os.path.join(GPU, "full")                     # published readout-swap run (read only)
REG_OUT = os.path.join(ROOT, "nn", "registers", "full")   # registers R4 run of 17 Nov 2026 (read only)

MEAN_IN, STD_IN = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FC.BACKBONES["dinov2reg"] = dict(hf="facebook/dinov2-with-registers-base", cls=True, nreg=4, mean=MEAN_IN, std=STD_IN)
FC.FSETS["dinov2reg_L12"] = dict(bb="dinov2reg", layer=12, cin=768)
FC.FSETS["dinov2reg_L6"] = dict(bb="dinov2reg", layer=6, cin=768)
FC.BB_FSETS["dinov2reg"] = ("dinov2reg_L12", "dinov2reg_L6")
FC.BACKBONES["dinov2ms"] = dict(hf="facebook/dinov2-base", cls=True, nreg=0, mean=MEAN_IN, std=STD_IN)
FC.FSETS["dinov2ms_L7L11"] = dict(bb="dinov2ms", layer=(7, 11), cin=1536)
FC.FSETS["dinov2ms_L11"] = dict(bb="dinov2ms", layer=11, cin=768)
FC.BB_FSETS["dinov2ms"] = ("dinov2ms_L7L11", "dinov2ms_L11")
assert C2.FSETS is FC.FSETS and C2.BACKBONES is FC.BACKBONES
NEW_BB = ("dinov2reg", "dinov2ms")

from fullcommon import (AB, DEVICE, S, FSETS, BB_FSETS, BACKBONES, CALIB_N, CALIB_SEED, Loader, make_head,  # noqa: E402,F401
                        calibrate_r1n, load_chstats, train_items, train_maps, paths, log, sync, tag_of, read_rgb,
                        mit_heldout, CC, NSS, V3, ARCHS_ALL, SEEDS)


class ReExtractor(C2.Extractor):
    """C2.Extractor for the added backbones: same preprocessing (INTER_AREA resize to s x s, ImageNet normalisation),
    model loaded with AutoModel from the local cache."""

    def __init__(self, bb):
        assert bb in NEW_BB, bb
        from transformers import AutoModel
        self.bb = bb
        self.m = AutoModel.from_pretrained(BACKBONES[bb]["hf"]).eval()
        if BACKBONES[bb]["nreg"]:
            assert self.m.config.num_register_tokens == BACKBONES[bb]["nreg"], self.m.config.num_register_tokens
        self.mean = np.array(BACKBONES[bb]["mean"], np.float32); self.std = np.array(BACKBONES[bb]["std"], np.float32)


def make_extractor(bb):
    if bb not in NEW_BB:
        return FC.make_extractor(bb)
    ex = ReExtractor(bb)
    ex.m = ex.m.to(DEVICE).eval()
    return ex


def prep_batch(ex, img_paths, s=S):
    return torch.stack([ex.prep(read_rgb(p), s) for p in img_paths])


def _grid(t, drop):
    t = t[:, drop:, :]
    g = int(round(t.shape[1] ** 0.5))
    assert g * g == t.shape[1], t.shape
    return t.transpose(1, 2).reshape(t.shape[0], -1, g, g).contiguous()


@torch.no_grad()
def feats(ex, x, s):
    """Plain backbones: fullcommon.feats unchanged. dinov2reg: regcommon's path (CLS and 4 registers dropped).
    dinov2ms: the plain DINOv2 path, with tuple layers concatenated along channels."""
    if ex.bb not in NEW_BB:
        return FC.feats(ex, x, s)
    x = x.to(DEVICE)
    if ex.bb == "dinov2reg":
        o = ex.m(pixel_values=x, output_hidden_states=True)
    else:
        o = ex.m(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True)
    hs = o.hidden_states
    drop = 1 + BACKBONES[ex.bb]["nreg"]
    out = {}
    for fs in BB_FSETS[ex.bb]:
        L = FSETS[fs]["layer"]
        out[fs] = torch.cat([_grid(hs[k], drop) for k in L], 1) if isinstance(L, tuple) else _grid(hs[L], drop)
        assert out[fs].shape[1] == FSETS[fs]["cin"], (fs, out[fs].shape)
    return out


Extractor = C2.Extractor                                  # name imported (unused) by the copied scripts


def _common_rigid():
    """scoring/common_rigid.py, imported lazily (appended to sys.path, so it shadows nothing)."""
    sc = os.path.join(RE, "scoring")
    if sc not in sys.path:
        sys.path.append(sc)
    import common_rigid
    return common_rigid


def require_public():
    """REANALYSIS_RULE.md Section 8.1 (added 18 November 2026 after the rule review): every script of this folder
    calls this before it trains, scores or analyses anything. Refuses unless the rule is fixed and public and every
    hashed file and every stimulus file is unchanged (scoring/common_rigid.require_public)."""
    return _common_rigid().require_public(RE)


def measures_extra(sal, a, it):
    """Exploratory AOI variants of rule Section 6 for the stim_rigid displays (scoring/common_rigid.measures_extra)."""
    return _common_rigid().measures_extra(sal, a, it)
