"""
Saliency models scored in the paper. Every function maps an RGB uint8 image (H x W x 3) to a non-negative
H x W saliency map. No model receives a centre-bias prior, because figure position is varied.

  DeepGaze IIE    (Linardos et al., 2021)   full-resolution input, uniform centre bias, predicted density
  DeepGaze MSDB   (Kummerer et al., 2025)   dataset-averaged parameters, half-resolution input at 17.5 px/deg
  xSoyelNet-ASPP  last layer of frozen DINOv2-B + ASPP readout, 448 px letterbox, centre bias off
  xSoyelNet-ML    blocks 6 + 10 of frozen DINOv2-B + ASPP readout, TTA over 280/336/392/448 px and flips
  Spectral residual (Hou & Zhang, 2007), own implementation
  Edge-energy null  Sobel magnitude blurred with sigma = 25 px
MSI-Net runs under TensorFlow; see msinet_maps.py.
The xSoyelNet readout weights are not distributed (they belong to a commercial product); train_ml.py reproduces
the xSoyelNet-ML recipe, and the layer readouts used in the paper's mechanism analysis are released.
"""
import math, os
import numpy as np, cv2, torch
from common import Readout, ASPPModule, DEV, WEIGHTS

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32); IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def deepgaze_iie():
    import deepgaze_pytorch
    m = deepgaze_pytorch.DeepGazeIIE(pretrained=True); m.eval()
    def f(rgb):
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None]
        cb = torch.full((1, h, w), -math.log(h * w))
        with torch.no_grad():
            return np.squeeze(torch.exp(m(x, cb)).numpy())
    return f


def deepgaze_msdb(scale=0.5, ppd=35.0):
    import deepgaze_pytorch
    m = deepgaze_pytorch.DeepGazeMSDB(pretrained=True); m.eval()   # do not chain .eval() on the constructor
    def f(rgb):
        H, W = rgb.shape[:2]
        sm = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(sm.astype(np.float32).transpose(2, 0, 1))[None]
        h, w = sm.shape[:2]; cb = torch.full((1, h, w), -math.log(h * w))
        with torch.no_grad():
            d = np.squeeze(torch.exp(m(x, cb, pixel_per_dva=ppd * scale, dataset=None)).numpy())
        return cv2.resize(d, (W, H))
    return f


def _dinov2():
    from transformers import AutoModel
    return AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()


def xsoyelnet_aspp(path=os.path.join(WEIGHTS, "xsoyelnet_aspp.pt"), size=448):
    """Readout on DINOv2-B last_hidden_state (with the final layer norm), letterboxed to a square."""
    bb = _dinov2()
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aspp = ASPPModule(768, 128, (1, 3, 6, 12)); self.head = torch.nn.Conv2d(128, 1, 1)
            self.center_bias = torch.nn.Parameter(torch.zeros(1, 1, 16, 16))
    m = M().to(DEV)
    sd = torch.load(path, map_location=DEV, weights_only=False); sd = sd.get("model", sd)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("backbone.")}); m.eval()
    def f(rgb):
        h, w = rgb.shape[:2]; s = size / max(h, w); nh, nw = round(h * s), round(w * s)
        r = cv2.resize(rgb, (nw, nh)); top, left = (size - nh) // 2, (size - nw) // 2
        p = cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left, cv2.BORDER_REPLICATE)
        x = torch.from_numpy(((p.astype(np.float32) / 255 - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))[None].to(DEV)
        with torch.no_grad():
            t = bb(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]
            g = int(round(t.shape[1] ** 0.5)); feat = t.transpose(1, 2).reshape(1, -1, g, g)
            sal = torch.nn.functional.interpolate(m.head(m.aspp(feat)), size=(size, size), mode="bilinear", align_corners=False)
            sal = torch.sigmoid(sal)[0, 0].cpu().numpy()                 # centre bias off
        return cv2.resize(sal[top:top + nh, left:left + nw], (w, h))
    return f


def dinov2_blocks_map(m, bb, idx, rgb, s, flip=False):
    """Readout `m` on concatenated DINOv2 hidden_states[idx] at a square input of s px (centre bias off)."""
    img = rgb[:, ::-1].copy() if flip else rgb
    x = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = torch.from_numpy(((x - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1))[None].to(DEV)
    with torch.no_grad():
        hs = bb(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
        fs = []
        for i in idx:
            t = hs[i][:, 1:, :]; g = int(round(t.shape[1] ** 0.5)); fs.append(t.transpose(1, 2).reshape(1, -1, g, g).contiguous())
        m.use_cb = False
        out = torch.sigmoid(m(torch.cat(fs, 1), (s, s)))[0, 0].cpu().numpy()
    out = cv2.resize(out.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    return out[:, ::-1] if flip else out


def xsoyelnet_ml(path=os.path.join(WEIGHTS, "xsoyelnet_ml.pt"), sizes=(280, 336, 392, 448)):
    """Blocks 6 and 10 (Hugging Face hidden_states[7] and [11]) concatenated; TTA over sizes and flips."""
    bb = _dinov2(); m = Readout(1536).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
    return lambda rgb: np.mean([dinov2_blocks_map(m, bb, (7, 11), rgb, s, fl) for s in sizes for fl in (False, True)], 0)


def spectral_residual(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape; small = cv2.resize(g, (64, int(round(64 * h / w))), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(small); L = np.log(np.abs(f) + 1e-8); P = np.angle(f)
    R = L - cv2.blur(L, (3, 3))
    s = np.abs(np.fft.ifft2(np.exp(R + 1j * P))) ** 2
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 2.5)
    return cv2.resize(s, (w, h))


def edge_null(rgb, sig=25):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), sig)


LOADERS = {"DeepGaze IIE": deepgaze_iie, "DeepGaze MSDB": deepgaze_msdb, "xSoyelNet-ASPP": xsoyelnet_aspp,
           "xSoyelNet-ML": xsoyelnet_ml, "Spectral residual": lambda: spectral_residual, "Edge-energy null": lambda: edge_null}
