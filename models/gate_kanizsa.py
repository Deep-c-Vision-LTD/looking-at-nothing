"""
The inside-share measure and the loaders of three survey models.

rel_inside(sal, inside, ind) = saliency mass inside the figure / (mass inside + mass on the inducers), map clipped at 0:
the analogue of the human dwell measure (Kavsek, 2024).

load_models(keys), keys among "aspp", "iie", "msdb"; every model maps an RGB uint8 image of 768 x 1024 to a map of the
same size, without a centre-bias prior (figure position is varied):
  aspp  xSoyelNet-ASPP: frozen DINOv2-B last_hidden_state + ASPP readout (models/src/model_aspp.py), 448 px letterbox
        with replicate padding, centre bias off. Weights: paths.XSOYELNET_ASPP (not distributed).
  iie   DeepGaze IIE (deepgaze_pytorch, pretrained), full-resolution input, uniform centre bias, predicted density.
  msdb  DeepGaze MSDB (deepgaze_pytorch, pretrained), half-resolution input at 17.5 px per degree, dataset=None
        (dataset-averaged parameters), uniform centre bias; the density is resized back to 768 x 1024.
"""
import json, math, os, sys, time
import numpy as np, cv2, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import paths as LP                           # noqa: E402

W, H, SS = 1024, 768, 4                      # canvas, supersampling factor for anti-aliasing


# ----------------------------------------------------------------- models
def load_models(which):
    M = {}
    if "aspp" in which:
        from src.model_aspp import ASPPSaliency
        m = ASPPSaliency("facebook/dinov2-base", aspp_ch=128, pretrained=False)
        c = torch.load(str(LP.XSOYELNET_ASPP), map_location="cpu", weights_only=False)
        m.load_state_dict(c.get("model", c)); m.eval(); m.use_center_bias = False
        MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
        def aspp(rgb, size=448):
            h, w = rgb.shape[:2]; s = size / max(h, w); nh, nw = round(h * s), round(w * s)
            r = cv2.resize(rgb, (nw, nh)); top = (size - nh) // 2; left = (size - nw) // 2
            p = cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left, cv2.BORDER_REPLICATE)
            x = torch.from_numpy(((p.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None]
            with torch.no_grad():
                sal = torch.sigmoid(m(x))[0, 0].numpy()
            return cv2.resize(sal[top:top + nh, left:left + nw], (w, h))
        M["xSoyelNet-ASPP"] = aspp
    if "iie" in which or "msdb" in which:
        import deepgaze_pytorch
        torch.set_num_threads(os.cpu_count() or 8)
    if "iie" in which:
        iie = deepgaze_pytorch.DeepGazeIIE(pretrained=True); iie.eval()
        def f_iie(rgb):
            x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None]
            cb = torch.full((1, H, W), -math.log(H * W))
            with torch.no_grad():
                return np.squeeze(torch.exp(iie(x, cb)).numpy())
        M["DeepGaze IIE"] = f_iie
    if "msdb" in which:
        msdb = deepgaze_pytorch.DeepGazeMSDB(pretrained=True); msdb.eval()
        def f_msdb(rgb, s=0.5):
            sm = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(sm.astype(np.float32).transpose(2, 0, 1))[None]
            h, w = sm.shape[:2]; cb = torch.full((1, h, w), -math.log(h * w))
            with torch.no_grad():
                d = np.squeeze(torch.exp(msdb(x, cb, pixel_per_dva=35.0 * s, dataset=None)).numpy())
            return cv2.resize(d, (W, H))
        M["DeepGaze MSDB"] = f_msdb
    return M


# ----------------------------------------------------------------- measure
def rel_inside(sal, ins, ind):
    s = np.clip(sal.astype(np.float64), 0, None)
    a, b = s[ins].sum(), s[ind].sum()
    return float(a / (a + b)) if a + b > 0 else float("nan")
