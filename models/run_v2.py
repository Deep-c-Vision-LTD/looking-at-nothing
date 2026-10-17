"""
Training-free survey models and the xSoyelNet-ML builder.

  spectral_residual  Hou & Zhang (2007), own implementation (64 px wide spectrum, Gaussian blur 2.5 px at that scale)
  edge_null          edge-energy null: Sobel gradient magnitude of the grey image blurred with sigma = 25 px
  build(keys)        {model name: map function} for keys among aspp, iie, msdb (gate_kanizsa.load_models), ml, sr, edge
                     ml = xSoyelNet-ML: DINOv2-B blocks 7 and 11 (Hugging Face hidden_states[7] and [11]) read by an
                     ASPP readout (ab_layers.Readout, weights paths.XSOYELNET_ML, not distributed), test-time average
                     over 280/336/392/448 px and horizontal flips, centre bias off (scales.sal_at).
No model receives a centre-bias prior (figures are placed at five positions).
"""
import json, os, sys, time
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gate_kanizsa as gk  # noqa: E402
import paths as LP         # noqa: E402  (gate_kanizsa puts the repository root on sys.path)


def spectral_residual(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape; small = cv2.resize(g, (64, int(round(64 * h / w))), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(small); L = np.log(np.abs(f) + 1e-8); P = np.angle(f)
    R = L - cv2.blur(L, (3, 3))
    s = np.abs(np.fft.ifft2(np.exp(R + 1j * P))) ** 2
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 2.5)            # ~8 px at 64-px width scaled
    return cv2.resize(s, (w, h))


def edge_null(rgb, sig=25):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), sig)


def build(keys):
    M = {}
    base = gk.load_models([k for k in ("aspp", "iie", "msdb") if k in keys])
    for n, f in base.items():
        M[n] = f
    if "ml" in keys:
        import ab_objective as AB, ab_layers as AL, scales as SC
        m = AL.Readout(1536).to(AB.DEV); m.load_state_dict(torch.load(str(LP.XSOYELNET_ML), map_location=AB.DEV)); m.eval()
        M["xSoyelNet-ML"] = lambda rgb: np.mean([SC.sal_at(m, (7, 11), rgb, s, False, fl) for s in (280, 336, 392, 448) for fl in (False, True)], 0)
    if "msinet" in keys:
        M["MSI-Net"] = None                                             # precomputed maps (msinet_maps.py)
    if "sr" in keys:
        M["Spectral residual"] = spectral_residual
    if "edge" in keys:
        M["Edge-energy null"] = edge_null
    return M
