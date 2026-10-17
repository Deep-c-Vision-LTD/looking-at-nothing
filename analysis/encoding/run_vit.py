"""
Encoding index, transformers: per-token components of the encoding index for every hidden state of one backbone at one
input size, on stim_v3.
Usage: python run_vit.py <dinov2|clip|siglip> <336|448> [max_configs]
Output: <results>/v2/encoding/tokens/<net>_s<S>/<config>.npz with, per configuration (G x G tokens, row-major):
  dot  [L, T, G*G]  <f(test) - f(ctrl), f(realC) - f(ctrl)> per token        (float32; T = K_TESTS or E_TESTS)
  nt   [L, T, G*G]  ||f(test) - f(ctrl)||^2
  nrc  [L, G*G]     ||f(realC) - f(ctrl)||^2
  same [I-1, G*G]   input patch pixel-identical to the control's patch, for each other image (order imgs[1:])
  inside_full [G*G] footprint entirely inside the 'inside' AOI;  far [G*G] footprint > 80 px from the figure
Conventions as sweep.py: Hugging Face hidden_states[0..N] (no final LayerNorm), CLS dropped for DINOv2 and CLIP,
square cv2.INTER_AREA resize of the 1024 x 768 image, each backbone's normalisation, float32 on CPU.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_enc import *  # noqa: F401,F403
from common_enc import (VIT, K_IMGS, E_IMGS, K_TESTS, E_TESTS, TOK, LOGS, W, H, configs, read_rgb, figure_dist,
                        footprints, region_masks, patch_equal, log)
import numpy as np, cv2, torch
torch.set_num_threads(3)
cv2.setNumThreads(1)

NET, S = sys.argv[1], int(sys.argv[2])
MAXC = int(sys.argv[3]) if len(sys.argv) > 3 else None
C = VIT[NET]; P = C["patch"]; G = S // P
assert G * P == S, (S, P)
MEAN, STD = np.array(C["mean"], np.float32), np.array(C["std"], np.float32)
OUT = os.environ.get("ENC_OUT") or os.path.join(TOK, f"{NET}_s{S}"); os.makedirs(OUT, exist_ok=True)   # ENC_OUT: test runs only
LOG = os.path.join(OUT, "test.log") if os.environ.get("ENC_OUT") else os.path.join(LOGS, f"encoding_{NET}_s{S}.log")


def load():
    if NET == "dinov2":
        from transformers import AutoModel; m = AutoModel.from_pretrained(C["hf"])
    elif NET == "clip":
        from transformers import CLIPVisionModel; m = CLIPVisionModel.from_pretrained(C["hf"])
    else:
        from transformers import SiglipVisionModel; m = SiglipVisionModel.from_pretrained(C["hf"])
    return m.to("cpu").eval()


def rs(rgb):
    return cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)


@torch.no_grad()
def run():
    log(LOG, f"=== start {NET} s{S} grid {G}x{G} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    m = load()
    cfgs = configs()[:MAXC] if MAXC else configs()
    fp = footprints(G, G, P, P, H / S, W / S)
    t0 = time.time()
    for k, cf in enumerate(cfgs):
        path = os.path.join(OUT, cf["config"] + ".npz")
        if os.path.exists(path):
            continue
        names = K_IMGS if cf["illusion"] == "kanizsa" else E_IMGS
        tests = K_TESTS if cf["illusion"] == "kanizsa" else E_TESTS
        imgs = [rs(read_rgb(cf["imgs"][c])) for c in names]
        same = np.stack([patch_equal(imgs[i], imgs[0], G, P).ravel() for i in range(1, len(names))])
        aoi = np.load(cf["aoi"])
        inside_full, far = region_masks(fp, aoi, figure_dist(aoi))
        x = torch.stack([torch.from_numpy(((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)) for im in imgs])
        hs = m(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
        assert len(hs) == C["n_hs"], len(hs)
        ir = names.index("realC"); it = [names.index(t) for t in tests]
        dot = np.zeros((len(hs), len(tests), G * G), np.float32); nt = np.zeros_like(dot)
        nrc = np.zeros((len(hs), G * G), np.float32)
        for L, h in enumerate(hs):
            f = (h[:, 1:, :] if C["cls"] else h).double()
            assert f.shape[1] == G * G, f.shape
            d = f - f[0:1]
            drc = d[ir]
            nrc[L] = (drc * drc).sum(-1).numpy()
            for j, i in enumerate(it):
                dot[L, j] = (d[i] * drc).sum(-1).numpy(); nt[L, j] = (d[i] * d[i]).sum(-1).numpy()
        np.savez_compressed(path, dot=dot, nt=nt, nrc=nrc, same=same, inside_full=inside_full.ravel(), far=far.ravel(),
                            imgs=np.array(names), tests=np.array(tests), grid=np.array([G, P, S]))
        if k % 10 == 0 or k == len(cfgs) - 1:
            log(LOG, f"  {k + 1}/{len(cfgs)} {cf['config']} interior-identical tokens "
                     f"{int((same.all(0) & inside_full.ravel()).sum())} far {int(far.sum())} {time.time() - t0:.0f}s")
    log(LOG, f"=== end {NET} s{S} {time.time() - t0:.0f}s {time.strftime('%Y-%m-%d %H:%M:%S')}")
    open(os.path.join(OUT, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    run()
