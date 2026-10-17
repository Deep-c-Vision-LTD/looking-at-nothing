"""
Shapley decomposition, DeepGaze IIE: per-backbone densities for the 60 Kanizsa illusory and control180 images.

IIE = MixtureModel over 4 backbone paths (features -> 30 readouts -> 30 finalizers -> normalised density each);
the output density is the mean of the 4 path densities. A path's features are substituted by substituting its density.
For every configuration, the 16 hybrids H(T) (backbones in T from the illusory image, the rest from control180) are
scored (AOI masses of the 768 x 1024 map). Maps saved: maps/iie/<image>.npz with p0..p3 = path density x (H*W) in float16
(mean 1); a hybrid is the mean of the selected paths.
Usage: python run_iie.py   -> scores/iie.json (resumable)
"""
import math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_d import *  # noqa
torch = torch_setup()
import deepgaze_pytorch  # noqa: E402

OUT = os.path.join(SCORES, "iie.json")
MD = os.path.join(MAPS, "iie"); os.makedirs(MD, exist_ok=True)


def main():
    cfgs = kanizsa_configs()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"configs": {}, "checks": {}}
    iie = deepgaze_pytorch.DeepGazeIIE(pretrained=True); iie.eval()
    cb = torch.full((1, H, W), -math.log(H * W))
    t0 = time.time()
    for ci, c in enumerate(cfgs):
        if c["config"] in res["configs"]:
            continue
        aoi = read_aoi(c["ill"])
        aoi_c = read_aoi(c["ctrl"])
        assert all((aoi[k] == aoi_c[k]).all() for k in AOI_KEYS)
        P = {}
        for tag, name in (("I", c["ill"]), ("C", c["ctrl"])):
            rgb = read_rgb(name)
            x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None]
            with torch.no_grad():
                ld = [iie.models[b](x, cb)[0, 0].double().exp().numpy() for b in range(4)]
                if ci < 2:   # check the decomposition against the model's own forward pass
                    full = torch.exp(iie(x, cb)).double().numpy().squeeze()
                    mix = np.mean(ld, 0)
                    res["checks"][name] = dict(max_abs_rel_diff_mixture_vs_forward=float(np.max(np.abs(mix - full)) / full.max()),
                                               sum_paths=[float(p.sum()) for p in ld])
            P[tag] = ld
            np.savez_compressed(os.path.join(MD, name + ".npz"), **{f"p{b}": (ld[b] * (H * W)).astype(np.float16) for b in range(4)})
        hyb = {}
        for T in range(16):
            m = np.mean([P["I" if T >> b & 1 else "C"][b] for b in range(4)], 0)
            hyb[format(T, "04b")[::-1]] = masses(m, aoi)          # key: char b = 1 if backbone b illusory
        res["configs"][c["config"]] = dict(hybrids=hyb)
        if (ci + 1) % 5 == 0 or ci == len(cfgs) - 1:
            save_json(res, OUT)
            print(f"iie {ci + 1}/{len(cfgs)} {time.time() - t0:.0f}s", flush=True)
    save_json(res, OUT)
    print("IIE DONE", flush=True)


if __name__ == "__main__":
    main()
