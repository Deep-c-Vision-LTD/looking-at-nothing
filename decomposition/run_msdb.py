"""
Shapley decomposition, DeepGaze MSDB.

MSDB: 10 scale branches x (CLIP RN50x64 layer4.2.conv2 [1024 ch] | DINOv2 block 10 [768] | DINOv2 block 6 [768]),
bilinearly resized to the 48 x 64 readout grid, summed with fixed dataset-averaged scale weights, then the saliency
network (first LayerNorm over all 2,560 channels and positions, 1 x 1 layers), bilinear upsampling, the finalizer
(Gaussian blur, uniform centre bias, normalisation). Per configuration the per-scale features of the illusory and the
control180 image are computed with the model's own _process_pixel_per_dva / _process_size, and these hybrids are
scored (AOI masses of the density resized to 768 x 1024 with cv2 bilinear, as in the published wrapper):
  grp_<abc>  feature-group game, players CLIP, DINOv2 block 10, DINOv2 block 6 (char = 1: that group illusory)
  scg_<abc>  scale-group game, players coarse (ppd 5, 10, size 128, 256), native (ppd 17.5, size 512),
             fine (ppd 24, 30, size 768, 1024)
  sc10_<k>   exact 10-scale game, 1,024 coalitions (bit j of k = scale j illusory; scales ordered ppd 5, 10, 17.5,
             24, 30, size 128, 256, 512, 768, 1024), evaluated with the exact LayerNorm + conv0 factorisation and
             checked against direct evaluation on 16 coalitions per configuration.
Maps saved: maps/msdb/<config>.npz (grp_* and scg_*, 384 x 512 model-output density x 384*512, float16).
Scale-weighted features of each image cached as float16 in cache/msdb_X/<image>.npy for run_lesion.py.
Usage: python run_msdb.py [n_configs]  -> scores/msdb.json (resumable)
"""
import gc, math, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_d import *  # noqa
torch = torch_setup()
import torch.nn.functional as F  # noqa: E402
from deepgaze_pytorch import deepgazemsdb as DGM  # noqa: E402

OUT = os.path.join(SCORES, "msdb.json")
MD = os.path.join(MAPS, "msdb"); XD = os.path.join(CACHE, "msdb_X")
for d in (MD, XD):
    os.makedirs(d, exist_ok=True)
SCALE = 0.5; PPD = 35.0 * SCALE
h, w = int(H * SCALE), int(W * SCALE)                       # 384 x 512
RS = [math.ceil(h / DGM._READOUT_FACTOR), math.ceil(w / DGM._READOUT_FACTOR)]     # 48 x 64
SAL = [math.ceil(h / DGM._SALIENCY_MAP_FACTOR), math.ceil(w / DGM._SALIENCY_MAP_FACTOR)]
GROUPS = {"CLIP": slice(0, 1024), "DINOv2_b10": slice(1024, 1792), "DINOv2_b6": slice(1792, 2560)}
SCALE_NAMES = [f"ppd{p:g}" for p in DGM._PIXEL_PER_DVA_SCALES] + [f"size{s}" for s in DGM._SIZE_SCALES]
SCALE_GROUPS = {"coarse": (0, 1, 5, 6), "native": (2, 7), "fine": (3, 4, 8, 9)}


class MSDB:
    def __init__(self):
        self.m = DGM.DeepGazeMSDB(pretrained=True); self.m.eval()
        fe = self.m.features
        pw = torch.exp(fe.pixel_per_dva_weights).mean(dim=1, keepdim=True)
        sw = torch.exp(fe.size_weights).mean(dim=1, keepdim=True)
        tot = pw.sum(dim=0, keepdim=True) + sw.sum(dim=0, keepdim=True)
        pw, sw = pw / tot, sw / tot
        z = torch.zeros(1, dtype=torch.long)
        self.wts = [p[z].view(-1, 1, 1, 1).detach() for p in pw] + [s[z].view(-1, 1, 1, 1).detach() for s in sw]   # as in forward
        self.cb = torch.full((1, h, w), -math.log(h * w))
        sn = self.m.saliency_network
        self.rest = torch.nn.Sequential(*list(sn.children())[2:])        # everything after conv0
        g = sn.layernorm0.weight.detach().double(); b = sn.layernorm0.bias.detach().double()
        W0 = sn.conv0.weight.detach()[:, :, 0, 0].double()
        self.Wg = W0 * g[None]; self.a = self.Wg.sum(1); self.c = W0 @ b; self.eps = sn.layernorm0.eps

    @torch.no_grad()
    def scale_feats(self, rgb):
        sm = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(sm.astype(np.float32).transpose(2, 0, 1))[None]
        fe = self.m.features
        out = [fe._process_pixel_per_dva(x, [PPD], t, RS) for t in fe.resolutions_pixel_per_dva]
        out += [fe._process_size(x, s, RS) for s in fe.resolutions_size]
        return x, out

    @torch.no_grad()
    def wsum(self, feats_by_scale):
        X = None
        for wt, f in zip(self.wts, feats_by_scale):
            X = wt * f if X is None else X + wt * f
        return X

    @torch.no_grad()
    def finish(self, s):
        """s: saliency-network output (B,1,48,64) -> log density (B,384,512)."""
        s = F.interpolate(s, SAL, mode="bilinear")
        B_ = s.shape[0]
        return self.m.finalizer(s, self.cb.repeat(B_, 1, 1), [PPD / DGM._SALIENCY_MAP_FACTOR] * B_, dataset_indices=None)

    @torch.no_grad()
    def head(self, X):
        return self.finish(self.m.saliency_network(X))

    @torch.no_grad()
    def head_from_z(self, z):
        return self.finish(self.rest(z))


def to_map(logd):
    d = torch.exp(logd).numpy().astype(np.float32)
    return d, cv2.resize(d, (W, H))


def main(nmax=None):
    cfgs = kanizsa_configs()[: nmax or None]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"configs": {}, "checks": {}}
    M = MSDB()
    res["scale_names"] = SCALE_NAMES
    res["scale_weights"] = [round(float(x), 6) for x in M.wts]
    res["scale_groups"] = {k: [SCALE_NAMES[i] for i in v] for k, v in SCALE_GROUPS.items()}
    res["feature_groups"] = {k: [v.start, v.stop] for k, v in GROUPS.items()}
    rng = np.random.RandomState(7)
    t0 = time.time()
    for ci, c in enumerate(cfgs):
        if c["config"] in res["configs"]:
            continue
        t1 = time.time()
        aoi = read_aoi(c["ill"])
        Fs, Xs, xin = {}, {}, {}
        for tag, name in (("I", c["ill"]), ("C", c["ctrl"])):
            xin[tag], Fs[tag] = M.scale_feats(read_rgb(name))
            assert all(f.shape == (1, 2560, RS[0], RS[1]) for f in Fs[tag])
            Xs[tag] = M.wsum(Fs[tag])
            np.save(os.path.join(XD, name + ".npy"), Xs[tag][0].numpy().astype(np.float16))
        chk = {}
        if ci == 0:     # head(sum of per-scale features) must equal the model's own forward pass
            with torch.no_grad():
                ld_full = M.m(xin["I"].clone(), M.cb, pixel_per_dva=PPD, dataset=None)
            chk["max_abs_logdens_diff_head_vs_forward"] = float((ld_full - M.head(Xs["I"])).abs().max())
            chk["X_absmax"] = float(max(Xs["I"].abs().max(), Xs["C"].abs().max()))
        out, maps = {}, {}
        # ---- feature-group game (direct, model modules)
        for T in range(8):
            X = Xs["C"].clone()
            for j, sl in enumerate(GROUPS.values()):
                if T >> j & 1:
                    X[:, sl] = Xs["I"][:, sl]
            d, mp = to_map(M.head(X)[0])
            key = "grp_" + "".join(str(T >> j & 1) for j in range(3))
            out[key] = masses(mp, aoi); maps[key] = (d * (h * w)).astype(np.float16)
        # ---- scale-group game (direct)
        sc_of = {}
        for j, (gname, ks) in enumerate(SCALE_GROUPS.items()):
            for k in ks:
                sc_of[k] = j
        for T in range(8):
            X = M.wsum([Fs["I" if T >> sc_of[k] & 1 else "C"][k] for k in range(10)])
            d, mp = to_map(M.head(X)[0])
            key = "scg_" + "".join(str(T >> j & 1) for j in range(3))
            out[key] = masses(mp, aoi); maps[key] = (d * (h * w)).astype(np.float16)
        np.savez_compressed(os.path.join(MD, c["config"] + ".npz"), **maps)
        # ---- exact 10-scale game via the LayerNorm / conv0 factorisation
        with torch.no_grad():
            # sources j: 0..9 illusory scales, 10..19 control180 scales; accumulated over channel chunks (memory)
            feats = [f[0] for tag in ("I", "C") for f in Fs[tag]]
            npos = RS[0] * RS[1]; n = 2560 * npos
            G = torch.zeros(20, 20, dtype=torch.float64); mu = torch.zeros(20, dtype=torch.float64)
            Pk = torch.zeros(20, 8, npos, dtype=torch.float64)
            for c0 in range(0, 2560, 256):
                Ac = torch.stack([f[c0:c0 + 256].reshape(256, npos) for f in feats]).double()   # (20, 256, npos)
                fl = Ac.reshape(20, -1)
                G += fl @ fl.T; mu += fl.sum(1)
                Pk += torch.einsum("oc,jcl->jol", M.Wg[:, c0:c0 + 256], Ac)
                del Ac, fl
            G /= n; mu /= n
            del feats
            wv = torch.tensor([float(x) for x in M.wts], dtype=torch.float64)
            masks = np.arange(1024)
            U = torch.zeros(1024, 20, dtype=torch.float64)
            for k in range(10):
                ill = torch.from_numpy((masks >> k & 1).astype(bool))
                U[ill, k] = wv[k]; U[~ill, 10 + k] = wv[k]
            P = torch.einsum("mj,jol->mol", U, Pk)
            mean = U @ mu; e2 = torch.einsum("mj,jk,mk->m", U, G, U)
            sd = torch.sqrt(e2 - mean ** 2 + M.eps)
            Z = ((P - mean[:, None, None] * M.a[None, :, None]) / sd[:, None, None] + M.c[None, :, None]).float()
            Z = Z.view(1024, 8, RS[0], RS[1])
            # direct check on the 8 scale-group coalitions and 8 random coalitions
            grp_masks = [sum(1 << k for k in range(10) if T >> sc_of[k] & 1) for T in range(8)]
            test = grp_masks + [int(v) for v in rng.choice(1024, 8, replace=False)]
            sc10 = {}
            logd_alg = {}
            for b0 in range(0, 1024, 128):
                ld = M.head_from_z(Z[b0:b0 + 128])
                for i in range(ld.shape[0]):
                    k = b0 + i
                    _, mp = to_map(ld[i]); sc10[str(k)] = masses(mp, aoi)
                    if k in test:
                        logd_alg[k] = ld[i].clone()
            dmax, rmax = 0.0, 0.0
            for k in test:
                X = M.wsum([Fs["I" if k >> s & 1 else "C"][s] for s in range(10)])
                ld = M.head(X)[0]
                dmax = max(dmax, float((ld - logd_alg[k]).abs().max()))
                _, mp = to_map(ld); mm = masses(mp, aoi)
                rmax = max(rmax, abs(r_from(mm) - r_from(sc10[str(k)])))
            chk["sc10_check_max_abs_logdens_diff"] = dmax; chk["sc10_check_max_abs_r_diff"] = rmax
            del logd_alg, P, Z, Pk
        out_cfg = dict(hybrids=out, sc10=sc10)
        res["configs"][c["config"]] = out_cfg
        res["checks"][c["config"]] = chk
        save_json(res, OUT)
        del Fs, Xs, xin, maps
        gc.collect()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"msdb {ci + 1}/{len(cfgs)} {c['config']} {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s) "
              f"maxrss {rss:.2f} GB check {chk}", flush=True)
    save_json(res, OUT)
    print("MSDB DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
