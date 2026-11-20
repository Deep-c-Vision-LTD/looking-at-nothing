"""
v_s4_a6.py: independent recomputation of S4 (`explore/a6_absolute.py`, rule Section 6 item 7, second part: CC, NSS and
information gain on held-out MIT1003 with the centre bias off) for 12 heads: dinov2_L12 R4, dinov2_L6 R2, dinov2_L6 R4,
siglip_L24 R4, seeds 0, 1, 2. Written 19 November 2026 for the verification of the remaining CPU steps.

Independent of the project code: the readout forward pass is re-implemented here with torch.nn.functional from the
checkpoint's state dict (ASPP readout as defined in saliency_research/src/model_aspp.py and v2/readout/common.py, read,
not imported: dilated 3x3 conv branches without bias -> BatchNorm (eval) -> GELU; R4 adds the global-average branch;
1x1 projection -> BatchNorm -> GELU; dropout is the identity in eval; 1x1 head; optional learned 16 x 16 centre bias
added after bilinear resizing (align_corners False); bilinear resize to 336 x 336; sigmoid). MIT1003 held-out list
re-derived (sorted ALLSTIMULI *.jpeg + *.jpg with a fixation map, last 200). CC, NSS and IG written here (IG: both maps
floored at 1e-12 x max, normalised to sum 1, mean over fixated pixels of log2 p_model - log2 p_baseline; baseline = mean
of all ab/maps.npy training maps, resized bilinearly to the image). Only analyze_re is imported, for require_public(),
which runs first. CPU only: torch device cpu, no MPS call; PYTORCH_ENABLE_MPS_FALLBACK removed.
Compares: centre bias ON per-image CC with the published mit_<tag>.json; per-head means (cb on, cb off) with
results/explore/A6_absolute.json; cell-level WCU (every image its own cluster) of the seed-averaged values.
Output: neuro/verify_rescore/cpu_remaining/v_s4_a6.json
"""
import os, sys, time
sys.dont_write_bytecode = True
os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V                                                       # noqa: E402

N_HASHED = V.guard()                                                      # require_public() first
import glob, json                                                         # noqa: E402
import numpy as np                                                        # noqa: E402
import cv2                                                                # noqa: E402
import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
DEV = torch.device("cpu")
ROOT, RE = V.ROOT, V.RE
OUT = os.path.join(V.HERE, "v_s4_a6.json")
CACHE = os.path.join(ROOT, "v2", "readout", "cache")
CKPT = os.path.join(ROOT, "v3", "gpu", "full", "ckpt")
PUB = os.path.join(ROOT, "v3", "gpu", "full", "scores")
MIT = os.path.join(os.environ.get("LAN_SAL_SRC") or os.path.join(os.path.dirname(ROOT), "saliency_research"), "data", "mit1003")   # as gpu/recommon.py SRC
CELLS = [("dinov2_L12", "R4"), ("dinov2_L6", "R2"), ("dinov2_L6", "R4"), ("siglip_L24", "R4")]
S = 336


def bn(x, sd, p):
    return F.batch_norm(x, sd[p + ".running_mean"], sd[p + ".running_var"], sd[p + ".weight"], sd[p + ".bias"],
                        training=False, eps=1e-5)


def readout(sd, f, arch, use_cb):
    used = set()
    def g(k):
        used.add(k); return sd[k]
    dil = {"R4": (1, 3, 6, 12), "R2": (1, 3, 6, 12), "R3": (1,)}[arch]
    outs = []
    for i, d in enumerate(dil):
        p = f"aspp.branches.{i}"
        y = F.conv2d(f, g(p + ".0.weight"), None, padding=d, dilation=d)
        for k in (".running_mean", ".running_var", ".weight", ".bias"):
            used.add(p + ".1" + k)
        outs.append(F.gelu(bn(y, sd, p + ".1")))
    if arch == "R4":
        gp = F.adaptive_avg_pool2d(f, 1)
        y = F.conv2d(gp, g("aspp.gap.1.weight"))
        for k in (".running_mean", ".running_var", ".weight", ".bias"):
            used.add("aspp.gap.2" + k)
        y = F.gelu(bn(y, sd, "aspp.gap.2"))
        outs.append(F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False))
    y = F.conv2d(torch.cat(outs, 1), g("aspp.proj.0.weight"))
    for k in (".running_mean", ".running_var", ".weight", ".bias"):
        used.add("aspp.proj.1" + k)
    y = F.gelu(bn(y, sd, "aspp.proj.1"))
    s = F.conv2d(y, g("head.weight"), g("head.bias"))
    cb = g("center_bias")
    if use_cb:
        s = s + F.interpolate(cb, size=s.shape[-2:], mode="bilinear", align_corners=False)
    s = F.interpolate(s, size=(S, S), mode="bilinear", align_corners=False)
    left = {k for k in sd if not k.endswith("num_batches_tracked")} - used
    assert not left, left
    return torch.sigmoid(s)[:, 0]


def cc(a, b):
    a = a.astype(np.float64) - a.mean(); b = b.astype(np.float64) - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def nss(s, fix):
    s = s.astype(np.float64)
    return float(((s - s.mean()) / s.std())[fix].mean())


def nlog2(m):
    m = np.asarray(m, np.float64)
    m = m + 1e-12 * m.max()
    return np.log2(m / m.sum())


def main():
    t0 = time.time()
    st = V.jl(os.path.join(RE, "results", "explore", "A6_absolute.json"))["result"]
    stim = os.path.join(MIT, "ALLSTIMULI"); fmap = os.path.join(MIT, "ALLFIXATIONMAPS")
    pairs = []
    for img in sorted(glob.glob(os.path.join(stim, "*.jpeg")) + glob.glob(os.path.join(stim, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        if os.path.exists(os.path.join(fmap, stem + "_fixMap.jpg")):
            pairs.append(stem)
    held = pairs[-200:]
    gts = [cv2.imread(os.path.join(fmap, s + "_fixMap.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for s in held]
    fxs = [cv2.imread(os.path.join(fmap, s + "_fixPts.jpg"), cv2.IMREAD_GRAYSCALE) > 128 for s in held]
    maps = np.load(os.path.join(ROOT, "ab", "maps.npy"), mmap_mode="r")
    acc = np.zeros(maps.shape[1:], np.float64)
    for b in range(0, maps.shape[0], 512):
        acc += np.asarray(maps[b:b + 512], np.float64).sum(0)
    mean_map = acc / maps.shape[0]
    base = [nlog2(cv2.resize(mean_map.astype(np.float32), (g.shape[1], g.shape[0]))) for g in gts]
    res = dict(n_hashed_verified=N_HASHED, n_images=len(held), n_train_maps=int(maps.shape[0]),
               mean_map_min=float(mean_map.min()), mean_map_max=float(mean_map.max()), heads={}, cells={})
    per = {}
    for fs, ar in CELLS:
        feats = torch.from_numpy(np.asarray(np.load(os.path.join(CACHE, f"{fs}_mit200.npy"), mmap_mode="r"),
                                            dtype=np.float16)).float()
        assert feats.shape[0] == 200
        for s in (0, 1, 2):
            tag = f"{fs}_{ar}_s{s}"
            sd = torch.load(os.path.join(CKPT, tag + ".pt"), map_location="cpu")
            rec = {}
            for cb in (True, False):
                with torch.no_grad():
                    sal = torch.cat([readout(sd, feats[b:b + 16].to(DEV), ar, cb) for b in range(0, 200, 16)]).numpy()
                m = dict(CC=[], NSS=[], IG=[])
                for i in range(200):
                    hh, ww = gts[i].shape
                    s_ = cv2.resize(sal[i], (ww, hh))
                    m["CC"].append(cc(s_, gts[i])); m["NSS"].append(nss(s_, fxs[i]))
                    m["IG"].append(float(np.mean(nlog2(s_)[fxs[i]] - base[i][fxs[i]])))
                rec["on" if cb else "off"] = {k: np.array(v) for k, v in m.items()}
            per[tag] = rec
            pub = V.jl(os.path.join(PUB, f"mit_{tag}.json"))
            sh = st["per_head"][tag]
            res["heads"][tag] = dict(
                gate_max_abs_per_image_CC=float(np.max(np.abs(rec["on"]["CC"] - np.array(pub["cc_per_image"])))),
                gate_abs_mean_CC=abs(float(rec["on"]["CC"].mean()) - pub["CC"]),
                gate_abs_mean_NSS=abs(float(rec["on"]["NSS"].mean()) - pub["NSS"]),
                cb_off={k: float(v.mean()) for k, v in rec["off"].items()},
                diff_vs_stored_cb_off={k: abs(float(rec["off"][k].mean()) - sh["cb_off"][k]) for k in ("CC", "NSS", "IG")},
                diff_vs_stored_cb_on={k: abs(float(rec["on"][k].mean()) - sh["cb_on"][k]) for k in ("CC", "NSS", "IG")})
            print(tag, json.dumps(res["heads"][tag]), f"{time.time() - t0:.0f}s", flush=True)
    for fs, ar in CELLS:
        tags = [f"{fs}_{ar}_s{s}" for s in (0, 1, 2)]
        sc = st["cells"][f"{fs}|{ar}"]
        out = {}
        for met in ("CC", "IG"):
            v = np.mean([per[t]["off"][met] for t in tags], 0)
            w = V.wild_iid(v)
            stc = sc["cb_off"][met]
            out[met + "_off"] = dict(mine=w, stored=dict(mean=stc["mean"], ci=stc["ci"], p=stc["p"]),
                                     max_abs_diff=max(abs(w["mean"] - stc["mean"]), abs(w["ci"][0] - stc["ci"][0]),
                                                      abs(w["ci"][1] - stc["ci"][1])))
            dv = np.mean([per[t]["off"][met] - per[t]["on"][met] for t in tags], 0)
            w2 = V.wild_iid(dv); s2 = sc["off_minus_on"][met]
            out[met + "_off_minus_on"] = dict(mine=w2, max_abs_diff=max(abs(w2["mean"] - s2["mean"]),
                                                                        abs(w2["ci"][0] - s2["ci"][0]),
                                                                        abs(w2["ci"][1] - s2["ci"][1]), abs(w2["p"] - s2["p"])))
        if ar != "R4":
            r4 = [f"{fs}_R4_s{s}" for s in (0, 1, 2)]
            dv = np.mean([per[t]["off"]["CC"] for t in tags], 0) - np.mean([per[t]["off"]["CC"] for t in r4], 0)
            w3 = V.wild_iid(dv); s3 = sc["minus_R4_cb_off"]["CC"]
            out["CC_minus_R4_off"] = dict(mine=w3, max_abs_diff=max(abs(w3["mean"] - s3["mean"]), abs(w3["ci"][0] - s3["ci"][0]),
                                                                    abs(w3["ci"][1] - s3["ci"][1]), abs(w3["p"] - s3["p"])))
        res["cells"][f"{fs}|{ar}"] = out
    res["seconds"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["cells"], indent=1)[:5000])


if __name__ == "__main__":
    main()
