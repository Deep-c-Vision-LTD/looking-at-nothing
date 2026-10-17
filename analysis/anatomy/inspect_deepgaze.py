"""
DeepGaze IIE and DeepGaze MSDB anatomy from the installed deepgaze_pytorch package and its cached checkpoints
(CPU only, 3 threads). Light compute: one forward pass per backbone on one display-sized input.

  1. Checkpoint inventory (torch.load): readout parameter counts, learned Gaussian sigmas, centre-bias weights,
     MSDB scale weights averaged over datasets (the dataset=None path used in our runs).
  2. Instantiate both models with pretrained=True, run the feature extractors on a half-resolution display
     (384 x 512, which is what both models feed their backbones in our runs) and record the channel count and
     spatial size of every hooked layer; time every backbone.
  3. Theoretical receptive fields of the hooked CNN layers (walk of the main-path conv / pool modules).
Output: <results>/v2/anatomy/facts_deepgaze.json. These facts (readout parameter counts, learned blur sigmas, MSDB
scale weights, channel counts) are the DeepGaze entries of <results>/v2/anatomy/anatomy.json.
"""
import json, math, os, time
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
import numpy as np, torch, cv2
torch.set_num_threads(3)
import deepgaze_pytorch
from deepgaze_pytorch import deepgaze2e as DG2E, deepgazemsdb as DGM

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths as LP  # noqa: E402
OUT = str(LP.V2 / "anatomy" / "facts_deepgaze.json"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
CK = os.path.join(torch.hub.get_dir(), "checkpoints")
STIM = str(LP.STIM_V2 / "img" / "square_sr50_dark_p0_illusory.png")
SKIP = ("downsample", "_se_reduce", "_se_expand", "attnpool", "fc", "classifier", "_fc")


def rf_walk(model, stop_name):
    """Theoretical receptive field (px of the backbone input) and jump of `stop_name` along the main path."""
    r, j = 1, 1
    for name, m in model.named_modules():
        if any(s in name.split(".")[-1] or s in name for s in SKIP):
            continue
        if isinstance(m, torch.nn.Conv2d):
            k = m.kernel_size[0]; d = m.dilation[0]; s = m.stride[0]
            r += (d * (k - 1)) * j; j *= s
        elif isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            r += (k - 1) * j; j *= s
        if name == stop_name:
            return r, j
    return None, None


def main():
    facts = {"package": dict(file=deepgaze_pytorch.__file__, version="1.2.1")}
    # ---------------- 1. checkpoints
    sd = torch.load(os.path.join(CK, "deepgaze2e.pth"), map_location="cpu", weights_only=False)
    rec = {"n_tensors": len(sd)}
    for b in range(4):
        pre = f"models.{b}."
        ks = [k for k in sd if k.startswith(pre)]
        feat = [k for k in ks if k.startswith(pre + "features.")]
        read = [k for k in ks if not k.startswith(pre + "features.")]
        sig = [float(sd[k]) for k in read if k.endswith("gauss.sigma")]
        cbw = [float(sd[k]) for k in read if k.endswith("center_bias_weight")]
        sal_in = [list(sd[k].shape) for k in read if k.endswith("saliency_networks.0.conv0.weight")]
        rec[f"backbone{b}"] = dict(type=DG2E.BACKBONES[b]["type"], used_features=DG2E.BACKBONES[b]["used_features"],
                                   channels_config=DG2E.BACKBONES[b]["channels"],
                                   n_feature_tensors_in_ckpt=len(feat), n_feature_params_in_ckpt=int(sum(sd[k].numel() for k in feat)),
                                   n_readout_params=int(sum(sd[k].numel() for k in read)),
                                   n_components=len({k.split(".")[3] for k in read if ".saliency_networks." in k}),
                                   saliency_conv0_shape=sal_in, sigma_mean=float(np.mean(sig)), sigma_min=float(np.min(sig)), sigma_max=float(np.max(sig)),
                                   center_bias_weight_mean=float(np.mean(cbw)), center_bias_weight_min=float(np.min(cbw)), center_bias_weight_max=float(np.max(cbw)))
    rec["n_readout_params_total"] = int(sum(rec[f"backbone{b}"]["n_readout_params"] for b in range(4)))
    allsig = [float(sd[k]) for k in sd if k.endswith("gauss.sigma")]
    rec["sigma_all_mean"] = float(np.mean(allsig)); rec["n_finalizers"] = len(allsig)
    facts["IIE_checkpoint"] = rec
    del sd
    sd = torch.load(os.path.join(CK, "deepgazemsdb.pth"), map_location="cpu", weights_only=False)
    rec = {"keys_shapes": {k: list(v.shape) for k, v in sd.items()}, "n_params": int(sum(v.numel() for v in sd.values()))}
    rec["n_saliency_network_params"] = int(sum(v.numel() for k, v in sd.items() if k.startswith("saliency_network.")))
    sig = sd["finalizer.gauss.dataset_sigmas"]; cbw = sd["finalizer.dataset_center_bias_weights"]
    rec["dataset_sigmas"] = sig.tolist(); rec["sigma_mean_used_when_dataset_None"] = float(sig.mean())
    rec["dataset_center_bias_weights"] = cbw.tolist()
    ppd_w = torch.exp(sd["features.pixel_per_dva_weights"]).mean(1); size_w = torch.exp(sd["features.size_weights"]).mean(1)
    tot = ppd_w.sum() + size_w.sum()
    rec["scale_weights_dataset_None"] = dict(pixel_per_dva={str(p): round(float(w / tot), 4) for p, w in zip(DGM._PIXEL_PER_DVA_SCALES, ppd_w)},
                                             size={str(s): round(float(w / tot), 4) for s, w in zip(DGM._SIZE_SCALES, size_w)})
    facts["MSDB_checkpoint"] = rec
    del sd
    print(json.dumps(facts["IIE_checkpoint"]["backbone0"], indent=0)[:600], flush=True)
    print("MSDB", facts["MSDB_checkpoint"]["keys_shapes"].get("saliency_network.conv0.weight"), facts["MSDB_checkpoint"]["scale_weights_dataset_None"], flush=True)

    rgb = cv2.cvtColor(cv2.imread(STIM), cv2.COLOR_BGR2RGB)
    half = cv2.resize(rgb, (512, 384), interpolation=cv2.INTER_AREA)
    xh = torch.from_numpy(half.astype(np.float32).transpose(2, 0, 1))[None]

    # ---------------- 2a. MSDB backbone channels
    msdb = DGM.DeepGazeMSDB(pretrained=True); msdb.eval()
    bbc = msdb.features.backbone
    with torch.no_grad():
        t0 = time.time(); outs = bbc(xh.clone()); t_bb = time.time() - t0
    names = [("CLIP RN50x64", t) for t in bbc.feature_extractors["CLIP"].targets] + [("DINOv2 ViT-B/14", t) for t in bbc.feature_extractors["DINOv2"].targets]
    facts["MSDB_backbone_outputs_at_384x512"] = [dict(net=n, target=t, shape=list(o.shape)) for (n, t), o in zip(names, outs)]
    facts["MSDB_total_channels"] = int(sum(o.shape[1] for o in outs))
    facts["MSDB_saliency_network_in_channels"] = int(msdb.saliency_network.conv0.weight.shape[1])
    facts["MSDB_backbone_time_s_at_384x512"] = round(t_bb, 2)
    clip_visual = bbc.feature_extractors["CLIP"].features
    facts["MSDB_CLIP_RF_layer4.2.conv2"] = rf_walk(clip_visual, "1.layer4.2.conv2")
    facts["MSDB_CLIP_layers"] = [len(getattr(clip_visual[1], f"layer{i}")) for i in range(1, 5)]
    facts["MSDB_DINOv2_n_blocks"] = len(bbc.feature_extractors["DINOv2"].features[1].blocks)
    facts["MSDB_DINOv2_has_final_norm_applied_to_hooked_outputs"] = False
    print("MSDB channels", facts["MSDB_backbone_outputs_at_384x512"], facts["MSDB_total_channels"], flush=True)
    with torch.no_grad():
        cb = torch.full((1, 384, 512), -math.log(384 * 512))
        t0 = time.time(); ld = msdb(xh.clone(), cb, pixel_per_dva=17.5, dataset=None); facts["MSDB_full_forward_time_s_at_384x512"] = round(time.time() - t0, 1)
    facts["MSDB_output_shape"] = list(ld.shape)
    facts["MSDB_readout_grid_at_384x512"] = [math.ceil(384 / DGM._READOUT_FACTOR), math.ceil(512 / DGM._READOUT_FACTOR)]
    facts["MSDB_blur_sigma_px_at_saliency_map_resolution"] = round(facts["MSDB_checkpoint"]["sigma_mean_used_when_dataset_None"] * 17.5 / DGM._SALIENCY_MAP_FACTOR, 3)
    facts["MSDB_blur_sigma_display_px"] = round(facts["MSDB_blur_sigma_px_at_saliency_map_resolution"] * 2 * 2, 2)  # 1/2 map of a 1/2-res input
    # scale inputs actually fed to the backbones in our runs (input 512 x 384 at 17.5 px/dva)
    facts["MSDB_backbone_input_sizes_in_our_runs"] = dict(
        pixel_per_dva={str(p): [math.ceil(384 * p / 17.5), math.ceil(512 * p / 17.5)] for p in DGM._PIXEL_PER_DVA_SCALES},
        size={str(s): [math.ceil(384 * s / 512), math.ceil(512 * s / 512)] for s in DGM._SIZE_SCALES})
    del msdb, bbc, clip_visual, outs
    print("MSDB done", facts["MSDB_full_forward_time_s_at_384x512"], flush=True)

    # ---------------- 2b. IIE backbones
    iie = DG2E.DeepGazeIIE(pretrained=True); iie.eval()
    facts["IIE_backbones"] = []
    for b, sub in enumerate(iie.models):
        fe = sub.features
        with torch.no_grad():
            t0 = time.time(); outs = fe(xh.clone()); dt = time.time() - t0
        rfs = [rf_walk(fe.features, t) for t in fe.targets]
        facts["IIE_backbones"].append(dict(type=DG2E.BACKBONES[b]["type"], targets=fe.targets,
                                           shapes_at_384x512=[list(o.shape) for o in outs], total_channels=int(sum(o.shape[1] for o in outs)),
                                           theoretical_rf_px_backbone_input=[r for r, _ in rfs], stride_px_backbone_input=[jj for _, jj in rfs],
                                           forward_time_s_at_384x512=round(dt, 2),
                                           n_backbone_params=int(sum(p.numel() for p in fe.features.parameters())),
                                           downsample=sub.downsample, readout_factor=sub.readout_factor, saliency_map_factor=sub.saliency_map_factor,
                                           readout_grid_for_768x1024_display=[math.ceil(768 / sub.downsample / sub.readout_factor), math.ceil(1024 / sub.downsample / sub.readout_factor)]))
        print("IIE", b, facts["IIE_backbones"][-1]["type"], facts["IIE_backbones"][-1]["total_channels"], rfs, round(dt, 2), flush=True)
    full = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1))[None]
    with torch.no_grad():
        cb = torch.full((1, 768, 1024), -math.log(768 * 1024))
        t0 = time.time(); ld = iie(full, cb); facts["IIE_full_forward_time_s_at_768x1024"] = round(time.time() - t0, 1)
    facts["IIE_output_shape"] = list(ld.shape)
    json.dump(facts, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
