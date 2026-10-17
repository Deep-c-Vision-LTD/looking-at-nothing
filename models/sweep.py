"""
Layer sweep: one ASPP readout (the R4 design) per selected block of a frozen vision transformer, all trained jointly
from a single frozen forward pass per batch (loss KL + 2(1 - CC); the 12,803 training images of ab_objective:
SALICON + MIT1003[:-200] + CAT2000; 336 x 336 input; 6 epochs; batch 32; AdamW 5e-4; cosine; seed 0; final epoch).
Evaluation of every readout: (a) the 200 held-out MIT1003 images (336 px, centre bias on), and (b) the first-version
Kanizsa and Ehrenstein sets (legacy/stimuli) at 336/448/560/672 px with the centre bias off. The paper's Appendix B
scores the same readouts on stim_v3 and stim_alt through scoring/score_v3.py and scoring/score_alt.py
(keys sweep:<backbone>), which import prep, feats, Readout and load_backbone from here.
Backbones: dinov2 = facebook/dinov2-base (12 blocks), clip = openai/clip-vit-large-patch14 (24 layers),
siglip = google/siglip-large-patch16-256 (24 layers). hidden_states[L] is the output of block L, before any final
layer norm; the CLS token is dropped (SigLIP has none).
Readouts: paths.LAYER_READOUTS/<bb>/readout_L<L>.pt (the 18 readouts of release v1.0, layer_readouts.zip, were trained
by this code). Evaluation output: paths.WORK/sweep/<bb>/{sweep_results,ehr_results}.json (the published copies are
legacy/results/layers/<bb>/).
Usage: python sweep.py <dinov2|clip|siglip> train|eval|ehr
"""
import json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ab_objective as AB, gate_kanizsa as gk
import paths as LP
from src.model_aspp import ASPPModule
from src.data import build_mit1003
from src.metrics import CC, NSS, SIM, AUC_Judd

DEV, S = AB.DEV, 336
CFG = {
    "dinov2": dict(hf="facebook/dinov2-base", layers=(2, 4, 6, 8, 10, 12), cls=True,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "clip": dict(hf="openai/clip-vit-large-patch14", layers=(4, 8, 12, 16, 20, 24), cls=True,
                 mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    "siglip": dict(hf="google/siglip-large-patch16-256", layers=(4, 8, 12, 16, 20, 24), cls=False,
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
BB = sys.argv[1]; C = CFG[BB]
WORK = str(LP.LAYER_READOUTS / BB)                           # readout_L<L>.pt
OUTD = str(LP.WORK / "sweep" / BB)                            # evaluation results
MEAN, STD = np.array(C["mean"], np.float32), np.array(C["std"], np.float32)

def load_backbone():
    if BB == "dinov2":
        from transformers import AutoModel; m = AutoModel.from_pretrained(C["hf"])
    elif BB == "clip":
        from transformers import CLIPVisionModel; m = CLIPVisionModel.from_pretrained(C["hf"])
    else:
        from transformers import SiglipVisionModel; m = SiglipVisionModel.from_pretrained(C["hf"])
    return m.to(DEV).eval()

def prep(rgb, s=S):
    x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))

@torch.no_grad()
def feats(bb, x):
    hs = bb(pixel_values=x.to(DEV), interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
    out = {}
    for L in C["layers"]:
        t = hs[L][:, 1:, :] if C["cls"] else hs[L]
        g = int(round(t.shape[1] ** 0.5))
        out[L] = t.transpose(1, 2).reshape(t.shape[0], -1, g, g).contiguous()
    return out

class Readout(torch.nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.aspp = ASPPModule(cin, 128, (1, 3, 6, 12)); self.head = torch.nn.Conv2d(128, 1, 1)
        self.center_bias = torch.nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True
    def forward(self, f, out_hw):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)

def train(epochs=6, bs=32, lr=5e-4):
    bb = load_backbone(); cin = bb.config.hidden_size
    meta = json.load(open(f"{AB.WORK}/meta.json")); its = meta["items"]
    maps = np.load(f"{AB.WORK}/maps.npy", mmap_mode="r")
    keep = [i for i in range(len(its)) if len(meta["fixes"][i]) > 0]
    torch.manual_seed(0); rng = np.random.RandomState(0)
    R = {L: Readout(cin).to(DEV) for L in C["layers"]}
    O = {L: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4) for L, m in R.items()}
    SC = {L: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs) for L, o in O.items()}
    for ep in range(epochs):
        order = rng.permutation(keep); tot = {L: 0.0 for L in R}; t0 = time.time(); nb = len(order) // bs
        for m in R.values(): m.train()
        for b in range(nb):
            idx = np.sort(order[b * bs:(b + 1) * bs])
            x = torch.stack([prep(cv2.cvtColor(cv2.imread(its[i]["image"]), cv2.COLOR_BGR2RGB)) for i in idx])
            fs = feats(bb, x)
            gt = torch.from_numpy(np.asarray(maps[idx], np.float32) / 255.0)[:, None].to(DEV)
            for L, m in R.items():
                loss = AB.loss_A(m(fs[L], (S, S)), gt)
                O[L].zero_grad(); loss.backward(); O[L].step(); tot[L] += loss.item()
            if b % 100 == 0: print(f"  {BB} ep {ep} batch {b}/{nb} {time.time() - t0:.0f}s", flush=True)
        for s in SC.values(): s.step()
        print(f"{BB} epoch {ep} " + " ".join(f"L{L} {v / nb:.4f}" for L, v in tot.items()) + f" {time.time() - t0:.0f}s", flush=True)
    os.makedirs(WORK, exist_ok=True)
    for L, m in R.items():
        torch.save(m.state_dict(), f"{WORK}/readout_L{L}.pt")

@torch.no_grad()
def evaluate():
    bb = load_backbone(); cin = bb.config.hidden_size
    R = {}
    for L in C["layers"]:
        m = Readout(cin).to(DEV); m.load_state_dict(torch.load(f"{WORK}/readout_L{L}.pt", map_location=DEV)); m.eval(); R[L] = m
    out = {"mit1003": {}, "kanizsa": {}}
    acc = {L: {"CC": [], "NSS": [], "AUC": []} for L in R}
    for p in build_mit1003(f"{AB.SR}/mit1003")[-200:]:
        rgb = cv2.cvtColor(cv2.imread(p["image"]), cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
        fs = feats(bb, prep(rgb)[None])
        gt = cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        fx = (cv2.imread(p["fix"], cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
        for L, m in R.items():
            m.use_cb = True
            sal = cv2.resize(torch.sigmoid(m(fs[L], (S, S)))[0, 0].cpu().numpy(), (w, h))
            acc[L]["CC"].append(CC(sal, gt)); acc[L]["NSS"].append(NSS(sal, fx)); acc[L]["AUC"].append(AUC_Judd(sal, fx))
    out["mit1003"] = {str(L): {k: round(float(np.mean(v)), 4) for k, v in a.items()} for L, a in acc.items()}
    print(BB, "MIT1003 held-out", out["mit1003"], flush=True)
    D = str(LP.STIM_V2); items = json.load(open(os.path.join(D, "items.json")))
    sel = [it for it in items if it["sr"] in (0.3, 0.5, 0.7) and it["cond"] in ("illusory", "control", "control90")]
    for s in (336, 448, 560, 672):
        res = {L: {} for L in R}
        for it in sel:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
            a = np.load(os.path.join(D, "img", it["name"] + "_aoi.npz"))
            fs = feats(bb, prep(rgb, s)[None])
            for L, m in R.items():
                m.use_cb = False
                sal = cv2.resize(torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
                res[L][it["name"]] = gk.rel_inside(sal, a["inside"], a["ind"])
        for L in R:
            d180 = [res[L][it["name"]] - res[L][it["name"].replace("_illusory", "_control")] for it in sel if it["cond"] == "illusory"]
            d90 = [res[L][it["name"]] - res[L][it["name"].replace("_illusory", "_control90")] for it in sel if it["cond"] == "illusory"]
            out["kanizsa"][f"L{L}_s{s}"] = dict(d180=round(float(np.mean(d180)), 4), p180=round(float(np.mean(np.array(d180) > 0)), 3),
                                                 d90=round(float(np.mean(d90)), 4), p90=round(float(np.mean(np.array(d90) > 0)), 3), raw=res[L])
            print(f"{BB} layer {L:2d} size {s}: illus-ctrl180 {np.mean(d180):+.3f} ({np.mean(np.array(d180) > 0) * 100:.0f}%)  "
                  f"illus-ctrl90 {np.mean(d90):+.3f} ({np.mean(np.array(d90) > 0) * 100:.0f}%)", flush=True)
    os.makedirs(OUTD, exist_ok=True); json.dump(out, open(f"{OUTD}/sweep_results.json", "w"), indent=1)

@torch.no_grad()
def evaluate_ehr():
    """Same readouts on the first-version Ehrenstein set: illusory vs control_end (primary) and control_mid."""
    bb = load_backbone(); cin = bb.config.hidden_size
    R = {}
    for L in C["layers"]:
        m = Readout(cin).to(DEV); m.load_state_dict(torch.load(f"{WORK}/readout_L{L}.pt", map_location=DEV)); m.eval(); m.use_cb = False; R[L] = m
    D = str(LP.STIM_EHR); items = json.load(open(os.path.join(D, "items.json")))
    sel = [it for it in items if it["cond"] in ("illusory", "control_end", "control_mid")]
    out = {}
    for s in (336, 448, 560, 672):
        res = {L: {} for L in R}
        for it in sel:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
            a = np.load(os.path.join(D, "img", it["name"] + "_aoi.npz"))
            fs = feats(bb, prep(rgb, s)[None])
            for L, m in R.items():
                sal = cv2.resize(torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
                res[L][it["name"]] = gk.rel_inside(sal, a["inside"], a["ind"])
        for L in R:
            ill = [it["name"] for it in sel if it["cond"] == "illusory"]
            de = np.array([res[L][n] - res[L][n.replace("_illusory", "_control_end")] for n in ill])
            dm = np.array([res[L][n] - res[L][n.replace("_illusory", "_control_mid")] for n in ill])
            out[f"L{L}_s{s}"] = dict(d_end=round(float(de.mean()), 4), p_end=round(float(np.mean(de > 0)), 3),
                                     d_mid=round(float(dm.mean()), 4), p_mid=round(float(np.mean(dm > 0)), 3), raw=res[L])
            print(f"{BB} EHR layer {L:2d} size {s}: illus-ctrl_end {de.mean():+.3f} ({np.mean(de > 0) * 100:.0f}%)  "
                  f"illus-ctrl_mid {dm.mean():+.3f} ({np.mean(dm > 0) * 100:.0f}%)", flush=True)
    os.makedirs(OUTD, exist_ok=True); json.dump(out, open(f"{OUTD}/ehr_results.json", "w"), indent=1)

if __name__ == "__main__":
    {"train": train, "eval": evaluate, "ehr": evaluate_ehr}[sys.argv[2]]()
