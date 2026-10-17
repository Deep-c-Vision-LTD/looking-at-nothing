"""
Mechanism analysis: one ASPP readout per layer of a frozen vision transformer, all trained jointly from one forward
pass per batch (loss KL + 2(1 - CC); SALICON + MIT1003[:-200] + CAT2000; 336 px; 6 epochs; batch 32; AdamW 5e-4;
cosine; seed 0; final epoch). Each readout is then scored on
  (a) the 200 held-out MIT1003 images (336 px, centre bias on)                     -> results/layers/<bb>/sweep_results.json
  (b) Kanizsa figures, support ratio 0.3/0.5/0.7, illusory vs 180 and 90 deg controls (same file)
  (c) all Ehrenstein figures, illusory vs control_end (primary) and control_mid     -> results/layers/<bb>/ehr_results.json
with the centre bias off, at 336/448/560/672 px.
Backbones: dinov2 = facebook/dinov2-base (hidden_states 2..12), clip = openai/clip-vit-large-patch14 and
siglip = google/siglip-large-patch16-256 (hidden_states 4..24). hidden_states[L] is the output of block L,
before any final layer norm.
Usage: python layer_sweep.py <dinov2|clip|siglip> train|eval|ehr   (train needs prepare_data.py first)
Released readouts: weights/layers/<bb>/readout_L<L>.pt (GitHub release).
"""
import json, os, sys, time
import numpy as np, cv2, torch
from common import Readout, loss_map, rel_inside, build_mit1003, CC, NSS, AUC_Judd, DEV, WEIGHTS, STIM, RESULTS

S = 336
CFG = {
    "dinov2": dict(hf="facebook/dinov2-base", layers=(2, 4, 6, 8, 10, 12), cls=True,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "clip": dict(hf="openai/clip-vit-large-patch14", layers=(4, 8, 12, 16, 20, 24), cls=True,
                 mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    "siglip": dict(hf="google/siglip-large-patch16-256", layers=(4, 8, 12, 16, 20, 24), cls=False,
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
SIZES = (336, 448, 560, 672)

def setup(bb_name):
    global BB, C, WDIR, RDIR, MEAN, STD
    BB, C = bb_name, CFG[bb_name]
    WDIR = os.path.join(WEIGHTS, "layers", BB); RDIR = os.path.join(RESULTS, "layers", BB)
    os.makedirs(WDIR, exist_ok=True); os.makedirs(RDIR, exist_ok=True)
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

def load_readouts(bb):
    R = {}
    for L in C["layers"]:
        m = Readout(bb.config.hidden_size).to(DEV)
        m.load_state_dict(torch.load(os.path.join(WDIR, f"readout_L{L}.pt"), map_location=DEV)); m.eval(); R[L] = m
    return R

def train(epochs=6, bs=32, lr=5e-4):
    from prepare_data import WORK
    bb = load_backbone(); cin = bb.config.hidden_size
    meta = json.load(open(f"{WORK}/meta.json")); its = meta["items"]; maps = np.load(f"{WORK}/maps.npy", mmap_mode="r")
    keep = [i for i in range(len(its)) if meta["nfix"][i] > 0]
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
                loss = loss_map(m(fs[L], (S, S)), gt)
                O[L].zero_grad(); loss.backward(); O[L].step(); tot[L] += loss.item()
        for s in SC.values(): s.step()
        print(f"{BB} epoch {ep} " + " ".join(f"L{L} {v / nb:.4f}" for L, v in tot.items()) + f" {time.time() - t0:.0f}s", flush=True)
    for L, m in R.items():
        torch.save(m.state_dict(), os.path.join(WDIR, f"readout_L{L}.pt"))

@torch.no_grad()
def score_set(bb, R, D, sel):
    out = {}
    for s in SIZES:
        res = {L: {} for L in R}
        for it in sel:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
            a = np.load(os.path.join(D, "img", it["name"] + "_aoi.npz"))
            fs = feats(bb, prep(rgb, s)[None])
            for L, m in R.items():
                m.use_cb = False
                sal = cv2.resize(torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
                res[L][it["name"]] = rel_inside(sal, a["inside"], a["ind"])
        out[s] = res
    return out

@torch.no_grad()
def evaluate():
    bb = load_backbone(); R = load_readouts(bb)
    out = {"mit1003": {}, "kanizsa": {}}
    acc = {L: {"CC": [], "NSS": [], "AUC": []} for L in R}
    for p in build_mit1003()[-200:]:
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
    D = STIM["kanizsa"]; items = json.load(open(os.path.join(D, "items.json")))
    sel = [it for it in items if it["sr"] in (0.3, 0.5, 0.7) and it["cond"] in ("illusory", "control", "control90")]
    for s, res in score_set(bb, R, D, sel).items():
        for L in R:
            ill = [it["name"] for it in sel if it["cond"] == "illusory"]
            d180 = np.array([res[L][n] - res[L][n.replace("_illusory", "_control")] for n in ill])
            d90 = np.array([res[L][n] - res[L][n.replace("_illusory", "_control90")] for n in ill])
            out["kanizsa"][f"L{L}_s{s}"] = dict(d180=round(float(d180.mean()), 4), p180=round(float(np.mean(d180 > 0)), 3),
                                                 d90=round(float(d90.mean()), 4), p90=round(float(np.mean(d90 > 0)), 3), raw=res[L])
            print(f"{BB} Kanizsa layer {L:2d} size {s}: illusory-control180 {d180.mean():+.3f} ({np.mean(d180 > 0) * 100:.0f}%)", flush=True)
    json.dump(out, open(os.path.join(RDIR, "sweep_results.json"), "w"), indent=1)

@torch.no_grad()
def evaluate_ehr():
    bb = load_backbone(); R = load_readouts(bb)
    D = STIM["ehrenstein"]; items = json.load(open(os.path.join(D, "items.json")))
    sel = [it for it in items if it["cond"] in ("illusory", "control_end", "control_mid")]
    out = {}
    for s, res in score_set(bb, R, D, sel).items():
        for L in R:
            ill = [it["name"] for it in sel if it["cond"] == "illusory"]
            de = np.array([res[L][n] - res[L][n.replace("_illusory", "_control_end")] for n in ill])
            dm = np.array([res[L][n] - res[L][n.replace("_illusory", "_control_mid")] for n in ill])
            out[f"L{L}_s{s}"] = dict(d_end=round(float(de.mean()), 4), p_end=round(float(np.mean(de > 0)), 3),
                                     d_mid=round(float(dm.mean()), 4), p_mid=round(float(np.mean(dm > 0)), 3), raw=res[L])
            print(f"{BB} Ehrenstein layer {L:2d} size {s}: illusory-control_end {de.mean():+.3f} ({np.mean(de > 0) * 100:.0f}%)", flush=True)
    json.dump(out, open(os.path.join(RDIR, "ehr_results.json"), "w"), indent=1)

if __name__ == "__main__":
    setup(sys.argv[1])
    {"train": train, "eval": evaluate, "ehr": evaluate_ehr}[sys.argv[2]]()
