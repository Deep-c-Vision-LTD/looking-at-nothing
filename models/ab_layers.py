"""
xSoyelNet-ML recipe: an ASPP readout of DINOv2-B blocks 7 and 11 (the two DINOv2 tensors that DeepGaze MSDB reads).

Variants, trained together from one frozen DINOv2-base forward pass per batch (features computed on the fly):
  M610: Hugging Face hidden_states[7] + hidden_states[11] concatenated (1536 channels) = xSoyelNet-ML
  M6:   hidden_states[7] only (768 channels)
Recipe: 336 x 336 input, ASPP readout + learned centre bias, loss KL + 2(1 - CC) (ab_objective.loss_A), the 12,803
training images of ab_objective (SALICON + MIT1003[:-200] + CAT2000), 8 epochs, batch 32, AdamW 5e-4 (weight decay
1e-4), cosine schedule, seed 0, final epoch. hidden_states[k] is the output of block k (1-based), before the final
layer norm; in 0-based timm/DINOv2 numbering these are blocks 6 and 10.
Usage: python ab_layers.py train   -> paths.WEIGHTS/readout_M610.pt, readout_M6.pt
"""
import json, os, sys, time
import numpy as np, cv2, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_objective as AB                        # noqa: E402
import paths as LP                               # noqa: E402
from src.model_aspp import ASPPModule            # noqa: E402

WORK = AB.WORK; S, G, DEV = AB.S, AB.G, AB.DEV
LAYERS = {"M610": (7, 11), "M6": (7,)}


class Readout(torch.nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.aspp = ASPPModule(cin, 128, (1, 3, 6, 12)); self.head = torch.nn.Conv2d(128, 1, 1)
        self.center_bias = torch.nn.Parameter(torch.zeros(1, 1, 16, 16)); self.use_cb = True
    def forward(self, f, out_hw=(S, S)):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def feats(batch):
    hs = AB.backbone()(pixel_values=batch.to(DEV), interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
    def grid(t):
        t = t[:, 1:, :]; return t.transpose(1, 2).reshape(t.shape[0], -1, G, G).contiguous()
    return {k: torch.cat([grid(hs[i]) for i in idx], 1) for k, idx in LAYERS.items()}


def train(epochs=8, bs=32, lr=5e-4):
    meta = json.load(open(f"{WORK}/meta.json")); its = meta["items"]
    maps = np.load(f"{WORK}/maps.npy", mmap_mode="r")
    keep = [i for i in range(len(its)) if len(meta["fixes"][i]) > 0]
    torch.manual_seed(0); rng = np.random.RandomState(0)
    models = {k: Readout(768 * len(v)).to(DEV) for k, v in LAYERS.items()}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4) for k, m in models.items()}
    scheds = {k: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs) for k, o in opts.items()}
    for ep in range(epochs):
        order = rng.permutation(keep); tot = {k: 0.0 for k in models}; t0 = time.time()
        for m in models.values(): m.train()
        nb = len(order) // bs
        for b in range(nb):
            idx = np.sort(order[b * bs:(b + 1) * bs])
            x = torch.stack([AB.prep(cv2.cvtColor(cv2.imread(its[i]["image"]), cv2.COLOR_BGR2RGB)) for i in idx])
            fs = feats(x)
            gt = torch.from_numpy(np.asarray(maps[idx], np.float32) / 255.0)[:, None].to(DEV)
            for k, m in models.items():
                loss = AB.loss_A(m(fs[k]), gt)
                opts[k].zero_grad(); loss.backward(); opts[k].step(); tot[k] += loss.item()
            if b % 100 == 0:
                print(f"  ep {ep} batch {b}/{nb} {time.time() - t0:.0f}s", flush=True)
        for s in scheds.values(): s.step()
        print(f"epoch {ep} " + " ".join(f"{k} {v / nb:.4f}" for k, v in tot.items()) + f" {time.time() - t0:.0f}s", flush=True)
    os.makedirs(LP.WEIGHTS, exist_ok=True)
    for k, m in models.items():
        torch.save(m.state_dict(), str(LP.WEIGHTS / f"readout_{k}.pt"))


if __name__ == "__main__":
    {"train": train}[sys.argv[1]]()
