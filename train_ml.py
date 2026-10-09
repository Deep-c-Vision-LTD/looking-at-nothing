"""
xSoyelNet-ML recipe: frozen DINOv2-B, blocks 6 and 10 (Hugging Face hidden_states[7] and [11]) concatenated
(1536 channels), ASPP readout + learned centre bias (7.36M parameters), loss KL + 2(1 - CC), 336 px input,
8 epochs, batch 32, AdamW 5e-4 (weight decay 1e-4), cosine schedule, seed 0, final epoch.
Needs prepare_data.py first. Writes weights/xsoyelnet_ml.pt.  Usage: python train_ml.py
"""
import json, os, time
import numpy as np, cv2, torch
from transformers import AutoModel
from common import Readout, loss_map, DEV, WEIGHTS
from prepare_data import WORK, S

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
IDX = (7, 11)

def prep(rgb):
    x = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))

def main(epochs=8, bs=32, lr=5e-4):
    bb = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    meta = json.load(open(f"{WORK}/meta.json")); its = meta["items"]; maps = np.load(f"{WORK}/maps.npy", mmap_mode="r")
    keep = [i for i in range(len(its)) if meta["nfix"][i] > 0]
    torch.manual_seed(0); rng = np.random.RandomState(0)
    m = Readout(768 * len(IDX)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        m.train(); order = rng.permutation(keep); tot, t0, nb = 0.0, time.time(), len(order) // bs
        for b in range(nb):
            idx = np.sort(order[b * bs:(b + 1) * bs])
            x = torch.stack([prep(cv2.cvtColor(cv2.imread(its[i]["image"]), cv2.COLOR_BGR2RGB)) for i in idx]).to(DEV)
            with torch.no_grad():
                hs = bb(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
                f = torch.cat([hs[i][:, 1:, :].transpose(1, 2).reshape(len(idx), -1, S // 14, S // 14).contiguous() for i in IDX], 1)
            gt = torch.from_numpy(np.asarray(maps[idx], np.float32) / 255.0)[:, None].to(DEV)
            loss = loss_map(m(f, (S, S)), gt); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step(); print(f"epoch {ep} loss {tot / nb:.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(WEIGHTS, exist_ok=True); torch.save(m.state_dict(), os.path.join(WEIGHTS, "xsoyelnet_ml.pt"))

if __name__ == "__main__":
    main()
