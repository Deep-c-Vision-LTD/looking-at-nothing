"""
Decomposition comparison model: xSoyelNet-ML (DINOv2-B hidden_states[7] and [11], ASPP readout paths.XSOYELNET_ML,
not distributed; TTA over 280/336/392/448 px and flips, centre bias off) on the 60 Kanizsa illusory and control180
images, through the code path of models/run_v2.py (scales.sal_at), on CPU.
Maps saved: maps/ml/<image>.npz (key sal, float16, 768 x 1024). Scores: scores/ml.json (AOI masses per image).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_d import *  # noqa
torch = torch_setup()
import ab_objective as AB  # noqa: E402
AB.DEV = "cpu"
import ab_layers as AL  # noqa: E402
AL.DEV = "cpu"
import scales as SC  # noqa: E402
SC.DEV = "cpu"
from transformers import AutoModel  # noqa: E402

OUT = os.path.join(SCORES, "ml.json")
MD = os.path.join(MAPS, "ml"); os.makedirs(MD, exist_ok=True)


def main():
    AB._bb = AutoModel.from_pretrained("facebook/dinov2-base").to("cpu").eval()
    m = AL.Readout(1536).to("cpu")
    m.load_state_dict(torch.load(str(LP.XSOYELNET_ML), map_location="cpu")); m.eval()
    assert next(m.parameters()).device.type == "cpu" and next(AB._bb.parameters()).device.type == "cpu"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    names = [n for c in kanizsa_configs() for n in (c["ill"], c["ctrl"])]
    for i, name in enumerate(names):
        if name in res:
            continue
        rgb = read_rgb(name)
        sal = np.mean([SC.sal_at(m, (7, 11), rgb, s, False, fl) for s in (280, 336, 392, 448) for fl in (False, True)], 0)
        res[name] = masses(sal, read_aoi(name))
        np.savez_compressed(os.path.join(MD, name + ".npz"), sal=sal.astype(np.float16))
        if (i + 1) % 10 == 0:
            save_json(res, OUT); print(f"ml {i + 1}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    save_json(res, OUT)
    print("ML DONE", flush=True)


if __name__ == "__main__":
    main()
