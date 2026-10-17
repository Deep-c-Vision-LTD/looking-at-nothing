"""Sensitivity of the Kanizsa illusion index to the real_filled grey level (DeepGaze IIE part).
Renders real_filled with the figure fill at 25% and 75% of the inducer-background contrast (stim_v3 uses grey 128 = 50%),
same renderer (stimuli/stimuli_v3.kanizsa_hi), runs DeepGaze IIE exactly as gate_kanizsa.load_models (CPU, 3 threads),
saves float16 maps (sal = map / max, scale = max) and per-image r on the inside AOI.
Writes stimuli/fill_variants/img/*.png (paths.FILL_VARIANTS), <results>/v2/verify/fill_variants/iie_scores.json and the
maps in <results>/v2/verify/fill_variants/maps_iie/ (not distributed)."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "3"; os.environ["MKL_NUM_THREADS"] = "3"
import numpy as np, torch
torch.set_num_threads(3)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "stimuli")); sys.path.insert(0, os.path.join(REPO, "models"))
import stimuli_v3 as S3
import gate_kanizsa as gk
LP = S3.LP
STIM, IMG = str(LP.STIM_V3), str(LP.FILL_VARIANTS / "img")
OUT = str(LP.V2 / "verify" / "fill_variants")
for d in (IMG, f"{OUT}/maps_iie"):
    os.makedirs(d, exist_ok=True)
fn = gk.load_models(["iie"])["DeepGaze IIE"]; torch.set_num_threads(3)
res = {}; t0 = time.time(); k = 0
for ill, p in S3.configs():
    if ill != "kanizsa": continue
    a = np.load(f"{STIM}/img/{S3.k_name(p['shape'], p['sr'], p['pol'], p['pos'], 'illusory')}_aoi.npz")
    for c in (0.25, 0.75):
        S3.GREY = int(round(p["bg"] + c * (p["fg"] - p["bg"])))
        img = S3.down(S3.render("kanizsa", p, "real_filled")).convert("RGB")
        name = S3.k_name(p["shape"], p["sr"], p["pol"], p["pos"], f"real_filled_c{int(c * 100)}")
        img.save(f"{IMG}/{name}.png")
        sal = fn(np.array(img)).astype(np.float64)
        np.savez_compressed(f"{OUT}/maps_iie/{name}.npz", sal=(sal / sal.max()).astype(np.float16), scale=np.float64(sal.max()))
        res[name] = dict(r_inside=float(gk.rel_inside(sal, a["inside"], a["ind"])), grey=S3.GREY)
        k += 1
        if k % 10 == 0: print(f"{k}/120 {time.time() - t0:.0f}s", flush=True)
S3.GREY = 128
json.dump(res, open(f"{OUT}/iie_scores.json", "w"), indent=1)
print("done", time.time() - t0)
