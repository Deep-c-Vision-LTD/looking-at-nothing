"""MSI-Net (Kroner et al., 2020; Hugging Face alexanderkroner/MSI-Net, original weights) maps for a stimulus set,
with the authors' preprocessing (aspect-preserving resize to one of 320x320 / 240x320 / 320x240, padding).
Needs TensorFlow 2 (Keras 3). Usage: python msinet_maps.py kanizsa|ehrenstein -> stimuli/<set>/maps_msinet/*.png"""
import json, os, sys
import numpy as np, cv2, tensorflow as tf, keras
from huggingface_hub import snapshot_download
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "stimuli", sys.argv[1]); out = os.path.join(D, "maps_msinet"); os.makedirs(out, exist_ok=True)
items = json.load(open(os.path.join(D, "items.json")))
model = keras.layers.TFSMLayer(snapshot_download("alexanderkroner/MSI-Net"), call_endpoint="serving_default")
def target(shape):
    ar = shape[0] / shape[1]; m = min(abs(ar - 1), abs(ar - .75), abs(ar - 4 / 3))
    return (320, 320) if m == abs(ar - 1) else ((240, 320) if m == abs(ar - .75) else (320, 240))
for it in items:
    dst = os.path.join(out, it["name"] + ".png")
    if os.path.exists(dst): continue
    rgb = cv2.cvtColor(cv2.imread(os.path.join(D, "img", it["name"] + ".png")), cv2.COLOR_BGR2RGB)
    shp = rgb.shape[:2]; t = target(shp)
    x = tf.image.resize(tf.expand_dims(rgb.astype(np.float32), 0), t, preserve_aspect_ratio=True)
    v, h = t[0] - x.shape[1], t[1] - x.shape[2]
    x = tf.pad(x, [[0, 0], [v // 2, v - v // 2], [h // 2, h - h // 2], [0, 0]])
    p = model(x); p = next(iter(p.values())) if isinstance(p, dict) else p
    p = p[:, v // 2: p.shape[1] - (v - v // 2), h // 2: p.shape[2] - (h - h // 2), :]
    p = tf.image.resize(p, shp).numpy().squeeze()
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    cv2.imwrite(dst, (p * 255).astype(np.uint8))
print("done", len(os.listdir(out)))
