"""MSI-Net (Kroner et al., 2020; Hugging Face alexanderkroner/MSI-Net, original weights) saliency maps for a stimulus
set, with the authors' preprocessing (aspect-preserving resize to one of 320 x 320 / 240 x 320 / 320 x 240, padding),
min-max normalised and written as uint8 PNGs. The SavedModel is loaded as a Keras 3 TFSMLayer (serving_default
endpoint). Needs TensorFlow 2 with Keras 3 (a separate environment; see scoring/run_msinet_v3.py).
Usage: python msinet_maps.py <items.json> <img_dir> <out_dir>"""
import json, os, sys
import numpy as np, cv2, tensorflow as tf, keras
from huggingface_hub import snapshot_download
items, img_dir, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
model = keras.layers.TFSMLayer(snapshot_download("alexanderkroner/MSI-Net"), call_endpoint="serving_default")
def target(shape):
    ar = shape[0] / shape[1]; m = min(abs(ar - 1), abs(ar - .75), abs(ar - 4 / 3))
    return (320, 320) if m == abs(ar - 1) else ((240, 320) if m == abs(ar - .75) else (320, 240))
for it in items:
    dst = os.path.join(out, it["name"] + ".png")
    if os.path.exists(dst): continue
    rgb = cv2.cvtColor(cv2.imread(os.path.join(img_dir, it["name"] + ".png")), cv2.COLOR_BGR2RGB)
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
