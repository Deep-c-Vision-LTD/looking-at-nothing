"""MSI-Net maps for stim_v3 with models/msinet_maps.py (original weights, authors' preprocessing), run in a TensorFlow 2 /
Keras 3 environment on the CPU with 3 threads. Writes uint8 PNG maps to <results>/v2/scores/maps_msinet_png/, which
score_v3.py msinet then scores.
Usage: <tensorflow python> scoring/run_msinet_v3.py"""
import os, runpy, sys
for k, v in (("OMP_NUM_THREADS", "3"), ("MKL_NUM_THREADS", "3"), ("TF_NUM_INTRAOP_THREADS", "3"),
             ("TF_NUM_INTEROP_THREADS", "1"), ("HF_HUB_OFFLINE", "1"), ("CUDA_VISIBLE_DEVICES", "")):
    os.environ[k] = os.environ.get(k, v) if k.endswith("OFFLINE") else v      # offline by default; =0 to download
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
import paths as LP  # noqa: E402
sys.argv = ["msinet_maps.py", str(LP.STIM_V3 / "items.json"), str(LP.STIM_V3 / "img"), str(LP.V2 / "scores" / "maps_msinet_png")]
print("devices:", tf.config.get_visible_devices(), flush=True)
runpy.run_path(os.path.join(REPO, "models", "msinet_maps.py"), run_name="__main__")
