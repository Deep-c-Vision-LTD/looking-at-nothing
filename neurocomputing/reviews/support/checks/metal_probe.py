"""Design review probe (no scoring): does the CPU import and model-load path of the re-analysis open a Metal client?
Stub first (as score_rigid.py and recommon.py do), then transformers AutoModel.from_pretrained (offline, CPU) and
import of deepgaze_pytorch (no model built). Prints the PID and sleeps so ioreg can be queried from outside."""
import os, sys, time
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "_stub"))   # <re-analysis folder>/_stub
import ab_objective as AB
assert AB.DEV == "cpu"
import torch
torch.set_num_threads(4)
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").eval()
import deepgaze_pytorch  # noqa: F401
print("pid", os.getpid(), "loaded", type(m).__name__, flush=True)
time.sleep(25)
print("exit", flush=True)
