"""
test_features_cpu.py: feature-path checks for the new feature sets of G3 and A14 (design review, finding 6b), run on
the CPU on one synthetic noise image (numpy RandomState(0), 768 x 1024 x 3, uint8). No stimulus, no readout, no score.
  1. dinov2ms_L7L11[:, 768:] equals dinov2ms_L11 bit for bit (same forward pass, channel order 7 then 11);
  2. dinov2ms_L11 equals hidden_states[11] of the published DINOv2-B extractor (v2/readout/common.Extractor, the
     convention-check path hidden_index=[11]) bit for bit, and dinov2ms_L7L11[:, :768] equals its hidden_states[7];
  3. the published GPU-side path (v3/gpu/fullcommon.feats) and the published extractor agree on dinov2_L12 and
     dinov2_L6, so check 2 ties the new path to the code that produced the readout-swap features;
  4. dinov2reg (DINOv2-B with registers): 4 register tokens, CLS + registers dropped, 768 channels on the token grid;
  at 336 and 448 px. The process checks with ioreg that its own PID holds no Metal client.
Usage: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 nice -n 10 python -B test_features_cpu.py > tests_features_output.txt
"""
import os, subprocess, sys, time
for k, v in (("V3_DEVICE", "cpu"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
             ("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))):
    os.environ[k] = v
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gpu"))
import recommon as RC                      # noqa: E402  (imports the CPU stub of ab_objective first)
import numpy as np, torch                  # noqa: E402
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
assert RC.DEVICE == "cpu" and RC._AB.DEV == "cpu"
FC, C2 = RC.FC, RC.C2

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def metal_clients():
    out = subprocess.run(["ioreg", "-l", "-w0", "-r", "-c", "AGXDeviceUserClient"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if f'"pid {os.getpid()},' in ln and "IOUserClientCreator" in ln]


def main():
    t0 = time.time()
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}; pid {os.getpid()}; device {RC.DEVICE}")
    rgb = np.random.RandomState(0).randint(0, 256, (768, 1024, 3)).astype(np.uint8)
    ex_ms = RC.make_extractor("dinov2ms")
    ex_pub = C2.Extractor("dinov2")
    ex_reg = RC.make_extractor("dinov2reg")
    check("registers checkpoint has 4 register tokens", ex_reg.m.config.num_register_tokens == 4,
          f"num_register_tokens {ex_reg.m.config.num_register_tokens}")
    for s in (336, 448):
        g = s // 14
        x = ex_ms.prep(rgb, s)[None]
        f = RC.feats(ex_ms, x, s)
        check(f"{s} px: dinov2ms shapes (1, 1536, {g}, {g}) and (1, 768, {g}, {g})",
              tuple(f["dinov2ms_L7L11"].shape) == (1, 1536, g, g) and tuple(f["dinov2ms_L11"].shape) == (1, 768, g, g))
        check(f"{s} px: dinov2ms_L7L11[:, 768:] equals dinov2ms_L11 bit for bit",
              torch.equal(f["dinov2ms_L7L11"][:, 768:], f["dinov2ms_L11"]))
        pub = ex_pub([rgb], s, hidden_index=[7, 11])
        d11 = float((pub["hs11"] - f["dinov2ms_L11"]).abs().max()); d7 = float((pub["hs7"] - f["dinov2ms_L7L11"][:, :768]).abs().max())
        check(f"{s} px: dinov2ms_L11 equals hidden_states[11] of the published extractor bit for bit",
              torch.equal(pub["hs11"], f["dinov2ms_L11"]), f"max |d| {d11:.2e}")
        check(f"{s} px: dinov2ms_L7L11[:, :768] equals hidden_states[7] of the published extractor bit for bit",
              torch.equal(pub["hs7"], f["dinov2ms_L7L11"][:, :768]), f"max |d| {d7:.2e}")
        fp = FC.feats(ex_pub, ex_pub.prep(rgb, s)[None], s)
        check(f"{s} px: published GPU-side path (fullcommon.feats) equals the published extractor for dinov2_L12 and dinov2_L6",
              torch.equal(fp["dinov2_L12"], pub["dinov2_L12"]) and torch.equal(fp["dinov2_L6"], pub["dinov2_L6"]))
        fr = RC.feats(ex_reg, ex_reg.prep(rgb, s)[None], s)
        check(f"{s} px: dinov2reg_L12 and dinov2reg_L6 are (1, 768, {g}, {g}) after dropping CLS and 4 registers",
              all(tuple(fr[k].shape) == (1, 768, g, g) for k in ("dinov2reg_L12", "dinov2reg_L6")))
    mc = metal_clients()
    check("this process holds no Metal (AGX) client", not mc, "; ".join(mc) if mc else "ioreg: none for this pid")
    print(f"\n{len(RES)} checks, {sum(not r for r in RES)} failed, {time.time() - t0:.0f} s")
    return sum(not r for r in RES)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
