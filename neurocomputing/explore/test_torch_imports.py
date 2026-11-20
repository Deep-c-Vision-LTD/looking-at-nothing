"""
Import smoke test of the CPU torch stack used by mapsrc.py (no weights, no data): the ab_objective stub first, the
repository gate_kanizsa, gpu/recommon.py with V3_DEVICE=cpu, sweep.py reloaded per backbone, and an untrained readout head
of each architecture run once on random features. 2 threads, a few seconds. The process checks itself with
excommon.metal_guard() at the end (no Metal client allowed); also check its pid with ioreg within 30 s of starting it.
Usage: OMP_NUM_THREADS=2 $PY -B explore/test_torch_imports.py
"""
import os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import excommon as X                                                      # noqa: E402
import mapsrc as M                                                        # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""), flush=True)


def main():
    t0 = time.time()
    print(f"test_torch_imports.py pid {os.getpid()}", flush=True)
    T = M.torch_env()
    torch, R = T["torch"], T["R"]
    check("stub ab_objective loaded (DEV = cpu)", T["AB"].DEV == "cpu" and "_stub" in T["AB"].__file__)
    check("recommon on the CPU", R.DEVICE == "cpu")
    check("torch threads = OMP_NUM_THREADS", torch.get_num_threads() == 2, torch.get_num_threads())
    import gate_kanizsa
    check("gate_kanizsa is the repository copy", os.path.join("looking-at-nothing", "models") in gate_kanizsa.__file__,
          gate_kanizsa.__file__)
    check("render_rigid's stimcode modules stay the stimcode copies",
          "stimcode" in sys.modules["stimuli_v2"].__file__ and "stimcode" in sys.modules["paths"].__file__)
    s1 = M.sweep_module("dinov2")
    s2 = M.sweep_module("clip")
    check("sweep.py reloaded per backbone (BB follows the request)", s1 is s2 and s2.BB == "clip" and s2.DEV == "cpu")
    s3 = M.sweep_module("siglip")
    check("and again for siglip", s3.BB == "siglip" and s3.C["cls"] is False)
    with torch.no_grad():
        for arch in ("R1n", "R1", "R2", "R3", "R4"):
            m = R.make_head(arch, 768).eval()                         # R1n: FixedNorm defaults (mu 0, sd 1)
            m.use_cb = False
            z = m(torch.randn(2, 768, 24, 24), (336, 336))
            check(f"untrained {arch} head runs on random features at 336 px", tuple(z.shape) == (2, 1, 336, 336),
                  tuple(z.shape))
    rd = s1.Readout(768).eval()
    rd.use_cb = False
    with torch.no_grad():
        z = rd(torch.randn(1, 768, 24, 24), (448, 448))
    check("sweep Readout runs at 448 px", tuple(z.shape) == (1, 1, 448, 448))
    check("this process holds no Metal client", X.metal_guard() in (True, None))
    print(f"{len(RES)} checks, {RES.count(False)} failed, {time.time() - t0:.1f}s")
    sys.exit(1 if not all(RES) else 0)


if __name__ == "__main__":
    main()
