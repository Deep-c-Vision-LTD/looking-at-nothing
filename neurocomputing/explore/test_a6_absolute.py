"""
Unit tests of explore/a6_absolute.py (RUN_PLAN.md S4) on synthetic inputs only: the information gain, the reproduction
gate, the per-image interval, the import guards, and (torch on the CPU, stub ab_objective, no checkpoint, no score file)
the centre-bias switch of untrained heads on random features. No project score, map, checkpoint or feature file is
read (analyze_re._load is wrapped to refuse every read). Writes only its output file (--out).
Usage: OMP_NUM_THREADS=2 V3_DEVICE=cpu $PY -B -I explore/test_a6_absolute.py --out explore/tests_a6_absolute_output.txt
Written 18 November 2026.
"""
import argparse, inspect, math, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402
import a6_absolute as A6                                                  # noqa: E402

AR = X.AR
RES, LINES = [], []


def _refuse(p):
    raise RuntimeError(f"test must not read project files: {p}")


AR._load = _refuse


def out(s):
    LINES.append(s); print(s, flush=True)


def check(name, ok, detail=""):
    RES.append(bool(ok))
    out(("PASS  " if ok else "FAIL  ") + name + (f"  {detail}" if detail != "" else ""))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "tests_a6_absolute_output.txt"))
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(3)
    out(f"test_a6_absolute.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    check("V3_DEVICE forced to cpu at import", os.environ.get("V3_DEVICE") == "cpu")
    check("PYTORCH_ENABLE_MPS_FALLBACK unset", "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ)
    check("torch not imported by the module import", "torch" not in sys.modules)
    src = inspect.getsource(A6.main)
    check("main: require_public before recommon, torch and any data", src.index("require_public") <
          min(src.index("import recommon"), src.index("import torch"), src.index("load_mit")))

    # information gain
    m = rng.rand(40, 50) + 0.1
    check("norm_log2 is a normalised log2 map", abs(np.sum(2.0 ** A6.norm_log2(m)) - 1) < 1e-12)
    fix = np.zeros((40, 50), bool); fix[rng.randint(0, 40, 30), rng.randint(0, 50, 30)] = True
    check("IG = 0 when the model equals the baseline", abs(A6.info_gain(m, fix, A6.norm_log2(m))) < 1e-12)
    check("IG = 0 for uniform model and uniform baseline", abs(A6.info_gain(np.ones((40, 50)), fix, A6.norm_log2(np.ones((40, 50))))) < 1e-12)
    aa, bb = 5.0, 1.0
    mm = np.where(fix, aa, bb); k, N = fix.sum(), fix.size
    ig_true = math.log2(aa / (k * aa + (N - k) * bb) * N)
    check("IG analytic value (model a on fixations, b elsewhere; uniform baseline)",
          abs(A6.info_gain(mm, fix, A6.norm_log2(np.ones((40, 50)))) - ig_true) < 1e-9, f"{ig_true:.6f}")
    check("IG antisymmetric in model and baseline",
          abs(A6.info_gain(mm, fix, A6.norm_log2(np.ones((40, 50)))) + A6.info_gain(np.ones((40, 50)), fix, A6.norm_log2(mm))) < 1e-12)
    check("IG nan without fixations", math.isnan(A6.info_gain(m, np.zeros((40, 50), bool), A6.norm_log2(m))))
    z = np.zeros((40, 50)); z[0, 0] = 1.0
    check("floor keeps the log finite for a map with zeros", np.isfinite(A6.norm_log2(z)).all())

    # gate
    cc = 0.7 + 0.05 * rng.randn(200); nss = 2.5 + 0.3 * rng.randn(200)
    pub = dict(CC=float(cc.mean()), NSS=float(nss.mean()), cc_per_image=[round(float(c), 5) for c in cc])
    check("gate passes on a faithful reproduction", A6.gate_compare(cc, nss, pub)["passed"])
    check("gate fails on a 2e-4 shift of the mean CC", not A6.gate_compare(cc + 2e-4, nss, pub)["passed"])
    c2 = cc.copy(); c2[7] += 3e-4
    check("gate fails on one per-image CC off by 3e-4", not A6.gate_compare(c2, nss, dict(pub, CC=float(c2.mean())))["passed"])
    check("gate fails on a 2e-4 shift of the mean NSS", not A6.gate_compare(cc, nss + 2e-4, pub)["passed"])
    check("gate fails on a different image count", not A6.gate_compare(cc[:199], nss[:199], pub)["passed"])

    # interval with images as clusters
    v = 0.1 + rng.randn(200)
    s = A6.summarise(v)
    check("summarise: mean, G = 200 images, 9,999 random draws", abs(s["mean"] - v.mean()) < 1e-15 and s["G"] == 200
          and s["draws"] == 9999 and not s["enumerated"])
    se_iid = v.std(ddof=1) / math.sqrt(200)
    check("summarise: CR1 se with singleton clusters equals the iid se", abs(s["se"] - se_iid) / se_iid < 1e-12)

    # torch part: untrained heads, random features, CPU
    sys.path.insert(0, os.path.join(X.RE, "gpu"))
    import recommon as RC
    import torch
    check("recommon on the CPU with the stub", RC.DEVICE == "cpu" and RC.FC.DEVICE == "cpu")
    torch.manual_seed(0)
    F = (0.5 * np.random.RandomState(1).randn(3, 8, 6, 6)).astype(np.float16)
    for arch in ("R4", "R1"):
        h = RC.make_head(arch, 8).eval()
        on = A6.head_maps(h, F, 24, True, torch, bs=2); off = A6.head_maps(h, F, 24, False, torch, bs=2)
        check(f"{arch}: zero centre bias, on equals off", np.allclose(on, off, atol=0, rtol=0))
        with torch.no_grad():
            h.center_bias.copy_(torch.linspace(-2, 2, 256).reshape(1, 1, 16, 16))
        on = A6.head_maps(h, F, 24, True, torch, bs=2); off = A6.head_maps(h, F, 24, False, torch, bs=2)
        check(f"{arch}: non-zero centre bias changes the map only when on", not np.allclose(on, off) and on.shape == (3, 24, 24))
        check(f"{arch}: use_cb left False after the off pass", h.use_cb is False)
    check("metal_guard: this process holds no Metal client", X.metal_guard() is True)

    n_fail = RES.count(False)
    out(f"# {len(RES)} checks, {n_fail} failed, {time.time() - t0:.1f}s")
    open(a.out, "w").write("\n".join(LINES) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
