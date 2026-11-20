"""
Unit tests of explore/tokens/a3_tokens_published.py (RUN_PLAN.md S5) on synthetic inputs only: the verdict rule (unrounded
and with the published 4-decimal rounding), the agreement rule, S20 from a ratio record, the configuration-to-design
mapping, and that the copies of explore/tokens/ redirect every output into neuro/ (source checks). No token, score or
result file is read; no torch. Writes only its output file (--out).
Usage: OMP_NUM_THREADS=2 $PY -B -I explore/tokens/test_a3_tokens.py --out explore/tokens/tests_a3_tokens_output.txt
Written 19 November 2026.
"""
import argparse, inspect, os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import a3_tokens_published as A3                                          # noqa: E402

X = A3.X
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
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "tests_a3_tokens_output.txt"))
    a = ap.parse_args()
    t0 = time.time()
    out(f"test_a3_tokens.py {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    check("no torch imported", "torch" not in sys.modules)
    src = inspect.getsource(A3.main)
    check("main: require_public before any read", src.index("require_public") < min(src.index("compare_files"), src.index("import analyze")))
    I = dict(ci=[0.30004, 0.5]); B = dict(ci=[0.1, 0.30001]); C = dict(ci=[0.0, 0.2])
    check("verdict: unrounded bounds", A3.verdict(I, [B, C]) is True)
    check("verdict: published rounding can change it (0.3000 > 0.3000 is False)", A3.verdict(I, [B, C], rounded=True) is False)
    check("verdict: fails when a null upper bound exceeds the interior lower bound", A3.verdict(dict(ci=[0.1, 0.4]), [B, C]) is False)
    check("verdict: undefined when a bound is missing", A3.verdict(dict(ci=[None, None]), [B]) is None)
    check("agrees: figure-specific with E above 0", A3.agrees(True, [0.01, 0.1]) is True)
    check("agrees: figure-specific with E including 0 disagrees", A3.agrees(True, [-0.01, 0.1]) is False)
    check("agrees: not figure-specific with E including 0 or below agrees",
          A3.agrees(False, [-0.01, 0.1]) is True and A3.agrees(False, [-0.2, -0.1]) is True)
    check("agrees: not figure-specific with E above 0 disagrees", A3.agrees(False, [0.02, 0.1]) is False)
    rat = dict(value=0.7, ci_bca=[0.6, 0.85], stable=True, reportable=True, flag=None, t_den=6.0)
    s = A3.s20(rat)
    check("S20 = 1 - U20 with the interval mapped", abs(s["value"] - 0.3) < 1e-15 and np.allclose(s["ci_bca"], [0.15, 0.4]))
    check("S20 keeps the flag", A3.s20(dict(rat, flag="unstable"))["flag"] == "unstable")
    check("ratio text: not reportable printed without a value",
          A3.rat_txt(dict(value=-180.3, ci_bca=[-2789.0, -2739.0], flag="not reportable (denominator interval includes 0)",
                          reportable=False)) == "not reportable (denominator interval includes 0)")
    check("ratio text: unstable ratio keeps value, interval and flag", "unstable" in A3.rat_txt(dict(
        value=0.5, ci_bca=[0.1, 0.9], flag="unstable", reportable=True)) and "0.5000" in A3.rat_txt(dict(
        value=0.5, ci_bca=[0.1, 0.9], flag="unstable", reportable=True)))
    src_sal = inspect.getsource(A3.saliency_rows)
    check("saliency rows: swap and sweep keys are prefixed (no collision of dinov2_L12, dinov2_L6, siglip_L24)",
          'f"swap:{fs}"' in src_sal and 'f"sweep:{bb}_L{L}"' in src_sal)
    # to_design on a synthetic configuration list in shuffled order
    meta = {f"{sh}_sr{int(sr * 100)}_{pol}_p{pos}": (sh, sr, pol, pos)
            for sh in ("square", "triangle") for sr in (0.3, 0.5, 0.7) for pol in ("dark", "light") for pos in range(5)}
    D = AR.Design(meta)
    names = list(meta)[::-1]
    CFGS = [dict(config=c) for c in names]
    vals = np.arange(len(names), dtype=float)
    arr = A3.to_design(vals, CFGS, D)
    check("to_design maps every configuration to its value", all(arr[k, p] == names.index(D.cells[k][p])
                                                                 for k in range(D.K) for p in range(D.P)))
    # copies redirect outputs into neuro/
    for f in ("analyze.py", "plots.py", "report.py", "encoding_nulls.py"):
        s_ = open(os.path.join(HERE, f)).read()
        check(f"copy {f}: outputs under A3_tokens/published", 'OUTDIR = ' in s_ and '"A3_tokens", "published"' in s_)
    s_ = open(os.path.join(HERE, "analyze.py")).read()
    check("copy analyze.py: no write into v2/encoding", 'os.path.join(ENC, "results.json")' not in s_ and
          'os.path.join(ENC, "per_config_indices.npz")' not in s_)
    check("copy analyze.py: require_public at the top when run", s_.index("require_public") < s_.index("from common_enc import"))
    s_ = open(os.path.join(HERE, "encoding_nulls.py")).read()
    check("copy encoding_nulls.py: no write into v2/verify and require_public first",
          '"verify", "encoding_nulls.json"' not in s_ and s_.index("require_public") < s_.index("from common_enc import"))
    for f in ("plots.py", "report.py"):
        s_ = open(os.path.join(HERE, f)).read()
        check(f"copy {f}: ENC is the redirected folder", "ENC = OUTDIR" in s_)
    n_fail = RES.count(False)
    out(f"# {len(RES)} checks, {n_fail} failed, {time.time() - t0:.1f}s")
    open(a.out, "w").write("\n".join(LINES) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
