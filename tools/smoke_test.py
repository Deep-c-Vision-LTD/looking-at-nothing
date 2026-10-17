"""
Smoke test of the repository on the CPU (a few minutes; no model weights or benchmark data needed).

  1. stimuli   re-render a sample of stim_v3, stim_alt and fill-variant displays with the generators in stimuli/ and
               compare the PNG bytes and the AOI arrays with the files in stimuli/
  2. scoring   score 2 stim_v3 images with the edge-energy null and the spectral residual model through
               scoring/score_v3.py (SCORE_SHARD=0/370, written to a temporary results tree) and compare every per-image
               measure with results/v2/scores/raw; recompute the 15 local nulls on the same 2 images with the code of
               scoring/local_nulls.py and compare with results/v2/verify/local_nulls_raw.json
  3. numbers   run paper_numbers/make_numbers.py into a temporary directory and compare every \\newcommand with
               paper_numbers/output/numbers.tex (comments, which name source files, are ignored)
Usage: python tools/smoke_test.py [stimuli] [scoring] [numbers]   (default: all three); exits non-zero on a mismatch.
"""
import json, os, subprocess, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import paths as LP  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def stimuli():
    sys.path.insert(0, os.path.join(REPO, "stimuli"))
    import stimuli_v3 as T
    import stimuli_alt as A
    picks = {"kanizsa": [0, 23, 59], "ehrenstein": [0, 39]}
    seen = {"kanizsa": -1, "ehrenstein": -1}
    for ill, p in T.configs():
        seen[ill] += 1
        if seen[ill] not in picks[ill]:
            continue
        conds = T.K_CONDS if ill == "kanizsa" else T.E_CONDS
        aoi = T.aoi_of(ill, p)
        for cond in conds:
            name = T.name_of(ill, p, cond)
            img = T.down(T.render(ill, p, cond)).convert("RGB")
            ref = os.path.join(LP.STIM_V3, "img", name + ".png")
            # PIL writes the file the same way as stimuli_v3.make(): img.convert("RGB").save(path)
            tmp = os.path.join(TMP, "s.png"); img.save(tmp)
            check(open(tmp, "rb").read() == open(ref, "rb").read(), f"stim_v3 {name}.png bytes")
            a = np.load(os.path.join(LP.STIM_V3, "img", name + "_aoi.npz"))
            check(sorted(a.files) == sorted(aoi) and all(np.array_equal(a[k], aoi[k]) for k in aoi), f"stim_v3 {name}_aoi.npz arrays")
    for k, p in enumerate(A.configs()):
        if k not in (0, 31, 59):
            continue
        cfg = A.config_name(p)
        for th in A.THETAS:
            for ph in ("a", "b"):
                img = T.down(A.kanizsa_hi_rots(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], A.alt_rots(p["shape"], th, ph), p["fg"], p["bg"]))
                tmp = os.path.join(TMP, "a.png"); img.convert("RGB").save(tmp)
                ref = os.path.join(LP.STIM_ALT, "img", f"{cfg}_alt{th}{ph}.png")
                check(open(tmp, "rb").read() == open(ref, "rb").read(), f"stim_alt {cfg}_alt{th}{ph}.png bytes")
        a = np.load(os.path.join(LP.STIM_ALT, "aoi", cfg + "_aoi.npz")); b = A.aoi_with_core(p)
        check(sorted(a.files) == sorted(b) and all(np.array_equal(a[x], b[x]) for x in b), f"stim_alt {cfg}_aoi.npz arrays")
    # fill variants (scoring/fill_contrast_iie.py renders them with stimuli_v3 and a changed GREY)
    for k, (ill, p) in enumerate(T.configs()):
        if ill != "kanizsa" or k not in (0, 37):
            continue
        for c in (0.25, 0.75):
            T.GREY = int(round(p["bg"] + c * (p["fg"] - p["bg"])))
            img = T.down(T.render("kanizsa", p, "real_filled")).convert("RGB")
            name = T.k_name(p["shape"], p["sr"], p["pol"], p["pos"], f"real_filled_c{int(c * 100)}")
            tmp = os.path.join(TMP, "f.png"); img.save(tmp)
            check(open(tmp, "rb").read() == open(os.path.join(LP.FILL_VARIANTS, "img", name + ".png"), "rb").read(), f"fill variant {name}.png bytes")
        T.GREY = 128


def scoring():
    res = os.path.join(TMP, "results")
    env = dict(os.environ, LAN_RESULTS=res, LAN_WORK=os.path.join(TMP, "work"), SCORE_SHARD="0/370")
    subprocess.run([sys.executable, os.path.join(REPO, "scoring", "score_v3.py"), "edge", "sr"], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    for tag in ("Edge-energy_null", "Spectral_residual"):
        new = json.load(open(os.path.join(res, "v2", "scores", "raw", f"{tag}.shard0of370.json")))
        old = LP.load_json(LP.V2 / "scores" / "raw" / f"{tag}.json")
        check(len(new) == 2, f"{tag}: {len(new)} images scored")
        for nm, rec in new.items():
            d = max(abs(rec[k] - old[nm][k]) for k in rec)
            check(d == 0.0, f"{tag} {nm}: max |difference| over {len(rec)} measures = {d:.1e}")
    # local nulls: maps() and the measures of scoring/local_nulls.py
    import cv2
    src = open(os.path.join(REPO, "scoring", "local_nulls.py")).read()
    ns = {"np": np, "cv2": cv2}
    exec(src[src.index("SIG = "):src.index("raw = {}")], ns)
    items = json.load(open(LP.STIM_V3 / "items.json"))
    ref = LP.load_json(LP.V2 / "verify" / "local_nulls_raw.json")
    for it in (items[0], items[370]):
        g = cv2.imread(str(LP.STIM_V3 / "img" / (it["name"] + ".png")), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        bg = 255.0 if it["pol"] == "dark" else 0.0
        a = np.load(LP.STIM_V3 / "img" / it["aoi"]); ins, ind, iea = a["inside"], a["ind"], a["inside_eroded_all"]
        d = 0.0
        for m, s in ns["maps"](g, bg).items():
            I, D, E = s[ins].sum(), s[ind].sum(), s[iea].sum()
            d = max(d, abs(float(I / (I + D)) - ref[m][it["name"]]["r_inside"]), abs(float(E / (E + D)) - ref[m][it["name"]]["r_eroded_all"]))
        check(d == 0.0, f"15 local nulls {it['name']}: max |difference| = {d:.1e}")


def numbers():
    out = os.path.join(TMP, "numbers")
    env = dict(os.environ, LAN_PAPER_OUT=out)
    subprocess.run([sys.executable, os.path.join(REPO, "paper_numbers", "make_numbers.py")], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    ref = os.path.join(REPO, "paper_numbers", "output")
    for f in ("numbers.tex", "tab_readout.tex", "tab_survey.tex", "tab_sweep.tex"):
        a = [l for l in open(os.path.join(ref, f)) if not l.startswith("%")]
        b = [l for l in open(os.path.join(out, f)) if not l.startswith("%")]
        n = len([l for l in a if l.startswith("\\newcommand")])
        check(a == b, f"{f}: {len(a)} non-comment lines" + (f" ({n} macros)" if n else "") + " identical")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["stimuli", "scoring", "numbers"]
    with tempfile.TemporaryDirectory() as TMP:
        for s in steps:
            print(f"== {s}", flush=True)
            globals()[s]()
    print(f"{len(FAIL)} failures")
    sys.exit(1 if FAIL else 0)
