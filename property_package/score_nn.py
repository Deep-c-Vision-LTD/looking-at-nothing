"""
Score sweep readouts on v2/stim_v3 (740 images) and v3/stim_alt (360 images) with the pipeline of
v2/scores/score_v3.py and v3/scores/score_alt.py (model code imported, measures imported or copied verbatim).

Per image, size s in (336, 448) and readout: sweep.prep -> sweep.feats -> sweep.Readout (use_cb False) -> sigmoid ->
cv2.resize to 1024 x 768 (INTER_LINEAR). CPU, 3 threads (ab_objective.DEV forced to "cpu" before sweep is imported).
Outputs (out_dir, default nn/results):
  raw_v2style/<tag>.json  v2 measures on the float32 map, all 740 stim_v3 images   (= v2/scores/raw/<tag>.json format)
  raw_v3maps/<tag>.json   common_alt.measures on common_alt.q16(map), 480 stim_v3 Kanizsa images in the 8 conditions
                          of common_alt.V3_CONDS                                    (= v3/scores/raw_v3maps/<tag>.json)
  raw_alt/<tag>.json      common_alt.measures on q16(map) + r_inside_f32, 360 stim_alt images (= v3/scores/raw/<tag>.json)
Resumable (images already scored for every tag are skipped). NN_SAVE_MAPS=1 also saves native s x s float16 maps.
Usage: python score_nn.py <dinov2|clip|siglip> <L[,L...]> [out_dir]
"""
import os, sys, time, json
sys.dont_write_bytecode = True
_T = os.environ.get("NN_THREADS", "3")
for k, v in (("OMP_NUM_THREADS", _T), ("MKL_NUM_THREADS", _T), ("VECLIB_MAXIMUM_THREADS", _T),
             ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[k] = v
import numpy as np, cv2, torch
torch.set_num_threads(int(_T))
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_nn as C                                   # noqa: E402
sys.path.insert(0, C.V3SC); sys.path.insert(0, C.ROOT)
import common_alt as CA                                 # noqa: E402
import ab_objective as AB                               # noqa: E402
AB.DEV = "cpu"
import gate_kanizsa as gk                               # noqa: E402

BB = sys.argv[1]
LS = tuple(int(x) for x in sys.argv[2].split(","))
OUT = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else C.RES
assert OUT.startswith(C.NN), OUT
SAVE_MAPS = os.environ.get("NN_SAVE_MAPS") == "1"
D_V2, D_V3, D_ALT = (os.path.join(OUT, d) for d in ("raw_v2style", "raw_v3maps", "raw_alt"))
for d in (D_V2, D_V3, D_ALT, C.LOGS):
    os.makedirs(d, exist_ok=True)
LOG = os.path.join(C.LOGS, f"score_{BB}_{'-'.join(map(str, LS))}.log")
ITEMS_V3 = json.load(open(os.path.join(C.STIM_V3, "items.json")))
ITEMS_ALT = CA.ALT_ITEMS


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


def measures_v2(sal, a):
    """Verbatim from v2/scores/score_v3.py measures()."""
    s = np.clip(sal.astype(np.float64), 0, None); tot = s.sum()
    I, E, EA, D = (s[a[k]].sum() for k in ("inside", "inside_eroded", "inside_eroded_all", "ind"))
    rr = lambda x: float(x / (x + D)) if x + D > 0 else float("nan")
    out = dict(r_inside=gk.rel_inside(sal, a["inside"], a["ind"]), r_eroded=rr(E), r_eroded_all=rr(EA),
               f_inside=float(I / tot), f_eroded=float(E / tot), f_eroded_all=float(EA / tot), f_ind=float(D / tot))
    assert abs(out["r_inside"] - rr(I)) < 1e-12
    return out


_aoi2 = {}
def aoi_v2(it):
    c = it["config"]
    if c not in _aoi2:
        a = np.load(os.path.join(C.STIM_V3, "img", it["aoi"]))
        _aoi2.clear()
        _aoi2[c] = {k: a[k] for k in ("inside", "ind", "inside_eroded", "inside_eroded_all")}
    return _aoi2[c]


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def main():
    argv = sys.argv; sys.argv = ["sweep.py", BB]
    import sweep                                        # module-level BB = sys.argv[1]; DEV = AB.DEV = "cpu"
    sys.argv = argv
    assert sweep.DEV == "cpu" and sweep.BB == BB
    bb = sweep.load_backbone(); cin = bb.config.hidden_size
    assert next(bb.parameters()).device.type == "cpu"
    R = {}
    for L in LS:
        m = sweep.Readout(cin)
        m.load_state_dict(torch.load(os.path.join(sweep.WORK, f"readout_L{L}.pt"), map_location="cpu")); m.eval(); m.use_cb = False
        R[L] = m
    tags = [C.tag(BB, L, s) for s in C.SIZES for L in R]
    log(f"start {BB} layers {LS} -> {OUT}; torch {torch.__version__}, threads {torch.get_num_threads()}")

    @torch.no_grad()
    def fn(rgb):
        out = {}
        for s in C.SIZES:
            fs = sweep.feats(bb, sweep.prep(rgb, s)[None])
            for L, m in R.items():
                m.use_cb = False
                nat = torch.sigmoid(m(fs[L], (s, s)))[0, 0].cpu().numpy()
                out[C.tag(BB, L, s)] = (cv2.resize(nat, (rgb.shape[1], rgb.shape[0])), nat)
        return out

    def load(d, t):
        p = os.path.join(d, t + ".json")
        return json.load(open(p)) if os.path.exists(p) else {}

    r2 = {t: load(D_V2, t) for t in tags}; r3 = {t: load(D_V3, t) for t in tags}; ra = {t: load(D_ALT, t) for t in tags}

    def dump():
        for t in tags:
            json.dump(r2[t], open(os.path.join(D_V2, t + ".json"), "w"))
            json.dump(r3[t], open(os.path.join(D_V3, t + ".json"), "w"))
            json.dump(ra[t], open(os.path.join(D_ALT, t + ".json"), "w"))

    def save_map(t, name, nat):
        if not SAVE_MAPS:
            return
        d = os.path.join(OUT, "maps", t); os.makedirs(d, exist_ok=True)
        mx = float(np.abs(nat).max())
        np.savez_compressed(os.path.join(d, name + ".npz"), sal=(nat / (mx if mx > 0 else 1.0)).astype(np.float16), scale=np.float64(mx))

    jobs = [("v3", it) for it in ITEMS_V3] + [("alt", it) for it in ITEMS_ALT]
    todo = [(k, it) for k, it in jobs if not all(it["name"] in (r2[t] if k == "v3" else ra[t]) for t in tags)]
    log(f"{len(todo)} of {len(jobs)} images to score, tags {tags}")
    t0 = time.time()
    for i, (kind, it) in enumerate(todo):
        nm = it["name"]
        if kind == "v3":
            maps = fn(read_rgb(os.path.join(C.STIM_V3, "img", nm + ".png")))
            a2 = aoi_v2(it)
            kz = it.get("illusion") == "kanizsa" and it["cond"] in CA.V3_CONDS
            a3 = CA.aoi(it["config"]) if kz else None
            for t, (full, nat) in maps.items():
                assert full.shape == (768, 1024) and np.isfinite(full).all(), (t, nm, full.shape)
                r2[t][nm] = measures_v2(full, a2)
                if kz:
                    r3[t][nm] = CA.measures(CA.q16(full), a3)
                save_map(t, nm, nat)
        else:
            maps = fn(read_rgb(os.path.join(C.STIM_ALT, "img", nm + ".png")))
            a = CA.aoi(it["config"])
            for t, (full, nat) in maps.items():
                assert full.shape == (768, 1024) and np.isfinite(full).all(), (t, nm, full.shape)
                rec = CA.measures(CA.q16(full), a)
                rec["r_inside_f32"] = gk.rel_inside(full, a["inside"], a["ind"])
                ra[t][nm] = rec
                save_map(t, nm, nat)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            dump()
            el = time.time() - t0
            log(f"  {i + 1}/{len(todo)} {el:.0f}s ({el / (i + 1):.2f} s/img)")
    dump()
    for t in tags:
        assert len(r2[t]) == 740 and len(r3[t]) == 480 and len(ra[t]) == 360, (t, len(r2[t]), len(r3[t]), len(ra[t]))
    log(f"done {BB} layers {LS}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
