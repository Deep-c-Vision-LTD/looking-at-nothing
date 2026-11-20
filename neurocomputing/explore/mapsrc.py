"""
neuro/reanalysis/explore/mapsrc.py: per-display maps of every model group for the map-based exploratory analyses of
REANALYSIS_RULE.md Section 6 (A4.4 log mass, A4.6 calibration, A12 Kavsek-style AOIs, A13 region variant) on stim_v3
and stim_alt. Written 18 November 2026 after the rule was public; not yet run on real data.

Groups and map sources (all read only; nothing is written outside neuro/):
  deepgaze   DeepGaze IIE and MSDB: saved maps (stim_v3: v2/scores/maps/<tag>/<name>.npz; stim_alt: v3/scores/maps/...),
             sal = density / max in float16 (the q16 form; the density up to a constant factor).
  msinet     MSI-Net: v2/scores/maps_msinet_png/<name>.png (8-bit min-max map; stim_v3 only), read as score_v3 does.
  nulls      the 15 local nulls: maps() copied verbatim from scoring/local_nulls_rigid.py (= v2/verify/local_nulls.py),
             float32, on the grey display (cv2), never quantised (as published).
  sweep      the 18 layer-sweep readouts at 336 and 448 px: sweep.py inference (trained readout_L<L>.pt, centre bias
             off), re-run on the CPU with the ab_objective stub (no Metal query), as scoring/score_rigid.py does.
  swap       the 60 readout-swap heads at 336 and 448 px: published checkpoints (v3/gpu/full/ckpt) through
             gpu/recommon.make_head; stim_v3 features from the float16 feature cache v2/readout/cache/stim (heads only,
             reproduces the published scores to 6.3e-6, data_inventory Section 6); stim_alt features by a backbone pass
             (gpu/recommon extractor and feats, features cast to float16 and back as eval_full does).
  registers  the 6 registers R4 heads (nn/registers/full/ckpt), backbone pass (dinov2reg path of gpu/recommon).
Published-form display map ("pub"), exactly the map that the published per-image measures were computed from:
  deepgaze sal; msinet the PNG; nulls the float32 map; sweep q16(cv2.resize(sigmoid(logit))); swap and registers
  q16(cv2.resize(float16(sigmoid(logit)))) (eval_full.score_map). Readouts also hand the logit (s x s, float32) and the
  float32 sigmoid to the analysis (A4.6 re-infers logits; they are never inverted from a float16 map).
Pipeline check: for every display with a published per-image r_inside, r_inside of "pub" is compared with it
(excommon.Check tolerances: saved maps 1e-6, nulls 1e-9, re-inferred readouts 1e-4, as rule Section 3.2); the analysis
scripts withhold the values of a unit that fails.
CPU only: V3_DEVICE=cpu, the stub of _stub/ab_objective.py first on sys.path, torch threads from OMP_NUM_THREADS (4),
and excommon.metal_guard() after every model load (the process aborts if it holds a Metal client).
"""
import json, os, sys, time
import numpy as np
import excommon as X

GROUPS = ("deepgaze", "msinet", "nulls", "sweep", "swap", "registers")
CACHE = os.path.join(X.ROOT, "v2", "readout", "cache", "stim")
SWAP_CKPT = os.path.join(X.ROOT, "v3", "gpu", "full", "ckpt")
REG_CKPT = os.path.join(X.ROOT, "nn", "registers", "full", "ckpt")
CHUNK = 64          # displays per feature chunk (cache path)
HB = 16             # head batch, as eval_full


# ====================================================================== displays
def displays(stimset, conds, configs=None):
    """[{stimset, name, cfg, cond, path, pol, inside (flat idx), ind (flat idx)}] in Design configuration order."""
    D = X.design()
    cfgs = configs or D.configs
    out = []
    for c in cfgs:
        a = X.v3_aoi(c) if stimset == "stim_v3" else X.alt_aoi(c)
        ins, ind = X.idx(a["inside"]), X.idx(a["ind"])
        for cond in conds:
            nm = f"{c}_{cond}"
            base = X.STIM_V3 if stimset == "stim_v3" else X.STIM_ALT
            out.append(dict(stimset=stimset, name=nm, cfg=c, cond=cond, path=os.path.join(base, "img", nm + ".png"),
                            pol=X.kz_params()[c]["pol"], inside=ins, ind=ind))
    return out


def units_of(group):
    return X.group_units(group)


# ====================================================================== published per-image r_inside (checks)
_PUB = {}


def published_r(group, unit, seed, size, stimset):
    key = (group, unit, seed, size, stimset)
    if key in _PUB:
        return _PUB[key]
    d = None
    if group == "deepgaze":
        tag = unit.replace(" ", "_")
        p = os.path.join(X.ROOT, "v3", "scores", "raw_v3maps" if stimset == "stim_v3" else "raw", tag + ".json")
        j = X.load_json(p)
        d = None if j is None else {k: v["r_inside"] for k, v in j.items()}
    elif group == "msinet":
        j = X.load_json(os.path.join(X.ROOT, "v2", "scores", "raw", "MSI-Net.json")) if stimset == "stim_v3" else None
        d = None if j is None else {k: v["r_inside"] for k, v in j.items()}
    elif group == "nulls":
        if "nulls" not in _PUB:
            _PUB["nulls"] = X.load_json(os.path.join(X.ROOT, "v3", "scores", "raw_nulls.json"))
        j = _PUB["nulls"]
        d = None if j is None else {k: v["r_inside"] for k, v in j[unit].items()}
    elif group == "sweep":
        bb, L = unit.split("_L")
        p = X.sweep_published_file(bb, int(L), size, stimset)
        j = X.load_json(p) if p else None
        d = None if j is None else {k: v["r_inside"] for k, v in j.items()}
    elif group in ("swap", "registers"):
        fs, ar = unit.split("|")
        sd = X.SWAP_SCORES if group == "swap" else X.REG_SCORES
        j = X.load_json(os.path.join(sd, f"{stimset}_{fs}_{ar}_s{seed}.json"))
        d = None if j is None else dict(j[str(size)]["r_inside"])
    _PUB[key] = d
    return d


# ====================================================================== null maps (verbatim)
SIG = [8, 16, 25, 40, 60]


def null_maps(g, bg):                                 # verbatim from scoring/local_nulls_rigid.py (= v2/verify/local_nulls.py)
    import cv2
    ink = np.abs(g - bg) / 255.0
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    har = np.clip(cv2.cornerHarris(g / 255.0, 5, 3, 0.04), 0, None)
    out = {}
    for s in SIG:
        out[f"ink_s{s}"] = cv2.GaussianBlur(ink, (0, 0), s)
        out[f"edge_s{s}"] = cv2.GaussianBlur(edge, (0, 0), s)
        out[f"corner_s{s}"] = cv2.GaussianBlur(har, (0, 0), s)
    return out


# ====================================================================== the CPU torch stack (lazy)
_T = None


def torch_env():
    """Import the CPU stack once: the ab_objective stub first (no Metal backend query), the repository gate_kanizsa (as
    scoring/score_rigid.py), then gpu/recommon.py with V3_DEVICE=cpu, torch with OMP_NUM_THREADS threads."""
    global _T
    if _T is not None:
        return _T
    X.kz_params()                                     # stimcode copies of the stimulus generators cached first
    os.environ["V3_DEVICE"] = "cpu"
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[k] = "1"
    sys.dont_write_bytecode = True
    stub = os.path.join(X.RE, "_stub")
    sys.path.insert(0, stub)
    import ab_objective as AB                         # noqa: E402  (the stub)
    assert AB.__file__.startswith(stub) and AB.DEV == "cpu", AB.__file__
    lan_repo = os.environ.get("LAN_REPO") or os.path.join(X.ROOT, "looking-at-nothing")
    saved = list(sys.path)
    shim = sys.modules.pop("paths", None)             # stimcode's `paths` shim (render_rigid) must not stand in for
    sys.path.insert(0, os.path.join(lan_repo, "models"))   # the repository paths.py that gate_kanizsa imports
    import gate_kanizsa as gk                         # noqa: E402,F401  (repository copy; sweep.py finds it cached)
    sys.path[:] = saved
    if shim is not None:
        sys.modules["paths"] = shim
    gpu = os.path.join(X.RE, "gpu")
    if gpu not in sys.path:
        sys.path.append(gpu)
    import recommon as R                              # noqa: E402  (fixed; V3_DEVICE=cpu, stub asserted inside)
    assert R.DEVICE == "cpu", R.DEVICE
    import torch, cv2                                 # noqa: E402
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cv2.setNumThreads(1)
    _T = dict(R=R, torch=torch, cv2=cv2, AB=AB)
    X.metal_guard()
    return _T


def sweep_module(bb):
    """sweep.py for backbone bb (module-level BB = sys.argv[1]); reloaded for each backbone, since a cached module keeps
    the first backbone (a plain second `import sweep` would return it unchanged)."""
    T = torch_env()
    import importlib
    argv = sys.argv
    sys.argv = ["sweep.py", bb]
    try:
        mod = importlib.reload(sys.modules["sweep"]) if "sweep" in sys.modules else importlib.import_module("sweep")
    finally:
        sys.argv = argv
    assert mod.DEV == "cpu" and mod.BB == bb and mod.AB is T["AB"], (mod.DEV, mod.BB)
    return mod


def load_head(ckpt, arch, fset):
    T = torch_env()
    R, torch = T["R"], T["torch"]
    m = R.make_head(arch, R.FSETS[fset]["cin"])
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    m.use_cb = False
    return m


def read_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# ====================================================================== the engine
def _pub_readout(group, sig):
    import cv2
    if group == "sweep":
        full = cv2.resize(np.asarray(sig, np.float32), (X.W, X.H))
    else:
        full = cv2.resize(np.asarray(sig, np.float16).astype(np.float32), (X.W, X.H))
    return X.clip64(X.q16(full))


class Engine:
    """run(groups, disps, fn): for every unit of the groups, every (seed, size) and every display, fn(ctx, pub, raw)
    returns a dict of floats that the Collector stores; the Check compares r_inside of pub with the published value."""

    def __init__(self, units=None, log_every=500):
        import cv2
        cv2.setNumThreads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))   # cv2 has its own pool: cap it too
        self.C, self.K = X.Collector(), X.Check()
        self.units = units or {}
        self.n, self.log_every, self.t0 = 0, log_every, time.time()

    def _units(self, group):
        return self.units.get(group) or units_of(group)

    def emit(self, group, unit, seed, size, disp, pub, raw, fn):
        pr = published_r(group, unit, seed, size, disp["stimset"])
        self.K.add(group, unit, X.r_from(pub, disp["inside"], disp["ind"]),
                   None if pr is None else pr.get(disp["name"]))
        ctx = dict(group=group, unit=unit, seed=seed, size=size, disp=disp)
        self.C.add(unit, seed, size, disp["name"], fn(ctx, pub, raw))
        self.n += 1
        if self.n % self.log_every == 0:
            X.log(f"  {self.n} unit-displays ({time.time() - self.t0:.0f}s); last {group} {unit} {disp['name']}")

    def run(self, groups, disps, fn):
        for g in groups:
            assert g in GROUPS, g
            X.log(f"group {g}: {len(self._units(g))} units, {len(disps)} displays")
            getattr(self, "_" + g)(disps, fn)
        return self.C, self.K

    # ---------------------------------------------------------------- saved maps and nulls
    def _deepgaze(self, disps, fn):
        for u in self._units("deepgaze"):
            tag = u.replace(" ", "_")
            for d in disps:
                base = os.path.join(X.ROOT, "v2" if d["stimset"] == "stim_v3" else "v3", "scores", "maps", tag)
                sal = np.load(os.path.join(base, d["name"] + ".npz"))["sal"]
                pub = X.clip64(sal)
                self.emit("deepgaze", u, None, None, d, pub, dict(sal=pub), fn)

    def _msinet(self, disps, fn):
        import cv2
        for d in disps:
            if d["stimset"] != "stim_v3":
                continue                                  # MSI-Net maps exist for stim_v3 only
            png = os.path.join(X.ROOT, "v2", "scores", "maps_msinet_png", d["name"] + ".png")
            m = cv2.imread(png, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            pub = X.clip64(m)
            self.emit("msinet", "MSI-Net", None, None, d, pub, dict(png=pub), fn)

    def _nulls(self, disps, fn):
        import cv2
        want = self._units("nulls")
        for d in disps:
            g = cv2.imread(d["path"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            bg = 255.0 if d["pol"] == "dark" else 0.0
            for u, m in null_maps(g, bg).items():
                if u in want:
                    pub = X.clip64(m)
                    self.emit("nulls", u, None, None, d, pub, dict(map=pub), fn)

    # ---------------------------------------------------------------- readouts
    def _readout_emit(self, group, unit, seed, s, d, z, sig, fn):
        pub = _pub_readout(group, sig)
        self.emit(group, unit, seed, s, d, pub, dict(logit=z, sig=sig, size=s), fn)

    def _sweep(self, disps, fn):
        T = torch_env()
        torch = T["torch"]
        want = set(self._units("sweep"))
        for bb, Ls in X.SWEEP_LAYERS.items():
            Ls = [L for L in Ls if f"{bb}_L{L}" in want]
            if not Ls:
                continue
            sw = sweep_module(bb)
            bbm = sw.load_backbone()
            assert next(bbm.parameters()).device.type == "cpu"
            cin = bbm.config.hidden_size
            heads = {}
            for L in Ls:
                m = sw.Readout(cin)
                m.load_state_dict(torch.load(os.path.join(sw.WORK, f"readout_L{L}.pt"), map_location="cpu"))
                m.eval()
                m.use_cb = False
                heads[L] = m
            X.metal_guard()
            X.log(f"  sweep {bb}: blocks {Ls}")
            with torch.no_grad():
                for b in range(0, len(disps), 4):
                    chunk = disps[b:b + 4]
                    rgbs = [read_rgb(d["path"]) for d in chunk]
                    for s in X.SIZES:
                        fs = sw.feats(bbm, torch.stack([sw.prep(r, s) for r in rgbs]))
                        for L, m in heads.items():
                            m.use_cb = False
                            z = m(fs[L], (s, s))[:, 0]
                            sig = torch.sigmoid(z)
                            for j, d in enumerate(chunk):
                                self._readout_emit("sweep", f"{bb}_L{L}", 0, s, d, z[j].numpy(), sig[j].numpy(), fn)
            del bbm, heads

    def _heads_for(self, group):
        """{fset: [(arch, seed, unit, module)]} for the requested units."""
        out = {}
        for u in self._units(group):
            fs, ar = u.split("|")
            for seed in (0, 1, 2):
                ck = os.path.join(SWAP_CKPT if group == "swap" else REG_CKPT, f"{fs}_{ar}_s{seed}.pt")
                out.setdefault(fs, []).append((ar, seed, u, load_head(ck, ar, fs)))
        return out

    def _swap(self, disps, fn):
        T = torch_env()
        torch = T["torch"]
        heads = self._heads_for("swap")
        X.metal_guard()
        v3 = [d for d in disps if d["stimset"] == "stim_v3"]
        other = [d for d in disps if d["stimset"] != "stim_v3"]
        names_v3 = json.load(open(os.path.join(CACHE, "stim_v3_names.json")))
        row = {n: i for i, n in enumerate(names_v3)}
        with torch.no_grad():
            for fs, hs in heads.items():                  # stim_v3: heads only, from the float16 feature cache
                for s in X.SIZES:
                    F = np.load(os.path.join(CACHE, f"stim_v3_{fs}_s{s}.npy"), mmap_mode="r")
                    assert F.shape[0] == len(names_v3), (F.shape, len(names_v3))
                    for b in range(0, len(v3), CHUNK):
                        chunk = v3[b:b + CHUNK]
                        Xc = torch.from_numpy(np.ascontiguousarray(F[[row[d["name"]] for d in chunk]]))
                        for ar, seed, u, m in hs:
                            for k in range(0, len(chunk), HB):
                                z = m(Xc[k:k + HB].float(), (s, s))[:, 0]
                                sig = torch.sigmoid(z)
                                for j, d in enumerate(chunk[k:k + HB]):
                                    self._readout_emit("swap", u, seed, s, d, z[j].numpy(), sig[j].numpy(), fn)
                    del F
            if other:                                     # stim_alt: backbone pass (no feature cache exists)
                self._backbone_heads("swap", {"dinov2": ("dinov2_L12", "dinov2_L6"), "siglip": ("siglip_L24",),
                                              "iie": ("iie_shapenetc",)}, heads, other, fn)

    def _registers(self, disps, fn):
        heads = self._heads_for("registers")
        self._backbone_heads("registers", {"dinov2reg": X.REG_FSETS}, heads, disps, fn)

    def _backbone_heads(self, group, bbs, heads, disps, fn):
        T = torch_env()
        R, torch = T["R"], T["torch"]
        with torch.no_grad():
            for bb, fsets in bbs.items():
                hs = [(fs, h) for fs in fsets for h in heads.get(fs, [])]
                if not hs:
                    continue
                ex = R.make_extractor(bb)
                X.metal_guard()
                X.log(f"  {group} backbone {bb}: {len(hs)} heads")
                for b in range(0, len(disps), HB):
                    chunk = disps[b:b + HB]
                    for s in X.SIZES:
                        f = R.feats(ex, R.prep_batch(ex, [d["path"] for d in chunk], s), s)
                        f = {k: v.half().cpu() for k, v in f.items()}      # as eval_full: float16 between backbone and head
                        for fs, (ar, seed, u, m) in hs:
                            z = m(f[fs].float(), (s, s))[:, 0]
                            sig = torch.sigmoid(z)
                            for j, d in enumerate(chunk):
                                self._readout_emit(group, u, seed, s, d, z[j].numpy(), sig[j].numpy(), fn)
                del ex
