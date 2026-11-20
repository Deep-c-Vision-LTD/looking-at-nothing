"""
neuro/reanalysis/explore/excommon.py: shared helpers for the exploratory scripts of REANALYSIS_RULE.md Section 6.

Written 18 November 2026, after the rule was fixed and made public (PUBLIC_PUSH.txt), before any of these scripts was
run on real data. Nothing here edits a fixed file; analyze_re.py and render_rigid.py are imported read only.

- No torch import (the readout engine is mapsrc.py and imports torch lazily, only inside the real runs).
- Statistics come from the fixed engine analyze_re.py (wild cluster bootstrap with Webb weights, BCa, Fieller, Design,
  the v2 two-stage indices); the only new statistic here is slope_pct(), a copy of v3/scores/analyze_alt.slope without
  its rounding (the published method for support-ratio slopes; rule Section 6, first paragraph).
- Every script calls require_public() first (rule Section 8.1), which is analyze_re.require_public().
- Outputs: one JSON per script in neuro/reanalysis/results/explore/ (write_json), labelled exploratory.
"""
import hashlib, json, math, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                                                    # neuro/reanalysis
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
RESULTS = os.path.join(RE, "results", "explore")
STIM_V3 = os.path.join(ROOT, "v2", "stim_v3")
STIM_ALT = os.path.join(ROOT, "v3", "stim_alt")
STIM_RIGID = os.path.join(RE, "stim_rigid")
SCORES_RIGID = os.path.join(RE, "scores_rigid")
W, H = 1024, 768
SIZES = (336, 448)
PHIS = (10, 20, 30, 45)

if RE not in sys.path:
    sys.path.insert(0, RE)
import analyze_re as AR                                                       # noqa: E402  (fixed engine, read only)

LABEL = "exploratory (REANALYSIS_RULE.md Section 6; unadjusted; never used for a headline claim)"


# ====================================================================== guard, provenance, output
def require_public():
    """Rule Section 8.1: refuse unless the rule and its code are public and unchanged (analyze_re.require_public)."""
    return AR.require_public(RE)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(o):
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None (reported as null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    return o


def write_json(name, result, script, inputs=(), notes=None, extra_code=()):
    """results/explore/<name>.json with provenance: the script and helper hashes, the rule hash and the public commit."""
    os.makedirs(RESULTS, exist_ok=True)
    pub = os.path.join(RE, "PUBLIC_PUSH.txt")
    code = [script, os.path.abspath(__file__)] + [os.path.abspath(p) for p in extra_code]
    rec = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), label=LABEL,
               code={os.path.relpath(p, RE): sha256_file(p) for p in code},
               rule_sha256=sha256_file(os.path.join(RE, "REANALYSIS_RULE.md")),
               public_commit=open(pub).read().split()[0] if os.path.exists(pub) else None,
               inputs=list(inputs), notes=notes, result=result)
    p = os.path.join(RESULTS, name)
    with open(p + ".tmp", "w") as f:
        json.dump(_clean(rec), f, indent=1)
    os.replace(p + ".tmp", p)
    print("wrote", p, flush=True)
    return p


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def metal_guard(pid=None):
    """CPU-only safety: abort this process if it holds a Metal client (an AGXDeviceUserClient created by its pid).
    Read-only ioreg query; called by mapsrc after every model load."""
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(["ioreg", "-l", "-w0", "-r", "-c", "AGXDeviceUserClient"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception as e:                                                    # ioreg missing: cannot check; say so
        log(f"metal_guard: ioreg not available ({type(e).__name__}); check the pid {pid} by hand")
        return None
    if f'"IOUserClientCreator" = "pid {pid},' in out:
        log(f"metal_guard: pid {pid} holds a Metal client; aborting (CPU-only run)")
        os._exit(3)
    return True


# ====================================================================== design and statistics (engine of analyze_re)
_DES = {}


def design(shape=None):
    """analyze_re.Design over the 60 Kanizsa configurations (12 clusters; 6 for one shape)."""
    if shape not in _DES:
        _DES[shape] = AR.Design(AR.configs_meta(), shape)
    return _DES[shape]


def lin(dKP, D):
    """Linear quantity (mean of per-configuration values): WCU interval and WCR p (rule Section 4.1)."""
    d = np.asarray(dKP, float).ravel()
    if not np.all(np.isfinite(d)):
        return dict(mean=None, ci=[None, None], p=None, note=f"not computable: {int(np.sum(~np.isfinite(d)))} "
                                                              "configuration values missing or not finite")
    return AR.wild_cluster(d, D.g)


def excl0(ci):
    return ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)


def ratio(numKP, denKP, D):
    """Ratio of means (rule Section 4.2 and 4.3): BCa (pairs cluster bootstrap, v2 indices), Fieller alongside, stability
    flag (|t| of the denominator, CR1 se, below 4 = unstable), reportable only if the denominator's WCU interval
    excludes 0."""
    num, den = np.asarray(numKP, float), np.asarray(denKP, float)
    if not (np.all(np.isfinite(num)) and np.all(np.isfinite(den))):
        return dict(value=None, note="not computable: missing configuration values")
    wd = AR.wild_cluster(den.ravel(), D.g)
    td = AR.t_den(den.ravel(), D.g)
    out = dict(value=float(num.mean() / den.mean()), t_den=td, stable=bool(td >= AR.T_STABLE),
               denominator=dict(mean=wd["mean"], ci=wd["ci"]), reportable=bool(excl0(wd["ci"])))
    b = AR.bca(lambda a, c: a / c, [num, den], D.idx)
    out.update(ci_bca=b["ci"], z0=b["z0"], acc=b["acc"], fieller=AR.fieller(num, den, D.g))
    if not out["reportable"]:
        out["flag"] = "not reportable (denominator interval includes 0)"
    elif not out["stable"]:
        out["flag"] = "unstable"
    return out


def ratio_diff(n1, d1, n2, d2, D):
    """Paired difference of two ratios of means (as analyze_re.p4): BCa with p by inversion at 0, both denominators
    screened (|t| >= 4 and WCU interval excluding 0), Fieller for both single ratios."""
    arrs = [np.asarray(a, float) for a in (n1, d1, n2, d2)]
    b = AR.bca(lambda a, c, e, f: a / c - e / f, arrs, D.idx, theta0=0.0)
    t1, t2 = AR.t_den(arrs[1].ravel(), D.g), AR.t_den(arrs[3].ravel(), D.g)
    w1, w2 = AR.wild_cluster(arrs[1].ravel(), D.g), AR.wild_cluster(arrs[3].ravel(), D.g)
    stable = bool(min(t1, t2) >= AR.T_STABLE)
    rep = bool(excl0(w1["ci"]) and excl0(w2["ci"]))
    return dict(diff=b["value"], ci_bca=b["ci"], p_bca=b["p"], z0=b["z0"], acc=b["acc"],
                ratio_1=float(arrs[0].mean() / arrs[1].mean()), ratio_2=float(arrs[2].mean() / arrs[3].mean()),
                t_den_1=t1, t_den_2=t2, stable=stable, denominators_exclude_zero=rep,
                fieller_1=AR.fieller(arrs[0], arrs[1], D.g), fieller_2=AR.fieller(arrs[2], arrs[3], D.g),
                flag=None if (stable and rep) else ("not reportable (a denominator interval includes 0)" if not rep
                                                   else "unstable"))


def slope_pct(vKP, D, alpha=0.05):
    """Support-ratio slope of per-configuration values (OLS on support ratio, per unit support ratio) with the two-stage
    percentile interval: a copy of v3/scores/analyze_alt.slope (published method) without its rounding."""
    v = np.asarray(vKP, float)
    ci, pi = D.idx
    E = v[ci[:, :, None], pi]
    X = np.broadcast_to(D.sr[ci][:, :, None], E.shape)
    xm, em = X.mean((1, 2), keepdims=True), E.mean((1, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = ((X - xm) * (E - em)).sum((1, 2)) / ((X - xm) ** 2).sum((1, 2))
    x = np.broadcast_to(D.sr[:, None], v.shape)
    pt = float(((x - x.mean()) * (v - v.mean())).sum() / ((x - x.mean()) ** 2).sum())
    lo, hi = np.nanpercentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(np.nanmean(b <= 0), np.nanmean(b >= 0)))
    by = {str(s): float(v[D.sr == s].mean()) for s in sorted(set(D.sr))}
    return dict(slope_per_unit_sr=pt, ci=[float(lo), float(hi)], p=float(p), mean_by_sr=by)


# ====================================================================== per-display bookkeeping
class Collector:
    """Per-display values of every unit (model, null, readout cell) at every (seed, size): add(), then per_display()
    averages over seeds and sizes (or over seeds within one size). Readouts: seeds averaged per configuration, then the
    336 and 448 px values averaged (rule Section 2.2); since both are plain means of the per-display value, the order
    does not matter. For a log mass the per-display value is the log of the share, so logs are averaged."""

    def __init__(self):
        self.v = {}                                               # unit -> key -> name -> {(seed, size): value}

    def add(self, unit, seed, size, name, vals):
        for k, x in vals.items():
            self.v.setdefault(unit, {}).setdefault(k, {}).setdefault(name, {})[(seed, size)] = float(x)

    def units(self):
        return list(self.v)

    def keys(self, unit):
        return list(self.v.get(unit, {}))

    def per_display(self, unit, key, size=None):
        """{name: mean over (seed, size)}; size restricts to one size. Every name must have the same set of
        (seed, size) cells (asserted), so an incomplete run cannot silently change the averaging."""
        d = self.v[unit][key]
        out, cells = {}, None
        for nm, cv in d.items():
            sel = {sc: x for sc, x in cv.items() if size is None or sc[1] == size}
            cs = frozenset(sel)
            if cells is None:
                cells = cs
            assert cs == cells, (unit, key, nm, sorted(cs, key=str), sorted(cells, key=str))
            out[nm] = float(np.mean(list(sel.values()))) if sel else float("nan")
        return out

    def sizes(self, unit, key):
        return sorted({sc[1] for cv in self.v[unit][key].values() for sc in cv if sc[1] is not None})


class Check:
    """Pipeline check bookkeeping (in the spirit of rule Section 3.2): the published-form map of every display that has a
    published per-image r_inside must reproduce it within the group's tolerance; a group that fails is reported as not
    computable (its values are withheld)."""
    TOL = dict(deepgaze=1e-6, msinet=1e-6, nulls=1e-9, sweep=1e-4, swap=1e-4, registers=1e-4)

    def __init__(self):
        self.worst, self.n, self.missing = {}, {}, {}

    def add(self, group, unit, value, published):
        if published is None:
            self.missing[group] = self.missing.get(group, 0) + 1
            return
        d = abs(float(value) - float(published))
        self.worst[(group, unit)] = max(self.worst.get((group, unit), 0.0), d)
        self.n[group] = self.n.get(group, 0) + 1

    def group_ok(self, group):
        ws = [w for (g, _), w in self.worst.items() if g == group]
        return bool(ws) and max(ws) <= self.TOL[group]

    def unit_ok(self, group, unit):
        return (group, unit) in self.worst and self.worst[(group, unit)] <= self.TOL[group]

    def report(self):
        groups = sorted({g for g, _ in self.worst} | set(self.missing))
        return {g: dict(tolerance=self.TOL[g], n_compared=self.n.get(g, 0), n_without_published_value=self.missing.get(g, 0),
                        max_abs_diff=max([w for (gg, _), w in self.worst.items() if gg == g], default=None),
                        passed=self.group_ok(g),
                        failed_units=sorted(u for (gg, u), w in self.worst.items() if gg == g and w > self.TOL[g]))
                for g in groups}


# ====================================================================== maps and regions
def q16(sal):
    """Verbatim from scoring/common_rigid.q16 (= v3/scores/common_alt.q16): map / max|map| in float16."""
    sal = np.asarray(sal, np.float32)
    mx = float(np.abs(sal).max())
    return (sal / (mx if mx > 0 else 1.0)).astype(np.float16)


def clip64(m):
    """As common_alt.measures: map cast to float32, then float64, clipped at 0."""
    return np.clip(np.asarray(m, np.float32).astype(np.float64), 0, None)


def mass(s, idx):
    """Sum of a clipped float64 map over flat pixel indices."""
    return float(s.ravel()[idx].sum())


def r_from(s, inside_idx, ind_idx):
    I, D = mass(s, inside_idx), mass(s, ind_idx)
    return I / (I + D) if I + D > 0 else float("nan")


def rank_equalise(m):
    """Rule Section 6, item 6 (a): each pixel replaced by its rank within the display divided by the number of pixels,
    ties averaged (scipy.stats.rankdata, method 'average')."""
    from scipy.stats import rankdata
    a = np.asarray(m, np.float64)
    return (rankdata(a.ravel(), method="average") / a.size).reshape(a.shape)


def idx(mask):
    return np.flatnonzero(np.asarray(mask, bool).ravel())


def grey_png(path):
    """Grey levels as render_rigid reads them (PIL, mode L)."""
    from PIL import Image
    return np.array(Image.open(path).convert("L"))


_RR = None


def render_rigid():
    """render_rigid.py (fixed; imported read only for notch_geom, ink_mask, polygon_exact, rigid_vertices, kz_configs)."""
    global _RR
    if _RR is None:
        import render_rigid as RR                                         # RE is on sys.path
        _RR = RR
    return _RR


_KZ = None


def kz_params():
    """{configuration name: stimulus parameters (shape, cx, cy, side, sr, pol, pos, fg, bg)} from the stim_v3 generator
    (render_rigid.kz_configs, the verbatim stimuli_v3 copy)."""
    global _KZ
    if _KZ is None:
        RR = render_rigid()
        _KZ = {RR.cfg_name(p): p for p in RR.kz_configs()}
        assert len(_KZ) == 60, len(_KZ)
    return _KZ


def v3_aoi(cfg):
    a = np.load(os.path.join(STIM_V3, "img", f"{cfg}_illusory_aoi.npz"))
    return {k: a[k].astype(bool) for k in a.files}


def alt_aoi(cfg):
    a = np.load(os.path.join(STIM_ALT, "aoi", f"{cfg}_aoi.npz"))
    return {k: a[k].astype(bool) for k in a.files}


ROT = {"illusory": 0, "control180": 180}


def v3_regions(cfg, cond, png_path=None, arr=None, aoi=None):
    """Regions of a stim_v3 display (phi = 0), rule Section 6 items 4, 10 and 11:
      inside   the inside AOI (identical across conditions);
      ind      the inducer-disc AOI (D);
      polygon  the whole contour area P (render_rigid.polygon_exact on the stim_v3 vertices, as the rule prescribes for
               phi = 0; the same function the stim_rigid AOIs use);
      notchgeom the analytic notch sectors of this condition (render_rigid.notch_geom, phi = 0, rotation 0 for the
               illusory and 180 for the control180 display; half-open angular test);
      ink      pixels at least 128 grey levels from the background in the rendered display (render_rigid.ink_mask);
      notch    notchgeom minus ink (the rule's notch sector).
    arr (grey display) and aoi may be passed directly (unit tests); otherwise they are read from v2/stim_v3."""
    RR = render_rigid()
    p = kz_params()[cfg]
    a = aoi if aoi is not None else v3_aoi(cfg)
    if arr is None:
        arr = grey_png(png_path or os.path.join(STIM_V3, "img", f"{cfg}_{cond}.png"))
    ink = RR.ink_mask(arr, p["bg"])
    ng = RR.notch_geom(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], 0, ROT[cond])
    V = RR.rigid_vertices(p["shape"], p["cx"], p["cy"], p["side"], 0)
    return dict(inside=a["inside"], ind=a["ind"], polygon=RR.polygon_exact(V), notchgeom=ng, ink=ink, notch=ng & ~ink)


def pd_masses(s, ix):
    """Masses of a clipped map in P (whole contour area), D (inducer discs), D minus P and P minus D."""
    return {k: mass(s, v) for k, v in ix.items()}


def pd_index_from(P, Dm):
    P, Dm = np.asarray(P, bool), np.asarray(Dm, bool)
    return dict(P=idx(P), D=idx(Dm), DnP=idx(Dm & ~P), PnD=idx(P & ~Dm))


def pd_index_v3(cfgs):
    """{cfg: flat indices of P, D, D minus P, P minus D} at phi = 0 (identical across the stim_v3 conditions):
    P = render_rigid.polygon_exact on the stim_v3 vertices, D = the inducer-disc AOI."""
    RR = render_rigid()
    out = {}
    for c in cfgs:
        p = kz_params()[c]
        V = RR.rigid_vertices(p["shape"], p["cx"], p["cy"], p["side"], 0)
        out[c] = pd_index_from(RR.polygon_exact(V), v3_aoi(c)["ind"])
    return out


def r_K(m, cond):
    """Kavsek-style r (rule Section 6, item 10): inner = P for illusory (and real) displays, P minus D for control
    displays; r_K = M(inner) / (M(inner) + M(D minus inner)). m: pd_masses."""
    if "control" in cond:
        inner, outer = m["PnD"], m["D"]                           # D minus (P minus D) = D
    else:
        inner, outer = m["P"], m["DnP"]
    return inner / (inner + outer) if inner + outer > 0 else float("nan")


def r_N(m):
    """Region variant (rule Section 6, item 11): r_N = M(P) / (M(P) + M(D minus P)), identical AOIs across conditions."""
    return m["P"] / (m["P"] + m["DnP"]) if m["P"] + m["DnP"] > 0 else float("nan")


def uniform_gain(nP, nDnP, nPnD, nD):
    """Gain in r_K that a uniform map would show from the AOI change alone (rule Section 6, item 10), from pixel counts:
    r_K,uniform(illusory) - r_K,uniform(control180) = |P| / (|P| + |D \\ P|) - |P \\ D| / (|P \\ D| + |D|)."""
    return nP / (nP + nDnP) - nPnD / (nPnD + nD)


def rigid_pd_counts(cfg, phi):
    """Pixel counts of P, D minus P, P minus D, D for a stim_rigid configuration and angle (its own rotated AOIs)."""
    a = np.load(os.path.join(STIM_RIGID, "aoi", f"{cfg}_rig{phi}_aoi.npz"))
    P, Dm = a["polygon"].astype(bool), a["ind"].astype(bool)
    return int(P.sum()), int((Dm & ~P).sum()), int((P & ~Dm).sum()), int(Dm.sum())


# ====================================================================== names, conditions, published files
def v3_name(cfg, cond):
    return f"{cfg}_{cond}"


def rig_name(cfg, phi, cond):
    return f"{cfg}_rig{phi}_{cond}"


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


SWAP_FSETS = AR.SWAP_FSETS
ARCHS = AR.ARCHS
NULLS = AR.NULLS
SWEEP_LAYERS = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
SWEEP_PRIMARY = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
REG_FSETS = ("dinov2reg_L12", "dinov2reg_L6")
SWAP_SCORES = os.path.join(ROOT, "v3", "gpu", "full", "scores")
REG_SCORES = os.path.join(ROOT, "nn", "registers", "full", "scores")


def sweep_tag(bb, L, s):
    return f"{bb}_L{L}_s{s}"


def sweep_published_file(bb, L, s, stimset):
    """Per-image q16 measures of a sweep readout: v3/scores (primary blocks) or nn/results (the other nine)."""
    t = sweep_tag(bb, L, s)
    if stimset == "stim_v3":
        for p in (os.path.join(ROOT, "v3", "scores", "raw_v3maps", t + ".json"),
                  os.path.join(ROOT, "nn", "results", "raw_v3maps", t + ".json")):
            if os.path.exists(p):
                return p
    else:
        for p in (os.path.join(ROOT, "v3", "scores", "raw", t + ".json"),
                  os.path.join(ROOT, "nn", "results", "raw_alt", t + ".json")):
            if os.path.exists(p):
                return p
    return None


def unit_sizes_seeds(group):
    if group in ("swap", "registers"):
        return (0, 1, 2), SIZES
    if group == "sweep":
        return (0,), SIZES
    return (None,), (None,)


# ====================================================================== rigid-rotation scores (written by Step 3)
def rigid_scores(group, unit):
    """{(seed, size): {display name: {measure: value}}} from neuro/reanalysis/scores_rigid, or None if not (yet)
    scored. Layouts: <tag>.json = {name: {measure: value}} (IIE, MSDB, sweep tags), nulls.json = {null: {name: ...}},
    heads/scores/stim_rigid_<tag>.json = {size: {measure: {name: value}}} (swap and registers heads)."""
    out = {}
    if group == "deepgaze":
        d = load_json(os.path.join(SCORES_RIGID, unit.replace(" ", "_") + ".json"))
        return None if d is None else {(None, None): d}
    if group == "nulls":
        d = load_json(os.path.join(SCORES_RIGID, "nulls.json"))
        return None if d is None or unit not in d else {(None, None): d[unit]}
    if group == "sweep":
        bb, L = unit.split("_L")
        for s in SIZES:
            d = load_json(os.path.join(SCORES_RIGID, sweep_tag(bb, int(L), s) + ".json"))
            if d is None:
                return None
            out[(0, s)] = d
        return out
    if group in ("swap", "registers"):
        fs, ar = unit.split("|")
        for seed in (0, 1, 2):
            d = load_json(os.path.join(SCORES_RIGID, "heads", "scores", f"stim_rigid_{fs}_{ar}_s{seed}.json"))
            if d is None:
                return None
            for s in SIZES:
                out[(seed, s)] = heads_layout(d, s)
        return out
    return None


def heads_layout(d, s):
    """{size: {measure: {name: value}}} (eval_re score file) -> {name: {measure: value}} for one size."""
    lay = {}
    for meas, per in d[str(s)].items():
        for nm, v in per.items():
            lay.setdefault(nm, {})[meas] = v
    return lay


def rigid_units():
    """Every unit that can have rigid-rotation scores, by group."""
    return dict(deepgaze=["DeepGaze IIE", "DeepGaze MSDB"], nulls=list(NULLS),
                sweep=[f"{bb}_L{L}" for bb in SWEEP_LAYERS for L in SWEEP_LAYERS[bb]],
                swap=[f"{f}|{a}" for f in SWAP_FSETS for a in ARCHS],
                registers=[f"{f}|R4" for f in REG_FSETS])


def group_units(group):
    """Units of a model group (MSI-Net is a group of its own and is not in the stim_rigid battery)."""
    return ["MSI-Net"] if group == "msinet" else rigid_units()[group]


def rigid_per_config(group, unit, fn, D, phis=PHIS, sc=None):
    """Per-configuration (K, P) arrays of fn(rec_illusory, rec_control180) for each phi, averaged over the unit's
    (seed, size) cells, from the scoring-pass records of the stim_rigid displays; None if the unit is not scored.
    fn may return NaN (for example a zero mass); the result then carries NaN for that configuration.
    sc (the rigid_scores layout) may be passed directly (unit tests)."""
    sc = rigid_scores(group, unit) if sc is None else sc
    if sc is None:
        return None
    out = {}
    for phi in phis:
        def val(c):
            xs = []
            for rec in sc.values():
                a, b = rec.get(rig_name(c, phi, "illusory")), rec.get(rig_name(c, phi, "control180"))
                if a is None or b is None:
                    return float("nan")
                xs.append(fn(a, b))
            return float(np.mean(xs))
        out[phi] = D.arr(val)
    return out


def safe_log(x):
    return math.log(x) if x is not None and x > 0 else float("nan")
