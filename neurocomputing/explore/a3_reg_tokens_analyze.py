"""
A3 (iv) registers-checkpoint token components, analysis step. Exploratory analysis of REANALYSIS_RULE.md Section 6,
item 8 (iv), under both nulls, with the paired excess of (iii). Written 18 November 2026 after the rule was fixed and
public; not yet run (it needs the token components of a3_reg_tokens_extract.py). No torch.

Rule text (Section 6, item 8, quoted verbatim):
  "A3 readout-free token analysis (decision D5; Critic fix 6): interior index E_int on pixel-identical interior tokens at
  every block of DINOv2-B, CLIP-L and SigLIP-L (336, 448 px and size average). (i) Analysis of record: the rule of 15
  October 2026 (figure-specific if the interior lower bound exceeds the upper bounds of both the background null and the
  control90 null). (ii) Post hoc: the near-exterior null at matched inducer distance (`encoding_nulls.py`). (iii) The
  paired interior-minus-near-exterior excess per configuration with a WCU interval and its retention under mis20 (BCa,
  stability flag). (iv) Registers checkpoint token components, required: all 12 blocks of
  `facebook/dinov2-with-registers-base` at 336 and 448 px, CLS and 4 register tokens dropped, same code path, both
  nulls; they are the only readout-free test of whether the block-12 change between checkpoints exists at the token
  level. (v) The comparison table for the 4 swap feature sets and the 18 sweep blocks (token verdict under each null,
  saliency E with interval, S20). No headline dissociation is drawn from either null."
  Wording (Section 7.6): "The registers token components are reported next to the registers saliency result."

Implementation (same code path as the published analyses, copied verbatim where marked):
  - Per-configuration indices: per_config_vit() of v2/encoding/analyze.py (sets "int" = tokens pixel-identical across
    all 7 images with the footprint inside the inside AOI, "bg" = pixel-identical tokens > 80 px from the figure;
    E = sum dot / sum nrc over the set) and per_config() of looking-at-nothing/analysis/encoding_nulls.py ("int",
    "int_far", "int_near", "ext_near"), read from neuro/reanalysis/A3_tokens/registers/dinov2reg_s<S>/; size average =
    nanmean over 336 and 448 (as both scripts).
  - (i) for every block L = 1..12 and the variants size average, 336, 448: interior, background null and control90 null
    (the int set under the control90 "test") with the cluster bootstrap of common_enc.Boot (10,000 resamples,
    RandomState(2026), percentile intervals); figure-specific if the interior lower bound exceeds both null upper bounds.
    The published code compared bounds rounded to 4 decimals (analyze.summ); the verdict is given on unrounded bounds
    and under the published rounding, and any disagreement is flagged.
  - (ii) for every block (size average): interior "int" against "ext_near"; passes if the interior lower bound exceeds
    the ext_near upper bound (encoding_nulls.py criterion), Boot percentile intervals, unrounded.
  - (iii) per configuration (size average): excess = E_int - E_ext_near (illusory test), WCU interval and WCR p
    (analyze_re.wild_cluster, 12 clusters); retention under mis20 = mean excess(mis20) / mean excess(illusory), BCa,
    stability flag, Fieller (excommon.ratio). Configurations without qualifying tokens make (iii) not computable.
  - Alongside, never as a contrast: the plain DINOv2-B verdicts at the same blocks from v2/encoding/results.json and
    v2/verify/encoding_nulls.json (read only).
Usage (after the rule is public and the extraction has finished): $PY -B explore/a3_reg_tokens_analyze.py
Output: results/explore/A3iv_registers_tokens.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
from scipy.ndimage import distance_transform_edt                          # noqa: E402
import excommon as X                                                      # noqa: E402
import a3_reg_tokens_extract as EX                                        # noqa: E402  (configs, footprints; no torch)

NET, N_HS = "dinov2reg", EX.N_HS
TOK = EX.OUT_ROOT
TESTS = EX.K_TESTS
TI = {t: i for i, t in enumerate(TESTS)}
SETS = ("int", "bg")
NSETS = ("int", "int_far", "int_near", "ext_near")
B, BOOT_SEED = 10000, 2026


# ------------------------------------------------------------------ verbatim from analysis/encoding/common_enc.py
def cells(cfgs, ill):
    """[K][P] configuration indices into cfgs (cells sorted, positions sorted)."""
    cl = {}
    for i, c in enumerate(cfgs):
        if c["illusion"] == ill:
            cl.setdefault(c["cell"], []).append((c["pos"], i))
    return [[i for _, i in sorted(v)] for _, v in sorted(cl.items(), key=lambda kv: str(kv[0]))]


class Boot:
    """Cluster bootstrap over cells (with replacement), positions resampled within the sampled cells.
    The same resample indices are used for every statistic of an illusion (paired). NaN configurations (no qualifying
    token or zero outline direction) are ignored in each resample's mean (nanmean)."""

    def __init__(self, cell_idx, seed=BOOT_SEED, b=B):
        self.cells = np.array(cell_idx)                          # [K, P] config indices
        K, P = self.cells.shape
        rng = np.random.RandomState(seed)
        ci = rng.randint(0, K, (b, K)); pi = rng.randint(0, P, (b, K, P))
        self.flat = self.cells[ci[:, :, None], pi].reshape(b, K * P)   # [B, K*P] resampled config indices

    def dist(self, vals):
        """vals: per-config values indexed like cfgs. Returns the bootstrap distribution of the mean."""
        v = np.asarray(vals, np.float64)[self.flat]
        with np.errstate(invalid="ignore"):
            return np.nanmean(v, 1) if np.isnan(v).any() else v.mean(1)

    def summary(self, vals, dist=None):
        v = np.asarray(vals, np.float64)[self.cells.ravel()]
        n = int(np.sum(~np.isnan(v)))
        if n == 0:
            return dict(mean=None, ci=[None, None], p=None, p_str=None, n=0)
        d = self.dist(vals) if dist is None else dist
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
        return dict(mean=float(np.nanmean(v)), ci=[float(lo), float(hi)], p=float(p), p_str=None, n=n)


# ------------------------------------------------------------------ from v2/encoding/analyze.py (adapted folder)
def per_config_vit(S, CFGS, tok=TOK, load=np.load):
    """analyze.per_config_vit for the registers token folder: E, cos {set: [NCFG, L, T]}, counts."""
    d = os.path.join(tok, f"{NET}_s{S}")
    nL = N_HS; T = len(TESTS); NCFG = len(CFGS)
    E = {s: np.full((NCFG, nL, T), np.nan) for s in SETS}
    cnt = {s: np.zeros(NCFG, int) for s in SETS}
    for ci, cf in enumerate(CFGS):
        if cf["illusion"] != "kanizsa":
            continue
        z = load(os.path.join(d, cf["config"] + ".npz"))
        imgs = list(z["imgs"]); tests = list(z["tests"]); same = z["same"]; ins = z["inside_full"]; far = z["far"]
        allsame = same.all(0)
        masks = {"int": allsame & ins, "bg": allsame & far}
        cnt["int"][ci] = masks["int"].sum(); cnt["bg"][ci] = masks["bg"].sum()
        dot, nrc = z["dot"].astype(np.float64), z["nrc"].astype(np.float64)
        for s in SETS:
            m = masks[s]
            if not m.any():
                continue
            for j, t in enumerate(tests):
                D_ = dot[:, j][:, m].sum(-1); R = nrc[:, m].sum(-1)
                ok = R > 0
                E[s][ci, ok, TI[t]] = D_[ok] / R[ok]
    return E, cnt


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def layer_block(BOOT, E, L):
    """analyze.layer_block (Kanizsa, test illusory): interior, background null, control90 null; figure-specific."""
    t = TI["illusory"]
    r = dict(interior=BOOT.summary(E["int"][:, L, t]), bg_null=BOOT.summary(E["bg"][:, L, t]),
             c90_null=BOOT.summary(E["int"][:, L, TI["control90"]]))
    nulls = [r["bg_null"], r["c90_null"]]

    def fs(a, nl, rnd):
        f = (lambda x: r4(x)) if rnd else (lambda x: x)
        if a["ci"][0] is None or any(n["ci"][1] is None for n in nl):
            return None
        return bool(all(f(a["ci"][0]) > f(n["ci"][1]) for n in nl))
    r["figure_specific"] = fs(r["interior"], nulls, False)
    r["figure_specific_published_rounding"] = fs(r["interior"], nulls, True)
    r["rounding_changes_verdict"] = r["figure_specific"] != r["figure_specific_published_rounding"]
    return r


# ------------------------------------------------------------------ from analysis/encoding_nulls.py (adapted folder)
_geo = {}


def token_geo(cf, G, P, S, load=np.load):
    key = (cf["config"], G, P, S)
    if key in _geo:
        return _geo[key]
    a = load(cf["aoi"]); ins, ind = a["inside"], a["ind"]; fig = ins | ind
    dI = distance_transform_edt(~ind)
    fp = EX.footprints(G, G, P, P, EX.H / S, EX.W / S)
    full_in = np.zeros(G * G, bool); full_out = np.zeros(G * G, bool); mind = np.full(G * G, np.inf)
    for r in range(G):
        for c in range(G):
            y0, y1, x0, x1 = fp[r, c]; k = r * G + c
            if y1 <= y0 or x1 <= x0:
                continue
            full_in[k] = ins[y0:y1, x0:x1].all(); full_out[k] = not fig[y0:y1, x0:x1].any(); mind[k] = dI[y0:y1, x0:x1].min()
    _geo[key] = (full_in, full_out, mind)
    return _geo[key]


def per_config_near(S, CFGS, tok=TOK, load=np.load):
    """encoding_nulls.per_config for the registers token folder."""
    nL = N_HS; NC = len(CFGS)
    KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
    E = {s: {t: np.full((NC, nL), np.nan) for t in ("illusory", "mis20", "control90")} for s in NSETS}
    cnt = {s: np.zeros(NC, int) for s in NSETS}
    for ci in KI:
        cf = CFGS[ci]; z = load(os.path.join(tok, f"{NET}_s{S}", cf["config"] + ".npz"))
        G, P, S_ = [int(v) for v in z["grid"]]; same = z["same"].all(0)
        full_in, full_out, mind = token_geo(cf, G, P, S_, load)
        intm = same & z["inside_full"]
        m = {"int": intm, "int_far": intm & (mind > 30), "int_near": intm & (mind <= 30)}
        dmax = mind[intm].max() if intm.any() else -1
        m["ext_near"] = same & full_out & (mind <= dmax)
        tests = list(z["tests"]); dot = z["dot"].astype(np.float64); nrc = z["nrc"].astype(np.float64)
        for s in NSETS:
            cnt[s][ci] = m[s].sum()
            if not m[s].any():
                continue
            R = nrc[:, m[s]].sum(-1)
            for t in ("illusory", "mis20", "control90"):
                D_ = dot[:, tests.index(t)][:, m[s]].sum(-1); ok = R > 0
                E[s][t][ci, ok] = D_[ok] / R[ok]
    return E, cnt


# ------------------------------------------------------------------ analysis
def size_avg(a, b):
    return np.nanmean(np.stack([a, b]), 0)


def to_design(vals, CFGS, D):
    """Per-config values indexed like CFGS -> (K, P) array in analyze_re Design order."""
    pos = {c["config"]: i for i, c in enumerate(CFGS)}
    return D.arr(lambda c: vals[pos[c]])


def analyse(CFGS, tok=TOK, load=np.load):
    D = X.design()
    BOOT = Boot(cells(CFGS, "kanizsa"))
    KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
    PC = {S: per_config_vit(S, CFGS, tok, load) for S in X.SIZES}
    variants = {"size_average": {s: size_avg(PC[336][0][s], PC[448][0][s]) for s in SETS},
                "px336": PC[336][0], "px448": PC[448][0]}
    out = dict(token_counts={str(S): {s: dict(min=int(PC[S][1][s][KI].min()), median=float(np.median(PC[S][1][s][KI])),
                                              configs_with_none=int((PC[S][1][s][KI] == 0).sum())) for s in SETS}
                             for S in X.SIZES})
    pre = {}
    for vn, E in variants.items():
        pre[vn] = {str(L): layer_block(BOOT, E, L) for L in range(1, N_HS)}
    out["i_prespecified"] = pre
    NE = {S: per_config_near(S, CFGS, tok, load) for S in X.SIZES}
    Eavg = {s: {t: size_avg(NE[336][0][s][t], NE[448][0][s][t]) for t in ("illusory", "mis20", "control90")} for s in NSETS}
    near, exc = {}, {}
    for L in range(1, N_HS):
        row = {s: BOOT.summary(np.where(np.isin(np.arange(len(CFGS)), KI), Eavg[s]["illusory"][:, L], np.nan))
               for s in NSETS}
        ok = row["int"]["ci"][0] is not None and row["ext_near"]["ci"][1] is not None
        row["passes_near_exterior"] = bool(ok and row["int"]["ci"][0] > row["ext_near"]["ci"][1]) if ok else None
        near[str(L)] = row
        ex_ill = to_design(Eavg["int"]["illusory"][:, L] - Eavg["ext_near"]["illusory"][:, L], CFGS, D)
        ex_mis = to_design(Eavg["int"]["mis20"][:, L] - Eavg["ext_near"]["mis20"][:, L], CFGS, D)
        exc[str(L)] = dict(excess=X.lin(ex_ill, D),
                           retention_mis20=(X.ratio(ex_mis, ex_ill, D) if np.all(np.isfinite(ex_mis)) and
                                            np.all(np.isfinite(ex_ill)) else "not computable (missing tokens)"))
    out["ii_near_exterior"] = near
    out["iii_paired_excess"] = exc
    out["near_exterior_token_counts"] = {str(S): {s: dict(median=float(np.median(NE[S][1][s][KI])),
                                                          configs_with_none=int((NE[S][1][s][KI] == 0).sum()))
                                                  for s in NSETS} for S in X.SIZES}
    return out


def plain_context():
    """Plain DINOv2-B verdicts at every block (read only; printed alongside, never as a contrast)."""
    res = X.load_json(os.path.join(X.ROOT, "v2", "encoding", "results.json"))
    nul = X.load_json(os.path.join(X.ROOT, "v2", "verify", "encoding_nulls.json"))
    out = {}
    for L in range(1, N_HS):
        rec = {}
        if res:
            b = res["networks"]["dinov2"]["kanizsa"]["avg"][L]
            rec["prespecified"] = dict(figure_specific=b["figure_specific"], interior=b["interior"], bg_null=b["bg_null"],
                                       c90_null=b.get("c90_null"))
        if nul:
            b = nul["dinov2"]["layers"][str(L)]
            rec["near_exterior"] = dict(int=b["int"], ext_near=b["ext_near"],
                                        passes=bool(b["int"]["ci"][0] is not None and b["ext_near"]["ci"][1] is not None
                                                    and b["int"]["ci"][0] > b["ext_near"]["ci"][1]))
        out[str(L)] = rec
    return out


def main():
    X.require_public()                                                # rule Section 8.1
    CFGS = EX.configs()
    for S in X.SIZES:
        d = os.path.join(TOK, f"{NET}_s{S}")
        missing = [c["config"] for c in CFGS if c["illusion"] == "kanizsa" and
                   not os.path.exists(os.path.join(d, c["config"] + ".npz"))]
        if missing:
            sys.exit(f"token components missing for {len(missing)} Kanizsa configurations at {S} px "
                     f"(run a3_reg_tokens_extract.py first): {missing[:3]}")
    res = analyse(CFGS)
    res["plain_dinov2_context"] = plain_context()
    res["definition"] = ("registers checkpoint facebook/dinov2-with-registers-base, CLS and 4 registers dropped; "
                         "(i) pre-specified rule of 15 October 2026, (ii) near-exterior null, (iii) paired excess; "
                         "no headline dissociation is drawn from either null")
    X.write_json("A3iv_registers_tokens.json", res, os.path.abspath(__file__),
                 inputs=["neuro/reanalysis/A3_tokens/registers/dinov2reg_s336", "neuro/reanalysis/A3_tokens/registers/dinov2reg_s448",
                         "v2/stim_v3 (AOIs)", "v2/encoding/results.json", "v2/verify/encoding_nulls.json"],
                 extra_code=[os.path.join(X.HERE, "a3_reg_tokens_extract.py")])


if __name__ == "__main__":
    main()
