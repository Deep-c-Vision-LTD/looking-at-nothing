"""
Secondary item 7 of the receptive-field rule with the corrected sign: exploratory analysis of REANALYSIS_RULE.md
Section 6, item 13 (RUN_PLAN.md Step 1, S6). Written 18 November 2026 after the rule was fixed and public; not yet run on
real data. No torch; saved scores only.

Rule text (Section 6, item 13, quoted verbatim):
  "Secondary item 7 of the receptive-field rule, corrected sign: L20 = sign(E) x [r(illusory) - r(mis20)] with the
  unrounded E (DINOv2-B L2: E = -0.0000288, L20 = -0.0052, not +0.0052); Spearman rho with ERF50, its stimulus-resampling
  CI and its 100,000-permutation p recomputed with a copy of the receptive-field rule's code; reported next to the
  stored value (rho = -0.06 [-0.18, +0.13], p = .807) as a logged departure from the code, not from the rule."
  Known (Section 1.1, item 11): "stored rho = -0.06; with the corrected sign rho = -0.023 (Section 6.13)."

The copy: the functions parts_for, spearman, perm_p, boot_rho, ols_r2, adj and assoc below are verbatim copies of
nn/analyze_nn.py (the code of nn/ANALYSIS_RULE.md; checked text-for-text by test_explore.py); X is the same function
with its default design bound when called instead of at import (so that nothing is read before require_public()). The
module-level names they use (C, AA, DA, NPERM, SEED) are bound as in that file: C is a shim with the attributes of
nn/common_nn.py that parts_for reads, AA is v3/scores/analyze_alt imported read only (as common_nn.import_v3 does),
DA = AA.DES["all"], NPERM = 100,000, SEED = 2026. The per-readout L20 and
its bootstrap rows follow nn/analyze_nn.py lines 265 to 267 and 394 to 395, with one change in the corrected run:
  stored code   sgn = np.sign(o["aligned"]["mean"]) or 1.0 for the point L20, and np.sign(recs[n]["E"]["mean"]) for the
                bootstrap rows Lb, where o["aligned"]["mean"] is analyze_alt.summ's mean rounded to 4 decimals. For
                DINOv2-B L2 the rounded mean is -0.0, so the point L20 took sign +1 (the error the rule names) and its
                bootstrap row took sign -0.0, i.e. a row of zeros in the stimulus-resampling CI;
  corrected     sgn = sign of the unrounded mean E for both the point L20 and the bootstrap row.
The script first recomputes the stored analysis with the stored sign logic and checks that it reproduces
nn/results/summary.json ("secondary" -> "all18_L20": rho, rho_ci, perm_p) exactly; then the corrected analysis. Both are
written, with the readouts whose sign changed. This is the departure from the code that the rule asks to be logged
(neuro/reanalysis/DEVIATIONS.md, entry of 18 November 2026).
Usage (after the rule is public): $PY -B explore/item7_corrected.py
Output: results/explore/item7_corrected.json
"""
import itertools, json, os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
from scipy.stats import rankdata                                          # noqa: E402
import excommon as EXC                                                    # noqa: E402

NPERM, SEED = 100_000, 2026
_AA = {}


class C:
    """Shim with the attributes of nn/common_nn.py that the copied functions read (values copied from that file)."""
    NN = os.path.join(EXC.ROOT, "nn")
    V3SC = os.path.join(EXC.ROOT, "v3", "scores")
    RES = os.path.join(NN, "results")
    SIZES = (336, 448)
    LAYERS = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
    SCORED = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}
    BBNAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}

    @staticmethod
    def tag(bb, L, s):
        return f"{bb}_L{L}_s{s}"

    @staticmethod
    def rname(bb, L):
        return f"{C.BBNAME[bb]} L{L}"

    @staticmethod
    def all_readouts():
        return [(bb, L) for bb in ("dinov2", "clip", "siglip") for L in C.LAYERS[bb]]

    @staticmethod
    def load_json(p):
        return json.load(open(p)) if os.path.exists(p) else None


def import_aa():
    """v3/scores/analyze_alt (imports common_alt and v2/scores/analyze_v3; no side effects), read only."""
    if "AA" not in _AA:
        for p in (C.V3SC, os.path.join(EXC.ROOT, "v2", "scores")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import analyze_alt
        _AA["AA"] = analyze_alt
    return _AA["AA"]


class _Lazy:
    def __getattr__(self, k):
        return getattr(import_aa(), k)


AA = _Lazy()
DA = None                                                                 # bound in bind() (AA.DES["all"])


def bind():
    global DA
    DA = import_aa().DES["all"]


# ------------------------------------------------------------------ verbatim from nn/analyze_nn.py
def parts_for(bb, L, source):
    """source 'stored' = v3/scores; otherwise a results directory with raw_v3maps/ and raw_alt/."""
    parts = []
    for s in C.SIZES:
        t = C.tag(bb, L, s)
        if source == "stored":
            v3, al = C.load_json(os.path.join(C.V3SC, "raw_v3maps", t + ".json")), C.load_json(os.path.join(C.V3SC, "raw", t + ".json"))
        else:
            v3, al = C.load_json(os.path.join(source, "raw_v3maps", t + ".json")), C.load_json(os.path.join(source, "raw_alt", t + ".json"))
        if v3 is None or al is None or len(v3) != 480 or len(al) != 360:
            return None
        parts.append(AA.with_alt({**v3, **al}))
    return parts


def X(parts, cond, m="r_inside", D=None):
    D = DA if D is None else D                                            # (original default D=DA, bound at import)
    return AA.arr(D, parts, cond, m)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y):
    n = len(x); rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum()); ry = (ry - ry.mean()) / np.sqrt(((ry - ry.mean()) ** 2).sum())
    obs = float(rx @ ry)
    if n <= 9:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64); kind = f"exact ({len(perms)} permutations)"
    else:
        rng = np.random.RandomState(SEED); perms = np.argsort(rng.rand(NPERM, n), 1); kind = f"{NPERM} random permutations"
    null = ry[perms] @ rx
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), kind


def boot_rho(x, Yb):
    """x: (n,), Yb: (n, B) bootstrap values; Spearman rho per resample."""
    rx = rankdata(x); rx = rx - rx.mean()
    ry = np.apply_along_axis(rankdata, 0, Yb); ry = ry - ry.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx @ ry) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum(0))
    return r


def ols_r2(y, cols):
    Xm = np.column_stack([np.ones(len(y))] + cols) if cols else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ beta; sst = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / sst) if sst > 0 else float("nan"), beta


def adj(r2, n, k):
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else None


def assoc(names, x, y, yb=None, groups=None):
    """Section 5 statistics for predictor x and outcome y over readouts 'names' (lists aligned)."""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(names)
    out = dict(n=n, readouts=names, x=[round(v, 3) for v in x], y=[round(v, 4) for v in y])
    if n < 3:
        out["note"] = "fewer than 3 readouts"; return out
    out["rho"] = spearman(x, y)
    out["perm_p"], out["perm_kind"] = perm_p(x, y)
    if yb is not None:
        rb = boot_rho(x, yb)
        out["rho_ci"] = [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]
        out["rho_boot_frac_pos"] = float(np.nanmean(rb > 0))
    groups = groups if groups is not None else [nm.split()[0] for nm in names]
    G = sorted(set(groups)); qual = [g for g in G if groups.count(g) >= 3]
    out["within"] = {}
    for g in G:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out["within"][g] = dict(n=len(idx), rho=(spearman(x[idx], y[idx]) if len(idx) >= 3 else None))
    if qual:
        idxs = [np.array([i for i, gg in enumerate(groups) if gg == g]) for g in qual]
        T = sum(len(ii) * spearman(x[ii], y[ii]) for ii in idxs) / sum(len(ii) for ii in idxs)
        rng = np.random.RandomState(SEED); null = np.zeros(NPERM)
        for ii in idxs:                                   # within-backbone permutation of x (ranks are permutation-equivariant)
            rx, ry = rankdata(x[ii]), rankdata(y[ii]); rx = rx - rx.mean(); ry = ry - ry.mean()
            den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
            perms = np.argsort(rng.rand(NPERM, len(ii)), 1)
            null += len(ii) * ((rx[perms] @ ry) / den if den > 0 else 0.0)
        null /= sum(len(ii) for ii in idxs)
        out["T"] = float(T); out["T_perm_p"] = float(np.mean(np.abs(null) >= abs(T) - 1e-12)); out["T_backbones"] = qual
        out["n_qual_backbones_rho_pos"] = int(sum(1 for g in qual if out["within"][g]["rho"] is not None and out["within"][g]["rho"] > 0))
    else:
        out["T"] = None; out["T_perm_p"] = None; out["T_backbones"] = []; out["n_qual_backbones_rho_pos"] = 0
    lx = np.log2(x)
    dummies = [np.array([1.0 if gg == g else 0.0 for gg in groups]) for g in G[1:]]
    r2_bb, _ = ols_r2(y, dummies); r2_erf, b1 = ols_r2(y, [lx]); r2_both, b2 = ols_r2(y, [lx] + dummies)
    out["R2_bb"], out["R2_x"], out["R2_both"] = r2_bb, r2_erf, r2_both
    out["adjR2_bb"], out["adjR2_x"], out["adjR2_both"] = adj(r2_bb, n, len(dummies)), adj(r2_erf, n, 1), adj(r2_both, n, 1 + len(dummies))
    out["partial_R2_x_given_bb"] = float((r2_both - r2_bb) / (1 - r2_bb)) if r2_bb < 1 else None
    out["coef_x_joint"] = float(b2[1]); out["coef_x_alone"] = float(b1[1])
    return out


# ------------------------------------------------------------------ L20 per readout (stored and corrected sign logic)
def l20_sign(E_rounded_mean, E_unrounded_mean, mode):
    """(sign for the point L20, sign for the bootstrap row). 'stored': nn/analyze_nn.py lines 265 and 394
    (np.sign(rounded mean) or 1.0; np.sign(rounded mean)); 'corrected': the sign of the unrounded mean for both."""
    if mode == "stored":
        return (np.sign(E_rounded_mean) or 1.0), np.sign(E_rounded_mean)
    s = np.sign(E_unrounded_mean)
    return s, s


def load_erf50():
    """ERF50 per readout: nn/results/erf/<bb>.json layers[L]['ro_sizeavg_all']['erf50'] (load_erf, non-test branch)."""
    out = {}
    for bb in C.LAYERS:
        p = os.path.join(C.RES, "erf", f"{bb}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            for L, rec in d["layers"].items():
                out[(bb, int(L))] = rec["ro_sizeavg_all"]["erf50"]
    return out


def per_readout():
    rows = {}
    for bb, L in C.all_readouts():
        src = "stored" if L in C.SCORED[bb] else C.RES
        parts = parts_for(bb, L, src)
        if parts is None:
            raise SystemExit(f"missing scores for {C.rname(bb, L)}")
        e = X(parts, "illusory") - X(parts, "control180")
        l20 = X(parts, "illusory") - X(parts, "mis20")
        rows[C.rname(bb, L)] = dict(bb=bb, L=L, E_unrounded=float(e.mean()), E_rounded=AA.summ(DA, e)["mean"],
                                    l20_mean=float(l20.mean()), l20_boot=DA.bmeans(l20))
    return rows


def analysis(rows, erf, mode):
    names = [n for n in rows if (rows[n]["bb"], rows[n]["L"]) in erf]
    y, Lb, signs = [], [], {}
    for n in names:
        sp, sb = l20_sign(rows[n]["E_rounded"], rows[n]["E_unrounded"], mode)
        y.append(float(sp * rows[n]["l20_mean"])); Lb.append(sb * rows[n]["l20_boot"])
        signs[n] = dict(point=float(sp), bootstrap=float(sb))
    a = assoc(names, [erf[(rows[n]["bb"], rows[n]["L"])] for n in names], y, np.vstack(Lb), [rows[n]["bb"] for n in names])
    a["signs"] = signs
    a["y_unrounded"] = y                                                  # assoc's "y" is rounded to 4 decimals (copied code)
    return a


def main():
    EXC.require_public()                                              # rule Section 8.1
    bind()
    rows, erf = per_readout(), load_erf50()
    stored_now = analysis(rows, erf, "stored")
    corrected = analysis(rows, erf, "corrected")
    summ = C.load_json(os.path.join(C.RES, "summary.json"))
    ref = (summ or {}).get("secondary", {}).get("all18_L20")
    repro = None
    if ref:
        repro = dict(rho_abs_diff=abs(stored_now["rho"] - ref["rho"]),
                     rho_ci_abs_diff=max(abs(a - b) for a, b in zip(stored_now["rho_ci"], ref["rho_ci"])),
                     perm_p_abs_diff=abs(stored_now["perm_p"] - ref["perm_p"]))
        repro["reproduced"] = bool(max(repro.values()) < 1e-12)
    changed = [n for n in corrected["signs"] if corrected["signs"][n] != stored_now["signs"][n]]
    res = dict(stored_value_quoted_by_rule="rho = -0.06 [-0.18, +0.13], p = .807",
               stored_recomputed=stored_now, stored_reference=ref, stored_reproduction=repro,
               corrected=corrected, readouts_with_changed_sign=changed,
               E_and_L20={n: dict(E_unrounded=rows[n]["E_unrounded"], E_rounded_as_stored=rows[n]["E_rounded"],
                                  L20_stored=stored_now["y_unrounded"][i], L20_corrected=corrected["y_unrounded"][i])
                          for i, n in enumerate(stored_now["readouts"])},
               note="a logged departure from the code of nn/ANALYSIS_RULE.md, not from the rule (REANALYSIS_RULE.md 6.13)")
    if repro is not None and not repro["reproduced"]:
        res["warning"] = "the stored analysis was not reproduced exactly; the corrected value is reported with this flag"
    EXC.write_json("item7_corrected.json", res, os.path.abspath(__file__),
                   inputs=["v3/scores/raw_v3maps", "v3/scores/raw", "nn/results/raw_v3maps", "nn/results/raw_alt",
                           "nn/results/erf/<bb>.json", "nn/results/summary.json"])


if __name__ == "__main__":
    main()
