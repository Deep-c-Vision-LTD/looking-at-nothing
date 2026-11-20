"""
a3_tokens_published.py: A3 readout-free token analysis for the three published ViTs (DINOv2-B, CLIP-L, SigLIP-L), items
(i), (ii), (iii) and (v) of REANALYSIS_RULE.md Section 6 item 8, under both nulls, from the saved token components
(RUN_PLAN.md Step 1, S5). Exploratory. Written 19 November 2026 (about 00:00 UTC), after P1 to P5, A3 (iv) for the
registers checkpoint and the other exploratory analyses had been run and read (DEVIATIONS.md D11). No torch.

Rule text (Section 6, item 8, quoted verbatim):
  "A3 readout-free token analysis (decision D5; Critic fix 6): interior index E_int on pixel-identical interior tokens at
  every block of DINOv2-B, CLIP-L and SigLIP-L (336, 448 px and size average). (i) Analysis of record: the rule of 15
  October 2026 (figure-specific if the interior lower bound exceeds the upper bounds of both the background null and the
  control90 null). (ii) Post hoc: the near-exterior null at matched inducer distance (`encoding_nulls.py`). (iii) The
  paired interior-minus-near-exterior excess per configuration with a WCU interval and its retention under mis20 (BCa,
  stability flag). (iv) Registers checkpoint token components, (...). (v) The comparison table for the 4 swap feature
  sets and the 18 sweep blocks (token verdict under each null, saliency E with interval, S20). No headline dissociation
  is drawn from either null."
  Wording (Section 7.6): "Both nulls named; the pre-specified one (background and control90) is the analysis of record.
  Mandatory content: CLIP-L passes at blocks 7 to 24 under it and at none under the near-exterior null; "Of the blocks
  read by a readout, only DINOv2-B block 6 passes both (block 7 also passes both)"; "for the three swap transformer
  blocks the pre-specified verdict agrees with the saliency sign; over the 18 sweep blocks it agrees in 12 and disagrees
  in 6", the six named."

This step runs after the two copied scripts of explore/tokens/ (RUN_PLAN S5), each its own process:
  analyze.py         (copy of v2/encoding/analyze.py) -> A3_tokens/published/results.json, per_config_indices.npz,
                     fig_*.png, report.md
  encoding_nulls.py  (copy of looking-at-nothing/analysis/encoding_nulls.py) -> A3_tokens/published/encoding_nulls.json
and compares their outputs leaf by leaf with the published v2/encoding/results.json, per_config_indices.npz and
v2/verify/encoding_nulls.json (the "created" time stamp excluded).

Implementation (choices fixed before any token result was read in this session):
  - (i) per block L = 1..n (hidden_states[L]; L = 0, the embeddings, is not a block) and variant (size average, 336,
    448): interior, background null and control90 null with the cluster bootstrap of common_enc.Boot (10,000,
    RandomState(2026), percentile), from the copied analyze.per_config_vit; figure-specific if the interior lower bound
    exceeds both null upper bounds. The published code compares bounds rounded to 4 decimals; the verdict is given on
    unrounded bounds (analysis here) and with the published rounding (continuity), and every disagreement is listed.
  - (ii) per block (size average): "int" against "ext_near" (per_config of encoding_nulls.py, copied verbatim below),
    passes if the int lower bound exceeds the ext_near upper bound (unrounded, as encoding_nulls.py).
  - (iii) per block (size average): per configuration excess = E_int - E_ext_near for the illusory test, WCU interval and
    WCR p (analyze_re.wild_cluster, 12 clusters, through excommon.lin); retention under mis20 = mean excess(mis20) / mean
    excess(illusory), BCa with the stability flag and Fieller (excommon.ratio). A configuration without qualifying tokens
    at both sizes makes (iii) not computable for that block (stated).
  - (v) rows: the three swap transformer blocks (dinov2_L12, dinov2_L6, siglip_L24) with R4's saliency E (seeds and
    sizes averaged) and S20 = 1 - U20 of R4; iie_shapenetc (CNN) with its saliency E and S20 and the per-layer CNN
    verdicts of the published analysis as context (the near-exterior null was not built for the CNN); the 18 sweep blocks
    (single seed, sizes averaged; rule measure r_inside). E: WCU interval (rule) and the published two-stage percentile
    interval (continuity, rule 1.1 item 9 was computed with it). S20: 1 - U20 with the BCa interval of U20 mapped by
    s -> 1 - s and the stability flag of U20. Agreement of a token verdict with the saliency sign (the reading that
    reproduces rule 1.1 item 9): "figure-specific" agrees if the E interval lies above 0, "not figure-specific" agrees if
    it does not. Counted under both intervals and both nulls.
Usage (after the two copies have run): $PY -B -I explore/tokens/a3_tokens_published.py
Outputs: results/explore/A3_tokens_published.json, results/explore/A3_tokens_published.md,
         A3_tokens/published/tab_tokens.tex
"""
import os, sys, time
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                                 # explore/ (excommon, explore_saved)
import numpy as np                                                        # noqa: E402
from scipy.ndimage import distance_transform_edt                          # noqa: E402
import excommon as X                                                      # noqa: E402

AR = X.AR
NETS = ("dinov2", "clip", "siglip")
NAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}
SWAP_BLOCKS = {"dinov2_L12": ("dinov2", 12), "dinov2_L6": ("dinov2", 6), "siglip_L24": ("siglip", 24)}
PUB_DIR = os.path.join(X.RE, "A3_tokens", "published")
NSETS = ("int", "int_far", "int_near", "ext_near")


def r4(x):
    return None if x is None else round(float(x), 4)


def verdict(interior, nulls, rounded=False):
    """Rule of 15 October 2026: the interior lower bound exceeds every null's upper bound (None if undefined)."""
    f = r4 if rounded else (lambda x: x)
    if interior["ci"][0] is None or any(n["ci"][1] is None for n in nulls):
        return None
    return bool(all(f(interior["ci"][0]) > f(n["ci"][1]) for n in nulls))


def agrees(fs_verdict, ci):
    """A token verdict agrees with the saliency sign: figure-specific <-> the E interval lies above 0."""
    if fs_verdict is None or ci is None or ci[0] is None:
        return None
    return bool(fs_verdict) == bool(ci[0] > 0)


def s20(rat):
    """S20 = 1 - U20 from an excommon.ratio record of U20 (BCa interval mapped by s -> 1 - s)."""
    if rat.get("value") is None:
        return dict(value=None, note=rat.get("note"))
    lo, hi = rat["ci_bca"]
    return dict(value=1 - rat["value"], ci_bca=[1 - hi, 1 - lo], U20=rat["value"], U20_ci_bca=rat["ci_bca"],
                stable=rat["stable"], reportable=rat["reportable"], flag=rat.get("flag"), t_den=rat["t_den"])


# ------------------------------------------------------------------ verbatim from analysis/encoding_nulls.py
_geo = {}


def token_geo(cf, G, P, S, CE):
    key = (cf["config"], G, P, S)
    if key in _geo:
        return _geo[key]
    a = np.load(cf["aoi"]); ins, ind = a["inside"], a["ind"]; fig = ins | ind
    dI = distance_transform_edt(~ind)
    fp = CE.footprints(G, G, P, P, CE.H / S, CE.W / S)
    full_in = np.zeros(G * G, bool); full_out = np.zeros(G * G, bool); mind = np.full(G * G, np.inf)
    for r in range(G):
        for c in range(G):
            y0, y1, x0, x1 = fp[r, c]; k = r * G + c
            if y1 <= y0 or x1 <= x0:
                continue
            full_in[k] = ins[y0:y1, x0:x1].all(); full_out[k] = not fig[y0:y1, x0:x1].any(); mind[k] = dI[y0:y1, x0:x1].min()
    _geo[key] = (full_in, full_out, mind)
    return _geo[key]


def per_config_near(net, S, CFGS, CE):
    nL = CE.VIT[net]["n_hs"]; NC = len(CFGS)
    KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
    E = {s: {t: np.full((NC, nL), np.nan) for t in ("illusory", "mis20", "control90")} for s in NSETS}
    cnt = {s: np.zeros(NC, int) for s in NSETS}
    for ci in KI:
        cf = CFGS[ci]; z = np.load(os.path.join(CE.TOK, f"{net}_s{S}", cf["config"] + ".npz"))
        G, P, S_ = [int(v) for v in z["grid"]]; same = z["same"].all(0)
        full_in, full_out, mind = token_geo(cf, G, P, S_, CE)
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


# ------------------------------------------------------------------ reproduction checks
def compare_files():
    import json
    from explore_saved import compare
    out = {}
    for name, pub in (("results.json", os.path.join(X.ROOT, "v2", "encoding", "results.json")),
                      ("encoding_nulls.json", os.path.join(X.ROOT, "v2", "verify", "encoding_nulls.json"))):
        a = json.load(open(os.path.join(PUB_DIR, name))); b = json.load(open(pub))
        if isinstance(a, dict):
            a.pop("created", None); b.pop("created", None)
        c = compare(a, b)
        out[name] = dict(c, reproduced=os.path.relpath(os.path.join(PUB_DIR, name), X.RE), published=os.path.relpath(pub, X.ROOT))
    za = np.load(os.path.join(PUB_DIR, "per_config_indices.npz")); zb = np.load(os.path.join(X.ROOT, "v2", "encoding", "per_config_indices.npz"))
    worst, n, keys_eq = 0.0, 0, sorted(za.files) == sorted(zb.files)
    for k in za.files:
        if k in zb.files:
            x, y = za[k], zb[k]
            if x.dtype.kind in "fc":
                d = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
                worst = max(worst, float(d)); n += 1
                assert np.array_equal(np.isnan(x), np.isnan(y)), k
            else:
                assert np.array_equal(x, y), k
    out["per_config_indices.npz"] = dict(same_keys=keys_eq, n_float_arrays=n, max_abs_diff=worst)
    return out


# ------------------------------------------------------------------ analysis
def to_design(vals, CFGS, D):
    pos = {c["config"]: i for i, c in enumerate(CFGS)}
    return D.arr(lambda c: vals[pos[c]])


def analyse_net(net, AN, CE, CFGS, BOOT, D, pub_res):
    nL = CE.VIT[net]["n_hs"]
    KI = [i for i, c in enumerate(CFGS) if c["illusion"] == "kanizsa"]
    PC = {S: AN.per_config_vit(net, S) for S in CE.SIZES}
    TI = AN.TI
    variants = {"size_average": {s: AN.size_avg(PC[336]["E"][s], PC[448]["E"][s]) for s in ("int", "bg")},
                "px336": PC[336]["E"], "px448": PC[448]["E"]}
    pub_key = {"size_average": "avg", "px336": "336", "px448": "448"}
    pre, disagree = {}, []
    for vn, E in variants.items():
        pre[vn] = {}
        for L in range(1, nL):
            blk = dict(interior=BOOT.summary(E["int"][:, L, TI["illusory"]]),
                       bg_null=BOOT.summary(E["bg"][:, L, TI["illusory"]]),
                       c90_null=BOOT.summary(E["int"][:, L, TI["control90"]]))
            nulls = [blk["bg_null"], blk["c90_null"]]
            blk["figure_specific"] = verdict(blk["interior"], nulls)
            blk["figure_specific_published_rounding"] = verdict(blk["interior"], nulls, rounded=True)
            blk["published_verdict"] = pub_res["networks"][net]["kanizsa"][pub_key[vn]][L]["figure_specific"]
            blk["reproduces_published_verdict"] = blk["figure_specific_published_rounding"] == blk["published_verdict"]
            if blk["figure_specific"] != blk["figure_specific_published_rounding"]:
                disagree.append(dict(variant=vn, block=L))
            pre[vn][str(L)] = blk
    NE = {S: per_config_near(net, S, CFGS, CE) for S in CE.SIZES}
    Eavg = {s: {t: AN.size_avg(NE[336][0][s][t], NE[448][0][s][t]) for t in ("illusory", "mis20", "control90")}
            for s in NSETS}
    near, exc = {}, {}
    mask = np.isin(np.arange(len(CFGS)), KI)
    for L in range(1, nL):
        row = {s: BOOT.summary(np.where(mask, Eavg[s]["illusory"][:, L], np.nan)) for s in NSETS}
        ok = row["int"]["ci"][0] is not None and row["ext_near"]["ci"][1] is not None
        row["passes_near_exterior"] = bool(row["int"]["ci"][0] > row["ext_near"]["ci"][1]) if ok else None
        near[str(L)] = row
        ex_ill = to_design(Eavg["int"]["illusory"][:, L] - Eavg["ext_near"]["illusory"][:, L], CFGS, D)
        ex_mis = to_design(Eavg["int"]["mis20"][:, L] - Eavg["ext_near"]["mis20"][:, L], CFGS, D)
        n_missing = int(np.sum(~np.isfinite(ex_ill)))
        exc[str(L)] = dict(excess=X.lin(ex_ill, D), n_configurations_without_tokens=n_missing,
                           retention_mis20=(X.ratio(ex_mis, ex_ill, D) if np.all(np.isfinite(ex_mis)) and n_missing == 0
                                            else "not computable (configurations without qualifying tokens)"))
    counts = {str(S): {s: dict(median=float(np.median(NE[S][1][s][KI])), configs_with_none=int((NE[S][1][s][KI] == 0).sum()))
                       for s in NSETS} for S in CE.SIZES}
    passes = dict(prespecified_size_average=[L for L in range(1, nL) if pre["size_average"][str(L)]["figure_specific"]],
                  prespecified_size_average_published_rounding=[L for L in range(1, nL)
                                                                if pre["size_average"][str(L)]["figure_specific_published_rounding"]],
                  near_exterior=[L for L in range(1, nL) if near[str(L)]["passes_near_exterior"]])
    passes["both"] = sorted(set(passes["prespecified_size_average"]) & set(passes["near_exterior"]))
    return dict(n_blocks=nL - 1, i_prespecified=pre, rounding_changes_verdict=disagree, ii_near_exterior=near,
                iii_paired_excess=exc, near_exterior_token_counts=counts, passes=passes)


def saliency_rows(D):
    """(v) saliency side: swap R4 cells (seeds and sizes averaged), iie_shapenetc R4, 18 sweep blocks."""
    from explore_saved import cell_getter, sweep_getter
    rows = {}
    units = [(f"{fs} (R4)", cell_getter(fs, "R4"), f"swap:{fs}") for fs in ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")]
    units += [(f"sweep {bb}_L{L}", sweep_getter(bb, L), f"sweep:{bb}_L{L}") for bb in NETS for L in X.SWEEP_LAYERS[bb]]
    for name, v, key in units:
        E = AR.per_config(D, v, "illusory", "control180")
        nU, dU = AR.ratio_parts(D, v, "U20")
        rows[key] = dict(unit=name, E=X.lin(E, D), E_percentile=AR.pct_summary(E, D.idx), S20=s20(X.ratio(nU, dU, D)))
    return rows


def table_v(nets, sal, pub_res):
    rows = []
    for key, (net, L) in SWAP_BLOCKS.items():
        rows.append((key, "swap R4", net, L))
    for net in NETS:
        for L in X.SWEEP_LAYERS[net]:
            rows.append((f"{net}_L{L}", "sweep", net, L))
    out, agree = [], {k: [] for k in ("pre_wcu", "pre_pct", "near_wcu", "near_pct")}
    for key, kind, net, L in rows:
        n = nets[net]
        pre = n["i_prespecified"]["size_average"][str(L)]
        nr = n["ii_near_exterior"][str(L)]
        s = sal[("swap:" if kind == "swap R4" else "sweep:") + key]
        rec = dict(row=key, kind=kind, network=NAME[net], block=L,
                   prespecified=pre["figure_specific"], prespecified_published_rounding=pre["figure_specific_published_rounding"],
                   near_exterior=nr["passes_near_exterior"], E=s["E"], E_percentile=s["E_percentile"], S20=s["S20"],
                   agree_prespecified_wcu=agrees(pre["figure_specific"], s["E"]["ci"]),
                   agree_prespecified_percentile=agrees(pre["figure_specific_published_rounding"], s["E_percentile"]["ci"]),
                   agree_near_exterior_wcu=agrees(nr["passes_near_exterior"], s["E"]["ci"]),
                   agree_near_exterior_percentile=agrees(nr["passes_near_exterior"], s["E_percentile"]["ci"]))
        out.append(rec)
        if kind == "sweep":
            agree["pre_wcu"].append((key, rec["agree_prespecified_wcu"]))
            agree["pre_pct"].append((key, rec["agree_prespecified_percentile"]))
            agree["near_wcu"].append((key, rec["agree_near_exterior_wcu"]))
            agree["near_pct"].append((key, rec["agree_near_exterior_percentile"]))
    summ = {k: dict(n=len(v), n_agree=sum(bool(x) for _, x in v), disagree=[r for r, x in v if not x]) for k, v in agree.items()}
    summ["swap_blocks_prespecified_agree_wcu"] = [r["row"] for r in out if r["kind"] == "swap R4" and r["agree_prespecified_wcu"]]
    summ["swap_blocks_prespecified_agree_percentile"] = [r["row"] for r in out if r["kind"] == "swap R4" and r["agree_prespecified_percentile"]]
    cnn = pub_res["cnn"].get("iie", {})
    iie = dict(row="iie_shapenetc", kind="swap R4 (CNN)", E=sal["swap:iie_shapenetc"]["E"],
               E_percentile=sal["swap:iie_shapenetc"]["E_percentile"], S20=sal["swap:iie_shapenetc"]["S20"],
               cnn_prespecified_per_layer={L: cnn["per_layer"][L]["kanizsa"]["figure_specific_plan"] for L in cnn.get("layers", [])},
               near_exterior="not built for the CNN (encoding_nulls.py covers the three ViTs only)")
    return out, iie, summ


def mandatory_checks(nets, summ):
    """Rule 7.6 statements against the recomputed verdicts (size average)."""
    read = {"dinov2": sorted(set(X.SWEEP_LAYERS["dinov2"]) | {6, 12}), "clip": list(X.SWEEP_LAYERS["clip"]),
            "siglip": sorted(set(X.SWEEP_LAYERS["siglip"]) | {24})}
    both_read = {net: [L for L in nets[net]["passes"]["both"] if L in read[net]] for net in NETS}
    pre_clip = nets["clip"]["passes"]["prespecified_size_average_published_rounding"]
    return dict(
        clip_prespecified_blocks=pre_clip, clip_prespecified_is_7_to_24=pre_clip == list(range(7, 25)),
        clip_near_exterior_blocks=nets["clip"]["passes"]["near_exterior"],
        clip_near_exterior_none=nets["clip"]["passes"]["near_exterior"] == [],
        blocks_read_by_a_readout_passing_both=both_read,
        only_dinov2_block6_among_read=(both_read == {"dinov2": [6], "clip": [], "siglip": []}),
        dinov2_block7_passes_both=7 in nets["dinov2"]["passes"]["both"],
        sweep_agreement_percentile=summ["pre_pct"], sweep_agreement_wcu=summ["pre_wcu"],
        rule_1_1_item9_six=["dinov2_L8", "clip_L20", "clip_L24", "siglip_L4", "siglip_L8", "siglip_L16"],
        six_reproduced_percentile=sorted(summ["pre_pct"]["disagree"]) == sorted(
            ["dinov2_L8", "clip_L20", "clip_L24", "siglip_L4", "siglip_L8", "siglip_L16"]),
        note="readout-read blocks: the 18 sweep blocks and the swap blocks DINOv2-B 6, 12 and SigLIP-L 24 (A14's blocks 7 "
             "and 11 are not trained yet)")


# ------------------------------------------------------------------ outputs
def f4(x):
    return "n/a" if x is None else (f"{x:+.2e}" if (x != 0 and abs(x) < 5e-5) else f"{x:+.4f}")


def ci4(c):
    return "n/a" if c is None or c[0] is None else f"[{f4(c[0])}, {f4(c[1])}]"


def rat_txt(r, key="ci_bca"):
    """A ratio record as text; a ratio whose denominator interval includes 0 is "not reportable" (rule 4.3), no value."""
    if not isinstance(r, dict):
        return str(r)
    if r.get("value") is None:
        return "n/a"
    if str(r.get("flag") or "").startswith("not reportable") or r.get("reportable") is False:
        return "not reportable (denominator interval includes 0)"
    return f"{r['value']:.4f} {ci4(r[key])}" + (f" ({r['flag']})" if r.get("flag") else "")


def yn(v):
    return "n/a" if v is None else ("pass" if v else "fail")


def write_md(res, path):
    L = []
    w = L.append
    w("# A3 token analysis of the published ViTs under both nulls (RUN_PLAN S5)\n")
    w("**EXPLORATORY** (REANALYSIS_RULE.md Section 6 item 8 (i), (ii), (iii), (v); never decision-bearing; no headline "
      "dissociation is drawn from either null, rule 7.6). The pre-specified rule of 15 October 2026 (background and "
      "control90 nulls) is the analysis of record; the near-exterior null is post hoc. Registers checkpoint (iv): "
      "`A3iv_registers_tokens.json` (earlier step).\n")
    rc = res["reproduction"]
    w("## 0. Reproduction of the published token analyses (copies in explore/tokens/, outputs in A3_tokens/published/)\n")
    for k in ("results.json", "encoding_nulls.json"):
        c = rc[k]
        w(f"- {k}: {c['n_leaves']} leaves ({c['n_numeric']} numeric), max |difference| {c['max_abs_diff']}, mismatches "
          f"{c['n_mismatch']} (identical: {c['identical']}).")
    p = rc["per_config_indices.npz"]
    w(f"- per_config_indices.npz: same keys {p['same_keys']}, {p['n_float_arrays']} float arrays, max |difference| "
      f"{p['max_abs_diff']}.")
    w(f"- Published verdicts reproduced from the bootstrap with the published rounding: {res['verdict_reproduction']['n_agree']} "
      f"of {res['verdict_reproduction']['n']} (all blocks, three variants, three ViTs). Verdict changes when unrounded "
      f"bounds are used: {res['verdict_reproduction']['rounding_changes']}.\n")
    w("## 1. Blocks that pass, by null (size average)\n")
    w("| network | (i) pre-specified, unrounded | (i) published rounding | (ii) near-exterior | both |")
    w("|---|---|---|---|---|")
    for net in NETS:
        ps = res["networks"][net]["passes"]
        w(f"| {NAME[net]} | {ps['prespecified_size_average']} | {ps['prespecified_size_average_published_rounding']} | "
          f"{ps['near_exterior']} | {ps['both']} |")
    m = res["mandatory_checks_rule_7_6"]
    w(f"\n- Rule 7.6: CLIP-L pre-specified 7 to 24: {m['clip_prespecified_is_7_to_24']}; CLIP-L near-exterior none: "
      f"{m['clip_near_exterior_none']}; blocks read by a readout passing both: {m['blocks_read_by_a_readout_passing_both']} "
      f"(only DINOv2-B 6: {m['only_dinov2_block6_among_read']}); DINOv2-B block 7 passes both: {m['dinov2_block7_passes_both']}.")
    w(f"- Sweep agreement (pre-specified verdict with the saliency sign): percentile intervals "
      f"{m['sweep_agreement_percentile']['n_agree']} of 18, disagreeing {m['sweep_agreement_percentile']['disagree']} "
      f"(rule 1.1 item 9 six reproduced: {m['six_reproduced_percentile']}); WCU intervals {m['sweep_agreement_wcu']['n_agree']} "
      f"of 18, disagreeing {m['sweep_agreement_wcu']['disagree']}.\n")
    w("## 2. (iii) Paired interior-minus-near-exterior excess (size average; WCU) and its retention under mis20 (BCa)\n")
    w("| network | block | excess [WCU], p | retention mis20 [BCa] |")
    w("|---|---|---|---|")
    for net in NETS:
        for Lb, e in res["networks"][net]["iii_paired_excess"].items():
            ex = e["excess"]
            rts = rat_txt(e["retention_mis20"])
            exs = (f"{f4(ex['mean'])} {ci4(ex['ci'])}, p {ex['p']:.4g}" if ex.get("mean") is not None else ex.get("note"))
            w(f"| {NAME[net]} | {Lb} | {exs} | {rts} |")
    w("\n## 3. (v) Comparison table: token verdict under each null, saliency E, S20\n")
    w("| row | token (i) | token (ii) | E [WCU], p | E [percentile] | S20 [BCa] | agree (i) WCU / pct | agree (ii) WCU / pct |")
    w("|---|---|---|---|---|---|---|---|")
    for r in res["table_v"]["rows"]:
        st = rat_txt(r["S20"])
        e = r["E"]
        w(f"| {r['row']} ({r['kind']}) | {yn(r['prespecified'])} | {yn(r['near_exterior'])} | {f4(e['mean'])} {ci4(e['ci'])}, "
          f"p {e['p']:.4g} | {ci4(r['E_percentile']['ci'])} | {st} | {r['agree_prespecified_wcu']} / "
          f"{r['agree_prespecified_percentile']} | {r['agree_near_exterior_wcu']} / {r['agree_near_exterior_percentile']} |")
    i = res["table_v"]["iie_shapenetc"]
    fsl = [k for k, v in i["cnn_prespecified_per_layer"].items() if v is True]
    w(f"| iie_shapenetc (swap R4, CNN) | per layer: passes at {fsl if fsl else 'no layer'} (published CNN analysis) | not built | "
      f"{f4(i['E']['mean'])} {ci4(i['E']['ci'])}, p {i['E']['p']:.4g} | {ci4(i['E_percentile']['ci'])} | "
      f"{rat_txt(i['S20'])} | n/a | n/a |")
    w("\nToken verdicts: (i) unrounded bounds; the published-rounding verdict differs only where listed in Section 0. "
      "Agreement: figure-specific agrees with a saliency E interval above 0, not figure-specific with one that is not.")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path, flush=True)


def write_tex(res, path):
    L = [r"% A3 (v) comparison table (exploratory; REANALYSIS_RULE.md Section 6 item 8 (v)); written by explore/tokens/a3_tokens_published.py",
         r"\begin{tabular}{lllll}", r"\hline",
         r"Block & Pre-specified & Near-exterior & $E$ [95\% WCU] & $S_{20}$ [95\% BCa] \\", r"\hline"]
    for r in res["table_v"]["rows"]:
        e, s = r["E"], r["S20"]
        nr = str(s.get("flag") or "").startswith("not reportable") or s.get("reportable") is False
        st = ("n/a" if s.get("value") is None else "not reportable" if nr else
              f"{s['value']:.3f} [{s['ci_bca'][0]:.3f}, {s['ci_bca'][1]:.3f}]" + ("$^u$" if s.get("flag") else ""))
        nm = r["row"].replace("_", r"\_")
        L.append(f"{nm} & {yn(r['prespecified'])} & {yn(r['near_exterior'])} & {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] & {st} \\\\")
    L += [r"\hline", r"\end{tabular}", "% u: ratio flagged unstable (denominator |t| < 4); not reportable: denominator interval includes 0"]
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path, flush=True)


def main():
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    n_hashed = X.require_public()                                     # rule Section 8.1
    for f in ("results.json", "per_config_indices.npz", "encoding_nulls.json"):
        if not os.path.exists(os.path.join(PUB_DIR, f)):
            sys.exit(f"missing {f} in A3_tokens/published (run explore/tokens/analyze.py and encoding_nulls.py first)")
    sys.path.insert(0, HERE)
    import analyze as AN                                              # the copy (module import: main() not run)
    import common_enc as CE                                           # v2/encoding (read only), via the copy's path
    import json
    pub_res = json.load(open(os.path.join(X.ROOT, "v2", "encoding", "results.json")))
    CFGS = AN.CFGS
    BOOT = AN.BOOT["kanizsa"]
    D = X.design()
    res = dict(reproduction=compare_files())
    X.log("reproduction compared")
    res["networks"] = {net: analyse_net(net, AN, CE, CFGS, BOOT, D, pub_res) for net in NETS}
    X.log("token analyses done")
    vr = [b["reproduces_published_verdict"] for net in NETS for vn in ("size_average", "px336", "px448")
          for b in res["networks"][net]["i_prespecified"][vn].values()]
    res["verdict_reproduction"] = dict(n=len(vr), n_agree=sum(vr),
                                       rounding_changes={net: res["networks"][net]["rounding_changes_verdict"] for net in NETS})
    sal = saliency_rows(D)
    rows, iie, summ = table_v(res["networks"], sal, pub_res)
    res["table_v"] = dict(rows=rows, iie_shapenetc=iie, agreement=summ,
                          definition="agreement: figure-specific <-> saliency E interval above 0 (the reading that "
                                     "reproduces rule 1.1 item 9); E of swap rows = R4 (seeds and sizes averaged)")
    res["mandatory_checks_rule_7_6"] = mandatory_checks(res["networks"], summ)
    res["run"] = dict(start_utc=start, end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), compute_s=time.time() - t0,
                      hashed_files_verified=n_hashed)
    X.write_json("A3_tokens_published.json", res, os.path.abspath(__file__),
                 inputs=["v2/encoding/tokens/<net>_s<S>/*.npz", "v2/stim_v3 (AOIs)", "v2/encoding/results.json",
                         "v2/encoding/per_config_indices.npz", "v2/verify/encoding_nulls.json",
                         "neuro/reanalysis/A3_tokens/published/{results.json, per_config_indices.npz, encoding_nulls.json}",
                         "saliency: v3/gpu/full/scores (R4), v3/scores and nn/results (sweep)"],
                 extra_code=[os.path.join(HERE, "analyze.py"), os.path.join(HERE, "encoding_nulls.py"),
                             os.path.join(HERE, "plots.py"), os.path.join(HERE, "report.py"),
                             os.path.join(os.path.dirname(HERE), "explore_saved.py")])
    clean = X._clean(res)
    write_md(clean, os.path.join(X.RESULTS, "A3_tokens_published.md"))
    write_tex(clean, os.path.join(PUB_DIR, "tab_tokens.tex"))


if __name__ == "__main__":
    main()
