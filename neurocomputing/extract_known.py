"""
extract_known.py: collect every value that was already known when REANALYSIS_RULE.md was written, from result
files that existed before the rule (read only). Used to write the rule's honesty preamble and to fix the list of
eligible P4 contrasts. Computes no new statistic: every number below is copied from an existing JSON, except
  (a) differences of two published point estimates (model minus null, readout minus R4), which follow from them, and
  (b) the percentile-approximate |t| of a published estimate, |mean| / ((hi - lo) / (2 x 1.96)), used only for the
      stability screen of the P4 list.
Revised 18 November 2026 after the rule reviews (before the rule was fixed): adds the square uniform loss that is the
denominator of the square drop ratio (with its percentile-approximate |t|), the pooled and triangle alternation values
that P2 leaves out, the values that bear on P6 (xSoyelNet-ML on the same two blocks, the MSDB Shapley shares of
DINOv2 blocks 7 and 11, the token verdicts of block 11), the registers R4 thresholds under the published rounding,
and the CC gap of every one of the 16 non-R4 cells (the earlier set of rounded gaps merged two pairs of equal values).
Usage: python extract_known.py  -> known_values.json (same folder)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(HERE))   # project root
V3R = json.load(open(os.path.join(ROOT, "v3/scores/results.json")))["models"]
FULL = json.load(open(os.path.join(ROOT, "v3/gpu/full/results_full.json")))
REG = json.load(open(os.path.join(ROOT, "nn/registers/registers_results.json")))
FSETS = ["dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc"]
ARCHS = ["R1n", "R1", "R2", "R3", "R4"]
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]


def tpct(m, ci):
    w = (ci[1] - ci[0]) / (2 * 1.96)
    return abs(m) / w if w > 0 else float("inf")


def v3(model, shape="all"):
    return V3R[model]["results"]["r_inside"][shape]


out = dict(source_files=["v3/scores/results.json (created 2026-10-15 23:05:52)",
                         "v3/gpu/full/results_full.json (created 2026-10-17 21:52:48)",
                         "nn/registers/registers_results.json (2026-11-17)",
                         "nn/registers/full/scores/mit_dinov2reg_*_R4_s*.json (2026-11-17)",
                         "v2/scores/results.json, v2/decompose/results.json, v2/encoding/results.json, "
                         "v2/verify/encoding_nulls.json (15 to 17 October 2026)"])

# ---------------- P1: figure-mass loss under uniform 20 deg rotation, all 60 configurations
p1 = {}
for m in ["DeepGaze IIE", "DeepGaze MSDB"] + ["null " + n for n in NULLS]:
    r = v3(m)
    p1[m] = dict(L20=r["drop_uni"]["20"]["mean"], ci=r["drop_uni"]["20"]["ci"], E=r["aligned"]["mean"],
                 E_ci=r["aligned"]["ci"])
nl = {n: p1["null " + n]["L20"] for n in NULLS}
for m in ("DeepGaze IIE", "DeepGaze MSDB"):
    p1[m]["diff_vs_nulls"] = {n: round(p1[m]["L20"] - v, 4) for n, v in nl.items()}
    p1[m]["min_diff"] = min(p1[m]["diff_vs_nulls"].values())
p1["largest_null_loss"] = max(nl.items(), key=lambda kv: kv[1])
out["P1"] = p1

# ---------------- P2: squares, Delta = r(alt20) - r(mis20)
p2 = {}
for m in ["DeepGaze IIE", "DeepGaze MSDB"] + ["null " + n for n in NULLS]:
    r = v3(m, "square")
    p2[m] = dict(Delta20=r["alt_minus_mis"]["20"]["mean"], ci=r["alt_minus_mis"]["20"]["ci"],
                 drop_ratio20=r["drop_ratio"]["20"]["value"], drop_ratio20_ci=r["drop_ratio"]["20"]["ci"],
                 L20_square=r["drop_uni"]["20"]["mean"], L20_square_ci=r["drop_uni"]["20"]["ci"],
                 L20_square_t_pct=round(tpct(r["drop_uni"]["20"]["mean"], r["drop_uni"]["20"]["ci"]), 2))
    for part in ("all", "triangle"):                      # left out of P2 (squares only); reported next to P2
        rp = v3(m, part)
        p2[m][f"Delta20_{part}"] = dict(mean=rp["alt_minus_mis"]["20"]["mean"], ci=rp["alt_minus_mis"]["20"]["ci"])
        p2[m][f"L20_{part}"] = dict(mean=rp["drop_uni"]["20"]["mean"], ci=rp["drop_uni"]["20"]["ci"])
    dr = v3(m, "triangle")["drop_ratio"]["20"]               # None where the published code did not form the ratio
    p2[m]["drop_ratio20_triangle"] = dict(value=dr["value"], ci=dr["ci"]) if dr else None
nd = {n: p2["null " + n]["Delta20"] for n in NULLS}
for m in ("DeepGaze IIE", "DeepGaze MSDB"):
    p2[m]["diff_vs_nulls"] = {n: round(p2[m]["Delta20"] - v, 4) for n, v in nd.items()}
    p2[m]["min_diff"] = min(p2[m]["diff_vs_nulls"].values())
p2["largest_null_Delta"] = max(nd.items(), key=lambda kv: kv[1])
out["P2"] = p2

# ---------------- swap cells: validity, E, seeds, 90 deg, rotation ratios
cells = {}
for f in FSETS:
    for a in ARCHS:
        mit = FULL["mit"][f][a]
        sb = FULL["section_B"][f][a]["inside"]["aligned"]
        k = FULL["stim_v3"][f][a]["kanizsa"]
        al = FULL["stim_alt"][f][a]["r_inside"]
        c = dict(cc=mit["mean"], cc_seeds=mit["seeds"], threshold=mit["threshold"], valid=mit["valid"],
                 cc_gap_to_R4=round(FULL["mit"][f]["R4"]["mean"] - mit["mean"], 4),
                 E=sb["mean"], E_ci=sb["ci"], E_seed_means=sb["seed_means"], E_seed_cis=sb["seed_cis"],
                 E_t_pct=round(tpct(sb["mean"], sb["ci"]), 2),
                 E_336=k["illusory-control180"]["336|inside"]["mean"], E_336_ci=k["illusory-control180"]["336|inside"]["ci"],
                 E_448=k["illusory-control180"]["448|inside"]["mean"], E_448_ci=k["illusory-control180"]["448|inside"]["ci"],
                 E90=k["illusory-control90"]["avg|inside"]["mean"], E90_ci=k["illusory-control90"]["avg|inside"]["ci"])
        for t in (10, 20, 30):
            for key in (f"ratio_uni{t}", f"ratio_alt{t}", f"drop_ratio{t}"):
                c[key] = dict(value=al[key]["value"], ci=al[key]["ci"], den_ci_excludes_0=al[key]["den_ci_excludes_0"])
            c[f"loss_uni{t}"] = round(al["effect_illusory"]["mean"] - al[f"effect_mis{t}"]["mean"], 4)
        cells[f"{f}|{a}"] = c
out["swap_cells"] = cells

# ---------------- P3: the 10 valid non-reference cells
p3 = []
for f in FSETS:
    sR4 = 1 if cells[f"{f}|R4"]["E"] > 0 else -1
    for a in ARCHS[:-1]:
        c = cells[f"{f}|{a}"]
        if c["valid"]:
            p3.append(dict(cell=f"{f}|{a}", E=c["E"], ci=c["E_ci"], sign_as_R4=(c["E"] > 0) == (sR4 > 0),
                           ci_excludes_0=c["E_ci"][0] > 0 or c["E_ci"][1] < 0, seed_means=c["E_seed_means"],
                           seed_ci_excl0=[ci[0] > 0 or ci[1] < 0 for ci in c["E_seed_cis"]]))
out["P3"] = dict(cells=p3, n=len(p3),
                 n_sign_and_excl0=sum(x["sign_as_R4"] and x["ci_excludes_0"] for x in p3),
                 n_seed_points_same_sign=sum(sum((s > 0) == (x["E"] > 0) for s in x["seed_means"]) for x in p3),
                 n_seed_cis_excl0=sum(sum(x["seed_ci_excl0"]) for x in p3))
below = [k for k, c in cells.items() if not c["valid"]]
out["below_criterion"] = dict(cells=below, all_same_sign_as_R4=all(
    (cells[k]["E"] > 0) == (cells[k.split("|")[0] + "|R4"]["E"] > 0) for k in below))

# ---------------- P4: candidate contrasts (valid non-R4 minus R4, U20 and A20) with stability screen
p4 = []
for x in p3:
    f, a = x["cell"].split("|")
    c, r = cells[x["cell"]], cells[f"{f}|R4"]
    tmin = min(c["E_t_pct"], r["E_t_pct"])
    for meas, key in (("U20", "ratio_uni20"), ("A20", "ratio_alt20")):
        p4.append(dict(contrast=f"{f} {a} minus R4, {meas}", readout=c[key], R4=r[key],
                       diff_point=round(c[key]["value"] - r[key]["value"], 4),
                       denominator_t_pct=dict(readout=c["E_t_pct"], R4=r["E_t_pct"]),
                       stable=tmin >= 4.0, cc_gap_to_R4=c["cc_gap_to_R4"],
                       margin_clearance=round(0.03 - c["cc_gap_to_R4"], 4)))
out["P4_candidates"] = p4
out["P4_eligible"] = [x["contrast"] for x in p4 if x["stable"]]
out["P4_dropped_unstable"] = [x["contrast"] for x in p4 if not x["stable"]]

# ---------------- validity margin curve facts (K.4 row 3)
gaps = sorted(c["cc_gap_to_R4"] for k, c in cells.items() if not k.endswith("R4"))     # one per cell (16)
out["cc_gaps_non_R4"] = gaps
out["cc_gaps_non_R4_unrounded"] = sorted(
    (round(float(FULL["mit"][k.split("|")[0]]["R4"]["mean"]) - float(c["cc"]), 4), k) for k, c in cells.items()
    if not k.endswith("R4"))
out["all_20_cells_sign_as_R4_and_excl0"] = all(
    ((c["E"] > 0) == (cells[k.split("|")[0] + "|R4"]["E"] > 0)) and (c["E_ci"][0] > 0 or c["E_ci"][1] < 0)
    for k, c in cells.items())

# ---------------- registers (R4 only, known) and the thresholds of the new registers cells (published rounding)
out["registers_file_keys"] = list(REG.keys())[:20]
thr = {}
for fs in ("dinov2reg_L12", "dinov2reg_L6"):
    cc = [json.load(open(os.path.join(ROOT, f"nn/registers/full/scores/mit_{fs}_R4_s{s}.json")))["CC"] for s in (0, 1, 2)]
    m = round(sum(cc) / 3, 4)
    thr[fs] = dict(R4_seed_cc=cc, R4_mean=m, threshold=round(m - 0.03, 4))
out["registers_thresholds"] = thr

# ---------------- values that bear on P6 (known; the P6 cells themselves are unseen)
S2 = json.load(open(os.path.join(ROOT, "v2/scores/results.json")))["models"]["xSoyelNet-ML"]["variants"]["native"]["kanizsa"]["effects"]
DEC = json.load(open(os.path.join(ROOT, "v2/decompose/results.json")))["games"]["MSDB_feature_tensors"]["aoi"]["inside"]["players"]
ENC = json.load(open(os.path.join(ROOT, "v2/encoding/results.json")))["networks"]["dinov2"]["kanizsa"]["avg"]
NUL = json.load(open(os.path.join(ROOT, "v2/verify/encoding_nulls.json")))["dinov2"]["layers"]
b11 = ENC[11]
out["P6_related"] = dict(
    xsoyelnet_ml_vs_control180=S2["illusory-control180"]["inside"], xsoyelnet_ml_vs_control90=S2["illusory-control90"]["inside"],
    msdb_shapley_share_dinov2_block11=dict(share=DEC["DINOv2 block 10"]["share"], ci=DEC["DINOv2 block 10"]["share_ci"],
                                           note="0-based block 10 = hidden_states[11]"),
    msdb_shapley_share_dinov2_block7=dict(share=DEC["DINOv2 block 6"]["share"], ci=DEC["DINOv2 block 6"]["share_ci"],
                                          note="0-based block 6 = hidden_states[7]"),
    token_block11_prespecified=dict(interior=b11["interior"], background_null=b11["bg_null"], c90_null=b11["c90_null"],
                                    figure_specific=b11["figure_specific"]),
    token_block11_near_exterior=dict(interior=NUL["11"]["int"], exterior_near=NUL["11"]["ext_near"],
                                     passes=NUL["11"]["int"]["ci"][0] > NUL["11"]["ext_near"]["ci"][1]),
    plain_block6_pointwise=dict(R1n=cells["dinov2_L6|R1n"]["cc"], R1=cells["dinov2_L6|R1"]["cc"],
                                threshold=cells["dinov2_L6|R1n"]["threshold"], valid=[cells["dinov2_L6|R1n"]["valid"],
                                                                                    cells["dinov2_L6|R1"]["valid"]]))

json.dump(out, open(os.path.join(HERE, "known_values.json"), "w"), indent=1)
print("P1 IIE", p1["DeepGaze IIE"]["L20"], p1["DeepGaze IIE"]["ci"], "min diff", p1["DeepGaze IIE"]["min_diff"])
print("P1 MSDB", p1["DeepGaze MSDB"]["L20"], p1["DeepGaze MSDB"]["ci"], "min diff", p1["DeepGaze MSDB"]["min_diff"])
print("largest null loss", p1["largest_null_loss"])
print("P2 IIE", p2["DeepGaze IIE"]["Delta20"], p2["DeepGaze IIE"]["ci"], "min diff", p2["DeepGaze IIE"]["min_diff"])
print("P2 MSDB", p2["DeepGaze MSDB"]["Delta20"], p2["DeepGaze MSDB"]["ci"], "min diff", p2["DeepGaze MSDB"]["min_diff"])
print("largest null Delta", p2["largest_null_Delta"])
print("P3", out["P3"]["n"], "cells;", out["P3"]["n_sign_and_excl0"], "sign+excl0; seed points", out["P3"]["n_seed_points_same_sign"],
      "; seed CIs excl0", out["P3"]["n_seed_cis_excl0"])
for x in p3:
    print("  ", x["cell"], x["E"], x["ci"], "t~", cells[x["cell"]]["E_t_pct"], "seeds", x["seed_means"])
print("below criterion", below, out["below_criterion"]["all_same_sign_as_R4"])
for x in p4:
    print("P4", x["contrast"], x["readout"]["value"], x["readout"]["ci"], "R4", x["R4"]["value"], x["R4"]["ci"],
          "diff", x["diff_point"], "t", x["denominator_t_pct"], "stable", x["stable"])
print("gaps", gaps)
print("all 20 cells sign+excl0:", out["all_20_cells_sign_as_R4_and_excl0"])
for f in FSETS:
    print(f, "R4 E", cells[f + "|R4"]["E"], cells[f + "|R4"]["E_ci"], "t", cells[f + "|R4"]["E_t_pct"])
