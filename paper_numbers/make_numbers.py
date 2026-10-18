#!/usr/bin/env python3
"""
paper_numbers/make_numbers.py: every number of the manuscript, with its source.

Reads result files only (paths relative to the results root, paths.RESULTS = <repo>/results unless LAN_RESULTS is set;
large files are read from their .json.xz copies when no plain .json exists):
  v2/conventions.md                        configuration counts and the human sample (children 97 of 127), by regex
  v2/scores/results.json                   survey: six saliency models + edge-energy null (stim_v3)
  v2/scores/raw/<tag>.json                 EXTRA SOURCE, polarity split only: per-image r of the same scoring run that
                                           v2/scores/results.json summarises (no file in the listed set holds the split)
  v2/verify/position_check.json            effect by figure position
  v2/verify/core_aoi_D30_835295.json       core-AOI effects for survey models not scored in v3
  v2/verify/local_nulls_stats.json         local-null Ehrenstein effects and illusion indices (with CIs)
  v2/verify/local_nulls_summary.json       local-null Ehrenstein point estimates for the sigmas not in the stats file
  v2/verify/fill_contrast_stats.json       illusion index at 25 / 50 / 75 % fill contrast
  v2/verify/encoding_nulls.json            readout-free encoding vs near-exterior null (supplementary limitation only)
  v2/decompose/results.json                Shapley decompositions of DeepGaze IIE and MSDB
  v2/anatomy/anatomy.json                  parameter counts, channels, blur scales, training-set counts
  v3/scores/results.json                   uniform vs alternating rotation, core measures, support-ratio slopes, nulls
  v3/gpu/full/results_full.json            full-data readout swap (feature set x readout x seed)
  paper/seed_bootstrap.json                two-level (seed) bootstrap check of the readout-swap intervals, written by
                                           paper_numbers/check_seed_bootstrap.py from v3/gpu/full/scores (run it first;
                                           build.sh does)

Writes <out>/numbers.tex (one \\newcommand per number, each preceded by a comment naming file and JSON path),
<out>/tab_readout.tex, tab_survey.tex and tab_sweep.tex (Appendix B), and prints the macro list (name: value: source) to
stdout. <out> = paper_numbers/output, or LAN_PAPER_OUT.

Runs unchanged as more readout-swap seeds land: every readout-swap cell carries its seed count and a provisional flag
(fewer than 3 seeds, or no scores yet); \\ifRSProvisional and \\RSProvisional give the global state.
Python 3.12.
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as LP  # noqa: E402

ROOT = LP.RESULTS
OUT = Path(os.environ.get("LAN_PAPER_OUT") or Path(__file__).resolve().parent / "output")
OUT.mkdir(parents=True, exist_ok=True)
SB = LP.PAPER / "seed_bootstrap.json"
PLANNED_SEEDS = (0, 1, 2)                       # readout_swap/README.md: seeds 0, 1, 2
FSETS = (("dinov2_L12", "DinoTwelve", "DINOv2-B block 12"),
         ("dinov2_L6", "DinoSix", "DINOv2-B block 6"),
         ("siglip_L24", "SiglipTwentyFour", "SigLIP-L block 24"),
         ("iie_shapenetc", "IIEFeat", "IIE ShapeNet-C"))
ARCHS = (("R1n", "RoneFixed"), ("R1", "Rone"), ("R2", "Rtwo"), ("R3", "Rthree"), ("R4", "Rfour"))
SURVEY = (("IIE", "DeepGaze IIE", "DeepGaze_IIE", "DeepGaze IIE"),
          ("MSDB", "DeepGaze MSDB", "DeepGaze_MSDB", "DeepGaze MSDB"),
          ("MSINet", "MSI-Net", "MSI-Net", "MSI-Net"),
          ("XSAspp", "xSoyelNet-ASPP", "xSoyelNet-ASPP", "xSoyelNet-ASPP"),
          ("XSMl", "xSoyelNet-ML", "xSoyelNet-ML", "xSoyelNet-ML"),
          ("SpecRes", "Spectral residual", "Spectral_residual", "Spectral residual"),
          ("EdgeNull", "Edge-energy null", "Edge-energy_null", "Edge-energy null"))
SIX = SURVEY[:6]
NULL_TYPES = (("corner", "Corner"), ("ink", "Ink"), ("edge", "Edge"))
SIGMAS = (8, 16, 25, 40, 60)
WORD = {2: "Two", 4: "Four", 3: "Three", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 10: "Ten", 12: "Twelve", 16: "Sixteen", 20: "Twenty",
        24: "TwentyFour", 25: "TwentyFive", 30: "Thirty", 40: "Forty", 45: "FortyFive", 50: "Fifty", 60: "Sixty", 75: "SeventyFive"}
MEAS = (("r_inside", ""), ("r_core30", "Core"), ("f_core30", "CoreShare"))
SUBSETS = (("all", ""), ("square", "Sq"), ("triangle", "Tri"))
INDUCER_ONLY = ("illusory", "control90", "mis10", "mis20", "mis30", "alt10", "alt20", "alt30")


# ------------------------------------------------------------------ file access with source strings
_CACHE = {}


def J(rel):
    if rel not in _CACHE:
        _CACHE[rel] = LP.load_json(ROOT / rel)
    return _CACHE[rel]


def src(rel, *path):
    return f"{rel} > " + " > ".join(str(p) for p in path)


def get(rel, *path, default=KeyError):
    o = J(rel)
    for p in path:
        try:
            o = o[p]
        except (KeyError, IndexError, TypeError):
            if default is KeyError:
                raise KeyError(src(rel, *path))
            return default
    return o


# ------------------------------------------------------------------ formatting
def fnum(x, d, signed=False):
    """(plain, tex) for one number; signed numbers and negatives go through \\ensuremath so the minus is a minus."""
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    if float(s) == 0 and d < 4 and round(x, 4) != 0 and (signed or x < 0):
        # keep the sign of a value that rounds to zero (an interval bound of -0.0004 must not print as 0.000)
        s = f"{x:+.4f}" if signed else f"{x:.4f}"
    elif float(s) == 0:
        s = f"{0:.{d}f}"
    return s, (r"\ensuremath{%s}" % s if s[0] in "+-" else s)


def fci(ci, d):
    a, b = fnum(ci[0], d), fnum(ci[1], d)
    return f"[{a[0]}, {b[0]}]", f"[{a[1]}, {b[1]}]"


def fint(n):
    s = f"{int(n):,}"
    return s, s


def fp(pstr):
    return pstr, r"\ensuremath{%s}" % pstr


class Pack:
    def __init__(self):
        self.items = []      # (name, tex, plain, source, ci_plain or None)
        self.names = {}
        self.sections = []

    def section(self, title):
        self.items.append(("__section__", title, None, None, None))

    def add(self, name, val, source, ci=None):
        """val = (plain, tex); ci = optional (plain, tex) written as name+'CI'."""
        assert re.fullmatch(r"[A-Za-z]+", name), name
        for n in (name, name + "CI") if ci else (name,):
            assert n not in self.names, f"duplicate macro {n}"
            self.names[n] = True
        self.items.append((name, val[1], val[0], source, ci))

    def text(self, name, s, source):
        t = s.replace("%", r"\%")
        self.add(name, (s, t), source)

    def interval(self, name, rec, source, d=3, signed=True, mean_key="mean"):
        self.add(name, fnum(rec[mean_key], d, signed), source, fci(rec["ci"], d))

    def ratio(self, name, rec, source, d=2):
        self.add(name, fnum(rec["value"], d), source, fci(rec["ci"], d))

    def write(self, path, header):
        L = list(header)
        for name, tex, plain, source, ci in self.items:
            if name == "__section__":
                L += ["", "% " + "=" * 100, f"% {tex}", "% " + "=" * 100]
                continue
            L.append(f"% {source}")
            L.append(r"\newcommand{\%s}{%s}" % (name, tex))
            if ci:
                L.append(f"% {source} (95% CI)")
                L.append(r"\newcommand{\%sCI}{%s}" % (name, ci[1]))
        path.write_text("\n".join(L) + "\n")

    def listing(self):
        out = ["# file codes: " + "; ".join(f"{v} = {k}" for k, v in ABBR)]
        for name, tex, plain, source, ci in self.items:
            if name == "__section__":
                out.append(f"# {tex}")
                continue
            for k, v in ABBR:
                source = source.replace(k, v)
            source = source.replace(" > ", "/")
            if ci:
                out.append(f"\\{name} (+CI): {plain} {ci[0]}: {source}")
            else:
                out.append(f"\\{name}: {plain}: {source}")
        return out


ABBR = (("v2/verify/core_aoi_D30_835295.json", "[core2]"), ("v2/verify/local_nulls_summary.json", "[lnsum]"),
        ("v2/verify/local_nulls_stats.json", "[lns]"), ("v2/verify/fill_contrast_stats.json", "[fill]"),
        ("v2/verify/position_check.json", "[pos]"), ("v2/scores/results.json", "[S2]"), ("v2/scores/raw/", "[S2raw]/"),
        ("v2/decompose/results.json", "[D2]"), ("v2/anatomy/anatomy.json", "[A2]"), ("v3/scores/results.json", "[S3]"),
        ("v3/gpu/full/results_full.json", "[F3]"), ("v2/conventions.md", "[conv]"),
        ("v2/verify/encoding_nulls.json", "[enc]"))
P = Pack()
WARN = []

# ================================================================== 0. design counts
P.section("0. Design counts (stimuli, statistics, training data, readout-swap design)")
RES2 = (ROOT / "v2/conventions.md").read_text()
m = re.search(r"Mean over configurations \((\d+) Kanizsa, (\d+)\s+Ehrenstein\)", RES2)
assert m, "configuration counts not found in v2/conventions.md"
nK, nE = int(m.group(1)), int(m.group(2))
assert nK == get("v2/decompose/results.json", "design", "n_configs")
P.add("NKzConfigs", fint(nK), src("v2/decompose/results.json", "design", "n_configs") + " (= v2/conventions.md)")
P.add("NEhrConfigs", fint(nE), "v2/conventions.md, conventions paragraph: 'Mean over configurations (60 Kanizsa, 40 Ehrenstein)'")
nKc = len(get("v2/scores/results.json", "cells", "kanizsa"))
nEc = len(get("v2/scores/results.json", "cells", "ehrenstein"))
npos = get("v2/decompose/results.json", "design", "positions_per_cell")
assert nKc * npos == nK and nE % nEc == 0
P.add("NKzCells", fint(nKc), src("v2/scores/results.json", "cells", "kanizsa") + " (length: shape x support ratio x polarity)")
P.add("NEhrCells", fint(nEc), src("v2/scores/results.json", "cells", "ehrenstein") + " (length: radius x line count x polarity)")
P.add("NPositions", fint(npos), src("v2/decompose/results.json", "design", "positions_per_cell"))
kz_eff = get("v2/scores/results.json", "models", "DeepGaze IIE", "variants", "native", "kanizsa", "effects")
eh_eff = get("v2/scores/results.json", "models", "DeepGaze IIE", "variants", "native", "ehrenstein", "effects")
kz_conds = {"control180"} | {k.split("-")[0] for k in kz_eff}
eh_conds = {"control_end"} | {k.split("-")[0] for k in eh_eff}
P.add("NKzConditions", fint(len(kz_conds)), "derived: conditions named in " + src("v2/scores/results.json", "models", "DeepGaze IIE", "variants", "native", "kanizsa", "effects") + " plus control180")
P.add("NEhrConditions", fint(len(eh_conds)), "derived: conditions named in ... ehrenstein > effects plus control_end")
P.add("NKzImages", fint(len(kz_conds) * nK), "derived: NKzConditions x NKzConfigs")
P.add("NEhrImages", fint(len(eh_conds) * nE), "derived: NEhrConditions x NEhrConfigs")
P.add("NStimImages", fint(len(kz_conds) * nK + len(eh_conds) * nE), "derived: NKzImages + NEhrImages")
phases = get("v3/scores/results.json", "models", "DeepGaze IIE", "results", "r_inside", "all", "ratio_alt_phase")
thetas = sorted({int(re.match(r"(\d+)", k).group(1)) for k in phases})
P.add("NAltImages", fint(len(phases) * nK), "derived: " + src("v3/scores/results.json", "models", "DeepGaze IIE", "results", "r_inside", "all", "ratio_alt_phase") + f" ({len(phases)} angle x phase keys) x NKzConfigs")
P.add("NAltAngles", fint(len(thetas)), "derived: angles in ... ratio_alt_phase keys")
P.text("AltAnglesList", ", ".join(str(t) for t in thetas[:-1]) + f" and {thetas[-1]}", "derived: angles in ... ratio_alt_phase keys (degrees)")
P.add("NBoot", fint(get("v2/scores/results.json", "bootstrap")), src("v2/scores/results.json", "bootstrap"))
assert get("v3/scores/results.json", "bootstrap") == get("v3/gpu/full/results_full.json", "bootstrap") == get("v2/scores/results.json", "bootstrap")
nulls = [k for k, v in get("v3/scores/results.json", "models").items() if v["kind"] == "local null"]
P.add("NLocalNulls", fint(len(nulls)), "derived: count of models with kind 'local null' in " + src("v3/scores/results.json", "models"))
P.text("LocalNullSigmas", ", ".join(str(s) for s in SIGMAS[:-1]) + f" and {SIGMAS[-1]}", "derived: sigma suffixes of the local-null names in v3/scores/results.json > models")
assert sorted(nulls) == sorted(f"null {t}_s{s}" for t, _ in NULL_TYPES for s in SIGMAS)
P.add("NSurveyModels", fint(len(SIX)), "derived: the six saliency models of " + src("v2/scores/results.json", "models") + " (edge-energy null excluded)")
# training data (sweep / full-data recipe)
sweep_rec = [r for r in get("v2/anatomy/anatomy.json", "records") if r["model"] == "sweep dinov2 L12"][0]
idx_sweep = get("v2/anatomy/anatomy.json", "records").index(sweep_rec)
ftd = sweep_rec["fixation_training_data"]
m = re.match(r"([\d,]+) images: SALICON train ([\d,]+) .*MIT1003 sorted first (\d+) \(last (\d+) held out\), CAT2000 train ([\d,]+)", ftd)
assert m, ftd
tot, sal, mit, held, cat = (int(g.replace(",", "")) for g in m.groups())
assert tot == sal + mit + cat
s_ftd = src("v2/anatomy/anatomy.json", "records", idx_sweep, "fixation_training_data")
P.add("NTrainImages", fint(tot), s_ftd)
P.add("NTrainSalicon", fint(sal), s_ftd + " (SALICON = mouse tracking)")
P.add("NTrainMIT", fint(mit), s_ftd)
P.add("NTrainCAT", fint(cat), s_ftd)
P.add("NMITHeldout", fint(held), s_ftd)
P.add("NTrainEpochs", fint(sweep_rec["training"]["epochs"]), src("v2/anatomy/anatomy.json", "records", idx_sweep, "training", "epochs") + " (full-data recipe = sweep recipe, readout_swap/README.md)")
P.add("NTrainBatch", fint(sweep_rec["training"]["batch"]), src("v2/anatomy/anatomy.json", "records", idx_sweep, "training", "batch"))
P.add("ValidityMargin", fnum(get("v3/gpu/full/results_full.json", "cc_margin"), 2), src("v3/gpu/full/results_full.json", "cc_margin"))
P.add("RSNFeatureSets", fint(len(FSETS)), "design: readout_swap/README.md (4 feature sets); keys of v3/gpu/full/results_full.json > mit")
P.add("RSNReadouts", fint(len(ARCHS)), "design: readout_swap/README.md (R1n, R1, R2, R3, R4)")
P.add("RSNSeedsPlanned", fint(len(PLANNED_SEEDS)), "design: readout_swap/README.md (seeds 0, 1, 2)")
P.add("RSNModelsPlanned", fint(len(FSETS) * len(ARCHS) * len(PLANNED_SEEDS)), "derived: RSNFeatureSets x RSNReadouts x RSNSeedsPlanned")
assert set(get("v3/gpu/full/results_full.json", "mit")) <= {f for f, _, _ in FSETS}

# ================================================================== 1. survey (v2/scores)
P.section("1. Survey: six saliency models and the edge-energy null (v2/scores/results.json; position and core from v2/verify)")
SV = "v2/scores/results.json"
pos = J("v2/verify/position_check.json")
core2 = "v2/verify/core_aoi_D30_835295.json"
survey = {}
n_pos_both = n_pos_all = n_sign_const = 0
raw_note = "EXTRA SOURCE v2/scores/raw/{tag}.json (per-image r_inside of the run summarised by v2/scores/results.json): mean over the {n} {pol} configurations of r(illusory) - r(control180)"
for key, name, tag, label in SURVEY:
    base = ("models", name, "variants", "native")
    k = get(SV, *base, "kanizsa")
    e = get(SV, *base, "ehrenstein")
    rec = dict(label=label)
    for mac, test in (("Kz", "illusory-control180"), ("KzNinety", "illusory-control90"), ("RealFilled", "real_filled-control180"),
                      ("Real", "real-control180")):
        r = k["effects"][test]["inside"]
        P.interval(f"Svy{key}{mac}", r, src(SV, *base, "kanizsa", "effects", test, "inside"))
        rec[mac] = r
    P.add(f"Svy{key}KzP", fp(k["effects"]["illusory-control180"]["inside"]["p_str"]), src(SV, *base, "kanizsa", "effects", "illusory-control180", "inside", "p_str"))
    r = e["effects"]["illusory-control_end"]["inside"]
    P.interval(f"Svy{key}Ehr", r, src(SV, *base, "ehrenstein", "effects", "illusory-control_end", "inside"))
    rec["Ehr"] = r
    for ill, mac in (("kanizsa", "KzIndex"), ("ehrenstein", "EhrIndex")):
        ix = get(SV, *base, ill, "index", "inside", "real_filled")
        if ix.get("computed"):
            P.ratio(f"Svy{key}{mac}", ix, src(SV, *base, ill, "index", "inside", "real_filled"))
            rec[mac] = ix
    sb = k["section_B"]["inside"]
    if sb["eligible"]:
        for t in (10, 20, 30):
            P.ratio(f"Svy{key}Uni{WORD[t]}", sb[f"ratio_{t}"], src(SV, *base, "kanizsa", "section_B", "inside", f"ratio_{t}"))
        for aoi, mac in (("inside_eroded", "Eroded"), ("inside_eroded_all", "ErodedAll")):
            P.ratio(f"Svy{key}UniTwenty{mac}", k["section_B"][aoi]["ratio_20"], src(SV, *base, "kanizsa", "section_B", aoi, "ratio_20"))
        rec["Uni20"] = sb["ratio_20"]
    # position
    pv = pos[tag]
    P.add(f"Svy{key}PosMin", fnum(min(pv), 3, True), f"derived: min over the {len(pv)} positions of v2/verify/position_check.json > {tag}")
    P.add(f"Svy{key}PosMax", fnum(max(pv), 3, True), f"derived: max over the {len(pv)} positions of v2/verify/position_check.json > {tag}")
    # polarity (extra source)
    raw = LP.load_json(ROOT / "v2/scores/raw" / f"{tag}.json")
    pol = {}
    for p_ in ("dark", "light"):
        cfg = sorted({kk.rsplit("_", 1)[0] for kk in raw if kk.endswith("_illusory") and f"_{p_}_" in kk and kk.startswith(("square", "triangle"))})
        vals = [raw[f"{c}_illusory"]["r_inside"] - raw[f"{c}_control180"]["r_inside"] for c in cfg]
        assert len(vals) == nK // 2
        pol[p_] = sum(vals) / len(vals)
        P.add(f"Svy{key}Kz{p_.capitalize()}", fnum(pol[p_], 3, True), raw_note.format(tag=tag, n=len(vals), pol=p_))
    allv = list(pv) + list(pol.values())
    const = all(v > 0 for v in allv) or all(v < 0 for v in allv)
    rec["sign_const"] = const
    if (key, name) in [(s[0], s[1]) for s in SIX]:
        n_sign_const += const
        if rec["Kz"]["ci"][0] > 0 and rec["KzNinety"]["ci"][0] > 0:
            n_pos_both += 1
            if all(v > 0 for v in allv):
                n_pos_all += 1
    # core (survey models not scored in v3)
    if tag in J(core2) and key not in ("IIE", "MSDB"):
        P.interval(f"Svy{key}KzCore", get(core2, tag, "illusory:r_core"), src(core2, tag, "illusory:r_core") + " (core r, D 30 px)")
        P.interval(f"Svy{key}KzCoreShare", get(core2, tag, "illusory:core"), src(core2, tag, "illusory:core") + " (core share of map mass, D 30 px)")
        rec["Core"] = get(core2, tag, "illusory:r_core")
    survey[key] = rec
P.add("SvyNPositiveBothControls", fint(n_pos_both), "derived: survey models (of NSurveyModels) whose Kanizsa CI lies above 0 against control180 and control90 (v2/scores/results.json)")
P.add("SvyNPositiveEverywhere", fint(n_pos_all), "derived: of those, models positive at every position (v2/verify/position_check.json) and polarity (v2/scores/raw)")
P.add("SvyNSignConstant", fint(n_sign_const), "derived: survey models whose Kanizsa effect keeps one sign over all positions and both polarities")
# eroded AOIs
mx = 0.0
for nm, recm in get(SV, "models").items():
    v = recm["variants"]; var = "native" if "native" in v else "avg"
    for ill, tests in (("kanizsa", ("illusory-control180", "illusory-control90")), ("ehrenstein", ("illusory-control_end",))):
        for t in tests:
            ee = v[var][ill]["effects"][t]
            for a in ("inside_eroded", "inside_eroded_all"):
                mx = max(mx, abs(ee[a]["mean"] - ee["inside"]["mean"]))
P.add("SvyMaxErodedChange", fnum(mx, 3), "derived: max over every model in v2/scores/results.json > models (size avg for layer readouts), Kanizsa vs control180 and control90 and Ehrenstein, of |effect(inside_eroded or inside_eroded_all) - effect(inside)|")
# largest model effect (for the corner-null comparison): survey + v2 layer readouts + full-data readouts
model_max = max(((r["Kz"]["mean"], k_) for k_, r in survey.items() if k_ != "EdgeNull"))
lay = [(get(SV, "models", nm, "variants", "avg", "kanizsa", "effects", "illusory-control180", "inside", "mean"), nm)
       for nm, recm in get(SV, "models").items() if recm["kind"] == "layer readout"]
full = J("v3/gpu/full/results_full.json")
fr = [(rr["kanizsa"]["illusory-control180"]["avg|inside"]["mean"], f"{fs} {ar}") for fs, d in full["stim_v3"].items()
      for ar, rr in d.items() if "kanizsa" in rr]
P.add("SvyMaxModelKz", fnum(model_max[0], 3, True), f"derived: largest Kanizsa effect among the six survey models ({model_max[1]})")
allmax = max(lay + fr + [model_max])
P.add("MaxAnyModelKz", fnum(allmax[0], 3, True), f"derived: largest Kanizsa effect among survey models, v2 layer readouts and full-data readouts ({allmax[1]})")

# ================================================================== 2. DeepGaze IIE / MSDB under rotation, core, support ratio (v3)
P.section("2. DeepGaze IIE and MSDB: core measures, uniform vs alternating rotation, support ratio (v3/scores/results.json)")
V3 = "v3/scores/results.json"
v3m = get(V3, "models")
for key, name in (("IIE", "DeepGaze IIE"), ("MSDB", "DeepGaze MSDB")):
    # consistency with v2
    a2 = survey[key]["Kz"]["mean"]; a3 = v3m[name]["results"]["r_inside"]["all"]["aligned"]["mean"]
    assert abs(a2 - a3) < 1e-3, (key, a2, a3)
    assert abs(survey[key]["Uni20"]["value"] - v3m[name]["results"]["r_inside"]["all"]["ratio_uni"]["20"]["value"]) < 1e-3
    for meas, mm in MEAS:
        for sub, ss in SUBSETS:
            o = v3m[name]["results"][meas][sub]
            sp = ("models", name, "results", meas, sub)
            if mm or ss:
                P.interval(f"Alt{key}{mm}{ss}Aligned", o["aligned"], src(V3, *sp, "aligned"))
            if sub == "all" and mm:
                P.interval(f"Svy{key}Kz{mm}", o["aligned"], src(V3, *sp, "aligned"))
                P.interval(f"Svy{key}Kz{mm}Ninety", o["effects_vs_control90"]["illusory"], src(V3, *sp, "effects_vs_control90", "illusory"))
            if o["ratios_computed"]:
                for t in (20, 30):
                    P.ratio(f"Alt{key}{mm}{ss}Uni{WORD[t]}", o["ratio_uni"][str(t)], src(V3, *sp, "ratio_uni", str(t)))
                    P.ratio(f"Alt{key}{mm}{ss}Alt{WORD[t]}", o["ratio_alt"][str(t)], src(V3, *sp, "ratio_alt", str(t)))
            for t in (20, 30):
                dr = o["drop_ratio"][str(t)]
                if dr is not None:
                    P.ratio(f"Alt{key}{mm}{ss}Drop{WORD[t]}", dr, src(V3, *sp, "drop_ratio", str(t)))
            if sub == "all":
                for t in (20, 30):
                    P.interval(f"Alt{key}{mm}LossUni{WORD[t]}", o["drop_uni"][str(t)], src(V3, *sp, "drop_uni", str(t)) + " (m(illusory) - m(mis); positive = figure loses mass)")
                    P.interval(f"Alt{key}{mm}LossAlt{WORD[t]}", o["drop_alt"][str(t)], src(V3, *sp, "drop_alt", str(t)) + " (m(illusory) - m(alt))")
                sl = o["sr_slope_vs_control180"]
                P.add(f"Svy{key}SRSlope{mm}", fnum(sl["slope_per_unit_sr"], 3, True), src(V3, *sp, "sr_slope_vs_control180", "slope_per_unit_sr") + " (per unit support ratio)",
                      fci(sl["ci"], 3))
                if not mm:
                    for srk, srv in sl["mean_by_sr"].items():
                        P.add(f"Svy{key}KzSR{WORD[int(round(float(srk) * 10))]}", fnum(srv, 3, True), src(V3, *sp, "sr_slope_vs_control180", "mean_by_sr", srk) + f" (aligned effect at support ratio {srk})")
                    survey[key]["Slope"] = sl
                if mm == "Core":
                    survey[key]["Core"] = o["aligned"]
                if not mm:
                    survey[key]["Alt20"] = o["ratio_alt"]["20"]

# ================================================================== 3. local nulls (v3 + v2/verify)
P.section("3. Local nulls: blurred corner, ink and edge-energy maps (v3/scores/results.json; Ehrenstein and indices from v2/verify)")
LNS, LNSUM = "v2/verify/local_nulls_stats.json", "v2/verify/local_nulls_summary.json"
iie_core = survey["IIE"]["Core"]["mean"]; iie_cshare = v3m["DeepGaze IIE"]["results"]["f_core30"]["all"]["aligned"]["mean"]
nullrec = {}
for t, T in NULL_TYPES:
    for s in SIGMAS:
        nm = f"null {t}_s{s}"; key = f"Null{T}{WORD[s]}"; R = v3m[nm]["results"]
        rec = {}
        for meas, mm in MEAS:
            o = R[meas]["all"]; sp = ("models", nm, "results", meas, "all")
            P.interval(f"{key}Kz{mm}", o["aligned"], src(V3, *sp, "aligned"))
            rec["Kz" + mm] = o["aligned"]
            if not mm:
                P.interval(f"{key}KzNinety", o["effects_vs_control90"]["illusory"], src(V3, *sp, "effects_vs_control90", "illusory"))
                rec["KzNinety"] = o["effects_vs_control90"]["illusory"]
            if o["ratios_computed"] and meas != "f_core30":
                for tt in (20,):
                    P.ratio(f"{key}{mm}Uni{WORD[tt]}", o["ratio_uni"][str(tt)], src(V3, *sp, "ratio_uni", str(tt)))
                    P.ratio(f"{key}{mm}Alt{WORD[tt]}", o["ratio_alt"][str(tt)], src(V3, *sp, "ratio_alt", str(tt)))
                if not mm:
                    rec["Uni20"], rec["Alt20"] = o["ratio_uni"]["20"], o["ratio_alt"]["20"]
            dr = o["drop_ratio"]["20"]
            if dr is not None and meas != "f_core30":
                P.ratio(f"{key}{mm}DropTwenty", dr, src(V3, *sp, "drop_ratio", "20"))
            if not mm:
                P.interval(f"{key}LossUniTwenty", o["drop_uni"]["20"], src(V3, *sp, "drop_uni", "20") + " (positive = figure loses mass)")
                P.interval(f"{key}LossAltTwenty", o["drop_alt"]["20"], src(V3, *sp, "drop_alt", "20"))
                rec["LossUni20"], rec["LossAlt20"] = o["drop_uni"]["20"], o["drop_alt"]["20"]
            if meas in ("r_inside", "r_core30"):
                sl = o["sr_slope_vs_control180"]
                P.add(f"{key}SRSlope{mm}", fnum(sl["slope_per_unit_sr"], 3, True), src(V3, *sp, "sr_slope_vs_control180", "slope_per_unit_sr"), fci(sl["ci"], 3))
                rec["Slope" + mm] = sl
            if meas in ("r_core30", "f_core30"):
                eff = o["effects_vs_control180"]
                best = max(INDUCER_ONLY, key=lambda c: abs(eff[c]["mean"]))
                P.add(f"{key}{mm}MaxInducerOnly", fnum(eff[best]["mean"], 3, True),
                      f"derived: largest |effect| over inducer-only conditions {', '.join(INDUCER_ONLY)} in {src(V3, *sp, 'effects_vs_control180')} (here {best})")
                rec["Max" + mm] = eff[best]["mean"]
        # Ehrenstein and index
        nk = f"{t}_s{s}"
        if nk in J(LNS):
            P.interval(f"{key}Ehr", get(LNS, nk, "E"), src(LNS, nk, "E"))
            P.ratio(f"{key}KzIndex", get(LNS, nk, "K_index"), src(LNS, nk, "K_index"))
        else:
            P.add(f"{key}Ehr", fnum(get(LNSUM, nk, "E_ill"), 3, True), src(LNSUM, nk, "E_ill") + " (point estimate; no CI in the verify files)")
        nullrec[(t, s)] = rec
# derived comparisons
corner_kz = [nullrec[("corner", s)]["Kz"]["mean"] for s in SIGMAS]
P.add("NullCornerKzMin", fnum(min(corner_kz), 3, True), "derived: min over sigma of NullCorner<sigma>Kz")
P.add("NullCornerKzMax", fnum(max(corner_kz), 3, True), "derived: max over sigma of NullCorner<sigma>Kz")
P.add("NullCornerMinOverIIE", fnum(min(corner_kz) / survey["IIE"]["Kz"]["mean"], 1), "derived: NullCornerKzMin / SvyIIEKz (times)")
for s in (16, 25):
    rc = nullrec[("corner", s)]
    P.text(f"NullCorner{WORD[s]}CoreIllPctIIE", f"{100 * rc['KzCore']['mean'] / iie_core:.0f}%", f"derived: NullCorner{WORD[s]}KzCore / SvyIIEKzCore (illusory - control180, core r, D 30)")
    P.text(f"NullCorner{WORD[s]}CoreMaxPctIIE", f"{100 * rc['MaxCore'] / iie_core:.0f}%", f"derived: NullCorner{WORD[s]}CoreMaxInducerOnly / SvyIIEKzCore")
    P.text(f"NullCorner{WORD[s]}CoreShareMaxPctIIE", f"{100 * rc['MaxCoreShare'] / iie_cshare:.0f}%", f"derived: NullCorner{WORD[s]}CoreShareMaxInducerOnly / SvyIIEKzCoreShare")
# ratios retained by positive-effect (corner) nulls; minimum pooled ratio over all nulls
ret = []
for s in SIGMAS:
    o = v3m[f"null corner_s{s}"]["results"]["r_inside"]["all"]
    if o["ratios_computed"]:
        for tt in ("10", "20", "30"):
            ret += [o["ratio_uni"][tt]["value"], o["ratio_alt"][tt]["value"]]
P.add("NullCornerMinRetained", fnum(min(ret), 2), "derived: min over corner nulls, angles 10/20/30 and both rotations of ratio_uni / ratio_alt (v3/scores/results.json > models > null corner_s* > results > r_inside > all)")
for meas, mm in (("r_inside", ""), ("r_core30", "Core")):
    vals = [(v3m[n]["results"][meas]["all"]["ratio_uni"]["20"]["value"], n) for n in nulls if v3m[n]["results"][meas]["all"]["ratios_computed"]]
    lo = min(vals)
    P.add(f"NullMin{mm}UniTwenty", fnum(lo[0], 2), f"derived: min over local nulls with ratios computed of ratio_uni 20 on {meas} ({lo[1]})")
lossu = [(nullrec[k_]["LossUni20"]["mean"], k_) for k_ in nullrec] + [(nullrec[k_]["LossAlt20"]["mean"], k_) for k_ in nullrec]
P.add("NullMaxLossTwenty", fnum(max(lossu)[0], 3, True), f"derived: max over local nulls and both rotations at 20 deg of m(illusory) - m(rotated) on r_inside ({max(lossu)[1]}); <= 0 means no null loses figure mass")
# drop-ratio expectation of the local nulls (as in v3/scores/table.md)
for meas, mm in MEAS:
    for sub, ss in SUBSETS:
        for tt in ("20", "30"):
            v = [v3m[n]["results"][meas][sub]["drop_ratio"][tt]["value"] for n in nulls if v3m[n]["results"][meas][sub]["drop_ratio"][tt] is not None]
            if v:
                sname = f"NullDrop{mm}{ss or 'All'}{WORD[int(tt)]}"
                P.add(sname + "Min", fnum(min(v), 2), f"derived: min over the {len(v)} local nulls with a uniform drop CI excluding 0 of drop_ratio {tt}, {meas}, {sub} (v3/scores/results.json)")
                P.add(sname + "Max", fnum(max(v), 2), f"derived: max over the same {len(v)} local nulls")
cs = [nullrec[("corner", s)]["SlopeCore"]["slope_per_unit_sr"] for s in SIGMAS if s >= 16]
P.add("NullCornerCoreSlopeLeast", fnum(max(cs), 2, True), "derived: least negative core-r SR slope over corner nulls sigma 16 to 60")
P.add("NullCornerCoreSlopeMost", fnum(min(cs), 2, True), "derived: most negative core-r SR slope over corner nulls sigma 16 to 60")
# negative-effect (ink and edge-energy) nulls: rotation profile, for comparison with the negative DINOv2 block-12 readouts
neg = [n for n in nulls if not n.startswith("null corner")]
assert all(v3m[n]["results"]["r_inside"]["all"]["aligned"]["ci"][1] < 0 for n in neg)
assert all(v3m[f"null corner_s{s}"]["results"]["r_inside"]["all"]["aligned"]["ci"][0] > 0 for s in SIGMAS)
nu = [v3m[n]["results"]["r_inside"]["all"]["ratio_uni"]["20"]["value"] for n in neg]
nd = [v3m[n]["results"]["r_inside"]["all"]["drop_ratio"]["20"]["value"] for n in neg if v3m[n]["results"]["r_inside"]["all"]["drop_ratio"]["20"] is not None]
P.add("NullNegUniTwentyMin", fnum(min(nu), 2), f"derived: min over the {len(nu)} ink and edge-energy nulls (all with aligned CI below 0) of ratio_uni 20, r_inside, all (v3/scores/results.json)")
P.add("NullNegUniTwentyMax", fnum(max(nu), 2), "derived: max over the same nulls")
P.add("NullNegDropTwentyMin", fnum(min(nd), 2), f"derived: min over the {len(nd)} ink and edge-energy nulls with a uniform drop CI excluding 0 of drop_ratio 20, r_inside, all")
P.add("NullNegDropTwentyMax", fnum(max(nd), 2), "derived: max over the same nulls")
P.add("NNullNeg", fint(len(neg)), "derived: number of local nulls with a negative aligned effect (ink and edge-energy)")
P.add("NNullCorner", fint(len(SIGMAS)), "derived: number of corner nulls (all with a positive aligned effect)")
# pooled over shapes: does any null lose figure mass with an interval above 0?
n_pool_pos = sum(1 for n in nulls for key in ("drop_uni", "drop_alt") for t in ("10", "20", "30")
                 if v3m[n]["results"]["r_inside"]["all"][key][t]["ci"][0] > 0)
P.add("NullNPooledLossAboveZero", fint(n_pool_pos), "derived: count of (null, rotation, angle 10/20/30) whose pooled figure-mass loss CI lies above 0 (v3/scores/results.json > r_inside > all > drop_uni / drop_alt)")
# triangles: corner nulls with large blur lose a little figure mass
tri = [(v3m[n]["results"]["r_inside"]["triangle"][key]["30"]["mean"], n, key) for n in nulls for key in ("drop_uni", "drop_alt")]
tmax = max(tri)
P.add("NullTriMaxLossThirty", fnum(tmax[0], 3), f"derived: max over local nulls and both rotations of the triangle figure-mass loss at 30 deg, r_inside ({tmax[1]}, {tmax[2]})")
tsig = sorted({int(n.split("_s")[1]) for v, n, k in tri if n.startswith("null corner") and
               v3m[n]["results"]["r_inside"]["triangle"][k]["30"]["ci"][0] > 0})
P.add("NullTriLossThirtySigmaMin", fint(min(tsig)), "derived: smallest corner-null sigma whose triangle loss at 30 deg has a CI above 0 (either rotation)")
P.interval("NullCornerSixtyTriLossUniTwenty", v3m["null corner_s60"]["results"]["r_inside"]["triangle"]["drop_uni"]["20"],
           src(V3, "models", "null corner_s60", "results", "r_inside", "triangle", "drop_uni", "20"))
P.interval("NullCornerSixtyTriLossAltTwenty", v3m["null corner_s60"]["results"]["r_inside"]["triangle"]["drop_alt"]["20"],
           src(V3, "models", "null corner_s60", "results", "r_inside", "triangle", "drop_alt", "20"))
for name, key in (("DeepGaze IIE", "IIE"), ("DeepGaze MSDB", "MSDB")):
    P.interval(f"Alt{key}TriLossUniThirty", v3m[name]["results"]["r_inside"]["triangle"]["drop_uni"]["30"],
               src(V3, "models", name, "results", "r_inside", "triangle", "drop_uni", "30"))
# the corner null orders the conditions as the human dwell data do (real filled > illusory > control)
P.interval("NullCornerTwentyFiveRealFilled", v3m["null corner_s25"]["results"]["r_inside"]["all"]["effects_vs_control180"]["real_filled"],
           src(V3, "models", "null corner_s25", "results", "r_inside", "all", "effects_vs_control180", "real_filled"))
P.add("NKzCellsPerShape", fint(nKc // 2), "derived: NKzCells / 2 shapes (clusters behind every square-only or triangle-only interval)")

# ================================================================== 4. smoothing scale and anatomy
P.section("4. Model anatomy: parameters, channels, scales, blur (v2/anatomy/anatomy.json)")
AN = "v2/anatomy/anatomy.json"
recs = get(AN, "records")
byname = {r["model"]: i for i, r in enumerate(recs)}
i = byname["DeepGaze IIE"]; bl = recs[i]["readout"]["blur_sigma_display_px"]
P.add("AnatIIEBlurMean", fnum(bl["mean"], 1), src(AN, "records", i, "readout", "blur_sigma_display_px", "mean") + " (display px)")
P.add("AnatIIEBlurMin", fnum(min(bl["per_backbone_mean"]), 0), src(AN, "records", i, "readout", "blur_sigma_display_px", "per_backbone_mean") + " (min, display px)")
P.add("AnatIIEBlurMax", fnum(max(bl["per_backbone_mean"]), 0), src(AN, "records", i, "readout", "blur_sigma_display_px", "per_backbone_mean") + " (max, display px)")
P.add("AnatIIENBackbones", fint(len(recs[i]["backbones"])), src(AN, "records", i, "backbones") + " (length)")
P.add("AnatIIENLayers", fint(sum(len(b["layers_read"]) for b in recs[i]["backbones"])), "derived: sum of len(layers_read) over " + src(AN, "records", i, "backbones"))
P.add("AnatIIEParamsM", fnum(recs[i]["readout"]["n_params"] / 1e6, 2), src(AN, "records", i, "readout", "n_params") + " (millions)")
i = byname["DeepGaze MSDB"]; bs = recs[i]["readout"]["blur_sigma"]
P.add("AnatMSDBBlurPx", fnum(bs["display_px"], 1), src(AN, "records", i, "readout", "blur_sigma", "display_px"))
P.add("AnatMSDBBlurDeg", fnum(bs["dva"], 2), src(AN, "records", i, "readout", "blur_sigma", "dva"))
ppd = bs["display_px"] / bs["dva"]
P.add("AnatMSDBPxPerDeg", fnum(ppd, 0), "derived: display_px / dva of " + src(AN, "records", i, "readout", "blur_sigma") + " (display pixels per degree at which MSDB was run)")
for s in (16, 25):
    P.add(f"NullSigmaDeg{WORD[s]}", fnum(s / ppd, 2), f"derived: {s} px / AnatMSDBPxPerDeg (null blur in degrees at the geometry MSDB was run at)")
ch = recs[i]["channels"]
P.add("AnatMSDBChannels", fint(ch["total"]), src(AN, "records", i, "channels", "total"))
P.add("AnatMSDBChannelsCLIP", fint(ch["per_layer"][0]), src(AN, "records", i, "channels", "per_layer", 0) + " (CLIP RN50x64 layer4.2.conv2)")
P.add("AnatMSDBChannelsDino", fint(ch["per_layer"][1]), src(AN, "records", i, "channels", "per_layer", 1) + " (each DINOv2 block, hs[7] and hs[11])")
assert ch["per_layer"][1] == ch["per_layer"][2] and sum(ch["per_layer"]) == ch["total"]
nsc = sum(len(v) for v in recs[i]["input"]["internal_scales"].values())
P.add("AnatMSDBNScales", fint(nsc), "derived: number of entries in " + src(AN, "records", i, "input", "internal_scales"))
P.add("AnatMSDBParams", fint(recs[i]["readout"]["n_params"]), src(AN, "records", i, "readout", "n_params"))
for nm, key in (("MSI-Net", "MSINet"), ("xSoyelNet-ASPP", "XSAspp"), ("xSoyelNet-ML", "XSMl"), ("sweep dinov2 L12", "SweepDino"), ("sweep siglip L24", "SweepSiglip"), ("sweep clip L24", "SweepClip")):
    i = byname[nm]; npar = recs[i]["readout"]["n_params"]
    P.add(f"Anat{key}ParamsM", fnum(npar / 1e6, 2), src(AN, "records", i, "readout", "n_params") + " (millions)")
    P.add(f"Anat{key}Params", fint(npar), src(AN, "records", i, "readout", "n_params"))
i = byname["xSoyelNet-ML"]
P.add("AnatXSMlCC", fnum(recs[i]["heldout_mit1003_cc"]["single_336"], 3), src(AN, "records", i, "heldout_mit1003_cc", "single_336"))
for nm, key in (("sweep dinov2 L12", "SweepDinoTwelveCC"), ("sweep dinov2 L6", "SweepDinoSixCC"), ("sweep siglip L24", "SweepSiglipTwentyFourCC"), ("ab/readout_A", "ReadoutACC")):
    i = byname[nm]
    P.add(f"Anat{key}", fnum(recs[i]["heldout_mit1003_cc"], 3), src(AN, "records", i, "heldout_mit1003_cc"))

# ================================================================== 5. decomposition (v2/decompose)
P.section("5. Shapley decomposition of the Kanizsa effect (v2/decompose/results.json); DINOv2 blocks in 1-based numbering")
DC = "v2/decompose/results.json"
g = ("games", "IIE_backbones", "aoi", "inside")
short = {"ShapeNetC ResNet-50": "ShapeNetC", "EfficientNet-B5": "EffNet", "DenseNet-201": "DenseNet", "ResNeXt-50": "ResNeXt",
         "CLIP RN50x64 layer4.2.conv2": "CLIP", "DINOv2 block 10": "DinoEleven", "DINOv2 block 6": "DinoSeven",
         "CLIP RN50x64": "CLIP", "DINOv2 (blocks 6 + 10)": "Dino",
         "coarse (ppd 5, 10; size 128, 256)": "Coarse", "native (ppd 17.5; size 512)": "Native", "fine (ppd 24, 30; size 768, 1024)": "Fine"}
P.add("DecIIENPlayers", fint(get(DC, "games", "IIE_backbones", "n_players")), src(DC, "games", "IIE_backbones", "n_players"))
P.add("DecIIENCoalitions", fint(get(DC, "games", "IIE_backbones", "n_coalitions")), src(DC, "games", "IIE_backbones", "n_coalitions"))
for pl, v in get(DC, *g, "players").items():
    P.add(f"DecIIEShare{short[pl]}", fnum(v["share"], 2), src(DC, *g, "players", pl, "share"), fci(v["share_ci"], 2))
alone = get(DC, "iie_path_alone_effects", "inside")
for pl, v in alone.items():
    P.interval(f"DecIIEAlone{short[pl]}", v, src(DC, "iie_path_alone_effects", "inside", pl))
P.add("DecIIEAloneMin", fnum(min(v["mean"] for v in alone.values()), 3, True), "derived: min over backbones of " + src(DC, "iie_path_alone_effects", "inside", "*", "mean"))
P.add("DecIIEAloneMax", fnum(max(v["mean"] for v in alone.values()), 3, True), "derived: max over backbones of the same")
P.add("DecIIESumInsertAlone", fnum(get(DC, *g, "sum_insert_alone_share"), 2), src(DC, *g, "sum_insert_alone_share") + " (x effect)")
P.add("DecIIESumRemoveAlone", fnum(get(DC, *g, "sum_remove_alone_share"), 2), src(DC, *g, "sum_remove_alone_share") + " (x effect)")
maxshare = max(v["share"] for v in get(DC, *g, "players").values())
P.add("DecIIEMaxShare", fnum(maxshare, 2), "derived: largest IIE backbone share (carries rule: share >= 0.5)")
for game, pre in (("MSDB_feature_tensors", "DecMSDBTensor"), ("MSDB_backbones", "DecMSDB"), ("MSDB_scale_groups", "DecMSDBScale")):
    gg = ("games", game, "aoi", "inside")
    for pl, v in get(DC, *gg, "players").items():
        P.add(f"{pre}Share{short[pl]}", fnum(v["share"], 2), src(DC, *gg, "players", pl, "share"), fci(v["share_ci"], 2))
    if game == "MSDB_backbones":
        for sub, ss in (("shape=square", "Sq"), ("shape=triangle", "Tri")):
            bsub = get(DC, *gg, "by_subset_point_estimates", sub)
            for pl, v in bsub["shares"].items():
                P.add(f"{pre}{ss}Share{short[pl]}", fnum(v, 2), src(DC, *gg, "by_subset_point_estimates", sub, "shares", pl), fci(bsub["share_cis"][pl], 2))
    if game == "MSDB_scale_groups":
        for pl, v in get(DC, *gg, "scale_weight_and_share_per_weight").items():
            P.add(f"{pre}Weight{short[pl]}", fnum(v["weight"], 2), src(DC, *gg, "scale_weight_and_share_per_weight", pl, "weight"))
            P.add(f"{pre}PerWeight{short[pl]}", fnum(v["share_per_weight"], 2), src(DC, *gg, "scale_weight_and_share_per_weight", pl, "share_per_weight"))
gg = ("games", "MSDB_scales10", "aoi", "inside")
pl10 = get(DC, *gg, "players")
top = max(pl10, key=lambda k_: pl10[k_]["share"])
P.add("DecMSDBLargestSingleScale", fnum(pl10[top]["share"], 2), src(DC, *gg, "players", top, "share") + f" (largest of 10 scales: {top})", fci(pl10[top]["share_ci"], 2))
P.text("DecMSDBLargestSingleScaleName", top.replace("size", "size ").replace("ppd", "ppd "), src(DC, *gg, "players") + " (argmax share)")

# ================================================================== 6. human comparison and Ehrenstein
P.section("6. Human comparison (descriptive only) and Ehrenstein")
P.add("HumKzIndex", fnum(get(SV, "human_index", "kanizsa"), 3), src(SV, "human_index", "kanizsa") + " (Kavsek pooled)")
P.add("HumEhrIndex", fnum(get(SV, "human_index", "ehrenstein"), 3), src(SV, "human_index", "ehrenstein") + " (Kavsek pooled)")
m = re.search(r"dominated by children \((\d+) of (\d+)\)", RES2)
assert m
P.add("HumNChildren", fint(int(m.group(1))), "v2/conventions.md, human reference: 'dominated by children (97 of 127)'")
P.add("HumNTotal", fint(int(m.group(2))), "v2/conventions.md, human reference: 'dominated by children (97 of 127)'")
FC = "v2/verify/fill_contrast_stats.json"
for who, key in (("IIE", "IIE"), ("corner_s25", "NullCornerTwentyFive")):
    for c in (25, 50, 75):
        P.ratio(f"Hum{key}IndexFill{WORD[c]}", get(FC, f"{who}_c{c}", "index"), src(FC, f"{who}_c{c}", "index") + f" (figure grey at {c}% of the inducer contrast)")
eb = ("models", "DeepGaze MSDB", "variants", "native", "ehrenstein", "mass", "illusory-control_end")
P.interval("EhrMSDBDiscLogMass", get(SV, *eb, "dlog_inside"), src(SV, *eb, "dlog_inside") + " (log map mass on the illusory disc)")
P.interval("EhrMSDBLineLogMass", get(SV, *eb, "dlog_ind"), src(SV, *eb, "dlog_ind") + " (log map mass on the inducing lines)")

# ================================================================== 6b. encoding (supplementary, limitation only)
P.section("6b. Readout-free encoding against a proximity-matched near-exterior null (supplementary; v2/verify/encoding_nulls.json)")
EN = "v2/verify/encoding_nulls.json"
for net, key in (("dinov2", "Dino"), ("clip", "Clip"), ("siglip", "Siglip")):
    lay = get(EN, net, "layers")
    ok = [int(Lk) for Lk, v in lay.items() if v["int"]["ci"][0] > v["ext_near"]["ci"][1]]
    P.add(f"Enc{key}NLayersPass", fint(len(ok)), f"derived: layers of {src(EN, net, 'layers')} whose interior CI lower bound exceeds the near-exterior CI upper bound")
    P.add(f"Enc{key}NLayers", fint(len(lay)), src(EN, net, "layers") + " (count)")
    okl = [str(x) for x in sorted(ok)]
    P.text(f"Enc{key}LayersPass", (", ".join(okl[:-1]) + " and " + okl[-1]) if len(okl) > 1 else (okl[0] if okl else "none"), f"derived: the passing layers (1-based hs index) in {src(EN, net, 'layers')}")
P.interval("EncClipTwentyFourInterior", get(EN, "clip", "layers", "24", "int"), src(EN, "clip", "layers", "24", "int"), signed=False)
P.interval("EncClipTwentyFourNearExterior", get(EN, "clip", "layers", "24", "ext_near"), src(EN, "clip", "layers", "24", "ext_near"), signed=False)

# ================================================================== 7. readout swap (v3/gpu/full)
P.section("7. Full-data readout swap: 4 feature sets x 5 readouts x 3 seeds (v3/gpu/full/results_full.json)")
FL = "v3/gpu/full/results_full.json"
P.text("RSResultsCreated", full["created"], src(FL, "created"))
cells = {}
n_models_done = 0
for fs, F, flabel in FSETS:
    mit = full["mit"].get(fs, {})
    r4 = mit.get("R4")
    thr = (r4["mean"] - full["cc_margin"]) if r4 else None
    if thr is not None:
        P.add(f"RS{F}CCThreshold", fnum(thr, 3), f"derived: {src(FL, 'mit', fs, 'R4', 'mean')} - cc_margin")
    for ar, A in ARCHS:
        nm = f"RS{F}{A}"
        c = dict(fs=fs, ar=ar, label=flabel)
        mrec = mit.get(ar)
        srec = full["stim_v3"].get(fs, {}).get(ar)
        n_cc = mrec["n"] if mrec else 0
        ok_stim = bool(srec) and "kanizsa" in srec
        n_st = len(srec["seeds_present"]) if ok_stim else 0
        n_models_done += n_st
        c["n_cc"], c["n"] = n_cc, n_st
        prov = (n_st < len(PLANNED_SEEDS)) or (n_cc < len(PLANNED_SEEDS))
        c["prov"] = prov
        P.add(f"{nm}Seeds", fint(n_st), (src(FL, "stim_v3", fs, ar, "seeds_present") + " (length)") if ok_stim else f"{FL} > stim_v3 > {fs} > {ar}: absent (no scores yet)")
        P.text(f"{nm}Provisional", "yes" if prov else "no", f"derived: fewer than {len(PLANNED_SEEDS)} seeds in {FL} > mit/stim_v3 > {fs} > {ar}")
        if mrec:
            P.add(f"{nm}CC", fnum(mrec["mean"], 3), src(FL, "mit", fs, ar, "mean") + f" (n {n_cc})")
            if mrec.get("valid") is not None:
                valid = mrec["valid"]; vsrc = src(FL, "mit", fs, ar, "valid")
            elif thr is not None:
                valid = mrec["mean"] >= thr; vsrc = f"derived: {src(FL, 'mit', fs, ar, 'mean')} >= RS{F}CCThreshold (provisional: seed counts differ from R4)"
            else:
                valid = None; vsrc = "no R4 CC yet"
            c["valid"] = valid; c["cc"] = mrec["mean"]
            P.text(f"{nm}Valid", {True: "valid", False: "invalid", None: "pending"}[valid], vsrc)
        else:
            c["valid"] = None; c["cc"] = None
            P.text(f"{nm}CC", "pending", f"{FL} > mit > {fs} > {ar}: absent")
            P.text(f"{nm}Valid", "pending", f"{FL} > mit > {fs} > {ar}: absent")
        if ok_stim:
            K = srec["kanizsa"]; E = srec["ehrenstein"]; sp = ("stim_v3", fs, ar)
            for mac, test, var in (("Kz", "illusory-control180", "inside"), ("KzCore", "illusory-control180", "r_core30"),
                                   ("KzNinety", "illusory-control90", "inside"), ("RealFilled", "real_filled-control180", "inside"),
                                   ("Real", "real-control180", "inside")):
                rr = K[test][f"avg|{var}"]
                P.interval(f"{nm}{mac}", rr, src(FL, *sp, "kanizsa", test, f"avg|{var}") + f" (size avg, seeds {rr['n_seeds']})")
                c[mac] = rr
            rr = E["illusory-control_end"]["avg|inside"]
            P.interval(f"{nm}Ehr", rr, src(FL, *sp, "ehrenstein", "illusory-control_end", "avg|inside"))
            sb = full["section_B"].get(fs, {}).get(ar, {}).get("inside", {})
            elig = bool(sb.get("eligible"))
            if elig:
                P.ratio(f"{nm}UniTwenty", sb["ratio_20"], src(FL, "section_B", fs, ar, "inside", "ratio_20"))
            else:
                P.add(f"{nm}UniTwenty", ("n.e.", "n.e."), src(FL, "section_B", fs, ar, "inside", "eligible") + " = false (aligned CI not above 0)", ("", ""))
            sa = full["stim_alt"].get(fs, {}).get(ar, {})
            for q, mac in (("ratio_alt20", "AltTwenty"), ("drop_ratio20", "DropTwenty")):
                qq = sa.get("r_inside", {}).get(q)
                if qq is None:
                    P.add(f"{nm}{mac}", ("pending", "pending"), f"{FL} > stim_alt > {fs} > {ar}: absent or incomplete", ("", ""))
                elif q == "ratio_alt20" and not elig:
                    P.add(f"{nm}{mac}", ("n.e.", "n.e."), "not eligible: aligned CI not above 0", ("", ""))
                elif not qq["den_ci_excludes_0"]:
                    P.add(f"{nm}{mac}", ("n.e.", "n.e."), src(FL, "stim_alt", fs, ar, "r_inside", q, "den_ci_excludes_0") + " = false", ("", ""))
                else:
                    P.ratio(f"{nm}{mac}", qq, src(FL, "stim_alt", fs, ar, "r_inside", q) + f" (seeds {sa['seeds']})")
            # rotation profile for every cell whose aligned interval excludes 0 on either side (the negative DINOv2
            # block-12 cells included): uniform and alternating ratio(20) and drop ratio(20) from the stim_alt set
            kz = K["illusory-control180"]["avg|inside"]
            excl = kz["ci"][0] > 0 or kz["ci"][1] < 0
            for q, mac in (("ratio_uni20", "RotUni"), ("ratio_alt20", "RotAlt"), ("drop_ratio20", "RotDrop")):
                qq = sa.get("r_inside", {}).get(q)
                if qq is None:
                    P.add(f"{nm}{mac}", ("pending", "pending"), f"{FL} > stim_alt > {fs} > {ar}: absent or incomplete", ("", ""))
                elif not excl or not qq["den_ci_excludes_0"]:
                    P.add(f"{nm}{mac}", ("n.e.", "n.e."), src(FL, "stim_alt", fs, ar, "r_inside", q) + ": aligned or denominator CI includes 0", ("", ""))
                else:
                    P.ratio(f"{nm}{mac}", qq, src(FL, "stim_alt", fs, ar, "r_inside", q) + f" (seeds {sa['seeds']})")
                    c[mac] = qq
        else:
            for mac in ("Kz", "KzCore", "KzNinety", "RealFilled", "Real", "Ehr", "UniTwenty", "AltTwenty", "DropTwenty",
                        "RotUni", "RotAlt", "RotDrop"):
                P.add(f"{nm}{mac}", ("pending", "pending"), f"{FL} > stim_v3 > {fs} > {ar}: absent (no scores yet)", ("", ""))
        cells[(fs, ar)] = c
    # paired contrasts (the five pairs of v3/gpu/analyze_full.py); placeholders keep the macro set stable
    for pr in ("R1n-R1", "R1n-R4", "R1-R4", "R1n-R2", "R1n-R3"):
        x, y = pr.split("-")
        rr = full["contrasts"].get(fs, {}).get(pr)
        nmc = f"RS{F}{dict(ARCHS)[x]}Minus{dict(ARCHS)[y]}"
        if rr:
            P.interval(nmc, rr["kanizsa|illusory-control180|avg|inside"],
                       src(FL, "contrasts", fs, pr, "kanizsa|illusory-control180|avg|inside") + f" (seeds {rr['n_seeds']})")
        else:
            P.add(nmc, ("pending", "pending"), f"{FL} > contrasts > {fs} > {pr}: absent", ("", ""))
# global readout-swap summaries
avail = [c for c in cells.values() if "Kz" in c]
ncomp = sum(1 for c in cells.values() if not c["prov"])
nprov = sum(1 for c in cells.values() if c["prov"] and "Kz" in c)
nmiss = sum(1 for c in cells.values() if "Kz" not in c)
P.add("RSNCellsPlanned", fint(len(cells)), "derived: RSNFeatureSets x RSNReadouts")
P.add("RSNCellsComplete", fint(ncomp), "derived: cells with all planned seeds (MIT1003 and stimuli)")
P.add("RSNCellsProvisional", fint(nprov), "derived: cells with scores from fewer than the planned seeds")
P.add("RSNCellsMissing", fint(nmiss), "derived: cells with no scores yet")
P.add("RSNModelsDone", fint(n_models_done), f"derived: sum of seeds_present over {FL} > stim_v3")
any_prov = any(c["prov"] for c in cells.values())
P.text("RSProvisional", "yes" if any_prov else "no", "derived: any readout-swap cell provisional or missing")
NUMW = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def nw(n):
    return NUMW.get(n, f"{n:,}")


parts = [f"{nw(ncomp) if ncomp else 'none'} of the {len(cells)} feature-readout cells had all {nw(len(PLANNED_SEEDS))} seeds"]
if nprov:
    parts.append(f"{nw(nprov)} had fewer")
if nmiss:
    parts.append(f"{nw(nmiss)} had no completed seed")
status = parts[0] + (", " + ", ".join(parts[1:-1]) + " and " + parts[-1] if len(parts) > 2 else (" and " + parts[1] if len(parts) == 2 else ""))
P.text("RSStatusText", status, "derived: RSNCellsComplete, RSNCellsProvisional, RSNCellsMissing in words")
fs_sign = {}
n_same = n_valid = n_valid_same = n_runs = n_runs_same = n_vruns = n_vruns_same = 0
multi_valid = []
for fs, F, flabel in FSETS:
    r4 = cells[(fs, "R4")]
    if "Kz" not in r4:
        continue
    sgn = 1 if r4["Kz"]["mean"] > 0 else -1
    fs_sign[fs] = sgn
    P.text(f"RS{F}Sign", "positive" if sgn > 0 else "negative", f"derived: sign of RS{F}RfourKz")
    fa = [c for (f_, a_), c in cells.items() if f_ == fs and "Kz" in c]
    same = [c for c in fa if (c["Kz"]["ci"][0] > 0 if sgn > 0 else c["Kz"]["ci"][1] < 0)]
    P.add(f"RS{F}NSameSign", fint(len(same)), f"derived: readouts on {fs} whose Kanizsa CI excludes 0 on the side of R4's sign")
    P.add(f"RS{F}NAvailable", fint(len(fa)), f"derived: readouts on {fs} with scores")
    vals = [c["Kz"]["mean"] for c in fa]
    P.add(f"RS{F}KzMin", fnum(min(vals), 3, True), f"derived: min Kanizsa effect over available readouts on {fs}")
    P.add(f"RS{F}KzMax", fnum(max(vals), 3, True), f"derived: max Kanizsa effect over available readouts on {fs}")
    vv = [c for c in fa if c["valid"]]
    if vv:
        P.add(f"RS{F}KzValidMin", fnum(min(c["Kz"]["mean"] for c in vv), 3, True), f"derived: min over valid readouts on {fs}")
        P.add(f"RS{F}KzValidMax", fnum(max(c["Kz"]["mean"] for c in vv), 3, True), f"derived: max over valid readouts on {fs}")
    n_same += len(same)
    n_valid += len(vv)
    n_valid_same += sum(1 for c in vv if c in same)
    P.add(f"RS{F}NValid", fint(len(vv)), f"derived: readouts on {fs} with scores that pass the validity rule")
    if len(vv) >= 2:
        multi_valid.append(flabel)
    ru = [c["RotUni"]["value"] for c in vv if "RotUni" in c]
    if ru:
        P.add(f"RS{F}RotUniValidMin", fnum(min(ru), 2), f"derived: min over valid readouts on {fs} of stim_alt ratio_uni20 (r_inside)")
        P.add(f"RS{F}RotUniValidMax", fnum(max(ru), 2), f"derived: max over valid readouts on {fs} of stim_alt ratio_uni20 (r_inside)")
    for c in fa:
        sm = c["Kz"].get("seed_means", [])
        n_runs += len(sm)
        n_runs_same += sum(1 for v in sm if v * sgn > 0)
        if c["valid"]:
            n_vruns += len(sm)
            n_vruns_same += sum(1 for v in sm if v * sgn > 0)
P.add("RSNCellsAvailable", fint(len(avail)), "derived: readout-swap cells with scores")
P.add("RSNCellsSameSign", fint(n_same), "derived: cells whose Kanizsa CI excludes 0 with the sign of R4 on the same features")
P.add("RSNValidCells", fint(n_valid), "derived: cells with scores that pass the MIT1003 validity rule")
P.add("RSNValidCellsSameSign", fint(n_valid_same), "derived: valid cells whose CI excludes 0 with the feature set's sign")
P.add("RSNSeedRuns", fint(n_runs), "derived: per-seed Kanizsa effects (seed_means) over all cells with scores")
P.add("RSNSeedRunsSameSign", fint(n_runs_same), "derived: per-seed effects with the feature set's sign")
P.add("RSNValidSeedRuns", fint(n_vruns), "derived: per-seed Kanizsa effects (seed_means) over the valid cells with scores")
P.add("RSNValidSeedRunsSameSign", fint(n_vruns_same), "derived: of those, per-seed effects with the feature set's sign")
P.add("RSNMultiValidFeatures", fint(len(multi_valid)), "derived: feature sets with at least two valid readouts with scores (where a readout comparison among valid readouts exists)")
def join_fs(labels):
    """'DINOv2-B block 12', 'DINOv2-B block 6' -> 'DINOv2-B blocks 12 and 6'; other labels joined with 'and'."""
    groups = {}
    for lab_ in labels:
        net_, _, blk_ = lab_.partition(" block ")
        groups.setdefault(net_, []).append(blk_)
    parts = [(f"{n_} block{'s' if len(b_) > 1 else ''} " + " and ".join(b_)) if b_[0] else n_ for n_, b_ in groups.items()]
    return " and ".join(parts) if parts else "none"


P.text("RSMultiValidFeatures", join_fs(multi_valid), "derived: the feature sets with at least two valid readouts with scores")
single_valid = [lab for fs, F, lab in FSETS if lab not in multi_valid]
P.text("RSSingleValidFeatures", join_fs(single_valid), "derived: feature sets with at most one valid readout with scores")
P.add("RSNSingleValidFeatures", fint(len(single_valid)), "derived: count of the same")
if single_valid:
    vc = (f"A comparison among valid readouts therefore exists on {join_fs(multi_valid)}; on {join_fs(single_valid)} "
          f"features only R4 is valid so far.")
else:
    vc = "A comparison among valid readouts therefore exists on every feature set."
P.text("RSValidComparisonText", vc, "derived: RSMultiValidFeatures / RSSingleValidFeatures as a sentence")
P.text("RSNMultiValidWord", {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(multi_valid), str(len(multi_valid))), "derived: RSNMultiValidFeatures in words")
# largest effect of any model or readout against the smallest corner-null effect (title claim)
assert allmax[0] < min(corner_kz), (allmax, min(corner_kz))
P.text("TitleCornerAboveAllModels", "yes", f"derived: asserted MaxAnyModelKz ({allmax[0]:+.3f}) < NullCornerKzMin ({min(corner_kz):+.3f})")
# two-level (seed) bootstrap sensitivity, paper_numbers/check_seed_bootstrap.py
if SB.exists():
    sbj = json.load(open(SB))
    P.add("SeedBootNChanged", fint(sbj["n_changed"]), "paper/seed_bootstrap.json > n_changed (intervals whose position relative to 0 changes when seeds are also resampled)")
    P.add("SeedBootNIntervals", fint(sbj["n_intervals"]), "paper/seed_bootstrap.json > n_intervals (inside and core Kanizsa effects of every cell with scores)")
    P.text("SeedBootText", f"{nw(sbj['n_changed'])} of the {sbj['n_intervals']}", "derived: SeedBootNChanged of SeedBootNIntervals, in words")
    if sbj["n_cells"] != len(avail):
        WARN.append(f"seed_bootstrap.json covers {sbj['n_cells']} cells but {len(avail)} cells have scores: rerun paper_numbers/check_seed_bootstrap.py")
else:
    WARN.append("paper/seed_bootstrap.json missing: run paper_numbers/check_seed_bootstrap.py")
    for nm_ in ("SeedBootNChanged", "SeedBootNIntervals", "SeedBootText"):
        P.add(nm_, ("pending", "pending"), "paper/seed_bootstrap.json missing")

# ================================================================== 8. layer sweep (Appendix B)
P.section("8. Layer sweep: one R4-type ASPP readout per block, seed 0 (v3/scores/results.json; held-out CC from v2/anatomy/anatomy.json)")
SWEEP = (("DINOv2-B", "dinov2", "Dino", (2, 8, 12)), ("CLIP-L/14", "clip", "Clip", (4, 12, 24)), ("SigLIP-L/16", "siglip", "Siglip", (4, 8, 24)))
sweep_rows = []
for lab, an, key, layers in SWEEP:
    for L_ in layers:
        nm3 = f"{lab.split('/')[0]} L{L_}"
        o = v3m[nm3]["results"]["r_inside"]["all"]; sp = ("models", nm3, "results", "r_inside", "all")
        mac = f"Sweep{key}{WORD[L_]}"
        ia = byname[f"sweep {an} L{L_}"]
        P.add(f"{mac}CC", fnum(recs[ia]["heldout_mit1003_cc"], 3), src(AN, "records", ia, "heldout_mit1003_cc"))
        P.interval(f"{mac}Kz", o["aligned"], src(V3, *sp, "aligned") + " (size avg 336/448 px, seed 0)")
        row_ = dict(label=f"{lab}, {L_}", cc=recs[ia]["heldout_mit1003_cc"], kz=o["aligned"], rc=o["ratios_computed"] and o["aligned"]["ci"][0] > 0)
        for q, m_ in (("ratio_uni", "RotUni"), ("ratio_alt", "RotAlt"), ("drop_ratio", "RotDrop")):
            qq = (o[q]["20"] if o["ratios_computed"] or q == "drop_ratio" else None)
            if qq is not None and (q == "drop_ratio" or o["ratios_computed"]):
                P.ratio(f"{mac}{m_}", qq, src(V3, *sp, q, "20"))
                row_[m_] = qq
            else:
                P.add(f"{mac}{m_}", ("n.e.", "n.e."), src(V3, *sp, q, "20") + ": not computed (aligned or uniform-drop CI includes 0)", ("", ""))
        sl = o["sr_slope_vs_control180"]
        P.add(f"{mac}SRSlope", fnum(sl["slope_per_unit_sr"], 3, True), src(V3, *sp, "sr_slope_vs_control180", "slope_per_unit_sr"), fci(sl["ci"], 3))
        row_["slope"] = sl
        sweep_rows.append(row_)
P.add("SweepNScored", fint(len(sweep_rows)), "derived: sweep readouts scored on these displays (rows of Appendix B)")
P.add("SweepNTotal", fint(sum(1 for r_ in recs if r_["model"].startswith("sweep "))), "derived: sweep readouts in v2/anatomy/anatomy.json")
P.add("SweepNUnscored", fint(sum(1 for r_ in recs if r_["model"].startswith("sweep ")) - len(sweep_rows)), "derived: SweepNTotal - SweepNScored (sweep blocks not scored on the present displays)")
pw_invalid = [F for fs, F, _ in FSETS if all(cells[(fs, a)]["valid"] is False for a in ("R1n", "R1"))]
pw_valid = [lab for fs, F, lab in FSETS if all(cells[(fs, a)]["valid"] is True for a in ("R1n", "R1"))]
P.add("RSNPointwiseInvalid", fint(len(pw_invalid)), "derived: feature sets on which both pointwise readouts (R1n, R1) fail the validity rule")
P.text("RSPointwiseValidFeatures", ", ".join(pw_valid) if pw_valid else "none", "derived: feature sets on which both pointwise readouts pass")
nonpw_invalid = [(fs, a) for (fs, a), c in cells.items() if a in ("R2", "R3", "R4") and c["valid"] is False]
P.add("RSNNonPointwiseInvalid", fint(len(nonpw_invalid)), "derived: R2/R3/R4 cells that fail the validity rule")
if any_prov:
    WARN.append(f"readout swap provisional: {nprov} cells with fewer than {len(PLANNED_SEEDS)} seeds, {nmiss} cells without scores")

# ================================================================== write numbers.tex
hdr = ["% numbers.tex: generated by paper_numbers/make_numbers.py on " + time.strftime("%Y-%m-%d %H:%M:%S") + "; do not edit by hand.",
       f"% Readout-swap results file created {full['created']}; provisional cells: {nprov}, missing cells: {nmiss}.",
       "% Each macro is preceded by '% <file> > <JSON path>' or '% derived: ...'. A macro X with a CI has a companion XCI = [lo, hi].",
       "% Effects are differences in r = map mass inside the figure / (inside + inducer AOI); signed values use \\ensuremath.",
       "% derived: true while any readout-swap cell in v3/gpu/full/results_full.json has fewer than 3 seeds or no scores",
       r"\newif\ifRSProvisional", r"\RSProvisional%s" % ("true" if any_prov else "false")]
P.write(OUT / "numbers.tex", hdr)


# ================================================================== tables
def cell2(rec, d=3, signed=True, key="mean"):
    """(value tex, CI tex) for a table cell."""
    if rec is None:
        return None
    v = fnum(rec[key] if key in rec else rec["value"], d, signed)[1]
    return v, fci(rec["ci"], d)[1]


def tab_readout():
    L = [r"% tab_readout.tex: generated by paper_numbers/make_numbers.py from v3/gpu/full/results_full.json; do not edit by hand.",
         r"% Needs \usepackage{booktabs}.",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}"]
    cap = (r"\caption{Readout swap on identical frozen features. Each readout was trained on the same " + f"{tot:,}" +
           r" images with the same recipe; rows give the mean over seeds. MIT1003 CC: held-out correlation on " + f"{held}" +
           r" images; $^{\mathrm{a}}$ below the validity threshold (CC of R4 minus " + f"{full['cc_margin']:.2f}" +
           r"), reported for completeness. Kanizsa effect: $r$(illusory) minus $r$(180$^\circ$ control) on the inside-figure AOI; core: the same contrast for figure pixels more than 30~px from the inducers; both averaged over 336 and 448~px inputs, 95\% cluster-bootstrap CI over " +
           f"{nK}" + r" configurations. R1n: pointwise readout with fixed per-channel normalisation; R1: DeepGaze-style pointwise readout (image-wide LayerNorm); R2: ASPP without global pooling; R3: dilation-1 branch only; R4: full ASPP with global pooling.")
    if any_prov:
        cap += r" $^\ast$Provisional: fewer than " + f"{len(PLANNED_SEEDS)}" + r" seeds (seed count shown)."
    cap += r"}"
    L += [cap, r"\label{tab:readout}",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"Readout & MIT1003 CC & Kanizsa effect [95\% CI] & Core effect [95\% CI] & Seeds \\", r"\midrule"]
    for gi, (fs, F, flabel) in enumerate(FSETS):
        if gi:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{5}{@{}l}{\textit{" + flabel + r"}} \\")
        for ai, (ar, A) in enumerate(ARCHS):
            c = cells[(fs, ar)]
            cc = "pending" if c["cc"] is None else f"{c['cc']:.3f}" + (r"$^{\mathrm{a}}$" if c["valid"] is False else "")
            if "Kz" in c:
                kv, kc = cell2(c["Kz"]); cv, ccc = cell2(c["KzCore"])
                kz, core = f"{kv} {kc}", f"{cv} {ccc}"
            else:
                kz = core = "pending"
            sd = f"{c['n']}" + (r"$^\ast$" if c["prov"] else "")
            L.append(f"\\quad {ar} & {cc} & {kz} & {core} & {sd} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_readout.tex").write_text("\n".join(L) + "\n")


def tab_survey():
    L = [r"% tab_survey.tex: generated by paper_numbers/make_numbers.py from v2/scores/results.json, v3/scores/results.json and",
         r"% v2/verify/core_aoi_D30_835295.json; do not edit by hand. Needs \usepackage{booktabs}.",
         r"\begin{sidewaystable}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         (r"\caption{Six saliency models and the local-null family on the Kanizsa displays. Effects are differences in $r$, the map mass inside the figure divided by the mass inside the figure plus on the inducers, between the illusory display and a control whose inducers are rotated by 180$^\circ$ or 90$^\circ$; means over " +
          f"{nK}" + r" configurations, with 95\% cluster-bootstrap CIs in brackets for the 180$^\circ$ effect, the core effect and the slope (the CIs of the 90$^\circ$ effects and ratios of DeepGaze IIE and MSDB are given in the text). Core: the same contrast for figure pixels more than 30~px from the inducer discs. Ratios: the effect with every inducer rotated by 20$^\circ$ in the same direction (uniform) or in alternating directions, divided by the aligned effect; computed only where the aligned CI lies above 0 (n.e. otherwise). Slope: change of the aligned effect per unit support ratio (0.3 to 0.7). Local nulls: corner-detector, ink and edge-energy maps blurred with a Gaussian of $\sigma$~px; the edge null at $\sigma = 25$~px is the edge-energy null of the survey. n.c.: not computed for this model.}"),
         r"\label{tab:survey}",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Kanizsa effect} & Core & \multicolumn{2}{c}{Ratio at 20$^\circ$} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
         r"Model & 180$^\circ$ control & 90$^\circ$ control & effect & Uniform & Alternating & Slope \\", r"\midrule"]

    def row(label, kz, k90, core, uni, alt, slope):
        # CIs for the 180-degree effect, the core effect and the slope; the 90-degree effect and the ratios are point
        # estimates here (the CIs of the two DeepGaze models are in the text; all CIs are in numbers.tex)
        cols = [kz, k90, core, uni, alt, slope]
        withci = [True, False, True, False, False, True]
        cells = [(f"{c[0]} {c[1]}" if w else c[0]) if isinstance(c, tuple) else c for c, w in zip(cols, withci)]
        L.append(f"{label} & " + " & ".join(cells) + r" \\")

    for key, name, tag, label in SIX:
        r = survey[key]
        elig = r["Kz"]["ci"][0] > 0
        row(label, cell2(r["Kz"]), cell2(r["KzNinety"]), cell2(r["Core"]) if "Core" in r else "n.c.",
            cell2(r["Uni20"], 2, False, "value") if elig else "n.e.",
            (cell2(r["Alt20"], 2, False, "value") if "Alt20" in r else "n.c.") if elig else "n.e.",
            cell2(r["Slope"], 3, True, "slope_per_unit_sr") if "Slope" in r else "n.c.")
    # edge-energy null of the survey = local null edge_s25 (asserted)
    e2, e3 = survey["EdgeNull"], nullrec[("edge", 25)]
    assert abs(e2["Kz"]["mean"] - e3["Kz"]["mean"]) < 1e-3 and abs(e2["KzNinety"]["mean"] - e3["KzNinety"]["mean"]) < 1e-3
    assert abs(e2["Core"]["mean"] - e3["KzCore"]["mean"]) < 1e-3
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{@{}l}{\textit{Local nulls}} \\")
    for t, T in NULL_TYPES:
        for s in SIGMAS:
            r = nullrec[(t, s)]
            elig = r["Kz"]["ci"][0] > 0
            lab = f"{T}, $\\sigma = {s}$"
            row(lab, cell2(r["Kz"]), cell2(r["KzNinety"]), cell2(r["KzCore"]),
                cell2(r["Uni20"], 2, False, "value") if elig and "Uni20" in r else "n.e.",
                cell2(r["Alt20"], 2, False, "value") if elig and "Alt20" in r else "n.e.",
                cell2(r["Slope"], 3, True, "slope_per_unit_sr"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    (OUT / "tab_survey.tex").write_text("\n".join(L) + "\n")


def tab_sweep():
    L = [r"% tab_sweep.tex: generated by paper_numbers/make_numbers.py from v3/scores/results.json and v2/anatomy/anatomy.json; do not edit by hand.",
         r"% Needs \usepackage{booktabs}.",
         r"\begin{sidewaystable}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         (r"\caption{Layer sweep: one readout of the R4 design (full atrous spatial pyramid pooling with global pooling) per block, trained with the data and recipe of the readout swap (seed 0 only) and scored on the same displays (size average of 336 and 448~px). Blocks are 1-based outputs before any final layer normalisation. CC: held-out MIT1003 correlation. Kanizsa effect: illusory minus 180$^\circ$ control on the inside region, 95\% cluster-bootstrap CI over " +
          f"{nK}" + r" configurations in brackets. Uniform and alternating: effect after a 20$^\circ$ rotation divided by the aligned effect, computed where the aligned CI excludes 0; drop: loss under alternating divided by loss under uniform rotation, computed where the uniform loss CI excludes 0 (n.e. otherwise). Slope: change of the effect per unit support ratio. For comparison, DeepGaze IIE: uniform " +
          MACRO["AltIIEUniTwenty"] + r", alternating " + MACRO["AltIIEAltTwenty"] + r", drop " + MACRO["AltIIEDropTwenty"] + r"; corner nulls: uniform and alternating at least " + MACRO["NullCornerMinRetained"] + r".}"),
         r"\label{tab:sweep}",
         r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Network, block & CC & Kanizsa effect & Uniform & Alternating & Drop & Slope \\", r"\midrule"]
    for r_ in sweep_rows:
        cols = [cell2(r_["kz"]), cell2(r_["RotUni"], 2, False, "value") if "RotUni" in r_ else "n.e.",
                cell2(r_["RotAlt"], 2, False, "value") if "RotAlt" in r_ else "n.e.",
                cell2(r_["RotDrop"], 2, False, "value") if "RotDrop" in r_ else "n.e.",
                cell2(r_["slope"], 3, True, "slope_per_unit_sr")]
        cells = [f"{c[0]} {c[1]}" if isinstance(c, tuple) else c for c in cols]
        L.append(f"{r_['label']} & {r_['cc']:.3f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable}"]
    (OUT / "tab_sweep.tex").write_text("\n".join(L) + "\n")


MACRO = {n: t for n, t, *_ in P.items if n != "__section__"}
tab_readout()
tab_survey()
tab_sweep()

# ================================================================== listing
lst = P.listing()
nmac = sum(1 + (1 if ci else 0) for n, _, _, _, ci in P.items if n != "__section__")
print(f"# numbers.tex: {nmac} macros ({sum(1 for n, *_ in P.items if n != '__section__')} quantities); tab_readout.tex; tab_survey.tex")
print(f"# readout swap: results created {full['created']}; complete {ncomp}, provisional {nprov}, missing {nmiss} of {len(cells)} cells")
for w in WARN:
    print("# WARNING " + w)
print("\n".join(lst))
