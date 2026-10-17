"""
Readout-swap pilot statistics (cluster bootstrap, validity rule, sign summaries); its design, effects, summarise and
boot functions are reused by readout_swap/analyze_full.py and paper_numbers/check_seed_bootstrap.py.
Reads <results>/v2/readout/scores/*.json, writes results.json and report.md there.
Usage: python analyze.py
"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RO, V2, SCORES, FSETS, ARCHS, SEEDS, stim_items

B = 10000
SIZES = ("336", "448")
VARIANTS = ("inside", "inside_eroded", "inside_eroded_all")
K_TESTS = ("illusory", "control90", "real", "real_filled", "realC", "mis10", "mis20", "mis30")
E_TESTS = ("illusory", "real", "real_filled", "realC")
CC_MARGIN = 0.03
EXPL = "R1x30ep"          # exploratory 30-epoch R1; never used for the primary label


def pstr(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def boot(vals, seed=2026):
    """vals: [K cells, P positions]. Cluster bootstrap: cells with replacement, positions within cells."""
    K, P = vals.shape
    rng = np.random.RandomState(seed)
    ci = rng.randint(0, K, (B, K)); pi = rng.randint(0, P, (B, K, P))
    bm = vals[ci[:, :, None], pi].mean((1, 2))
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(bm <= 0), np.mean(bm >= 0)))
    return dict(mean=round(float(vals.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)], p=float(p), p_str=pstr(p))


def design(stimset):
    """configs grouped into cells; returns (illusion -> [K][P] config keys, primary control, tests)."""
    items = stim_items(stimset)
    out = {}
    if stimset == "stim_v2":
        cfg = sorted({it["name"].rsplit("_", 1)[0] for it in items})
        cells = {}
        for c in cfg:
            sh, sr, pol, pos = c.split("_"); cells.setdefault((sh, sr, pol), []).append(c)
        out["kanizsa"] = dict(cells=[sorted(v) for _, v in sorted(cells.items())], ctrl="control", tests=("illusory", "control90"))
        return out
    for ill, ctrl, tests in (("kanizsa", "control180", K_TESTS), ("ehrenstein", "control_end", E_TESTS)):
        cells = {}
        for it in items:
            if it["illusion"] == ill and it["cond"] == "illusory":
                key = (it["shape"], it["sr"], it["pol"]) if ill == "kanizsa" else (it["R"], it["n"], it["pol"])
                cells.setdefault(key, []).append(it["config"])
        out[ill] = dict(cells=[sorted(v) for _, v in sorted(cells.items(), key=lambda kv: str(kv[0]))], ctrl=ctrl, tests=tests)
    return out


def effects(sc, d, test, ctrl, size, var):
    """[K, P] array of r(test) - r(ctrl); size 'avg' = mean of the 336 and 448 px effects."""
    sizes = SIZES if size == "avg" else (size,)
    return np.mean([[[sc[s][var][f"{c}_{test}"] - sc[s][var][f"{c}_{ctrl}"] for c in cell] for cell in d["cells"]]
                    for s in sizes], 0)


def summarise(scs, d, test, ctrl, size, var):
    """scs: list of per-seed score dicts. Seed-averaged effect + bootstrap, and per-seed summaries."""
    per = [effects(sc, d, test, ctrl, size, var) for sc in scs]
    r = boot(np.mean(per, 0))
    r["seed_means"] = [round(float(e.mean()), 4) for e in per]
    if len(per) > 1:
        r["seed_cis"] = [boot(e)["ci"] for e in per]
    r["n_seeds"] = len(per)
    return r


def load_scores(stimset, fset, tags):
    out = {}
    for t in tags:
        p = os.path.join(SCORES, f"{stimset}_{fset}_{t}.json")
        if os.path.exists(p):
            out[t] = json.load(open(p))
    return out


def main():
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), bootstrap=B, cc_margin=CC_MARGIN, mit={}, stim_v3={}, stim_v2_interim={})
    # ---------------- held-out MIT1003 CC and validity
    for fs in FSETS:
        res["mit"][fs] = {}
        for a in ARCHS + (EXPL,):
            cc = {s: json.load(open(p))["CC"] for s in SEEDS if os.path.exists(p := os.path.join(SCORES, f"mit_{fs}_{a}_s{s}.json"))}
            if cc:
                res["mit"][fs][a] = dict(seeds={str(k): round(v, 4) for k, v in cc.items()}, mean=round(float(np.mean(list(cc.values()))), 4), n=len(cc))
        r4 = res["mit"][fs].get("R4")
        for a, v in res["mit"][fs].items():
            if r4 and r4["n"] == 3 and v["n"] == 3:
                v["threshold"] = round(r4["mean"] - CC_MARGIN, 4); v["valid"] = bool(v["mean"] >= r4["mean"] - CC_MARGIN)
            else:
                v["valid"] = None
    conv = os.path.join(SCORES, "convention_check.json")
    if os.path.exists(conv):
        res["convention_check"] = json.load(open(conv))
    # ---------------- stimulus effects
    for stimset, key in (("stim_v3", "stim_v3"), ("stim_v2", "stim_v2_interim")):
        if not glob.glob(os.path.join(SCORES, f"{stimset}_*.json")):
            continue
        D = design(stimset)
        for fs in FSETS:
            groups = [(a, [f"{a}_s{s}" for s in SEEDS]) for a in ARCHS + (EXPL,)] + [("R4_fulldata_sweep", ["R4_ssweep"])]
            for group, tags in groups:
                scs = load_scores(stimset, fs, tags)
                if not scs:
                    continue
                rec = dict(seeds_present=sorted(scs))
                for ill, d in D.items():
                    rec[ill] = {}
                    tests = [(t, d["ctrl"]) for t in d["tests"]] + ([("illusory", "control90")] if ill == "kanizsa" else [])
                    for test, ctrl in tests:
                        nm = f"{test}-{ctrl}"
                        rec[ill][nm] = {}
                        for size in ("avg",) + SIZES:
                            for var in (VARIANTS if stimset == "stim_v3" else ("inside",)):
                                rec[ill][nm][f"{size}|{var}"] = summarise(list(scs.values()), d, test, ctrl, size, var)
                res[key].setdefault(fs, {})[group] = rec
    # ---------------- gate and decision
    res["gate_interim_stim_v2"] = gate(res["stim_v2_interim"], "kanizsa", "illusory-control")
    res["gate"] = gate(res["stim_v3"], "kanizsa", "illusory-control180")
    res["decision"] = {sz: decide(res, sz) for sz in ("avg", "336", "448")}
    res["decision_sensitivity_R1x30ep"] = {sz: decide(res, sz, r1=EXPL) for sz in ("avg", "336", "448")}
    json.dump(res, open(os.path.join(RO, "results.json"), "w"), indent=1)
    report(res)
    print(json.dumps(res["gate_interim_stim_v2"]), "\n", json.dumps(res["gate"]), "\n", json.dumps({k: v["label"] for k, v in res["decision"].items()}))


def get(res, fs, a, nm="illusory-control180", size="avg", var="inside", ill="kanizsa"):
    try:
        return res["stim_v3"][fs][a][ill][nm][f"{size}|{var}"]
    except KeyError:
        return None


def gate(block, ill, nm):
    out = {}
    for size in ("avg",) + SIZES:
        try:
            e = block["dinov2_L12"]["R4"][ill][nm][f"{size}|inside"]
        except KeyError:
            continue
        out[size] = dict(effect=e, n_seeds=e["n_seeds"], passed=bool(e["ci"][1] < 0))
    return out


def decide(res, size, r1="R1"):
    """Pilot decision rules, applied literally. r1=EXPL: sensitivity with the 30-epoch R1."""
    need = [(fs, a) for fs in FSETS for a in ARCHS]
    eff = {(fs, a): get(res, fs, r1 if a == "R1" else a, size=size) for fs, a in need}
    missing = [f"{fs}/{a}" for (fs, a), e in eff.items() if e is None or e["n_seeds"] < 3]
    valid = {(fs, a): res["mit"].get(fs, {}).get(r1 if a == "R1" else a, {}).get("valid") for fs, a in need}
    g = eff[("dinov2_L12", "R4")]
    out = dict(size=size, missing=missing, invalid=[f"{fs}/{a}" for k, v in valid.items() if v is False for fs, a in [k]])
    if g is None:
        out["label"] = "incomplete (gate readout missing)"; return out
    out["gate_passed"] = bool(g["ci"][1] < 0)
    if not out["gate_passed"]:
        out["label"] = "non-diagnostic"; return out
    L = {a: eff[("dinov2_L12", a)] for a in ARCHS}
    pos = lambda e: e is not None and e["ci"][0] > 0 and all(m > 0 for m in e["seed_means"])
    neg = lambda e: e is not None and e["ci"][1] < 0 and all(m < 0 for m in e["seed_means"])
    notbelow = lambda e: e is not None and e["ci"][1] >= 0
    v = lambda a, fs="dinov2_L12": valid[(fs, a)] is True
    out["R1_above_0_all_seeds"] = pos(L["R1"]); out["R4_below_0_all_seeds"] = neg(L["R4"])
    readout = v("R1") and v("R4") and pos(L["R1"]) and neg(L["R4"])
    gp = readout and v("R2") and v("R3") and notbelow(L["R2"]) and notbelow(L["R3"])
    signs = {fs: {a: (None if eff[(fs, a)] is None else int(np.sign(eff[(fs, a)]["mean"]))) for a in ARCHS} for fs in FSETS}
    out["signs"] = signs
    strict_feat = (not missing) and all(valid[(fs, a)] is True for fs, a in need) and \
        all(len({signs[fs][a] for a in ARCHS}) == 1 for fs in FSETS)
    valid_only = {fs: {signs[fs][a] for a in ARCHS if valid[(fs, a)] is True and signs[fs][a] is not None} for fs in FSETS}
    out["features_share_sign_valid_only"] = all(len(s) <= 1 for s in valid_only.values())
    if gp:
        out["label"] = "global pooling is the critical component"
    elif readout:
        out["label"] = "readout decides"
    elif strict_feat:
        out["label"] = "features decide"
    else:
        out["label"] = "mixed"
    if missing:
        out["label"] += " (provisional: missing " + ", ".join(missing) + ")"
    return out


# ------------------------------------------------------------------ report
def fmt(e):
    if e is None:
        return "n/a"
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] ({', '.join(f'{m:+.3f}' for m in e['seed_means'])})"


def report(res):
    L = []
    w = L.append
    w("# Readout swap, CPU pilot\n")
    w(f"Generated by readout_swap/pilot/analyze.py on {res['created']}.\n")
    w("Effects are r(test) - r(control); r = map mass inside the figure / (inside + inducers). Cells: mean over configurations "
      "and the three seeds, cluster-bootstrap 95% CI (10,000 resamples over shape x support ratio x polarity cells, positions "
      "resampled within cells), then the three per-seed means in parentheses. Primary size = mean of the 336 and 448 px effects.\n")
    d = res["decision"]["avg"]
    w("## Outcome\n")
    w(f"**Primary label (size-averaged): {d['label']}.** At 336 px: {res['decision']['336']['label']}. At 448 px: "
      f"{res['decision']['448']['label']}.\n")
    w("## Gate\n")
    for nm, gk_ in (("Interim gate on stim_v2 (illusory vs 180 deg control; not the primary gate)", "gate_interim_stim_v2"),
                    ("Gate on stim_v3 (illusory vs control180)", "gate")):
        w(f"- {nm}: " + "; ".join(f"{s} px: {fmt(v['effect'])}, {v['effect']['p_str']}, {'passed' if v['passed'] else 'FAILED'}"
                                     for s, v in res[gk_].items()) if res[gk_] else f"- {nm}: not available")
    w("")
    w("## Held-out MIT1003 CC (200 images) and validity (CC >= R4 - 0.03)\n")
    w("| features | R1 | R2 | R3 | R4 | R1x30ep (exploratory) |\n|---|---|---|---|---|---|")
    for fs in FSETS:
        row = []
        for a in ARCHS + (EXPL,):
            v = res["mit"].get(fs, {}).get(a)
            row.append("n/a" if v is None else f"{v['mean']:.4f} ({', '.join(f'{x:.3f}' for x in v['seeds'].values())})"
                       + ("" if v.get("valid") is None else (" valid" if v["valid"] else " INVALID")))
        w(f"| {fs} | " + " | ".join(row) + " |")
    w("")
    for ill, nm0 in (("kanizsa", "illusory-control180"), ("ehrenstein", "illusory-control_end")):
        w(f"## {ill.capitalize()} effect ({nm0}), stim_v3\n")
        for size in ("avg",) + SIZES:
            w(f"Input {size if size != 'avg' else 'mean of 336 and 448'} px, full inside AOI:\n")
            w("| features | R1 | R2 | R3 | R4 | R1x30ep (exploratory) | R4 full-data sweep readout |\n|---|---|---|---|---|---|---|")
            for fs in FSETS:
                cells = [fmt(get(res, fs, a, nm0, size, "inside", ill)) for a in ARCHS + (EXPL,)]
                cells.append(fmt(get(res, fs, "R4_fulldata_sweep", nm0, size, "inside", ill)) if fs != "iie_shapenetc" else "n/a")
                w(f"| {fs} | " + " | ".join(cells) + " |")
            w("")
    w("## Kanizsa, all conditions vs control180 (size-averaged, full inside AOI; p from the seed-averaged bootstrap)\n")
    tests = [f"{t}-control180" for t in K_TESTS] + ["illusory-control90"]
    w("| features | readout | " + " | ".join(tests) + " |\n|---|---|" + "---|" * len(tests))
    for fs in FSETS:
        for a in ARCHS + (EXPL, "R4_fulldata_sweep"):
            es = [get(res, fs, a, t) for t in tests]
            if all(e is None for e in es):
                continue
            w(f"| {fs} | {a} | " + " | ".join("n/a" if e is None else f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]" for e in es) + " |")
    w("")
    w("## Ehrenstein, all conditions vs control_end (size-averaged, full inside AOI)\n")
    tests = [f"{t}-control_end" for t in E_TESTS]
    w("| features | readout | " + " | ".join(tests) + " |\n|---|---|" + "---|" * len(tests))
    for fs in FSETS:
        for a in ARCHS + (EXPL, "R4_fulldata_sweep"):
            es = [get(res, fs, a, t, ill="ehrenstein") for t in tests]
            if all(e is None for e in es):
                continue
            w(f"| {fs} | {a} | " + " | ".join("n/a" if e is None else f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]" for e in es) + " |")
    w("")
    w("## Kanizsa illusory-control180 with the eroded inside AOIs (size-averaged)\n")
    w("| features | readout | inside | inside_eroded (8 px from contour) | inside_eroded_all |\n|---|---|---|---|---|")
    for fs in FSETS:
        for a in ARCHS + (EXPL,):
            es = [get(res, fs, a, var=v) for v in VARIANTS]
            if all(e is None for e in es):
                continue
            w(f"| {fs} | {a} | " + " | ".join(fmt(e) for e in es) + " |")
    w("")
    w("## Decision details\n")
    for sz, dd in res["decision"].items():
        w(f"- {sz}: label = {dd['label']}; gate passed = {dd.get('gate_passed')}; R1 above 0 (all seeds) = "
          f"{dd.get('R1_above_0_all_seeds')}; R4 below 0 (all seeds) = {dd.get('R4_below_0_all_seeds')}; invalid readouts = "
          f"{dd.get('invalid') or 'none'}; signs of the point estimates = {dd.get('signs')}; readouts share the sign on every "
          f"feature set when only valid readouts are counted = {dd.get('features_share_sign_valid_only')}")
    w("")
    w("Sensitivity (exploratory, not the primary label): the same rules with the 30-epoch R1 (R1x30ep) in place of R1:\n")
    for sz, dd in res["decision_sensitivity_R1x30ep"].items():
        w(f"- {sz}: label = {dd['label']}; invalid readouts = {dd.get('invalid') or 'none'}; R1x30ep above 0 (all seeds) = "
          f"{dd.get('R1_above_0_all_seeds')}; signs = {dd.get('signs')}")
    w("")
    notes = os.path.join(RO, "report_notes.md")
    if os.path.exists(notes):
        w(open(notes).read())
    open(os.path.join(RO, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
