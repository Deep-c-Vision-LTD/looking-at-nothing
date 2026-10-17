"""
Statistics for the full-data readouts (readout swap). Reads DIR/scores, writes DIR/results_full.json and
DIR/report_full.md. Usage: python analyze_full.py [--out DIR]   (default DIR = <results>/v3/gpu/full)

Code reused, not rewritten:
  readout_swap/pilot/analyze.py: design, effects, summarise, boot = cluster bootstrap 10,000 over shape x support
  ratio x polarity cells (Ehrenstein: radius x line count x polarity), positions resampled within cells, numpy
  RandomState(2026), percentile 95% CI, two-sided bootstrap p floored at "p < .001"; seeds averaged per configuration
  before the bootstrap; size "avg" = mean of the 336 and 448 px effects. Validity: MIT1003 CC >= CC(R4) - 0.03.
  survey rule for ratio(20) = effect(mis20) / effect(illusory), eligible if the aligned CI is above 0.
  analysis/analyze_alt.py definitions for the alternating-rotation set, with the same bootstrap indices:
    ratio_uni(t) = effect(mis t) / effect(illusory); ratio_alt(t) = effect(alt t) / effect(illusory);
    diff(t) = (m(alt t) - m(mis t)) / effect(illusory); drop_ratio(t) = (m(illusory) - m(alt t)) / (m(illusory) - m(mis t));
    alt t = mean of phases a and b per configuration; effects against control180; ratios = bootstrap ratio of
    resampled means, flagged when the denominator CI does not exclude 0.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fullcommon  # noqa: F401  (puts readout_swap/pilot on the path)
import analyze as A2                                        # readout_swap/pilot/analyze.py
from fullcommon import V3, FSETS, ARCHS_ALL, SEEDS, OUT_DEFAULT, LP

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT_DEFAULT); a = ap.parse_args()
SC = os.path.join(a.out, "scores")
B, CC_MARGIN = A2.B, A2.CC_MARGIN
SIZES = ("avg", "336", "448")
VARS = ("inside", "inside_eroded", "inside_eroded_all",                       # v2 readout measures
        "r_inside", "r_core30", "f_core30", "r_core45", "f_core45", "r_eroded_all", "f_inside", "f_ind")   # v3 measures
ALT_MEAS = ("r_inside", "r_core30", "f_core30")
THETAS = (10, 20, 30)
_IDX = {}


def bidx(K, P):
    """The bootstrap indices of pilot analyze.boot and analysis/analyze_v3 (RandomState(2026))."""
    if (K, P) not in _IDX:
        rng = np.random.RandomState(2026)
        _IDX[(K, P)] = (rng.randint(0, K, (B, K)), rng.randint(0, P, (B, K, P)))
    return _IDX[(K, P)]


def bmeans(v):
    ci, pi = bidx(*v.shape)
    return v[ci[:, :, None], pi].mean((1, 2))


def ratio(num, den):
    bn, bd = bmeans(num), bmeans(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bn / bd
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    dlo, dhi = np.percentile(bd, [2.5, 97.5])
    return dict(value=round(float(num.mean() / den.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                den_ci=[round(float(dlo), 4), round(float(dhi), 4)], den_ci_excludes_0=bool(dlo > 0 or dhi < 0))


def load(stimset, fs, arch):
    out = {}
    for s in SEEDS:
        p = os.path.join(SC, f"{stimset}_{fs}_{arch}_s{s}.json")
        if LP.json_exists(p):
            out[s] = LP.load_json(p)
    return out


def seedavg(scs, d, test, ctrl, size, var):
    return np.mean([A2.effects(sc, d, test, ctrl, size, var) for sc in scs], 0)


def fmt(e):
    return f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"


def fmtr(r):
    return f"{r['value']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" + ("" if r["den_ci_excludes_0"] else " (den. CI includes 0)")


def main():
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), bootstrap=B, cc_margin=CC_MARGIN, mit={}, stim_v3={},
               section_B={}, contrasts={}, stim_alt={})
    # ---------------- MIT1003 held-out CC and validity
    for fs in FSETS:
        rec = {}
        for ar in ARCHS_ALL:
            cc = {s: LP.load_json(p)["CC"] for s in SEEDS if LP.json_exists(p := os.path.join(SC, f"mit_{fs}_{ar}_s{s}.json"))}
            if cc:
                rec[ar] = dict(seeds={str(k): round(v, 4) for k, v in cc.items()}, mean=round(float(np.mean(list(cc.values()))), 4), n=len(cc))
        r4 = rec.get("R4")
        for ar, v in rec.items():
            if r4 and r4["n"] == v["n"]:
                v["threshold"] = round(r4["mean"] - CC_MARGIN, 4); v["valid"] = bool(v["mean"] >= v["threshold"])
            else:
                v["valid"] = None
        if rec:
            res["mit"][fs] = rec
    # ---------------- stim_v3 effects, section B
    D = A2.design("stim_v3")
    n_items = len(A2.stim_items("stim_v3"))
    full = {}
    for fs in FSETS:
        for ar in ARCHS_ALL:
            scs = load("stim_v3", fs, ar)
            if not scs:
                continue
            if any(len(sc["336"]["inside"]) != n_items for sc in scs.values()):
                res["stim_v3"].setdefault(fs, {})[ar] = dict(incomplete=True, seeds_present=sorted(scs)); continue
            full[(fs, ar)] = scs
            vals = list(scs.values()); rec = dict(seeds_present=sorted(scs))
            for ill, d in D.items():
                rec[ill] = {}
                tests = [(t, d["ctrl"]) for t in d["tests"]] + ([("illusory", "control90")] if ill == "kanizsa" else [])
                for test, ctrl in tests:
                    nm = f"{test}-{ctrl}"; rec[ill][nm] = {}
                    for size in SIZES:
                        for var in VARS:
                            rec[ill][nm][f"{size}|{var}"] = A2.summarise(vals, d, test, ctrl, size, var)
            res["stim_v3"].setdefault(fs, {})[ar] = rec
            kd = D["kanizsa"]; rb_all = {}
            for var in ("inside", "inside_eroded", "r_core30"):
                al = rec["kanizsa"]["illusory-control180"]["avg|" + var]
                rb = dict(aligned=al, eligible=bool(al["ci"][0] > 0))
                if rb["eligible"]:
                    r20 = ratio(seedavg(vals, kd, "mis20", "control180", "avg", var), seedavg(vals, kd, "illusory", "control180", "avg", var))
                    rb["ratio_20"] = r20
                    rb["label"] = ("relatability-tuned" if r20["value"] <= 0.5 and r20["ci"][1] < 0.75 else
                                   "inward-notch cue" if r20["ci"][0] >= 0.5 else "intermediate")
                rb_all[var] = rb
            res["section_B"].setdefault(fs, {})[ar] = rb_all
    # ---------------- paired readout contrasts on the same configurations
    for fs in FSETS:
        for x, y in (("R1n", "R1"), ("R1n", "R4"), ("R1", "R4"), ("R1n", "R2"), ("R1n", "R3")):
            if (fs, x) not in full or (fs, y) not in full:
                continue
            sx, sy = list(full[(fs, x)].values()), list(full[(fs, y)].values()); out = {}
            for ill, test, ctrl in (("kanizsa", "illusory", "control180"), ("kanizsa", "illusory", "control90"),
                                    ("ehrenstein", "illusory", "control_end")):
                for var in ("inside", "r_core30"):
                    for size in SIZES:
                        v = seedavg(sx, D[ill], test, ctrl, size, var) - seedavg(sy, D[ill], test, ctrl, size, var)
                        out[f"{ill}|{test}-{ctrl}|{size}|{var}"] = A2.boot(v)
            res["contrasts"].setdefault(fs, {})[f"{x}-{y}"] = dict(n_seeds=[len(sx), len(sy)], **out)
    # ---------------- alternating rotation (stim_alt), analysis/analyze_alt.py definitions
    if os.path.exists(os.path.join(str(LP.STIM_ALT), "items.json")):
        cells = D["kanizsa"]["cells"]
        for (fs, ar), s3 in full.items():
            sa = load("stim_alt", fs, ar)
            seeds = sorted(set(s3) & set(sa))
            if not seeds:
                continue
            try:
                def m(cond, meas, sizes=("336", "448")):
                    def one(c, sd):
                        if cond.startswith("alt"):
                            return np.mean([sa[sd][z][meas][f"{c}_{cond}{ph}"] for z in sizes for ph in "ab"])
                        return np.mean([s3[sd][z][meas][f"{c}_{cond}"] for z in sizes])
                    return np.array([[np.mean([one(c, sd) for sd in seeds]) for c in cell] for cell in cells])
                rec = dict(seeds=seeds)
                for meas in ALT_MEAS:
                    ill, c180 = m("illusory", meas), m("control180", meas)
                    e_ill = ill - c180; r = dict(effect_illusory=A2.boot(e_ill))
                    for t in THETAS:
                        mis, alt = m(f"mis{t}", meas), m(f"alt{t}", meas)
                        r[f"effect_mis{t}"] = A2.boot(mis - c180); r[f"effect_alt{t}"] = A2.boot(alt - c180)
                        r[f"alt{t}_minus_mis{t}"] = A2.boot(alt - mis)
                        r[f"ratio_uni{t}"] = ratio(mis - c180, e_ill); r[f"ratio_alt{t}"] = ratio(alt - c180, e_ill)
                        r[f"diff{t}"] = ratio(alt - mis, e_ill); r[f"drop_ratio{t}"] = ratio(ill - alt, ill - mis)
                    rec[meas] = r
                res["stim_alt"].setdefault(fs, {})[ar] = rec
            except KeyError as e:
                res["stim_alt"].setdefault(fs, {})[ar] = dict(incomplete=f"missing score {e}")
    json.dump(res, open(os.path.join(a.out, "results_full.json"), "w"), indent=1)
    report(res)


def report(res):
    L = [f"# Full-data readouts: results ({res['created']})", "",
         "Cluster bootstrap 10,000 (cells, positions within cells), 95% CI; seeds averaged per configuration; size avg = "
         "mean of the 336 and 448 px effects. Validity: MIT1003 CC >= CC(R4) - 0.03.", "",
         "## MIT1003 held-out CC (mean of seeds)", "", "| features | " + " | ".join(ARCHS_ALL) + " |", "|---|" + "---|" * len(ARCHS_ALL)]
    for fs, rec in res["mit"].items():
        cell = lambda ar: (f"{rec[ar]['mean']:.3f} (n {rec[ar]['n']}{', invalid' if rec[ar].get('valid') is False else ''})" if ar in rec else "")
        L.append(f"| {fs} | " + " | ".join(cell(ar) for ar in ARCHS_ALL) + " |")
    for title, ill, nm, var in (("Kanizsa illusory - control180, inside AOI", "kanizsa", "illusory-control180", "inside"),
                                ("Kanizsa illusory - control180, core AOI (r_core30)", "kanizsa", "illusory-control180", "r_core30"),
                                ("Kanizsa illusory - control90, inside AOI", "kanizsa", "illusory-control90", "inside"),
                                ("Kanizsa real_filled - control180, inside AOI", "kanizsa", "real_filled-control180", "inside"),
                                ("Ehrenstein illusory - control_end, inside AOI", "ehrenstein", "illusory-control_end", "inside")):
        L += ["", f"## {title} (size avg)", "", "| features | " + " | ".join(ARCHS_ALL) + " |", "|---|" + "---|" * len(ARCHS_ALL)]
        for fs, recs in res["stim_v3"].items():
            row = []
            for ar in ARCHS_ALL:
                r = recs.get(ar, {})
                e = r.get(ill, {}).get(nm, {}).get(f"avg|{var}")
                row.append(f"{fmt(e)} {e['p_str']} (seeds {e['n_seeds']})" if e else ("incomplete" if r.get("incomplete") else ""))
            L.append(f"| {fs} | " + " | ".join(row) + " |")
    L += ["", "## Paired readout contrasts, Kanizsa illusory - control180 (size avg)", ""]
    for fs, recs in res["contrasts"].items():
        for pr, r in recs.items():
            L.append(f"- {fs} {pr}: inside {fmt(r['kanizsa|illusory-control180|avg|inside'])}; core30 "
                     f"{fmt(r['kanizsa|illusory-control180|avg|r_core30'])}; Ehrenstein {fmt(r['ehrenstein|illusory-control_end|avg|inside'])}")
    L += ["", "## Section B ratio(20), size avg (eligible readouts only)", ""]
    for fs, recs in res["section_B"].items():
        for ar, rb in recs.items():
            for var, r in rb.items():
                if r.get("eligible"):
                    L.append(f"- {fs} {ar} {var}: {fmtr(r['ratio_20'])} {r['label']}")
    if res["stim_alt"]:
        L += ["", "## Alternating rotation (stim_alt), theta 20, size avg", "",
              "| features | readout | measure | ratio_uni20 | ratio_alt20 | diff20 | drop_ratio20 |", "|---|---|---|---|---|---|---|"]
        for fs, recs in res["stim_alt"].items():
            for ar, r in recs.items():
                if "incomplete" in r:
                    L.append(f"| {fs} | {ar} | {r['incomplete']} | | | | |"); continue
                for meas in ALT_MEAS:
                    q = r[meas]
                    L.append(f"| {fs} | {ar} | {meas} | {fmtr(q['ratio_uni20'])} | {fmtr(q['ratio_alt20'])} | {fmtr(q['diff20'])} | {fmtr(q['drop_ratio20'])} |")
    open(os.path.join(a.out, "report_full.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:80]))


if __name__ == "__main__":
    main()
