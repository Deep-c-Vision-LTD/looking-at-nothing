"""
neuro/reanalysis/explore/ttr/review/review_stats.py: independent recomputation of selected numbers in
results/explore/ttr_results.json, written for the adversarial review of the test-time-registers (TTR) code
(explore/ttr/REVIEW.md, 18 November 2026). EXPLORATORY check; decides nothing.

Independence: this file does NOT import ttr_common, ttr_analyze, excommon or mapsrc. It reads the raw per-display
score files in scores_ttr/, the published per-image files, and the stored per-image norm table, and recomputes from
first principles:
  - configurations and clusters parsed from the display names (shape x support ratio x polarity, 12 clusters;
    clusters ordered by the sorted (shape, sr, polarity) tuple, as the rule's "sorted by the string of the key");
  - per-configuration values: seeds averaged, then the 336 and 448 px values averaged;
  - WCU interval and WCR p re-implemented from the definitions in the rule (analyze_re.py docstring, Section 4.1):
    Webb six-point weights, 9,999 draws from numpy RandomState(2026).randint(0, 6, (9999, 12)), CR1 standard error,
    percentile-t interval [mu - q(0.975) se, mu - q(0.025) se], WCR p = 2 (1 + min(#t* >= t, #t* <= t)) / (B + 1);
  - A20 = mean(alt20 - c180) / mean(ill - c180), alt20 = mean of phases a and b (point estimates only);
  - held-out MIT1003 CC from the stored per-image CCs;
  - SigLIP-L block 4: ttr against reg0 display by display (no register neuron lies below block 4);
  - gate K percentiles from the stored per-image ratio table;
  - gate P spot checks against the published per-image files.
analyze_re is imported ONLY for require_public() (rule Section 8.1), which runs first.
Run (one process): nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    $PY -B -I explore/ttr/review/review_stats.py      ($PY: the project's Python 3.12 interpreter)
Output: explore/ttr/review/review_stats_output.json (and the same lines on stdout).
"""
import json, math, os, sys
sys.dont_write_bytecode = True
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
sys.path.insert(0, RE)
import analyze_re                                                # noqa: E402  (only for the guard)
analyze_re.require_public(RE)                                    # rule Section 8.1, before anything is computed
import numpy as np                                               # noqa: E402

SC = os.path.join(RE, "scores_ttr")
OUTP = os.path.join(RE, "explore", "ttr", "review", "review_stats_output.json")
V3C = ("illusory", "control180", "control90", "real", "real_filled", "mis10", "mis20", "mis30")
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
LINES, OUT = [], {}


def say(s):
    print(s, flush=True)
    LINES.append(s)


def split_name(nm):
    for c in sorted(V3C, key=len, reverse=True) + [f"alt{t}{p}" for t in (10, 20, 30) for p in "ab"]:
        if nm.endswith("_" + c):
            return nm[: -len(c) - 1], c
    raise ValueError(nm)


def cluster_of(cfg):
    shape, sr, pol, _ = cfg.split("_")
    return (shape, int(sr[2:]), pol)


def load(arm, tag, seeds, stimset):
    """{name: value}: mean over seeds and sizes (equal weights; = seeds first, then sizes)."""
    acc = {}
    for sd in seeds:
        j = json.load(open(os.path.join(SC, arm, f"{stimset}_{tag}_s{sd}.json")))
        for sz in ("336", "448"):
            for nm, v in j[sz]["r_inside"].items():
                acc.setdefault(nm, []).append(v)
    n = {len(v) for v in acc.values()}
    assert n == {2 * len(seeds)}, n
    return {k: float(np.mean(v)) for k, v in acc.items()}


def unit(arm, u, with_alt=False):
    tag, seeds = (u.replace("|", "_"), (0, 1, 2)) if "|" in u else (u, (0,))
    d = load(arm, tag, seeds, "stim_v3")
    if with_alt:
        d.update(load(arm, tag, seeds, "stim_alt"))
    return d


def configs(vals):
    cf = sorted({split_name(n)[0] for n in vals if split_name(n)[1] == "illusory"})
    assert len(cf) == 60, len(cf)
    cl = sorted({cluster_of(c) for c in cf})
    assert len(cl) == 12
    g = np.array([cl.index(cluster_of(c)) for c in cf])
    assert np.all(np.bincount(g) == 5)
    return cf, g


def contrast(vals, cf, a, b):
    def v(c, cond):
        if cond == "alt20":
            return 0.5 * (vals[f"{c}_alt20a"] + vals[f"{c}_alt20b"])
        return vals[f"{c}_{cond}"]
    return np.array([v(c, a) - v(c, b) for c in cf])


def cr1(d, g):
    G, N = g.max() + 1, d.size
    S = np.bincount(g, weights=d - d.mean(), minlength=G)
    return math.sqrt(G / (G - 1) * float((S ** 2).sum())) / N


def wild(d, g, B=9999, seed=2026):
    """WCU interval and WCR p, written from the rule's definitions (not from analyze_re's code)."""
    G, N = int(g.max()) + 1, d.size
    n = np.bincount(g, minlength=G).astype(float)
    W = WEBB[np.random.RandomState(seed).randint(0, 6, (B, G))]
    mu, se = float(d.mean()), cr1(d, g)
    t = mu / se

    def tstar(resid):
        Eg = np.bincount(g, weights=resid, minlength=G)          # cluster sums of the residuals
        dm = (W * Eg).sum(1) / N                                 # mean* minus the centre
        S = W * Eg - n * dm[:, None]                             # cluster sums of the bootstrap-sample residuals
        sst = np.sqrt(G / (G - 1) * (S ** 2).sum(1)) / N
        return dm / sst
    tu = tstar(d - mu)                                           # unrestricted (WCU)
    q025, q975 = np.quantile(tu, [0.025, 0.975])
    t0 = tstar(d)                                                # H0: mu = 0 imposed (WCR)
    p = min(1.0, 2 * (1 + min(int((t0 >= t - 1e-12).sum()), int((t0 <= t + 1e-12).sum()))) / (B + 1))
    return dict(mean=mu, ci=[mu - q975 * se, mu - q025 * se], p=p)


def cmp(label, mine, theirs, tol=1e-9):
    diff = max(abs(a - b) for a, b in zip(mine, theirs))
    ok = diff <= tol
    say(f"{'MATCH' if ok else 'DIFF '}  {label}: mine {['%+.6f' % x for x in mine]} | ttr_results {['%+.6f' % x for x in theirs]}"
        f" | max |diff| {diff:.2e}")
    OUT[label] = dict(mine=list(mine), ttr_results=list(theirs), max_abs_diff=diff, match=bool(ok))
    return ok


def main():
    R = json.load(open(os.path.join(RE, "results", "explore", "ttr_results.json")))["result"]
    U = R["units"]
    say("review_stats.py: independent recomputation (numbers in ttr_results.json)")

    # 1. SigLIP-L block 24 swap R4: E plain and the paired change ttr - plain
    p4, t4 = unit("plain", "siglip_L24|R4", True), unit("ttr", "siglip_L24|R4", True)
    cf, g = configs(p4)
    e = wild(contrast(p4, cf, "illusory", "control180"), g)
    j = U["siglip_L24|R4"]["plain"]["E"]
    cmp("1a siglip_L24 R4 plain E (mean, WCU lo, hi, WCR p)", [e["mean"], *e["ci"], e["p"]], [j["mean"], *j["ci"], j["p"]])
    dE = contrast(t4, cf, "illusory", "control180") - contrast(p4, cf, "illusory", "control180")
    e = wild(dE, g)
    j = U["siglip_L24|R4"]["ttr_minus_plain"]["dE"]
    cmp("1b siglip_L24 R4 ttr - plain dE (mean, WCU lo, hi, WCR p)", [e["mean"], *e["ci"], e["p"]], [j["mean"], *j["ci"], j["p"]])

    # 2. CLIP-L block 24 sweep: ttr - plain
    pc, tc = unit("plain", "clip_L24"), unit("ttr", "clip_L24")
    cf2, g2 = configs(pc)
    e = wild(contrast(tc, cf2, "illusory", "control180") - contrast(pc, cf2, "illusory", "control180"), g2)
    j = U["clip_L24"]["ttr_minus_plain"]["dE"]
    cmp("2 clip_L24 ttr - plain dE (mean, WCU lo, hi)", [e["mean"], *e["ci"]], [j["mean"], *j["ci"]])

    # 3. SigLIP-L block 24 swap R3 under ttr (the interval that includes 0)
    t3 = unit("ttr", "siglip_L24|R3", True)
    e = wild(contrast(t3, cf, "illusory", "control180"), g)
    j = U["siglip_L24|R3"]["ttr"]["E"]
    cmp("3 siglip_L24 R3 ttr E (mean, WCU lo, hi, WCR p)", [e["mean"], *e["ci"], e["p"]], [j["mean"], *j["ci"], j["p"]])

    # 4. CLIP-L block 16 under ttr
    e = wild(contrast(unit("ttr", "clip_L16"), cf2, "illusory", "control180"), g2)
    j = U["clip_L16"]["ttr"]["E"]
    cmp("4 clip_L16 ttr E (mean, WCU lo, hi)", [e["mean"], *e["ci"]], [j["mean"], *j["ci"]])

    # 5. P4-like (R2 - R4) A20 under ttr and plain: point estimates
    for arm, t4x in (("ttr", t4), ("plain", p4)):
        t2 = unit(arm, "siglip_L24|R2", True)
        a2 = contrast(t2, cf, "alt20", "control180").mean() / contrast(t2, cf, "illusory", "control180").mean()
        a4 = contrast(t4x, cf, "alt20", "control180").mean() / contrast(t4x, cf, "illusory", "control180").mean()
        j = R["swap"][f"P4_like_{arm}"]["A20"]
        cmp(f"5 P4-like {arm} A20 (R2, R4, R2 - R4)", [a2, a4, a2 - a4], [j["ratio_1"], j["ratio_2"], j["diff"]])

    # 6. held-out MIT1003 CC change for R4 (mean of the seed means of the stored per-image CCs)
    cc = {}
    for arm in ("plain", "ttr"):
        cc[arm] = float(np.mean([np.mean(json.load(open(os.path.join(SC, arm, f"mit_siglip_L24_R4_s{s}.json")))["cc_per_image"])
                                 for s in (0, 1, 2)]))
    j = R["accuracy"]["siglip_L24|R4"]
    cmp("6 MIT1003 CC R4 (plain, ttr, ttr - plain)", [cc["plain"], cc["ttr"], cc["ttr"] - cc["plain"]],
        [j["plain"]["CC"], j["ttr"]["CC"], j["dCC_ttr"]], tol=1e-12)

    # 7. SigLIP-L block 4: ttr must equal reg0 display by display (no register neuron below block 4)
    md, mdp, n = 0.0, 0.0, 0
    for st in ("stim_v3",):
        a = json.load(open(os.path.join(SC, "ttr", f"{st}_siglip_L4_s0.json")))
        b = json.load(open(os.path.join(SC, "reg0", f"{st}_siglip_L4_s0.json")))
        c = json.load(open(os.path.join(SC, "plain", f"{st}_siglip_L4_s0.json")))
        for sz in ("336", "448"):
            for nm, v in a[sz]["r_inside"].items():
                md = max(md, abs(v - b[sz]["r_inside"][nm])); mdp = max(mdp, abs(v - c[sz]["r_inside"][nm])); n += 1
    say(f"{'MATCH' if md == 0 else 'DIFF '}  7 siglip_L4 ttr vs reg0 on stim_v3: {n} values, max |diff| {md:.3g} "
        f"(ttr vs plain {mdp:.3g})")
    OUT["7 siglip_L4 ttr vs reg0"] = dict(n=n, max_abs_diff=md, ttr_vs_plain=mdp)

    # 8. gate K percentiles from the stored per-image ratios
    pi = json.load(open(os.path.join(SC, "norms", "gateK_per_image.json")))
    gk = R["gate_K"]
    for bb, b in (("siglip", 24), ("clip", 16), ("clip", 12)):
        mine = [float(np.percentile(pi[f"{bb}|kanizsa|336|{arm}|{b}"]["ratio"], 90)) for arm in ("plain", "ttr")]
        theirs = [gk[bb]["summary"][f"kanizsa|336|{arm}|{b}"]["p90"] for arm in ("plain", "ttr")]
        cmp(f"8 gate K {bb} block {b} Kanizsa 336 p90 (plain, ttr)", mine, theirs, tol=1e-12)
    mine = [float(np.mean(pi[f"clip|kanizsa|336|{arm}|16"]["share"])) for arm in ("plain", "ttr")]
    theirs = [gk["clip"]["summary"][f"kanizsa|336|{arm}|16"]["share_mean"] for arm in ("plain", "ttr")]
    cmp("8b gate K clip block 16 Kanizsa 336 share > 2x median (plain, ttr)", mine, theirs, tol=1e-12)

    # 9. gate P spot checks: plain arm against the published per-image files
    pub = json.load(open(os.path.join(ROOT, "nn", "results", "raw_v3maps", "clip_L16_s336.json")))
    mine = json.load(open(os.path.join(SC, "plain", "stim_v3_clip_L16_s0.json")))["336"]["r_inside"]
    d = max(abs(v - pub[k]["r_inside"]) for k, v in mine.items())
    say(f"{'MATCH' if d <= 1e-4 else 'DIFF '}  9a gate P clip_L16 stim_v3 336: {len(mine)} values, max |diff| {d:.3g}")
    OUT["9a gate P clip_L16 stim_v3 336"] = dict(n=len(mine), max_abs_diff=d)
    pub = json.load(open(os.path.join(ROOT, "v3", "gpu", "full", "scores", "stim_alt_siglip_L24_R4_s1.json")))["448"]["r_inside"]
    mine = json.load(open(os.path.join(SC, "plain", "stim_alt_siglip_L24_R4_s1.json")))["448"]["r_inside"]
    d = max(abs(v - pub[k]) for k, v in mine.items())
    say(f"{'MATCH' if d <= 1e-4 else 'DIFF '}  9b gate P siglip_L24 R4 s1 stim_alt 448: {len(mine)} values, max |diff| {d:.3g}")
    OUT["9b gate P siglip_L24 R4 s1 stim_alt 448"] = dict(n=len(mine), max_abs_diff=d)

    # 10. selection images: disjoint from the 24 A15(a) MIT1003 images, no display name
    sel = set(json.load(open(os.path.join(RE, "results", "explore", "ttr", "TTR_select_siglip.json")))["result"]["selection_images"])
    sel2 = set(json.load(open(os.path.join(RE, "results", "explore", "ttr", "TTR_select_clip.json")))["result"]["selection_images"])
    a15 = set(json.load(open(os.path.join(RE, "results", "explore", "A15a_token_norms.json")))["result"]["images"]["mit1003"])
    bad = [s for s in sel | sel2 if s.endswith(".png") or "_sr" in s]
    say(f"{'MATCH' if (sel == sel2 and not (sel & a15) and not bad and len(sel) == 100) else 'DIFF '}  10 selection images: "
        f"{len(sel)} (same list for both backbones: {sel == sel2}); overlap with the 24 A15(a) MIT1003 images "
        f"{len(sel & a15)}; display-like names {len(bad)}")
    OUT["10 selection images"] = dict(n=len(sel), same_both=sel == sel2, overlap_a15=len(sel & a15), display_like=len(bad))

    with open(OUTP + ".tmp", "w") as f:
        json.dump(dict(lines=LINES, checks=OUT), f, indent=1)
    os.replace(OUTP + ".tmp", OUTP)
    say(f"wrote {OUTP}")


if __name__ == "__main__":
    main()
