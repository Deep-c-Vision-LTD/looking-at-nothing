"""
verify_known.py (independent reviewer): re-read the result files and print every number that
REANALYSIS_RULE.md Section 1.1 items 2 and 4 and Section 5.4 cite, for a side-by-side check.
Reads JSON only (no scoring, no new statistic on real scores; percentile-approximate |t| is
arithmetic on published intervals, exactly as the rule does).
"""
import json, os

ROOT = "/Users/hamitsoyel/Downloads/illusion_research"
F = json.load(open(os.path.join(ROOT, "v3/gpu/full/results_full.json")))
FS = [("dinov2_L12", "DINOv2-B block 12"), ("dinov2_L6", "DINOv2-B block 6"),
      ("siglip_L24", "SigLIP-L block 24"), ("iie_shapenetc", "ShapeNet-C")]
AR = ["R1n", "R1", "R2", "R3", "R4"]


def tp(m, ci):
    return abs(m) / ((ci[1] - ci[0]) / 3.92)


rows, gaps, seedci = [], [], []
for fs, name in FS:
    mit = F["mit"][fs]
    for a in AR:
        k = F["stim_v3"][fs][a]["kanizsa"]
        e = k["illusory-control180"]["avg|r_inside"]
        e336 = k["illusory-control180"]["336|r_inside"]["mean"]
        e448 = k["illusory-control180"]["448|r_inside"]["mean"]
        e90 = k["illusory-control90"]["avg|r_inside"]
        al = F["stim_alt"][fs][a]["r_inside"]
        u, A = al["ratio_uni20"], al["ratio_alt20"]
        cc = mit[a]["mean"]; gap = mit["R4"]["mean"] - cc
        if a != "R4":
            gaps.append((round(gap, 4), name, a, mit[a]["valid"]))
        for s, sc in zip(e["seed_means"], e["seed_cis"]):
            seedci.append((name, a, mit[a]["valid"], s, sc, (sc[0] > 0) or (sc[1] < 0)))
        print(f"{name:18s} {a:3s} valid={mit[a]['valid']!s:5s} CC {cc:.4f} gap {gap:.4f} thr {mit[a]['threshold']:.4f} | "
              f"E {e['mean']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] t~{tp(e['mean'], e['ci']):.2f} seeds "
              f"{', '.join(f'{x:+.3f}' for x in e['seed_means'])} | 336 {e336:+.4f} 448 {e448:+.4f} | "
              f"E90 {e90['mean']:+.4f} [{e90['ci'][0]:+.4f}, {e90['ci'][1]:+.4f}] | U20 {u['value']:.2f} "
              f"[{u['ci'][0]:.2f}, {u['ci'][1]:.2f}] A20 {A['value']:.2f} [{A['ci'][0]:.2f}, {A['ci'][1]:.2f}]")
print()
print("gaps sorted:", sorted(gaps))
gs = sorted(set(g for g, *_ in gaps))
print("distinct gap values in (0, 0.06]:", len([g for g in gs if g <= 0.06]), gs)
nval = [x for x in seedci if x[2]]
print(f"per-seed intervals excluding 0: valid non-R4 {sum(x[5] for x in nval if x[1] != 'R4')}/{sum(1 for x in nval if x[1] != 'R4')}; "
      f"all cells {sum(x[5] for x in seedci)}/{len(seedci)}")
print("per-seed intervals including 0:", [(x[0], x[1], x[3], x[4]) for x in seedci if not x[5]])
# signs vs R4
for fs, name in FS:
    r4 = F["stim_v3"][fs]["R4"]["kanizsa"]["illusory-control180"]["avg|r_inside"]["mean"]
    for a in AR:
        e = F["stim_v3"][fs][a]["kanizsa"]["illusory-control180"]["avg|r_inside"]
        e90 = F["stim_v3"][fs][a]["kanizsa"]["illusory-control90"]["avg|r_inside"]["mean"]
        same = (e["mean"] > 0) == (r4 > 0)
        excl = e["ci"][0] > 0 or e["ci"][1] < 0
        sseeds = sum((s > 0) == (r4 > 0) for s in e["seed_means"])
        if not (same and excl and sseeds == 3 and (e90 > 0) == (e["mean"] > 0)):
            print("SIGN CHECK FAIL", name, a)
print("sign checks done")

# P4 table: percentile-approximate |t| of both denominators and the point differences
print()
for fs, name in FS:
    r4 = F["stim_v3"][fs]["R4"]["kanizsa"]["illusory-control180"]["avg|r_inside"]
    for a in ["R1n", "R1", "R2", "R3"]:
        if not F["mit"][fs][a]["valid"]:
            continue
        e = F["stim_v3"][fs][a]["kanizsa"]["illusory-control180"]["avg|r_inside"]
        for m in ("ratio_uni20", "ratio_alt20"):
            x = F["stim_alt"][fs][a]["r_inside"][m]["value"]; y = F["stim_alt"][fs]["R4"]["r_inside"][m]["value"]
            print(f"P4 {name:18s} {a:3s} {m}: diff {x - y:+.3f}  |t| {tp(e['mean'], e['ci']):.2f}, {tp(r4['mean'], r4['ci']):.2f}")
