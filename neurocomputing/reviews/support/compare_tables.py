"""compare_tables.py: parse the two known-value tables of REANALYSIS_RULE.md Section 1.1 and compare every number with
the result files (read only)."""
import json, os, re
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
rule = open(f"{ROOT}/neuro/reanalysis/REANALYSIS_RULE.md").read().splitlines()
F = json.load(open(f"{ROOT}/v3/gpu/full/results_full.json"))
V = json.load(open(f"{ROOT}/v3/scores/results.json"))["models"]
num = lambda s: [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s)]
FS = {"DINOv2-B block 12": "dinov2_L12", "DINOv2-B block 6": "dinov2_L6", "SigLIP-L block 24": "siglip_L24", "ShapeNet-C": "iie_shapenetc"}
bad = n = 0
for ln in rule:
    m = re.match(r"\| (DINOv2-B block 12|DINOv2-B block 6|SigLIP-L block 24|ShapeNet-C) (R1n|R1|R2|R3|R4) \|", ln)
    if not m: continue
    fs, a = FS[m.group(1)], m.group(2)
    cells = [c.strip() for c in ln.strip("|").split("|")]
    k = F["stim_v3"][fs][a]["kanizsa"]; mit = F["mit"][fs]
    e = k["illusory-control180"]["avg|r_inside"]; e90 = k["illusory-control90"]["avg|r_inside"]
    al = F["stim_alt"][fs][a]["r_inside"]
    exp = [round(mit[a]["mean"], 4)] + ([] if a == "R4" else [round(mit["R4"]["mean"] - mit[a]["mean"], 4)]) \
        + [e["mean"]] + e["ci"] + [round(x, 3) for x in e["seed_means"]] \
        + [k["illusory-control180"]["336|r_inside"]["mean"], k["illusory-control180"]["448|r_inside"]["mean"], e90["mean"]] + e90["ci"] \
        + [round(al["ratio_uni20"]["value"], 2)] + [round(x, 2) for x in al["ratio_uni20"]["ci"]] \
        + [round(al["ratio_alt20"]["value"], 2)] + [round(x, 2) for x in al["ratio_alt20"]["ci"]]
    got = num(" ".join(cells[2:]))
    n += 1
    if len(got) != len(exp) or any(abs(g - x) > 6e-4 for g, x in zip(got, exp)):
        bad += 1; print("MISMATCH", m.group(1), a, got, exp)
print(f"swap table: {n} rows checked, {bad} mismatches")
bad = n = 0
for ln in rule:
    m = re.match(r"\| (DeepGaze IIE|DeepGaze MSDB|(corner|edge|ink) sigma (\d+)) \|", ln)
    if not m: continue
    name = m.group(1) if m.group(1).startswith("DeepGaze") else f"null {m.group(2)}_s{m.group(3)}"
    r = V[name]["results"]["r_inside"]
    E = r["all"]["effects_vs_control180"]["illusory"]; L = r["all"]["drop_uni"]["20"]
    D = r["square"]["alt_minus_mis"]["20"]; R = r["square"]["drop_ratio"]["20"]
    exp = [E["mean"]] + E["ci"] + [L["mean"]] + L["ci"] + [D["mean"]] + D["ci"] + [R["value"]] + R["ci"]
    cells = ln.strip("|").split("|")
    got = num(" ".join(cells[1:]))
    n += 1
    if len(got) != len(exp) or any(abs(g - x) > 6e-4 for g, x in zip(got, exp)):
        bad += 1; print("MISMATCH", name, got, exp)
print(f"P1/P2 table: {n} rows checked, {bad} mismatches")
