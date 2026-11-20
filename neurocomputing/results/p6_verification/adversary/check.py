# Adversarial read-only check: own loader, own CR1 SE, own validity, own (d) and narrowing; wild_cluster imported read-only.
import json, os, sys
import numpy as np
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # re-analysis folder
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
sys.path.insert(0, RE)
import analyze_re as A   # read-only import; no step function is called

items = json.load(open(os.path.join(ROOT, "v2/stim_v3/items.json")))
meta = {it["config"]: (it["shape"], it["sr"], it["pol"], it["pos"]) for it in items
        if it["illusion"] == "kanizsa" and it["cond"] == "illusory"}
cl = {}
for c, (sh, sr, pol, pos) in meta.items():
    cl.setdefault((sh, sr, pol), []).append((pos, c))
keys = sorted(cl, key=str)
configs = [c for k in keys for _, c in sorted(cl[k])]
g = np.repeat(np.arange(len(keys)), 5)
assert len(configs) == 60 and len(keys) == 12

def load_cell(src, tag, ctrl="control180"):
    vals = []
    for s in (0, 1, 2):
        d = json.load(open(os.path.join(src, f"stim_v3_{tag}_s{s}.json")))
        for z in ("336", "448"):
            m = d[z]["r_inside"]
            vals.append([m[f"{c}_illusory"] - m[f"{c}_{ctrl}"] for c in configs])
    return np.mean(np.array(vals), axis=0)

def cc(src, tag):
    return float(np.mean([json.load(open(os.path.join(src, f"mit_{tag}_s{s}.json")))["CC"] for s in (0, 1, 2)]))

def cr1(d):
    G = 12; N = len(d); r = d - d.mean()
    s = np.array([r[g == k].sum() for k in range(G)])
    return np.sqrt(G / (G - 1) * (s ** 2).sum()) / N

P6 = json.load(open(os.path.join(RE, "out/primary_P6.json")))["result"]
sets = {"dinov2reg_L6": "gpu/g3", "dinov2reg_L12": "gpu/g3", "dinov2ms_L7L11": "gpu/a14", "dinov2ms_L11": "gpu/a14"}
out = {}
maxdiff = 0.0
for fs, sd in sets.items():
    res = {}
    for ar in ("R1n", "R1", "R2", "R3", "R4"):
        src = os.path.join(RE, sd, "scores")
        if fs.startswith("dinov2reg") and ar == "R4":
            src = os.path.join(ROOT, "nn/registers/full/scores")
        tag = f"{fs}_{ar}"
        d = load_cell(src, tag)
        d90 = load_cell(src, tag, "control90")
        w = A.wild_cluster(d, g)
        w90 = A.wild_cluster(d90, g)
        res[ar] = dict(E=float(d.mean()), se=float(cr1(d)), ci=w["ci"], p=w["p"], cc=cc(src, tag),
                       E90=float(d90.mean()), ci90=w90["ci"], p90=w90["p"])
    r4 = res["R4"]; sgn = int(np.sign(r4["E"])); ref_ok = r4["p"] < 0.05
    thr = round(round(r4["cc"], 4) - 0.03, 4)
    valid = [a for a in ("R1n", "R1", "R2", "R3") if round(res[a]["cc"], 4) >= thr]
    adj, rej = A.holm([res[a]["p"] for a in valid])
    for a in ("R1n", "R1", "R2", "R3"):
        res[a]["valid"] = a in valid
        res[a]["opp"] = bool((sgn > 0 and res[a]["ci"][1] < 0) or (sgn < 0 and res[a]["ci"][0] > 0))
    for a, pa, j in zip(valid, adj, rej):
        res[a]["p_holm"] = float(pa); res[a]["keep"] = bool(j and ref_ok and np.sign(res[a]["E"]) == sgn)
    # compare against JSON
    J = P6["P6"][fs]
    maxdiff = max(maxdiff, abs(J["R4"]["mean"] - r4["E"]), abs(J["R4"]["se"] - r4["se"]))
    for a in ("R1n", "R1", "R2", "R3"):
        c = J["cells"][a]
        maxdiff = max(maxdiff, abs(c["mean"] - res[a]["E"]), abs(c["se"] - res[a]["se"]), abs(c["cc"] - res[a]["cc"]),
                      abs(c["ci"][0] - res[a]["ci"][0]), abs(c["ci"][1] - res[a]["ci"][1]))
        assert c["valid"] == res[a]["valid"] and c["opposite_sign_interval"] == res[a]["opp"], (fs, a)
        if res[a]["valid"]:
            assert abs(c["p_holm"] - res[a]["p_holm"]) < 1e-12 and c["keeps_sign"] == res[a]["keep"], (fs, a)
    out[fs] = dict(thr=thr, sign=sgn, ref_ok=bool(ref_ok), valid=valid, cells=res)
print("max |diff| vs primary_P6.json:", maxdiff)
for fs, v in out.items():
    print(fs, "thr", v["thr"], "R4 sign", v["sign"], "valid", v["valid"])
    for a, r in v["cells"].items():
        print("   %-3s E %+.4f [%+.4f, %+.4f] p %.4f holm %s cc %.4f opp %s keep %s | E90 %+.4f [%+.4f, %+.4f] p90 %.4f" % (
            a, r["E"], r["ci"][0], r["ci"][1], r["p"], r.get("p_holm"), r["cc"], r.get("opp"), r.get("keep"),
            r["E90"], r["ci90"][0], r["ci90"][1], r["p90"]))
# narrowing: feature sets with no valid opposite-sign interval (P3 sets included)
P3 = P6["P3"]
p3sets = {}
for k, c in P3["cells"].items():
    fs = k.split("|")[0]; p3sets.setdefault(fs, []).append(bool(c["opposite_sign_interval"]))
covered = [fs for fs, l in p3sets.items() if not any(l)] + [fs for fs, v in out.items() if not any(v["cells"][a]["opp"] for a in v["valid"])]
print("sets without a valid opposite-sign interval:", covered, len(covered))
print("P3 reference:", {k: round(v["mean"], 4) for k, v in P3["reference"].items()})
json.dump(out, open(os.path.join(os.path.dirname(__file__), "check_out.json"), "w"), indent=1, default=float)
