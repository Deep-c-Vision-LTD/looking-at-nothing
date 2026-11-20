"""
v_ttr.py: independent recomputation of a sample of the test-time-registers results (rule Section 6 item 12;
`explore/ttr/`, results/explore/ttr_results.json) from the per-display score files in scores_ttr/ and the per-image
norm file. Written 19 November 2026 for the verification of the remaining CPU steps; chosen to differ from the numbers
the TTR review (explore/ttr/REVIEW.md) already recomputed. Imports no project module except analyze_re for
require_public(), which runs first. numpy only.
Recomputed (rule 2.2 aggregation: seeds averaged per display, then 336 and 448 px averaged; WCU 12 clusters):
  - E (ttr arm) of SigLIP-L 24 R2 and of every sweep unit; ttr - plain dE and dE_real for every unit;
  - per-seed E of SigLIP-L 24 R3 under ttr (P3-like seed count), accuracy analogue from the MIT1003 files;
  - A20 ratios of R2 and R4 under ttr (P4-like point values);
  - SigLIP-L block 4: ttr against reg0 display by display (stim_v3; M2);
  - held-out MIT1003 CC change (ttr - plain) of the 15 units;
  - gate K 90th percentiles and the number of A15(a) displays with a token above twice the median (M5).
Output: neuro/verify_rescore/cpu_remaining/v_ttr.json
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V                                                       # noqa: E402

N_HASHED = V.guard()                                                      # require_public() first
import json                                                               # noqa: E402
import numpy as np                                                        # noqa: E402

RE = V.RE
SC = os.path.join(RE, "scores_ttr")
OUT = os.path.join(V.HERE, "v_ttr.json")
SWEEP = [f"clip_L{L}" for L in (12, 16, 20, 24)] + [f"siglip_L{L}" for L in (4, 8, 12, 16, 20, 24)]
SWAP = [f"siglip_L24|{a}" for a in ("R1n", "R1", "R2", "R3", "R4")]
KEYS, GRID = V.design()


def files(unit):
    if "|" in unit:
        fs, ar = unit.split("|")
        return [f"{fs}_{ar}_s{s}" for s in (0, 1, 2)]
    return [f"{unit}_s0"]


def scores(arm, unit, seeds=None, sets=("stim_v3", "stim_alt")):
    tags = files(unit) if seeds is None else [files(unit)[s] for s in seeds]
    out = {}
    for st in sets:
        per = [V.jl(os.path.join(SC, arm, f"{st}_{t}.json")) for t in tags]
        for name in per[0]["336"]["r_inside"]:
            out[name] = float(np.mean([np.mean([p[z]["r_inside"][name] for p in per]) for z in ("336", "448")]))
    return out


def r(sc, c, cond):
    if cond.startswith("alt"):
        return 0.5 * (sc[f"{c}_{cond}a"] + sc[f"{c}_{cond}b"])
    return sc[f"{c}_{cond}"]


def E(sc, test="illusory"):
    return V.per_config(GRID, lambda c: r(sc, c, test) - r(sc, c, "control180"))


def cmpw(w, st):
    return max(abs(w["mean"] - st["mean"]), abs(w["ci"][0] - st["ci"][0]), abs(w["ci"][1] - st["ci"][1]),
               abs(w["p"] - st["p"]))


def main():
    t0 = time.time()
    T = V.jl(os.path.join(RE, "results", "explore", "ttr_results.json"))["result"]
    res = dict(n_hashed_verified=N_HASHED, units={})
    dmax = 0.0
    for u in SWEEP + SWAP:
        pl, tt = scores("plain", u), scores("ttr", u)
        e_p, e_t = E(pl), E(tt)
        er_p, er_t = E(pl, "real"), E(tt, "real")
        w = dict(E_plain=V.wild(e_p), E_ttr=V.wild(e_t), dE=V.wild(e_t - e_p), dE_real=V.wild(er_t - er_p))
        st = T["units"][u]
        d = max(cmpw(w["E_plain"], st["plain"]["E"]), cmpw(w["E_ttr"], st["ttr"]["E"]),
                cmpw(w["dE"], st["ttr_minus_plain"]["dE"]), cmpw(w["dE_real"], st["ttr_minus_plain"]["dE_real"]))
        dmax = max(dmax, d)
        side = lambda ci: "above 0" if ci[0] > 0 else ("below 0" if ci[1] < 0 else "includes 0")
        res["units"][u] = dict(E_plain=w["E_plain"]["mean"], E_plain_ci=w["E_plain"]["ci"], E_plain_side=side(w["E_plain"]["ci"]),
                               E_ttr=w["E_ttr"]["mean"], E_ttr_ci=w["E_ttr"]["ci"], E_ttr_p=w["E_ttr"]["p"],
                               E_ttr_side=side(w["E_ttr"]["ci"]),
                               dE=w["dE"]["mean"], dE_ci=w["dE"]["ci"], dE_real=w["dE_real"]["mean"],
                               dE_real_ci=w["dE_real"]["ci"], max_abs_diff_vs_stored=d)
    res["units_max_abs_diff_vs_stored"] = dmax

    # P3-like seeds of R3 and R2 under ttr; accuracy analogue
    seeds = {}
    for a in ("R2", "R3"):
        seeds[a] = [float(E(scores("ttr", f"siglip_L24|{a}", seeds=(s,))).mean()) for s in (0, 1, 2)]
    res["ttr_seed_point_estimates"] = seeds

    def mitcc(arm, unit):
        return [float(np.mean(V.jl(os.path.join(SC, arm, f"mit_{t}.json"))["cc_per_image"])) for t in files(unit)]
    acc = {}
    for arm in ("plain", "ttr"):
        r4 = np.mean(mitcc(arm, "siglip_L24|R4"))
        thr = round(round(r4, 4) - 0.03, 4)
        acc[arm] = {a: bool(round(float(np.mean(mitcc(arm, f"siglip_L24|{a}"))), 4) >= thr) for a in ("R1n", "R1", "R2", "R3")}
    res["accuracy_analogue_valid"] = acc
    res["mit_cc_change_ttr_minus_plain"] = {u: float(np.mean(mitcc("ttr", u)) - np.mean(mitcc("plain", u)))
                                            for u in SWEEP + SWAP}
    ch = list(res["mit_cc_change_ttr_minus_plain"].values())
    res["mit_cc_change_range"] = [min(ch), max(ch)]

    # P4-like point values (A20 = E(alt20) / E(illusory), ratio of means)
    p4 = {}
    for arm in ("plain", "ttr"):
        a2 = scores(arm, "siglip_L24|R2"); a4 = scores(arm, "siglip_L24|R4")
        A2 = float(E(a2, "alt20").mean() / E(a2).mean()); A4 = float(E(a4, "alt20").mean() / E(a4).mean())
        p4[arm] = dict(A20_R2=A2, A20_R4=A4, diff=A2 - A4, stored=T["swap"][f"P4_like_{arm}"]["A20"]["diff"])
    res["P4_like_A20"] = p4

    # M2: SigLIP-L block 4, ttr against reg0 (stim_v3, both sizes)
    a = V.jl(os.path.join(SC, "ttr", "stim_v3_siglip_L4_s0.json")); b = V.jl(os.path.join(SC, "reg0", "stim_v3_siglip_L4_s0.json"))
    vals = [(a[z]["r_inside"][n], b[z]["r_inside"][n]) for z in ("336", "448") for n in a[z]["r_inside"]]
    res["siglip_L4_ttr_vs_reg0"] = dict(n=len(vals), max_abs_diff=float(max(abs(x - y) for x, y in vals)))

    # gate K and M5 from the per-image norm file
    K = V.jl(os.path.join(SC, "norms", "gateK_per_image.json"))
    gk = {}
    for bb, blocks in (("clip", (12, 16, 20, 24)), ("siglip", (4, 8, 12, 16, 20, 24))):
        for L in blocks:
            for z in ("336", "448"):
                rec = {}
                for arm in ("plain", "ttr"):
                    k = f"{bb}|kanizsa|{z}|{arm}|{L}"
                    if k not in K:
                        continue
                    rec[arm + "_p90"] = float(np.percentile(K[k]["ratio"], 90))
                    rec[arm + "_n_displays_count_gt0"] = int(np.sum(np.array(K[k]["count"]) > 0))
                    rec[arm + "_n"] = len(K[k]["ratio"])
                gk[f"{bb} {L} {z}"] = rec
    res["gate_K"] = gk
    res["seconds"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:9000])


if __name__ == "__main__":
    main()
