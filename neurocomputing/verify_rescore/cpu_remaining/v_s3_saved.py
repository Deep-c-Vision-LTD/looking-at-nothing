"""
v_s3_saved.py: independent recomputation of S3 (`explore/explore_saved.py`, rule Section 6 items 1, 2, 3, 5, 17, 18) from
the raw per-display score files. Written 19 November 2026 for the verification of the remaining CPU steps. Imports no
project module except analyze_re for require_public() (rule Section 8.1), which runs first. numpy only. Exploratory
verification; nothing here is decision-bearing.

Checks (definitions from rule Sections 2.3 and 4; every quantity recomputed here, then compared with
results/explore/explore_saved.json):
  item 1  per-size E (WCU, WCR p), L20, U20, A20, D20 point values for the 20 swap cells and 2 registers R4 cells;
          count of per-size E intervals excluding 0.
  item 2  E90 = r(illusory) - r(control90) for the 20 swap cells, IIE, MSDB and the 15 nulls; same-sign and
          interval-excludes-0 counts; E90 / E180 point values.
  item 3  per-seed E (seed fixed, sizes averaged) with WCU and with the published two-stage percentile interval for the
          20 swap cells; per-seed counts on R4's side (valid = the 10 P3 cells; overall = 20 cells); validity recomputed
          from the stored MIT1003 CCs (rule 2.7 rounding).
  item 5  f_core30 illusory minus control180 for every unit with a stored f_core30.
  item 17 point values of the below-criterion cells and of SigLIP-L block 24 R3 minus R4 (U20, A20), and the R3
          denominator |t|.
  item 18 E per support ratio (4 clusters, all 1,296 vectors) for the 18 sweep readouts; counts.
  text    rule 1.1 item 5 sweep E of DINOv2-B block 10 and CLIP-L block 12 (D9 text observation).
Output: neuro/verify_rescore/cpu_remaining/v_s3_saved.json
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V                                                       # noqa: E402

N_HASHED = V.guard()                                                      # require_public() first
import json                                                               # noqa: E402
import numpy as np                                                        # noqa: E402

ROOT, RE = V.ROOT, V.RE
OUT = os.path.join(V.HERE, "v_s3_saved.json")
FSETS = ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")
ARCHS = ("R1n", "R1", "R2", "R3", "R4")
P3_CELLS = [("dinov2_L12", "R1n"), ("dinov2_L12", "R1"), ("dinov2_L12", "R2"), ("dinov2_L12", "R3"),
            ("dinov2_L6", "R2"), ("dinov2_L6", "R3"), ("siglip_L24", "R2"), ("siglip_L24", "R3"),
            ("iie_shapenetc", "R2"), ("iie_shapenetc", "R3")]
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
SWEEP = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
SWAP_DIR = os.path.join(ROOT, "v3", "gpu", "full", "scores")
REG_DIR = os.path.join(ROOT, "nn", "registers", "full", "scores")


# ------------------------------------------------------------------ loaders (read only)
def model(tag):
    d = dict(V.jl(os.path.join(ROOT, "v3", "scores", "raw_v3maps", tag + ".json")))
    d.update(V.jl(os.path.join(ROOT, "v3", "scores", "raw", tag + ".json")))
    return d                                                              # {name: {measure: value}}


NULLD = V.jl(os.path.join(ROOT, "v3", "scores", "raw_nulls.json"))


def head(fs, ar, seeds=(0, 1, 2), sizes=("336", "448"), meas="r_inside"):
    """{name: {meas: value}}: seeds averaged per display, then the sizes averaged (rule 2.2)."""
    dr = REG_DIR if fs.startswith("dinov2reg") else SWAP_DIR
    acc = {}
    for st in ("stim_v3", "stim_alt"):
        per = [V.jl(os.path.join(dr, f"{st}_{fs}_{ar}_s{s}.json")) for s in seeds]
        for z in sizes:
            for name in per[0][z][meas]:
                acc.setdefault(name, {}).setdefault(z, []).append(np.mean([p[z][meas][name] for p in per]))
    return {n: {meas: float(np.mean([np.mean(v[z]) for z in sizes]))} for n, v in acc.items()}


def sweep(bb, L, meas="r_inside"):
    out = {}
    for z in ("336", "448"):
        for sub_v3, sub_alt in ((("v3", "scores", "raw_v3maps"), ("v3", "scores", "raw")),
                                (("nn", "results", "raw_v3maps"), ("nn", "results", "raw_alt"))):
            p = os.path.join(ROOT, *sub_v3, f"{bb}_L{L}_s{z}.json")
            if os.path.exists(p):
                d = dict(V.jl(p))
                d.update(V.jl(os.path.join(ROOT, *sub_alt, f"{bb}_L{L}_s{z}.json")))
                for n, v in d.items():
                    out.setdefault(n, []).append(v[meas])
                break
    assert all(len(v) == 2 for v in out.values()), (bb, L)
    return {n: {meas: float(np.mean(v))} for n, v in out.items()}


def r(sc, c, cond, meas="r_inside"):
    if cond.startswith("alt"):
        return 0.5 * (sc[f"{c}_{cond}a"][meas] + sc[f"{c}_{cond}b"][meas])
    return sc[f"{c}_{cond}"][meas]


KEYS, GRID = V.design()


def arr(sc, f):
    return V.per_config(GRID, lambda c: f(sc, c))


def E(sc, test="illusory", ctl="control180", meas="r_inside"):
    return arr(sc, lambda s, c: r(s, c, test, meas) - r(s, c, ctl, meas))


def Lth(sc, th):
    return arr(sc, lambda s, c: r(s, c, "illusory") - r(s, c, f"mis{th}"))


def ratio(num, den):
    return float(num.mean() / den.mean())


def wild_cmp(mine, st):
    return max(abs(mine["mean"] - st["mean"]), abs(mine["ci"][0] - st["ci"][0]), abs(mine["ci"][1] - st["ci"][1]),
               abs(mine["p"] - st["p"]))


def main():
    t0 = time.time()
    S = V.jl(os.path.join(RE, "results", "explore", "explore_saved.json"))["result"]
    res = dict(n_hashed_verified=N_HASHED)

    # ---------------------------------------------------------- item 1 per size
    units1 = [(f, a) for f in FSETS for a in ARCHS] + [("dinov2reg_L12", "R4"), ("dinov2reg_L6", "R4")]
    d1, n_excl, n_tot, samples = 0.0, 0, 0, {}
    for f, a in units1:
        for z in ("336", "448"):
            sc = head(f, a, sizes=(z,))
            e = V.wild(E(sc)); l20 = V.wild(Lth(sc, 20))
            st = S["item1_A4_1_per_size"][f"{f}|{a}"][f"px{z}"]
            d1 = max(d1, wild_cmp(e, st["E"]), wild_cmp(l20, st["L20"]))
            u = ratio(E(sc, "mis20"), E(sc)); aa = ratio(E(sc, "alt20"), E(sc))
            d1 = max(d1, abs(u - st["U20"]["value"]), abs(aa - st["A20"]["value"]))
            if st["D20"].get("value") is not None:
                dd = ratio(arr(sc, lambda s, c: r(s, c, "illusory") - r(s, c, "alt20")), Lth(sc, 20))
                d1 = max(d1, abs(dd - st["D20"]["value"]))
            n_tot += 1
            n_excl += int(e["ci"][0] > 0 or e["ci"][1] < 0)
            if (f, a, z) in (("siglip_L24", "R4", "448"), ("dinov2reg_L12", "R4", "336"), ("dinov2_L6", "R1n", "336")):
                samples[f"{f}|{a} {z}px E"] = e
    res["item1"] = dict(max_abs_diff_vs_stored=d1, per_size_E_intervals_excluding_0=f"{n_excl} of {n_tot}",
                        samples=samples)

    # ---------------------------------------------------------- item 2 control90
    units2 = {f"{f}|{a}": head(f, a) for f in FSETS for a in ARCHS}
    units2["DeepGaze IIE"] = model("DeepGaze_IIE"); units2["DeepGaze MSDB"] = model("DeepGaze_MSDB")
    for n in NULLS:
        units2[f"null {n}"] = NULLD[n]
    d2, same, excl, samp2 = 0.0, 0, 0, {}
    for u, sc in units2.items():
        e180, e90 = E(sc), E(sc, "illusory", "control90")
        w90 = V.wild(e90)
        st = S["item2_A4_2_control90"][u]
        d2 = max(d2, wild_cmp(w90, st["E90"]), abs(ratio(e90, e180) - st["ratio_E90_over_E180"]["value"]))
        same += int(np.sign(e90.mean()) == np.sign(e180.mean()))
        excl += int(w90["ci"][0] > 0 or w90["ci"][1] < 0)
        if u in ("DeepGaze MSDB", "DeepGaze IIE"):
            samp2[u] = dict(E90=w90, ratio=ratio(e90, e180))
    res["item2"] = dict(max_abs_diff_vs_stored=d2, n=len(units2), same_sign=same, e90_excl0=excl, samples=samp2)

    # ---------------------------------------------------------- validity (rule 2.7) from the stored MIT1003 CCs
    def ccm(f, a):
        return float(np.mean([V.jl(os.path.join(SWAP_DIR, f"mit_{f}_{a}_s{s}.json"))["CC"] for s in (0, 1, 2)]))
    valid = {}
    for f in FSETS:
        thr = round(round(ccm(f, "R4"), 4) - 0.03, 4)
        for a in ARCHS[:-1]:
            valid[(f, a)] = dict(valid=round(ccm(f, a), 4) >= thr, cc=ccm(f, a), thr=thr,
                                 gap=ccm(f, "R4") - ccm(f, a))
    res["validity"] = dict(valid_cells=sorted(f"{f}|{a}" for (f, a), v in valid.items() if v["valid"]),
                           matches_P3_cells=sorted(k for k, v in valid.items() if v["valid"]) == sorted(P3_CELLS),
                           below=[dict(cell=f"{f}|{a}", cc=v["cc"], gap=v["gap"]) for (f, a), v in valid.items()
                                  if not v["valid"]])

    # ---------------------------------------------------------- item 3 per seed
    d3, cnt, exceptions = 0.0, {}, []
    for f in FSETS:
        r4 = np.sign(E(head(f, "R4")).mean())
        for a in ARCHS:
            grp = "reference_R4" if a == "R4" else ("valid" if (f, a) in P3_CELLS else "below_criterion")
            for s in (0, 1, 2):
                sc = head(f, a, seeds=(s,))
                e = E(sc); w = V.wild(e); pc = V.pct_ci(e)
                st = S["item3_A4_3_profiles"]["per_seed"][f"{f}|{a}"][str(s)]
                d3 = max(d3, wild_cmp(w, st["E"]), abs(pc[0] - st["E_percentile"]["ci"][0]),
                         abs(pc[1] - st["E_percentile"]["ci"][1]))
                side = lambda ci: bool(ci[0] > 0) if r4 > 0 else bool(ci[1] < 0)
                for g in (grp, "overall"):
                    c = cnt.setdefault(g, dict(n=0, point=0, wcu=0, pct=0))
                    c["n"] += 1; c["point"] += int(np.sign(w["mean"]) == r4)
                    c["wcu"] += int(side(w["ci"])); c["pct"] += int(side(pc))
                if not side(w["ci"]) or not side(pc):
                    exceptions.append(dict(cell=f"{f}|{a}", seed=s, mean=w["mean"], wcu=w["ci"], pct=pc))
    stc = S["item3_A4_3_profiles"]["per_seed_counts"]
    res["item3"] = dict(max_abs_diff_vs_stored=d3, counts=cnt, exceptions=exceptions,
                        stored_counts={k: stc[k] for k in stc})

    # ---------------------------------------------------------- item 5 core share
    units5 = dict(units2)
    del_keys = [k for k in units5 if "|" in k]
    for k in del_keys:
        f, a = k.split("|")
        units5[k] = head(f, a, meas="f_core30")
    units5["dinov2reg_L12|R4"] = head("dinov2reg_L12", "R4", meas="f_core30")
    units5["dinov2reg_L6|R4"] = head("dinov2reg_L6", "R4", meas="f_core30")
    for bb, Ls in SWEEP.items():
        for L in Ls:
            units5[f"sweep {bb}_L{L}"] = sweep(bb, L, meas="f_core30")
    d5, samp5 = 0.0, {}
    for u, sc in units5.items():
        w = V.wild(E(sc, meas="f_core30"))
        d5 = max(d5, wild_cmp(w, S["item5_A4_5_core_share"][u]))
        if u in ("DeepGaze IIE", "dinov2reg_L12|R4", "sweep siglip_L4"):
            samp5[u] = w
    res["item5"] = dict(max_abs_diff_vs_stored=d5, n_units=len(units5), samples=samp5)

    # ---------------------------------------------------------- item 17
    i17 = S["item17_below_criterion_and_excluded_P4"]
    d17 = 0.0
    for k, v in i17["below_criterion"].items():
        f, a = k.split("|")
        sc = head(f, a)
        d17 = max(d17, wild_cmp(V.wild(E(sc)), v["E"]), abs(ratio(E(sc, "mis20"), E(sc)) - v["U20"]["value"]),
                  abs(valid[(f, a)]["cc"] - v["accuracy"]["cc"]), abs(valid[(f, a)]["thr"] - v["accuracy"]["threshold"]))
    r3, r4h = head("siglip_L24", "R3"), head("siglip_L24", "R4")
    u3, u4 = ratio(E(r3, "mis20"), E(r3)), ratio(E(r4h, "mis20"), E(r4h))
    a3, a4 = ratio(E(r3, "alt20"), E(r3)), ratio(E(r4h, "alt20"), E(r4h))
    ex = i17["siglip_L24_R3_excluded_P4"]
    t3 = V.wild(E(r3))["t"]
    d17 = max(d17, abs((u3 - u4) - ex["siglip_L24|R3|U20"]["diff"]), abs((a3 - a4) - ex["siglip_L24|R3|A20"]["diff"]),
              abs(t3 - ex["siglip_L24|R3|U20"]["t_den_1"]))
    res["item17"] = dict(max_abs_diff_vs_stored=d17, U20_R3=u3, U20_R4=u4, U20_diff=u3 - u4, A20_R3=a3, A20_R4=a4,
                         A20_diff=a3 - a4, R3_denominator_t=t3)

    # ---------------------------------------------------------- item 18 per support ratio
    sub = {sr: V.design(sr)[1] for sr in (0.3, 0.5, 0.7)}
    d18, cnt18 = 0.0, {str(sr): dict(ci_excl0=0, p_lt_05=0, disagree=0) for sr in sub}
    samp18 = {}
    for bb, Ls in SWEEP.items():
        for L in Ls:
            sc = sweep(bb, L)
            for sr, g in sub.items():
                d = V.per_config(g, lambda c: r(sc, c, "illusory") - r(sc, c, "control180"))
                w = V.wild(d)
                st = S["item18_A10_per_support_ratio"]["units"][f"{bb}_L{L}"][str(sr)]
                d18 = max(d18, wild_cmp(w, st))
                ex0 = w["ci"][0] > 0 or w["ci"][1] < 0
                c = cnt18[str(sr)]
                c["ci_excl0"] += int(ex0); c["p_lt_05"] += int(w["p"] < 0.05); c["disagree"] += int(ex0 != (w["p"] < 0.05))
                if (bb, L, sr) in (("siglip", 24, 0.7), ("dinov2", 6, 0.3)):
                    samp18[f"{bb}_L{L} SR {sr}"] = w
    res["item18"] = dict(max_abs_diff_vs_stored=d18, counts=cnt18, samples=samp18,
                         smallest_attainable_p=6 / 6 ** 4)

    # ---------------------------------------------------------- rule 1.1 item 5 text observation
    res["rule_1_1_item5"] = {f"{bb}_L{L}": float(E(sweep(bb, L)).mean()) for bb, L in (("dinov2", 10), ("clip", 12))}

    res["seconds"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v if k in ("validity",) else {kk: vv for kk, vv in v.items() if kk != "samples"})
                      if isinstance(v, dict) else v for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
