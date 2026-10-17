"""
Decomposition statistics: exact Shapley decompositions of the Kanizsa effect (illusory - control180) in DeepGaze IIE
and DeepGaze MSDB, MSDB lesion variants and the xSoyelNet-ML comparison.
Reads <results>/v2/decompose/scores/{iie,msdb,lesion,ml}.json and, for each IIE backbone path's own effect, the
per-path maps that run_iie.py saves (maps/iie, not distributed); writes <results>/v2/decompose/results.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_d import *  # noqa

CARRY = 0.5
GAMES = {
    "IIE_backbones": dict(model="DeepGaze IIE", src="iie", n=4, primary=True,
                          players=["ShapeNetC ResNet-50", "EfficientNet-B5", "DenseNet-201", "ResNeXt-50"]),
    "MSDB_feature_tensors": dict(model="DeepGaze MSDB", src="msdb", n=3, primary=True,
                                 players=["CLIP RN50x64 layer4.2.conv2", "DINOv2 block 10", "DINOv2 block 6"]),
    "MSDB_backbones": dict(model="DeepGaze MSDB", src="msdb", n=2, primary=False, players=["CLIP RN50x64", "DINOv2 (blocks 6 + 10)"]),
    "MSDB_scale_groups": dict(model="DeepGaze MSDB", src="msdb", n=3, primary=True,
                              players=["coarse (ppd 5, 10; size 128, 256)", "native (ppd 17.5; size 512)",
                                       "fine (ppd 24, 30; size 768, 1024)"]),
    "MSDB_scales10": dict(model="DeepGaze MSDB", src="msdb", n=10, primary=False, players=None),
}


def hybrid_masses(game, cfg_scores, T):
    if game == "IIE_backbones":
        return cfg_scores["hybrids"]["".join(str(T >> b & 1) for b in range(4))]
    if game == "MSDB_feature_tensors":
        return cfg_scores["hybrids"]["grp_" + "".join(str(T >> j & 1) for j in range(3))]
    if game == "MSDB_backbones":
        c, d = T & 1, T >> 1 & 1
        return cfg_scores["hybrids"][f"grp_{c}{d}{d}"]
    if game == "MSDB_scale_groups":
        return cfg_scores["hybrids"]["scg_" + "".join(str(T >> j & 1) for j in range(3))]
    if game == "MSDB_scales10":
        return cfg_scores["sc10"][str(T)]
    raise KeyError(game)


def v2_name(name):
    return name.replace("_control180", "_control")


def main():
    cfgs = kanizsa_configs()
    idx, cell_keys = cell_index(cfgs)
    K, P = idx.shape
    draws = boot_draws(K, P)
    kp = lambda arr: np.asarray(arr)[idx]                    # per-config array (60,) -> [K, P]
    S = {k: LP.load_json(os.path.join(SCORES, k + ".json")) for k in ("iie", "msdb", "lesion", "ml")
         if LP.json_exists(os.path.join(SCORES, k + ".json"))}
    for k in list(S):
        n_have = len(S[k]["configs"]) if "configs" in S[k] else len(S[k]) // 2
        if n_have < len(cfgs):
            print(f"skipping incomplete source {k}: {n_have}/{len(cfgs)}"); del S[k]
    msdb_names = S["msdb"]["scale_names"] if "msdb" in S else []
    GAMES["MSDB_scales10"]["players"] = msdb_names
    out = dict(design=dict(n_configs=len(cfgs), cells=[list(map(str, k)) for k in cell_keys], positions_per_cell=P,
                           bootstrap=B, comparison="illusory - control180", carries_rule=f"share >= {CARRY}"),
               checks={}, games={}, lesion={}, notes=[])
    if "msdb" in S:
        out["msdb_scale_weights"] = dict(zip(msdb_names, S["msdb"]["scale_weights"]))
        out["msdb_scale_groups"] = S["msdb"]["scale_groups"]

    # ------------------------------------------------------------ checks against the published stim_v2 scores
    pub = {m: json.load(open(os.path.join(LEG, "kanizsa", f"results_{m}.json"))) for m in ("DeepGaze_IIE", "DeepGaze_MSDB", "xSoyelNet-ML")}
    def pubdiff(model, getr):
        d = [abs(getr(c, tag) - pub[model][v2_name(c[tag])]) for c in cfgs for tag in ("ill", "ctrl")]
        return dict(max_abs_r_diff_vs_published_stim_v2=float(np.max(d)), mean_abs=float(np.mean(d)))
    if "iie" in S:
        C = S["iie"]["configs"]
        out["checks"]["IIE_vs_published"] = pubdiff("DeepGaze_IIE", lambda c, t: r_from(C[c["config"]]["hybrids"]["1111" if t == "ill" else "0000"]))
        out["checks"]["IIE_mixture_vs_forward"] = S["iie"]["checks"]
    if "msdb" in S:
        C = S["msdb"]["configs"]
        out["checks"]["MSDB_vs_published"] = pubdiff("DeepGaze_MSDB", lambda c, t: r_from(C[c["config"]]["hybrids"]["grp_111" if t == "ill" else "grp_000"]))
        ch = S["msdb"]["checks"]
        out["checks"]["MSDB_head_vs_forward"] = {k: v for c in ch.values() for k, v in c.items() if "forward" in k or "absmax" in k}
        out["checks"]["MSDB_sc10_factorisation_vs_direct"] = dict(
            max_abs_logdens_diff=max(c["sc10_check_max_abs_logdens_diff"] for c in ch.values()),
            max_abs_r_diff=max(c["sc10_check_max_abs_r_diff"] for c in ch.values()))
        dd = []
        for c in cfgs:
            x = C[c["config"]]
            for a, b in (("grp_111", "scg_111"), ("grp_000", "scg_000")):
                dd.append(abs(r_from(x["hybrids"][a]) - r_from(x["hybrids"][b])))
            dd.append(abs(r_from(x["hybrids"]["grp_111"]) - r_from(x["sc10"]["1023"])))
            dd.append(abs(r_from(x["hybrids"]["grp_000"]) - r_from(x["sc10"]["0"])))
        out["checks"]["MSDB_endpoints_consistent_across_games_max_abs_r_diff"] = float(max(dd))
    if "ml" in S:
        out["checks"]["ML_vs_published"] = pubdiff("xSoyelNet-ML", lambda c, t: r_from(S["ml"][c[t]]))

    # ------------------------------------------------------------ games
    for g, spec in GAMES.items():
        if spec["src"] not in S:
            continue
        C = S[spec["src"]]["configs"]
        n = spec["n"]
        gres = dict(model=spec["model"], players=spec["players"], n_players=n, n_coalitions=1 << n, primary=spec["primary"], aoi={})
        for var in VARIANTS:
            per = [shapley_both(n, lambda T, c=c: r_from(hybrid_masses(g, C[c["config"]], T), var)) for c in cfgs]
            phi = np.array([p["phi"] for p in per]); eff = np.array([p["effect"] for p in per])
            add_err = np.abs(phi.sum(1) - eff)
            dir_diff = np.max(np.abs(np.array([p["phi_ins"] - p["phi_rem"] for p in per])))
            ins1 = np.array([p["ins_alone"] for p in per]); rem1 = np.array([p["rem_alone"] for p in per])
            vr = dict(effect=summ(kp(eff), draws),
                      additivity_max_abs_err=float(add_err.max()),
                      direction_max_abs_diff_phi_ins_vs_rem=float(dir_diff),
                      players={})
            for i, pl in enumerate(spec["players"]):
                sh = summ_ratio(kp(phi[:, i]), kp(eff), draws)
                vr["players"][pl] = dict(
                    phi=summ(kp(phi[:, i]), draws, 5), share=sh["share"], share_ci=sh["ci"],
                    carries=bool(sh["share"] >= CARRY), carries_ci_lower_ge_rule=bool(sh["ci"][0] >= CARRY),
                    insert_alone=summ(kp(ins1[:, i]), draws, 5), remove_alone=summ(kp(rem1[:, i]), draws, 5),
                    insert_alone_share=summ_ratio(kp(ins1[:, i]), kp(eff), draws)["share"],
                    remove_alone_share=summ_ratio(kp(rem1[:, i]), kp(eff), draws)["share"])
            vr["sum_insert_alone_share"] = round(float(ins1.sum(1).mean() / eff.mean()), 3)
            vr["sum_remove_alone_share"] = round(float(rem1.sum(1).mean() / eff.mean()), 3)
            # descriptive breakdown of shares by shape and support ratio (point estimates)
            br = {}
            for key, sel in [(f"shape={s}", lambda c, s=s: c["shape"] == s) for s in ("square", "triangle")] + \
                            [(f"sr={r}", lambda c, r=r: c["sr"] == r) for r in (0.3, 0.5, 0.7)]:
                m = np.array([sel(c) for c in cfgs])
                rows = [k for k in range(K) if all(m[j] for j in idx[k])]       # cells inside the subset
                sidx = idx[rows]; sd = boot_draws(len(rows), P)
                sk = lambda a: np.asarray(a)[sidx]
                br[key] = dict(effect=round(float(eff[m].mean()), 4), n_cells=len(rows),
                               shares={pl: round(float(phi[m, i].mean() / eff[m].mean()), 3) for i, pl in enumerate(spec["players"])},
                               share_cis={pl: summ_ratio(sk(phi[:, i]), sk(eff), sd)["ci"] for i, pl in enumerate(spec["players"])})
            vr["by_subset_point_estimates"] = br
            if g in ("MSDB_scale_groups", "MSDB_scales10") and "msdb" in S:
                wts = dict(zip(msdb_names, S["msdb"]["scale_weights"]))
                grp_w = {pl: sum(wts[nm] for nm in S["msdb"]["scale_groups"][k]) for pl, k in zip(spec["players"], ("coarse", "native", "fine"))} \
                    if g == "MSDB_scale_groups" else {pl: wts[pl] for pl in spec["players"]}
                vr["scale_weight_and_share_per_weight"] = {pl: dict(weight=round(grp_w[pl], 4), share_per_weight=round(vr["players"][pl]["share"] / grp_w[pl], 2))
                                                           for pl in spec["players"]}
            vr["per_config"] = {c["config"]: dict(effect=round(float(eff[j]), 5), phi=[round(float(x), 5) for x in phi[j]])
                                for j, c in enumerate(cfgs)}
            gres["aoi"][var] = vr
        carried = [pl for pl, v in gres["aoi"]["inside"]["players"].items() if v["carries"]]
        gres["label_inside"] = (f"carried by {carried[0]}" if len(carried) == 1 else
                                "no single component carries the effect (no share >= 0.5)" if not carried else
                                "several components have share >= 0.5: " + ", ".join(carried))
        out["games"][g] = gres

    # ------------------------------------------------------------ IIE: each path's own effect (from the saved float16 maps)
    have_maps = all(os.path.exists(os.path.join(MAPS, "iie", c[t] + ".npz")) for c in cfgs for t in ("ill", "ctrl"))
    if "iie" in S and not have_maps:
        out["notes"].append("iie_path_alone_effects not computed: the per-path maps of run_iie.py are absent")
    if "iie" in S and have_maps:
        names = GAMES["IIE_backbones"]["players"]
        own = {var: np.zeros((len(cfgs), 4)) for var in VARIANTS}
        mix16 = {var: np.zeros(len(cfgs)) for var in VARIANTS}
        for j, c in enumerate(cfgs):
            aoi = read_aoi(c["ill"])
            mp = {t: np.load(os.path.join(MAPS, "iie", c[t] + ".npz")) for t in ("ill", "ctrl")}
            pm = {t: [mp[t][f"p{b}"].astype(np.float32) for b in range(4)] for t in mp}
            ms = {t: [masses(x, aoi) for x in pm[t]] for t in pm}
            mm = {t: masses(np.mean(pm[t], 0), aoi) for t in pm}
            for var in VARIANTS:
                own[var][j] = [r_from(ms["ill"][b], var) - r_from(ms["ctrl"][b], var) for b in range(4)]
                mix16[var][j] = r_from(mm["ill"], var) - r_from(mm["ctrl"], var)
        eff32 = np.array([r_from(S["iie"]["configs"][c["config"]]["hybrids"]["1111"]) - r_from(S["iie"]["configs"][c["config"]]["hybrids"]["0000"]) for c in cfgs])
        out["checks"]["IIE_effect_from_saved_fp16_maps_vs_fp64_max_abs_diff"] = float(np.max(np.abs(mix16["inside"] - eff32)))
        out["iie_path_alone_effects"] = {var: {names[b]: summ(kp(own[var][:, b]), draws) for b in range(4)} for var in VARIANTS}

    # ------------------------------------------------------------ lesions and xSoyelNet-ML
    if "lesion" in S and "msdb" in S:
        L = S["lesion"]["configs"]; C = S["msdb"]["configs"]
        eff_of = {}
        for var in VARIANTS:
            e = {}
            e["MSDB full (fp32)"] = np.array([r_from(C[c["config"]]["hybrids"]["grp_111"], var) - r_from(C[c["config"]]["hybrids"]["grp_000"], var) for c in cfgs])
            for key, lab in (("full_fp16", "MSDB full (fp16 cache)"), ("dino_only", "MSDB DINOv2-only (CLIP clamped)"),
                             ("clip_only", "MSDB CLIP-only (DINOv2 clamped)")):
                e[lab] = np.array([r_from(L[c["config"]][f"{key}_I"], var) - r_from(L[c["config"]][f"{key}_C"], var) for c in cfgs])
            if "ml" in S:
                e["xSoyelNet-ML"] = np.array([r_from(S["ml"][c["ill"]], var) - r_from(S["ml"][c["ctrl"]], var) for c in cfgs])
            eff_of[var] = e
            vr = {lab: summ(kp(v), draws) for lab, v in e.items()}
            vr["fp16_vs_fp32_full_effect_max_abs_diff"] = float(np.max(np.abs(e["MSDB full (fp16 cache)"] - e["MSDB full (fp32)"])))
            if "xSoyelNet-ML" in e:
                vr["diff: MSDB DINOv2-only minus xSoyelNet-ML"] = summ(kp(e["MSDB DINOv2-only (CLIP clamped)"] - e["xSoyelNet-ML"]), draws)
                vr["diff: MSDB full minus xSoyelNet-ML"] = summ(kp(e["MSDB full (fp32)"] - e["xSoyelNet-ML"]), draws)
                vr["sign agreement DINOv2-only vs ML (% configs)"] = round(float(np.mean(np.sign(e["MSDB DINOv2-only (CLIP clamped)"]) == np.sign(e["xSoyelNet-ML"])) * 100), 1)
            vr["diff: MSDB full minus DINOv2-only"] = summ(kp(e["MSDB full (fp16 cache)"] - e["MSDB DINOv2-only (CLIP clamped)"]), draws)
            vr["diff: MSDB full minus CLIP-only"] = summ(kp(e["MSDB full (fp16 cache)"] - e["MSDB CLIP-only (DINOv2 clamped)"]), draws)
            # per-image r levels (illusory, control) to see what moves
            lv = {}
            for key, lab in (("full_fp16", "MSDB full"), ("dino_only", "MSDB DINOv2-only"), ("clip_only", "MSDB CLIP-only")):
                lv[lab] = dict(r_illusory=round(float(np.mean([r_from(L[c["config"]][f"{key}_I"], var) for c in cfgs])), 4),
                               r_control180=round(float(np.mean([r_from(L[c["config"]][f"{key}_C"], var) for c in cfgs])), 4))
            if "ml" in S:
                lv["xSoyelNet-ML"] = dict(r_illusory=round(float(np.mean([r_from(S["ml"][c["ill"]], var) for c in cfgs])), 4),
                                          r_control180=round(float(np.mean([r_from(S["ml"][c["ctrl"]], var) for c in cfgs])), 4))
            vr["mean_r_levels"] = lv
            out["lesion"][var] = vr
    save_json(out, os.path.join(DD, "results.json"))
    json.dump(out, open(os.path.join(DD, "results.json"), "w"), indent=1)
    print(json.dumps({g: {"label": v["label_inside"], "effect": v["aoi"]["inside"]["effect"],
                          "shares": {p: (x["share"], x["share_ci"]) for p, x in v["aoi"]["inside"]["players"].items()}}
                      for g, v in out["games"].items()}, indent=1))
    print(json.dumps(out["checks"], indent=1))
    if out["lesion"]:
        print(json.dumps({k: v for k, v in out["lesion"]["inside"].items()}, indent=1))


if __name__ == "__main__":
    main()
