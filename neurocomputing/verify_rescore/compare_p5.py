"""
Compare the independent recomputation (verify_rescore/recompute_p5.json) with the R7 output
(reanalysis/out/primary_P5.json) key by key; also: the IIE part of R7 against the D3 partial run
(reanalysis/results/primary_P5_partial.json) and the P1 / P2 blocks of R7 against S2 (reanalysis/out/primary_P1_P2_P3_P4.json).
Reads JSON outputs only; no project module is imported. Calls an independent copy of require_public() first.
Output: verify_rescore/compare_p5.json.
"""
import hashlib, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                          # re-analysis folder (neurocomputing/ in the repository)
OUT = os.path.join(HERE, "compare_p5.json")
NULLS = [f"{k}_s{s}" for k in ("corner", "edge", "ink") for s in (8, 16, 25, 40, 60)]
MODELS = ("DeepGaze IIE", "DeepGaze MSDB")
TOL = 1e-12


def require_public():
    rec, pub = os.path.join(RE, "REANALYSIS_RULE.sha256"), os.path.join(RE, "PUBLIC_PUSH.txt")
    assert os.path.exists(rec) and os.path.exists(pub)
    tok = open(pub).read().split()
    assert tok and len(tok[0]) >= 7 and all(c in "0123456789abcdef" for c in tok[0])
    n = 0
    for line in open(rec):
        p = line.split()
        if len(p) == 2 and len(p[0]) == 64:
            assert hashlib.sha256(open(os.path.join(RE, p[1]), "rb").read()).hexdigest() == p[0], p[1]
            n += 1
    man = json.load(open(os.path.join(RE, "stim_rigid", "manifest.json")))
    for f, h in man["files"].items():
        assert hashlib.md5(open(os.path.join(RE, "stim_rigid", f), "rb").read()).hexdigest() == h, f
    return n


rows = []


def add(path, kind, rep, mine, note="", info=False):
    if isinstance(rep, (bool, str, type(None), list)) or isinstance(mine, (bool, str, type(None), list)):
        if isinstance(rep, list) and rep and isinstance(rep[0], float):
            for i, (a, b) in enumerate(zip(rep, mine)):
                add(f"{path}[{i}]", kind, a, b, note, info)
            return
        ok = rep == mine
        rows.append(dict(path=path, kind=kind, reported=rep, recomputed=mine, abs_diff=None, agree=bool(ok), note=note,
                         informational=info))
        return
    d = abs(float(rep) - float(mine))
    rows.append(dict(path=path, kind=kind, reported=rep, recomputed=mine, abs_diff=d, agree=bool(d <= TOL), note=note,
                     informational=info))


def wild_block(path, rep, mine):
    add(path + ".mean", "mean", rep["mean"], mine["mean"])
    add(path + ".se", "se", rep["se"], mine["se"])
    add(path + ".t", "t", rep["t"], mine["t"])
    add(path + ".p", "p", rep["p"], mine["p"], "WCR p, ties counted")
    add(path + ".p(strict float count)", "p_strict_info", rep["p"], mine["p_strict"],
        "informational: strict floating-point count of t* >= t misses exact ties (VERIFY.md discrepancy 2)", True)
    add(path + ".p_min", "p_min", rep["p_min"], mine["p_min"])
    add(path + ".ci", "ci", rep["ci"], mine["ci"])
    add(path + ".draws", "int", rep["draws"], mine["n_vectors"])
    add(path + ".enumerated", "bool", rep["enumerated"], mine["enumerated"])
    add(path + ".G", "int", rep["G"], mine["G"])
    add(path + ".N", "int", rep["N"], mine["N"])
    add(path + ".skew_cluster_means", "skew", rep["skew_cluster_means"], mine["skew_cluster_means"])


def f3(x):
    return f"{x:+.3f}"


def ci3(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{pre}.{k}" if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, o


def main():
    n_hashed = require_public()
    R = json.load(open(os.path.join(RE, "out/primary_P5.json")))["result"]
    M = json.load(open(os.path.join(HERE, "recompute_p5.json")))

    for m in MODELS:
        # P1
        r, x = R["P1"][m], M["P1"][m]
        wild_block(f"P1.{m}.L_model", r["L_model"], x["L_model"])
        for k in NULLS:
            wild_block(f"P1.{m}.vs_null.{k}", r["vs_null"][k], x["vs_null"][k])
        add(f"P1.{m}.met", "decision", r["met"], x["met"])
        add(f"P1.{m}.met (strict float p)", "decision_info", r["met"], x["met_strict"], "informational", True)
        add(f"P1.{m}.null_with_largest_loss", "str", r["null_with_largest_loss"], x["null_with_largest_loss"])
        add(f"P1.{m}.n_separated", "count", r["n_separated"], x["n_separated"])
        # P2
        r, x = R["P2"][m], M["P2"][m]
        wild_block(f"P2.{m}.Delta_model", r["Delta_model"], x["Delta_model"])
        for k in NULLS:
            wild_block(f"P2.{m}.vs_null.{k}", r["vs_null"][k], x["vs_null"][k])
        add(f"P2.{m}.met", "decision", r["met"], x["met"])
        add(f"P2.{m}.met (strict float p)", "decision_info", r["met"], x["met_strict"], "informational", True)
        add(f"P2.{m}.n_separated", "count", r["n_separated"], x["n_separated"])
        # P5
        r, x = R["P5"][m], M["P5"][m]
        add(f"P5.{m}.computable", "decision", r["computable"], x["computable"])
        for key in ("L_uni", "L_rig", "align_effect", "half_criterion"):
            wild_block(f"P5.{m}.{key}", r[key], x[key])
        for k in NULLS:
            wild_block(f"P5.{m}.vs_null.{k}", r["vs_null"][k], x["vs_null"][k])
        add(f"P5.{m}.iut_met", "decision", r["iut_met"], x["iut_met"])
        add(f"P5.{m}.iut_met (strict float p)", "decision_info", r["iut_met"], x["iut_met_strict"], "informational", True)
        add(f"P5.{m}.n_separated", "count", r["n_separated"], x["n_separated"])
        rs, xs = r["rigid_share"], x["rigid_share"]
        add(f"P5.{m}.rigid_share.value", "ratio", rs["value"], xs["value"])
        add(f"P5.{m}.rigid_share.ci (BCa)", "ci_bca", rs["ci"], xs["ci"])
        add(f"P5.{m}.rigid_share.z0", "z0", rs["z0"], xs["z0"])
        add(f"P5.{m}.rigid_share.acc", "acc", rs["acc"], xs["acc"])
        add(f"P5.{m}.rigid_share.B", "int", rs["B"], xs["B"])
        add(f"P5.{m}.rigid_share.alpha_adj", "alpha_adj", rs["alpha_adj"], xs["alpha_adj"])
        add(f"P5.{m}.rigid_share.t_den", "t", rs["t_den"], xs["t_den"])
        add(f"P5.{m}.rigid_share.stable", "decision", rs["stable"], xs["stable"])
        add(f"P5.{m}.rigid_share.denominator_excludes_zero", "decision", rs["denominator_excludes_zero"],
            xs["denominator_excludes_zero"])
        fr, fx = rs["fieller"], xs["fieller"]
        add(f"P5.{m}.rigid_share.fieller.value", "ratio", fr["value"], fx["value"])
        add(f"P5.{m}.rigid_share.fieller.bounded", "decision", fr["bounded"], fx["bounded"])
        if fr["ci"] is None or fx["ci"] is None:
            add(f"P5.{m}.rigid_share.fieller.ci", "ci_fieller", fr["ci"], fx["ci"])
        else:
            add(f"P5.{m}.rigid_share.fieller.ci", "ci_fieller", fr["ci"], fx["ci"])
        add(f"P5.{m}.rigid_share.fieller.t_den", "t", fr["t_den"], fx["t_den"])
        for key in ("A_drafted", "A_control_free", "control_rigid_change"):
            wild_block(f"P5.{m}.secondary.{key}", r["secondary"][key], x["secondary"][key])
        add(f"P5.{m}.category", "decision", r["category"], x["category"])
    add("P5.both_category_1", "decision", R["P5"]["both_category_1"], M["P5"]["both_category_1"])
    ab = M["abstract_rotation"]
    add("P5.abstract_leads_with_alternation", "decision", R["P5"]["abstract_leads_with_alternation"],
        ab["branch"].startswith("A not empty"))
    ra = R["abstract_rotation"]
    add("abstract_rotation.uniform_models", "decision", ra["uniform_models"], ab["uniform_models"])
    add("abstract_rotation.alternation_models", "decision", ra["alternation_models"], ab["alternation_models"])
    exp_abs = ("square alternation for IIE and MSDB; no uniform-rotation clause"
               if ab["branch"].startswith("A not empty") and len(ab["alternation_models"]) == 2 else ab["branch"])
    add("abstract_rotation.abstract", "decision", ra["abstract"], exp_abs, "engine's label vs the branch of rule 5.5")
    add("abstract_rotation.highlight4", "str", ra["highlight4"], ab["highlight4"])
    add("abstract_rotation.highlight4_chars", "count", ra["highlight4_chars"], ab["highlight4_chars"])
    add("abstract_rotation.highlight4_chars (len of reported string)", "count", ra["highlight4_chars"], len(ra["highlight4"]))
    add("abstract_rotation.highlight4_chars (stated in rule 5.5)", "count", ra["highlight4_chars"],
        ab["highlight4_chars_stated_in_rule"])

    # P1 and P2 wording strings of R7, against the rule 7.1 templates filled with the recomputed values (3 decimals,
    # the engine's precision for these two sentences)
    P1, P2 = M["P1"], M["P2"]
    w1 = ("Unlike all 15 local nulls, both DeepGaze models lost figure mass under a uniform 20 degree inducer rotation "
          "(differences against the null with the largest loss: "
          + ", ".join(f"{m.replace('DeepGaze ', '')} {f3(P1[m]['vs_null'][P1[m]['null_with_largest_loss']]['mean'])} "
                      f"{ci3(P1[m]['vs_null'][P1[m]['null_with_largest_loss']]['ci'])}" for m in MODELS)
          + "), a manipulation that also turns the notch edges oblique.")
    w2 = ("On squares, alternating the rotation direction, which keeps the axis mirror symmetries, kept more mass inside "
          f"the figure than uniform rotation (difference {f3(P2['DeepGaze IIE']['Delta_model']['mean'])} "
          f"{ci3(P2['DeepGaze IIE']['Delta_model']['ci'])} for IIE and {f3(P2['DeepGaze MSDB']['Delta_model']['mean'])} "
          f"{ci3(P2['DeepGaze MSDB']['Delta_model']['ci'])} for MSDB), more than for every local null, for which the two "
          "rotations left the same figure mass (differences 0.0000 to +0.0003).")
    add("P1.wording", "str", R["P1"]["wording"], w1, "rule 7.1 template, P1 met for both")
    add("P2.wording", "str", R["P2"]["wording"], w2, "rule 7.1 template, P2 met for both")
    nd = M["P2"]["null_square_Delta20_means"]
    null_range = [min(nd.values()), max(nd.values())]

    # extra 1: IIE part of R7 equals the D3 partial run (results/primary_P5_partial.json)
    part = json.load(open(os.path.join(RE, "results/primary_P5_partial.json")))
    pr = part["result"]["P5"]["DeepGaze IIE"] if "result" in part else part["P5"]["DeepGaze IIE"]
    a = dict(leaves(pr)); b = dict(leaves(R["P5"]["DeepGaze IIE"]))
    common = sorted(set(a) & set(b))
    diffs = [(k, a[k], b[k]) for k in common if a[k] != b[k]]
    extra_partial = dict(n_common_leaves=len(common), only_in_partial=sorted(set(a) - set(b)),
                         only_in_R7=sorted(set(b) - set(a)), n_different=len(diffs), different=diffs[:20])
    # extra 2: P1 / P2 blocks of R7 equal those of S2
    s2 = json.load(open(os.path.join(RE, "out/primary_P1_P2_P3_P4.json")))["result"]
    ex2 = {}
    for fam in ("P1", "P2"):
        a = dict(leaves(s2[fam])); b = dict(leaves(R[fam]))
        common = sorted(set(a) & set(b))
        ex2[fam] = dict(n_common_leaves=len(common), only_in_S2=sorted(set(a) - set(b)), only_in_R7=sorted(set(b) - set(a)),
                        n_different=sum(a[k] != b[k] for k in common),
                        different=[(k, a[k], b[k]) for k in common if a[k] != b[k]][:20])

    # aggregate by kind
    agg = {}
    for r in rows:
        g = agg.setdefault(r["kind"], dict(n=0, n_disagree=0, max_abs_diff=0.0, informational=r["informational"]))
        g["n"] += 1
        g["n_disagree"] += (not r["agree"])
        if r["abs_diff"] is not None:
            g["max_abs_diff"] = max(g["max_abs_diff"], r["abs_diff"])
    out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), hashed_files_verified=n_hashed,
               reported_file="reanalysis/out/primary_P5.json", reported_created_utc=json.load(open(os.path.join(RE, "out/primary_P5.json")))["created_utc"],
               recomputed_file="verify_rescore/recompute_p5.json", recomputed_created_utc=M["created_utc"], tolerance=TOL,
               n_rows=len(rows), n_rows_decisive=sum(not r["informational"] for r in rows),
               n_disagree=sum((not r["agree"]) and (not r["informational"]) for r in rows),
               n_disagree_informational=sum((not r["agree"]) and r["informational"] for r in rows),
               aggregate_by_kind=agg,
               disagreements=[r for r in rows if not r["agree"] and not r["informational"]],
               disagreements_informational=[r for r in rows if not r["agree"] and r["informational"]],
               null_square_Delta20_range=null_range,
               extra_iie_R7_vs_partial=extra_partial, extra_P1P2_R7_vs_S2=ex2, rows=rows)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} rows ({out['n_rows_decisive']} decisive), {out['n_disagree']} disagree, "
          f"{out['n_disagree_informational']} informational rows differ")
    for k, g in agg.items():
        print(f"  {k:14s} n={g['n']:4d} disagree={g['n_disagree']:3d} max|diff|={g['max_abs_diff']:.3g} info={g['informational']}")
    for r in out["disagreements"]:
        print("  DISAGREE", r["path"], r["reported"], r["recomputed"], r["abs_diff"])
    print("null square Delta20 range", null_range)
    print("partial vs R7 (IIE):", {k: v for k, v in extra_partial.items() if k != "different"})
    print("S2 vs R7:", {f: {k: v for k, v in d.items() if k != "different"} for f, d in ex2.items()})


if __name__ == "__main__":
    main()
