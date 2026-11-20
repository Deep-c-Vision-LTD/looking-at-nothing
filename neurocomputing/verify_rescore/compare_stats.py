"""
Compare neuro/verify_rescore/recompute_stats.json (independent recomputation) with the reported
neuro/reanalysis/out/primary_P1_P2_P3_P4.json, key by key. Writes neuro/verify_rescore/compare_stats.json and prints a
summary. Read only on the reported file. No project module imported.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                          # re-analysis folder (neurocomputing/ in the repository)
mine = json.load(open(os.path.join(HERE, "recompute_stats.json")))
rep = json.load(open(os.path.join(RE, "out", "primary_P1_P2_P3_P4.json")))["result"]
rows = []
add_tail = []


def add(claim, path_rep, r, m, kind, decision=False):
    if isinstance(r, list):
        diff = max(abs(a - b) for a, b in zip(r, m))
    elif isinstance(r, bool) or isinstance(m, bool) or isinstance(r, str):
        diff = 0.0 if r == m else 1.0
    else:
        diff = abs(r - m)
    rows.append(dict(claim=claim, reported_key=path_rep, reported=r, recomputed=m, abs_diff=diff, kind=kind,
                     decision=decision))


for fam in ("P1", "P2"):
    for mod in ("DeepGaze IIE", "DeepGaze MSDB"):
        R, M = rep[fam][mod], mine[fam][mod]
        own_r = R.get("L_model") or R.get("Delta_model")
        own_m = M.get("L_model") or M.get("Delta_model")
        if own_r is not None:
            for k in ("mean", "se", "t", "p"):
                add(f"{fam} {mod} model value {k}", f"result.{fam}.{mod}.{'L_model' if fam=='P1' else 'Delta_model'}.{k}",
                    own_r[k], own_m[k], k)
            add(f"{fam} {mod} model value WCU", f"result.{fam}.{mod}...ci", own_r["ci"], own_m["ci"], "ci")
        for nk, cr in R["vs_null"].items():
            cm = M["vs_null"][nk]
            for k in ("mean", "se", "t", "p"):
                add(f"{fam} {mod} minus {nk} {k}", f"result.{fam}.{mod}.vs_null.{nk}.{k}", cr[k], cm[k], k)
            add(f"{fam} {mod} minus {nk} WCU", f"result.{fam}.{mod}.vs_null.{nk}.ci", cr["ci"], cm["ci"], "ci")
            if "skew_cluster_means" in cr:
                add(f"{fam} {mod} minus {nk} skew", f"result.{fam}.{mod}.vs_null.{nk}.skew_cluster_means",
                    cr["skew_cluster_means"], cm["skew_cluster_means"], "skew")
        add(f"{fam} {mod} met", f"result.{fam}.{mod}.met", R["met"], M["met"], "decision", True)
        if "n_separated" in R:
            add(f"{fam} {mod} n separated", f"result.{fam}.{mod}.n_separated", R["n_separated"], M["n_separated"], "count", True)

R3, M3 = rep["P3"], mine["P3"]
for fs, r in R3["reference"].items():
    m = M3["reference"][fs]
    for k in ("mean", "p", "t"):
        if k in r:
            add(f"P3 reference {fs} R4 {k}", f"result.P3.reference.{fs}.{k}", r[k], m[k], k)
    if "ci" in r:
        add(f"P3 reference {fs} R4 WCU", f"result.P3.reference.{fs}.ci", r["ci"], m["ci"], "ci")
    if "defined" in r:
        add(f"P3 reference {fs} defined", f"result.P3.reference.{fs}.defined", r["defined"], m["defined"], "decision", True)
for ck, r in R3["cells"].items():
    m = M3["cells"][ck]
    for k in ("mean", "se", "t", "p", "p_holm"):
        if k in r:
            add(f"P3 {ck} {k}", f"result.P3.cells.{ck}.{k}", r[k], m[k], k)
    add(f"P3 {ck} WCU", f"result.P3.cells.{ck}.ci", r["ci"], m["ci"], "ci")
    if "skew_cluster_means" in r:
        add(f"P3 {ck} skew", f"result.P3.cells.{ck}.skew_cluster_means", r["skew_cluster_means"], m["skew_cluster_means"], "skew")
    add(f"P3 {ck} keeps sign", f"result.P3.cells.{ck}.keeps_sign", r["keeps_sign"], m["keeps_sign"], "decision", True)
    add(f"P3 {ck} opposite interval", f"result.P3.cells.{ck}.opposite_sign_interval", r["opposite_sign_interval"], m["opposite_interval"], "decision", True)

R4, M4 = rep["P4"], mine["P4"]
for ck, r in R4["contrasts"].items():
    m = M4["contrasts"][ck]
    add(f"P4 {ck} difference", f"result.P4.contrasts.{ck}.diff", r["diff"], m["est"], "mean")
    add(f"P4 {ck} BCa CI", f"result.P4.contrasts.{ck}.ci", r["ci"], m["ci"], "ci")
    add(f"P4 {ck} BCa p", f"result.P4.contrasts.{ck}.p", r["p"], m["p"], "p")
    add(f"P4 {ck} Holm p", f"result.P4.contrasts.{ck}.p_holm", r["p_holm"], m["p_holm"], "p_holm")
    add(f"P4 {ck} min |t| den", f"result.P4.contrasts.{ck}.t_den_min", r["t_den_min"], m["t_den_min"], "t")
    add(f"P4 {ck} ratio readout", f"result.P4.contrasts.{ck}.ratio_readout", r.get("ratio_readout"), m["ratio_readout"], "mean") if "ratio_readout" in r else None
    add(f"P4 {ck} ratio R4", f"result.P4.contrasts.{ck}.ratio_R4", r.get("ratio_R4"), m["ratio_R4"], "mean") if "ratio_R4" in r else None
    add(f"P4 {ck} ratio readout BCa", f"result.P4.contrasts.{ck}.ratio_readout_bca.ci", r["ratio_readout_bca"]["ci"], m["ratio_readout_bca"], "ci")
    add(f"P4 {ck} ratio R4 BCa", f"result.P4.contrasts.{ck}.ratio_R4_bca.ci", r["ratio_R4_bca"]["ci"], m["ratio_R4_bca"], "ci")
    add(f"P4 {ck} z0", f"result.P4.contrasts.{ck}.z0", r["z0"], m["z0"], "z0")
    add(f"P4 {ck} acc", f"result.P4.contrasts.{ck}.acc", r["acc"], m["acc"], "acc")
    add(f"P4 {ck} counted", f"result.P4.contrasts.{ck}.counted", r["counted"], m["counted"], "decision", True)
    add(f"P4 {ck} BCa p (clip convention)", f"result.P4.contrasts.{ck}.p", r["p"], m["p_clip"], "p_clipconv")
    add(f"P4 {ck} Holm p (clip convention)", f"result.P4.contrasts.{ck}.p_holm", r["p_holm"], m["p_holm_clip"], "p_holm_clipconv")
    add(f"P4 {ck} counted (clip convention)", f"result.P4.contrasts.{ck}.counted", r["counted"], m["counted_clip"], "decision", True)

import sys
sys.path.insert(0, HERE)
from recompute_stats import holm as my_holm
ck3 = list(R3["cells"]); h3 = my_holm([R3["cells"][k]["p"] for k in ck3])
for k, h in zip(ck3, h3):
    add(f"P3 {k} Holm of reported raw p", f"result.P3.cells.{k}.p_holm", R3["cells"][k]["p_holm"], h, "holm_of_reported")
ck4 = list(R4["contrasts"]); h4 = my_holm([R4["contrasts"][k]["p"] for k in ck4])
for k, h in zip(ck4, h4):
    add(f"P4 {k} Holm of reported raw p", f"result.P4.contrasts.{k}.p_holm", R4["contrasts"][k]["p_holm"], h, "holm_of_reported")
add("P3 n keep", "result.P3.n_keep", R3["n_keep"], M3["n_keep"], "count", True)
add("P3 any opposite", "result.P3.any_opposite", R3["any_opposite"], M3["n_opposite"] > 0, "decision", True)
add("P4 n counted", "result.P4.n_counted", R4["n_counted"], M4["n_counted"], "count", True)
add("P4 n counted (clip convention)", "result.P4.n_counted", R4["n_counted"], M4["n_counted_clip"], "count", True)
for fam in ("P1", "P2"):
    for mod in ("DeepGaze IIE", "DeepGaze MSDB"):
        add(f"{fam} {mod} met (strict float p)", f"result.{fam}.{mod}.met", rep[fam][mod]["met"], mine[fam][mod]["met_strict"], "decision", True)
add("P3 n keep (strict float p)", "result.P3.n_keep", R3["n_keep"], M3["n_keep_strict"], "count", True)
json.dump(rows, open(os.path.join(HERE, "compare_stats.json"), "w"), indent=1)
by = {}
for r in rows:
    by.setdefault(r["kind"], []).append(r["abs_diff"])
for k, v in sorted(by.items()):
    print(f"{k:10s} n={len(v):4d} max|diff|={max(v):.3g}")
bad = [r for r in rows if r["decision"] and r["abs_diff"] != 0]
print("decision mismatches:", len(bad))
for r in bad:
    print("  ", r["claim"], r["reported"], r["recomputed"])
