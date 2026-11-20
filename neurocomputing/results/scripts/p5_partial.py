"""
p5_partial.py: P5 of REANALYSIS_RULE.md Section 5.5 as far as it can be computed before DeepGaze MSDB has been scored
on stim_rigid (RUN_PLAN.md R6 still to run). Written 18 November 2026; not a hashed file; changes nothing in the rule.

What it does, with the fixed engine only (analyze_re.py is imported, never edited):
  1. analyze_re.require_public() first (rule Section 8.1).
  2. analyze_re.step_primary({"P5"}), the exact code path of RUN_PLAN.md R7. That call computes P1, P2 and P5. While
     scores_rigid/DeepGaze_MSDB.json does not exist the engine marks MSDB "not computable" and evaluates the abstract
     rule of Section 5.5 with MSDB outside category 1. Neither is a result here: MSDB is PENDING (its scoring has not
     run, it did not fail), so this script replaces MSDB's P5 entry by a pending note and moves the abstract-rule keys
     (both_category_1, abstract_leads_with_alternation, abstract_rotation) out of the result into "withheld", marked
     not decided. DeepGaze IIE's P5 entry is kept exactly as the engine returned it. Every random draw of the engine is
     seeded per call (RandomState(2026); the two-stage indices are fixed by Design), so IIE's values will be identical
     in the final R7 run.
  3. The 15 nulls at their own values (rule Section 5.5, "Secondary quantities, reported for every model and null"):
     for each null, L_uni, L_rig, A*, C* and the three secondary quantities with WCU intervals and WCR p-values, and
     the rigid share s* (BCa, stability flag, denominator check, Fieller), with the engine's functions. Exploratory
     bookkeeping only; P5's decision uses the model-minus-null differences that step 2 computes.
Output: results/primary_P5_partial.json
Run: nice -n 10 env OMP_NUM_THREADS=4 $PY -B results/scripts/p5_partial.py   (numpy and scipy only)
"""
import json, os, sys, time
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RE)
import analyze_re as A          # noqa: E402

n_hashed = A.require_public()
rig = os.path.join(RE, "scores_rigid")
assert not os.path.exists(os.path.join(rig, "DeepGaze_MSDB.json")), "MSDB scores exist: run analyze_re.py primary --which P5"
assert os.path.exists(os.path.join(rig, "DeepGaze_IIE.json")) and os.path.exists(os.path.join(rig, "nulls.json"))

t0 = time.time()
res = A.step_primary({"P5"})
p5 = res["P5"]
assert p5["DeepGaze MSDB"].get("computable") is False and p5["DeepGaze IIE"].get("computable") is True
withheld = {k: p5.pop(k) for k in ("both_category_1", "abstract_leads_with_alternation")}
withheld["abstract_rotation"] = res.pop("abstract_rotation", None)
p5["DeepGaze MSDB"] = dict(status="PENDING", note="DeepGaze MSDB has not been scored on stim_rigid yet (RUN_PLAN.md R6 "
                           "launched after this script); its P5 waits. This is not the rule's 'not computable'.")

# the 15 nulls at their own values
meta = A.configs_meta(); D = A.Design(meta)
N, NR = A.null_scores(), A._load(os.path.join(rig, "nulls.json"))
nulls_own = {}
for k in A.NULLS:
    v = A.getter({**N[k], **NR[k]})
    L_uni, L_rig, Aal, sec = A.p5_parts(D, v, 20)
    share = A.bca(lambda a, b: a / b, [L_rig, L_uni], D.idx)
    wl = A.wild_cluster(L_uni.ravel(), D.g)
    tden = A.t_den(L_uni.ravel(), D.g)
    nulls_own[k] = dict(L_uni=wl, L_rig=A.wild_cluster(L_rig.ravel(), D.g), align_effect=A.wild_cluster(Aal.ravel(), D.g),
                        half_criterion=A.wild_cluster((L_uni - 2 * L_rig).ravel(), D.g),
                        rigid_share=dict(share, t_den=tden, stable=bool(tden >= A.T_STABLE),
                                         denominator_excludes_zero=bool(wl["ci"][0] > 0 or wl["ci"][1] < 0),
                                         fieller=A.fieller(L_rig, L_uni, D.g)),
                        secondary={s: A.wild_cluster(x.ravel(), D.g) for s, x in sec.items()})

out = dict(note="PARTIAL P5 (rule Section 5.5): DeepGaze IIE and the 15 nulls only; DeepGaze MSDB pending; the abstract "
                "rule of Section 5.5 is NOT decided (it needs MSDB's category). P1 and P2 are recomputed by the same "
                "engine call and must equal out/primary_P1_P2_P3_P4.json.",
           theta=20, hashed_files_verified=n_hashed, public_push=open(os.path.join(RE, "PUBLIC_PUSH.txt")).read().split()[0],
           engine_sha256=A.sha256(os.path.join(RE, "analyze_re.py")), script_sha256=A.sha256(os.path.abspath(__file__)),
           result=res, nulls_own_values=nulls_own,
           withheld_not_decided=withheld, wall_s=round(time.time() - t0, 1))


def conv(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


p = os.path.join(RE, "results", "primary_P5_partial.json")
json.dump(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **out), open(p, "w"), indent=1, default=conv)
print("wrote", p)
