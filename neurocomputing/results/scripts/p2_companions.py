"""
p2_companions.py: the P2 companions of REANALYSIS_RULE.md Section 5.2 and Section 6 item 20 (EXPLORATORY), computed with
the fixed engine (analyze_re.py imported, never edited). Written 18 November 2026; not a hashed file.

Rule text (Section 6, item 20): "P2 companions: the pooled and triangle Delta(20) of IIE and MSDB (Section 5.2), and the
square drop ratios with their flags."
Section 5.2: pooled and triangle Delta(20) "are reported next to P2 in the Results with WCU intervals (12 and 6 clusters),
labelled exploratory"; the square drop ratios D(20) = (r(illusory) - r(alt20)) / (r(illusory) - r(mis20)) "appear in the
Results only if their denominator passes the screen of Section 4.3 (|t| >= 4 with the CR1 se, WCU interval excluding 0);
otherwise they are given with the flag 'unstable'"; "Null ratios are given only for nulls whose denominator passes the
screen".
Implementation: Delta(20) per configuration = r(alt20) - r(mis20) (alt20 = mean of phases a and b), analyze_re.wild_cluster
(12 clusters pooled with 9,999 Webb draws; 6 clusters per shape with full enumeration). Square drop ratio: BCa (two-stage
pairs resample of the square design, 10,000, RandomState(2026)) via analyze_re.bca, Fieller via analyze_re.fieller,
denominator |t| via analyze_re.t_den and its WCU interval via analyze_re.wild_cluster. Unadjusted (exploratory).
Output: results/p2_companions_exploratory.json
"""
import json, os, sys, time
RE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RE)
import analyze_re as A          # noqa: E402

n_hashed = A.require_public()
meta = A.configs_meta()
D, Dsq, Dtri = A.Design(meta), A.Design(meta, "square"), A.Design(meta, "triangle")
models, nulls = A.load_survey_and_nulls()
units = list(models.items()) + [("null " + k, v) for k, v in nulls.items()]
out = {}
for name, v in units:
    rec = {}
    for lab, des in (("pooled_12_clusters", D), ("square_6_clusters", Dsq), ("triangle_6_clusters", Dtri)):
        rec["Delta20_" + lab] = A.wild_cluster(A.per_config(des, v, "alt20", "mis20").ravel(), des.g)
    num = A.per_config(Dsq, v, "illusory", "alt20")
    den = A.per_config(Dsq, v, "illusory", "mis20")
    w = A.wild_cluster(den.ravel(), Dsq.g)
    tden = A.t_den(den.ravel(), Dsq.g)
    passes = bool(tden >= A.T_STABLE and (w["ci"][0] > 0 or w["ci"][1] < 0))
    rec["square_uniform_loss_denominator"] = dict(w, t_den=tden)
    rec["square_drop_ratio_D20"] = dict(
        A.bca(lambda a, b: a / b, [num, den], Dsq.idx), fieller=A.fieller(num, den, Dsq.g),
        stable=bool(tden >= A.T_STABLE), denominator_excludes_zero=bool(w["ci"][0] > 0 or w["ci"][1] < 0),
        passes_screen=passes,
        flag=None if passes else ("unstable" if (w["ci"][0] > 0 or w["ci"][1] < 0) else "not reportable"),
        report=("yes" if passes else ("with the flag 'unstable'" if not name.startswith("null") else
                                      "no (null ratios only when the denominator passes the screen)")))
    out[name] = rec


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


p = os.path.join(RE, "results", "p2_companions_exploratory.json")
json.dump(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               label="EXPLORATORY (REANALYSIS_RULE.md Section 6 item 20; unadjusted)", hashed_files_verified=n_hashed,
               public_push=open(os.path.join(RE, "PUBLIC_PUSH.txt")).read().split()[0],
               engine_sha256=A.sha256(os.path.join(RE, "analyze_re.py")), script_sha256=A.sha256(os.path.abspath(__file__)),
               result=out), open(p, "w"), indent=1, default=conv)
print("wrote", p)
