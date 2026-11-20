"""
v_s3_extra.py: two spot checks of S3 cross-check claims, from the raw score files with this folder's own loaders.
(1) D9 (i): a published profile loss L(theta) that differs from the computed L at 4 decimals equals the difference of
    the two effects E(illusory) and E(mis theta), each rounded to 4 decimals (checked for dinov2_L12 R1n theta 10 and
    dinov2_L12 R4 theta 20 against known_values.json as listed in explore_saved.json).
(2) the A10 mean of the three support-ratio means equals the all-60 E for two sweep readouts (equal cell sizes).
Written 19 November 2026. require_public() first. numpy only. Output: v_s3_extra.json
"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V                                                       # noqa: E402

V.guard()
import json                                                               # noqa: E402
import numpy as np                                                        # noqa: E402
import v_s3_saved as S                                                    # noqa: E402  (this folder's loaders)


def main():
    st = V.jl(os.path.join(V.RE, "results", "explore", "explore_saved.json"))["result"]["crosschecks"]
    rows = {(r["cell"], r["theta"]): r for r in st["published_values"]["L_mismatches"]["rows"]}
    out = {"L_mismatch_rows": []}
    for (f, a, th) in (("dinov2_L12", "R1n", 10), ("dinov2_L12", "R4", 20)):
        sc = S.head(f, a)
        e_ill = float(S.E(sc).mean()); e_mis = float(S.E(sc, f"mis{th}").mean())
        L = e_ill - e_mis
        r = rows[(f"{f}|{a}", th)]
        out["L_mismatch_rows"].append(dict(cell=f"{f}|{a}", theta=th, L=L, L_rounded=round(L, 4),
                                           rounded_components=round(round(e_ill, 4) - round(e_mis, 4), 4),
                                           published=r["published"],
                                           explained=abs(round(round(e_ill, 4) - round(e_mis, 4), 4) - r["published"]) < 1e-9))
    a10 = {}
    for bb, L in (("clip", 16), ("siglip", 24)):
        sc = S.sweep(bb, L)
        m = [float(V.per_config(V.design(sr)[1], lambda c: S.r(sc, c, "illusory") - S.r(sc, c, "control180")).mean())
             for sr in (0.3, 0.5, 0.7)]
        a10[f"{bb}_L{L}"] = dict(mean_of_sr_means=float(np.mean(m)), all60=float(S.E(sc).mean()))
    out["a10_mean_of_sr_means"] = a10
    with open(os.path.join(V.HERE, "v_s3_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
