#!/usr/bin/env python3
"""
paper_numbers/check_seed_bootstrap.py: sensitivity of the readout-swap intervals to seed-to-seed training variance.

The intervals in the manuscript (v3/gpu/full/results_full.json, readout_swap/analyze_full.py) average the seeds within each
configuration before the cluster bootstrap, so they describe consistency across displays, not across training runs.
This check repeats the bootstrap with a second level: in every resample the seeds of a cell are also drawn with
replacement (the scheme of the earlier pilot check, applied to the full-data runs). It reports, for every
feature-set x readout cell with scores, the primary ("plan") interval, the two-level interval and whether the two-level
interval changes the interval's position relative to zero.

Reads <results>/v3/gpu/full/scores/stim_v3_<fs>_<arch>_s<seed>.json(.xz) (per-image r of the full-data runs) and the
stimulus design of readout_swap/pilot/analyze.py. Writes <results>/paper/seed_bootstrap.json, which make_numbers.py reads.
Run: python paper_numbers/check_seed_bootstrap.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "readout_swap" / "pilot"))
import analyze as A2  # noqa: E402  (readout_swap/pilot/analyze.py: design, effects, B)
from common import LP  # noqa: E402

SC = LP.V3 / "gpu" / "full" / "scores"
FSETS = ("dinov2_L12", "dinov2_L6", "siglip_L24", "iie_shapenetc")
ARCHS = ("R1n", "R1", "R2", "R3", "R4")
SEEDS = (0, 1, 2)
MEASURES = (("inside", "illusory", "control180"), ("r_core30", "illusory", "control180"))

D = A2.design("stim_v3")["kanizsa"]
K, P = len(D["cells"]), len(D["cells"][0])
B = A2.B
rng = np.random.RandomState(2026)
ci = rng.randint(0, K, (B, K))
pi = rng.randint(0, P, (B, K, P))
n_items = len(A2.stim_items("stim_v3"))


def position(lo, hi):
    return "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")


out = dict(scheme="two-level cluster bootstrap: cells with replacement, positions within cells, seeds with replacement "
                  "per resample (RandomState(2026), %d resamples); plan scheme: seeds averaged first" % B,
           cells={})
for fs in FSETS:
    for ar in ARCHS:
        scs = []
        for s in SEEDS:
            p = SC / f"stim_v3_{fs}_{ar}_s{s}.json"
            if LP.json_exists(p):
                sc = LP.load_json(p)
                if len(sc["336"]["inside"]) == n_items:
                    scs.append(sc)
        if not scs:
            continue
        n = len(scs)
        si = np.random.RandomState(2027).randint(0, n, (B, n))
        rec = dict(n_seeds=n)
        for var, test, ctrl in MEASURES:
            per = np.array([A2.effects(sc, D, test, ctrl, "avg", var) for sc in scs])        # [n, K, P]
            plan = per.mean(0)[ci[:, :, None], pi].mean((1, 2))
            two = np.stack([per[si[:, j]][np.arange(B)[:, None, None], ci[:, :, None], pi] for j in range(n)]).mean(0).mean((1, 2))
            cp = np.percentile(plan, [2.5, 97.5]); ct = np.percentile(two, [2.5, 97.5])
            rec[var] = dict(mean=round(float(per.mean()), 4), ci_plan=[round(float(x), 4) for x in cp],
                            ci_two_level=[round(float(x), 4) for x in ct], position_plan=position(*cp),
                            position_two_level=position(*ct), changed=position(*cp) != position(*ct))
        out["cells"][f"{fs} {ar}"] = rec
        print(fs, ar, n, {v: (rec[v]["ci_plan"], rec[v]["ci_two_level"], rec[v]["changed"]) for v, _, _ in MEASURES}, flush=True)
out["n_cells"] = len(out["cells"])
out["n_changed"] = sum(1 for r in out["cells"].values() for v, _, _ in MEASURES if r[v]["changed"])
out["n_intervals"] = sum(len(MEASURES) for _ in out["cells"])
LP.PAPER.mkdir(parents=True, exist_ok=True)
json.dump(out, open(LP.PAPER / "seed_bootstrap.json", "w"), indent=1)
print(f"# {out['n_changed']} of {out['n_intervals']} intervals change position relative to 0 under the two-level bootstrap")
