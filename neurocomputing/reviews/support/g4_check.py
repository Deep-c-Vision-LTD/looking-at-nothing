"""g4_check.py: coverage of the rule's WCU interval with G = 4 clusters (rule Section 6 item 18, A10), synthetic only."""
import sys
sys.argv = [sys.argv[0]]
import numpy as np
import independent_check as I
out = []
for kind in ("sym", "skew", "lever"):
    cov, size, w = I.coverage(4, kind, 300, seed=404 + len(kind), invert=False)
    s = f"G = 4 {kind:5s} n = 300: WCU {cov['wcu']:.3f} | CR1 t(3) {cov['t']:.3f} | two-stage percentile {cov['pct']:.3f} | WCR size {size:.3f} | median width WCU {w['wcu']:.3f}, t(3) {w['t']:.3f}"
    print(s, flush=True); out.append(s)
open("g4_check_output.txt", "w").write("\n".join(out) + "\n")
