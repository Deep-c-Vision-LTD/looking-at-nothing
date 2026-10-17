"""Shared definitions for the stim_alt scoring (results in <results>/v3/scores): AOIs, per-image measures, the condition
lists and the map quantisation.

Measures (map clipped at 0, float64 sums; the map is first normalised by its maximum and rounded to float16, exactly the
form in which v2/scores saved its maps, so stim_v3 measures computed from the v2 maps and stim_alt measures computed at
scoring time go through the same quantisation):
  r_inside     = M(inside) / (M(inside) + M(ind))                 the published measure (v2: gate_kanizsa.rel_inside)
  r_core30     = M(core30) / (M(core30) + M(ind))                 analysis/core_aoi.py "r_core" with D = 30
  f_core30     = M(core30) / M(all pixels)                        analysis/core_aoi.py "core" share of map mass
  r_core45, f_core45                                              same with D = 45 (sensitivity, as v2 PV2)
  r_eroded_all = M(inside_eroded_all) / (M(inside_eroded_all) + M(ind))
  f_inside, f_ind                                                 shares of total map mass
core<D> = inside-AOI pixels whose Euclidean distance to the inducer-disc AOI (ind) is > D px
(scipy distance_transform_edt(~ind) > D), exactly as analysis/core_aoi.py.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as LP          # noqa: E402

V3 = str(LP.V3); V2 = str(LP.V2)                          # result trees
STIM_ALT = str(LP.STIM_ALT); STIM_V3 = str(LP.STIM_V3)
V3_CONDS = ("illusory", "control180", "control90", "real", "real_filled", "mis10", "mis20", "mis30")
ALT_CONDS = tuple(f"alt{t}{ph}" for t in (10, 20, 30) for ph in ("a", "b"))
MEASURES = ("r_inside", "r_core30", "f_core30", "r_core45", "f_core45", "r_eroded_all", "f_inside", "f_ind")

ALT_ITEMS = json.load(open(os.path.join(STIM_ALT, "items.json")))
CONFIGS = sorted({it["config"] for it in ALT_ITEMS})
assert len(CONFIGS) == 60

_aoi = {}
def aoi(cfg):
    if cfg not in _aoi:
        a = np.load(os.path.join(STIM_ALT, "aoi", cfg + "_aoi.npz"))
        _aoi[cfg] = {k: a[k] for k in a.files}
    return _aoi[cfg]


def q16(sal):
    """Map as v2/scores saved it: sal / max|sal| in float16 (scale dropped; all measures are scale free)."""
    sal = np.asarray(sal, np.float32)
    mx = float(np.abs(sal).max())
    return (sal / (mx if mx > 0 else 1.0)).astype(np.float16)


def measures(sal, a):
    s = np.clip(np.asarray(sal, np.float32).astype(np.float64), 0, None); tot = s.sum()
    I, D, EA = s[a["inside"]].sum(), s[a["ind"]].sum(), s[a["inside_eroded_all"]].sum()
    C30, C45 = s[a["core30"]].sum(), s[a["core45"]].sum()
    rr = lambda x: float(x / (x + D)) if x + D > 0 else float("nan")
    return dict(r_inside=rr(I), r_core30=rr(C30), f_core30=float(C30 / tot), r_core45=rr(C45), f_core45=float(C45 / tot),
                r_eroded_all=rr(EA), f_inside=float(I / tot), f_ind=float(D / tot))
