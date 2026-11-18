"""
Copy of v3/scores/common_alt.py for the whole-display rotation set (neuro/reanalysis/stim_rigid). q16() and measures()
are verbatim; only the stimulus set changes. AOIs are per configuration and rotation angle (items.json field "config",
e.g. square_sr30_dark_p0_rig20), identical for the illusory and control180 display of that angle; the orientation-
matched controls of the uniform rotation (config e.g. square_sr30_dark_p0_mis20c, one display each) carry the stim_v3
AOIs of their configuration.
Added on 18 November 2026 after the rule review, before the rule was fixed:
  - require_public(): every scoring and analysis script calls it before any display is scored; it refuses unless
    REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt exist, every file hashed in REANALYSIS_RULE.sha256 is unchanged, and
    every stimulus file matches its md5 in stim_rigid/manifest.json (REANALYSIS_RULE.md Section 8.1);
  - measures_extra(): the exploratory AOI variants of rule Section 6 (items 4, 10, 11) computed in the same scoring
    pass, so that no rigid-rotation map has to be stored (design review, finding 8, option b).
Read only: stim_rigid/ (this package), v2/stim_v3 (pipeline checks). Nothing outside neuro/ is written by importers.
"""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RE = os.path.dirname(HERE)                                     # neuro/reanalysis
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(RE))   # project root
V2, V3 = os.path.join(ROOT, "v2"), os.path.join(ROOT, "v3")
STIM_V3 = os.path.join(V2, "stim_v3")
STIM_RIGID = os.path.join(RE, "stim_rigid")
SCORES = os.path.join(RE, "scores_rigid")
LOGS = os.path.join(RE, "logs")
MEASURES = ("r_inside", "r_core30", "f_core30", "r_core45", "f_core45", "r_eroded_all", "f_inside", "f_ind")
N_ITEMS, N_CONFIGS = 660, 420          # 480 rigid displays (240 AOI configs) + 180 mis<theta>_control180 (180 configs)

RIGID_ITEMS = json.load(open(os.path.join(STIM_RIGID, "items.json")))
RCONFIGS = sorted({it["config"] for it in RIGID_ITEMS})
assert len(RIGID_ITEMS) == N_ITEMS and len(RCONFIGS) == N_CONFIGS, (len(RIGID_ITEMS), len(RCONFIGS))


def _digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_public(re_dir=RE):
    """Refuse to run unless the rule is fixed and public and nothing fixed with it has changed (rule Section 8.1)."""
    rec, pub = os.path.join(re_dir, "REANALYSIS_RULE.sha256"), os.path.join(re_dir, "PUBLIC_PUSH.txt")
    if not (os.path.exists(rec) and os.path.exists(pub)):
        sys.exit("refusing: REANALYSIS_RULE.sha256 and PUBLIC_PUSH.txt must exist (rule fixed and public first)")
    tok = open(pub).read().split()
    if not tok or len(tok[0]) < 7 or any(c not in "0123456789abcdef" for c in tok[0]):
        sys.exit("refusing: PUBLIC_PUSH.txt must start with the public commit hash")
    n = 0
    for line in open(rec):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            p = os.path.join(re_dir, parts[1])
            if not os.path.exists(p) or _digest(p, "sha256") != parts[0]:
                sys.exit(f"refusing: {parts[1]} is missing or changed after the rule was fixed")
            n += 1
    if n == 0:
        sys.exit("refusing: REANALYSIS_RULE.sha256 lists no file")
    man = json.load(open(os.path.join(re_dir, "stim_rigid", "manifest.json")))
    bad = [f for f, m in man["files"].items()
           if not os.path.exists(os.path.join(re_dir, "stim_rigid", f)) or _digest(os.path.join(re_dir, "stim_rigid", f), "md5") != m]
    if bad:
        sys.exit(f"refusing: {len(bad)} stimulus files differ from stim_rigid/manifest.json (first: {bad[0]})")
    return n


_aoi = {}
def aoi(rcfg):
    if rcfg not in _aoi:
        a = np.load(os.path.join(STIM_RIGID, "aoi", rcfg + "_aoi.npz"))
        _aoi[rcfg] = {k: a[k] for k in a.files}
    return _aoi[rcfg]


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


def cond_suffix(a, it):
    """Key suffix of the per-condition AOIs (ink_<s>, notch_<s>): the base condition for the rigid displays
    (ink_illusory, ink_control180), the condition itself for mis<theta>_control180."""
    return it["base_cond"] if f"ink_{it['base_cond']}" in a else it["cond"]


def measures_extra(sal, a, it):
    """Exploratory AOI variants of REANALYSIS_RULE.md Section 6, on the same map as measures():
      r_N (item 11, region variant): M(P) / (M(P) + M(D minus P)), P = whole contour area (`polygon`), D = discs;
      r_K (item 10, Kavsek-style): inner = P for illusory displays, P minus D for control displays;
           r_K = M(inner) / (M(inner) + M(D minus inner));
      f_polygon, f_notch, f_ink (item 4, log mass): share of the map's total mass in P, in the notch sectors minus ink
           and in the inducer ink of this display's condition (log m_R is taken in the analysis)."""
    s = np.clip(np.asarray(sal, np.float32).astype(np.float64), 0, None); tot = s.sum()
    P, D = a["polygon"].astype(bool), a["ind"].astype(bool)
    suf = cond_suffix(a, it)
    inner = (P & ~D) if "control" in it["cond"] else P
    MP, MDnP, Mi, MDni = s[P].sum(), s[D & ~P].sum(), s[inner].sum(), s[D & ~inner].sum()
    rr = lambda x, y: float(x / (x + y)) if x + y > 0 else float("nan")
    return dict(r_N=rr(MP, MDnP), r_K=rr(Mi, MDni), f_polygon=float(MP / tot),
                f_notch=float(s[a[f"notch_{suf}"].astype(bool)].sum() / tot),
                f_ink=float(s[a[f"ink_{suf}"].astype(bool)].sum() / tot))
