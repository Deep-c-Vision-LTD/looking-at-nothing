"""
Shared paths and loaders for nn/ (property package: rotation sensitivity vs effective receptive field).

Read-only inputs: the project tree (sweep.py, ab_objective.py, gate_kanizsa.py), v2/stim_v3, v3/stim_alt, v3/scores
(common_alt.py, analyze_alt.py, raw_v3maps/, raw/, raw_nulls.json, results.json), v2/scores (analyze_v3.py, raw/),
v2/anatomy/anatomy.json. Nothing outside nn/ is written: byte-code writing is disabled before any project module is
imported, so no __pycache__ appears in v2/ or v3/.
"""
import os, sys, json
sys.dont_write_bytecode = True
for _k, _v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[_k] = _v

NN = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(NN)
V2 = os.path.join(ROOT, "v2")
V3 = os.path.join(ROOT, "v3")
V3SC = os.path.join(V3, "scores")
V2SC = os.path.join(V2, "scores")
RES = os.path.join(NN, "results")
LOGS = os.path.join(NN, "logs")
STIM_V3 = os.path.join(V2, "stim_v3")
STIM_ALT = os.path.join(V3, "stim_alt")
RULE = os.path.join(NN, "ANALYSIS_RULE.md")
RULE_SHA = os.path.join(NN, "ANALYSIS_RULE.sha256")

LAYERS = {"dinov2": (2, 4, 6, 8, 10, 12), "clip": (4, 8, 12, 16, 20, 24), "siglip": (4, 8, 12, 16, 20, 24)}
NBLOCKS = {"dinov2": 12, "clip": 24, "siglip": 24}
PATCH = {"dinov2": 14, "clip": 14, "siglip": 16}
SCORED = {"dinov2": (2, 8, 12), "clip": (4, 12, 24), "siglip": (4, 8, 24)}         # scored in v2/v3 (Appendix B)
UNSCORED = {bb: tuple(L for L in LAYERS[bb] if L not in SCORED[bb]) for bb in LAYERS}
BBNAME = {"dinov2": "DINOv2-B", "clip": "CLIP-L", "siglip": "SigLIP-L"}           # names used in v3/scores/results.json
OBJECTIVE = {"dinov2": "self-supervised", "clip": "image-text", "siglip": "image-text"}
SIZES = (336, 448)


def tag(bb, L, s):
    return f"{bb}_L{L}_s{s}"


def rname(bb, L):
    return f"{BBNAME[bb]} L{L}"


def all_readouts():
    return [(bb, L) for bb in ("dinov2", "clip", "siglip") for L in LAYERS[bb]]


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rule_ok():
    """True if ANALYSIS_RULE.md still has the recorded SHA-256."""
    import hashlib
    h = hashlib.sha256(open(RULE, "rb").read()).hexdigest()
    rec = open(RULE_SHA).read().split()[0]
    return h == rec, h, rec


def import_v3():
    """common_alt and analyze_alt from v3/scores (analyze_alt imports v2/scores/analyze_v3); read-only imports."""
    for p in (V3SC, V2SC, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import common_alt as CA
    import analyze_alt as AA
    return CA, AA


def heldout_cc():
    d = json.load(open(os.path.join(V2, "anatomy", "anatomy.json")))
    out = {}
    for r in d["records"]:
        if r["model"].startswith("sweep "):
            _, bb, Ls = r["model"].split()
            out[(bb, int(Ls[1:]))] = r["heldout_mit1003_cc"]
    return out
