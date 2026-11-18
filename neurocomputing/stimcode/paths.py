"""
Shim for the verbatim copies of looking-at-nothing/stimuli/stimuli_{v2,v3,ehr}.py in this folder.
The copies import `paths` only for their output folders, which render_rigid.py never uses (it never calls their
make()). Every location points inside neuro/reanalysis/stimcode/_never_written, so even an accidental make() could
not write outside neuro/.
"""
from pathlib import Path

_BASE = Path(__file__).resolve().parent / "_never_written"
STIM_V3 = _BASE / "stim_v3"
STIM_ALT = _BASE / "stim_alt"
STIM_V2 = _BASE / "stim_v2"
STIM_EHR = _BASE / "stim_ehr"
