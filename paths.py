"""
Single configurable root for every script in this repository.

Every location is repository-relative by default and can be moved with an environment variable:

  LAN_RESULTS   result tree, default <repo>/results. It mirrors the project tree in which the paper's numbers were
                computed, so the source comments in paper_numbers/output/numbers.tex (for example
                'v3/scores/results.json > models > ...') name files under this root. Scripts write their outputs here;
                bulky intermediates (saliency maps, feature caches, checkpoints, training states) go to subfolders
                that .gitignore excludes.
  LAN_STIMULI   rendered stimulus sets, default <repo>/stimuli (stim_v3/, stim_alt/, fill_variants/)
  LAN_WORK      training cache (meta.json, maps.npy) and logs, default <repo>/work
  LAN_WEIGHTS   model weights, default <repo>/weights:
                  layers/<dinov2|clip|siglip>/readout_L<L>.pt   18 layer-sweep readouts (release v1.0, layer_readouts.zip)
                  xsoyelnet_aspp.pt, readout_M610.pt             the two xSoyelNet readouts (not distributed)
  SAL_DATA      public fixation data, default <repo>/data: mit1003/ (ALLSTIMULI, ALLFIXATIONMAPS) and
                salicon/ (images/train, train/*.png, train/*.mat)
  CAT2000       CAT2000 training set, default $SAL_DATA/cat2000 (Stimuli/, FIXATIONMAPS/, FIXATIONLOCS/)
Pretrained backbones come from the Hugging Face and torch.hub caches (HF_HOME, TORCH_HOME).

Large per-image score files are stored compressed (<name>.json.xz). load_json() reads <name>.json if it exists and
otherwise <name>.json.xz (or .json.gz), so a rerun that writes a plain .json takes precedence over the shipped file.
"""
import gzip
import json
import lzma
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _env(name, default):
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


RESULTS = _env("LAN_RESULTS", REPO / "results")
V2 = RESULTS / "v2"                         # survey, local nulls, core AOI, Shapley decomposition, model anatomy
V3 = RESULTS / "v3"                         # rotation controls (scores) and the full-data readout swap (gpu/full)
PAPER = RESULTS / "paper"                   # seed_bootstrap.json
STIMULI = _env("LAN_STIMULI", REPO / "stimuli")
STIM_V3 = STIMULI / "stim_v3"               # 740 displays (540 Kanizsa, 200 Ehrenstein) with AOIs
STIM_ALT = STIMULI / "stim_alt"             # 360 alternating-rotation displays with AOIs
FILL_VARIANTS = STIMULI / "fill_variants"   # real_filled at 25 % and 75 % fill contrast (120 displays)
LEGACY = REPO / "legacy"
STIM_V2 = LEGACY / "stimuli" / "kanizsa"    # first-version Kanizsa set (byte-identity reference)
STIM_EHR = LEGACY / "stimuli" / "ehrenstein"
LEGACY_RESULTS = LEGACY / "results"         # first-version scores (cross-checks)
WORK = _env("LAN_WORK", REPO / "work")
TRAIN_CACHE = WORK / "train_cache"          # meta.json and maps.npy of the 12,803 training images
LOGS = WORK / "logs"
WEIGHTS = _env("LAN_WEIGHTS", REPO / "weights")
LAYER_READOUTS = WEIGHTS / "layers"
XSOYELNET_ASPP = WEIGHTS / "xsoyelnet_aspp.pt"
XSOYELNET_ML = WEIGHTS / "readout_M610.pt"
DATA = _env("SAL_DATA", REPO / "data")
CAT2000 = _env("CAT2000", DATA / "cat2000")
CODE = {k: REPO / k for k in ("models", "stimuli", "scoring", "analysis", "decomposition", "readout_swap",
                              "paper_numbers")}


def use(*folders):
    """Put code folders of this repository (keys of CODE, or paths relative to the repository) on sys.path."""
    for f in reversed(folders):
        p = str(CODE.get(f, REPO / f))
        if p not in sys.path:
            sys.path.insert(0, p)


def json_path(path):
    """The file load_json() would read for `path` (plain .json first, then .json.xz, then .json.gz), or None."""
    path = Path(path)
    for p in (path, path.with_name(path.name + ".xz"), path.with_name(path.name + ".gz")):
        if p.exists():
            return p
    return None


def json_exists(path):
    return json_path(path) is not None


def load_json(path):
    p = json_path(path)
    if p is None:
        raise FileNotFoundError(path)
    if p.suffix == ".xz":
        with lzma.open(p, "rt") as f:
            return json.load(f)
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            return json.load(f)
    with open(p) as f:
        return json.load(f)
