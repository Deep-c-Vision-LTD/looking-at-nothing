"""
neuro/reanalysis/gpu copy of nn/registers/eval_reg.py (= v3/gpu/eval_full.py, the published readout-swap evaluation).
Changes, nothing else:
  - shared definitions from recommon.py (adds dinov2reg and dinov2ms; with V3_DEVICE=cpu the ab_objective stub is used);
  - --ckpt-dir DIR: score checkpoints from another run (for example the published v3/gpu/full/ckpt or
    nn/registers/full/ckpt, read only) while writing scores under --out (inside neuro/); done markers are then not
    required; --tags a,b limits the heads; --skip-mit skips the MIT1003 scoring;
  - stimulus set "stim_rigid" = neuro/reanalysis/stim_rigid (whole-display rotation; AOIs per configuration and angle);
  - V3_MAPS defaults to "none" (no float16 maps are saved; the disk is nearly full). Scores are always computed from
    the float16 maps in memory, as published;
  - added 18 November 2026 (rule review, before the rule was fixed): require_public() before anything else; on
    stim_rigid the exploratory AOI variants of rule Section 6 (recommon.measures_extra) are stored with each display.
Docstring of the copied file follows.
nn/registers copy of v3/gpu/eval_full.py (the published readout-swap evaluation), unchanged except that the shared
definitions come from regcommon.py (adds the DINOv2-with-registers backbone `dinov2reg`; plain backbones keep the
published code path). Original docstring follows.
Evaluate the full-data readouts of one backbone (every trained head that is not yet scored).
  MIT1003 held-out 200: 336 px, centre bias on, sigmoid map resized to the image, CC and NSS (= v2 train.score_mit).
  Stimuli (v2/stim_v3 Kanizsa + Ehrenstein, and v3/stim_alt if it exists when this runs) at 336 and 448 px, centre
  bias off: sigmoid map at s x s -> float16 (saved as v2: maps/<stimset>/<tag>_px<s>.npz with maps, names) -> cast
  back to float32 and cv2-bilinear resized to the 1024 x 768 canvas (v2/readout/evaluate.py), then per image
    v2 measures: inside, inside_eroded, inside_eroded_all = mass(variant) / (mass(variant) + mass(ind))
                 (v2/readout/evaluate.rel, unchanged)
    v3 measures: r_inside, r_core30, f_core30, r_core45, f_core45, r_eroded_all, f_inside, f_ind
                 (v3/scores/common_alt.measures on common_alt.q16 of the resized map, i.e. Task 1's code; core<D> =
                 inside pixels with Euclidean distance to the inducer AOI > D px, v2/verify/core_aoi.py; taken from the
                 AOI file when it has them (stim_alt), else computed the same way and checked against the file)
Usage: python eval_full.py --bb dinov2 --out DIR [--stimsets stim_v3,stim_alt] [--mit-limit N] [--stim-limit N]
Env V3_MAPS (which float16 maps are saved; scores are always computed from the float16 maps in memory):
  "default": every model on the new stimulus set(s) only (anything but stim_v3), per the v3 disk rule (float16 maps
             only for the new stimuli); stim_v3 maps are scored in memory and not kept; they can be regenerated from the
             saved checkpoints with V3_MAPS=all eval_full.py --force (minutes per backbone on MPS)
  "all" | "none" | a comma list of stimsets (all seeds).
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from scipy import ndimage
from recommon import (make_extractor, V3, DEVICE, S, FSETS, BB_FSETS, Extractor, Loader, prep_batch, feats, make_head, read_rgb,
                        mit_heldout, CC, NSS, paths, log, sync, require_public, measures_extra)
require_public()                                             # REANALYSIS_RULE.md Section 8.1 (added 18 Nov 2026)
import common as C2
from evaluate import rel                                     # v2/readout/evaluate.py
sys.path.insert(0, os.path.join(V3, "scores"))
import common_alt as CA                                      # v3/scores/common_alt.py (Task 1): q16, measures

V2_VARIANTS = ("inside", "inside_eroded", "inside_eroded_all")

SIZES = (336, 448)
CORE_D = (30, 45)                                           # core AOIs as common_alt
BLOCK = 800
ap = argparse.ArgumentParser()
ap.add_argument("--bb", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--stimsets", default=None); ap.add_argument("--mit-limit", type=int, default=None)
ap.add_argument("--stim-limit", type=int, default=None); ap.add_argument("--force", action="store_true")
ap.add_argument("--ckpt-dir", default=None); ap.add_argument("--tags", default=None)
ap.add_argument("--skip-mit", action="store_true")
a = ap.parse_args()
P = paths(a.out)
LOG = os.path.join(P["logs"], f"eval_{a.bb}.log")
SAVE = os.environ.get("V3_MAPS", "none")


def save_maps(stimset, seed):
    if SAVE == "default":
        return stimset != "stim_v3"
    return SAVE == "all" or stimset in SAVE.split(",")


def stim_items(stimset):
    """v2/stim_v3 via v2 common.stim_items; v3/stim_alt: images in img/<name>.png, AOI path relative to the set."""
    if stimset == "stim_v3":
        return C2.stim_items("stim_v3")
    D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stim_rigid") if stimset == "stim_rigid" \
        else os.path.join(V3, stimset)
    items = json.load(open(os.path.join(D, "items.json")))
    out = []
    for it in items:
        aoi = it.get("aoi") or os.path.join("img", it["name"] + "_aoi.npz")
        out.append(dict(it, img=os.path.join(D, "img", it["name"] + ".png"), aoi=aoi if os.path.isabs(aoi) else os.path.join(D, aoi)))
    return out


def masks_for(items):
    """Per configuration: boolean AOIs for common_alt.measures (inside, ind, inside_eroded_all, core30, core45) and flat
    indices of the v2 inside variants. AOIs must be identical across a configuration's conditions (asserted, as v2
    evaluate.aoi_index); core AOIs present in the file must equal the v2/verify/core_aoi.py construction."""
    cache, out = {}, {}
    for it in items:
        key = it.get("config") or it["name"].rsplit("_", 1)[0]
        if key not in cache:
            A = np.load(it["aoi"]); ab = {k: A[k].astype(bool) for k in A.files}
            dd = ndimage.distance_transform_edt(~ab["ind"])
            for D in CORE_D:
                core = ab["inside"] & (dd > D)
                if f"core{D}" in ab:
                    assert np.array_equal(core, ab[f"core{D}"]), (key, D)
                ab[f"core{D}"] = core
            cache[key] = dict(shape=ab["inside"].shape, a=ab, ind=np.flatnonzero(ab["ind"].ravel()),
                              var={k: np.flatnonzero(ab[k].ravel()) for k in V2_VARIANTS if k in ab}, aoi=it["aoi"])
        elif it["aoi"] != cache[key]["aoi"]:
            A = np.load(it["aoi"])
            for k in A.files:
                assert np.array_equal(A[k].astype(bool), cache[key]["a"][k]), (it["name"], k)
        out[it["name"]] = cache[key]
    return out


def score_map(mp16, M, it=None):
    H, W = M["shape"]
    full = cv2.resize(mp16.astype(np.float32), (W, H))
    r = {v: rel(full.ravel(), idx, M["ind"]) for v, idx in M["var"].items()}
    q = CA.q16(full)
    r.update(CA.measures(q, M["a"]))
    if it is not None and "polygon" in M["a"]:                 # stim_rigid only (added 18 Nov 2026)
        r.update(measures_extra(q, M["a"], it))
    return r


def dump_atomic(obj, path):
    """Score files are read by a concurrent analysis: write to a temporary name, then rename."""
    json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def discover():
    out = []
    cdir = a.ckpt_dir or P["ckpt"]
    want = set(a.tags.split(",")) if a.tags else None
    for fs in BB_FSETS[a.bb]:
        for p in sorted(glob.glob(os.path.join(cdir, f"{fs}_R*_s*.pt"))):
            tag = os.path.basename(p)[:-3]
            if want is not None and tag not in want:
                continue
            if not a.ckpt_dir and not os.path.exists(os.path.join(P["done"], f"train_{tag}")):
                continue
            arch, seed = tag[len(fs) + 1:].split("_s")
            out.append(dict(tag=tag, fset=fs, arch=arch, seed=int(seed), ckpt=p))
    return out


def load_head(h):
    m = make_head(h["arch"], FSETS[h["fset"]]["cin"])
    m.load_state_dict(torch.load(h["ckpt"], map_location="cpu")); return m.to(DEVICE).eval()


@torch.no_grad()
def run_mit(ex, heads):
    todo = [h for h in heads if a.force or not os.path.exists(os.path.join(P["scores"], f"mit_{h['tag']}.json"))]
    if not todo:
        return
    items = mit_heldout()[:a.mit_limit] if a.mit_limit else mit_heldout()
    t0 = time.time(); fcache = {fs: [] for fs in {h["fset"] for h in todo}}
    for x in Loader(ex, [[p["image"] for p in items[b:b + 16]] for b in range(0, len(items), 16)], S):
        f = feats(ex, x, S)
        for fs in fcache:
            fcache[fs].append(f[fs].half().cpu())
    fcache = {fs: torch.cat(v) for fs, v in fcache.items()}
    gts = [cv2.imread(p["gt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in items]
    fxs = [(cv2.imread(p["fix"], cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32) for p in items]
    log(LOG, f"MIT features {len(items)} images {time.time() - t0:.0f}s")
    for h in todo:
        m = load_head(h); m.use_cb = True; cc, nss = [], []
        F32 = fcache[h["fset"]]
        for b in range(0, len(items), 16):
            sal = torch.sigmoid(m(F32[b:b + 16].float().to(DEVICE), (S, S)))[:, 0].cpu().numpy()
            for i in range(sal.shape[0]):
                gt = gts[b + i]; hh, ww = gt.shape
                s_ = cv2.resize(sal[i], (ww, hh))
                cc.append(CC(s_, gt)); nss.append(NSS(s_, fxs[b + i]))
        rec = dict(tag=h["tag"], fset=h["fset"], arch=h["arch"], seed=h["seed"], n=len(cc), CC=float(np.mean(cc)),
                   NSS=float(np.mean(nss)), cc_per_image=[round(float(c), 5) for c in cc], device=DEVICE,
                   features="float32 on the fly, held in float16 between backbone and readout")
        dump_atomic(rec, os.path.join(P["scores"], f"mit_{h['tag']}.json"))
        log(LOG, f"{h['tag']} MIT1003 held-out CC {rec['CC']:.4f} NSS {rec['NSS']:.4f} (n {len(cc)})")


@torch.no_grad()
def run_stim(ex, heads, stimset):
    todo = [h for h in heads if a.force or not os.path.exists(os.path.join(P["scores"], f"{stimset}_{h['tag']}.json"))]
    if not todo:
        return
    items = stim_items(stimset)
    if a.stim_limit:
        items = items[:a.stim_limit]
    names = [it["name"] for it in items]
    M = masks_for(items)
    os.makedirs(os.path.join(P["maps"], stimset), exist_ok=True)
    res = {h["tag"]: {} for h in todo}
    models = {h["tag"]: load_head(h) for h in todo}
    for m in models.values():
        m.use_cb = False
    for s in SIZES:
        for h in todo:
            res[h["tag"]][str(s)] = {}
        blocks = [list(range(b, min(b + BLOCK, len(items)))) for b in range(0, len(items), BLOCK)]
        for k, blk in enumerate(blocks):
            t0 = time.time(); fc = {fs: [] for fs in {h["fset"] for h in todo}}
            for x in Loader(ex, [[items[i]["img"] for i in blk[b:b + 16]] for b in range(0, len(blk), 16)], s):
                f = feats(ex, x, s)
                for fs in fc:
                    fc[fs].append(f[fs].half().cpu())
            fc = {fs: torch.cat(v) for fs, v in fc.items()}
            log(LOG, f"{stimset} px{s} block {k + 1}/{len(blocks)} features {len(blk)} images {time.time() - t0:.0f}s")
            for h in todo:
                m = models[h["tag"]]; mp = np.zeros((len(blk), s, s), np.float16)
                for b in range(0, len(blk), 16):
                    mp[b:b + 16] = torch.sigmoid(m(fc[h["fset"]][b:b + 16].float().to(DEVICE), (s, s)))[:, 0].cpu().numpy().astype(np.float16)
                if save_maps(stimset, h["seed"]):
                    suf = "" if len(blocks) == 1 else f"_part{k}"
                    np.savez_compressed(os.path.join(P["maps"], stimset, f"{h['tag']}_px{s}{suf}.npz"), maps=mp,
                                        names=np.array([names[i] for i in blk]))
                R = res[h["tag"]][str(s)]
                for j, i in enumerate(blk):
                    for key, v in score_map(mp[j], M[names[i]], items[i] if stimset == "stim_rigid" else None).items():
                        R.setdefault(key, {})[names[i]] = v
            del fc
    for h in todo:
        dump_atomic(res[h["tag"]], os.path.join(P["scores"], f"{stimset}_{h['tag']}.json"))
        log(LOG, f"{stimset} {h['tag']} scored {len(names)} images x {len(SIZES)} sizes (maps saved: {save_maps(stimset, h['seed'])})")


if __name__ == "__main__":
    heads = discover()
    if not heads:
        log(LOG, f"no trained heads for {a.bb}"); sys.exit(0)
    stimsets = a.stimsets.split(",") if a.stimsets else ["stim_v3"] + (["stim_alt"] if os.path.exists(os.path.join(V3, "stim_alt", "items.json")) else [])
    log(LOG, f"=== eval {a.bb} on {DEVICE}: {len(heads)} heads, stimsets {stimsets}, maps saved for: {SAVE}")
    ex = make_extractor(a.bb)
    if not a.skip_mit:
        run_mit(ex, heads)
    for st in stimsets:
        try:
            run_stim(ex, heads, st)
        except Exception as e:                                  # a malformed new stimulus set must not stop the queue
            if st == "stim_v3":
                raise
            log(LOG, f"FAILED {st}: {type(e).__name__}: {e}")
    log(LOG, f"=== eval {a.bb} done")
