"""
Evaluate the full-data readouts of one backbone (every trained head that is not yet scored).
  MIT1003 held-out 200: 336 px, centre bias on, sigmoid map resized to the image, CC and NSS (= pilot train.score_mit).
  Stimuli (stim_v3 Kanizsa + Ehrenstein, and stim_alt if it exists when this runs) at 336 and 448 px, centre
  bias off: sigmoid map at s x s -> float16 (saved as the pilot: maps/<stimset>/<tag>_px<s>.npz with maps, names) ->
  cast back to float32 and cv2-bilinear resized to the 1024 x 768 canvas (pilot/evaluate.py), then per image
    v2 measures: inside, inside_eroded, inside_eroded_all = mass(variant) / (mass(variant) + mass(ind))
                 (pilot/evaluate.rel, unchanged)
    v3 measures: r_inside, r_core30, f_core30, r_core45, f_core45, r_eroded_all, f_inside, f_ind
                 (scoring/common_alt.measures on common_alt.q16 of the resized map; core<D> = inside pixels with
                 Euclidean distance to the inducer AOI > D px, analysis/core_aoi.py; taken from the AOI file when it
                 has them (stim_alt), else computed the same way and checked against the file)
Usage: python eval_full.py --bb dinov2 [--out DIR] [--stimsets stim_v3,stim_alt] [--mit-limit N] [--stim-limit N]
Env V3_MAPS (which float16 maps are saved; scores are always computed from the float16 maps in memory):
  "default": every model on the new stimulus set(s) only (anything but stim_v3); stim_v3 maps are scored in memory and
             not kept; they can be regenerated from the saved checkpoints with V3_MAPS=all eval_full.py --force
  "all" | "none" | a comma list of stimsets (all seeds).
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
from scipy import ndimage
from fullcommon import (make_extractor, V3, DEVICE, S, FSETS, BB_FSETS, Extractor, Loader, prep_batch, feats, make_head, read_rgb,
                        mit_heldout, CC, NSS, paths, log, sync, OUT_DEFAULT, LP)
import common as C2
from evaluate import rel                                     # readout_swap/pilot/evaluate.py
sys.path.insert(0, os.path.join(str(LP.REPO), "scoring"))
import common_alt as CA                                      # scoring/common_alt.py: q16, measures

V2_VARIANTS = ("inside", "inside_eroded", "inside_eroded_all")

SIZES = (336, 448)
CORE_D = (30, 45)                                           # core AOIs as common_alt
BLOCK = 800
ap = argparse.ArgumentParser()
ap.add_argument("--bb", required=True); ap.add_argument("--out", default=OUT_DEFAULT)
ap.add_argument("--stimsets", default=None); ap.add_argument("--mit-limit", type=int, default=None)
ap.add_argument("--stim-limit", type=int, default=None); ap.add_argument("--force", action="store_true")
a = ap.parse_args()
P = paths(a.out)
LOG = os.path.join(P["logs"], f"eval_{a.bb}.log")
SAVE = os.environ.get("V3_MAPS", "default")


def save_maps(stimset, seed):
    if SAVE == "default":
        return stimset != "stim_v3"
    return SAVE == "all" or stimset in SAVE.split(",")


def stim_items(stimset):
    """stim_v3 via pilot common.stim_items; stim_alt: images in img/<name>.png, AOI path relative to the set."""
    if stimset == "stim_v3":
        return C2.stim_items("stim_v3")
    D = os.path.join(str(LP.STIMULI), stimset)
    items = json.load(open(os.path.join(D, "items.json")))
    out = []
    for it in items:
        aoi = it.get("aoi") or os.path.join("img", it["name"] + "_aoi.npz")
        out.append(dict(it, img=os.path.join(D, "img", it["name"] + ".png"), aoi=aoi if os.path.isabs(aoi) else os.path.join(D, aoi)))
    return out


def masks_for(items):
    """Per configuration: boolean AOIs for common_alt.measures (inside, ind, inside_eroded_all, core30, core45) and flat
    indices of the v2 inside variants. AOIs must be identical across a configuration's conditions (asserted, as v2
    evaluate.aoi_index); core AOIs present in the file must equal the analysis/core_aoi.py construction."""
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


def score_map(mp16, M):
    H, W = M["shape"]
    full = cv2.resize(mp16.astype(np.float32), (W, H))
    r = {v: rel(full.ravel(), idx, M["ind"]) for v, idx in M["var"].items()}
    r.update(CA.measures(CA.q16(full), M["a"]))
    return r


def dump_atomic(obj, path):
    """Score files are read by a concurrent analysis: write to a temporary name, then rename."""
    json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def discover():
    out = []
    for fs in BB_FSETS[a.bb]:
        for p in sorted(glob.glob(os.path.join(P["ckpt"], f"{fs}_R*_s*.pt"))):
            tag = os.path.basename(p)[:-3]
            if not os.path.exists(os.path.join(P["done"], f"train_{tag}")):
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
                    for key, v in score_map(mp[j], M[names[i]]).items():
                        R.setdefault(key, {})[names[i]] = v
            del fc
    for h in todo:
        dump_atomic(res[h["tag"]], os.path.join(P["scores"], f"{stimset}_{h['tag']}.json"))
        log(LOG, f"{stimset} {h['tag']} scored {len(names)} images x {len(SIZES)} sizes (maps saved: {save_maps(stimset, h['seed'])})")


if __name__ == "__main__":
    heads = discover()
    if not heads:
        log(LOG, f"no trained heads for {a.bb}"); sys.exit(0)
    stimsets = a.stimsets.split(",") if a.stimsets else ["stim_v3"] + (["stim_alt"] if os.path.exists(os.path.join(str(LP.STIM_ALT), "items.json")) else [])
    log(LOG, f"=== eval {a.bb} on {DEVICE}: {len(heads)} heads, stimsets {stimsets}, maps saved for: {SAVE}")
    ex = make_extractor(a.bb)
    run_mit(ex, heads)
    for st in stimsets:
        try:
            run_stim(ex, heads, st)
        except Exception as e:                                  # a malformed new stimulus set must not stop the queue
            if st == "stim_v3":
                raise
            log(LOG, f"FAILED {st}: {type(e).__name__}: {e}")
    log(LOG, f"=== eval {a.bb} done")
