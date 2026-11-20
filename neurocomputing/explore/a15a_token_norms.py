"""
A15(a) token norms of CLIP-L and SigLIP-L, with the Critic fix 18 condition (RUN_PLAN.md Step 2, T2). Exploratory
analysis of REANALYSIS_RULE.md Section 6, item 12. Written 18 November 2026 after the rule was fixed and public; not yet
run.

Rule text (Section 6, item 12, quoted verbatim):
  "A15(a) token norms of CLIP-L and SigLIP-L: every block read by a readout (SigLIP-L 24 and the 12 sweep blocks of
  CLIP-L and SigLIP-L), on the 24 Kanizsa displays of the receptive-field rule (12 configurations, one per design cell,
  cell i at position i mod 5, illusory and control180) and the 24 MIT1003 images of `nn/registers/diag/norm_diag.json`,
  at 336 and 448 px: per image the ratio of the largest to the median patch-token L2 norm and the share of tokens above
  twice the median. Condition (Critic fix 18): if any block read shows at least one patch token above twice the image
  median on any Kanizsa display, test-time registers (Jiang et al. 2025) are applied to that block before submission
  (new CPU code; exploratory); otherwise the norm result is stated as the reason for not running them."

Implementation (adapted from nn/registers/norm_diag.py, which measured plain and registers DINOv2-B):
  - Backbones and inputs as the sweep readouts see them (sweep.py CFG and prep(), copied verbatim below): CLIP-L/14
    (openai/clip-vit-large-patch14, CLS dropped) and SigLIP-L/16 (google/siglip-large-patch16-256, no CLS), the 1024 x
    768 display squashed to s x s with cv2 INTER_AREA, each backbone's normalisation, hidden_states[L] with
    interpolate_pos_encoding=True (block L = Hugging Face hidden_states[L], rule Section 2.5). The SigLIP-L block-24
    swap features use the same preprocessing and the same tensor (v2/readout/common.py Extractor), so SigLIP-L 24 is
    measured once and serves both readouts. Blocks: 4, 8, 12, 16, 20, 24 of each backbone (12 blocks read).
  - Displays: the receptive-field rule's 24 (cells ordered as analyze_re.Design keys, i.e. sorted by str of (shape,
    support ratio, polarity) as v3/scores/analyze_alt.Design; cell i at position i mod 5; illusory and control180;
    nn/erf_nn.py display_set()). MIT1003: the first 24 held-out images (build_mit1003(<SR>/mit1003)[-200:][:24], the
    list norm_diag.py used through regcommon.mit_heldout), with SR from the ab_objective stub.
  - Per image, block and size: patch-token L2 norms; ratio = max / median; share = fraction of tokens > 2 x median;
    any = at least one token > 2 x median. Summaries per (backbone, block, size, set).
  - Condition: a block read is flagged if any Kanizsa display at 336 or 448 px has any = True; the flagged blocks are
    listed (test-time registers would then be applied to them, a later step not written here).
  - CPU only: torch threads from OMP_NUM_THREADS (4), the ab_objective stub (DEV = cpu), excommon.metal_guard() after
    every model load; require_public() first.
Usage (after the rule is public; CPU, one process, about 0.1 to 0.5 h):
  nice -n 10 env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -B explore/a15a_token_norms.py
Output: results/explore/A15a_token_norms.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "4"))     # at most 4 threads (CPU shared)
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_k] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                        # noqa: E402
import excommon as X                                                      # noqa: E402

CFG = {                                                                   # verbatim from sweep.py CFG (clip, siglip)
    "clip": dict(hf="openai/clip-vit-large-patch14", layers=(4, 8, 12, 16, 20, 24), cls=True,
                 mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    "siglip": dict(hf="google/siglip-large-patch16-256", layers=(4, 8, 12, 16, 20, 24), cls=False,
                   mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
BATCH = 6


def rf_displays():
    """The receptive-field rule's 24 Kanizsa displays (nn/erf_nn.py display_set)."""
    D = X.design()
    assert len(D.keys) == 12
    out = []
    for i, (sh, sr, pol) in enumerate(D.keys):
        cfg = f"{sh}_sr{int(round(sr * 100))}_{pol}_p{i % 5}"
        assert cfg in D.configs, cfg
        for cond in ("illusory", "control180"):
            out.append(os.path.join(X.STIM_V3, "img", f"{cfg}_{cond}.png"))
    return out


def norm_stats(n):
    """n: (images, tokens) patch-token L2 norms -> per image ratio max/median, share > 2 x median, any > 2 x median."""
    n = np.asarray(n, np.float64)
    med = np.median(n, 1, keepdims=True)
    gt2 = n > 2 * med
    return dict(max_over_median=(n.max(1) / med[:, 0]).tolist(), share_gt2xmedian=gt2.mean(1).tolist(),
                any_gt2xmedian=gt2.any(1).tolist(), n_gt2xmedian=gt2.sum(1).astype(int).tolist(),
                median_norm=med[:, 0].tolist())


def summary(per):
    r = np.array(per["max_over_median"]); s = np.array(per["share_gt2xmedian"]); a = np.array(per["any_gt2xmedian"])
    return dict(n_images=int(r.size), max_over_median_median=float(np.median(r)), max_over_median_max=float(r.max()),
                share_gt2xmedian_mean=float(s.mean()), n_images_with_token_gt2xmedian=int(a.sum()))


def condition(res):
    """Critic fix 18: blocks read with at least one patch token above twice the image median on any Kanizsa display."""
    flagged = []
    for bb, blocks in res.items():
        for L, rec in blocks.items():
            if any(any(rec[f"{s}|kanizsa"]["per_image"]["any_gt2xmedian"]) for s in X.SIZES):
                flagged.append(f"{bb} block {L}")
    return dict(flagged_blocks=flagged, test_time_registers_required=bool(flagged),
                statement=("test-time registers (Jiang et al. 2025) to be applied to: " + ", ".join(flagged)) if flagged
                else "no block read showed a patch token above twice the image median on any Kanizsa display; this "
                     "norm result is the reason for not running test-time registers")


def main():
    X.require_public()                                                # rule Section 8.1
    import cv2, torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cv2.setNumThreads(1)
    stub = os.path.join(X.RE, "_stub")
    sys.path.insert(0, stub)
    import ab_objective as AB                                         # the stub: DEV = "cpu", SR from the environment
    assert AB.__file__.startswith(stub) and AB.DEV == "cpu", AB.__file__
    from src.data import build_mit1003
    mit = [p["image"] for p in build_mit1003(f"{AB.SR}/mit1003")[-200:][:24]]
    assert len(mit) == 24, len(mit)
    sets = {"kanizsa": rf_displays(), "mit1003": mit}
    from transformers import CLIPVisionModel, SiglipVisionModel
    res = {}
    for bb, C in CFG.items():
        m = (CLIPVisionModel if bb == "clip" else SiglipVisionModel).from_pretrained(C["hf"]).to("cpu").eval()
        X.metal_guard()
        MEAN, STD = np.array(C["mean"], np.float32), np.array(C["std"], np.float32)

        def prep(rgb, s):                                                 # verbatim from sweep.prep
            x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1))
        res[bb] = {L: {} for L in C["layers"]}
        for s in X.SIZES:
            for sname, paths in sets.items():
                norms = {L: [] for L in C["layers"]}
                for b in range(0, len(paths), BATCH):
                    x = torch.stack([prep(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), s) for p in paths[b:b + BATCH]])
                    with torch.no_grad():
                        hs = m(pixel_values=x, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states
                    for L in C["layers"]:
                        t = hs[L][:, 1:, :] if C["cls"] else hs[L]
                        norms[L].append(t.norm(dim=-1).numpy())
                for L in C["layers"]:
                    per = norm_stats(np.concatenate(norms[L]))
                    res[bb][L][f"{s}|{sname}"] = dict(summary=summary(per), per_image=per)
                X.log(f"  {bb} {s}px {sname} done")
        del m
    out = dict(images=dict(kanizsa=[os.path.basename(p) for p in sets["kanizsa"]],
                           mit1003=[os.path.basename(p) for p in mit]),
               blocks={bb: list(C["layers"]) for bb, C in CFG.items()}, sizes=list(X.SIZES), results=res,
               critic_fix_18=condition(res))
    X.write_json("A15a_token_norms.json", out, os.path.abspath(__file__),
                 inputs=["v2/stim_v3/img (24 displays)", "MIT1003 held-out (first 24)", "HF cache: clip-vit-large-patch14, "
                                                                                        "siglip-large-patch16-256"])


if __name__ == "__main__":
    main()
