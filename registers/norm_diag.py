"""
Diagnostic (CPU): high-norm patch tokens at blocks 6 and 12 of plain DINOv2-B and of DINOv2-B with registers, on the
inputs the readouts see (336 and 448 px, the pipeline's preprocessing). Per image: patch-token L2 norms (CLS and registers
excluded); outliers = norm > 2 x the image's median patch norm and norm > median + 6 x MAD. Displays: the first
illusory and control180 configuration of every shape x support ratio cell of v2/stim_v3 at the first position
(up to 24 displays), and 24 held-out MIT1003 images.
Usage: V3_DEVICE=cpu python norm_diag.py   -> diag/norm_diag.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import regcommon as R

OUT = os.path.join(R.HERE, "diag"); os.makedirs(OUT, exist_ok=True)
items = R.C2.stim_items("stim_v3")
kz = [it for it in items if it.get("illusion", "kanizsa") == "kanizsa" and it["cond"] in ("illusory", "control180")]
seen, sel = set(), []
for it in kz:
    key = (it["shape"], it["sr"], it["pol"], it["cond"])
    if key not in seen:
        seen.add(key); sel.append(it)
sel = sel[:24]
mit = [p["image"] for p in R.mit_heldout()[:24]]
sets = {"kanizsa": [it["img"] for it in sel], "mit1003": mit}
res = dict(n={k: len(v) for k, v in sets.items()}, models={})
for bb, drop in (("dinov2", 1), ("dinov2reg", 5)):
    ex = R.make_extractor(bb)
    res["models"][bb] = {}
    for nm, paths_ in [(f"{px}|{k}", v) for px in (336, 448) for k, v in sets.items()]:
        x = R.prep_batch(ex, paths_, int(nm.split("|")[0]))
        with torch.no_grad():
            kw = dict(interpolate_pos_encoding=True) if bb == "dinov2" else {}
            hs = ex.m(pixel_values=x.to(R.DEVICE), output_hidden_states=True, **kw).hidden_states
        rec = {}
        for L in (6, 12):
            n = hs[L][:, drop:, :].norm(dim=-1).cpu().numpy()            # (B, tokens)
            med = np.median(n, 1, keepdims=True); mad = np.median(np.abs(n - med), 1, keepdims=True)
            out2 = (n > 2 * med); out6 = (n > med + 6 * mad)
            rec[f"L{L}"] = dict(median_norm=float(np.median(med)), max_over_median=float(np.median(n.max(1) / med[:, 0])),
                                max_over_median_max=float(np.max(n.max(1) / med[:, 0])),
                                frac_tokens_gt2xmedian=float(out2.mean()), frac_images_with_gt2xmedian=float(out2.any(1).mean()),
                                frac_tokens_gt6mad=float(out6.mean()), frac_tokens_gt1p5xmedian=float((n > 1.5 * med).mean()))
            if bb == "dinov2reg":
                rn = hs[L][:, 1:5, :].norm(dim=-1).cpu().numpy()
                rec[f"L{L}"]["register_norm_over_patch_median"] = float(np.median(rn / med))
        res["models"][bb][nm] = rec
    del ex
json.dump(res, open(os.path.join(OUT, "norm_diag.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
