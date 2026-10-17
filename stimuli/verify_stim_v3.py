"""
Pixel-level verification of stimuli/stim_v3 (run after stimuli/stimuli_v3.py). Writes stim_v3/verify.json and prints
a summary; exits non-zero if any hard check fails. The first-version sets (stim_v2, stim_ehr) are those in
legacy/stimuli/kanizsa and legacy/stimuli/ehrenstein.

Hard checks
  H1  saved PNG == fresh render (determinism); every condition the original renderers can draw is byte-identical to
      stimuli_v2.figure / stimuli_ehr.figure (mis* via stimuli_v2.ROT[cond] = theta); configurations shared with
      stim_v2 / stim_ehr are byte-identical to the PNGs and AOIs on disk.
  H2  illusory vs control180 (and vs control90, mis*): at supersampled resolution every differing pixel lies inside an
      inducer disc (radius r, 1 supersampled pixel tolerance for the rasteriser); at output resolution every differing
      pixel lies within the LANCZOS support (3 px) of a disc.
  H3  realC vs control180 (Kanizsa) and realC vs control_end (Ehrenstein): at supersampled resolution the differing
      pixels are a subset of the outline stroke; at output resolution within 3 px of it. Same for real vs illusory.
      real_filled vs illusory: subset of the figure fill.
  H4  misaligned inducer area (supersampled ink pixel count, and output-resolution ink mass) within 1% of illusory.
  H5  AOIs (inside, ind, inside_eroded, inside_eroded_all) identical across all conditions of a configuration; inside
      and ind disjoint; inside_eroded subset of inside and non-empty; inside_eroded pixels > 8 px from the contour.
  H6  outline pixels never fall in inside_eroded; real_filled is exactly GREY on inside_eroded.
  H7  items.json complete: counts, unique names, files present, siblings / controls resolve.
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stimuli_v3 as T          # noqa: E402
S, E = T.S, T.E
OUT = T.OUT; SS, W, H = T.SS, T.W, T.H
STIM_V2, STIM_EHR = str(T.LP.STIM_V2), str(T.LP.STIM_EHR)
LZ = 3                           # LANCZOS support in output pixels for a 4x downsample

fails = []
def check(ok, msg):
    if not ok:
        fails.append(msg); print("FAIL:", msg, flush=True)
    return ok

def arr(img): return np.asarray(img, dtype=np.int16)
def load_png(path): return np.asarray(Image.open(path).convert("L"), dtype=np.int16)
def load_png_rgb(path): return np.asarray(Image.open(path))

def blockmax(m):
    return m.reshape(H, SS, W, SS).max(axis=(1, 3))

def near(mask_lo, px=LZ):
    return maximum_filter(mask_lo, size=2 * px + 1)

def discs_hi(V, r, tol):
    m = Image.new("1", (W * SS, H * SS), 0); d = ImageDraw.Draw(m)
    R = r * SS + tol
    for x, y in V:
        d.ellipse([x * SS - R, y * SS - R, x * SS + R, y * SS + R], fill=1)
    return np.asarray(m, bool)

def discs_lo(V, r, extra=0.0):
    yy, xx = np.mgrid[0:H, 0:W]; m = np.zeros((H, W), bool)
    for x, y in V:
        m |= (xx - x) ** 2 + (yy - y) ** 2 <= (r + extra) ** 2
    return m

def kanizsa_outline_hi(V):
    m = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in V], outline=255, width=3 * SS)
    return np.asarray(m) > 0

def kanizsa_fill_hi(V):
    m = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(m).polygon([(x * SS, y * SS) for x, y in V], fill=255)
    return np.asarray(m) > 0

def ehr_outline_hi(cx, cy, R):
    m = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(m).ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], outline=255, width=3 * SS)
    return np.asarray(m) > 0

def ehr_fill_hi(cx, cy, R):
    m = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(m).ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], fill=255)
    return np.asarray(m) > 0

def subset_report(diff, allowed):
    """Number of differing pixels and number outside the allowed mask."""
    return int(diff.sum()), int((diff & ~allowed).sum())

rep = dict(kanizsa={}, ehrenstein={}, summary={})
items = json.load(open(os.path.join(OUT, "items.json")))
by_name = {it["name"]: it for it in items}

# ---------------- H7 items.json ----------------
check(len(items) == 60 * 9 + 40 * 5, f"item count {len(items)} != 740")
check(len(by_name) == len(items), "duplicate names in items.json")
need = {"name", "illusion", "config", "shape", "sr", "R", "n", "pol", "pos", "cond", "primary_control", "controls",
        "siblings", "aoi"}
for it in items:
    check(need <= set(it), f"{it['name']}: missing fields {need - set(it)}")
    check(os.path.exists(os.path.join(OUT, "img", it["name"] + ".png")), f"{it['name']}.png missing")
    check(os.path.exists(os.path.join(OUT, "img", it["aoi"])), f"{it['aoi']} missing")
    check(it["primary_control"] in by_name, f"{it['name']}: primary control not found")
    for c, nm in list(it["controls"].items()) + list(it["siblings"].items()):
        check(nm in by_name and by_name[nm]["config"] == it["config"], f"{it['name']}: sibling {c} {nm} bad")
nconf = {"kanizsa": len({i["config"] for i in items if i["illusion"] == "kanizsa"}),
         "ehrenstein": len({i["config"] for i in items if i["illusion"] == "ehrenstein"})}
check(nconf == {"kanizsa": 60, "ehrenstein": 40}, f"config counts {nconf}")

stats = dict(k_ill_ctrl_lo_outside_disc=0, k_ill_ctrl_lo_outside_disc_maxabs=0, k_area_ratio=[], k_ink_ratio=[],
             min_inside=10 ** 9, min_eroded=10 ** 9, min_eroded_all=10 ** 9, e_ctrl_line_px_in_inside=[])

for ill, p in T.configs():
    conds = T.K_CONDS if ill == "kanizsa" else T.E_CONDS
    cfg = T.name_of(ill, p, "illusory")[: -len("_illusory")]
    hi = {c: arr(T.render(ill, p, c)) for c in conds}
    lo = {c: load_png(os.path.join(OUT, "img", T.name_of(ill, p, c) + ".png")) for c in conds}
    r_ = {}
    # ---------- H1 determinism and identity with the original renderers ----------
    for c in conds:
        rgb = load_png_rgb(os.path.join(OUT, "img", T.name_of(ill, p, c) + ".png"))
        check(rgb.ndim == 3 and rgb.shape == (H, W, 3) and (rgb[..., 0] == rgb[..., 1]).all()
              and (rgb[..., 0] == rgb[..., 2]).all(), f"{cfg} {c}: not a grey RGB {W}x{H} image")
        fresh = arr(T.down(Image.fromarray(hi[c].astype(np.uint8))))
        check((fresh == lo[c]).all(), f"{cfg} {c}: saved PNG differs from fresh render")
    if ill == "kanizsa":
        sh, cx, cy, side, sr, fg, bg = p["shape"], p["cx"], p["cy"], p["side"], p["sr"], p["fg"], p["bg"]
        for c in conds:
            if c in T.K_V2NAME:
                o, _, _ = S.figure(sh, cx, cy, side, sr, T.K_V2NAME[c], fg, bg)
            elif c.startswith("mis"):
                S.ROT[c] = T.K_CONDS[c][0]
                o, _, _ = S.figure(sh, cx, cy, side, sr, c, fg, bg); del S.ROT[c]
            else:
                continue
            check((arr(o) == lo[c]).all(), f"{cfg} {c}: differs from stimuli_v2.figure")
            r_.setdefault("identical_to_stimuli_v2_figure", []).append(c)
        v2n = f"{sh}_sr{int(sr * 100)}_{p['pol']}_p{p['pos']}"
        for c, c2 in T.K_V2NAME.items():
            old = load_png(os.path.join(STIM_V2, "img", f"{v2n}_{c2}.png"))
            check((old == lo[c]).all(), f"{cfg} {c}: differs from stim_v2/{v2n}_{c2}.png")
        r_["identical_to_stim_v2_png"] = list(T.K_V2NAME)
        V = S.vertices(sh, cx, cy, side); r = sr * side / 2
    else:
        cx, cy, R, n, fg, bg = p["cx"], p["cy"], p["R"], p["n"], p["fg"], p["bg"]
        for c in T.E_ORIG:
            o, _, _ = E.figure(cx, cy, R, n, c, fg, bg)
            check((arr(o) == lo[c]).all(), f"{cfg} {c}: differs from stimuli_ehr.figure")
            old = load_png(os.path.join(STIM_EHR, "img", f"{T.e_name(R, n, p['pol'], p['pos'], c)}.png"))
            check((old == lo[c]).all(), f"{cfg} {c}: differs from stim_ehr png")
        r_["identical_to_stimuli_ehr_and_png"] = list(T.E_ORIG)

    # ---------- H5 AOIs ----------
    A = [np.load(os.path.join(OUT, "img", T.name_of(ill, p, c) + "_aoi.npz")) for c in conds]
    keys = sorted(A[0].files)
    check(keys == ["ind", "inside", "inside_eroded", "inside_eroded_all"], f"{cfg}: AOI fields {keys}")
    for a, c in zip(A[1:], list(conds)[1:]):
        check(sorted(a.files) == keys and all((a[k] == A[0][k]).all() for k in keys), f"{cfg} {c}: AOI differs")
    a = {k: A[0][k] for k in keys}
    check(all(a[k].dtype == bool and a[k].shape == (H, W) for k in keys), f"{cfg}: AOI dtype/shape")
    check(not (a["inside"] & a["ind"]).any(), f"{cfg}: inside and ind overlap")
    check(not (a["inside_eroded"] & ~a["inside"]).any(), f"{cfg}: inside_eroded not subset of inside")
    check(not (a["inside_eroded_all"] & ~a["inside_eroded"]).any(), f"{cfg}: inside_eroded_all not subset")
    if ill == "kanizsa":
        old = np.load(os.path.join(STIM_V2, "img", f"{v2n}_illusory_aoi.npz"))
        check((old["inside"] == a["inside"]).all() and (old["ind"] == a["ind"]).all(), f"{cfg}: AOI != stim_v2")
        poly = Image.new("L", (W, H), 0); ImageDraw.Draw(poly).polygon(V, fill=1); poly = np.array(poly, bool)
        dpoly = distance_transform_edt(poly)
        check(dpoly[a["inside_eroded"]].min() > T.ERODE_PX, f"{cfg}: eroded pixel within 8 px of contour")
        # every inside pixel farther than 8 px from the contour is kept
        check((a["inside_eroded"] == (a["inside"] & (dpoly > T.ERODE_PX))).all(), f"{cfg}: eroded definition")
    else:
        old = np.load(os.path.join(STIM_EHR, "img", f"{T.e_name(R, n, p['pol'], p['pos'], 'illusory')}_aoi.npz"))
        check((old["inside"] == a["inside"]).all() and (old["ind"] == a["ind"]).all(), f"{cfg}: AOI != stim_ehr")
        yy, xx = np.mgrid[0:H, 0:W]; rr = np.hypot(xx - cx, yy - cy)
        check(rr[a["inside_eroded"]].max() <= R - T.ERODE_PX + 1.0, f"{cfg}: Ehrenstein eroded disc not ~ r <= R-8")
        r_["eroded_vs_analytic_r_le_R_minus_8_mismatch_px"] = int((a["inside_eroded"] != (rr <= R - T.ERODE_PX)).sum())
    r_["aoi_px"] = {k: int(a[k].sum()) for k in keys}
    stats["min_inside"] = min(stats["min_inside"], r_["aoi_px"]["inside"])
    stats["min_eroded"] = min(stats["min_eroded"], r_["aoi_px"]["inside_eroded"])
    stats["min_eroded_all"] = min(stats["min_eroded_all"], r_["aoi_px"]["inside_eroded_all"])
    check(r_["aoi_px"]["inside_eroded_all"] > 0, f"{cfg}: empty eroded AOI")

    # ---------- H2 / H3 / H4 / H6 ----------
    if ill == "kanizsa":
        dh = discs_hi(V, r, 1.0)
        dl_lz = discs_lo(V, r, LZ + 0.5)
        for c in ("control180", "control90", "mis10", "mis20", "mis30"):
            n_hi, out_hi = subset_report(hi[c] != hi["illusory"], dh)
            check(out_hi == 0, f"{cfg} illusory vs {c}: {out_hi} supersampled diff px outside discs")
            dlo = lo[c] != lo["illusory"]
            n_lo, out_lo = subset_report(dlo, dl_lz)
            check(out_lo == 0, f"{cfg} illusory vs {c}: {out_lo} output diff px beyond disc + {LZ} px")
            strict = dlo & ~a["ind"]
            r_[f"ill_vs_{c}"] = dict(diff_px_hi=n_hi, diff_px=n_lo, diff_px_outside_ind=int(strict.sum()),
                                     maxabs_outside_ind=int(np.abs(lo[c] - lo["illusory"])[strict].max()) if strict.any() else 0,
                                     diff_px_in_inside_eroded=int((dlo & a["inside_eroded"]).sum()),
                                     diff_px_in_inside_eroded_all=int((dlo & a["inside_eroded_all"]).sum()))
            # inside_eroded is eroded from the figure contour only, so it still borders the disc arcs and receives
            # the LANCZOS ringing (<= 3 px) of the inducer change; inside_eroded_all must not.
            check(not (dlo & a["inside_eroded_all"]).any(), f"{cfg} illusory vs {c}: diff in inside_eroded_all")
            if c == "control180":
                stats["k_ill_ctrl_lo_outside_disc"] += int(strict.sum())
                stats["k_ill_ctrl_lo_outside_disc_maxabs"] = max(stats["k_ill_ctrl_lo_outside_disc_maxabs"],
                                                                 r_[f"ill_vs_{c}"]["maxabs_outside_ind"])
        ol_hi = kanizsa_outline_hi(V); ol_lo = near(blockmax(ol_hi))
        for test, base in (("realC", "control180"), ("real", "illusory")):
            n_hi, out_hi = subset_report(hi[test] != hi[base], ol_hi)
            check(out_hi == 0, f"{cfg} {test} vs {base}: {out_hi} supersampled diff px off the outline")
            dlo = lo[test] != lo[base]
            n_lo, out_lo = subset_report(dlo, ol_lo)
            check(out_lo == 0, f"{cfg} {test} vs {base}: {out_lo} output diff px beyond outline + {LZ} px")
            check(not (dlo & a["inside_eroded"]).any(), f"{cfg} {test} vs {base}: outline reaches inside_eroded")
            r_[f"{test}_vs_{base}"] = dict(diff_px_hi=n_hi, diff_px=n_lo, outline_px_hi=int(ol_hi.sum()))
        fl_hi = kanizsa_fill_hi(V)
        n_hi, out_hi = subset_report(hi["real_filled"] != hi["illusory"], fl_hi)
        check(out_hi == 0, f"{cfg} real_filled vs illusory: {out_hi} supersampled diff px off the figure")
        check((lo["real_filled"][a["inside_eroded"]] == T.GREY).all(), f"{cfg}: real_filled not grey on eroded AOI")
        fg_ = p["fg"]
        ink_hi = {c: int((hi[c] == fg_).sum()) for c in conds}
        ink_lo = {c: float(np.abs(lo[c] - p["bg"]).sum() / 255) for c in conds}
        for c in ("control180", "control90", "mis10", "mis20", "mis30"):
            ra, rl = ink_hi[c] / ink_hi["illusory"], ink_lo[c] / ink_lo["illusory"]
            if c.startswith("mis"):
                stats["k_area_ratio"].append(ra); stats["k_ink_ratio"].append(rl)
                check(abs(ra - 1) < 0.01 and abs(rl - 1) < 0.01, f"{cfg} {c}: inducer area ratio {ra:.4f}/{rl:.4f}")
            r_[f"area_ratio_{c}"] = dict(supersampled_px=round(ra, 5), output_ink=round(rl, 5))
        rep["kanizsa"][cfg] = r_
    else:
        ol_hi = ehr_outline_hi(cx, cy, R); ol_lo = near(blockmax(ol_hi))
        for test, base in (("realC", "control_end"), ("real", "illusory")):
            n_hi, out_hi = subset_report(hi[test] != hi[base], ol_hi)
            check(out_hi == 0, f"{cfg} {test} vs {base}: {out_hi} supersampled diff px off the outline")
            dlo = lo[test] != lo[base]
            n_lo, out_lo = subset_report(dlo, ol_lo)
            check(out_lo == 0, f"{cfg} {test} vs {base}: {out_lo} output diff px beyond outline + {LZ} px")
            check(not (dlo & a["inside_eroded"]).any(), f"{cfg} {test} vs {base}: outline reaches inside_eroded")
            r_[f"{test}_vs_{base}"] = dict(diff_px_hi=n_hi, diff_px=n_lo, outline_px_hi=int(ol_hi.sum()))
        fl_hi = ehr_fill_hi(cx, cy, R)
        n_hi, out_hi = subset_report(hi["real_filled"] != hi["illusory"], fl_hi)
        check(out_hi == 0, f"{cfg} real_filled vs illusory: {out_hi} supersampled diff px off the disc")
        check((lo["real_filled"][a["inside_eroded"]] == T.GREY).all(), f"{cfg}: real_filled not grey on eroded AOI")
        # illusory vs control_end: lines only, so differences are confined to inside | ind (info, plus the known
        # ~1.4 px intrusion of control_end line corners into the inside disc that stim_ehr already had)
        dlo = lo["control_end"] != lo["illusory"]
        r_["ill_vs_control_end"] = dict(diff_px=int(dlo.sum()), diff_px_outside_inside_or_ind=int((dlo & ~(a["inside"] | a["ind"])).sum()),
                                        diff_px_in_inside=int((dlo & a["inside"]).sum()),
                                        diff_px_in_inside_eroded=int((dlo & a["inside_eroded"]).sum()))
        check(not (dlo & a["inside_eroded"]).any(), f"{cfg} illusory vs control_end: diff in inside_eroded")
        ink_ce = int((hi["control_end"] == fg).sum()); ink_il = int((hi["illusory"] == fg).sum())
        r_["area_ratio_control_end"] = round(ink_ce / ink_il, 5)
        stats["e_ctrl_line_px_in_inside"].append(int(((lo["control_end"] != bg) & a["inside"]).sum()))
        rep["ehrenstein"][cfg] = r_
    print(f"checked {cfg}", flush=True)

rep["summary"] = dict(
    n_images=len(items), n_kanizsa=sum(i["illusion"] == "kanizsa" for i in items),
    n_ehrenstein=sum(i["illusion"] == "ehrenstein" for i in items), configs=nconf,
    kanizsa_mis_area_ratio_supersampled=[round(min(stats["k_area_ratio"]), 5), round(max(stats["k_area_ratio"]), 5)],
    kanizsa_mis_ink_ratio_output=[round(min(stats["k_ink_ratio"]), 5), round(max(stats["k_ink_ratio"]), 5)],
    kanizsa_ill_vs_control180_output_diff_px_outside_disc_aoi_total=stats["k_ill_ctrl_lo_outside_disc"],
    kanizsa_ill_vs_control180_output_diff_maxabs_outside_disc_aoi=stats["k_ill_ctrl_lo_outside_disc_maxabs"],
    min_aoi_px=dict(inside=stats["min_inside"], inside_eroded=stats["min_eroded"], inside_eroded_all=stats["min_eroded_all"]),
    ehrenstein_control_end_nonbg_px_in_inside=[min(stats["e_ctrl_line_px_in_inside"]), max(stats["e_ctrl_line_px_in_inside"])],
    failures=fails, passed=not fails)
json.dump(rep, open(os.path.join(OUT, "verify.json"), "w"), indent=1)
print(json.dumps(rep["summary"], indent=1))
sys.exit(1 if fails else 0)
