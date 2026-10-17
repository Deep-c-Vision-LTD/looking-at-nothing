"""
Pixel-level verification and stimulus measures for stimuli/stim_alt (run after stimuli/stimuli_alt.py).
Writes stim_alt/verify.json and stim_alt/measures.md; exits non-zero if a hard check fails.

Hard checks
  H1  the per-inducer renderer (stimuli_alt.kanizsa_hi_rots) with one shared rotation is byte-identical to
      stimuli_v3.kanizsa_hi at supersampled resolution, and its output-resolution image is pixel-identical to the
      stim_v3 PNG on disk, for illusory (0), control180, control90, mis10, mis20, mis30, all 60 configurations.
  H2  saved alt PNGs == fresh render; R = G = B.
  H3  alt vs illusory: at supersampled resolution every differing pixel lies inside an inducer disc (radius r plus one
      supersampled pixel); at output resolution within r + 3.5 px of the disc centres (the verify_stim_v3.py band: 3 px LANCZOS
      support + 0.5 px); no differing pixel in
      core30, core45 or inside_eroded_all.
  H4  ink area of every inducer (supersampled ink pixel count) within 1% of the same inducer in illusory (all rotated
      conditions, controls included).
  H5  measured notch direction of every inducer (circular mean angle of the notch pixels in an annulus 0.3 r to 0.9 r,
      supersampled) equals the illusory notch direction + the intended rotation, within 1 deg (rasterisation error at the
      smallest radius is about 0.5 deg; the smallest intended rotation is 10 deg); this checks the sign
      of the alternation inducer by inducer (also run on mis* to confirm the uniform convention).
  H6  AOIs saved in stim_alt/aoi equal the stim_v3 AOIs; core AOIs non-empty and inside the inside AOI.
Measures (all conditions: stim_v3 illusory, control180, control90, mis10/20/30 and stim_alt alt10/20/30 a and b)
  inner-sector ink: notch sector of inducer k = pixels (integer centres, as the AOIs) within radius r whose angle lies
      within +-gap/2 of the illusory notch direction. ink = |I - background| / 255 on the output image.
      frac_sector_inked = mean ink over the notch-sector pixels (task definition: fraction of notch-sector pixels inked);
      share_of_inducer_ink = ink in the notch sectors / ink in the inducer AOI (the v2 verifier's number, "7 to 8% at
      mis20 against 1% in illusory").
  mirror symmetry, two ways, for every mirror axis of the illusory figure (square: vertical, horizontal, 2 diagonals;
      triangle: the 3 vertex axes):
      analytic: exact inducer geometry on a 0.5 px grid; S = |ink and reflected ink| / |ink| (1 = exact symmetry);
      image: on the output image, reflection about the axis through the illusory image's ink centroid (bilinear),
      S_img = 1 - sum|k - k_refl| / sum(k + k_refl) over the figure window.
  edge geometry per figure side (analytic): collinear, relatable (forward extensions of the two notch edges meet, turn
      <= 90 deg) or not relatable; turn angle; offset of the far edge tip from the near edge's line.
Also writes the contact sheets stim_alt/contact_alt.png (all 60 configurations) and contact_alt_zoom.png.
"""
import json, math, os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter, binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stimuli_alt as A          # noqa: E402
T, S = A.T, A.S
W, H, SS = A.W, A.H, A.SS
OUT = A.OUT; STIM_V3 = str(T.LP.STIM_V3)
LZ = 3
V3COND = {"illusory": 0, "control180": 180, "control90": 90, "mis10": 10, "mis20": 20, "mis30": 30}

fails = []
def check(ok, msg):
    if not ok:
        fails.append(msg); print("FAIL:", msg, flush=True)
    return bool(ok)

items = json.load(open(os.path.join(OUT, "items.json")))
cfgs = list(A.configs())
assert len(cfgs) == 60 and len(items) == 360


def rots_of(p, cond):
    n = len(S.vertices(p["shape"], 0, 0, 1))
    if cond in V3COND:
        return [V3COND[cond]] * n
    th, ph = int(cond[3:-1]), cond[-1]
    return A.alt_rots(p["shape"], th, ph)


def load_L(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return a[..., 0], a


def img_path(cfg, cond):
    return (os.path.join(STIM_V3, "img", f"{cfg}_{cond}.png") if cond in V3COND
            else os.path.join(OUT, "img", f"{cfg}_{cond}.png"))


ALT = [f"alt{t}{ph}" for t in A.THETAS for ph in ("a", "b")]
ALLC = list(V3COND) + ALT
yy, xx = np.mgrid[0:H, 0:W]


def discs_lo(V, r, extra=0.0):
    m = np.zeros((H, W), bool)
    for x, y in V:
        m |= (xx - x) ** 2 + (yy - y) ** 2 <= (r + extra) ** 2
    return m


def discs_hi(V, r, tol=1):
    m = Image.new("1", (W * SS, H * SS), 0); d = ImageDraw.Draw(m); R = r * SS + tol
    for x, y in V:
        d.ellipse([x * SS - R, y * SS - R, x * SS + R, y * SS + R], fill=1)
    return np.asarray(m, bool)


def ang_in(a, lo, hi):
    """a in the circular interval [lo, hi] (degrees, hi - lo < 360)."""
    return ((a - lo) % 360) <= ((hi - lo) % 360)


# ------------------------------------------------------------------ analytic geometry
def ink_analytic(px, py, V, r, dirs, rots, gap):
    ink = np.zeros(px.shape, bool)
    for (x, y), th, rot in zip(V, dirs, rots):
        dx, dy = px - x, py - y
        inside = dx * dx + dy * dy <= r * r
        a = np.degrees(np.arctan2(dy, dx))
        c = th + rot
        notch = ang_in(a, c - gap / 2, c + gap / 2)
        ink |= inside & ~notch
    return ink


def axes_of(shape, dirs):
    """Mirror axes of the illusory figure as line directions (deg, screen coordinates) through the centroid."""
    if shape == "square":
        return {"vertical": 90.0, "horizontal": 0.0, "diag_TL_BR": 45.0, "diag_TR_BL": 135.0}
    # triangle: the axis through each vertex and the centroid (vertex directions from the centroid: -90, 30, 150)
    return {"apex(vertical)": 90.0, "BR": 30.0, "BL": 150.0}


def reflect(px, py, cx, cy, phi):
    c, s = math.cos(math.radians(2 * phi)), math.sin(math.radians(2 * phi))
    dx, dy = px - cx, py - cy
    return cx + c * dx + s * dy, cy + s * dx - c * dy


_symcache = {}
def sym_analytic(p, rots):
    key = (p["shape"], p["sr"], tuple(rots))
    if key not in _symcache:
        _symcache[key] = _sym_analytic(p, rots)
    return _symcache[key]


def _sym_analytic(p, rots):
    V, dirs, (gx, gy) = A.inward_dirs(p["shape"], p["cx"], p["cy"], p["side"])
    r = p["sr"] * p["side"] / 2; gap = A.gap_of(p["shape"])
    xs = [v[0] for v in V]; ys = [v[1] for v in V]
    gxs = np.arange(min(xs) - r - 1, max(xs) + r + 1, 0.5); gys = np.arange(min(ys) - r - 1, max(ys) + r + 1, 0.5)
    px, py = np.meshgrid(gxs, gys)
    ink = ink_analytic(px, py, V, r, dirs, rots, gap)
    out = {}
    for nm, phi in axes_of(p["shape"], dirs).items():
        qx, qy = reflect(px, py, gx, gy, phi)
        ink2 = ink_analytic(qx, qy, V, r, dirs, rots, gap)
        out[nm] = float((ink & ink2).sum() / ink.sum())
    return out


def sym_image(k, centre, shape, win):
    cx, cy = centre
    out = {}
    x0, x1, y0, y1 = win
    for nm, phi in axes_of(shape, None).items():
        c, s = math.cos(math.radians(2 * phi)), math.sin(math.radians(2 * phi))
        # destination (x, y) samples source at reflect(x, y): x' = cx + c(x-cx) + s(y-cy), y' = cy + s(x-cx) - c(y-cy)
        M = np.array([[c, s, cx - c * cx - s * cy], [s, -c, cy - s * cx + c * cy]], np.float32)
        kr = cv2.warpAffine(k, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        a, b = k[y0:y1, x0:x1], kr[y0:y1, x0:x1]
        out[nm] = float(1 - np.abs(a - b).sum() / (a + b).sum())
    return out


def edge_geometry(p, rots):
    """Per side (i, j adjacent): notch edge of i toward j and of j toward i, after rotation."""
    V, dirs, _ = A.inward_dirs(p["shape"], p["cx"], p["cy"], p["side"])
    r = p["sr"] * p["side"] / 2; n = len(V)
    sides = []
    for i in range(n):
        j = (i + 1) % n
        vi, vj = np.array(V[i]), np.array(V[j])
        bi = math.degrees(math.atan2(*(vj - vi)[::-1])) + rots[i]
        bj = math.degrees(math.atan2(*(vi - vj)[::-1])) + rots[j]
        di = np.array([math.cos(math.radians(bi)), math.sin(math.radians(bi))])
        dj = np.array([math.cos(math.radians(bj)), math.sin(math.radians(bj))])
        ti, tj = vi + r * di, vj + r * dj                      # edge tips
        # turn angle between travelling direction along i's edge (di) and the direction of arrival along j's edge (-dj)
        turn = math.degrees(math.acos(np.clip(np.dot(di, -dj), -1, 1)))
        # do the forward rays ti + s di and tj + u dj (s, u > 0) intersect?
        Mx = np.array([di, -dj]).T
        cross = lambda u, v: float(u[0] * v[1] - u[1] * v[0])
        if abs(np.linalg.det(Mx)) < 1e-9:
            off = abs(cross(di, tj - ti))
            facing = float(np.dot(di, tj - ti)) > 0                # tips face each other along the side
            if off < 1e-6 and facing:
                kind = "collinear"
            elif off < 1e-6:
                kind = "collinear, edges point away (not relatable)"
            else:
                kind = "parallel offset (not relatable)"
        else:
            s_, u_ = np.linalg.solve(Mx, tj - ti)
            rel = (s_ > 0) and (u_ > 0) and turn <= 90
            kind = "relatable bend" if rel else "not relatable"
        off = abs(cross(di, tj - ti))
        sides.append(dict(side=f"{i}-{j}", turn_deg=round(turn, 2), tip_offset_px=round(off, 2), kind=kind))
    return sides


# ------------------------------------------------------------------ main loop
res = dict(H1={}, H2={}, H3={}, H4={}, H5={}, H6={})
meas = {}
contact_rows = []
for p in cfgs:
    cfg = A.config_name(p)
    V, dirs, (gx, gy) = A.inward_dirs(p["shape"], p["cx"], p["cy"], p["side"])
    r = p["sr"] * p["side"] / 2; gap = A.gap_of(p["shape"]); n = len(V)
    bgv = p["bg"]
    # H6 AOIs
    a3 = np.load(os.path.join(OUT, "aoi", cfg + "_aoi.npz"))
    a2 = np.load(os.path.join(STIM_V3, "img", cfg + "_illusory_aoi.npz"))
    ok = all(np.array_equal(a3[k], a2[k]) for k in a2.files)
    ok &= all(a3[f"core{D}"].sum() > 0 and not (a3[f"core{D}"] & ~a3["inside"]).any() for D in A.CORE_D)
    check(ok, f"H6 {cfg}")
    ind = a3["ind"]
    # sector masks (illusory notch sectors) at output resolution
    sect = []
    for (x, y), th in zip(V, dirs):
        dx, dy = xx - x, yy - y
        m = (dx * dx + dy * dy <= r * r) & ang_in(np.degrees(np.arctan2(dy, dx)), th - gap / 2, th + gap / 2)
        sect.append(m)
    near_disc = discs_lo(V, r, LZ + 0.5)              # same band as verify_stim_v3.py (disc radius + 3.5 px)
    dhi = discs_hi(V, r)
    ill_hi = np.asarray(A.kanizsa_hi_rots(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], [0] * n, p["fg"], p["bg"]), np.int16)
    ill_lo, _ = load_L(img_path(cfg, "illusory"))
    # per-inducer ink area at supersampled resolution (bounding square of each disc)
    def ink_counts(hi):
        out = []
        for x, y in V:
            x0, x1 = int((x - r) * SS) - 2, int((x + r) * SS) + 3; y0, y1 = int((y - r) * SS) - 2, int((y + r) * SS) + 3
            out.append(int((hi[y0:y1, x0:x1] != bgv).sum()))
        return out
    ink_ill = ink_counts(ill_hi)
    # ink centroid of the illusory image (symmetry centre for the image-based measure)
    k_ill = np.abs(ill_lo.astype(np.float32) - bgv) / 255.0
    cen = (float((k_ill * xx).sum() / k_ill.sum()), float((k_ill * yy).sum() / k_ill.sum()))
    xs = [v[0] for v in V]; ys = [v[1] for v in V]
    win = (int(min(xs) - r - 8), int(max(xs) + r + 9), int(min(ys) - r - 8), int(max(ys) + r + 9))
    for cond in ALLC:
        rots = rots_of(p, cond)
        hi = np.asarray(A.kanizsa_hi_rots(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], rots, p["fg"], p["bg"]), np.int16)
        lo_fresh = np.asarray(T.down(Image.fromarray(hi.astype(np.uint8))), np.int16)
        lo, rgb = load_L(img_path(cfg, cond))
        if cond in V3COND:
            old_hi = np.asarray(T.kanizsa_hi(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], V3COND[cond], False, False, p["fg"], p["bg"]), np.int16)
            res["H1"][f"{cfg}_{cond}"] = check(np.array_equal(hi, old_hi) and np.array_equal(lo_fresh, lo), f"H1 {cfg} {cond}")
        else:
            res["H2"][f"{cfg}_{cond}"] = check(np.array_equal(lo_fresh, lo) and (rgb[..., 0] == rgb[..., 1]).all()
                                               and (rgb[..., 0] == rgb[..., 2]).all(), f"H2 {cfg} {cond}")
            dh = hi != ill_hi
            dl = lo != ill_lo
            h3 = dict(hi_outside_disc=int((dh & ~dhi).sum()), lo_outside_band=int((dl & ~near_disc).sum()),
                      lo_in_core30=int((dl & a3["core30"]).sum()), lo_in_core45=int((dl & a3["core45"]).sum()),
                      lo_in_inside_eroded_all=int((dl & a3["inside_eroded_all"]).sum()), lo_in_inside=int((dl & a3["inside"]).sum()))
            res["H3"][f"{cfg}_{cond}"] = h3
            check(h3["hi_outside_disc"] == 0 and h3["lo_outside_band"] == 0 and h3["lo_in_core30"] == 0
                  and h3["lo_in_core45"] == 0 and h3["lo_in_inside_eroded_all"] == 0, f"H3 {cfg} {cond} {h3}")
        if cond not in ("illusory",):
            ic = ink_counts(hi)
            rel = [abs(a_ - b_) / b_ for a_, b_ in zip(ic, ink_ill)]
            res["H4"][f"{cfg}_{cond}"] = round(max(rel), 5)
            check(max(rel) < 0.01, f"H4 {cfg} {cond} {rel}")
        # H5 measured notch direction per inducer
        errs = []
        for (x, y), th, rot in zip(V, dirs, rots):
            x0, x1 = int((x - r) * SS) - 2, int((x + r) * SS) + 3; y0, y1 = int((y - r) * SS) - 2, int((y + r) * SS) + 3
            sub = hi[y0:y1, x0:x1]
            gy_, gx_ = np.mgrid[y0:y1, x0:x1]
            dx, dy = (gx_ + 0.5) / SS - x, (gy_ + 0.5) / SS - y
            rr = np.hypot(dx, dy)
            m = (rr > 0.3 * r) & (rr < 0.9 * r) & (sub == bgv)
            ang = np.arctan2(dy[m], dx[m])
            mean = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
            errs.append(((mean - (th + rot) + 180) % 360) - 180)
        res["H5"][f"{cfg}_{cond}"] = round(float(np.max(np.abs(errs))), 4)
        check(np.max(np.abs(errs)) < 1.0, f"H5 {cfg} {cond} {errs}")
        # measures
        k = np.abs(lo.astype(np.float32) - bgv) / 255.0
        frac = [float(k[m].mean()) for m in sect]
        sec_all = np.zeros_like(ind)
        for m in sect:
            sec_all |= m
        share = float(k[sec_all].sum() / k[ind].sum())
        meas[f"{cfg}_{cond}"] = dict(frac_sector_inked=frac, frac_sector_inked_mean=float(np.mean(frac)),
                                     share_of_inducer_ink=share, sym_analytic=sym_analytic(p, rots),
                                     sym_image=sym_image(k, cen, p["shape"], win))
    print(cfg, "checked", flush=True)

# ------------------------------------------------------------------ summaries
def agg(shape, cond, f):
    return float(np.mean([f(meas[f"{A.config_name(p)}_{cond}"]) for p in cfgs if p["shape"] == shape]))

SUMC = ["illusory", "control180", "control90", "mis10", "mis20", "mis30"] + ALT
summary = {}
for shape in ("square", "triangle"):
    rows = {}
    p0 = next(p for p in cfgs if p["shape"] == shape and p["sr"] == 0.5)
    for cond in SUMC:
        m0 = meas[f"{A.config_name(p0)}_{cond}"]
        rows[cond] = dict(
            frac_sector_inked=agg(shape, cond, lambda m: m["frac_sector_inked_mean"]),
            frac_sector_inked_by_inducer=[agg(shape, cond, lambda m, i=i: m["frac_sector_inked"][i]) for i in range(len(m0["frac_sector_inked"]))],
            share_of_inducer_ink=agg(shape, cond, lambda m: m["share_of_inducer_ink"]),
            sym_analytic={ax: agg(shape, cond, lambda m, ax=ax: m["sym_analytic"][ax]) for ax in m0["sym_analytic"]},
            sym_image={ax: agg(shape, cond, lambda m, ax=ax: m["sym_image"][ax]) for ax in m0["sym_image"]},
            sym_image_min_over_configs={ax: float(np.min([meas[f"{A.config_name(p)}_{cond}"]["sym_image"][ax] for p in cfgs if p["shape"] == shape])) for ax in m0["sym_image"]},
            n_exact_mirrors=int(sum(v > 0.9999 for v in m0["sym_analytic"].values())),
            edges_sr50=edge_geometry(p0, rots_of(p0, cond)))
    summary[shape] = rows

h3max = {k: max(v[k] for v in res["H3"].values()) for k in next(iter(res["H3"].values()))}
out = dict(fails=fails, n_fail=len(fails),
           H1_all_identical=all(res["H1"].values()), H1_n=len(res["H1"]),
           H2_all_identical=all(res["H2"].values()), H2_n=len(res["H2"]),
           H3_max_counts=h3max, H4_max_rel_ink_diff_mis_alt=max(res["H4"].values()),
           H5_max_abs_err_deg=max(res["H5"].values()),
           summary=summary, per_image=meas)
json.dump(out, open(os.path.join(OUT, "verify.json"), "w"), indent=1)

# measures.md
L = []; w = L.append
w("# stim_alt: stimulus measures (stimuli/verify_stim_alt.py)\n")
w(f"Hard checks: {'all passed' if not fails else str(len(fails)) + ' FAILED'}. H1 renderer identity on {out['H1_n']} stim_v3 images; "
  f"H2 {out['H2_n']} alt images re-rendered identically; H3 max differing-pixel counts vs illusory: {h3max}; "
  f"H4 max per-inducer ink-area difference (mis, alt) {out['H4_max_rel_ink_diff_mis_alt']:.4f}; H5 max notch-direction error {out['H5_max_abs_err_deg']:.3f} deg.\n")
for shape in ("square", "triangle"):
    w(f"## {shape}\n")
    axs = list(summary[shape]["illusory"]["sym_analytic"])
    w("| condition | notch-sector pixels inked | by inducer | share of inducer ink in notch sectors | exact mirrors | "
      + " | ".join(f"S analytic {a}" for a in axs) + " | " + " | ".join(f"S image {a}" for a in axs) + " |")
    w("|---|---|---|---|---|" + "---|" * (2 * len(axs)))
    for cond in SUMC:
        rw = summary[shape][cond]
        w(f"| {cond} | {rw['frac_sector_inked']:.3f} | {', '.join(f'{v:.3f}' for v in rw['frac_sector_inked_by_inducer'])} | "
          f"{rw['share_of_inducer_ink']:.3f} | {rw['n_exact_mirrors']} | "
          + " | ".join(f"{rw['sym_analytic'][a]:.3f}" for a in axs) + " | " + " | ".join(f"{rw['sym_image'][a]:.3f}" for a in axs) + " |")
    w("")
    w("Edge geometry per side at sr 0.5 (turn angle deg / far tip offset from the near edge's line px / kind):\n")
    w("| condition | " + " | ".join(e["side"] for e in summary[shape]["illusory"]["edges_sr50"]) + " |")
    w("|---|" + "---|" * len(summary[shape]["illusory"]["edges_sr50"]))
    for cond in SUMC:
        w(f"| {cond} | " + " | ".join(f"{e['turn_deg']:.0f} / {e['tip_offset_px']:.1f} / {e['kind']}" for e in summary[shape][cond]["edges_sr50"]) + " |")
    w("")
open(os.path.join(OUT, "measures.md"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ contact sheets
FONT = ImageFont.load_default(size=15)
GUT, LAB, HEAD = 4, 250, 34
COLS = ["illusory", "mis10", "mis20", "mis30", "alt10a", "alt10b", "alt20a", "alt20b", "alt30a", "alt30b", "control180", "AOI"]


def boundary(m):
    return m & ~binary_erosion(m)


def tile(cfg, p, cond, half, dy, scale):
    _, rgb = load_L(img_path(cfg, "illusory" if cond == "AOI" else cond))
    rgb = rgb.astype(np.uint8).copy()
    if cond == "AOI":
        a = np.load(os.path.join(OUT, "aoi", cfg + "_aoi.npz"))
        for kk, col in (("ind", (0, 200, 255)), ("inside", (0, 200, 0)), ("core30", (255, 0, 255))):
            b = boundary(a[kk]); b = b | np.roll(b, 1, 0) | np.roll(b, 1, 1); rgb[b] = col
    cx, cy = p["cx"], p["cy"] + dy
    crop = Image.fromarray(rgb).crop((cx - half, cy - half, cx + half, cy + half))
    return crop.resize((int(2 * half * scale), int(2 * half * scale)), Image.LANCZOS)


def sheet(rows, half=225, scale=0.5):
    t = int(2 * half * scale)
    Wd = LAB + len(COLS) * (t + GUT) + GUT; Hd = HEAD + len(rows) * (t + GUT) + GUT
    Sh = Image.new("RGB", (Wd, Hd), (150, 110, 190)); d = ImageDraw.Draw(Sh)
    d.rectangle([0, 0, Wd, HEAD], fill=(255, 255, 255)); d.rectangle([0, 0, LAB, Hd], fill=(255, 255, 255))
    for j, c in enumerate(COLS):
        d.text((LAB + GUT + j * (t + GUT) + 6, 9), c, fill=(0, 0, 0), font=FONT)
    for i, p in enumerate(rows):
        cfg = A.config_name(p); y = HEAD + GUT + i * (t + GUT); lab = cfg.split("_")
        d.text((8, y + t // 2 - 18), "_".join(lab[:2]), fill=(0, 0, 0), font=FONT)
        d.text((8, y + t // 2 + 2), "_".join(lab[2:]), fill=(0, 0, 0), font=FONT)
        for j, c in enumerate(COLS):
            Sh.paste(tile(cfg, p, c, half, -33 if p["shape"] == "triangle" else 0, scale), (LAB + GUT + j * (t + GUT), y))
    return Sh


sheet(cfgs).save(os.path.join(OUT, "contact_alt.png"))
pick = [p for p in cfgs if (p["shape"], p["sr"], p["pol"], p["pos"]) in
        (("square", 0.3, "dark", 0), ("square", 0.7, "light", 3), ("triangle", 0.3, "light", 0), ("triangle", 0.7, "dark", 2))]
sheet(pick, scale=0.8).save(os.path.join(OUT, "contact_alt_zoom.png"))
print(f"done; {len(fails)} failures")
sys.exit(1 if fails else 0)
