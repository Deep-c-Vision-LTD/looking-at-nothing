"""
render_rigid.py: whole-display rigid rotation set for primary contrast P5 of REANALYSIS_RULE.md (mock review A9).

Starts from looking-at-nothing/stimuli/stimuli_v3.py, copied verbatim to stimcode/ (with stimuli_v2.py and
stimuli_ehr.py, which it imports; stimcode/paths.py is a shim whose folders are never written). Same canvas
(1024 x 768), background and polarity colours, 4x supersampling with LANCZOS downsampling, inducer radius rule
(r = sr x side / 2), notch width (90 deg squares, 60 deg triangles), positions and 60 Kanizsa configurations.

The only change: before drawing, every inducer centre is rotated about the figure centre (the centroid of the
vertices, which equals the configuration's (cx, cy) for both shapes) by phi = 10, 20, 30 or 45 deg, in the PIL angle
convention used for the stim_v3 misaligned inducers (positive = clockwise on screen). Each notch direction is
computed exactly as in stimuli_v2.figure (towards the centroid, plus the condition's own rotation), so it turns by phi
together with its centre. The illusory figure is therefore kept (all notches still aligned on the figure's edges,
which are now oblique), and the 180 deg control is the same rigid rotation of the stim_v3 control180 display.
For every phi the notch edges have exactly the orientations of the stim_v3 uniform rotation by phi (each notch turned
by phi); what differs from uniform rotation is that the inducer centres move with the figure, so alignment is kept.

Conditions per configuration and phi: rig<phi>_illusory, rig<phi>_control180 (2 x 4 x 60 = 480 displays).

Orientation-matched control of the uniform rotation (added 18 November 2026 after the design review, finding 1, before
the rule was fixed): mis<theta>_control180 for theta = 10, 20, 30 (3 x 60 = 180 displays): every control180 inducer
turned by a further +theta about its own centre, inducer centres unrotated, i.e. stimuli_v3.kanizsa_hi with rotation
180 + theta (the stim_v3 mis<theta> display has rotation theta). Its notch edges therefore have exactly the
orientations of stim_v3 mis<theta> (a line through the inducer centre at angle a and at a + 180 is the same line),
with outward notches, so that every term of P5 is a contrast against a control with the same edge orientation.
AOIs: the stim_v3 AOIs of the configuration (identical across stim_v3 conditions), stored per display as
aoi/<cfg>_mis<theta>c_aoi.npz. Checks: the same code with rotation theta reproduces the stim_v3 mis<theta> PNG pixel
for pixel (same renderer and sign convention); notch directions equal those of stim_v3 mis<theta> plus 180 exactly;
the magnitude-weighted gradient-orientation histogram (mod 180 deg, 36 bins) correlates more with stim_v3 mis<theta>
than with the unrotated stim_v3 control180 for every display; ink mass within 0.5% of stim_v3 control180.
AOIs, computed with the stimuli_v2 / stimuli_v3 / stimuli_alt code applied to the rotated vertices (so they rotate
identically): inside (polygon minus the inducer discs), ind (union of the discs), inside_eroded, inside_eroded_all,
core30, core45, polygon (the whole contour area, exact pixel-centre raster, for the Kavsek-style and region-variant
AOIs of A12 and A13; see polygon_exact), and per
condition ink (|pixel - background| >= 128 in the rendered display, i.e. at least half the inducer contrast; the
plan's "pixel differs from the background" would add the faint LANCZOS ringing halo, up to 20 grey levels and about
3 px wide) and notch (analytic notch sector minus ink), plus the analytic notch sectors (notchgeom, half-open angular
test).

Checks (all written to manifest.json; every comparison is against phi = 0 measured the same way):
  - phi = 0 rendered in memory with this code equals the stim_v3 illusory and control180 PNGs pixel for pixel, and
    its AOIs equal the stim_v3 AOI files (and core30, core45 the stim_alt AOI files), for all 60 configurations;
  - figure centre: the centroid of the rotated inducer centres equals (cx, cy) (to 1e-9 px); the centroid of the ind
    AOI moves by at most 0.15 px and the centroid of the rendered ink mass by at most 0.5 px from its phi = 0 value
    (the stim_v3 renderer draws the inducers about 0.38 px up and left of their nominal centres, in every condition;
    rotation keeps that offset); the inside and polygon centroids are reported (limit 0.75 px: PIL fills a polygon
    with fractional vertices about half a pixel up and left, which the published triangle AOIs share);
  - AOI areas within 1% of phi = 0 (absolute pixel differences reported; the 45 px core is reported separately because
    it is the smallest region); inducer ink mass (sum of |pixel - background| / 255) within 0.5% of phi = 0 (the
    binary count of antialiased ink pixels depends on edge orientation and is reported, not required);
  - notch: no ink inside the analytic notch sector (pixels more than 1.5 px inside its boundary); notch-sector area
    compared with phi = 0 (displays above 1% listed: small sectors, about 600 px each, carry pixel noise);
  - configurations where any inducer disc would come within 1 px of the canvas edge are refused (counted), and no
    rendered ink may touch the canvas border.
Notch definition used by the rule (A4.4): analytic notch sector minus ink. "Disc minus ink" is not used, because the
0.38 px drawing offset leaves a rim of uninked disc pixels outside the notch (counted and reported here).
Usage: python render_rigid.py            -> stim_rigid/{img/*.png, aoi/*_aoi.npz, items.json, manifest.json, contact_sheet.png}
       python render_rigid.py --check-only   (renders in memory, writes nothing but prints the checks)
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stimcode"))
import stimuli_v3 as T            # noqa: E402  (verbatim copy; imports stimuli_v2 as T.S)
S = T.S
W, H, SS = T.W, T.H, T.SS
ROOT = os.environ.get("LAN_PROJECT_ROOT") or os.path.dirname(os.path.dirname(HERE))   # project root (neuro/reanalysis is two levels down)
STIM_V3 = os.path.join(ROOT, "v2", "stim_v3", "img")            # read only (reference for the phi = 0 checks)
STIM_ALT_AOI = os.path.join(ROOT, "v3", "stim_alt", "aoi")      # read only
OUT = os.path.join(HERE, "stim_rigid")
PHIS = (10, 20, 30, 45)
CONDS = {"illusory": 0, "control180": 180}
MIS = (10, 20, 30)                 # orientation-matched controls of the stim_v3 uniform rotations (design review 1)
OBINS = 36                         # gradient-orientation histogram bins (mod 180 deg)
CORE_D = (30, 45)
ERODE_PX = T.ERODE_PX
MARGIN = 1.0


def rigid_vertices(shape, cx, cy, side, phi):
    V = S.vertices(shape, cx, cy, side)
    if phi == 0:
        return V
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return [(gx + c * (x - gx) - s * (y - gy), gy + s * (x - gx) + c * (y - gy)) for x, y in V]


def kanizsa_hi_rigid(shape, cx, cy, side, sr, rot, phi, fg, bg):
    """stimuli_v3.kanizsa_hi (no outline, no fill) with the vertices rotated rigidly by phi about the centroid."""
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    V = rigid_vertices(shape, cx, cy, side, phi); r = sr * side / 2
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    gap = 90 if shape == "square" else 60
    for x, y in V:
        th = math.degrees(math.atan2(gy - y, gx - x)) + rot
        d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], th + gap / 2, th - gap / 2 + 360, fill=fg)
    return img


def notch_dirs(shape, cx, cy, side, phi, rot):
    V = rigid_vertices(shape, cx, cy, side, phi)
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    return V, [math.degrees(math.atan2(gy - y, gx - x)) + rot for x, y in V]


def aoi_rigid(shape, cx, cy, side, sr, phi):
    """AOIs exactly as stimuli_v2.figure / stimuli_v3.kanizsa_aoi / stimuli_alt.aoi_with_core build them, from the
    rotated vertices."""
    V = rigid_vertices(shape, cx, cy, side, phi); r = sr * side / 2
    yy, xx = np.mgrid[0:H, 0:W]
    discs = np.zeros((H, W), bool)
    for x, y in V:
        discs |= (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    poly = Image.new("L", (W, H), 0); ImageDraw.Draw(poly).polygon(V, fill=1)
    poly = np.array(poly, bool)                         # the published raster (defines inside, as in stim_v3)
    inside = poly & ~discs
    a = dict(inside=inside, ind=discs,
             inside_eroded=inside & (distance_transform_edt(poly) > ERODE_PX),
             inside_eroded_all=inside & (distance_transform_edt(inside) > ERODE_PX),
             polygon=polygon_exact(V))
    dd = distance_transform_edt(~discs)
    for D in CORE_D:
        a[f"core{D}"] = inside & (dd > D)
    return a


def polygon_exact(V):
    """Whole contour area for the exploratory A12 (Kavsek-style) and A13 (region variant) AOIs: pixel centres inside
    the convex polygon, with a fixed sub-pixel perturbation as tie-break so that a pixel centre on an edge counts on
    exactly one side. Unlike PIL's fill (which the published inside AOI uses and which shifts fractional polygons by
    about half a pixel), this raster is rotation-consistent; the rule applies the same function at phi = 0 to the
    stim_v3 displays."""
    yy, xx = np.mgrid[0:H, 0:W]
    xs, ys = xx + 3.7e-7, yy + 6.1e-7
    sgn = []
    n = len(V)
    for k in range(n):
        (x1, y1), (x2, y2) = V[k], V[(k + 1) % n]
        sgn.append((x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1))
    sgn = np.stack(sgn)
    return np.all(sgn > 0, 0) | np.all(sgn < 0, 0)


def notch_geom(shape, cx, cy, side, sr, phi, rot):
    """Analytic notch sectors: disc pixels whose direction from the inducer centre lies within gap / 2 of the notch."""
    V, ths = notch_dirs(shape, cx, cy, side, phi, rot); r = sr * side / 2
    gap = 90 if shape == "square" else 60
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((H, W), bool)
    for (x, y), th in zip(V, ths):
        inn = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        ang = np.degrees(np.arctan2(yy - y, xx - x))
        dif = (ang - th + 180.0) % 360.0 - 180.0
        out |= inn & (dif >= -gap / 2) & (dif < gap / 2)          # half-open, so a pixel on an edge counts once
    return out


def fits(shape, cx, cy, side, sr, phi):
    V = rigid_vertices(shape, cx, cy, side, phi); r = sr * side / 2
    return all(x - r >= MARGIN and x + r <= W - 1 - MARGIN and y - r >= MARGIN and y + r <= H - 1 - MARGIN for x, y in V)


def kz_configs():
    for ill, p in T.configs():
        if ill == "kanizsa":
            yield p


def cfg_name(p):
    return T.k_name(p["shape"], p["sr"], p["pol"], p["pos"], "illusory")[: -len("_illusory")]


def render(p, cond, phi):
    return T.down(kanizsa_hi_rigid(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], CONDS[cond], phi, p["fg"], p["bg"]))


INK_MIN = 128          # ink = |pixel - background| >= 128 (half the inducer contrast); excludes the faint LANCZOS
                       # ringing (up to 20 grey levels, up to about 3 px from an edge) that "pixel != background" picks up


def ink_mask(arr, bg):
    return np.abs(arr.astype(np.int16) - bg) >= INK_MIN


def ink_stats(arr, bg):
    w = np.abs(arr.astype(np.float64) - bg) / 255.0
    yy, xx = np.mgrid[0:H, 0:W]
    m = w.sum()
    return dict(mass=float(m), count=int(ink_mask(arr, bg).sum()),
                cx=float((w * xx).sum() / m), cy=float((w * yy).sum() / m))


def centroid(mask):
    yy, xx = np.nonzero(mask)
    return float(xx.mean()), float(yy.mean())


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def render_rot(p, rot):
    """stimuli_v3.kanizsa_hi (no outline, no fill) with inducer rotation rot, unrotated positions, downsampled."""
    return T.down(T.kanizsa_hi(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], rot, False, False, p["fg"], p["bg"]))


def orient_hist(arr):
    """Gradient-magnitude-weighted histogram of gradient orientation mod 180 deg (OBINS bins), normalised to sum 1."""
    a = arr.astype(np.float64)
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]; gy[1:-1, :] = a[2:, :] - a[:-2, :]
    mag = np.hypot(gx, gy); ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    h = np.bincount(np.minimum((ang / (180.0 / OBINS)).astype(int), OBINS - 1).ravel(), weights=mag.ravel(),
                    minlength=OBINS)
    return h / h.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check-only", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    if not a.check_only:
        os.makedirs(os.path.join(OUT, "img"), exist_ok=True); os.makedirs(os.path.join(OUT, "aoi"), exist_ok=True)
    items, refused = [], []
    checks = dict(n_configs=0, phi0_pixels_equal=0, phi0_aoi_equal=0, phi0_core_equal=0, mis_inward_pixels_equal=0)
    worst = dict(aoi_rel={}, aoi_abs={}, ink_mass_rel=0.0, ink_count_rel_reported=0.0, vertex_centroid_px=0.0,
                 ind_centroid_shift_px=0.0, ink_centroid_shift_px=0.0, inside_centroid_shift_px=0.0,
                 polygon_centroid_shift_px=0.0, notch_ink_px=0, notch_area_rel=0.0, rim_px_phi0_max=0, rim_px_rot_max=0,
                 mis_notch_dir_dev_deg=0.0, mis_orient_corr_with_mis_min=1.0, mis_orient_corr_with_control180_max=-1.0,
                 mis_ink_mass_rel=0.0)
    over1 = {}
    fails = []
    keys_rel = ("inside", "ind", "polygon", "inside_eroded", "inside_eroded_all", "core30", "core45")
    for p in kz_configs():
        cfg = cfg_name(p); checks["n_configs"] += 1
        # ---- phi = 0 reference checks against the published files
        a0 = aoi_rigid(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], 0)
        ref = np.load(os.path.join(STIM_V3, f"{cfg}_illusory_aoi.npz"))
        if all(np.array_equal(a0[k], ref[k]) for k in ("inside", "ind", "inside_eroded", "inside_eroded_all")):
            checks["phi0_aoi_equal"] += 1
        else:
            fails.append(f"{cfg}: phi = 0 AOI differs from stim_v3")
        refc = np.load(os.path.join(STIM_ALT_AOI, f"{cfg}_aoi.npz"))
        if all(np.array_equal(a0[f"core{D}"], refc[f"core{D}"]) for D in CORE_D):
            checks["phi0_core_equal"] += 1
        else:
            fails.append(f"{cfg}: phi = 0 core AOI differs from stim_alt")
        base, ng0 = {}, {}
        for cond in CONDS:
            im0 = np.array(render(p, cond, 0))
            pub = np.array(Image.open(os.path.join(STIM_V3, f"{cfg}_{cond}.png")).convert("L"))
            if np.array_equal(im0, pub):
                checks["phi0_pixels_equal"] += 1
            else:
                fails.append(f"{cfg} {cond}: phi = 0 pixels differ from stim_v3 (max |d| {np.abs(im0.astype(int) - pub).max()})")
            base[cond] = ink_stats(im0, p["bg"])
            ng0[cond] = notch_geom(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], 0, CONDS[cond])
            rim0 = int((a0["ind"] & ~ink_mask(im0, p["bg"]) & ~(distance_transform_edt(~ng0[cond]) <= 1.5)).sum())
            worst["rim_px_phi0_max"] = max(worst["rim_px_phi0_max"], rim0)
        area0 = {k: int(a0[k].sum()) for k in keys_rel}
        c0 = {k: centroid(a0[k]) for k in ("ind", "inside", "polygon")}
        for phi in PHIS:
            if not fits(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], phi):
                refused.append(f"{cfg}_rig{phi}"); continue
            V = rigid_vertices(p["shape"], p["cx"], p["cy"], p["side"], phi)
            ev = math.hypot(np.mean([v[0] for v in V]) - p["cx"], np.mean([v[1] for v in V]) - p["cy"])
            worst["vertex_centroid_px"] = max(worst["vertex_centroid_px"], ev)
            if ev > 1e-9:
                fails.append(f"{cfg} rig{phi}: vertex centroid {ev:.2e} px from the figure centre")
            A = aoi_rigid(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], phi)
            for k in keys_rel:
                ar = int(A[k].sum()); d = abs(ar - area0[k]); rel = d / max(area0[k], 1)
                worst["aoi_rel"][k] = max(worst["aoi_rel"].get(k, 0.0), rel)
                worst["aoi_abs"][k] = max(worst["aoi_abs"].get(k, 0), d)
                if rel > 0.01:
                    over1.setdefault(k, []).append(f"{cfg}_rig{phi} ({100 * rel:.2f}%, {d} of {area0[k]} px)")
                    if k != "core45":
                        fails.append(f"{cfg} rig{phi}: area of {k} differs by {100 * rel:.2f}% ({d} px of {area0[k]})")
            for k, lim in (("ind", 0.15), ("inside", 0.75), ("polygon", 0.75)):
                x, y = centroid(A[k]); e = math.hypot(x - c0[k][0], y - c0[k][1])
                worst[f"{k}_centroid_shift_px"] = max(worst[f"{k}_centroid_shift_px"], e)
                if e > lim:
                    fails.append(f"{cfg} rig{phi}: centroid of {k} moved {e:.2f} px from phi = 0")
            store = {k: A[k] for k in keys_rel}
            for cond in CONDS:
                img = render(p, cond, phi)
                arr = np.array(img)
                st = ink_stats(arr, p["bg"])
                rm = abs(st["mass"] - base[cond]["mass"]) / base[cond]["mass"]
                rc = abs(st["count"] - base[cond]["count"]) / base[cond]["count"]
                e = math.hypot(st["cx"] - base[cond]["cx"], st["cy"] - base[cond]["cy"])
                worst["ink_mass_rel"] = max(worst["ink_mass_rel"], rm)
                worst["ink_count_rel_reported"] = max(worst["ink_count_rel_reported"], rc)
                worst["ink_centroid_shift_px"] = max(worst["ink_centroid_shift_px"], e)
                if rm > 0.005 or e > 0.5:
                    fails.append(f"{cfg} rig{phi} {cond}: ink mass {100 * rm:.3f}%, ink centroid moved {e:.2f} px")
                border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
                if (border != p["bg"]).any():
                    fails.append(f"{cfg} rig{phi} {cond}: ink on the canvas border")
                ink = ink_mask(arr, p["bg"])
                ng = notch_geom(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], phi, CONDS[cond])
                ni = int((ng & (distance_transform_edt(ng) > 1.5) & ink).sum())
                na = abs(int(ng.sum()) - int(ng0[cond].sum())) / int(ng0[cond].sum())
                rim = int((A["ind"] & ~ink & ~(distance_transform_edt(~ng) <= 1.5)).sum())
                worst["notch_ink_px"] = max(worst["notch_ink_px"], ni)
                worst["notch_area_rel"] = max(worst["notch_area_rel"], na)
                worst["rim_px_rot_max"] = max(worst["rim_px_rot_max"], rim)
                if ni > 0:
                    fails.append(f"{cfg} rig{phi} {cond}: {ni} inked px inside the notch")
                if na > 0.01:
                    over1.setdefault("notch", []).append(f"{cfg}_rig{phi}_{cond} ({100 * na:.2f}%)")
                store[f"ink_{cond}"] = ink; store[f"notch_{cond}"] = ng & ~ink; store[f"notchgeom_{cond}"] = ng
                name = f"{cfg}_rig{phi}_{cond}"
                if not a.check_only:
                    img.convert("RGB").save(os.path.join(OUT, "img", name + ".png"))
                Vc, ths = notch_dirs(p["shape"], p["cx"], p["cy"], p["side"], phi, CONDS[cond])
                items.append(dict(name=name, illusion="kanizsa", config=f"{cfg}_rig{phi}", base_config=cfg,
                                  shape=p["shape"], sr=p["sr"], pol=p["pol"], pos=p["pos"], cond=f"rig{phi}_{cond}",
                                  base_cond=cond, rig_deg=phi, centre=[p["cx"], p["cy"]],
                                  inducer_centres=[[round(x, 4), round(y, 4)] for x, y in Vc],
                                  notch_dirs_deg=[round(t, 4) for t in ths], aoi=f"aoi/{cfg}_rig{phi}_aoi.npz",
                                  stim_v3_reference=f"v2/stim_v3/img/{cfg}_{cond}.png"))
            if not a.check_only:
                np.savez_compressed(os.path.join(OUT, "aoi", f"{cfg}_rig{phi}_aoi.npz"), **store)
        # ---- orientation-matched controls of the uniform rotation: mis<theta>_control180 (design review, finding 1)
        h_c180 = orient_hist(np.array(Image.open(os.path.join(STIM_V3, f"{cfg}_control180.png")).convert("L")))
        for th in MIS:
            pub_mis = np.array(Image.open(os.path.join(STIM_V3, f"{cfg}_mis{th}.png")).convert("L"))
            if np.array_equal(np.array(render_rot(p, th)), pub_mis):
                checks["mis_inward_pixels_equal"] += 1
            else:
                fails.append(f"{cfg} mis{th}: rotation {th} does not reproduce the stim_v3 mis{th} PNG")
            img = render_rot(p, 180 + th); arr = np.array(img)
            cond = f"mis{th}_control180"
            _, d_in = notch_dirs(p["shape"], p["cx"], p["cy"], p["side"], 0, th)
            Vc, ths = notch_dirs(p["shape"], p["cx"], p["cy"], p["side"], 0, 180 + th)
            dd = max(abs(((b - a_ - 180.0) + 180.0) % 360.0 - 180.0) for a_, b in zip(d_in, ths))
            worst["mis_notch_dir_dev_deg"] = max(worst["mis_notch_dir_dev_deg"], dd)
            if dd > 1e-9:
                fails.append(f"{cfg} {cond}: notch directions differ from stim_v3 mis{th} + 180 by {dd:.2e} deg")
            h = orient_hist(arr)
            c_mis = float(np.corrcoef(h, orient_hist(pub_mis))[0, 1]); c_c180 = float(np.corrcoef(h, h_c180)[0, 1])
            worst["mis_orient_corr_with_mis_min"] = min(worst["mis_orient_corr_with_mis_min"], c_mis)
            worst["mis_orient_corr_with_control180_max"] = max(worst["mis_orient_corr_with_control180_max"], c_c180)
            if not c_mis > c_c180:
                fails.append(f"{cfg} {cond}: orientation histogram closer to control180 ({c_c180:.3f}) than to mis{th} ({c_mis:.3f})")
            st = ink_stats(arr, p["bg"])
            rm = abs(st["mass"] - base["control180"]["mass"]) / base["control180"]["mass"]
            worst["mis_ink_mass_rel"] = max(worst["mis_ink_mass_rel"], rm)
            if rm > 0.005:
                fails.append(f"{cfg} {cond}: ink mass differs from stim_v3 control180 by {100 * rm:.3f}%")
            border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
            if (border != p["bg"]).any():
                fails.append(f"{cfg} {cond}: ink on the canvas border")
            ink = ink_mask(arr, p["bg"])
            ng = notch_geom(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], 0, 180 + th)
            ni = int((ng & (distance_transform_edt(ng) > 1.5) & ink).sum())
            worst["notch_ink_px"] = max(worst["notch_ink_px"], ni)
            if ni > 0:
                fails.append(f"{cfg} {cond}: {ni} inked px inside the notch")
            store = {k: a0[k] for k in keys_rel}                 # stim_v3 AOIs (checked equal at phi = 0 above)
            store[f"ink_{cond}"] = ink; store[f"notch_{cond}"] = ng & ~ink; store[f"notchgeom_{cond}"] = ng
            name = f"{cfg}_{cond}"
            if not a.check_only:
                img.convert("RGB").save(os.path.join(OUT, "img", name + ".png"))
                np.savez_compressed(os.path.join(OUT, "aoi", f"{cfg}_mis{th}c_aoi.npz"), **store)
            items.append(dict(name=name, illusion="kanizsa", config=f"{cfg}_mis{th}c", base_config=cfg,
                              shape=p["shape"], sr=p["sr"], pol=p["pol"], pos=p["pos"], cond=cond,
                              base_cond="control180", rig_deg=0, mis_deg=th, centre=[p["cx"], p["cy"]],
                              inducer_centres=[[round(x, 4), round(y, 4)] for x, y in Vc],
                              notch_dirs_deg=[round(t, 4) for t in ths], aoi=f"aoi/{cfg}_mis{th}c_aoi.npz",
                              stim_v3_reference=f"v2/stim_v3/img/{cfg}_mis{th}.png"))
        print(f"{cfg}: {2 * len(PHIS) + len(MIS)} displays ({time.time() - t0:.0f}s)", flush=True)
    summary = dict(n_displays=len(items), n_rigid=sum(1 for it in items if it["rig_deg"]),
                   n_mis_control180=sum(1 for it in items if not it["rig_deg"]), n_configs=checks["n_configs"],
                   phis=list(PHIS), conds=list(CONDS), mis_thetas=list(MIS),
                   refused=refused, n_refused=len(refused), checks=checks, worst=worst, aoi_over_1pct=over1,
                   failures=fails,
                   n_failures=len(fails))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    for f in fails[:40]:
        print("FAIL", f)
    if a.check_only:
        return
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    contact_sheet()
    files = sorted(os.path.relpath(os.path.join(dp, f), OUT) for dp, _, fs in os.walk(OUT) for f in fs
                   if f != "manifest.json")
    man = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               generator="neuro/reanalysis/render_rigid.py", generator_md5=md5(os.path.abspath(__file__)),
               source_copies={f: md5(os.path.join(HERE, "stimcode", f)) for f in
                              ("stimuli_v3.py", "stimuli_v2.py", "stimuli_ehr.py", "paths.py")},
               pillow=PIL.__version__, numpy=np.__version__, canvas=[W, H], supersampling=SS,
               rotation_convention="PIL angle convention: positive = clockwise on screen, about the vertex centroid",
               **summary, files={f: md5(os.path.join(OUT, f)) for f in files})
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in files) + os.path.getsize(os.path.join(OUT, "manifest.json"))
    print(f"wrote {len(files)} files + manifest.json, {tot / 1e6:.1f} MB, {time.time() - t0:.0f}s")


def contact_sheet():
    """Crops around the figure: stim_v3 illusory, stim_v3 mis20 (uniform), rigid 10/20/30/45 illusory, rigid 20
    control180, mis20 control180 (orientation-matched control of stim_v3 mis20); AOI outlines (red inside, blue inducer discs, green core30) on the rigid tiles; plus full-canvas
    thumbnails of the rigid 45 deg display for an off-centre position."""
    sel = ["square_sr50_dark_p0", "triangle_sr50_light_p0", "square_sr70_light_p3", "triangle_sr30_dark_p4"]
    meta = {cfg_name(p): p for p in kz_configs()}
    tile, crop = 230, 440
    cols = ["stim_v3 illusory", "stim_v3 mis20", "rig10", "rig20", "rig30", "rig45", "rig20 control180",
            "mis20 control180", "rig45 canvas"]
    sheet = Image.new("RGB", (tile * len(cols), (tile + 18) * len(sel) + 22), (128, 128, 128))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for j, c in enumerate(cols):
        dr.text((j * tile + 4, 4), c, fill=(255, 255, 0), font=font)
    for i, cfg in enumerate(sel):
        p = meta[cfg]
        x0, y0 = int(p["cx"] - crop / 2), int(p["cy"] - crop / 2)
        box = (max(x0, 0), max(y0, 0), min(x0 + crop, W), min(y0 + crop, H))
        tiles = [Image.open(os.path.join(STIM_V3, f"{cfg}_illusory.png")).convert("RGB"),
                 Image.open(os.path.join(STIM_V3, f"{cfg}_mis20.png")).convert("RGB")]
        for phi, cond in ((10, "illusory"), (20, "illusory"), (30, "illusory"), (45, "illusory"), (20, "control180")):
            im = Image.open(os.path.join(OUT, "img", f"{cfg}_rig{phi}_{cond}.png")).convert("RGB")
            A = np.load(os.path.join(OUT, "aoi", f"{cfg}_rig{phi}_aoi.npz"))
            arr = np.array(im)
            for k, col in (("inside", (255, 0, 0)), ("ind", (0, 90, 255)), ("core30", (0, 200, 0))):
                m = A[k]
                edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
                arr[edge] = col
            tiles.append(Image.fromarray(arr))
        tiles.append(Image.open(os.path.join(OUT, "img", f"{cfg}_mis20_control180.png")).convert("RGB"))
        y = 22 + i * (tile + 18)
        for j, im in enumerate(tiles):
            sheet.paste(im.crop(box).resize((tile, tile), Image.LANCZOS), (j * tile, y))
        full = Image.open(os.path.join(OUT, "img", f"{cfg}_rig45_illusory.png")).convert("RGB")
        th = full.resize((tile, int(tile * H / W)), Image.LANCZOS)
        sheet.paste(th, (8 * tile, y + (tile - th.size[1]) // 2))
        dr.text((4, y + tile + 3), cfg, fill=(255, 255, 0), font=font)
    sheet.save(os.path.join(OUT, "contact_sheet.png"))


if __name__ == "__main__":
    main()
