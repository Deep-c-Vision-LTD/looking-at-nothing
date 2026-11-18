"""
Independent checks of neuro/reanalysis/stim_rigid (design review, 18 Nov 2026). Reads only; writes a JSON summary and
two small PNG crops into this folder. No torch, no model, no scoring. Run with nice -n 10, OMP_NUM_THREADS=4.
Independent of render_rigid.py: geometry recomputed from the stated design (square side 190, triangle side 230, r =
sr x side / 2, positions, notch 90 / 60 deg), the rigid display compared with a bicubic rotation of the re-rendered
stim_v3 display, notch direction measured from the pixels, AOIs recomputed with a half-plane raster.
"""
import hashlib, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

RE = "/Users/hamitsoyel/Downloads/illusion_research/neuro/reanalysis"
ROOT = "/Users/hamitsoyel/Downloads/illusion_research"
SR = os.path.join(RE, "stim_rigid")
V3 = os.path.join(ROOT, "v2", "stim_v3", "img")
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RE, "stimcode"))
import stimuli_v3 as T  # verbatim copy (used only to re-render the phi = 0 hi-res display for the rotation test)

W, H, SS = 1024, 768, 4
POS = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)]
out = {}

# ---------------------------------------------------------------- 1. inventory and md5
man = json.load(open(os.path.join(SR, "manifest.json")))
bad = [f for f, m in man["files"].items() if hashlib.md5(open(os.path.join(SR, f), "rb").read()).hexdigest() != m]
on_disk = sorted(os.path.relpath(os.path.join(dp, f), SR) for dp, _, fs in os.walk(SR) for f in fs if f != "manifest.json")
items = json.load(open(os.path.join(SR, "items.json")))
exp_names = set()
for shape in ("square", "triangle"):
    for sr in (30, 50, 70):
        for pol in ("dark", "light"):
            for p in range(5):
                for phi in (10, 20, 30, 45):
                    for c in ("illusory", "control180"):
                        exp_names.add(f"{shape}_sr{sr}_{pol}_p{p}_rig{phi}_{c}")
pngs = {f[4:-4] for f in on_disk if f.startswith("img/")}
aois = {f[4:-8] for f in on_disk if f.startswith("aoi/")}
out["inventory"] = dict(md5_mismatch=len(bad), files_in_manifest=len(man["files"]), files_on_disk=len(on_disk),
                        manifest_equals_disk=sorted(man["files"]) == on_disk, png=len(pngs), aoi=len(aois),
                        expected_png=len(exp_names), missing_png=sorted(exp_names - pngs), extra_png=sorted(pngs - exp_names),
                        items=len(items), items_equal_png=set(i["name"] for i in items) == pngs,
                        missing_aoi=sorted({n.rsplit("_", 1)[0] for n in exp_names} - aois))


# ---------------------------------------------------------------- 2. geometry from the design
def verts(shape, cx, cy, side, phi):
    if shape == "square":
        h = side / 2; V = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    else:
        R = side / math.sqrt(3); V = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))   # y down: positive phi = clockwise on screen
    return [(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)) for x, y in V]


def parse(name):
    sh, sr, pol, p, rig, cond = name.split("_")
    return sh, int(sr[2:]) / 100, pol, int(p[1:]), int(rig[3:]), cond


geo_err_c, geo_err_n = 0.0, 0.0
for it in items:
    sh, sr, pol, p, phi, cond = parse(it["name"])
    side = 190 if sh == "square" else 230
    cx, cy = POS[p]
    V = verts(sh, cx, cy, side, phi)
    geo_err_c = max(geo_err_c, max(math.hypot(a - b[0], c - b[1]) for (a, c), b in zip(V, it["inducer_centres"])))
    rot = 0 if cond == "illusory" else 180
    for (x, y), th in zip(V, it["notch_dirs_deg"]):
        want = math.degrees(math.atan2(cy - y, cx - x)) + rot
        geo_err_n = max(geo_err_n, abs((want - th + 180) % 360 - 180))
out["geometry_vs_items"] = dict(max_centre_err_px=geo_err_c, max_notch_dir_err_deg=geo_err_n)

# ---------------------------------------------------------------- 3. pixels: stats, rotation test, measured notch
def load(path):
    im = Image.open(path)
    a = np.array(im)
    return im.mode, a


def stats(a, bg):
    g = a[..., 0].astype(np.int16)
    d = np.abs(g - bg)
    return dict(mass=float(d.sum() / 255), full=int((d == 255).sum()), aa=int(((d > 20) & (d < 235)).sum()),
                faint=int(((d > 0) & (d <= 20)).sum()), ink128=int((d >= 128).sum()),
                n_unique=int(len(np.unique(g))), border_nonbg=int(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]).__ne__(bg).sum()))


rot_rows, stat_rows, notch_rows, rgb_equal, modes = [], [], [], True, set()
cfgs = sorted({it["base_config"] for it in items})
for k, cfg in enumerate(cfgs):
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230
    cx, cy = POS[p]
    fg, bg = (0, 255) if pol == "dark" else (255, 0)
    r = sr * side / 2
    s0 = {}
    for cond in ("illusory", "control180", "mis20"):
        md, a = load(os.path.join(V3, f"{cfg}_{cond}.png")); s0[cond] = stats(a, bg); modes.add(("v3", md, a.shape))
    for phi in (10, 20, 30, 45):
        for cond, rot in (("illusory", 0), ("control180", 180)):
            md, a = load(os.path.join(SR, "img", f"{cfg}_rig{phi}_{cond}.png")); modes.add(("rig", md, a.shape))
            rgb_equal &= bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
            st = stats(a, bg)
            stat_rows.append(dict(name=f"{cfg}_rig{phi}_{cond}", phi=phi, cond=cond,
                                  mass_rel=st["mass"] / s0[cond]["mass"] - 1, aa_rel=st["aa"] / s0[cond]["aa"] - 1,
                                  aa_rel_vs_mis20=st["aa"] / s0["mis20"]["aa"] - 1 if cond == "illusory" and phi == 20 else None,
                                  full_rel=st["full"] / s0[cond]["full"] - 1, ink128_rel=st["ink128"] / s0[cond]["ink128"] - 1,
                                  n_unique=st["n_unique"], n_unique_v3=s0[cond]["n_unique"], border=st["border_nonbg"],
                                  bg_mode=int(np.bincount(a[..., 0].ravel()).argmax())))
            g = a[..., 0].astype(np.float64)
            # (a) rotation test on a subset: bicubic rotation of the re-rendered stim_v3 hi-res display about (cx, cy)
            if k % 6 == 0:
                hi = T.kanizsa_hi(sh, cx, cy, side, sr, rot, False, False, fg, bg)
                ref = np.array(T.down(hi.rotate(-phi, resample=Image.BICUBIC, center=(cx * SS, cy * SS), fillcolor=bg))).astype(np.float64)
                wrong = np.array(T.down(hi.rotate(+phi, resample=Image.BICUBIC, center=(cx * SS, cy * SS), fillcolor=bg))).astype(np.float64)
                v3img = load(os.path.join(V3, f"{cfg}_{cond}.png"))[1][..., 0].astype(np.float64)
                rot_rows.append(dict(name=f"{cfg}_rig{phi}_{cond}", mad=float(np.abs(g - ref).mean()), maxd=float(np.abs(g - ref).max()),
                                     p999=float(np.quantile(np.abs(g - ref), 0.9999)), frac_gt32=float((np.abs(g - ref) > 32).mean()),
                                     mad_wrong_dir=float(np.abs(g - wrong).mean()), mad_unrotated=float(np.abs(g - v3img).mean())))
            # (b) measured notch direction and width for each inducer, from the pixels (ring at 0.6 r)
            V = verts(sh, cx, cy, side, phi)
            gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
            for (x, y) in V:
                angs = np.arange(0, 360, 0.25)
                rr = 0.6 * r
                xs = x + rr * np.cos(np.radians(angs)); ys = y + rr * np.sin(np.radians(angs))
                vals = g[np.round(ys).astype(int), np.round(xs).astype(int)]
                empty = np.abs(vals - bg) < 128
                # circular run of empty samples
                e2 = np.concatenate([empty, empty]); best, cur, bstart = 0, 0, 0
                for i, v in enumerate(e2):
                    cur = cur + 1 if v else 0
                    if cur > best:
                        best, bstart = cur, i - cur + 1
                best = min(best, len(angs))
                centre = (angs[bstart % len(angs)] + best * 0.25 / 2) % 360
                want = (math.degrees(math.atan2(gy - y, gx - x)) + rot) % 360
                notch_rows.append(dict(err=abs((centre - want + 180) % 360 - 180), width=best * 0.25,
                                       gap=90 if sh == "square" else 60))
        if k % 10 == 0:
            print(cfg, flush=True)

out["pixels"] = dict(modes=sorted(map(str, modes)), rgb_channels_equal=rgb_equal,
                     bg_mode_ok=all(r_["bg_mode"] in (0, 255) for r_ in stat_rows),
                     border_nonbg_max=max(r_["border"] for r_ in stat_rows),
                     ink_mass_rel=[min(r_["mass_rel"] for r_ in stat_rows), max(r_["mass_rel"] for r_ in stat_rows)],
                     ink128_count_rel=[min(r_["ink128_rel"] for r_ in stat_rows), max(r_["ink128_rel"] for r_ in stat_rows)],
                     full_ink_count_rel=[min(r_["full_rel"] for r_ in stat_rows), max(r_["full_rel"] for r_ in stat_rows)],
                     aa_count_rel_vs_phi0=[min(r_["aa_rel"] for r_ in stat_rows), max(r_["aa_rel"] for r_ in stat_rows)],
                     aa_count_rel_rig20_vs_mis20=[min(r_["aa_rel_vs_mis20"] for r_ in stat_rows if r_["aa_rel_vs_mis20"] is not None),
                                                  max(r_["aa_rel_vs_mis20"] for r_ in stat_rows if r_["aa_rel_vs_mis20"] is not None)],
                     n_unique_grey=[min(r_["n_unique"] for r_ in stat_rows), max(r_["n_unique"] for r_ in stat_rows)],
                     n_unique_grey_v3=[min(r_["n_unique_v3"] for r_ in stat_rows), max(r_["n_unique_v3"] for r_ in stat_rows)])
out["rotation_test"] = dict(n=len(rot_rows), mad_max=max(r_["mad"] for r_ in rot_rows),
                            maxdiff_max=max(r_["maxd"] for r_ in rot_rows), frac_gt32_max=max(r_["frac_gt32"] for r_ in rot_rows),
                            mad_wrong_direction_min=min(r_["mad_wrong_dir"] for r_ in rot_rows),
                            mad_unrotated_min=min(r_["mad_unrotated"] for r_ in rot_rows))
out["notch_measured"] = dict(n=len(notch_rows), max_dir_err_deg=max(r_["err"] for r_ in notch_rows),
                             width_minus_gap=[min(r_["width"] - r_["gap"] for r_ in notch_rows), max(r_["width"] - r_["gap"] for r_ in notch_rows)])

# ---------------------------------------------------------------- 4. AOIs recomputed from geometry
def halfplane(V):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    n = len(V); sg = []
    for i in range(n):
        (x1, y1), (x2, y2) = V[i], V[(i + 1) % n]
        sg.append((x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1))
    sg = np.stack(sg)
    return np.all(sg >= 0, 0) | np.all(sg <= 0, 0)


aoi_rows = []
yy, xx = np.mgrid[0:H, 0:W]
for cfg in cfgs[::3]:
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230
    cx, cy = POS[p]; r = sr * side / 2
    a0 = np.load(os.path.join(V3, f"{cfg}_illusory_aoi.npz"))
    for phi in (10, 20, 30, 45):
        A = np.load(os.path.join(SR, "aoi", f"{cfg}_rig{phi}_aoi.npz"))
        V = verts(sh, cx, cy, side, phi)
        discs = np.zeros((H, W), bool)
        for x, y in V:
            discs |= (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        poly_pil = np.array((lambda im: (ImageDraw.Draw(im).polygon(V, fill=1), im)[1])(Image.new("L", (W, H), 0)), bool)
        poly_hp = halfplane(V)
        ins_pil, ins_hp = poly_pil & ~discs, poly_hp & ~discs
        dd = distance_transform_edt(~discs)
        # distance of disagreeing pixels from the true polygon boundary
        dis = ins_hp ^ A["inside"]
        bd = distance_transform_edt(~(poly_hp ^ distance_transform_edt(poly_hp).__gt__(1))) if False else None
        edge = poly_hp & ~(np.roll(poly_hp, 1, 0) & np.roll(poly_hp, -1, 0) & np.roll(poly_hp, 1, 1) & np.roll(poly_hp, -1, 1))
        dist_to_edge = distance_transform_edt(~edge)
        # rotation consistency: inverse-map phi = 0 AOIs (stim_v3 files) by -phi about (cx, cy), nearest neighbour
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        xs = cx + c * (xx - cx) + s * (yy - cy); ys = cy - s * (xx - cx) + c * (yy - cy)
        xi, yi = np.clip(np.round(xs).astype(int), 0, W - 1), np.clip(np.round(ys).astype(int), 0, H - 1)
        iou = {}
        for key in ("inside", "ind"):
            rm = a0[key][yi, xi]; st_ = A[key].astype(bool)
            iou[key] = float((rm & st_).sum() / (rm | st_).sum())
        aoi_rows.append(dict(name=f"{cfg}_rig{phi}", ind_exact=bool(np.array_equal(discs, A["ind"])),
                             inside_equals_pil=bool(np.array_equal(ins_pil, A["inside"])),
                             core30_exact=bool(np.array_equal(ins_pil & (dd > 30), A["core30"])),
                             core45_exact=bool(np.array_equal(ins_pil & (dd > 45), A["core45"])),
                             inside_eroded_exact=bool(np.array_equal(ins_pil & (distance_transform_edt(poly_pil) > 8), A["inside_eroded"])),
                             inside_vs_halfplane_px=int(dis.sum()),
                             inside_vs_halfplane_max_dist_to_edge=float(dist_to_edge[dis].max()) if dis.any() else 0.0,
                             polygon_vs_halfplane_px=int((A["polygon"] ^ poly_hp).sum()),
                             iou_inside_rotated_phi0=iou["inside"], iou_ind_rotated_phi0=iou["ind"],
                             inside_area_rel_phi0=float(A["inside"].sum() / a0["inside"].sum() - 1),
                             ink_outside_ind_dilated=int((A["ink_illusory"] & ~(distance_transform_edt(~A["ind"]) <= 1.0)).sum())))
out["aoi"] = dict(n=len(aoi_rows), ind_exact=all(r_["ind_exact"] for r_ in aoi_rows),
                  inside_equals_pil=all(r_["inside_equals_pil"] for r_ in aoi_rows),
                  core30_exact=all(r_["core30_exact"] for r_ in aoi_rows), core45_exact=all(r_["core45_exact"] for r_ in aoi_rows),
                  inside_eroded_exact=all(r_["inside_eroded_exact"] for r_ in aoi_rows),
                  inside_vs_halfplane_px_max=max(r_["inside_vs_halfplane_px"] for r_ in aoi_rows),
                  inside_vs_halfplane_max_dist_to_edge=max(r_["inside_vs_halfplane_max_dist_to_edge"] for r_ in aoi_rows),
                  polygon_vs_halfplane_px_max=max(r_["polygon_vs_halfplane_px"] for r_ in aoi_rows),
                  iou_inside_min=min(r_["iou_inside_rotated_phi0"] for r_ in aoi_rows),
                  iou_ind_min=min(r_["iou_ind_rotated_phi0"] for r_ in aoi_rows),
                  inside_area_rel_phi0=[min(r_["inside_area_rel_phi0"] for r_ in aoi_rows), max(r_["inside_area_rel_phi0"] for r_ in aoi_rows)],
                  ink_outside_ind_max=max(r_["ink_outside_ind_dilated"] for r_ in aoi_rows))

# ---------------------------------------------------------------- 5. edge orientation: rig20 vs mis20 vs illusory
def orient_peaks(g):
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]; gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy); th = (np.degrees(np.arctan2(gy, gx)) + 90) % 180      # edge orientation, y down
    h, _ = np.histogram(th[mag > 60], bins=180, range=(0, 180), weights=mag[mag > 60])
    h = np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5) / 5, "valid")
    return sorted(int(i) for i in np.argsort(h)[-2:])


ori = {}
for cfg in ("square_sr50_dark_p0", "square_sr70_light_p3", "triangle_sr50_dark_p1"):
    ori[cfg] = dict(illusory=orient_peaks(load(os.path.join(V3, f"{cfg}_illusory.png"))[1][..., 0].astype(float)),
                    mis20=orient_peaks(load(os.path.join(V3, f"{cfg}_mis20.png"))[1][..., 0].astype(float)),
                    rig20=orient_peaks(load(os.path.join(SR, "img", f"{cfg}_rig20_illusory.png"))[1][..., 0].astype(float)),
                    rig20_control=orient_peaks(load(os.path.join(SR, "img", f"{cfg}_rig20_control180.png"))[1][..., 0].astype(float)),
                    control180=orient_peaks(load(os.path.join(V3, f"{cfg}_control180.png"))[1][..., 0].astype(float)))
out["edge_orientation_peaks_deg"] = ori

# ---------------------------------------------------------------- 6. crops for viewing
def crop(path, cfg, size=300):
    p = int(cfg.split("_p")[1][0]); cx, cy = POS[p]
    return Image.open(path).convert("RGB").crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))


for cfg in ("square_sr50_dark_p0", "triangle_sr50_light_p2"):
    tiles = [crop(os.path.join(V3, f"{cfg}_{c}.png"), cfg) for c in ("illusory", "mis20", "control180")]
    tiles += [crop(os.path.join(SR, "img", f"{cfg}_rig20_{c}.png"), cfg) for c in ("illusory", "control180")]
    sheet = Image.new("RGB", (300 * len(tiles), 300), (128, 128, 128))
    for i, t in enumerate(tiles):
        sheet.paste(t, (300 * i, 0))
    sheet.resize((150 * len(tiles), 150)).save(os.path.join(OUTD, f"crop_{cfg}_v3ill_mis20_ctrl_rig20ill_rig20ctrl.png"))

json.dump(out, open(os.path.join(OUTD, "design_checks.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
