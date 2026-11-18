"""Follow-up checks (design review): edge-orientation peaks near the straight notch edges only; rotation test with the
correct and the mirrored direction per display; location of the largest pixel differences; notch measurement on the
stim_v3 displays as the baseline of the same pixel method. Reads only; writes design_checks2.json here."""
import json, math, os, sys
import numpy as np
from PIL import Image

RE = "/Users/hamitsoyel/Downloads/illusion_research/neuro/reanalysis"
V3 = "/Users/hamitsoyel/Downloads/illusion_research/v2/stim_v3/img"
SR = os.path.join(RE, "stim_rigid")
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RE, "stimcode"))
import stimuli_v3 as T
W, H, SS = 1024, 768, 4
POS = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)]


def verts(shape, cx, cy, side, phi):
    if shape == "square":
        h = side / 2; V = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    else:
        R = side / math.sqrt(3); V = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return [(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)) for x, y in V]


def g_of(path):
    return np.array(Image.open(path))[..., 0].astype(np.float64)


def peaks(g, V, r):
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros((H, W), bool)
    for x, y in V:
        d = np.hypot(xx - x, yy - y); m |= (d > 0.25 * r) & (d < 0.85 * r)      # straight notch edges only, no arcs
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]; gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy); th = (np.degrees(np.arctan2(gy, gx)) + 90) % 180
    sel = m & (mag > 60)
    h, _ = np.histogram(th[sel], bins=180, range=(0, 180), weights=mag[sel])
    hs = np.array([h[(np.arange(i - 2, i + 3)) % 180].sum() for i in range(180)])
    p1 = int(hs.argmax()); hs2 = hs.copy()
    for d in range(-15, 16):
        hs2[(p1 + d) % 180] = 0
    p2 = int(hs2.argmax())
    return sorted([p1, p2]), float(hs2.max() / hs.max())


out = {"orientation": {}, "rotation": [], "maxdiff_loc": [], "notch_baseline": {}}
for cfg in ("square_sr50_dark_p0", "square_sr70_light_p3", "square_sr30_dark_p1", "triangle_sr50_dark_p1", "triangle_sr70_light_p4"):
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230; cx, cy = POS[p]; r = sr * side / 2
    row = {}
    for lab, path, phi in (("v3_illusory", f"{V3}/{cfg}_illusory.png", 0), ("v3_control180", f"{V3}/{cfg}_control180.png", 0),
                           ("v3_mis20", f"{V3}/{cfg}_mis20.png", 0), ("rig20_illusory", f"{SR}/img/{cfg}_rig20_illusory.png", 20),
                           ("rig20_control180", f"{SR}/img/{cfg}_rig20_control180.png", 20),
                           ("v3_mis10", f"{V3}/{cfg}_mis10.png", 0), ("rig10_illusory", f"{SR}/img/{cfg}_rig10_illusory.png", 10),
                           ("v3_mis30", f"{V3}/{cfg}_mis30.png", 0), ("rig30_illusory", f"{SR}/img/{cfg}_rig30_illusory.png", 30)):
        row[lab] = peaks(g_of(path), verts(sh, cx, cy, side, phi), r)
    out["orientation"][cfg] = row

yy, xx = np.mgrid[0:H, 0:W]
cfgs = sorted({f.rsplit("_rig", 1)[0] for f in os.listdir(os.path.join(SR, "img"))})
for k, cfg in enumerate(cfgs):
    if k % 4:
        continue
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230; cx, cy = POS[p]; r = sr * side / 2
    fg, bg = (0, 255) if pol == "dark" else (255, 0)
    for cond, rot in (("illusory", 0), ("control180", 180)):
        hi = T.kanizsa_hi(sh, cx, cy, side, sr, rot, False, False, fg, bg)
        for phi in (10, 20, 30, 45):
            g = g_of(f"{SR}/img/{cfg}_rig{phi}_{cond}.png")
            ok = np.array(T.down(hi.rotate(-phi, resample=Image.BICUBIC, center=(cx * SS, cy * SS), fillcolor=bg))).astype(float)
            wr = np.array(T.down(hi.rotate(+phi, resample=Image.BICUBIC, center=(cx * SS, cy * SS), fillcolor=bg))).astype(float)
            d = np.abs(g - ok)
            symmetric = (sh == "square" and phi == 45)
            out["rotation"].append(dict(name=f"{cfg}_rig{phi}_{cond}", mad_ok=float(d.mean()), mad_mirror=float(np.abs(g - wr).mean()),
                                        mirror_identical_by_symmetry=symmetric, max_ok=float(d.max()),
                                        n_gt64=int((d > 64).sum())))
            if d.max() > 64:
                V = verts(sh, cx, cy, side, phi)
                iy, ix = np.unravel_index(int(d.argmax()), d.shape)
                dc = min(math.hypot(ix - x, iy - y) for x, y in V)
                out["maxdiff_loc"].append(dict(name=f"{cfg}_rig{phi}_{cond}", max=float(d.max()), dist_to_inducer_centre_px=dc,
                                               r=r, at_rim=abs(dc - r) < 2.5, at_apex=dc < 2.5))

# notch measured by the same pixel method on stim_v3 (baseline error of the method)
errs = {"v3": [], "rig": []}
for cfg in cfgs:
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230; cx, cy = POS[p]; r = sr * side / 2
    bg = 255 if pol == "dark" else 0
    for src, path, phi, rot in (("v3", f"{V3}/{cfg}_illusory.png", 0, 0), ("v3", f"{V3}/{cfg}_control180.png", 0, 180),
                                ("rig", f"{SR}/img/{cfg}_rig20_illusory.png", 20, 0), ("rig", f"{SR}/img/{cfg}_rig45_control180.png", 45, 180)):
        g = g_of(path); V = verts(sh, cx, cy, side, phi)
        gx_, gy_ = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
        for x, y in V:
            angs = np.arange(0, 360, 0.25); rr = 0.6 * r
            vals = g[np.round(y + rr * np.sin(np.radians(angs))).astype(int), np.round(x + rr * np.cos(np.radians(angs))).astype(int)]
            e = np.abs(vals - bg) < 128; e2 = np.r_[e, e]; best = cur = bs = 0
            for i, v in enumerate(e2):
                cur = cur + 1 if v else 0
                if cur > best:
                    best, bs = cur, i - cur + 1
            c_ = (angs[bs % 1440] + best * 0.125) % 360
            want = (math.degrees(math.atan2(gy_ - y, gx_ - x)) + rot) % 360
            errs[src].append((abs((c_ - want + 180) % 360 - 180), best * 0.25 - (90 if sh == "square" else 60), sr))
for src in errs:
    a = np.array(errs[src])
    out["notch_baseline"][src] = dict(n=len(a), max_dir_err=float(a[:, 0].max()), mean_dir_err=float(a[:, 0].mean()),
                                      width_err_range=[float(a[:, 1].min()), float(a[:, 1].max())],
                                      max_dir_err_sr70=float(a[a[:, 2] == 0.7, 0].max()))
R = out["rotation"]
out["rotation_summary"] = dict(n=len(R), mad_ok_max=max(x["mad_ok"] for x in R),
                               mad_mirror_min_nonsym=min(x["mad_mirror"] for x in R if not x["mirror_identical_by_symmetry"]),
                               ratio_mirror_over_ok_min_nonsym=min(x["mad_mirror"] / x["mad_ok"] for x in R if not x["mirror_identical_by_symmetry"]),
                               max_ok=max(x["max_ok"] for x in R), n_gt64_max=max(x["n_gt64"] for x in R),
                               maxdiff_at_rim=sum(x["at_rim"] for x in out["maxdiff_loc"]),
                               maxdiff_at_apex=sum(x["at_apex"] for x in out["maxdiff_loc"]), maxdiff_n=len(out["maxdiff_loc"]))
json.dump(out, open(os.path.join(OUTD, "design_checks2.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("rotation", "maxdiff_loc")}, indent=1))
print(out["maxdiff_loc"][:6])
