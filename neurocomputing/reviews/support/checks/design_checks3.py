"""Edge-orientation histogram correlation (straight notch edges only): rig theta illusory vs stim_v3 mis theta, vs the
stim_v3 illusory display, for every configuration and theta in 10, 20, 30. Reads only; writes design_checks3.json."""
import json, math, os
import numpy as np
from PIL import Image
V3 = "/Users/hamitsoyel/Downloads/illusion_research/v2/stim_v3/img"
SR = "/Users/hamitsoyel/Downloads/illusion_research/neuro/reanalysis/stim_rigid/img"
W, H = 1024, 768
POS = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)]
def verts(shape, cx, cy, side, phi):
    if shape == "square":
        h = side / 2; V = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    else:
        R = side / math.sqrt(3); V = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return [(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)) for x, y in V]
yy, xx = np.mgrid[0:H, 0:W]
def hist(path, V, r):
    g = np.array(Image.open(path))[..., 0].astype(float)
    m = np.zeros((H, W), bool)
    for x, y in V:
        d = np.hypot(xx - x, yy - y); m |= (d > 0.25 * r) & (d < 0.85 * r)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]; gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy); th = (np.degrees(np.arctan2(gy, gx)) + 90) % 180
    sel = m & (mag > 30)
    h, _ = np.histogram(th[sel], bins=90, range=(0, 180), weights=mag[sel])
    return h / h.sum()
rows = []
cfgs = sorted({f.rsplit("_rig", 1)[0] for f in os.listdir(SR)})
for cfg in cfgs:
    sh, srs, pol, ps = cfg.split("_"); sr = int(srs[2:]) / 100; p = int(ps[1:])
    side = 190 if sh == "square" else 230; cx, cy = POS[p]; r = sr * side / 2
    h_ill = hist(f"{V3}/{cfg}_illusory.png", verts(sh, cx, cy, side, 0), r)
    for th in (10, 20, 30):
        h_mis = hist(f"{V3}/{cfg}_mis{th}.png", verts(sh, cx, cy, side, 0), r)
        h_rig = hist(f"{SR}/{cfg}_rig{th}_illusory.png", verts(sh, cx, cy, side, th), r)
        h_rc = hist(f"{SR}/{cfg}_rig{th}_control180.png", verts(sh, cx, cy, side, th), r)
        rows.append(dict(cfg=cfg, theta=th, corr_rig_mis=float(np.corrcoef(h_rig, h_mis)[0, 1]),
                         corr_rig_ill=float(np.corrcoef(h_rig, h_ill)[0, 1]), corr_rigctrl_mis=float(np.corrcoef(h_rc, h_mis)[0, 1]),
                         l1_rig_mis=float(np.abs(h_rig - h_mis).sum()), l1_rig_ill=float(np.abs(h_rig - h_ill).sum())))
out = dict(n=len(rows), corr_rig_mis_min=min(r_["corr_rig_mis"] for r_ in rows), corr_rig_ill_max=max(r_["corr_rig_ill"] for r_ in rows),
           corr_rigctrl_mis_min=min(r_["corr_rigctrl_mis"] for r_ in rows),
           l1_rig_mis_max=max(r_["l1_rig_mis"] for r_ in rows), l1_rig_ill_min=min(r_["l1_rig_ill"] for r_ in rows))
json.dump(dict(summary=out, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "design_checks3.json"), "w"), indent=1)
print(out)
