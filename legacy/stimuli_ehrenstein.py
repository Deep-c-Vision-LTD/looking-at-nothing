"""
Ehrenstein figures (radial lines whose inner ends induce an illusory bright disc).
Kavsek (2024) used Ehrenstein figures alongside Kanizsa squares, so the same human free-viewing reference applies.

Graded factor: number of lines n in {4, 6, 8, 12, 16, 24} (illusory brightness/contour strength grows with the
number of line ends, analogous to the Kanizsa support ratio).

Conditions per configuration (identical lines, so identical ink, length and count; only their arrangement changes):
  illusory    : radial lines from radius R to R + L (inner ends on a circle of radius R)
  real        : illusory display plus the drawn circle of radius R
  control_mid : every line rotated 45 deg about its MIDPOINT (pinwheel; inner ends move outward,
                so a low-level edge model predicts LESS saliency inside -> confounded in favour of the illusion)
  control_end : every line rotated 45 deg about its INNER END (inner ends stay on the circle, the rest of each
                line moves closer to it -> a low-level edge model predicts MORE saliency inside, i.e. this control
                is confounded AGAINST the illusion and is the conservative test)
Rule set before any model was run: the primary control is the one for which the edge-energy null
does not predict illusory > control; the other is reported as secondary.
Varied: n (6) x size R in {70, 100} x polarity (2) x position (5) x condition (4) = 480 images; L = 0.9 R, width 4 px.
AOIs: inside = disc r <= R; inducers = annulus R < r <= R + L (all line pixels lie in it in every condition).
"""
import json, math, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "ehrenstein"); os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
W, H, SS = 1024, 768, 4
POSITIONS = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)]
NS = (4, 6, 8, 12, 16, 24)
CONDS = ("illusory", "real", "control_mid", "control_end")

def segment(cx, cy, R, L, t, cond):
    """End points of one line at polar angle t."""
    ux, uy = math.cos(t), math.sin(t)
    a = math.pi / 4
    if cond == "control_mid":
        mx, my = cx + (R + L / 2) * ux, cy + (R + L / 2) * uy
        vx, vy = math.cos(t + a), math.sin(t + a)
        return (mx - vx * L / 2, my - vy * L / 2), (mx + vx * L / 2, my + vy * L / 2)
    x0, y0 = cx + R * ux, cy + R * uy
    if cond == "control_end":
        vx, vy = math.cos(t + a), math.sin(t + a)
    else:
        vx, vy = ux, uy
    return (x0, y0), (x0 + vx * L, y0 + vy * L)

def figure(cx, cy, R, n, cond, fg, bg, width=4):
    L = 0.9 * R
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    t0 = math.pi / n                                           # no line exactly horizontal/vertical for even n
    for k in range(n):
        p, q = segment(cx, cy, R, L, t0 + 2 * math.pi * k / n, cond)
        d.line([(p[0] * SS, p[1] * SS), (q[0] * SS, q[1] * SS)], fill=fg, width=width * SS)
    if cond == "real":
        d.ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], outline=fg, width=3 * SS)
    img = img.resize((W, H), Image.LANCZOS)
    yy, xx = np.mgrid[0:H, 0:W]; rr = np.hypot(xx - cx, yy - cy)
    return img, rr <= R, (rr > R) & (rr <= R + L + width)

def make():
    items = []
    for R in (70, 100):
        for n in NS:
            for pol, (fg, bg) in (("dark", (0, 255)), ("light", (255, 0))):
                for pi, (cx, cy) in enumerate(POSITIONS):
                    for cond in CONDS:
                        img, ins, ind = figure(cx, cy, R, n, cond, fg, bg)
                        name = f"ehr_R{R}_n{n}_{pol}_p{pi}_{cond}"
                        img.convert("RGB").save(os.path.join(OUT, "img", name + ".png"))
                        np.savez_compressed(os.path.join(OUT, "img", name + "_aoi.npz"), inside=ins, ind=ind)
                        items.append(dict(name=name, shape="ehrenstein", R=R, n=n, pol=pol, pos=pi, cond=cond))
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    return items

if __name__ == "__main__":
    print(len(make()), "stimuli")
