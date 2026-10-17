"""
Kanizsa squares and triangles with a graded support ratio (480 images).

Support ratio (SR) = proportion of each figure edge specified by the inducers = 2r / side
(Shipley & Kellman 1992: illusory-contour strength increases with SR). Levels SR = 0.2 ... 0.7.

Conditions per configuration:
  illusory  : inducer gaps face the figure centre (illusory figure)
  real      : illusory display plus the drawn outline of the figure
  control   : every inducer rotated 180 deg (gaps face outward; no figure)
  control90 : every inducer rotated 90 deg (gaps sideways; no figure)
Varied: shape (square, triangle) x SR (6) x polarity (2) x position (5) x condition (4) = 480 images.
AOIs: inside = figure interior minus the full inducer discs; inducers = union of the full discs.
"""
import json, math, os
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stimuli", "kanizsa"); os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
W, H, SS = 1024, 768, 4
POSITIONS = [(512, 384), (360, 300), (664, 300), (360, 468), (664, 468)]
SRS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
ROT = {"illusory": 0, "real": 0, "control": 180, "control90": 90}

def vertices(shape, cx, cy, side):
    if shape == "square":
        h = side / 2
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    R = side / math.sqrt(3)                                   # circumradius of the equilateral triangle
    return [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in (-90, 30, 150)]

def figure(shape, cx, cy, side, sr, cond, fg, bg):
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    V = vertices(shape, cx, cy, side); r = sr * side / 2
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    gap = 90 if shape == "square" else 60                       # interior angle of the figure at each vertex
    for x, y in V:
        th = math.degrees(math.atan2(gy - y, gx - x)) + ROT[cond]  # gap direction (PIL: clockwise from +x)
        d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], th + gap / 2, th - gap / 2 + 360, fill=fg)
    if cond == "real":
        d.polygon([(x * SS, y * SS) for x, y in V], outline=fg, width=3 * SS)
    img = img.resize((W, H), Image.LANCZOS)
    yy, xx = np.mgrid[0:H, 0:W]
    discs = np.zeros((H, W), bool)
    for x, y in V:
        discs |= (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    poly = Image.new("L", (W, H), 0); ImageDraw.Draw(poly).polygon(V, fill=1)
    inside = np.array(poly, bool) & ~discs
    return img, inside, discs

def make():
    items = []
    for shape, side in (("square", 190), ("triangle", 230)):
        for sr in SRS:
            for pol, (fg, bg) in (("dark", (0, 255)), ("light", (255, 0))):
                for pi, (cx, cy) in enumerate(POSITIONS):
                    for cond in ROT:
                        img, ins, ind = figure(shape, cx, cy, side, sr, cond, fg, bg)
                        name = f"{shape}_sr{int(sr * 100)}_{pol}_p{pi}_{cond}"
                        img.convert("RGB").save(os.path.join(OUT, "img", name + ".png"))
                        np.savez_compressed(os.path.join(OUT, "img", name + "_aoi.npz"), inside=ins, ind=ind)
                        items.append(dict(name=name, shape=shape, sr=sr, pol=pol, pos=pi, cond=cond))
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    return items

if __name__ == "__main__":
    print(len(make()), "stimuli")
