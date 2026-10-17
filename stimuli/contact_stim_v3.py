"""
Contact sheets for stimuli/stim_v3: one configuration per row, all conditions as columns (each tile is a crop centred on
the figure at half resolution, same scale for every tile), plus a last column showing the illusory image with the AOI
outlines (green = inside, cyan = inducer region, magenta = inside_eroded).
Outputs stim_v3/contact_kanizsa.png and stim_v3/contact_ehrenstein.png.
Optional: python stimuli/contact_stim_v3.py <dir> [rows_per_part] also writes the sheets in row chunks to <dir> for viewing.
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stimuli_v3 as T   # noqa: E402

OUT = T.OUT
GUT, LAB, HEAD = 4, 250, 34
GUTTER_RGB = (150, 110, 190)
FONT = ImageFont.load_default(size=15)


def boundary(m):
    return m & ~binary_erosion(m)


def tile(it, cond, half, dy, scale, aoi=False):
    nm = it["siblings"][cond]
    rgb = np.array(Image.open(os.path.join(OUT, "img", nm + ".png")).convert("RGB"))
    if aoi:
        a = np.load(os.path.join(OUT, "img", it["aoi"]))
        for k, col in (("ind", (0, 200, 255)), ("inside", (0, 200, 0)), ("inside_eroded", (255, 0, 255))):
            b = boundary(a[k])
            b = b | np.roll(b, 1, 0) | np.roll(b, 1, 1)                       # 2 px so it survives the downscale
            rgb[b] = col
    cx, cy = T.POSITIONS[it["pos"]]
    cy = cy + dy
    crop = Image.fromarray(rgb).crop((cx - half, cy - half, cx + half, cy + half))
    return crop.resize((int(2 * half * scale), int(2 * half * scale)), Image.LANCZOS)


def sheet(ill, conds, half, dy_fn, scale=0.5):
    items = json.load(open(os.path.join(OUT, "items.json")))
    rows = [it for it in items if it["illusion"] == ill and it["cond"] == "illusory"]
    cols = list(conds) + ["AOI"]
    t = int(2 * half * scale)
    Wd = LAB + len(cols) * (t + GUT) + GUT
    Hd = HEAD + len(rows) * (t + GUT) + GUT
    S = Image.new("RGB", (Wd, Hd), GUTTER_RGB); d = ImageDraw.Draw(S)
    d.rectangle([0, 0, Wd, HEAD], fill=(255, 255, 255)); d.rectangle([0, 0, LAB, Hd], fill=(255, 255, 255))
    for j, c in enumerate(cols):
        d.text((LAB + GUT + j * (t + GUT) + 6, 9), c, fill=(0, 0, 0), font=FONT)
    for i, it in enumerate(rows):
        y = HEAD + GUT + i * (t + GUT)
        lab = it["config"].split("_")
        d.text((8, y + t // 2 - 18), "_".join(lab[:2]), fill=(0, 0, 0), font=FONT)
        d.text((8, y + t // 2 + 2), "_".join(lab[2:]), fill=(0, 0, 0), font=FONT)
        for j, c in enumerate(cols):
            im = tile(it, "illusory" if c == "AOI" else c, half, dy_fn(it), scale, aoi=(c == "AOI"))
            S.paste(im, (LAB + GUT + j * (t + GUT), y))
    return S, t


def parts(S, t, k, prefix, outdir):
    n = (S.height - HEAD - GUT) // (t + GUT)
    paths = []
    for s in range(0, n, k):
        top = HEAD + GUT + s * (t + GUT); bot = min(S.height, top + k * (t + GUT))
        P = Image.new("RGB", (S.width, HEAD + bot - top), (255, 255, 255))
        P.paste(S.crop((0, 0, S.width, HEAD)), (0, 0)); P.paste(S.crop((0, top, S.width, bot)), (0, HEAD))
        p = os.path.join(outdir, f"{prefix}_rows{s:02d}.png"); P.save(p); paths.append(p)
    return paths


if __name__ == "__main__":
    K, tk = sheet("kanizsa", T.K_CONDS, 225, lambda it: -33 if it["shape"] == "triangle" else 0)
    K.save(os.path.join(OUT, "contact_kanizsa.png"))
    Eh, te = sheet("ehrenstein", T.E_CONDS, 200, lambda it: 0)
    Eh.save(os.path.join(OUT, "contact_ehrenstein.png"))
    print("kanizsa", K.size, "ehrenstein", Eh.size)
    if len(sys.argv) > 1:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        print(parts(K, tk, k, "kan", sys.argv[1])); print(parts(Eh, te, k, "ehr", sys.argv[1]))
