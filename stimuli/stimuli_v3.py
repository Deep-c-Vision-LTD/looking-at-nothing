"""
stim_v3: the 740 displays of the paper (540 Kanizsa, 200 Ehrenstein) with their AOIs.

Drawing reuses stimuli_v2.py (Kanizsa) and stimuli_ehr.py (Ehrenstein) by import: same canvas (1024 x 768),
supersampling (4x, LANCZOS down), positions, polarity colours, inducer radius rule (r = sr * side / 2),
notch angle (90 deg squares, 60 deg triangles), Ehrenstein line geometry (L = 0.9 R, width 4 px) and outline width
(3 px). The renderers below are the stimuli_v2 / stimuli_ehr renderers with two extra options (a figure fill and an
outline on any inducer arrangement); verify_stim_v3.py checks that they are byte-identical to the original renderers
for every condition the originals can draw.

Kanizsa (60 configurations = square/triangle x sr 0.3/0.5/0.7 x dark/light x 5 positions), 9 conditions:
  illusory     notches face the centroid (aligned)
  control180   every inducer rotated 180 deg (primary control; called "control" in stim_v2)
  control90    every inducer rotated 90 deg
  real         illusory + figure outline (3 px), identical to stim_v2 "real"
  real_filled  illusory + figure surface filled with grey 128 (no outline), drawn over the inducers
  realC        control180 inducers + figure outline (3 px)
  mis10/20/30  every inducer rotated by +10/+20/+30 deg (PIL angle convention, i.e. clockwise on screen) about its
               own centre, all in the same direction
Ehrenstein (40 configurations = n 8/16 x R 70/100 x dark/light x 5 positions), 5 conditions:
  illusory, control_end (primary control; lines rotated 45 deg about their inner ends), real (illusory + circle
  outline, identical to stim_ehr "real"), real_filled (illusory + disc of radius R filled grey 128), realC
  (control_end lines + circle outline).
AOIs (identical across the conditions of a configuration; same definitions as stim_v2 / stim_ehr):
  inside, ind              as in stim_v2 / stim_ehr (taken from the original renderers' AOI output)
  inside_eroded            inside minus every pixel within 8 px of the figure contour (Kanizsa: the polygon
                           boundary; Ehrenstein: the circle of radius R), via the Euclidean distance transform of
                           the figure mask (keep distance > 8)
  inside_eroded_all        extra: inside minus every pixel within 8 px of ANY boundary of the inside region (polygon
                           and inducer discs); identical to inside_eroded for Ehrenstein
Usage: python stimuli/stimuli_v3.py   -> stimuli/stim_v3/{img/*.png, img/*_aoi.npz, items.json} (paths.STIM_V3)
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimuli_v2 as S        # noqa: E402  (first-version Kanizsa renderer)
import stimuli_ehr as E       # noqa: E402  (first-version Ehrenstein renderer)
import paths as LP            # noqa: E402

OUT = str(LP.STIM_V3)
W, H, SS = S.W, S.H, S.SS
assert (W, H, SS) == (E.W, E.H, E.SS) and S.POSITIONS == E.POSITIONS
POSITIONS = S.POSITIONS
POLS = (("dark", (0, 255)), ("light", (255, 0)))                       # (fg, bg), as in stimuli_v2 / stimuli_ehr
GREY = 128                                                              # mid grey between 0 and 255
ERODE_PX = 8

K_SHAPES = (("square", 190), ("triangle", 230))
K_SRS = (0.3, 0.5, 0.7)
# condition -> (inducer rotation in deg, outline, fill)
K_CONDS = {
    "illusory": (0, False, False),
    "control180": (180, False, False),
    "control90": (90, False, False),
    "real": (0, True, False),
    "real_filled": (0, False, True),
    "realC": (180, True, False),
    "mis10": (10, False, False),
    "mis20": (20, False, False),
    "mis30": (30, False, False),
}
K_V2NAME = {"illusory": "illusory", "control180": "control", "control90": "control90", "real": "real"}

E_RS = (70, 100)              # both radii of stimuli_ehr.py
E_NS = (8, 16)                # two of stimuli_ehr.NS = (4, 6, 8, 12, 16, 24)
# condition -> (line arrangement, outline, fill)
E_CONDS = {
    "illusory": ("illusory", False, False),
    "control_end": ("control_end", False, False),
    "real": ("illusory", True, False),
    "real_filled": ("illusory", False, True),
    "realC": ("control_end", True, False),
}
E_ORIG = ("illusory", "control_end", "real")


def kanizsa_hi(shape, cx, cy, side, sr, rot, outline, fill, fg, bg):
    """Supersampled Kanizsa display; inducer code identical to stimuli_v2.figure."""
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    V = S.vertices(shape, cx, cy, side); r = sr * side / 2
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    gap = 90 if shape == "square" else 60
    for x, y in V:
        th = math.degrees(math.atan2(gy - y, gx - x)) + rot
        d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], th + gap / 2, th - gap / 2 + 360, fill=fg)
    if fill:
        d.polygon([(x * SS, y * SS) for x, y in V], fill=GREY)        # occluding grey surface (no outline)
    if outline:
        d.polygon([(x * SS, y * SS) for x, y in V], outline=fg, width=3 * SS)
    return img


def ehrenstein_hi(cx, cy, R, n, lines, outline, fill, fg, bg, width=4):
    """Supersampled Ehrenstein display; line code identical to stimuli_ehr.figure."""
    L = 0.9 * R
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    t0 = math.pi / n
    for k in range(n):
        p, q = E.segment(cx, cy, R, L, t0 + 2 * math.pi * k / n, lines)
        d.line([(p[0] * SS, p[1] * SS), (q[0] * SS, q[1] * SS)], fill=fg, width=width * SS)
    if fill:
        d.ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], fill=GREY)
    if outline:
        d.ellipse([(cx - R) * SS, (cy - R) * SS, (cx + R) * SS, (cy + R) * SS], outline=fg, width=3 * SS)
    return img


def down(img_hi):
    return img_hi.resize((W, H), Image.LANCZOS)


def kanizsa_aoi(shape, cx, cy, side, sr, fg, bg):
    """AOIs exactly as stimuli_v2.figure returns them, plus the eroded variants."""
    _, inside, ind = S.figure(shape, cx, cy, side, sr, "illusory", fg, bg)
    V = S.vertices(shape, cx, cy, side)
    poly = Image.new("L", (W, H), 0); ImageDraw.Draw(poly).polygon(V, fill=1)   # same raster as stimuli_v2
    poly = np.array(poly, bool)
    assert not (inside & ~poly).any()
    ero = inside & (distance_transform_edt(poly) > ERODE_PX)
    ero_all = inside & (distance_transform_edt(inside) > ERODE_PX)
    return dict(inside=inside, ind=ind, inside_eroded=ero, inside_eroded_all=ero_all)


def ehrenstein_aoi(cx, cy, R, n, fg, bg):
    _, inside, ind = E.figure(cx, cy, R, n, "illusory", fg, bg)
    ero = inside & (distance_transform_edt(inside) > ERODE_PX)
    return dict(inside=inside, ind=ind, inside_eroded=ero, inside_eroded_all=ero.copy())


def k_name(shape, sr, pol, pi, cond):
    return f"{shape}_sr{int(sr * 100)}_{pol}_p{pi}_{cond}"


def e_name(R, n, pol, pi, cond):
    return f"ehr_R{R}_n{n}_{pol}_p{pi}_{cond}"


def configs():
    """Yield (illusion, params, name_fn) for every configuration in a fixed order."""
    for shape, side in K_SHAPES:
        for sr in K_SRS:
            for pol, (fg, bg) in POLS:
                for pi, (cx, cy) in enumerate(POSITIONS):
                    yield "kanizsa", dict(shape=shape, side=side, sr=sr, pol=pol, fg=fg, bg=bg, pos=pi, cx=cx, cy=cy)
    for R in E_RS:
        for n in E_NS:
            for pol, (fg, bg) in POLS:
                for pi, (cx, cy) in enumerate(POSITIONS):
                    yield "ehrenstein", dict(R=R, n=n, pol=pol, fg=fg, bg=bg, pos=pi, cx=cx, cy=cy)


def render(ill, p, cond):
    """Supersampled image of one condition of one configuration."""
    if ill == "kanizsa":
        rot, outline, fill = K_CONDS[cond]
        return kanizsa_hi(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], rot, outline, fill, p["fg"], p["bg"])
    lines, outline, fill = E_CONDS[cond]
    return ehrenstein_hi(p["cx"], p["cy"], p["R"], p["n"], lines, outline, fill, p["fg"], p["bg"])


def name_of(ill, p, cond):
    return (k_name(p["shape"], p["sr"], p["pol"], p["pos"], cond) if ill == "kanizsa"
            else e_name(p["R"], p["n"], p["pol"], p["pos"], cond))


def aoi_of(ill, p):
    return (kanizsa_aoi(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], p["fg"], p["bg"]) if ill == "kanizsa"
            else ehrenstein_aoi(p["cx"], p["cy"], p["R"], p["n"], p["fg"], p["bg"]))


def make():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    items = []
    for ill, p in configs():
        conds = K_CONDS if ill == "kanizsa" else E_CONDS
        aoi = aoi_of(ill, p)
        names = {c: name_of(ill, p, c) for c in conds}
        config = names["illusory"][: -len("_illusory")]
        ctrl_names = ({"control180": names["control180"], "control90": names["control90"]} if ill == "kanizsa"
                      else {"control_end": names["control_end"]})
        primary = names["control180"] if ill == "kanizsa" else names["control_end"]
        for cond in conds:
            img = down(render(ill, p, cond))
            img.convert("RGB").save(os.path.join(OUT, "img", names[cond] + ".png"))
            np.savez_compressed(os.path.join(OUT, "img", names[cond] + "_aoi.npz"), **aoi)
            it = dict(name=names[cond], illusion=ill, config=config,
                      shape=p["shape"] if ill == "kanizsa" else "ehrenstein",
                      sr=p["sr"] if ill == "kanizsa" else None,
                      R=p.get("R"), n=p.get("n"), pol=p["pol"], pos=p["pos"], cond=cond,
                      rot_deg=K_CONDS[cond][0] if ill == "kanizsa" else None,
                      primary_control=primary, controls=ctrl_names, siblings=names,
                      aoi=names[cond] + "_aoi.npz")
            items.append(it)
        print(f"{config}: {len(conds)} images", flush=True)
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    return items


if __name__ == "__main__":
    its = make()
    print(len(its), "stimuli;", sum(i["illusion"] == "kanizsa" for i in its), "Kanizsa,",
          sum(i["illusion"] == "ehrenstein" for i in its), "Ehrenstein")
