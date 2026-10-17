"""
stim_alt: alternating-rotation misalignment control for the 60 Kanizsa configurations of stim_v3.

Same shapes (square side 190, triangle side 230), support ratios (0.3, 0.5, 0.7), polarities, positions, canvas,
supersampling and inducer drawing as stimuli_v3.py (imported, not copied: vertices, positions, polarity colours,
radius rule r = sr * side / 2, notch width 90 deg squares / 60 deg triangles, 4x supersampling + LANCZOS down).
The only change is that each inducer k gets its own rotation rot_k = sign_k * theta (PIL angle convention: positive =
clockwise on screen) about its own centre, instead of one rotation shared by all inducers.

Uniform rotation (stim_v3 mis10/20/30): sign = +1 for every inducer.
Alternating rotation (this set), theta = 10, 20, 30 deg, two phases per configuration:
  square   (vertex order TL, TR, BR, BL = clockwise around the figure)
           phase a: (+1, -1, +1, -1)  -> top and bottom sides pinched (both edge tips tilt into the figure),
                                         left and right sides bulge (both tilt out)
           phase b: (-1, +1, -1, +1)  -> top and bottom bulge, left and right pinched
           Neighbours always rotate in opposite directions. Each side's two notch edges are mirror images about that
           side's perpendicular bisector, so collinearity is broken at every side but the two edges stay relatable
           (extensions meet on the bisector with a turn of 2 * theta). Mirror axes kept: the vertical and horizontal
           axes (2 of the illusory square's 4); the diagonal mirrors and the 90 deg rotational symmetry are lost
           (180 deg rotational symmetry is kept). Inner-sector ink is identical to uniform rotation (every inducer
           rotated by |theta|).
  triangle (vertex order apex (top), bottom-right, bottom-left)
           A strict alternation is impossible with 3 inducers (odd cycle: two neighbours must share a direction), and
           with every inducer rotated by |theta| > 0 no mirror symmetry can survive (an inducer on a mirror axis maps
           onto itself with its rotation reversed, so it must have rotation 0). Definition used: the apex inducer,
           which lies on the vertical mirror axis, is not rotated; the two base inducers rotate by the same angle in
           opposite directions (mirror images of each other).
           phase a: (0, +1, -1)  -> base side pinched, lateral sides: base-inducer edge tilts outward
           phase b: (0, -1, +1)  -> base side bulges, lateral sides: base-inducer edge tilts inward
           Kept: the vertical mirror axis (1 of 3); lost: the other two mirrors and the 120 deg rotation.
           Collinearity is broken at all three sides; the base pair is relatable (turn 2 * theta); each lateral side
           has one unrotated and one rotated edge whose extensions meet only at the base inducer's centre, behind the
           rotated edge, so the lateral sides are not relatable (as in uniform rotation, with a smaller offset).
           Inner-sector ink is 2/3 of uniform rotation (the apex inducer contributes none).
Per configuration and theta, the analysis averages phases a and b ("alt<theta>"), so bulge and pinch (and, for the
square, the two orientations of the pattern on the 4:3 canvas) are balanced.

AOIs are those of stim_v3 (identical across conditions); stim_alt/aoi/<config>_aoi.npz holds them plus the core
AOIs (inside pixels with Euclidean distance > 30 or > 45 px from the inducer-disc AOI, analysis/core_aoi.py).
Usage: python stimuli/stimuli_alt.py  -> stimuli/stim_alt/{img/*.png, aoi/*_aoi.npz, items.json} (paths.STIM_ALT)
(items.json keeps the field stim_v3_aoi = "v2/stim_v3/img/<config>_illusory_aoi.npz", the provenance string of the
original run; the same file is stimuli/stim_v3/img/<config>_illusory_aoi.npz here.)
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stimuli_v3 as T            # noqa: E402  (stim_v3 renderer; imports stimuli_v2 as T.S)
S = T.S
W, H, SS = T.W, T.H, T.SS
OUT = str(T.LP.STIM_ALT)
THETAS = (10, 20, 30)
SIGNS = {"square": {"a": (1, -1, 1, -1), "b": (-1, 1, -1, 1)},
         "triangle": {"a": (0, 1, -1), "b": (0, -1, 1)}}
CORE_D = (30, 45)


def gap_of(shape):
    return 90 if shape == "square" else 60


def inward_dirs(shape, cx, cy, side):
    """Vertices and the notch (inward) direction of each inducer, exactly as stimuli_v2.figure computes them."""
    V = S.vertices(shape, cx, cy, side)
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    return V, [math.degrees(math.atan2(gy - y, gx - x)) for x, y in V], (gx, gy)


def kanizsa_hi_rots(shape, cx, cy, side, sr, rots, fg, bg):
    """Supersampled Kanizsa display with one rotation per inducer. With rots = [rot] * n this is line for line the
    inducer code of stimuli_v3.kanizsa_hi (and stimuli_v2.figure) with rotation rot (checked byte-identical by
    verify_stim_alt.py)."""
    img = Image.new("L", (W * SS, H * SS), bg); d = ImageDraw.Draw(img)
    V = S.vertices(shape, cx, cy, side); r = sr * side / 2
    gx, gy = np.mean([v[0] for v in V]), np.mean([v[1] for v in V])
    gap = 90 if shape == "square" else 60
    assert len(rots) == len(V)
    for (x, y), rot in zip(V, rots):
        th = math.degrees(math.atan2(gy - y, gx - x)) + rot
        d.pieslice([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], th + gap / 2, th - gap / 2 + 360, fill=fg)
    return img


def alt_rots(shape, theta, phase):
    return [s * theta for s in SIGNS[shape][phase]]


def configs():
    for ill, p in T.configs():
        if ill == "kanizsa":
            yield p


def config_name(p):
    return T.k_name(p["shape"], p["sr"], p["pol"], p["pos"], "illusory")[: -len("_illusory")]


def aoi_with_core(p):
    a = T.kanizsa_aoi(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], p["fg"], p["bg"])
    dd = ndimage.distance_transform_edt(~a["ind"])
    for D in CORE_D:
        a[f"core{D}"] = a["inside"] & (dd > D)
    return a


def make():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True); os.makedirs(os.path.join(OUT, "aoi"), exist_ok=True)
    items = []
    for p in configs():
        cfg = config_name(p)
        a = aoi_with_core(p)
        np.savez_compressed(os.path.join(OUT, "aoi", cfg + "_aoi.npz"), **a)
        for th in THETAS:
            for ph in ("a", "b"):
                rots = alt_rots(p["shape"], th, ph)
                img = T.down(kanizsa_hi_rots(p["shape"], p["cx"], p["cy"], p["side"], p["sr"], rots, p["fg"], p["bg"]))
                name = f"{cfg}_alt{th}{ph}"
                img.convert("RGB").save(os.path.join(OUT, "img", name + ".png"))
                items.append(dict(name=name, illusion="kanizsa", config=cfg, shape=p["shape"], sr=p["sr"], pol=p["pol"],
                                  pos=p["pos"], cond=f"alt{th}{ph}", alt_theta=th, phase=ph, rots_deg=rots,
                                  aoi=f"aoi/{cfg}_aoi.npz", stim_v3_aoi=f"v2/stim_v3/img/{cfg}_illusory_aoi.npz"))
        print(f"{cfg}: 6 images", flush=True)
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    return items


if __name__ == "__main__":
    its = make()
    print(len(its), "stimuli")
