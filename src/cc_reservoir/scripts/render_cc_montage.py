"""Stage 2: shaded montage of every clean CC model (run on a GL machine, e.g. the Mac).

All models are drawn at a COMMON physical scale (parallel projection, shared parallel_scale)
so real size differences are visible — auto-fitting each tile would hide exactly the volume
signal Phase-2 is about. Green/white/SSAO as in render_cc_pyvista; labels are tinted by
condition (baseline blue, angiotensin red) and annotated with the integrated-HU volume.

  CC_MONTAGE_MESHES=/tmp/cc_montage.npz python3 -m cc_reservoir.scripts.render_cc_montage
"""
import os

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

MESHES = os.environ.get("CC_MONTAGE_MESHES", "/tmp/cc_montage.npz")
OUT = os.environ.get("CC_RENDER_OUT", "/tmp/cc_pv")
GREEN = "#5aa469"
COND_TINT = {"baseline": (225, 236, 250), "angiotensin": (252, 226, 223)}
COND_BAR = {"baseline": (31, 111, 224), "angiotensin": (216, 57, 43)}
os.makedirs(OUT, exist_ok=True)
pv.global_theme.font.color = "black"

d = np.load(MESHES)
n = int(d["n"])
labels, conditions, volumes = d["labels"], d["conditions"], d["volumes"]
def keep_substantial(mesh, frac=0.20):
    """Keep connected components >= frac of the largest. Drops noise specks the half-max
    surface caught, but (unlike 'largest') preserves a CC that fragmented into a few real
    pieces. Surface-only cleanup; the integrated-HU volume is unaffected."""
    conn = mesh.connectivity("all")
    rid = np.asarray(conn.cell_data["RegionId"]).astype(int)
    counts = np.bincount(rid)
    keep = np.where(counts >= frac * counts.max())[0]
    return conn.extract_cells(np.isin(rid, keep)).extract_surface()


meshes, radii = [], []
for k in range(n):
    v = d[f"verts_{k}"].astype(float)
    faces = d[f"faces_{k}"]
    fa = np.hstack([np.full((len(faces), 1), 3, int), faces]).ravel()
    m = keep_substantial(pv.PolyData(v, fa)).smooth(n_iter=20, relaxation_factor=0.1)
    m.points = m.points - m.points.mean(0)                      # center at origin
    meshes.append(m)
    radii.append(float(np.linalg.norm(m.points, axis=1).max()))
SCALE = max(radii) * 1.12                                        # common scale -> true relative size

TILE = (430, 560)
tiles = []
for k, m in enumerate(meshes):
    pl = pv.Plotter(off_screen=True, window_size=list(TILE), border=False)
    pl.background_color = "white"; pl.enable_parallel_projection(); pl.enable_anti_aliasing("ssaa")
    try:
        pl.enable_ssao(radius=2.0, bias=0.5)
    except Exception as e:
        print("ssao note:", e)
    pl.add_mesh(m, color=GREEN, smooth_shading=True, specular=0.45, specular_power=18,
                ambient=0.32, diffuse=0.70)
    pl.camera_position = "yz"; pl.camera.azimuth = 35; pl.camera.elevation = 8
    pl.camera.parallel_scale = SCALE                            # identical for every tile
    f = f"{OUT}/_montage_{k}.png"; pl.screenshot(f, scale=1); pl.close()
    tiles.append(Image.open(f).convert("RGB"))

cols = 4
rows = (n + cols - 1) // cols
LABH = 46
cell_w, cell_h = TILE[0], TILE[1] + LABH
canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), "white")
dr = ImageDraw.Draw(canvas)
dr.text((16, 14), "Phase-2: all clean CC models (common physical scale) — green = lymph, "
        "labeled with integrated-HU volume", fill=(20, 20, 20))
for k in range(n):
    r, c = divmod(k, cols)
    x, y = c * cell_w, 40 + r * cell_h
    cond = str(conditions[k])
    dr.rectangle([x, y, x + cell_w, y + cell_h], fill=COND_TINT.get(cond, (245, 245, 245)))
    canvas.paste(tiles[k], (x + (cell_w - TILE[0]) // 2, y))
    dr.rectangle([x, y + TILE[1], x + 6, y + cell_h], fill=COND_BAR.get(cond, (150, 150, 150)))
    dr.text((x + 14, y + TILE[1] + 6), f"{labels[k]}  ({cond})", fill=(20, 20, 20))
    dr.text((x + 14, y + TILE[1] + 24), f"V = {volumes[k]:.0f} mm³", fill=(60, 60, 60))
canvas.save(f"{OUT}/cc_models_montage.png")
print(f"saved {OUT}/cc_models_montage.png  ({n} models, common scale {SCALE:.1f} mm)")
