"""Stage 2 of CC 3D rendering (run on a machine with an OpenGL context, e.g. the Mac):
load the meshes built by compute_cc_meshes.py and produce shaded, white-background,
green ("lymph") renders with depth cues (SSAO + specular) so the CC's tortuous form
and bulbous reservoir end are readable.

Outputs (to CC_RENDER_OUT, default /tmp/cc_pv), named to match assets/realdata/:
  cc_compare_prev_vs_new.png  previous tp1-binary mask vs new integrated-HU surface
  cc_segmentation_3d.png      hero still of the new accurate surface
  cc_model_rotate.gif         orbit of the new accurate surface
  cc_fill_4d.gif              per-timepoint contrast filling, colored by enhancement + colorbar

  CC_MESHES=/tmp/cc_meshes.npz python3 -m cc_reservoir.scripts.render_cc_pyvista
"""
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

MESHES = os.environ.get("CC_MESHES", "/tmp/cc_meshes.npz")
OUT = os.environ.get("CC_RENDER_OUT", "/tmp/cc_pv")
GREEN = "#5aa469"          # lymph convention (matches cc_model_rotate.gif solid color)
FILL_CMAP = LinearSegmentedColormap.from_list("cc_fill", [GREEN, "#ffffe5"])  # 0 = that same green → fills to soft cream (softer than pure white; reads on the gray bg)
FILL_BG = (0.85, 0.85, 0.85)  # light-gray bg for the fill GIF so the pale near-white peak still pops (white bg washed it out)
CLIM = 400.0              # surface enhancement spans ~0-400 HU
VIEW, AZ, EL = "yz", 35, 8   # long (z) axis vertical on screen, slight 3/4 turn

os.makedirs(OUT, exist_ok=True)
pv.global_theme.font.color = "black"
d = np.load(MESHES)
vn, fn, vo, fo = d["vn"], d["fn"], d["vo"], d["fo"]
fill_vals, times = d["fill_vals"], d["times"]
peak_i, peak_enh = int(d["peak_i"]), float(d["peak_enh"])


def mk(verts, faces):
    fa = np.hstack([np.full((len(faces), 1), 3, int), faces]).ravel()
    return pv.PolyData(verts, fa)


new = mk(vn, fn).smooth(n_iter=20, relaxation_factor=0.1)     # soften marching-cubes staircase
old = mk(vo, fo).smooth(n_iter=8, relaxation_factor=0.1)


def style(pl, bg="white"):
    pl.background_color = bg
    pl.enable_anti_aliasing("ssaa")
    try:
        pl.enable_ssao(radius=2.0, bias=0.5)                  # depth via ambient occlusion
    except Exception as e:
        print("ssao note:", e)


def fit(pl, zoom=1.0):
    pl.camera_position = VIEW
    pl.camera.azimuth = AZ; pl.camera.elevation = EL
    pl.reset_camera(); pl.camera.zoom(zoom)


def add_solid(pl, m):
    pl.add_mesh(m, color=GREEN, smooth_shading=True, specular=0.45, specular_power=18,
                ambient=0.32, diffuse=0.70)


# ---------- A. comparison (render each panel solo; subplot+SSAO interact badly), concat with PIL ----------
def panel(mesh, title, fname):
    pl = pv.Plotter(off_screen=True, window_size=[660, 1080], border=False)
    style(pl); add_solid(pl, mesh)
    pl.add_text(title, position="lower_edge", color="black", font_size=14)
    fit(pl, 0.92); pl.screenshot(fname, scale=1); pl.close()

panel(old, "Previous: tp1 binary mask", f"{OUT}/_old.png")
panel(new, "New: integrated-HU accurate", f"{OUT}/_new.png")
a, b = Image.open(f"{OUT}/_old.png"), Image.open(f"{OUT}/_new.png")
canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height) + 46), "white")
canvas.paste(a, (0, 46)); canvas.paste(b, (a.width, 46))
dr = ImageDraw.Draw(canvas)
dr.line([(a.width, 50), (a.width, a.height + 30)], fill=(210, 210, 210), width=2)
dr.text((a.width // 2 - 70, 16), "voxelized, staircased", fill=(90, 90, 90))
dr.text((a.width + b.width // 2 - 130, 16), "sub-voxel half-max surface, TD excluded", fill=(90, 90, 90))
canvas.save(f"{OUT}/cc_compare_prev_vs_new.png")
print("saved cc_compare_prev_vs_new.png")

# ---------- B. orbit GIF ----------
pl = pv.Plotter(off_screen=True, window_size=[920, 1080], border=False)
style(pl); add_solid(pl, new)
pl.add_text("Cisterna chyli — accurate segmentation", position="lower_edge", color="black", font_size=13)
fit(pl, 1.0)
path = pl.generate_orbital_path(n_points=48, factor=2.0, viewup=[0, 0, 1], shift=0.0)
pl.open_gif(f"{OUT}/cc_model_rotate.gif", fps=14)
pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.0)
pl.close()
print("saved cc_model_rotate.gif")

# ---------- C. 4D contrast-fill GIF (one screenshot per timepoint, assembled with PIL) ----------
sb = dict(title="CC enhancement (HU)", color="black", title_font_size=22, label_font_size=17,
          n_labels=5, fmt="%.0f", vertical=True, position_x=0.84, position_y=0.20, height=0.60, width=0.07)
frames = []
for k in range(len(times)):
    pl = pv.Plotter(off_screen=True, window_size=[1000, 1080], border=False)
    style(pl, FILL_BG)
    m = mk(vn, fn); m.cell_data["enh"] = np.clip(fill_vals[k], 0, CLIM)
    m = m.smooth(n_iter=20, relaxation_factor=0.1).cell_data_to_point_data()
    tag = "  (peak)" if k == peak_i else ""
    pl.add_mesh(m, scalars="enh", cmap=FILL_CMAP, clim=[0, CLIM], smooth_shading=True,
                specular=0.4, specular_power=16, ambient=0.32, diffuse=0.72, scalar_bar_args=sb)
    pl.add_text(f"CC contrast filling    timepoint {int(times[k])}{tag}",
                position="lower_edge", color="black", font_size=15)
    fit(pl, 1.0)
    f = f"{OUT}/_fill_{k}.png"; pl.screenshot(f, scale=1); pl.close()
    frames.append(Image.open(f).convert("RGB"))
hold = [1100] * len(frames); hold[peak_i] = 2400      # linear sweep, slightly longer per frame + extra pause on the peak (tp4)
frames[0].save(f"{OUT}/cc_fill_4d.gif", save_all=True, append_images=frames[1:], duration=hold, loop=0)
print("saved cc_fill_4d.gif")

# ---------- D. hero still ----------
pl = pv.Plotter(off_screen=True, window_size=[980, 1120], border=False)
style(pl); add_solid(pl, new); fit(pl, 1.0)
pl.add_text("Cisterna chyli — accurate segmentation", position="lower_edge", color="black", font_size=14)
pl.screenshot(f"{OUT}/cc_segmentation_3d.png", scale=1); pl.close()
print(f"saved cc_segmentation_3d.png | new {len(fn)} faces, old {len(fo)} faces, peak_enh {peak_enh:.0f} HU")
