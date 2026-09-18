"""Render the CC+TD lumen (Mac, OpenGL) from the phase2c integrated-HU mesh npz:

  (a) cc_td_rotate.gif    — smooth orbit of the anatomical structure (master extent): CC green
                            + TD teal, white bg + SSAO.
  (b) cc_td_structure.png — a hero still of the same.
  (c) cc_td_fill_4d.gif   — the per-timepoint lumen (opacified frames only), each from that
                            frame's integrated-HU occupancy surface (occ=0.5 iso, sub-voxel,
                            continuous). CC green / TD teal; the shape & volume change frame to
                            frame as the swine breathes. Fixed camera so the deformation reads.

The cranial TD ends flat because it exits the scan field of view (the contrast-filled duct
continues past the FOV edge), not a segmentation cut.

CC_ZOOM_CM frames the caudal N cm instead of the whole duct, and writes `_zoom` outputs. A
hand-segmented TD runs ~40 cm, and the duct's respiratory bumping is a ~6 mm swing -- 1.5% of a
40 cm frame, i.e. invisible. Over the caudal ~14 cm (the cistern plus the TD leaving it) the same
swing is ~5% and reads, which is the scale the earlier renders happened to be at.

  CC_FIXEDMESH=/tmp/cc_phase2c/fixedmesh_09_07_22_data_Acq16.npz CC_RENDER_OUT=/tmp/cc_pv \
    [CC_ZOOM_CM=14] python3 -m cc_reservoir.scripts.render_cc_td_4d
"""
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

MESH = os.environ["CC_FIXEDMESH"]
OUT = os.environ.get("CC_RENDER_OUT", "/tmp/cc_pv"); os.makedirs(OUT, exist_ok=True)
ZOOM_CM = float(os.environ.get("CC_ZOOM_CM", "0"))          # 0 = frame the whole duct
SFX = "_zoom" if ZOOM_CM > 0 else ""
GREEN, TEAL = "#5aa469", "#3a8f9c"                          # CC green / TD teal (structure hue)
# hue = CC/TD identity; brightness → pale = contrast concentration washing in (separate channel)
CC_CMAP = LinearSegmentedColormap.from_list("cc", ["#2f7d4f", GREEN, "#ecfff2"])
TD_CMAP = LinearSegmentedColormap.from_list("td", ["#246b77", TEAL, "#eaffff"])
pv.global_theme.font.color = "black"

d = np.load(MESH)
opac = [int(x) for x in d["opac"]]
Vcc, Vtd = d["Vcc"], d["Vtd"]
allenh = np.concatenate([d[f"wenh_{i}"] for i in opac])
CLO, CHI = 0.0, float(np.percentile(allenh, 92))             # 0 = no contrast (dark) → pale = peak; shows the wavefront


def poly(v, f):
    return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3, int), f]).ravel())


def setup(pl):
    pl.enable_anti_aliasing("ssaa")
    try:
        pl.enable_ssao(radius=2.0, bias=0.5)
    except Exception as e:
        print("ssao note:", e)


def add_split(pl, mesh, is_cc):
    for sel, col in ((is_cc, GREEN), (~is_cc, TEAL)):
        if sel.any():
            sub = mesh.extract_cells(np.where(sel)[0]).extract_surface()
            pl.add_mesh(sub, color=col, smooth_shading=True, specular=0.45, specular_power=18,
                        ambient=0.32, diffuse=0.72)


def add_split_enh(pl, mesh, is_cc, enh):
    """CC faces green→pale, TD faces teal→pale, coloured by the per-face contrast (enh)."""
    mesh = mesh.copy(); mesh.cell_data["E"] = enh
    for sel, cmap in ((is_cc, CC_CMAP), (~is_cc, TD_CMAP)):
        if sel.any():
            sub = mesh.extract_cells(np.where(sel)[0]).extract_surface().cell_data_to_point_data()
            pl.add_mesh(sub, scalars="E", cmap=cmap, clim=[CLO, CHI], smooth_shading=True,
                        specular=0.4, specular_power=16, ambient=0.30, diffuse=0.74, show_scalar_bar=False)


def fmt(v):
    return f"{v:.0f}" if np.isfinite(v) else "—"


master = poly(d["vn"].astype(float), d["fn"]).smooth_taubin(n_iter=40, pass_band=0.05)
mis_cc = d["is_cc"]

# ---------- (a) structure orbit + (b) hero still (the whole duct; the zoom run skips them) ----------
if not SFX:
    pl = pv.Plotter(off_screen=True, window_size=[920, 1080], border=False)
    pl.background_color = "white"; setup(pl); add_split(pl, master, mis_cc)
    pl.add_text("CC (green, caudal reservoir) + TD (teal, ascending duct)", position="lower_edge",
                color="black", font_size=13)
    pl.camera_position = "yz"; pl.reset_camera()
    pl.screenshot(f"{OUT}/cc_td_structure.png", scale=1)
    path = pl.generate_orbital_path(n_points=48, factor=2.0, viewup=[0, 0, 1], shift=0.0)
    pl.open_gif(f"{OUT}/cc_td_rotate.gif", fps=14)
    pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.0)
    pl.close()
    print("saved cc_td_structure.png + cc_td_rotate.gif")

# ---------- fixed camera from the master extent ----------
ap = pv.Plotter(off_screen=True, window_size=[1000, 1080], border=False)
ap.add_mesh(master); ap.camera_position = "yz"; ap.camera.azimuth = 30; ap.camera.elevation = 8
if ZOOM_CM > 0:
    # the caudal N cm, measured from the duct's caudal tip (the cistern's blind end). Bound x/y to
    # what actually lives in that z band, or the cranial TD's wander would pad the frame back out.
    p = master.points
    z0 = float(p[:, 2].min()); z1 = z0 + ZOOM_CM * 10.0
    q = p[(p[:, 2] >= z0) & (p[:, 2] <= z1)]
    ap.reset_camera(bounds=[q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), z0, z1])
    print(f"zoom: caudal {ZOOM_CM:.0f} cm  z {z0:.0f}..{z1:.0f} mm")
else:
    ap.reset_camera()
CAM = ap.camera_position; ap.close()

# ---------- (c) 4D per-timepoint integrated-HU lumen (opacified frames, deforming) ----------
frames = []
for k, i in enumerate(opac):
    pl = pv.Plotter(off_screen=True, window_size=[1000, 1080], border=False)
    pl.background_color = "white"; setup(pl)
    w = poly(d[f"wv_{i}"].astype(float), d[f"wf_{i}"]).smooth_taubin(n_iter=40, pass_band=0.05)
    add_split_enh(pl, w, d[f"wcc_{i}"], d[f"wenh_{i}"])
    pl.add_text(f"timepoint {i}", position="upper_left", color="black", font_size=26)
    pl.add_text(f"V_CC {fmt(Vcc[k])}   V_TD {fmt(Vtd[k])} mm³", position="lower_left",
                color="#555555", font_size=12)
    pl.camera_position = CAM
    fp = f"{OUT}/_fix{SFX}_{i}.png"; pl.screenshot(fp, scale=1); pl.close()
    frames.append(Image.open(fp).convert("RGB"))
seq = frames + [frames[-1]] * 2
seq[0].save(f"{OUT}/cc_td_fill_4d{SFX}.gif", save_all=True, append_images=seq[1:], duration=900, loop=0)
print(f"saved cc_td_fill_4d{SFX}.gif")
