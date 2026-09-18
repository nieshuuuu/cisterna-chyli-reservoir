"""Stage 2 of the PAINTED-mask CC rendering (needs an OpenGL context).

Same visual language as render_cc_pyvista / render_cc_montage — green "#5aa469" lymph on
white, SSAO + specular depth cues, yz camera at azimuth 35 / elevation 8, montage tiles at a
COMMON physical scale so real size differences are visible — plus the interior views:

  cc_painted_montage.png        all 4 CC surfaces, common scale, solid
  <acq>_inside3d.png            solid | semi-transparent + 3 orthogonal cut planes | cutaway
  <acq>_crosssections.png       orthogonal MPRs + a short-axis ladder along the CC long axis
  <acq>_rotate.gif              orbit of the solid surface

Labels carry the PROBE flow only. CT-derived flow is not carried by PAINTED_SET, so the probe
reading is the only flow value that can appear here.

  CC_PAINTED_MESHES=<in.npz> CC_RENDER_OUT=<dir> python -m cc_reservoir.scripts.render_cc_painted
"""
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

MESHES = os.environ.get("CC_PAINTED_MESHES",
                        os.path.join(os.environ.get("TEMP", "/tmp"), "cc_painted_meshes.npz"))
OUT = os.environ.get("CC_RENDER_OUT", os.path.join(os.environ.get("TEMP", "/tmp"), "cc_painted"))

GREEN = "#5aa469"          # lymph convention, shared with render_cc_pyvista
TEAL = "#3a8f9c"           # TD, same as viz.labels label 2 — CC green vs TD teal is the
                           # painter's own colour pairing, so a render reads like the paint UI
TD_BLUE = "#1f6fe0"        # TD contour on the 2D CT panels (dashed; teal is illegible on gray)
# Dense contrast core: soft cream, the same "fills to cream" endpoint render_cc_pyvista uses for
# peak enhancement. Slightly off-white so it still separates from the white page.
CORE_CREAM = "#f6f0d8"
COND_TINT = {"baseline": (225, 236, 250), "angiotensin": (252, 226, 223)}
COND_BAR = {"baseline": (31, 111, 224), "angiotensin": (216, 57, 43)}
VIEW, AZ, EL = "yz", 35, 8
CT_WIN = dict(cmap="gray", vmin=-150, vmax=300)
INSIDE_CMAP = LinearSegmentedColormap.from_list("cc_inside", ["#08103a", "#2f6fb5", "#5aa469",
                                                              "#e8e46a", "#fffdf0"])

os.makedirs(OUT, exist_ok=True)
pv.global_theme.font.color = "black"
D = np.load(MESHES, allow_pickle=True)
N = int(D["n"])


# ---------------------------------------------------------------- shared helpers
def mk(verts, faces):
    fa = np.hstack([np.full((len(faces), 1), 3, int), faces]).ravel()
    return pv.PolyData(verts, fa)


def keep_substantial(mesh, frac=0.20):
    """Keep connected components >= frac of the largest — drops marching-cubes specks but
    preserves a CC that genuinely fragmented into a few pieces. (Same rule as the montage.)"""
    conn = mesh.connectivity("all")
    rid = np.asarray(conn.cell_data["RegionId"]).astype(int)
    counts = np.bincount(rid)
    keep = np.where(counts >= frac * counts.max())[0]
    return conn.extract_cells(np.isin(rid, keep)).extract_surface()


def style(pl, bg="white"):
    pl.background_color = bg
    pl.enable_anti_aliasing("ssaa")
    try:
        pl.enable_ssao(radius=2.0, bias=0.5)
    except Exception as e:
        print("ssao note:", e)


def add_solid(pl, m, opacity=1.0):
    pl.add_mesh(m, color=GREEN, smooth_shading=True, specular=0.45, specular_power=18,
                ambient=0.32, diffuse=0.70, opacity=opacity)


def fit(pl, zoom=1.0):
    pl.camera_position = VIEW
    pl.camera.azimuth = AZ
    pl.camera.elevation = EL
    pl.reset_camera()
    pl.camera.zoom(zoom)


def grid_for(k, field):
    """pv.ImageData over the acquisition's patch, in the SAME mm frame as the mesh (the
    marching-cubes verts are spacing-scaled from the patch origin, so origin is 0).
    Point-centred, so the grid can be contoured and sampled onto cut planes."""
    vox = D[f"voxel_{k}"].astype(float)
    a = np.asarray(D[f"{field}_{k}"], np.float32)
    g = pv.ImageData(dimensions=a.shape, spacing=vox, origin=(0.0, 0.0, 0.0))
    g.point_data[field] = a.ravel(order="F")
    return g


def long_axis(mask, vox):
    """(centroid_mm, unit long axis, unit u, unit v) from the painted CC via PCA."""
    pts = np.argwhere(mask) * np.asarray(vox, float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, tmp); u /= np.linalg.norm(u)
    v = np.cross(d, u); v /= np.linalg.norm(v)
    return c, d, u, v


def sample_plane(field, vox, centre_mm, e1, e2, half_mm, n=140):
    """Bilinear resample of `field` on a square mm-grid plane spanned by e1,e2."""
    s = np.linspace(-half_mm, half_mm, n)
    a, b = np.meshgrid(s, s, indexing="ij")
    pts = centre_mm[None, None, :] + a[..., None] * e1 + b[..., None] * e2
    idx = (pts / np.asarray(vox, float)).reshape(-1, 3).T
    out = map_coordinates(field.astype(np.float32), idx, order=1, mode="nearest")
    return out.reshape(n, n)


# ---------------------------------------------------------------- load models
# THE RENDERED CC IS THE HAND-PAINTED MASK (operator decision, 2026-07-27). The half-max
# enhancement surface is kept, but only as the comparison panel in the seg figure.
#
# Why: measure.surface sets its iso level at 0.5 * the single brightest enhancement voxel.
# On these acquisitions the peak is 1186-1867 HU while the median inside the traced CC is
# 157-301 HU, so that level cuts deep inside the lumen — the half-max surface enclosed only
# 13-19% of the traced volume (88/102/101/65 vs 629/562/796/483 mm3). The tracing is the
# trusted segmentation here, so it is what gets drawn and measured.
def polish(m):
    """Taubin, not Laplacian. `.smooth()` shrinks a closed surface every iteration, which on a
    4 mm duct is a real fraction of the lumen; Taubin's alternating positive/negative pass
    removes the same high-frequency raster ripple with no net shrinkage. Volume is a reported
    quantity here, so the smoother must not quietly move it."""
    return m.smooth_taubin(n_iter=30, pass_band=0.08)


models, traced, halfmax, labels, conds, probes = [], [], [], [], [], []
mesh_vols, trace_vols = [], []
for k in range(N):
    # PRIMARY = the tracing snapped to the local half-max lumen boundary (measure.refine).
    m = polish(mk(D[f"refined_v_{k}"].astype(float), D[f"refined_f_{k}"]))
    models.append(m)
    t = polish(mk(D[f"ccmask_v_{k}"].astype(float), D[f"ccmask_f_{k}"]))
    traced.append(t)
    h = polish(keep_substantial(mk(D[f"verts_{k}"].astype(float), D[f"faces_{k}"])))
    halfmax.append(h)
    labels.append(str(D["labels"][k]))
    conds.append(str(D["conditions"][k]))
    probes.append(float(D["probe_flows"][k]))
    # Volume ENCLOSED BY THE RENDERED SURFACE — a pure geometric property of the object on
    # screen. Deliberately not the integrated-HU volume, whose frame-to-frame spread on this
    # set is as large as the value itself, which would be a misleading label.
    mesh_vols.append(float(m.volume))
    trace_vols.append(float(t.volume))

SCALE = max(float(np.linalg.norm(m.points - m.points.mean(0), axis=1).max()) for m in models) * 1.12


# ---------------------------------------------------------------- A. montage, common scale
TILE = (430, 560)
tiles = []
for k, m in enumerate(models):
    c = m.copy(); c.points = c.points - c.points.mean(0)
    pl = pv.Plotter(off_screen=True, window_size=list(TILE), border=False)
    style(pl); pl.enable_parallel_projection(); add_solid(pl, c)
    pl.camera_position = VIEW; pl.camera.azimuth = AZ; pl.camera.elevation = EL
    pl.camera.parallel_scale = SCALE
    f = f"{OUT}/_tile_{k}.png"; pl.screenshot(f, scale=1); pl.close()
    tiles.append(Image.open(f).convert("RGB"))

cols = min(4, N); rows = (N + cols - 1) // cols
LABH = 62
cell_w, cell_h = TILE[0], TILE[1] + LABH
canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + 44), "white")
dr = ImageDraw.Draw(canvas)
dr.text((16, 12), "Cisterna chyli lumen - common physical scale (tiles are directly "
                  "size-comparable). Green = lymph.", fill=(20, 20, 20))
dr.text((16, 27), "Surface is the hand tracing snapped to 0.25 x the LOCAL peak enhancement "
                  "(boundary-coverage setting, not FWHM). Flow shown is the Transonic probe "
                  "reference standard; CT-derived flow is not used in this figure.",
        fill=(110, 110, 110))
for k in range(N):
    r, c = divmod(k, cols)
    x, y = c * cell_w, 44 + r * cell_h
    dr.rectangle([x, y, x + cell_w, y + cell_h], fill=COND_TINT.get(conds[k], (245, 245, 245)))
    canvas.paste(tiles[k], (x + (cell_w - TILE[0]) // 2, y))
    dr.rectangle([x, y + TILE[1], x + 6, y + cell_h], fill=COND_BAR.get(conds[k], (150, 150, 150)))
    dr.text((x + 14, y + TILE[1] + 6), f"{labels[k]}  ({conds[k]})", fill=(20, 20, 20))
    dr.text((x + 14, y + TILE[1] + 24), f"probe {probes[k]:.3f} mL/min", fill=(60, 60, 60))
    # "mm3" not "mm³": PIL's default bitmap font has no superscript glyph and draws a tofu box.
    dr.text((x + 14, y + TILE[1] + 42),
            f"lumen {mesh_vols[k]:.0f} mm3   (raw tracing was {trace_vols[k]:.0f})",
            fill=(60, 60, 60))
canvas.save(f"{OUT}/cc_painted_montage.png")
print("saved cc_painted_montage.png")


# ---------------------------------------------------------------- B. inside: 3D transparent + cuts
# NB: full-extent orthogonal slice planes are useless here — the patch is much wider than the
# duct, so an opaque plane through the centroid simply hides the CC behind a wall of abdomen.
# The interior is shown instead by (2) a translucent shell over the dense contrast core,
# (3) small short-axis cut planes at stations along the long axis, and (4) a lengthwise cutaway.
WIN = [620, 780]

for k, m in enumerate(models):
    tag = labels[k].replace("/", "_")
    vox = D[f"voxel_{k}"].astype(float)
    mask = D[f"mask_{k}"].astype(bool)
    lvl = 0.5 * float(D[f"peak_enh_{k}"])
    hu_g = grid_for(k, "hu")
    enh_g = grid_for(k, "enh")
    ctr = np.array(m.center)
    c_mm, d_ax, u_ax, v_ax = long_axis(mask, vox)
    panels = []

    def shot(fname, caption, build, zoom=0.95, elev=None, after=None):
        pl = pv.Plotter(off_screen=True, window_size=WIN, border=False)
        style(pl)
        build(pl)
        fit(pl, zoom)
        if elev is not None:                      # extra tilt for panels that need an oblique view
            pl.camera.elevation = elev
            pl.reset_camera(); pl.camera.zoom(zoom)
        if after is not None:                     # runs once the camera is known (view-aligned cuts)
            after(pl)
        pl.add_text(caption, position="lower_edge", color="black", font_size=11)
        pl.screenshot(fname); pl.close()
        panels.append(fname)

    # B1 solid, for reference
    shot(f"{OUT}/_in0_{k}.png", "solid surface", lambda pl: add_solid(pl, m))

    # B2 translucent shell over the dense core (contrast at >=80% of peak enhancement):
    #    where the lumen actually holds concentrated contrast, seen through the wall.
    core = enh_g.contour([1.6 * lvl])                       # 1.6 x half-max = 0.8 x peak
    def b2(pl):
        add_solid(pl, m, opacity=0.22)
        if core.n_points:
            pl.add_mesh(core.smooth(n_iter=12, relaxation_factor=0.1), color=CORE_CREAM,
                        smooth_shading=True, specular=0.5, specular_power=22, ambient=0.35)
    shot(f"{OUT}/_in1_{k}.png", "shell 22% opaque, cream = dense contrast core", b2)

    # B3 translucent shell + short-axis CT cuts at 5 stations along the long axis. Small
    #    (22 mm) squares so they read as cross-sections instead of occluding the duct. Viewed
    #    from a steep elevation: the planes are perpendicular to the (near-vertical) long axis,
    #    so at the default elevation they are edge-on and collapse to slivers.
    pts = np.argwhere(mask) * vox
    t = (pts - c_mm) @ d_ax
    stations = np.linspace(t.min(), t.max(), 5)
    def b3(pl):
        for tj in stations:
            pln = pv.Plane(center=c_mm + tj * d_ax, direction=d_ax, i_size=14, j_size=14,
                           i_resolution=60, j_resolution=60).sample(hu_g)
            pl.add_mesh(pln, scalars="hu", cmap="gray", clim=[-150, 300],
                        show_scalar_bar=False, lighting=False)
        add_solid(pl, m, opacity=0.45)
    shot(f"{OUT}/_in2_{k}.png", "shell 45% opaque + short-axis CT cuts", b3, zoom=0.9, elev=28)

    # B4 LONG-axis cut: one CT plane containing the long axis, so the duct is sectioned end to
    #    end — the complement to B3's short-axis cuts. (A solid "cutaway" was tried here and
    #    dropped: a flat clip plane only stays inside the lumen where the duct happens to run
    #    along it, so on a curved 4 mm tube it mostly shaves the outer wall and shows nothing.)
    length = float(t.max() - t.min())
    def b4(pl):
        pln = pv.Plane(center=c_mm, direction=u_ax, i_size=length + 14, j_size=26,
                       i_resolution=160, j_resolution=90)
        # pv.Plane's i axis is arbitrary; align it to the duct so the section runs lengthwise.
        pln.points = (c_mm + (pln.points - c_mm) @ np.stack([d_ax, v_ax, u_ax]))
        pl.add_mesh(pln.sample(hu_g), scalars="hu", cmap="gray", clim=[-150, 300],
                    show_scalar_bar=False, lighting=False)
        add_solid(pl, m, opacity=0.42)
    shot(f"{OUT}/_in3_{k}.png", "shell 42% opaque + long-axis CT cut", b4, zoom=0.9, elev=-22)

    ims = [Image.open(p).convert("RGB") for p in panels]
    W = sum(i.width for i in ims); H = max(i.height for i in ims)
    out = Image.new("RGB", (W, H + 40), "white")
    d2 = ImageDraw.Draw(out)
    d2.text((14, 13), f"{labels[k]} - CC interior   (probe {probes[k]:.3f} mL/min)   "
                      f"half-max boundary at {lvl:.0f} HU enhancement", fill=(20, 20, 20))
    x = 0
    for i in ims:
        out.paste(i, (x, 40)); x += i.width
    out.save(f"{OUT}/{tag}_inside3d.png")
    print(f"saved {tag}_inside3d.png")


# ---------------------------------------------------------------- B5. painted seg: CC + TD
# What the operator actually traced, both labels together, at three viewing angles — plus the
# same CC as the half-max surface for comparison. Everything else in this script renders the
# ENHANCEMENT-derived surface, which is not the paint; when a render "looks wrong" this figure
# is what tells you whether it is the tracing or the half-max level that is off.
# Tall narrow tiles: the TD runs ~4x the CC's length, so a square tile is mostly white margin.
SEG_WIN = [360, 900]
for k in range(N):
    tag = labels[k].replace("/", "_")
    cc_m = mk(D[f"ccmask_v_{k}"].astype(float), D[f"ccmask_f_{k}"])
    td_v, td_f = D[f"tdmask_v_{k}"].astype(float), D[f"tdmask_f_{k}"]
    td_m = mk(td_v, td_f) if len(td_f) else None
    # Heavier smoothing than the half-max surface needs: a binary mask marching-cubes is a
    # voxel staircase, and the staircase is an artifact of the raster, not of the tracing.
    cc_s = polish(cc_m)
    td_s = polish(td_m) if td_m is not None else None
    half = halfmax[k]
    cc_ctr = np.array(cc_s.center)
    cc_rad = float(np.linalg.norm(cc_s.points - cc_ctr, axis=1).max())
    # reset_camera() fits the bounding SPHERE, which for a 150 mm x 4 mm duct leaves the object
    # a thread in a sea of white. Frame on the actual z-extent instead (the yz camera puts z
    # vertical), under parallel projection so the scale is exact.
    allpts = np.vstack([cc_s.points] + ([td_s.points] if td_s is not None else []))
    z_half = (allpts[:, 2].max() - allpts[:, 2].min()) / 2
    seg_ctr = np.array([allpts[:, 0].mean(), allpts[:, 1].mean(),
                        (allpts[:, 2].max() + allpts[:, 2].min()) / 2])

    def seg_panel(fname, caption, az, cc_mesh=None, td_op=1.0, zoom_cc=False, win=SEG_WIN,
                  ghost_trace=False):
        pl = pv.Plotter(off_screen=True, window_size=list(win), border=False)
        style(pl)
        if ghost_trace:      # raw tracing as a translucent envelope around the refined lumen
            pl.add_mesh(cc_s, color=GREEN, opacity=0.20, smooth_shading=True)
        pl.add_mesh(cc_mesh if cc_mesh is not None else cc_s, color=GREEN, smooth_shading=True,
                    specular=0.42, specular_power=17, ambient=0.33, diffuse=0.70)
        if td_s is not None:
            pl.add_mesh(td_s, color=TEAL, smooth_shading=True, specular=0.42,
                        specular_power=17, ambient=0.33, diffuse=0.70, opacity=td_op)
        pl.camera_position = VIEW
        pl.camera.azimuth = az
        pl.camera.elevation = EL
        pl.reset_camera()
        pl.enable_parallel_projection()
        if zoom_cc:
            # Frame the CC only, so the CC/TD boundary the operator drew is actually legible
            # instead of a few pixels at the bottom of a full-length TD.
            pl.camera.focal_point = cc_ctr
            pl.camera.parallel_scale = cc_rad * 1.35
        else:
            pl.camera.focal_point = seg_ctr
            pl.camera.parallel_scale = z_half * 1.06
        pl.add_text(caption, position="lower_edge", color="black", font_size=11)
        pl.screenshot(fname); pl.close()
        return fname

    ps = [seg_panel(f"{OUT}/_seg0_{k}.png", "raw tracing + TD  (front)", 0),
          seg_panel(f"{OUT}/_seg1_{k}.png", "raw tracing + TD  (35 deg)", 35),
          seg_panel(f"{OUT}/_seg2_{k}.png", "refined lumen inside tracing", 35,
                    cc_mesh=models[k], ghost_trace=True, td_op=0.25),
          seg_panel(f"{OUT}/_seg3_{k}.png", "global half-max (rejected)", 35,
                    cc_mesh=half, td_op=0.25),
          seg_panel(f"{OUT}/_seg4_{k}.png", "zoom: CC/TD boundary", 35, zoom_cc=True,
                    win=[560, 900])]

    ims = [Image.open(p).convert("RGB") for p in ps]
    W = sum(i.width for i in ims); H = max(i.height for i in ims)
    out = Image.new("RGB", (W, H + 60), "white")
    d2 = ImageDraw.Draw(out)
    fin = D[f"refined_levels_{k}"][np.isfinite(D[f"refined_levels_{k}"])]
    d2.text((14, 12), f"{labels[k]} - segmentation: GREEN = CC (label 1), "
                      f"TEAL = TD (label 2)", fill=(20, 20, 20))
    d2.text((14, 30), f"peak frame f{int(D[f'peak_i_{k}'])}   raw tracing "
                      f"{trace_vols[k]:.0f} mm3 -> refined lumen {mesh_vols[k]:.0f} mm3 "
                      f"({100 * mesh_vols[k] / trace_vols[k]:.0f}%, boundary at 0.25 x local "
                      f"peak = {fin.min():.0f}-{fin.max():.0f} HU)   TD "
                      f"{int(D[f'td_vox_{k}'])} vox   probe {probes[k]:.3f} mL/min   - panel 4 "
                      f"is the GLOBAL half-max ({halfmax[k].volume:.0f} mm3), rejected: one "
                      f"level for the whole duct", fill=(110, 110, 110))
    x = 0
    for i in ims:
        out.paste(i, (x, 60)); x += i.width
    out.save(f"{OUT}/{tag}_seg_cc_td.png")
    print(f"saved {tag}_seg_cc_td.png")


# ---------------------------------------------------------------- C. cross-sections
for k in range(N):
    tag = labels[k].replace("/", "_")
    vox = D[f"voxel_{k}"].astype(float)
    hu = D[f"hu_{k}"].astype(np.float32)
    enh = D[f"enh_{k}"].astype(np.float32)
    mask = D[f"mask_{k}"].astype(bool)
    td = D[f"td_{k}"].astype(bool)
    maskf = mask.astype(np.float32)                # raw tracing, drawn as a dashed envelope
    reff = D[f"refined_mask_{k}"].astype(np.float32)   # refined lumen, the solid boundary
    lvl = 0.5 * float(D[f"peak_enh_{k}"])          # global half-max, quoted for reference only

    c_mm, d_ax, u_ax, v_ax = long_axis(mask, vox)
    pts = np.argwhere(mask) * vox
    t = (pts - c_mm) @ d_ax
    NCUT, HALF = 8, 11.0
    ts = np.linspace(t.min(), t.max(), NCUT)

    fig = plt.figure(figsize=(15.5, 8.2), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.0], hspace=0.28)
    gs_top = gs[0].subgridspec(1, 3, wspace=0.08)
    gs_ct = gs[1].subgridspec(1, NCUT, wspace=0.10)
    gs_enh = gs[2].subgridspec(1, NCUT, wspace=0.10)

    # row 0: the three orthogonal MPRs through the CC centroid
    ci = np.round(c_mm / vox).astype(int)
    ci = np.clip(ci, 0, np.array(hu.shape) - 1)
    planes = [("axial  (x-y)", hu[:, :, ci[2]], maskf[:, :, ci[2]], reff[:, :, ci[2]],
               td[:, :, ci[2]]),
              ("coronal  (x-z)", hu[:, ci[1], :], maskf[:, ci[1], :], reff[:, ci[1], :],
               td[:, ci[1], :]),
              ("sagittal  (y-z)", hu[ci[0], :, :], maskf[ci[0], :, :], reff[ci[0], :, :],
               td[ci[0], :, :])]
    for j, (ttl, ct, mp, rp, tdp) in enumerate(planes):
        ax = fig.add_subplot(gs_top[0, j])
        ax.imshow(ct.T, origin="lower", aspect="auto", **CT_WIN)
        if mp.any():
            ax.contour(mp.T, levels=[0.5], colors=["#9dbfa6"], linewidths=1.1,
                       linestyles="dashed")
        if rp.any():
            ax.contour(rp.T, levels=[0.5], colors=[GREEN], linewidths=1.8)
        if tdp.any():
            ax.contour(tdp.T.astype(float), levels=[0.5], colors=[TD_BLUE], linewidths=1.0,
                       linestyles="dashed")
        ax.set_title(ttl, fontsize=9); ax.axis("off")

    # rows 1-2: short-axis cuts perpendicular to the CC long axis, caudal -> cranial.
    # Row 1 is the CT (what it looks like); row 2 is the enhancement (what the surface is cut from).
    for j, tj in enumerate(ts):
        cen = c_mm + tj * d_ax
        ct = sample_plane(hu, vox, cen, u_ax, v_ax, HALF)
        e = sample_plane(enh, vox, cen, u_ax, v_ax, HALF)
        mp = sample_plane(maskf, vox, cen, u_ax, v_ax, HALF)
        rp = sample_plane(reff, vox, cen, u_ax, v_ax, HALF)
        ax = fig.add_subplot(gs_ct[0, j])
        ax.imshow(ct.T, origin="lower", **CT_WIN)
        if mp.max() > 0.5:
            ax.contour(mp.T, levels=[0.5], colors=["#9dbfa6"], linewidths=1.0,
                       linestyles="dashed")
        if rp.max() > 0.5:
            ax.contour(rp.T, levels=[0.5], colors=[GREEN], linewidths=1.3)
        ax.set_title(f"{tj - t.min():.0f} mm", fontsize=7.5, pad=2); ax.axis("off")
        ax2 = fig.add_subplot(gs_enh[0, j])
        ax2.imshow(np.clip(e, 0, None).T, origin="lower", cmap=INSIDE_CMAP, vmin=0, vmax=2 * lvl)
        ax2.axis("off")

    fig.text(0.012, 0.395, "short-axis CT", rotation=90, fontsize=8.5, va="center")
    fig.text(0.012, 0.16, "enhancement", rotation=90, fontsize=8.5, va="center")
    fig.suptitle(f"{labels[k]}  -  CC cross-sections    solid green = refined boundary "
                 f"(0.25 x local peak enhancement), pale dashed = raw hand tracing, "
                 f"blue dash = TD    probe {probes[k]:.3f} mL/min    [true FWHM would sit "
                 f"further in; one global half-max would have been {lvl:.0f} HU]",
                 fontsize=10.5)
    fig.savefig(f"{OUT}/{tag}_crosssections.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {tag}_crosssections.png")


# ---------------------------------------------------------------- C2. same-animal consistency
# All four acquisitions are one scan day = ONE PIG (operator, 2026-07-27). The duct's COURSE and
# LENGTH are therefore fixed anatomy and must agree across acquisitions; only calibre may move,
# through bolus filling / distension. So a large spread in length is a segmentation failure,
# while a spread in cross-sectional area is the physiology this project is about. This figure
# separates the two: if the profiles have the same shape and differ mainly in amplitude, the
# segmentations are mutually consistent and the volume differences are real.
CSA_STATION_MM = 2.0
prof, summ = [], []
for k in range(N):
    vox = D[f"voxel_{k}"].astype(float)
    ref = D[f"refined_mask_{k}"].astype(bool)
    pts = np.argwhere(ref) * vox
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    t = (pts - c) @ d
    t = t - t.min()
    L = float(t.max())
    nb = max(int(np.ceil(L / CSA_STATION_MM)), 1)
    edges = np.linspace(0, L, nb + 1)
    cnt, _ = np.histogram(t, bins=edges)
    csa = cnt * float(np.prod(vox)) / CSA_STATION_MM       # mm^3 per mm = mm^2
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof.append((mid, csa, L))
    summ.append((labels[k], L, mesh_vols[k], float(np.mean(csa)), probes[k]))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=140)
cols = plt.cm.viridis(np.linspace(0.08, 0.82, N))
for k, (mid, csa, L) in enumerate(prof):
    axes[0].plot(mid, csa, color=cols[k], lw=1.7,
                 label=f"{labels[k]}  L={L:.0f} mm  V={mesh_vols[k]:.0f} mm3")
    axes[1].plot(mid / L, csa, color=cols[k], lw=1.7)
axes[0].set_xlabel("distance from caudal end (mm)")
axes[1].set_xlabel("normalised position along the duct  (0 = caudal, 1 = cranial)")
for a in axes:
    a.set_ylabel("cross-sectional area (mm$^2$)")
    a.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=7.5, frameon=False)
Ls = np.array([s[1] for s in summ]); Vs = np.array([s[2] for s in summ])
fig.suptitle("8_31_22 Acq3-6 = one animal, one scan day: course/length is fixed anatomy, "
             "calibre is not\n"
             f"length {Ls.mean():.0f} +/- {Ls.std():.0f} mm (CV {100*Ls.std()/Ls.mean():.0f}%)   |   "
             f"volume {Vs.mean():.0f} +/- {Vs.std():.0f} mm3 (CV {100*Vs.std()/Vs.mean():.0f}%)"
             "   - length CV is the segmentation check, volume CV is the filling signal",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(f"{OUT}/cc_same_animal_consistency.png", facecolor="white")
plt.close(fig)
print("saved cc_same_animal_consistency.png")
print(f"{'acq':<16}{'length mm':>10}{'volume mm3':>12}{'mean CSA mm2':>14}{'probe':>8}")
for lb, L, V, A, p in summ:
    print(f"{lb:<16}{L:>10.1f}{V:>12.0f}{A:>14.2f}{p:>8.3f}")


# ---------------------------------------------------------------- D. orbit GIFs
for k, m in enumerate(models):
    tag = labels[k].replace("/", "_")
    pl = pv.Plotter(off_screen=True, window_size=[760, 900], border=False)
    style(pl); add_solid(pl, m)
    pl.add_text(f"{labels[k]} - cisterna chyli", position="lower_edge", color="black",
                font_size=12)
    fit(pl, 1.0)
    path = pl.generate_orbital_path(n_points=48, factor=2.0, viewup=[0, 0, 1], shift=0.0)
    pl.open_gif(f"{OUT}/{tag}_rotate.gif", fps=14)
    pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.0)
    pl.close()
    print(f"saved {tag}_rotate.gif")

for f in os.listdir(OUT):
    if f.startswith("_"):
        os.remove(os.path.join(OUT, f))
print(f"\nall outputs in {OUT}")
