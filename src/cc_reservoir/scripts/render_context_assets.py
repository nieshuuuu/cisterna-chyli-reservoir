"""Per-acq deliverables in the new context-viewer style, from a context4d_<tag>/ directory.

Replaces the old direction-less centerline.png with an oriented 3-plane MPR (+3D), and regenerates
rotate.gif / fill.gif / structure.png / volume.csv / tdc.csv with bone context and the correct
lay-down orientation. The segmentation itself is the integrated-HU pipeline already baked into the
context4d (find-all-contrast extent -> occupancy-FWHM boundary); this script only presents it.

  CC_CTX_DIR=/tmp/context4d_<tag> CC_ASSET_OUT=/tmp/assets_<tag> \
    PYTHONPATH=src python3 -m cc_reservoir.scripts.render_context_assets
"""
import csv
import os

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cc_reservoir.viz.slice_canvas import slice_2d, _AXIS, plane_vox
from cc_reservoir.viz.labels import overlay_rgba, LABEL_RGBA, legend
from cc_reservoir.io.context import load_seg

CTX = os.environ["CC_CTX_DIR"]
OUT = os.environ.get("CC_ASSET_OUT", "/tmp/assets"); os.makedirs(OUT, exist_ok=True)
tag = os.path.basename(CTX.rstrip("/")).replace("context4d_", "")
meta = np.load(os.path.join(CTX, "meta.npz"))
vox = tuple(float(v) for v in meta["voxel"]); peak = int(meta["peak_i"]); nF = int(meta["n_frames"])
frames = [np.load(os.path.join(CTX, f"f{i}.npz")) for i in range(nF)]


def _poly(v, f):
    return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3, int), f]).ravel())


def new_plotter(size):
    """Off-screen plotter with the silky/shiny look: SSAA anti-aliasing + SSAO depth shading."""
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("white")
    for fn in (lambda: p.enable_anti_aliasing("ssaa"), p.enable_ssao):
        try:
            fn()
        except Exception:
            pass
    return p


def _mesh(plotter, fr, opac=1.0):
    """Add the bone(static) + CC/TD(of frame fr) meshes; return duct bounds or None."""
    if meta["bone_v"].size:
        plotter.add_mesh(_poly(meta["bone_v"].astype(float), meta["bone_f"]),
                         color=tuple(LABEL_RGBA[4, :3]), opacity=0.12, smooth_shading=True)
    vs = []
    for tg, lid in (("cc", 1), ("td", 2)):
        v, f = fr[f"{tg}_v"], fr[f"{tg}_f"]
        if len(v):
            # Taubin-smooth the SURFACE, not the mask. The duct is only ~2-3 voxels across in-plane
            # (1.56 mm grid), so blurring the mask enough to round its staircase also shrinks it
            # below the iso-level and breaks the tube. Taubin is volume-preserving: the staircase
            # goes, the calibre stays.
            pd = _poly(v.astype(float), f)
            try:
                pd = pd.smooth_taubin(n_iter=40, pass_band=0.05)
            except Exception:
                pass
            plotter.add_mesh(pd, color=tuple(LABEL_RGBA[lid, :3]), opacity=opac,
                             smooth_shading=True, specular=0.6, specular_power=18, ambient=0.25,
                             diffuse=0.8); vs.append(v)
    return np.vstack(vs) if vs else None


def _duct_of(fr):
    """CC+TD verts of one frame stacked (mm), or None if nothing is segmented in it."""
    vs = [fr[f"{t}_v"] for t in ("cc", "td") if len(fr[f"{t}_v"])]
    return np.vstack(vs) if vs else None


def _aim(plotter, duct, az=20, el=8, margin=15):
    plotter.camera_position = "yz"
    if duct is not None:
        lo, hi = duct.min(0) - margin, duct.max(0) + margin
        plotter.reset_camera(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
    else:
        plotter.reset_camera()
    plotter.camera.azimuth = az; plotter.camera.elevation = el


def crop_to_content(imgs, pad=14, bg=250, min_w=190):
    """Crop renders to the ONE box that holds all their ink.

    A hand-segmented duct is ~40 cm long and ~5 mm wide -- 80:1 -- so it fills a square render's
    height and leaves ~95% of it white. Narrowing the window does not help: VTK's reset_camera
    fits the bounding SPHERE, so a narrow frame just shrinks the duct. Render square (the duct
    lands at full height) and cut the white off afterwards instead.

    One box across every frame, not per-frame: a box that retracked each frame would make the duct
    swim as it fills. min_w keeps room for the caption drawn after the crop.
    """
    arrs = [np.asarray(im.convert("RGB")) for im in imgs]
    ink = np.zeros(arrs[0].shape[:2], bool)
    for a in arrs:
        ink |= (a < bg).any(-1)
    if not ink.any():
        return imgs
    ys, xs = np.where(ink)
    H, W = ink.shape
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, H)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, W)
    if x1 - x0 < min_w:                                   # widen about the duct, stay inside the frame
        cx = (x0 + x1) // 2
        x0, x1 = max(cx - min_w // 2, 0), min(max(cx - min_w // 2, 0) + min_w, W)
        x0 = max(x1 - min_w, 0)
    return [im.crop((x0, y0, x1, y1)) for im in imgs]


def shot3d(fr, px=820):
    p = new_plotter((px, px))                             # square: VTK fits the duct to full height
    duct = _mesh(p, fr); _aim(p, duct)
    img = p.screenshot(return_img=True); p.close()
    return Image.fromarray(img).convert("RGB")


HAS_RAW = "raw_table" in meta.files      # hand-seg bake: the table came from the raw DICOM already


def raw_col(name):
    """A column of the raw V/HU table (raw_duct_table.py), carried through meta by the bake."""
    return meta["raw_table"][:, list(meta["raw_fields"]).index(name)]


# ---- the per-frame duct table ----
if HAS_RAW:
    # ONE table, already measured off the full-res unclipped DICOM under each frame's own hand seg
    # (raw_duct_table.py). Just copy it out: recomputing anything here would mean using this
    # context's `hu`, which is 2x2 mean-pooled and (pre-2026-07-15 builds) clipped at 3071.
    with open(os.path.join(OUT, "duct.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(list(meta["raw_fields"]))
        for r in meta["raw_table"]:
            w.writerow([int(r[0])] + [round(float(x), 1) for x in r[1:]])
    vcc_mean = float(np.mean(raw_col("V_cc_uL")))
    print(f"{tag}: mean V_cc {vcc_mean:.0f} uL (segmented extent, raw grid)  frames {nF}")
else:
    # V/E come straight from the producer's meta.vol_table = per-frame PER-Z ∫occ over the CC/TD master
    # split (frame, V_cc, V_td, E_cc, E_td). ∫occ is the PV-corrected volume; the duct's concentration
    # varies along z, so a single-reference S_O would read it ~1.5-2.5x low.
    rows = [dict(frame=int(r[0]), V_cc=float(r[1]), V_td=float(r[2]), E_cc=float(r[3]),
                 E_td=float(r[4])) for r in meta["vol_table"]]
    # volume.csv = opacified frames only (V is meaningless pre-contrast); tdc.csv = all frames (the
    # pre-contrast E is the bolus-arrival upslope the transport model needs).
    opac = [m for m in rows if m["E_cc"] >= 80.0 and np.isfinite(m["V_cc"])]
    with open(os.path.join(OUT, "volume.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "V_cc", "V_td", "E_cc", "E_td"]); w.writeheader()
        for m in opac:
            w.writerow({k: (round(m[k], 1) if isinstance(m[k], float) else m[k]) for k in w.fieldnames})
    with open(os.path.join(OUT, "tdc.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "E_cc", "E_td"])
        for m in rows:
            w.writerow([m["frame"], round(m["E_cc"], 1), round(m["E_td"], 1)])
    vcc_mean = np.mean([m["V_cc"] for m in opac]) if opac else float("nan")
    print(f"{tag}: mean V_cc {vcc_mean:.0f} uL (paper ~302)  frames {nF}")


# ---- mpr.png: 3 oriented MPR planes (peak frame) + 3D, with bone context ----
def mpr_rgb(hu, seg, view, idx, duct_slab=0, level=80.0, window=900.0):
    """Everything drawn on ONE real CT slice at idx — NO MIP anywhere (HU, duct, AND bone are the
    single slice). A MIP projects the brightest voxel through a slab and blends genuine narrowings /
    defects shut and the meander into a phantom tube; the single slice shows the true cross-section."""
    lo = level - window / 2
    g = (np.clip((slice_2d(hu, view, idx, 0) - lo) / window, 0, 1) * 255).astype(np.uint8)
    duct_seg = seg.copy(); duct_seg[duct_seg > 2] = 0
    bone_seg = seg.copy(); bone_seg[bone_seg != 4] = 0
    ds = seg.shape[_AXIS[view]] if duct_slab is None else duct_slab
    lab = slice_2d(duct_seg, view, idx, ds).astype(np.uint8)
    bone = slice_2d(bone_seg, view, idx, 0).astype(np.uint8)            # single slice, not MIP
    lab[lab == 0] = bone[lab == 0]
    return overlay_rgba(g, lab)[..., :3]


fr = frames[peak]
peak_seg = load_seg(os.path.join(CTX, f"f{peak}.npz"))   # prefer hand-edit sidecar (io.context)
c = (np.asarray(meta["duct_centroid"]) / np.array(vox)).astype(int)
M = 60                                                  # crop window around the duct (keep spine context)
dz = np.where(((peak_seg >= 1) & (peak_seg <= 2)).any(axis=(0, 1)))[0]   # duct z-extent
zlo, zhi = (max(int(dz.min()) - 15, 0), min(int(dz.max()) + 15, peak_seg.shape[2])) if len(dz) \
    else (0, peak_seg.shape[2])


def crop_panel(rgb, view):
    """Zoom to the duct + margin: axial = box around (cx,cy); coronal/sagittal = strip around the
    perpendicular axis, cropped along z to the duct extent so the duct fills the panel."""
    if view == "axial":
        return rgb[max(c[0] - M, 0):c[0] + M, max(c[1] - M, 0):c[1] + M]
    r = c[1] if view == "coronal" else c[0]             # coronal rows=y, sagittal rows=x; cols=z
    return rgb[max(r - M, 0):r + M, zlo:zhi]


fig, ax = plt.subplots(2, 2, figsize=(12, 9))
panels = [("axial", c[2]), ("coronal", c[0]), ("sagittal", c[1])]
leg = legend(set(np.unique(peak_seg)) - {0})            # name only the labels this context actually has
for a, (view, idx) in zip(ax.ravel()[:3], panels):
    r_mm, c_mm = plane_vox(view, vox)                   # z is ~3x finer than in-plane: square
    a.imshow(crop_panel(mpr_rgb(fr["hu"], peak_seg, view, idx), view),   # pixels would stretch the
             origin="upper", aspect=r_mm / c_mm)                         # duct 3x along its length
    a.set_title(f"{view}  ({leg})", fontsize=11); a.axis("off")
ax[1, 1].imshow(crop_to_content([shot3d(fr, px=900)])[0])
ax[1, 1].set_title("3D duct + spine" if meta["bone_v"].size else "3D duct", fontsize=11)
ax[1, 1].axis("off")
fig.suptitle(f"{tag} — context MPR (peak frame {peak})  mean V_CC {vcc_mean:.0f} uL", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mpr.png"), dpi=110); plt.close(fig)

# ---- structure.png: 3D snapshot (peak) ----
crop_to_content([shot3d(frames[peak], px=1600)])[0].save(os.path.join(OUT, "structure.png"))

# fixed camera bounds from the peak-frame duct (so fill.gif doesn't drift as the duct grows)
duct_pk = _duct_of(frames[peak])
GIF_PX = 1100

# ---- rotate.gif: orbit the peak-frame duct + spine (fresh plotter per frame = clean azimuth) ----
rot = []
for az in range(0, 360, 15):
    rp = new_plotter((GIF_PX, GIF_PX))
    _mesh(rp, frames[peak]); _aim(rp, duct_pk, az=20 + az)
    rot.append(Image.fromarray(rp.screenshot(return_img=True)).convert("RGB")); rp.close()
rot = [im.convert("P") for im in crop_to_content(rot)]    # one box over the whole orbit = no swim
rot[0].save(os.path.join(OUT, "rotate.gif"), save_all=True, append_images=rot[1:], duration=90, loop=0)

# ---- fill.gif: the duct filling / deforming across timepoints (fixed camera) ----
# Caption the CISTERN's raw CT number (median: the duct's HU is right-skewed and its rim is
# partial-volume dark). Falls back to the auto-context's mean enhancement, a different quantity.
cap = ([f"CC {v:.0f} HU" for v in raw_col("HU_cc_median")] if HAS_RAW else
       [f"enh {v:.0f} HU" for v in (meta["enh_means"] if "enh_means" in meta.files else [0] * nF)])
fill = []
for i in range(nF):
    p = new_plotter((GIF_PX, GIF_PX))
    _mesh(p, frames[i]); _aim(p, duct_pk, az=20)         # same bounds+angle every frame
    fill.append(Image.fromarray(p.screenshot(return_img=True)).convert("RGB")); p.close()
# caption after the crop: drawn in the render it would be ink, and the crop box would stretch to
# hold it -- pinning the frame to the text width instead of the duct.
fill = crop_to_content(fill)
for i, im in enumerate(fill):
    ImageDraw.Draw(im).text((6, 6), f"frame {i}   {cap[i]}", fill=(0, 0, 0))
fill = [im.convert("P") for im in fill]
fill[0].save(os.path.join(OUT, "fill.gif"), save_all=True, append_images=fill[1:], duration=600, loop=0)
print(f"saved {OUT}/  (mpr.png, structure.png, rotate.gif, fill.gif, "
      f"{'duct.csv' if HAS_RAW else 'volume.csv, tdc.csv'})")
