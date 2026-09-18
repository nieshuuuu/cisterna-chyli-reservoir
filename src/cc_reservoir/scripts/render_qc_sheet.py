"""One-page QC sheet for a hand-drawn CC/TD mask, in the painter's display convention.

Seven views, so a mask can be judged against the CT at a glance:

  full-slab coronal    projection through the whole crop, labels filled: anatomic context
  duct-slab coronal    projection through +-PAD_VOX around the duct's own depth, re-centred
                       on every axial slice so a tortuous duct stays inside it; CT alone and
                       with the mask outlined
  duct-slab sagittal   the same, in the other plane
  three axial cuts     true single slices at the CC and at two TD levels, each with a
                       zoomed copy, mask outlined

The slab panels are projections and can lay structures that are apart in depth on top of
each other. The axial cuts are not projections; they are the depth-exact check.

  python -m cc_reservoir.scripts.render_qc_sheet <context4d_dir> <frame> [--mask <f_edit.npz>] [--out <png>]

<context4d_dir> holds <frame>.npz (key `hu`) and meta.npz (key `voxel`). The mask defaults to
<frame>_edit.npz beside the CT; pass --mask to check a mask kept elsewhere.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cc_reservoir.viz.ipad_paint import AXIS, LABELS, inplane_mm, to_display

CC, TD = 1, 2
PAD_VOX = 7                    # depth added on each side of the mask's own extent in the duct slabs
CROP_MARGIN = (60, 60, 30)     # voxels kept around the mask in x, y, z for anatomic context
ZOOM_HALF = 20                 # half-width, in voxels, of the zoomed axial panels
HU_FLOOR = -150
BG = "#0e1013"
INK = "#c9ccd1"


def load_frame(context_dir, frame, mask_path=None):
    """(hu, seg, voxel_mm) for one frame. Fails if the mask and the CT disagree in shape."""
    with np.load(os.path.join(context_dir, frame + ".npz")) as d:
        hu = d["hu"]
    mask_path = mask_path or os.path.join(context_dir, frame + "_edit.npz")
    with np.load(mask_path) as d:
        seg = d["seg"].astype(np.uint8)
    with np.load(os.path.join(context_dir, "meta.npz"), allow_pickle=True) as m:
        voxel = tuple(float(v) for v in m["voxel"])
    if hu.shape != seg.shape:
        raise ValueError(f"CT {hu.shape} and mask {seg.shape} differ: {mask_path}")
    return hu, seg, voxel


def duct_box(seg, margin):
    """[lo, hi) per axis around every CC|TD voxel, grown by `margin` and clipped to the volume."""
    idx = np.argwhere(np.isin(seg, (CC, TD)))
    if idx.size == 0:
        raise ValueError("mask has no CC or TD voxels")
    lo = np.maximum(idx.min(0) - np.asarray(margin), 0)
    hi = np.minimum(idx.max(0) + 1 + np.asarray(margin), seg.shape)
    return lo, hi


def slab_covering(lo, hi):
    """(centre, half) such that to_display's [centre-half, centre+half] covers [lo, hi)."""
    centre = (lo + hi - 1) // 2
    return int(centre), int(max(centre - lo, hi - 1 - centre))


def duct_slab(vol, duct, view, half):
    """Max projection over +-half voxels around the duct's position on the view's cut axis,
    re-centred on every z. Slices without duct voxels take the interpolated centre. Same
    cranial-up orientation as viz.ipad_paint.to_display."""
    ax = AXIS[view]
    x, y, z = np.nonzero(duct)
    nz = vol.shape[2]
    per_z = np.bincount(z, minlength=nz)
    have = per_z > 0
    zs = np.arange(nz)
    centre = np.rint(np.interp(zs, zs[have],
                               np.bincount(z, weights=(x, y)[ax], minlength=nz)[have] / per_z[have]))
    cut = np.clip(centre.astype(int)[None, :] + np.arange(-half, half + 1)[:, None],
                  0, vol.shape[ax] - 1)                                  # (offset, z)
    other = np.arange(vol.shape[1 - ax])[None, :, None]
    ix = (cut[:, None, :], other, zs) if ax == 0 else (other, cut[:, None, :], zs)
    return vol[ix].max(axis=0).T[::-1]


def axial_levels(seg):
    """Three z indices: the CC's median slice and TD's 1/3 and 2/3 quantiles (TD-only if no CC)."""
    z_cc = np.nonzero(seg == CC)[2]
    z_td = np.nonzero(seg == TD)[2]
    if z_td.size == 0 and z_cc.size == 0:
        raise ValueError("mask has no CC or TD voxels")
    if z_cc.size == 0:
        return [int(np.quantile(z_td, q)) for q in (0.25, 0.5, 0.75)]
    if z_td.size == 0:
        return [int(np.quantile(z_cc, q)) for q in (0.25, 0.5, 0.75)]
    return [int(np.median(z_cc))] + [int(np.quantile(z_td, q)) for q in (1 / 3, 2 / 3)]


def _show(ax, plane, view, voxel, window, title):
    mm_row, mm_col = inplane_mm(view, voxel)
    ax.imshow(plane, cmap="gray", vmin=window[0], vmax=window[1], aspect=mm_row / mm_col,
              interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _outline(ax, masks):
    for lid, m in masks.items():
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[LABELS[lid][1]], linewidths=0.9)


def _fill(ax, lab_plane):
    rgba = np.zeros(lab_plane.shape + (4,))
    for lid, (_name, hexc) in LABELS.items():
        m = lab_plane == lid
        rgba[m, :3] = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgba[m, 3] = 0.6
    ax.imshow(rgba, aspect=ax.get_aspect(), interpolation="nearest")


def render_sheet(hu, seg, voxel, title, out_png):
    lo, hi = duct_box(seg, CROP_MARGIN)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    hu_c, seg_c = hu[box], seg[box]
    ml = float(np.prod(voxel)) / 1000.0
    n_cc, n_td = int((seg == CC).sum()), int((seg == TD).sum())
    duct = np.isin(seg_c, (CC, TD))
    window = (HU_FLOOR, int(round(float(np.percentile(hu_c[duct], 99.5)))))

    full_cor = slab_covering(0, seg_c.shape[0])
    label = {lid: (seg_c == lid).astype(np.uint8) for lid in (CC, TD)}

    def proj(view):
        return (duct_slab(hu_c, duct, view, PAD_VOX),
                {lid: duct_slab(m, duct, view, PAD_VOX) > 0 for lid, m in label.items()})

    fig = plt.figure(figsize=(20, 10), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(3, 7, left=0.01, right=0.99, top=0.9, bottom=0.05,
                          wspace=0.05, hspace=0.18, width_ratios=[1, 1, 1, 1, 1, 1.25, 1.25])
    ax = fig.add_subplot(gs[:, 0])
    _show(ax, to_display(hu_c, "coronal", *full_cor), "coronal", voxel, window,
          "full-slab coronal\n(anatomic context)")
    _fill(ax, to_display(seg_c, "coronal", *full_cor))
    for col, view in enumerate(("coronal", "sagittal")):
        ct, lab_planes = proj(view)
        _show(fig.add_subplot(gs[:, 1 + 2 * col]), ct, view, voxel, window,
              f"duct-slab {view}\nCT only")
        ax = fig.add_subplot(gs[:, 2 + 2 * col])
        _show(ax, ct, view, voxel, window, f"duct-slab {view}\n+ mask outline")
        _outline(ax, lab_planes)

    for row, z in enumerate(axial_levels(seg_c)):
        ct = to_display(hu_c, "axial", z)
        masks = {lid: to_display(m, "axial", z) > 0 for lid, m in label.items()}
        ax = fig.add_subplot(gs[row, 5])
        _show(ax, ct, "axial", voxel, window, f"axial z={z + lo[2]}")
        _outline(ax, masks)
        here = np.argwhere(masks[CC] | masks[TD])
        cx, cy = (here.mean(0) if here.size else np.argwhere(duct[:, :, z] | duct.any(2)).mean(0))
        x0, y0 = int(max(cx - ZOOM_HALF, 0)), int(max(cy - ZOOM_HALF, 0))
        win = (slice(x0, x0 + 2 * ZOOM_HALF), slice(y0, y0 + 2 * ZOOM_HALF))
        ax = fig.add_subplot(gs[row, 6])
        _show(ax, ct[win], "axial", voxel, window, "zoom + outline")
        _outline(ax, {lid: m[win] for lid, m in masks.items()})

    fig.suptitle(
        f"{title}      CC {n_cc:,} vox / {n_cc * ml:.2f} mL    TD {n_td:,} vox / {n_td * ml:.2f} mL\n"
        f"crop x[{lo[0]}:{hi[0]}] y[{lo[1]}:{hi[1]}] z[{lo[2]}:{hi[2]}]    voxel "
        f"{voxel[0]:.3f}×{voxel[1]:.3f}×{voxel[2]:.3f} mm    window [{window[0]}, {window[1]}] HU"
        f"    slab ±{PAD_VOX} vox around the duct",
        color=INK, fontsize=10)
    fig.legend(handles=[Patch(color=LABELS[l][1], label=LABELS[l][0]) for l in (CC, TD)],
               loc="lower left", frameon=False, labelcolor=INK, fontsize=8, ncol=2)
    fig.savefig(out_png, facecolor=BG)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("context_dir")
    ap.add_argument("frame", help="frame stem, e.g. f4")
    ap.add_argument("--mask", help="mask npz (default: <frame>_edit.npz beside the CT)")
    ap.add_argument("--out", help="output PNG (default: qc_<acquisition>_<frame>.png here)")
    a = ap.parse_args()
    acq = os.path.basename(os.path.normpath(a.context_dir)).removeprefix("context4d_")
    hu, seg, voxel = load_frame(a.context_dir, a.frame, a.mask)
    print(render_sheet(hu, seg, voxel, f"{acq}    {a.frame}", a.out or f"qc_{acq}_{a.frame}.png"))


if __name__ == "__main__":
    main()
