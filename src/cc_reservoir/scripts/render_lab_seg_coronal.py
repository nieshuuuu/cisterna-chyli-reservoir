"""Overlay the ORIGINAL shipped CC/TD segmentation on the whole-body coronal MIP.

Why this exists: the masks that ship with the data (`SEGMENT_dcm/CC_dcm_01.mat`,
`TD_dcm_01.mat`) are the rough seed the pipeline starts from — this renders them as-is,
on the CT, so their extent can be compared against the anatomy directly.
Both structures are drawn, never unioned, in the repo's seed palette (yellow CC / magenta
TD from viz.labels) — NOT the green/teal locked to our own seg, so provenance stays legible.

THE VIEW IS ipad_paint's DEFAULT (Shu-directed): whole coronal field, no crop, slab-MIP
±8, contrast window, index = median of the duct — i.e. exactly what opens in the iPad
painter. Every one of those is imported or mirrored from viz.ipad_paint rather than
re-derived, so this figure IS the view Shu paints on.

A MIP is the point here, not a lapse: the duct meanders ~30 mm across the coronal axis, so
NO single plane holds its course, and a cross-section can only ever show a fragment of the
seed. The projection is what makes the seed's whole extent — and the duct it misses —
visible in one image. The slab thickness is stated on the figure so the projection is never
mistaken for a cross-section. (Use `--slab 0` for a true single slice.)

    PYTHONPATH=src python3 -m cc_reservoir.scripts.render_lab_seg_coronal \
        <context4d_dir> -o lab_seg_coronal.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from cc_reservoir.viz.ipad_paint import MARKS, WL_PRESETS, _parse_acq, _wl, inplane_mm, to_display
from cc_reservoir.viz.labels import LAB_CC_RGB, LAB_TD_RGB

SAFE_MAX_PX = 1920          # API rejects >2000 px images in multi-image conversations
BLEND = 0.65                # label opacity over the CT, matching viz.mip_panel's overlay
VIEW = "coronal"
DEFAULT_SLAB = 8            # ipad_paint's opening slab (viz/ipad_paint.py: `S={... slab:8 ...}`)
BAR_CHOICES = (5, 10, 20, 50, 100, 200)


def bbox(mask):
    """(lo, hi) voxel index per axis for the True voxels of `mask`; None if empty."""
    nz = np.argwhere(mask)
    if not len(nz):
        return None
    return nz.min(axis=0), nz.max(axis=0)


def default_index(seg, lab_cc, lab_td, axis=0):
    """ipad_paint's opening slice: the median of the duct along the cut axis (its
    Session._load_frame rule, viz/ipad_paint.py:485-488). Uses the frame's own `seg` (our
    pipeline's duct) exactly as the painter does; falls back to the seed if absent."""
    duct = np.argwhere((seg >= 1) & (seg <= 2)) if seg is not None else []
    if not len(duct):
        duct = np.argwhere(lab_cc | lab_td)
    return int(np.median(duct[:, axis]))


def extent_mm(mask, voxel):
    """Physical (x, y, z) extent in mm of a mask's bounding box — the number that makes the
    seed CC's size falsifiable against the anatomy."""
    bb = bbox(mask)
    if bb is None:
        return np.zeros(3)
    lo, hi = bb
    return (hi - lo + 1) * np.asarray(voxel)


def auto_slab(x, lab_cc, lab_td, floor=DEFAULT_SLAB):
    """Smallest slab that still projects the WHOLE seed, never thinner than ipad_paint's ±8.

    The default index is the median of OUR duct, but the seed sits where it sits: in the
    8_31_22 acqs our duct's median lands ~10 voxels posterior of the seed's CC, so a fixed ±8
    silently drops it — Acq4 rendered 0 of 67 CC voxels while the legend still advertised a CC
    patch. A figure of the seed that omits the seed is worse than a thicker slab, and the slab
    is printed on the figure either way. ipad_paint's own slab is a 0-24 slider, so widening
    stays inside the view it opens with."""
    xs = np.argwhere(lab_cc | lab_td)[:, 0]
    return max(int(floor), int(abs(x - xs.min())), int(abs(x - xs.max())))


def scale_bar_mm(panel_w_mm):
    """Largest round bar under ~15% of the panel width — a 20 mm bar is invisible on a
    400 mm whole-body field, and eyeballing the length is how scale bars end up lying."""
    return max([b for b in BAR_CHOICES if b <= 0.15 * panel_w_mm] or [BAR_CHOICES[0]])


def render(ctx_dir, out_png, x=None, slab=None, margin_mm=None, window="contrast"):
    ctx_dir = Path(ctx_dir)
    meta = np.load(ctx_dir / "meta.npz", allow_pickle=True)
    voxel = tuple(float(v) for v in meta["voxel"])
    lab_cc, lab_td = meta["lab_cc"] > 0, meta["lab_td"] > 0
    if not lab_cc.any() and not lab_td.any():
        raise SystemExit(f"[lab_seg] {ctx_dir.name} carries no lab_cc/lab_td seed")

    peak_i = int(meta["peak_i"])
    with np.load(ctx_dir / f"f{peak_i}.npz") as d:
        hu = d["hu"]
        seg = d["seg"] if "seg" in d.files else None
    if hu.shape != lab_cc.shape:
        raise SystemExit(f"[lab_seg] hu {hu.shape} != seed {lab_cc.shape} — different crop")

    x = default_index(seg, lab_cc, lab_td) if x is None else int(x)
    auto = slab is None
    slab = auto_slab(x, lab_cc, lab_td) if auto else int(slab)
    level, win = WL_PRESETS[window]
    # Same slab for CT and seed: projecting the backdrop but not the labels would show the
    # duct's whole course under a fragment of the mask that is supposed to cover it.
    gray = _wl(to_display(hu, VIEW, x, slab).astype(np.float32), level, win)
    d_cc = to_display(lab_cc, VIEW, x, slab)
    d_td = to_display(lab_td, VIEW, x, slab)

    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for m, col in ((d_td, LAB_TD_RGB), (d_cc, LAB_CC_RGB)):    # CC last: it is tiny, never bury it
        if m.any():
            rgb[m] = (1 - BLEND) * rgb[m] + BLEND * col

    nz = hu.shape[2]
    if margin_mm is None:                                      # whole field — ipad_paint's default
        view_rgb = rgb.astype(np.uint8)
    else:
        lo, hi = bbox(lab_cc | lab_td)
        pz, py = int(round(margin_mm / voxel[2])), int(round(margin_mm / voxel[1]))
        z0, z1 = max(0, lo[2] - pz), min(nz - 1, hi[2] + pz)
        y0, y1 = max(0, lo[1] - py), min(hu.shape[1] - 1, hi[1] + py)
        r0, r1 = nz - 1 - z1, nz - 1 - z0                      # to_display flips z -> cranial up
        view_rgb = rgb[r0:r1 + 1, y0:y1 + 1].astype(np.uint8)

    mm_r, mm_c = inplane_mm(VIEW, voxel)                       # (mm/row, mm/col) = (z, y)
    aspect = mm_r / mm_c
    nrow, ncol = view_rgb.shape[:2]
    ph, pw = nrow * mm_r, ncol * mm_c                          # physical size of the panel, mm

    fig_h = 9.5
    fig_w = max(fig_h * pw / ph, 2.6) + 3.1                    # + room for the legend column
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(view_rgb, aspect=aspect, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444")

    top, bot, left, right = MARKS[VIEW]                        # Cr / Cd / R / L, as the iPad shows
    for t, xy, ha, va in ((top, (0.5, 1.0), "center", "bottom"), (bot, (0.5, 0.0), "center", "top"),
                          (left, (0.0, 0.5), "right", "center"), (right, (1.0, 0.5), "left", "center")):
        ax.annotate(t, xy=xy, xycoords="axes fraction", ha=ha, va=va, color="#7cf",
                    fontsize=11, fontweight="bold")

    bar_mm = scale_bar_mm(pw)
    bar_c = bar_mm / mm_c
    ax.add_patch(Rectangle((ncol - bar_c - ncol * 0.03, nrow - nrow * 0.02), bar_c, 2.0 / mm_r,
                           color="w", ec="none"))
    ax.text(ncol - bar_c / 2 - ncol * 0.03, nrow - nrow * 0.025, f"{bar_mm:.0f} mm", color="w",
            ha="center", va="bottom", fontsize=9)

    # What the projection actually carries: seed voxels inside the slab, not the whole mask.
    lo_x, hi_x = (max(0, x - slab), min(hu.shape[0] - 1, x + slab)) if slab > 0 else (x, x)
    in_cc = int(lab_cc[lo_x:hi_x + 1].sum())
    in_td = int(lab_td[lo_x:hi_x + 1].sum())
    # A legend patch with zero pixels behind it is a lie, not a rendering quirk — auto-slab
    # exists to make that impossible, so hold it to that rather than let it regress quietly.
    if auto and (in_cc, in_td) != (int(lab_cc.sum()), int(lab_td.sum())):
        raise SystemExit(f"[lab_seg] auto-slab ±{slab} at x={x} projects only CC {in_cc}/{int(lab_cc.sum())}, "
                         f"TD {in_td}/{int(lab_td.sum())} — the legend would overstate what is drawn")
    e_cc, e_td = extent_mm(lab_cc, voxel), extent_mm(lab_td, voxel)
    handles = [
        Patch(facecolor=LAB_CC_RGB / 255, edgecolor="none",
              label=f"CC — cisterna chyli\nwhole mask: {int(lab_cc.sum())} vox · {e_cc[2]:.1f} mm cranio-caudal\n"
                    f"in this slab: {in_cc} vox"),
        Patch(facecolor=LAB_TD_RGB / 255, edgecolor="none",
              label=f"TD — thoracic duct\nwhole mask: {int(lab_td.sum())} vox · {e_td[2]:.1f} mm cranio-caudal\n"
                    f"in this slab: {in_td} vox"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              labelspacing=1.4, handlelength=1.6, fontsize=9.5,
              title="Original segmentation\n(shipped .mat seed)", title_fontsize=11, alignment="left")

    date, cond, acq = _parse_acq(ctx_dir.name)
    acq_line = " · ".join(v for v in (date, cond, acq) if v)
    sess, sub = str(meta["raw_session"]), str(meta["raw_sub"])
    proj = (f"coronal slab-MIP ±{slab} ({(2 * slab + 1) * voxel[0]:.1f} mm) at x={x}"
            if slab > 0 else f"coronal single slice x={x} (no MIP)")
    ax.set_title(f"Data acquisition: {acq_line}", fontsize=13, fontweight="bold", loc="left", pad=16)
    fig.text(0.012, 0.012,
             f"{sess} · {sub}   |   frame f{peak_i} of {int(meta['n_frames'])} (peak enhancement)   |   "
             f"{proj}   |   {'whole field' if margin_mm is None else f'crop +{margin_mm:.0f} mm'}"
             f" {ph:.0f}×{pw:.0f} mm   |   voxel {voxel[0]:.3f}×{voxel[1]:.3f}×{voxel[2]:.3f} mm   |   "
             f"window {window} (L{level:.0f}/W{win:.0f})",
             fontsize=8, color="#555", ha="left", va="bottom")

    # bbox_inches="tight" grows the canvas by an amount only it knows, so aim, measure, correct
    # against the real file rather than trusting the estimate.
    dpi = min(200, 0.9 * SAFE_MAX_PX / max(fig_w, fig_h))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    im_h, im_w = plt.imread(out_png).shape[:2]
    if max(im_h, im_w) > SAFE_MAX_PX:
        dpi *= SAFE_MAX_PX / max(im_h, im_w)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
        im_h, im_w = plt.imread(out_png).shape[:2]
    plt.close(fig)
    if max(im_h, im_w) > 2000:
        raise SystemExit(f"[lab_seg] {out_png} is {im_w}x{im_h}px — over the 2000px cap")
    print(f"[lab_seg] {ctx_dir.name}  x={x} slab=±{slab}  CC {in_cc}/{int(lab_cc.sum())} vox in slab, "
          f"TD {in_td}/{int(lab_td.sum())}  ->  {out_png} ({im_w}x{im_h}px)")
    return out_png


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ctx_dir", help="context4d_<tag>/ directory (needs meta.npz + f<peak>.npz)")
    p.add_argument("-o", "--out", default="lab_seg_coronal.png")
    p.add_argument("--x", type=int, default=None, help="coronal index (default: ipad_paint's median-of-duct)")
    p.add_argument("--slab", type=int, default=None,
                   help=f"MIP half-thickness; 0 = single slice (default: >={DEFAULT_SLAB}, widened to hold the whole seed)")
    p.add_argument("--margin-mm", type=float, default=None,
                   help="crop to the seed + this margin (default: whole field, as ipad_paint opens)")
    p.add_argument("--window", default="contrast", choices=sorted(WL_PRESETS))
    a = p.parse_args(argv)
    render(a.ctx_dir, a.out, x=a.x, slab=a.slab, margin_mm=a.margin_mm, window=a.window)


if __name__ == "__main__":
    main()
