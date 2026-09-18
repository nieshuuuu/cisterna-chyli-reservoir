"""Browser-based CC/TD seg painter — draw on an iPad with the Apple Pencil.

Why this exists: napari can't run on an iPad, and mouse painting on the Mac is
imprecise. This serves a single-page paint tool over your local WiFi; open it in
Safari on the iPad and draw with the Pencil. Rendering happens server-side (this
Mac), so only small image tiles cross the wire — the 166 MB HU volume never does.
The grayscale backdrop is sent as JPEG (small + fast, so scrubbing slices is
smooth); the seg overlay is lossless PNG so label edges stay crisp.

Orientation: CRANIAL-UP, matching the whole-body MIP viewer (mip_panel.py's
`.T[::-1]  # flip z -> cranial up`) — head at the top, tail at the bottom, NOT
the "lying down" napari layout. Coronal is the default view; sagittal and axial
are one tap away. All views use the standard radiological convention (anatomical
Right on the left of the screen), consistent with the desktop viewer.

Labels (same ids as the rest of the pipeline, io/context + viz/labels):
    1 = CC  (green)      2 = TD  (teal)
    3 = lymph (gold)    4 = bone (tan)
Big labelled buttons pick the active label; CC/TD are the two you paint most.

Backdrop is an adjustable rolling slab-MIP of HU (thick = locate the tortuous
duct as one continuous streak; 0 = a single true CT slice for depth-exact edges).
MIP depth-follow (default "grow") back-projects a stroke onto the duct's TRUE
slice(s) inside the slab instead of the centre slice (same idea as the napari
editor's `b` key). Because that spreads the label over several slices, the overlay
is drawn in TWO layers: THIS slice's labels solid, plus a dim GHOST of the labels
anywhere in the ±slab — so you can see whether the duct is covered across the slab
you are tracing on, without mistaking "somewhere in the slab" for "on this slice".
Slab 0 = pure single-slice truth, no ghost.

Saves to the SAME sidecar the napari editor and every viewer already read:
`<stem>_edit.npz` (key `seg`, in original [x,y,z] orientation). Nothing else in
the pipeline changes — io.context.load_seg picks it up automatically.

Run (on the Mac that has the data):
    PYTHONPATH=src python3 -m cc_reservoir.viz.ipad_paint cc_contexts/context4d_<tag>/
    PYTHONPATH=src python3 -m cc_reservoir.viz.ipad_paint /path/f4.npz
    PYTHONPATH=src python3 -m cc_reservoir.viz.ipad_paint <input> --port 8000
    PYTHONPATH=src python3 -m cc_reservoir.viz.ipad_paint --selftest   # no server, checks the math
Then open the printed http://<mac-lan-ip>:<port>/ URL in Safari on the iPad
(Mac and iPad on the same WiFi). Needs only numpy + Pillow (already installed).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry — cranial-up display convention (matches viz/mip_panel.py)
# Data axes [x,y,z]: x = anterior(0)->posterior, y = left-right,
#                    z = caudal(0)->cranial(max).
#   sagittal (cut y): disp = vol[:,y,:].T[::-1]  rows=z(cranial top), cols=x(A->P)
#   coronal  (cut x): disp = vol[x,:,:].T[::-1]  rows=z(cranial top), cols=y(L->R)
#   axial    (cut z): disp = vol[:,:,z]          rows=x(A top),       cols=y(L->R)
# --------------------------------------------------------------------------- #
AXIS = {"sagittal": 1, "coronal": 0, "axial": 2}
# edge labels shown on the canvas: (top, bottom, left, right)
MARKS = {"sagittal": ("Cr", "Cd", "A", "P"),
         "coronal":  ("Cr", "Cd", "R", "L"),
         "axial":    ("A", "P", "R", "L")}
LABELS = {1: ("CC", "#5aa469"), 2: ("TD", "#3a8f9c"),
          3: ("lymph", "#e0a33a"), 4: ("bone", "#b5916a")}
GHOST_ALPHA = 110         # slab-projected label ("painted somewhere in ±slab"), vs 255 on THIS slice
GATE_RGB = (80, 150, 255)  # "Show gate" wash: the pixels the HU band lets you paint
GATE_ALPHA = 64            # faint enough to read the CT through, strong enough to see the edges


def disp_shape(view, shape):
    nx, ny, nz = shape
    return {"sagittal": (nz, nx), "coronal": (nz, ny), "axial": (nx, ny)}[view]


def to_display(vol, view, idx, slab=0):
    """2D display plane, cranial-up. slab>0 -> max-intensity projection over ±slab."""
    ax = AXIS[view]
    idx = int(np.clip(idx, 0, vol.shape[ax] - 1))
    if slab > 0:
        sl = [slice(None)] * 3
        sl[ax] = slice(max(idx - slab, 0), min(idx + slab + 1, vol.shape[ax]))
        plane = vol[tuple(sl)].max(axis=ax)
    else:
        sl = [slice(None)] * 3
        sl[ax] = idx
        plane = vol[tuple(sl)]
    if view == "axial":
        return plane                      # (x, y) anterior-up
    return plane.T[::-1]                  # (z, in-plane) cranial-up


def disp_to_voxel(view, row, col, idx, shape):
    """Inverse of to_display: a display pixel (row,col) at this view's idx -> (x,y,z)."""
    nx, ny, nz = shape
    if view == "sagittal":                # rows=z flipped, cols=x ; y=idx
        return (int(col), int(idx), int(nz - 1 - row))
    if view == "coronal":                 # rows=z flipped, cols=y ; x=idx
        return (int(idx), int(col), int(nz - 1 - row))
    return (int(row), int(col), int(idx))  # axial: rows=x, cols=y ; z=idx


def voxel_to_disp(view, x, y, z, shape):
    """Forward map (x,y,z) -> (row,col) in this view's display array (for crosshair)."""
    nx, ny, nz = shape
    if view == "sagittal":
        return (int(nz - 1 - z), int(x))
    if view == "coronal":
        return (int(nz - 1 - z), int(y))
    return (int(x), int(y))               # axial


def inplane_mm(view, voxel_mm):
    """(mm_per_row, mm_per_col) for the display plane — used for physical brush + aspect."""
    vx, vy, vz = voxel_mm
    return {"sagittal": (vz, vx), "coronal": (vz, vy), "axial": (vx, vy)}[view]


# --------------------------------------------------------------------------- #
# On-screen 90° rotation (display-only ergonomics — lay the long cranio-caudal
# duct horizontal for comfortable Pencil tracing). rot = # of CCW 90° turns
# (np.rot90 convention). It is PURELY a view transform: rendered tiles are rot90'd
# and pointer strokes are un-rotated back to unrotated display space before they
# touch the paint core — so disp_to_voxel / paint_stroke never see rotation and
# the voxel a stroke lands on is provably rotation-invariant (tested).
# --------------------------------------------------------------------------- #
def rot_dims(view, shape, rot):
    """(rows, cols) of the display AFTER rot90 k=rot (swaps for odd turns)."""
    H0, W0 = disp_shape(view, shape)
    return (W0, H0) if rot % 2 else (H0, W0)


def unrotate_rc(r, c, rot, H0, W0):
    """Map a pixel (r,c) in the rot90'd display back to the UNROTATED (H0×W0) display.
    Affine, so sub-pixel float coords map correctly. Exact inverse of np.rot90(·, rot)."""
    rot %= 4
    if rot == 0:
        return (r, c)
    if rot == 1:                       # inverse of np.rot90 k=1: rotated(i,j)=unrot(j, W0-1-i)
        return (c, W0 - 1 - r)
    if rot == 2:
        return (H0 - 1 - r, W0 - 1 - c)
    return (H0 - 1 - c, r)             # rot == 3


def rot_marks(view, rot):
    """Edge labels (top,bottom,left,right) after rot90 k=rot. One CCW turn sends the
    top edge to the left, right to top, etc. — kept in lockstep with the pixel rotation."""
    t, b, l, r = MARKS[view]
    for _ in range(rot % 4):
        t, b, l, r = r, l, t, b        # CCW 90°: new top<-old right, new left<-old top, ...
    return (t, b, l, r)


# --------------------------------------------------------------------------- #
# The HU gate.
#
# THE BUG THIS FIXES: [floor, ceiling] used to be consulted by exactly one code path —
# paint_stroke_depth's choice of WHICH slice in the slab to snap to. It was a depth-selection
# rule, never a restriction, so:
#   * with depth-follow Off (or slab 0) the band was not read at all and the brush painted every
#     voxel it touched, whatever its HU. Setting a threshold did nothing.
#   * with depth-follow on, ANY in-band voxel anywhere in the +/-slab column was enough to make
#     the stroke paint. On a real thorax a 17-voxel column almost always holds something above
#     120 HU (rib, aorta, contrast), so the "gate" passed nearly everywhere — which is what
#     "even set a threshold, I can still draw freely" was.
#
# Now the band is a HARD PER-VOXEL GATE on every write, in every tool and every depth mode:
# a voxel is painted only if floor <= HU <= ceiling. Erase (label 0) is always gate-free, so
# you can never end up unable to remove a label you can see.
#
# `gate=False` (UI: HU gate Off) opens the band completely — the old free-drawing behaviour,
# now something you choose rather than something you get by accident.
# --------------------------------------------------------------------------- #
def band_bounds(floor, ceiling, gate=True):
    """(lo, hi) HU limits. None on either side = open. gate=False = fully open."""
    if not gate:
        return (-np.inf, np.inf)
    lo = -np.inf if floor is None else float(floor)
    hi = np.inf if ceiling is None else float(ceiling)
    return (lo, hi)


def _band_is_open(lo, hi):
    return lo == -np.inf and hi == np.inf


def disp_to_voxel_arr(view, rows, cols, idx, shape):
    """Vectorised disp_to_voxel: (rows, cols) arrays -> (x, y, z) arrays. Same map, exactly."""
    nx, ny, nz = shape
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    fill = np.full(rows.shape, int(idx), np.int64)
    if view == "sagittal":                 # rows=z flipped, cols=x ; y=idx
        return cols, fill, (nz - 1 - rows)
    if view == "coronal":                  # rows=z flipped, cols=y ; x=idx
        return fill, cols, (nz - 1 - rows)
    return rows, cols, fill                # axial: rows=x, cols=y ; z=idx


def _write_voxels(seg, x, y, z, label, record=None):
    """Set seg[x,y,z]=label wherever that changes something; append (x,y,z,old) to `record`.
    (x,y,z) must be duplicate-free — every caller here derives them from distinct display
    pixels at distinct depths, so they are."""
    if x.size == 0:
        return 0
    old = seg[x, y, z]
    chg = old != label
    if not chg.any():
        return 0
    x, y, z, old = x[chg], y[chg], z[chg], old[chg]
    if record is not None:
        record.extend(zip(x.tolist(), y.tolist(), z.tolist(), old.astype(np.int64).tolist()))
    seg[x, y, z] = label
    return int(x.size)


def _paint_pixels_flat(seg, hu_shape, view, idx, rows, cols, label,
                       hu=None, floor=None, ceiling=None, gate=True, record=None):
    """Write `label` into ONE slice at the given display pixels, gated by the HU band.

    hu=None means no gate is possible (the caller has only a shape), so every touched voxel is
    written — which is what flat painting ALWAYS used to do, and was the bug. Callers that
    want the gate must hand the volume in. label 0 (erase) is never gated."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    if rows.size == 0:
        return 0
    x, y, z = disp_to_voxel_arr(view, rows, cols, idx, hu_shape)
    lo, hi = band_bounds(floor, ceiling, gate)
    if hu is not None and label != 0 and not _band_is_open(lo, hi):
        v = hu[x, y, z].astype(np.float32)
        keep = (v >= lo) & (v <= hi)
        if not keep.all():
            x, y, z = x[keep], y[keep], z[keep]
    return _write_voxels(seg, x, y, z, label, record)


# --------------------------------------------------------------------------- #
# Brush + depth back-projection (pure, tested in _selftest)
# --------------------------------------------------------------------------- #
def _interp(points, max_step=1.0):
    """Densify a display-space polyline so disk stamps leave no gaps."""
    if len(points) <= 1:
        return list(points)
    out = []
    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
        d = max(abs(r1 - r0), abs(c1 - c0))
        n = max(int(d / max_step), 1)
        for k in range(n):
            out.append((r0 + (r1 - r0) * k / n, c0 + (c1 - c0) * k / n))
    out.append(points[-1])
    return out


def _footprint_pixels(view, hu_shape, points_rc, radius_mm, voxel_mm):
    """Set of (row,col) display pixels a physical-mm disk brush covers along a polyline."""
    H, W = disp_shape(view, hu_shape)
    mm_r, mm_c = inplane_mm(view, voxel_mm)
    rr, cc = radius_mm / mm_r, radius_mm / mm_c
    dr = np.arange(-int(np.ceil(rr)), int(np.ceil(rr)) + 1)
    dc = np.arange(-int(np.ceil(cc)), int(np.ceil(cc)) + 1)
    DR, DC = np.meshgrid(dr, dc, indexing="ij")
    foot = ((DR * mm_r) ** 2 + (DC * mm_c) ** 2) <= radius_mm ** 2
    foff, coff = DR[foot], DC[foot]
    seen = set()
    for (r, c) in _interp([(float(a), float(b)) for a, b in points_rc]):
        rs = np.round(r + foff).astype(int)
        csr = np.round(c + coff).astype(int)
        m = (rs >= 0) & (rs < H) & (csr >= 0) & (csr < W)
        for rp, cp in zip(rs[m], csr[m]):
            seen.add((int(rp), int(cp)))
    return seen


def _pixels_array(pixels):
    """A {(row,col)} set -> two int arrays, in a deterministic order."""
    if not pixels:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.array(sorted(pixels), np.int64)
    return a[:, 0], a[:, 1]


def paint_stroke(seg, hu_shape, view, idx, points_rc, radius_mm, voxel_mm, label,
                 record=None, hu=None, floor=None, ceiling=None, gate=True):
    """Stamp a physical-mm disk brush along a display-space polyline onto ONE slice.

    Mutates seg in place ([x,y,z]). label 0 erases. Returns #voxels changed.
    The footprint is a true circle in millimetres, so it is not distorted by the
    anisotropic voxel (z is 0.5 mm, x/y 1.56 mm here). If `record` is a list, each
    change is appended as (x, y, z, old_label) so the caller can undo it.

    Pass `hu` (the HU volume) to enforce the [floor, ceiling] gate: only voxels inside the
    band are painted. Erase (label 0) is never gated."""
    rows, cols = _pixels_array(_footprint_pixels(view, hu_shape, points_rc, radius_mm, voxel_mm))
    return _paint_pixels_flat(seg, hu_shape, view, idx, rows, cols, label,
                              hu=hu, floor=floor, ceiling=ceiling, gate=gate, record=record)


DEPTH_CHUNK = 65_536      # display pixels per pass; see _paint_pixels_depth


def _paint_pixels_depth(hu, seg, view, idx, rows, cols, slab, label,
                        floor=120.0, ceiling=None, gate=True, frac=0.4, maxrun=6,
                        grow=True, record=None):
    """Back-project a set of display pixels through the ±slab and paint the duct's TRUE depth.

    Vectorised over the pixels (a lasso hands this tens of thousands at a time, which the
    old per-pixel Python loop could not carry). Semantics are unchanged from the per-pixel
    version and are pinned by the depth tests:

      * pick, in each pixel's depth column, the BRIGHTEST voxel whose HU is inside
        [floor, ceiling]; a column with no in-band voxel is skipped entirely;
      * `grow` additionally labels the contiguous in-band run around that peak, out to
        `maxrun` slices each way, down to thr = p20 + frac*(peak - p20), never below floor;
      * label 0 (erase) ignores the band completely and snaps to the brightest voxel.

    Every voxel written therefore satisfies the band — the gate is on the WRITE, not merely
    on the choice of depth."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    if rows.size == 0:
        return 0
    a = AXIS[view]
    others = [k for k in range(3) if k != a]
    lo_d, hi_d = max(0, idx - slab), min(hu.shape[a], idx + slab + 1)
    if hi_d <= lo_d:
        return 0
    # A whole-field lasso is ~360k pixels; times a 49-deep slab that is a 70 MB float32 column
    # block per temporary, and np.percentile sorts it. Chunk the pixels so peak memory is bounded
    # no matter how big the gesture -- every column is independent, so this changes nothing.
    if rows.size > DEPTH_CHUNK:
        n = 0
        for k in range(0, rows.size, DEPTH_CHUNK):
            n += _paint_pixels_depth(hu, seg, view, idx, rows[k:k + DEPTH_CHUNK],
                                     cols[k:k + DEPTH_CHUNK], slab, label, floor=floor,
                                     ceiling=ceiling, gate=gate, frac=frac, maxrun=maxrun,
                                     grow=grow, record=record)
        return n
    vx = disp_to_voxel_arr(view, rows, cols, idx, hu.shape)
    o0, o1 = vx[others[0]], vx[others[1]]
    hum = np.moveaxis(hu, a, 0)                       # (depth, o0, o1) view
    cols_hu = hum[lo_d:hi_d, o0, o1].astype(np.float32)          # (D, N)
    D = cols_hu.shape[0]
    lo_b, hi_b = band_bounds(floor, ceiling, gate)

    if label != 0:
        inband = (cols_hu >= lo_b) & (cols_hu <= hi_b)
        keep = inband.any(axis=0)
        if not keep.any():                            # no in-band (duct) voxel anywhere -> skip
            return 0
        if not keep.all():
            cols_hu = cols_hu[:, keep]
            inband = inband[:, keep]
            o0, o1 = o0[keep], o1[keep]
        pk = np.argmax(np.where(inband, cols_hu, -np.inf), axis=0)
    else:
        pk = np.argmax(cols_hu, axis=0)               # erase snaps to the brightest, gate-free

    N = cols_hu.shape[1]
    ar = np.arange(N)
    run = np.zeros((D, N), bool)
    run[pk, ar] = True
    if grow:
        base = np.percentile(cols_hu, 20, axis=0)     # local background level in each column
        peak = cols_hu[pk, ar]
        thr = base + frac * (peak - base)
        ok = cols_hu >= thr[None, :]
        if label != 0:
            thr = np.maximum(thr, lo_b)               # stay within the duct band while growing
            ok = (cols_hu >= thr[None, :]) & (cols_hu <= hi_b)
        for sign in (-1, 1):                          # walk outward, contiguous, bounded by maxrun
            alive = np.ones(N, bool)
            for step in range(1, maxrun + 1):
                d = pk + sign * step
                valid = alive & (d >= 0) & (d < D)
                dc = np.clip(d, 0, D - 1)
                good = valid & ok[dc, ar]
                if not good.any():
                    break
                run[dc[good], ar[good]] = True
                alive = good

    dd, jj = np.nonzero(run)
    if dd.size == 0:
        return 0
    coord = [None, None, None]
    coord[a] = lo_d + dd
    coord[others[0]] = o0[jj]
    coord[others[1]] = o1[jj]
    return _write_voxels(seg, coord[0], coord[1], coord[2], label, record)


def paint_stroke_depth(hu, seg, view, idx, points_rc, radius_mm, voxel_mm, slab, label,
                       floor=120.0, ceiling=None, gate=True, frac=0.4, maxrun=6,
                       grow=True, record=None):
    """MIP-aware brush: draw ON the slab-MIP, land on the TRUE slice(s).

    The problem this solves: you can only SEE the whole tortuous duct on a thick slab-MIP,
    but a flat stroke lands on one slice while the duct actually sits at DIFFERENT depths
    along its length (median ~2 slices, up to 5-7). This back-projects every footprint
    pixel through the ±slab window along the cut axis and snaps to the duct's true depth.

    Target selection uses an HU BAND [floor, ceiling]:
      * pick the brightest voxel in the column WHOSE HU is within [floor, ceiling]; if none
        qualifies, paint nothing here. This is the "比肌肉/脂肪高一点、但别抓到别的亮结构" gate.
      * POST-contrast (peak frames): the duct is bright (~120-500+ HU) and muscle ~45-100,
        so floor=120, ceiling=None (open top) cleanly isolates it.
      * PRE-contrast (early frames, e.g. f0): the chyle-filled duct is near WATER (~0-40 HU),
        DIMMER than the muscle/aorta beside it (~60-100 HU). A floor alone can't help — snap
        would grab the muscle. Set a low floor AND a ceiling BELOW muscle (e.g. band [-20,45])
        so the brightest IN-BAND voxel is the water-density duct, not the brighter muscle.

    grow: also label the CONTIGUOUS in-band run around the peak (bounded to `maxrun` slices
    each way, clamped in-bounds), so a duct several slices thick gets all of them; if not
    grow, label only the single peak slice ("snap to the correct slice").

    label=0 erases: removes any existing label along the same run with NO band gate, so you
    can always erase. Mutates seg in place; `record` collects (x,y,z,old) for undo.
    Returns #voxels changed."""
    rows, cols = _pixels_array(_footprint_pixels(view, hu.shape, points_rc, radius_mm, voxel_mm))
    return _paint_pixels_depth(hu, seg, view, idx, rows, cols, slab, label,
                               floor=floor, ceiling=ceiling, gate=gate, frac=frac,
                               maxrun=maxrun, grow=grow, record=record)


# --------------------------------------------------------------------------- #
# Lasso: enclose a region, let the HU gate pick the duct out of it.
#
# This is what a working threshold is FOR. Tracing a tortuous duct voxel by voxel is the slow
# way; circling it loosely and keeping only the in-band voxels inside the loop is the fast
# one, and it is exactly as accurate as the band is — which you can now see, because the
# gate is previewable on the overlay.
# --------------------------------------------------------------------------- #
def polygon_pixels(poly_rc, H, W):
    """Display pixels enclosed by the closed polyline `poly_rc`, plus the outline itself.

    Even-odd scanline fill, clipped to the HxW display. The outline is always included so a
    degenerate lasso (a scribble, two points, a single tap) still marks what it touched rather
    than silently doing nothing. Returns (rows, cols) int arrays."""
    pts = [(float(r), float(c)) for r, c in poly_rc]
    if not pts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    mask = np.zeros((H, W), bool)

    # ---- outline (and the whole answer when there are fewer than 3 points) ----
    loop = pts + [pts[0]] if len(pts) > 1 else pts
    dense = np.array(_interp(loop), np.float64)
    rr = np.rint(dense[:, 0]).astype(np.int64)
    cc = np.rint(dense[:, 1]).astype(np.int64)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    mask[rr[ok], cc[ok]] = True
    if len(pts) < 3:
        r_i, c_i = np.nonzero(mask)
        return r_i.astype(np.int64), c_i.astype(np.int64)

    # ---- interior: one scanline per row, every edge tested at once ----
    # Per-row rather than one big (rows x cols) broadcast: a Pencil lasso arrives with hundreds
    # of coalesced points, and a full-field boolean per edge is seconds of work.
    P = np.array(pts, np.float64)
    ay, ax = P[:, 0], P[:, 1]
    by, bx = np.roll(ay, -1), np.roll(ax, -1)
    live = ay != by                              # horizontal edges are never crossed
    ay, ax, by, bx = ay[live], ax[live], by[live], bx[live]
    if ay.size:
        r0 = max(0, int(np.floor(P[:, 0].min())))
        r1 = min(H - 1, int(np.ceil(P[:, 0].max())))
        for r in range(r0, r1 + 1):
            crossed = (ay > r) != (by > r)
            if not crossed.any():
                continue
            xs = np.sort(ax[crossed] + (r - ay[crossed]) * (bx[crossed] - ax[crossed])
                         / (by[crossed] - ay[crossed]))
            for k in range(0, xs.size - 1, 2):
                c_lo = max(0, int(np.ceil(xs[k])))
                c_hi = min(W - 1, int(np.floor(xs[k + 1])))
                if c_hi >= c_lo:
                    mask[r, c_lo:c_hi + 1] = True
    r_i, c_i = np.nonzero(mask)
    return r_i.astype(np.int64), c_i.astype(np.int64)


def lasso_fill(hu, seg, view, idx, poly_rc, voxel_mm, slab, label,
               depth="off", floor=None, ceiling=None, gate=True, record=None):
    """Fill everything the lasso encloses, subject to the same HU gate as the brush.

    depth 'snap'/'grow' back-project the enclosed pixels through the slab exactly as the
    brush does, so a lasso drawn on a thick slab-MIP lands on the duct's real slices.
    label 0 erases everything enclosed, gate-free (the cleanup tool)."""
    H, W = disp_shape(view, hu.shape)
    rows, cols = polygon_pixels(poly_rc, H, W)
    if rows.size == 0:
        return 0
    if depth in ("snap", "grow") and slab > 0:
        return _paint_pixels_depth(hu, seg, view, idx, rows, cols, slab, label,
                                   floor=floor, ceiling=ceiling, gate=gate,
                                   grow=(depth == "grow"), record=record)
    return _paint_pixels_flat(seg, hu.shape, view, idx, rows, cols, label,
                              hu=hu, floor=floor, ceiling=ceiling, gate=gate, record=record)


# --------------------------------------------------------------------------- #
# Wand: one tap -> the connected in-band blob around it.
# --------------------------------------------------------------------------- #
def _flood_numpy(band, seed):
    """6-connected flood fill of `band` from `seed`, numpy only (no scipy).

    This is what runs on the painter's stated dependency set (numpy + Pillow), so it is the
    reference implementation and _selftest checks scipy against IT, not the other way round."""
    cur = np.zeros_like(band)
    cur[seed] = True
    while True:
        nxt = cur.copy()
        nxt[1:, :, :] |= cur[:-1, :, :]
        nxt[:-1, :, :] |= cur[1:, :, :]
        nxt[:, 1:, :] |= cur[:, :-1, :]
        nxt[:, :-1, :] |= cur[:, 1:, :]
        nxt[:, :, 1:] |= cur[:, :, :-1]
        nxt[:, :, :-1] |= cur[:, :, 1:]
        nxt &= band
        if nxt.sum() == cur.sum():
            return cur
        cur = nxt


def _connected_from(band, seed):
    """The 6-connected component of `band` containing `seed` (an (i,j,k) index tuple).

    scipy is used when importable because it is far faster on a big box; without it the numpy
    flood fill gives the identical answer, just slower."""
    if not band[seed]:
        return np.zeros_like(band)
    try:
        from scipy import ndimage                      # optional; identical result
        lab, _n = ndimage.label(band)
        return lab == lab[seed]
    except Exception:
        return _flood_numpy(band, seed)


def wand_fill(hu, seg, view, idx, row, col, voxel_mm, label, slab=0,
              floor=None, ceiling=None, gate=True, radius_mm=12.0,
              max_vox=150_000, record=None):
    """One tap -> label the whole connected in-band structure around it, within radius_mm.

    This is the auto-correction: where the threshold DOES separate the duct from its
    neighbours, one tap replaces a whole traced stroke. It is deliberately bounded by a
    physical ball (radius_mm) and a voxel budget, because on a badly-set band the connected
    component is the entire skeleton and an unbounded fill looks unrecoverable.

    The seed is the brightest in-band voxel in the tapped pixel's ±slab column (the rule the
    depth brush uses), so tapping on a slab-MIP works without first finding the exact slice.
    label 0 erases the connected LABELLED blob instead (band ignored).
    Returns (#voxels changed, note); note is '' on success, else why nothing happened."""
    lo, hi = band_bounds(floor, ceiling, gate)
    H, W = disp_shape(view, hu.shape)
    r = int(np.clip(round(float(row)), 0, H - 1))         # a tap can land just off the canvas
    c = int(np.clip(round(float(col)), 0, W - 1))
    seed = list(disp_to_voxel(view, r, c, int(idx), hu.shape))
    a = AXIS[view]

    # ---- pick the seed's depth inside the ±slab column ----
    lo_d, hi_d = max(0, idx - slab), min(hu.shape[a], idx + slab + 1)
    sl = [slice(None)] * 3
    for k in range(3):
        sl[k] = slice(lo_d, hi_d) if k == a else seed[k]
    colvals = hu[tuple(sl)].astype(np.float32)
    if colvals.size == 0:
        return 0, "outside the volume"
    if label != 0:
        inb = (colvals >= lo) & (colvals <= hi)
        if not inb.any():
            return 0, "nothing inside the HU band under the tap"
        seed[a] = lo_d + int(np.argmax(np.where(inb, colvals, -np.inf)))
    else:
        seed[a] = lo_d + int(np.argmax(colvals))

    # ---- a physical ball around the seed, so one tap can never run away ----
    vx = [float(v) for v in voxel_mm]
    half = [max(1, int(np.ceil(radius_mm / v))) for v in vx]
    b0 = [max(0, seed[k] - half[k]) for k in range(3)]
    b1 = [min(hu.shape[k], seed[k] + half[k] + 1) for k in range(3)]
    sub = hu[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]]
    grids = np.meshgrid(*[(np.arange(b0[k], b1[k]) - seed[k]) * vx[k] for k in range(3)],
                        indexing="ij")
    ball = (grids[0] ** 2 + grids[1] ** 2 + grids[2] ** 2) <= radius_mm ** 2

    if label != 0:
        band = (sub >= lo) & (sub <= hi) & ball
    else:                                              # erase: the connected LABELLED blob
        segsub = seg[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]]
        band = (segsub > 0) & ball
    comp = _connected_from(band, tuple(seed[k] - b0[k] for k in range(3)))
    n = int(comp.sum())
    if n == 0:
        return 0, "the tapped voxel is outside the HU band"
    if n > max_vox:
        return 0, f"that would fill {n} voxels — tighten the band or shorten the reach"
    ii, jj, kk = np.nonzero(comp)
    return _write_voxels(seg, ii + b0[0], jj + b0[1], kk + b0[2], label, record), ""


def backproject(hu, seg, view, idx, slab, label, record=None):
    """Snap `label` voxels on the current slice to their brightest slice within ±slab
    along the cut axis (recovers depth after thick-slab painting). Mutates seg.
    If `record` is a list, appends (x,y,z,old) for every voxel touched (undo)."""
    if slab <= 0:
        return 0
    a = AXIS[view]
    n = seg.shape[a]
    lo, hi = max(0, idx - slab), min(n, idx + slab + 1)
    hu_m = np.moveaxis(hu, a, 0)          # (n_a, o0, o1) view
    seg_m = np.moveaxis(seg, a, 0)        # a view -> writes propagate to seg
    mask2d = (seg_m[idx] == label)
    if not mask2d.any():
        return 0
    ii, jj = np.nonzero(mask2d)
    window = hu_m[lo:hi][:, ii, jj]       # (win, N) — uniform for every view
    best = lo + np.argmax(window, axis=0)

    def _xyz(a_i, o0, o1):
        v = [0, 0, 0]
        v[a] = a_i
        others = [k for k in range(3) if k != a]
        v[others[0]] = o0
        v[others[1]] = o1
        return tuple(int(t) for t in v)

    moved = 0
    for k in range(len(ii)):
        if best[k] == idx:
            continue
        if record is not None:
            src = _xyz(idx, ii[k], jj[k])
            dst = _xyz(int(best[k]), ii[k], jj[k])
            record.append((*src, int(seg[src])))
            record.append((*dst, int(seg[dst])))
        seg_m[idx, ii[k], jj[k]] = 0
        seg_m[int(best[k]), ii[k], jj[k]] = label
        moved += 1
    return moved


# --------------------------------------------------------------------------- #
# Volume + seg loading (mirrors io/context.load_seg sidecar behaviour)
# --------------------------------------------------------------------------- #
def seed_stamp(seg, seed, seed_ok=True):
    """Provenance keys recording whether `seg` differs from the producer's auto-seed `seed`.

    n_diff_seed = number of voxels that differ, or:
      -1  the frame genuinely has no seed, so everything here came from a human (edited=True)
      -2  the seed could not be read, so provenance is UNKNOWN (edited=False, i.e. untrusted)

    The two sentinels must stay distinct. Mapping an unreadable seed to -1 would let an I/O error
    certify a mask as hand work, which is the exact failure this stamp exists to prevent."""
    if not seed_ok:
        return {"edited": np.bool_(False), "n_diff_seed": np.int64(-2), "n_seed": np.int64(-1)}
    if seed is None or seed.shape != seg.shape:
        return {"edited": np.bool_(True), "n_diff_seed": np.int64(-1), "n_seed": np.int64(-1)}
    n_diff = int((seg != seed).sum())
    return {"edited": np.bool_(bool(n_diff)),
            "n_diff_seed": np.int64(n_diff),
            "n_seed": np.int64(int((seed > 0).sum()))}


def _load_seg_sidecar(npz_path):
    """seg (uint8), preferring the `<stem>_edit.npz` sidecar if it exists."""
    side = os.path.splitext(str(npz_path))[0] + "_edit.npz"
    src = side if os.path.exists(side) else str(npz_path)
    with np.load(src) as d:
        return d["seg"].astype(np.uint8), (src == side)


def _voxel(src):
    """(vx,vy,vz) mm from the npz's own 'voxel', else a sibling meta.npz, else (1,1,1)."""
    src = Path(src)
    with np.load(src) as d:
        if "voxel" in d.files:
            return tuple(float(v) for v in d["voxel"])
    meta = src.with_name("meta.npz")
    if meta.exists():
        with np.load(meta) as m:
            if "voxel" in m.files:
                return tuple(float(v) for v in m["voxel"])
    return (1.0, 1.0, 1.0)


def _resolve_frames(path):
    """Return [(label, npz_path), ...]: a context4d dir -> its f*.npz frames; a file -> itself.

    EXCLUDES the `f<N>_edit.npz` seg sidecars. They are an EDIT OF a frame, not a frame: they hold
    only `seg` (no `hu`). A bare `f*.npz` glob swept them in, so saving made bogus 'f0_edit' entries
    appear in the frame picker, and clicking one hit the missing-`hu` guard — which raised SystemExit
    (not an Exception, so the handler's `except Exception` never caught it) and silently killed the
    request thread. Hence "clicking does nothing"."""
    path = Path(path)
    if path.is_dir():
        fs = sorted((p for p in path.glob("f*.npz") if not p.stem.endswith("_edit")),
                    key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or -1))
        if fs:
            return [(p.stem, p) for p in fs]
    return [(path.stem, path)]


def _parse_acq(dirname):
    """Split a 'context4d_<date>_data_<cond?>_Acq<n>' folder name into (date, cond, acq).
    Handles the inconsistent date formats (07_20_22 / 09_07_22 / 8_31_22) and the fact that
    only some acquisitions carry a condition label (Baseline / Angiotensin)."""
    stem = dirname
    if stem.startswith("context4d_"):
        stem = stem[len("context4d_"):]
    date, sep, rest = stem.partition("_data_")
    if not sep:                                          # no _data_ token -> use the whole thing
        date, rest = "", stem
    date = date.replace("_", "/")                        # 07_20_22 -> 07/20/22
    # peel a trailing Acq<n>
    parts = rest.split("_")
    acq = ""
    if parts and parts[-1].lower().startswith("acq"):
        acq = parts.pop()
    cond = "_".join(parts)                               # what's left is the condition (may be "")
    return date, cond, acq


def _discover_acqs(path):
    """Given any acquisition dir or a frame file inside one, find all sibling context4d_* dirs.
    Returns (acqs, current_index) where acqs = [{path, name, date, cond, acq, label}, ...],
    sorted by (date, cond, acq). current_index points at the acquisition `path` belongs to."""
    path = Path(path).resolve()
    acq_dir = path if path.is_dir() and path.name.startswith("context4d_") else path.parent
    parent = acq_dir.parent
    dirs = sorted(d for d in parent.glob("context4d_*") if d.is_dir())
    acqs = []
    for d in dirs:
        date, cond, acq = _parse_acq(d.name)
        label = " · ".join(x for x in (cond or None, acq or None) if x) or d.name
        acqs.append({"path": str(d), "name": d.name,
                     "date": date, "cond": cond, "acq": acq, "label": label})

    def _datekey(s):                                     # "07/20/22" -> (22,7,20); missing -> big
        try:
            m, d2, y = (int(x) for x in s.split("/"))
            return (y, m, d2)
        except Exception:
            return (9999, 99, 99)

    def _acqseq(name):                                   # every Acq<n> number, in folder-name order
        # the acq NUMBER is the dynamic series' clock, so it (not the condition label) defines the
        # timeline within a date: Baseline/Angiotensin share one increasing run and must interleave
        # by number (Baseline Acq6 precedes Angiotensin Acq8), and the nested 2023 series
        # (Acq06_ANG_Acq02) order by (outer, inner). Condition stays a display label / tiebreak only.
        seq = []
        for tok in name.split("_"):
            if tok.lower().startswith("acq"):
                digits = "".join(ch for ch in tok if ch.isdigit())
                if digits:
                    seq.append(int(digits))
        return tuple(seq)

    # order by the true acquisition timeline: date, then the acq-number sequence, then condition.
    acqs.sort(key=lambda a: (_datekey(a["date"]), _acqseq(a["name"]), a["cond"]))
    cur = next((i for i, a in enumerate(acqs) if a["path"] == str(acq_dir)), 0)
    return acqs, cur


WL_PRESETS = {"contrast": (275.0, 850.0), "soft": (40.0, 400.0), "bone": (300.0, 1500.0)}


class Session:
    """Holds one acquisition in RAM: HU volume, editable seg (from sidecar if any),
    voxel spacing, undo stack. One active frame at a time; switching frames swaps both."""

    def __init__(self, path):
        self.lock = threading.RLock()
        self._fcache = {}                                # fi -> loaded frame state (background-warmed)
        self._prefetch = None
        self.acqs, self.ai = _discover_acqs(path)        # all sibling acquisitions + current idx
        self._open_acq_path(path)

    def _open_acq_path(self, path):
        """(Re)point the session at an acquisition dir (or single npz) and load its first frame."""
        self.frames = _resolve_frames(path)
        self.root = Path(path)
        self.fi = 0
        self._fcache = {}                                # new acquisition -> discard the old frames' cache
        self._load_frame(0)
        self._start_prefetch()                           # warm the other frames so a switch is instant

    def data_sig(self):
        """Cache-key token = this acq index + its on-disk build time. A rebuilt or switched
        acquisition mints a distinct token, so /backdrop URLs never collide across acqs/builds in
        the browser HTTP cache (the 'scrubbing jumps / different pig' bug). Stable within one build,
        so the max-age scrub-back cache still works. The server ignores it when rendering (RAM
        already holds the right acq); it only varies the browser cache key."""
        d = self.root if self.root.is_dir() else self.root.parent
        try:
            mt = int((d / "meta.npz").stat().st_mtime)
        except Exception:
            mt = 0
        return f"{self.ai}-{mt}"

    def set_acq(self, ai):
        """Switch to acquisition index `ai` (from self.acqs). Discards in-memory edits of the
        current acquisition — the caller/UI must have saved or confirmed first. Returns the
        new acquisition dict."""
        with self.lock:
            ai = int(np.clip(ai, 0, len(self.acqs) - 1))
            self.ai = ai
            self._open_acq_path(self.acqs[ai]["path"])
            return self.acqs[ai]

    # -- frame data --
    # A frame switch used to reload the whole volume synchronously (hu decompress ~2.4s on a full-res
    # whole-body frame, plus label counts) -> a multi-second stall. Now each frame is loaded once into
    # `_fcache` and the OTHERS are warmed in a background thread, so switching time-frames is instant.
    FRAME_CACHE_MAX = 8                          # bound RAM (~1 GB/frame full-res); these acqs have <=6

    def _read_frame(self, npz):
        """Load ONE frame's arrays + stats into a state dict. Pure (touches no self) so the prefetch
        thread can call it off-lock. Holds the slow work: hu decompress, seg load, label counts. (The
        old per-load np.percentile clim was dead code -- nothing read that value -- so it is gone.) The
        default slice is an approximate duct-median from a 2x-subsampled seg: a starting slice never
        needs voxel precision, and the full argwhere over ~370M voxels was a needless ~0.7s."""
        with np.load(npz) as d:
            if "hu" not in d.files or "seg" not in d.files:
                # ValueError, NOT SystemExit: this runs inside request/prefetch threads, where
                # SystemExit slips past `except Exception` and kills the thread silently.
                raise ValueError(f"[ipad_paint] {npz} lacks hu/seg (found {list(d.files)})")
            hu = np.ascontiguousarray(d["hu"])
        seg, resumed = _load_seg_sidecar(npz)
        seg = np.ascontiguousarray(seg)
        voxel = _voxel(npz)
        count = {k: int((seg == k).sum()) for k in (1, 2, 3, 4)}
        # Opening slice = the median of the duct along each cut axis. Sampled every 2nd voxel
        # because this runs on every frame load; but a 2x sample MISSES a duct whose voxels all
        # sit on odd indices, and the viewer then silently opened on the geometric middle of the
        # volume instead of on the seg. Fall back to the exact pass when the sample comes up
        # empty -- which only happens on a seg that is (near) empty, so it costs one scan.
        s = seg[::2, ::2, ::2]
        duct, scale = np.argwhere((s >= 1) & (s <= 2)), 2
        if not len(duct):
            duct, scale = np.argwhere((seg >= 1) & (seg <= 2)), 1
        idx = {}
        for v in ("sagittal", "coronal", "axial"):
            ax = AXIS[v]
            idx[v] = int(np.median(duct[:, ax]) * scale) if len(duct) else hu.shape[ax] // 2
        return {"hu": hu, "seg": seg, "voxel": voxel, "resumed": resumed,
                "count": count, "idx": idx, "npz": npz}

    def _cache_put(self, fi, st):
        self._fcache[fi] = st
        while len(self._fcache) > self.FRAME_CACHE_MAX:      # evict the oldest, never the current frame
            del self._fcache[next(k for k in self._fcache if k != self.fi)]

    def _load_frame(self, fi):
        fi = int(np.clip(fi, 0, len(self.frames) - 1))
        st = self._fcache.get(fi)
        if st is None:
            st = self._read_frame(self.frames[fi][1])
            self._cache_put(fi, st)
        self.hu = st["hu"]                                   # hu is read-only (never edited) -> share it
        self.seg = st["seg"].copy()                          # a live copy: painting must not touch the cache
        self.voxel = st["voxel"]
        self.npz = Path(st["npz"])
        self.sidecar = Path(os.path.splitext(str(st["npz"]))[0] + "_edit.npz")
        self.fi = fi
        self.undo = []                          # list of stroke-records (each a list of (x,y,z,old))
        self.redo = []                          # records undone, re-appliable (each (x,y,z,new))
        self.resumed = st["resumed"]
        # per-voxel physical volume, in mL (1 mm^3 = 1e-3 mL)
        self.voxvol_ml = float(np.prod(self.voxel)) / 1000.0
        # label voxel counts, kept incrementally so /counts is O(stroke) not O(340M voxels)
        self._count = dict(st["count"])
        self.idx = dict(st["idx"])

    def _start_prefetch(self):
        """Warm the remaining frames in a daemon thread so a time-frame tap is instant. The heavy
        _read_frame runs OFF the lock; only the tiny cache insert takes it. Abandons if the acq switches."""
        frames = self.frames
        def worker():
            for j in range(len(frames)):
                with self.lock:
                    if self.frames is not frames:
                        return                               # acq switched -> this warmup is stale
                    have = j in self._fcache
                if have:
                    continue
                try:
                    st = self._read_frame(frames[j][1])      # slow load, no lock held
                except Exception:
                    continue
                with self.lock:
                    if self.frames is frames and j not in self._fcache:
                        self._cache_put(j, st)
        self._prefetch = threading.Thread(target=worker, daemon=True)
        self._prefetch.start()

    def set_frame(self, fi):
        with self.lock:
            if fi != self.fi:
                self._load_frame(fi)

    # -- edits --
    def _tally_forward(self, rec):
        """A record is [(x,y,z,old), ...]; the NEW label is whatever seg holds now.
        Update the incremental count cache for applying the record."""
        c = self._count
        for (x, y, z, old) in rec:
            new = int(self.seg[x, y, z])
            if old in c: c[old] -= 1
            if new in c: c[new] += 1

    def _commit(self, rec):
        """Fold a finished edit record into the counts + undo stack. Call under the lock."""
        if rec:
            self._tally_forward(rec)
            self.undo.append(rec)
            self.redo.clear()                   # a new edit invalidates the redo stack
        return len(rec)

    def apply_stroke(self, view, idx, points_rc, radius_mm, label, slab,
                     depth="off", floor=120.0, ceiling=None, gate=True):
        """depth: 'off' = paint flat on this one slice (classic);
                  'snap' = back-project each point to the brightest IN-BAND slice in ±slab;
                  'grow' = back-project AND fill the contiguous in-band run (multi-slice).

        The HU band [floor, ceiling] gates EVERY write in every mode, including 'off' and
        slab 0 — the flat path used to skip it, which is why setting a threshold appeared to
        do nothing. gate=False opens the band (free drawing, by choice)."""
        with self.lock:
            rec = []
            if depth in ("snap", "grow") and slab > 0:
                paint_stroke_depth(self.hu, self.seg, view, idx, points_rc,
                                   radius_mm, self.voxel, slab, label,
                                   floor=floor, ceiling=ceiling, gate=gate,
                                   grow=(depth == "grow"), record=rec)
            else:
                paint_stroke(self.seg, self.hu.shape, view, idx, points_rc,
                             radius_mm, self.voxel, label, record=rec,
                             hu=self.hu, floor=floor, ceiling=ceiling, gate=gate)
            return self._commit(rec)

    def apply_lasso(self, view, idx, poly_rc, label, slab,
                    depth="off", floor=120.0, ceiling=None, gate=True):
        """Fill the region a closed pen loop encloses, under the same HU gate as the brush."""
        with self.lock:
            rec = []
            lasso_fill(self.hu, self.seg, view, idx, poly_rc, self.voxel, slab, label,
                       depth=depth, floor=floor, ceiling=ceiling, gate=gate, record=rec)
            return self._commit(rec)

    def apply_wand(self, view, idx, row, col, label, slab,
                   floor=120.0, ceiling=None, gate=True, radius_mm=12.0):
        """One tap -> the connected in-band blob around it. Returns (n_changed, note)."""
        with self.lock:
            rec = []
            _n, note = wand_fill(self.hu, self.seg, view, idx, row, col, self.voxel, label,
                                 slab=slab, floor=floor, ceiling=ceiling, gate=gate,
                                 radius_mm=radius_mm, record=rec)
            return self._commit(rec), note

    def clear_all(self):
        """Wipe ALL labels in the whole volume (every slice, every label) -> background.
        Recorded as ONE undo entry, so a single Undo brings the entire seg back."""
        with self.lock:
            nz = np.argwhere(self.seg > 0)
            if nz.size == 0:
                return 0
            olds = self.seg[nz[:, 0], nz[:, 1], nz[:, 2]]
            rec = [(int(x), int(y), int(z), int(o)) for (x, y, z), o in zip(nz, olds)]
            self.seg[nz[:, 0], nz[:, 1], nz[:, 2]] = 0
            for k in (1, 2, 3, 4):
                self._count[k] = 0
            self.undo.append(rec)
            self.redo.clear()                   # a new edit invalidates the redo stack
            return len(rec)

    def undo_last(self):
        with self.lock:
            if not self.undo:
                return 0
            rec = self.undo.pop()
            c = self._count
            redo_rec = []
            for (x, y, z, old) in reversed(rec):
                prev = int(self.seg[x, y, z])   # label being undone (what redo must restore)
                self.seg[x, y, z] = old
                if prev in c: c[prev] -= 1
                if old in c: c[old] += 1
                redo_rec.append((x, y, z, prev))
            self.redo.append(redo_rec)
            return len(rec)

    def redo_last(self):
        """Re-apply the most recently undone edit. Mirror of undo_last."""
        with self.lock:
            if not self.redo:
                return 0
            rec = self.redo.pop()
            c = self._count
            undo_rec = []
            for (x, y, z, new) in rec:
                old = int(self.seg[x, y, z])    # current (undone) value
                self.seg[x, y, z] = new
                if old in c: c[old] -= 1
                if new in c: c[new] += 1
                undo_rec.append((x, y, z, old))
            self.undo.append(undo_rec)
            return len(rec)

    def _builder_seed(self, npz):
        """The PRODUCER's automatic seg for `npz`, read straight from that file.

        Takes the path explicitly rather than reading self.npz: the caller reads this with the lock
        released, so self.npz can move underneath it. Deliberately not through _load_seg_sidecar --
        this is the thing a save has to be compared against.

        Returns (seg, ok). ok=False means the seed could not be READ (unreadable file), which is not
        the same as the frame having no seed. Collapsing the two would let an unreadable seed
        silently certify a mask as hand work."""
        try:
            with np.load(npz) as d:
                if "seg" not in d.files:
                    return None, True                    # genuinely seedless frame
                return d["seg"].astype(np.uint8), True
        except Exception:
            return None, False                           # could not tell

    def save(self):
        """Write the sidecar, and record whether a human actually changed anything.

        Without the `edited` stamp a frame that was merely opened and saved is indistinguishable on
        disk from a hand-painted one -- the sidecar just echoes the builder's auto-seed, and every
        consumer (io.context.load_seg, measure.context.load_context_seg, the mesh scripts) reads it as
        manual ground truth. That is how a threshold seed sitting on rib cortex ended up being treated
        as a painted lymph label. `n_diff_seed` is the voxel count that differs from the seed;
        -1 means the frame had no seed to compare against, so the sidecar is human by construction;
        -2 means the seed could not be read, so provenance is unknown and must not be trusted.

        Returns (sidecar, voxels, n_diff_seed), or (None, 0, None) when the frame or acquisition
        changed while the seed was being read -- see below."""
        # Snapshot the target under the lock, read the seed for THAT path with the lock released
        # (it decompresses ~350 MB and must not stall painting), then re-take the lock.
        with self.lock:
            npz_at_start = self.npz
        seed, seed_ok = self._builder_seed(npz_at_start)
        with self.lock:
            if self.npz != npz_at_start:
                # A frame or acquisition switch landed mid-save. self.seg now holds a DIFFERENT
                # frame's labels, so there is no correct file to write: the old target would get
                # the new frame's mask, the new target a save nobody asked for. Refuse, and let
                # the caller keep the unsaved-changes flag up so the user can press Save again.
                return None, 0, None
            self.sidecar.parent.mkdir(parents=True, exist_ok=True)
            seg = self.seg.astype(np.uint8)
            stamp = seed_stamp(seg, seed if seed_ok else None, seed_ok=seed_ok)
            n_diff = int(stamp["n_diff_seed"])   # np.int64 is not JSON-serialisable; /save returns it
            np.savez_compressed(self.sidecar, seg=seg, source=self.npz.name, **stamp)
            st = self._fcache.get(self.fi)                   # keep the warm cache current after a save
            if st is not None:
                st["seg"] = self.seg.copy()
                st["count"] = dict(self._count)
            return self.sidecar, int((self.seg > 0).sum()), n_diff

    def frames_edited(self):
        """Which frames already carry a saved manual-edit sidecar (`f<N>_edit.npz`), so the frame
        picker can mark them. That sidecar is what this tool AND io.context.load_seg auto-read in
        preference to the frame's built-in seg — i.e. a marked frame is one whose seg is yours."""
        return [os.path.exists(os.path.splitext(str(p))[0] + "_edit.npz") for _lbl, p in self.frames]

    def counts(self):
        """Voxel counts (cached, O(1)) plus physical volume in mL per label."""
        with self.lock:
            vox = {int(k): int(v) for k, v in self._count.items()}
            ml = {int(k): round(v * self.voxvol_ml, 4) for k, v in self._count.items()}
            return {"vox": vox, "ml": ml, "voxvol_ml": self.voxvol_ml}


# --------------------------------------------------------------------------- #
# Rendering — backdrop PNG (grayscale slab-MIP) + overlay PNG (RGBA labels)
# --------------------------------------------------------------------------- #
def _wl(arr, level, window):
    lo = level - window / 2.0
    return (np.clip((arr - lo) / max(window, 1e-6), 0.0, 1.0) * 255).astype(np.uint8)


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_backdrop_jpeg(sess, view, idx, slab, level, window, rot=0):
    """Grayscale slab-MIP of HU at NATIVE display resolution, JPEG-encoded.

    We deliberately do NOT upsample on the server: the browser draws this into a
    physical-mm canvas rect (so anatomy keeps correct proportions) with one clean
    bilinear pass. An earlier version pre-upsampled the coarse in-plane axis with a
    Lanczos (windowed-sinc) filter, whose negative side-lobes rang around every bright
    rib and contrast-filled duct — the "contour-map" / horizontal-banding artifact.
    Sending native pixels + a single bilinear stretch removes that entirely.

    JPEG (not PNG) because the backdrop is what you scrub: q90 grayscale is ~2.3x
    smaller than PNG and encodes ~20x faster, so dragging the slice slider is smooth.
    It is visually lossless for a windowed CT; the seg overlay stays lossless PNG."""
    from PIL import Image
    disp = to_display(sess.hu, view, idx, slab)
    g = _wl(disp.astype(np.float32), level, window)
    if rot % 4:
        g = np.ascontiguousarray(np.rot90(g, rot))       # display-only turn; paint core unaffected
    buf = io.BytesIO()
    Image.fromarray(g, "L").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def gate_plane(hu, view, idx, slab, lo, hi):
    """Bool display plane: does this column hold a voxel with lo <= HU <= hi?

    Exactly mirrors to_display's projection and cranial-up transform, so the tint lands on the
    same pixels as the backdrop it explains. With slab>0 it is an ANY over the window, not a max
    over it: a column whose brightest voxel is above the ceiling can still contain an in-band
    one, and that column IS paintable, so a max-based preview would lie about it."""
    ax = AXIS[view]
    idx = int(np.clip(idx, 0, hu.shape[ax] - 1))
    sl = [slice(None)] * 3
    if slab > 0:
        sl[ax] = slice(max(idx - slab, 0), min(idx + slab + 1, hu.shape[ax]))
        sub = hu[tuple(sl)]
        plane = ((sub >= lo) & (sub <= hi)).any(axis=ax)
    else:
        sl[ax] = idx
        sub = hu[tuple(sl)]
        plane = (sub >= lo) & (sub <= hi)
    if view == "axial":
        return plane
    return plane.T[::-1]


def render_overlay_png(sess, view, idx, visible=None, rot=0, slab=0, gate=None):
    """RGBA overlay in TWO layers, so you can judge coverage on a thick slab:

      * SOLID (alpha 255) = the label on THIS slice — always the TRUE per-slice seg.
      * GHOST (alpha GHOST_ALPHA) = the label ANYWHERE in the ±slab, projected like the CT backdrop.

    Why: depth-follow ('grow') deliberately paints the duct across the several slices it really spans,
    but a single-slice overlay only ever showed the centre one — so a correctly-painted duct looked
    unpainted and you couldn't tell if you'd covered it. The ghost shows the slab-wide coverage while
    the solid layer keeps "what is on this slice" unambiguous. slab=0 -> no ghost = pure single-slice
    truth (the delivery/verification view; the projection is an EDITING AID only).

    `visible` is an iterable of label ids to DRAW; labels not in it are hidden (fully transparent).
    None means show all. Display only — the seg is untouched by hiding.

    `gate` = (floor, ceiling) draws the HU gate itself as a faint blue wash UNDER the labels:
    every pixel the brush/lasso is ALLOWED to paint. Without it the threshold is invisible and
    you can only find out what it does by painting and undoing — which is exactly how a band
    that gates nothing (floor 120, open ceiling, on a thorax full of rib and contrast) passes
    for a band that is broken."""
    from PIL import Image
    show = set(LABELS) if visible is None else {int(v) for v in visible}
    cur = to_display(sess.seg, view, idx, 0).astype(np.uint8)
    # max-project the labels over the same window as the backdrop. Safe here: CC/TD are split along z,
    # so a single column of the cut axis never holds two different duct labels.
    proj = to_display(sess.seg, view, idx, slab).astype(np.uint8) if slab > 0 else cur
    band = None
    if gate is not None:
        lo, hi = band_bounds(gate[0], gate[1])
        if not _band_is_open(lo, hi):
            band = gate_plane(sess.hu, view, idx, slab, lo, hi)
    if rot % 4:                                            # same turn as the backdrop tile
        cur = np.ascontiguousarray(np.rot90(cur, rot))
        proj = np.ascontiguousarray(np.rot90(proj, rot))
        if band is not None:
            band = np.ascontiguousarray(np.rot90(band, rot))
    H, W = cur.shape
    rgba = np.zeros((H, W, 4), np.uint8)
    if band is not None and band.any():                    # the gate wash, beneath everything
        rgba[band, :3] = GATE_RGB
        rgba[band, 3] = GATE_ALPHA
    for lid, (_name, hexc) in LABELS.items():              # all ghosts first...
        if lid not in show:
            continue
        m = proj == lid
        if m.any():
            rgba[m, :3] = _hex_rgb(hexc)
            rgba[m, 3] = GHOST_ALPHA
    for lid, (_name, hexc) in LABELS.items():              # ...then this slice's labels solid on top
        if lid not in show:
            continue
        m = cur == lid
        if m.any():
            rgba[m, :3] = _hex_rgb(hexc)
            rgba[m, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Front-end (single page). Apple Pencil via Pointer Events + getCoalescedEvents.
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>CC/TD paint</title>
<style>
  :root{ --bg:#111; --panel:#1c1c1e; --ink:#eee; --line:#333; color-scheme:dark; }
  *{ box-sizing:border-box; -webkit-user-select:none; user-select:none; -webkit-touch-callout:none; }
  html,body{ margin:0; height:100%; background:var(--bg); color:var(--ink);
    font:14px -apple-system,system-ui,sans-serif; overflow:hidden; touch-action:none; }
  #wrap{ display:flex; height:100%; }
  /* Layout: [left: set-once settings] [mid: image + full-width slice scrub] [right: the hot controls].
     depth-follow/HU-target used to sit BETWEEN undo/redo and slice/slab/brush, splitting the controls
     you actually click; and the slice slider was stuck at ~80px of travel inside the 186px sidebar
     (~3 slices/px coronal, ~17 axial -> impossible not to overshoot). The scrub bar under the image
     is ~10x longer. */
  #left{ width:178px; background:var(--panel); border-right:1px solid var(--line);
    padding:10px; display:flex; flex-direction:column; gap:9px; overflow-y:auto; }
  #mid{ flex:1; display:flex; flex-direction:column; min-width:0; }
  #stage{ flex:1; position:relative; overflow:hidden; background:#000; touch-action:none; }
  #scrub{ background:var(--panel); border-top:1px solid var(--line); padding:6px 14px 10px; }
  #cv{ position:absolute; left:0; top:0; touch-action:none; }
  #marks span{ position:absolute; color:#7cf; font-weight:600; text-shadow:0 0 3px #000; pointer-events:none; }
  #side{ width:186px; background:var(--panel); border-left:1px solid var(--line);
    padding:10px; display:flex; flex-direction:column; gap:9px; overflow-y:auto; }
  .lab{ display:flex; align-items:center; gap:8px; padding:11px 10px; border-radius:11px;
    border:2px solid transparent; background:#2a2a2c; font-weight:600; font-size:16px; }
  .lab .sw{ width:20px; height:20px; border-radius:5px; border:1px solid #0006; }
  .lab.on{ border-color:#fff; background:#3a3a3d; }
  .lab .hot{ margin-left:auto; color:#888; font-weight:400; font-size:12px; }
  .lab .eye{ width:20px; text-align:center; font-size:17px; color:#7cf; cursor:pointer; }
  .lab .eye.off{ color:#666; }                            /* hidden label = dim hollow ring */
  .btn{ padding:11px; border-radius:11px; background:#2a2a2c; border:1px solid #0000;
    text-align:center; font-weight:600; font-size:15px; }
  .btn:active{ background:#3a3a3d; }
  .btn.save{ background:#0a84ff; } .btn.erase.on{ background:#8a5; color:#000; }
  .btn.danger{ background:#3a2323; color:#ff8f8f; border-color:#5a2a2a; }
  .btn.on{ background:#0a84ff; color:#fff; }
  .row{ display:flex; gap:7px; } .row>*{ flex:1; }
  label.sl{ display:block; font-size:12px; color:#aab; margin:2px 0 -2px; }
  input[type=range]{ width:100%; }
  .stepper{ display:flex; align-items:center; gap:6px; }
  .stepper input[type=range]{ flex:1; min-width:0; }
  .stepbtn{ width:46px; min-width:46px; padding:9px 0; text-align:center; font-size:20px;
    font-weight:700; line-height:1; border-radius:9px; background:#2a2a2c; color:var(--ink);
    border:1px solid var(--line); color-scheme:dark; -webkit-appearance:none; appearance:none;
    -webkit-user-select:none; user-select:none; touch-action:manipulation; }
  .stepbtn:active{ background:#0a84ff; color:#fff; }
  /* HU floor/ceiling need one-step precision on a 178px rail, but a 46px button each side
     would leave ~60px of slider. 32px keeps both usable. */
  .stepbtn.sm{ width:32px; min-width:32px; padding:7px 0; font-size:17px; }
  #gatehint{ font-size:11px; color:#8ab; line-height:1.4; min-height:2.4em; }
  #gatehint.warn{ color:#e0a33a; }
  select,.mini{ width:100%; padding:8px; border-radius:9px; background:#2a2a2c; color:var(--ink);
    border:1px solid var(--line); font-size:14px; color-scheme:dark;
    -webkit-appearance:none; appearance:none; }
  /* iOS Safari renders the native <select> value + popup with system colours and ignores
     the `color` above unless color-scheme is dark — that was the dark-on-dark text bug. */
  option{ background:#2a2a2c; color:#eee; }
  optgroup{ background:#1c1c1e; color:#8ab; font-weight:600; }
  #hud{ font-size:12px; color:#9aa; line-height:1.5; }
  .seg{ display:flex; gap:5px; } .seg>*{ flex:1; padding:8px 0; text-align:center;
    background:#2a2a2c; border-radius:8px; font-size:13px; } .seg>.on{ background:#0a84ff; }
  #toast{ position:absolute; left:50%; bottom:16px; transform:translateX(-50%);
    background:#000c; padding:8px 14px; border-radius:20px; opacity:0; transition:.2s; font-size:13px; }
</style></head><body>
<div id="wrap">
  <div id="left">
    <label class="sl">Tool</label>
    <div class="seg" id="toolsel">
      <div data-t="brush" class="on">Brush</div><div data-t="lasso">Lasso</div><div data-t="wand">Wand</div>
    </div>
    <label class="sl">MIP depth-follow</label>
    <div class="seg" id="depthsel">
      <div data-d="off">Off</div><div data-d="snap">Snap</div><div data-d="grow" class="on">Grow</div>
    </div>
    <label class="sl">HU gate</label>
    <div class="seg" id="gatesel">
      <div data-g="1" class="on">On</div><div data-g="0">Off</div>
    </div>
    <div id="bandbox">
      <label class="sl">Floor <span id="floorv"></span> HU</label>
      <div class="stepper">
        <div class="stepbtn sm" id="floordn">&minus;</div>
        <input type="range" id="floor" min="-200" max="800" step="5" value="120">
        <div class="stepbtn sm" id="floorup">+</div>
      </div>
      <label class="sl">Ceiling <span id="ceilv"></span> HU</label>
      <div class="stepper">
        <div class="stepbtn sm" id="ceildn">&minus;</div>
        <input type="range" id="ceil" min="0" max="4000" step="5" value="4000">
        <div class="stepbtn sm" id="ceilup">+</div>
      </div>
      <div class="btn" id="showgate">Show gate</div>
      <div id="gatehint"></div>
    </div>
    <div id="wandbox">
      <label class="sl">Wand reach <span id="wandrv"></span> mm</label>
      <input type="range" id="wandr" min="3" max="25" step="1" value="12">
    </div>
  </div>
  <div id="mid">
    <div id="stage"><canvas id="cv"></canvas><div id="marks"></div><div id="toast"></div></div>
    <div id="scrub">
      <label class="sl">Slice <span id="idxv"></span></label>
      <div class="stepper">
        <div class="stepbtn" id="idxdn">&minus;</div>
        <input type="range" id="idx" min="0" max="1" value="0">
        <div class="stepbtn" id="idxup">+</div>
      </div>
    </div>
  </div>
  <div id="side">
    <label class="sl">Acquisition</label>
    <select id="acq"></select>
    <div class="seg" id="viewsel">
      <div data-v="coronal" class="on">Cor</div><div data-v="sagittal">Sag</div><div data-v="axial">Ax</div>
    </div>
    <div class="btn" id="rotate">⟳ Rotate 90°</div>
    <div id="labs"></div>
    <div class="btn erase" id="erase">Erase (e)</div>
    <div class="row"><div class="btn" id="undo">Undo</div><div class="btn" id="redo">Redo</div></div>
    <div class="btn danger" id="clear">Clear all</div>
    <label class="sl">Slab ± <span id="slabv"></span></label>
    <input type="range" id="slab" min="0" max="24" value="8">
    <label class="sl">Brush <span id="brushv"></span> mm</label>
    <input type="range" id="brush" min="1" max="12" step="0.5" value="3">
    <label class="sl">Zoom <span id="zoomv"></span>×</label>
    <input type="range" id="zoom" min="1" max="8" step="0.25" value="1">
    <select id="win">
      <option value="contrast">window: contrast</option>
      <option value="soft">window: soft tissue</option>
      <option value="bone">window: bone</option>
    </select>
    <select id="frame"></select>
    <div class="btn save" id="save">Save (⌘S)</div>
    <div id="hud"></div>
  </div>
</div>
<script>
const S={ view:"coronal", idx:0, slab:8, brush:3, label:1, erase:false, win:"contrast",
  frame:0, sig:"", shape:null, overlayImg:new Image(), backImg:new Image(), scale:1, ox:0, oy:0,
  W:0,H:0, drawing:false, pts:[], busy:false,
  zoom:1, panX:0, panY:0, mmPerPx:1, dpr:(window.devicePixelRatio||1),
  depth:"grow", floor:120, ceiling:null,        // depth-follow ON by default: a stroke on the slab-MIP
                                                // back-projects onto the true duct slice(s), not the
                                                // centre slice (which was ~80% wrong). off/snap still available.
  tool:"brush",                                 // brush | lasso | wand
  gate:true, showGate:false, wandR:12,          // gate: [floor,ceiling] restricts EVERY write, in every
                                                // tool and every depth mode. Off = free drawing.
  rot:0,                                        // on-screen 90° turns (display only)
  vis:new Set([1]),                             // which labels are DISPLAYED (default: only CC)
  dirty:false, acq:0 };                         // unsaved edits? / current acquisition index
const cv=document.getElementById("cv"), ctx=cv.getContext("2d"), stage=document.getElementById("stage");
const $=id=>document.getElementById(id);
function toast(t){ const el=$("toast"); el.textContent=t; el.style.opacity=1; clearTimeout(el._t);
  el._t=setTimeout(()=>el.style.opacity=0,1200); }
async function jget(u){ const r=await fetch(u); return r.json(); }
async function jpost(u,b){ const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(b)}); return r.json(); }

// ---- layout: fit the display plane into the stage, keep physical aspect ----
// The canvas holds the base-fit image at internal resolution S.W×S.H; zoom/pan are
// a CSS transform on top, so image<->canvas mapping is zoom-independent and crisp.
function layout(){
  const availW=stage.clientWidth, availH=stage.clientHeight;
  if(!availW||!availH||!S.dispW_mm||!S.dispH_mm) return;   // stage not sized yet (boot/flex race): a ResizeObserver re-runs layout once it has a real size, so we never commit a 0 scale (which left the canvas 0x0 / mis-placed)
  const s=Math.min(availW/S.dispW_mm, availH/S.dispH_mm);   // mm->px, isotropic in mm
  S.scale=s;
  S.W=Math.round(S.dispW_mm*s); S.H=Math.round(S.dispH_mm*s);
  S.ox=Math.round((availW-S.W)/2); S.oy=Math.round((availH-S.H)/2);
  // Retina backing store: CSS box stays S.W×S.H (mm-fit) but the bitmap is dpr× bigger,
  // so the crisp square-rendered PNG isn't softened by a 1x canvas on a 2x screen.
  S.dpr=window.devicePixelRatio||1;
  cv.width=Math.round(S.W*S.dpr); cv.height=Math.round(S.H*S.dpr);
  cv.style.width=S.W+"px"; cv.style.height=S.H+"px";
  cv.style.left=S.ox+"px"; cv.style.top=S.oy+"px";
  cv.style.transformOrigin="0 0";
  S.mmPerPx=S.dispW_mm/S.W;
  clampPan();                                 // stage just changed size (iPad rotation): a pan that
  applyTransform();                           // was legal at the old size can strand the image off-screen
  redraw();
}
function applyTransform(){
  cv.style.transform=`translate(${S.panX}px,${S.panY}px) scale(${S.zoom})`;
  placeMarks();
}
function placeMarks(){
  const m=S.marks, box=$("marks"); box.innerHTML="";
  const mk=(t,x,y)=>{ const e=document.createElement("span"); e.textContent=t;
    e.style.left=x+"px"; e.style.top=y+"px"; box.appendChild(e); };
  const x0=S.ox+S.panX, y0=S.oy+S.panY, W=S.W*S.zoom, H=S.H*S.zoom;   // post-transform box
  const cx=x0+W/2, cy=y0+H/2;
  mk(m[0],cx-6,Math.max(2,y0)+4); mk(m[1],cx-6,Math.min(stage.clientHeight-24,y0+H-22));
  mk(m[2],Math.max(2,x0)+4,cy-8); mk(m[3],Math.min(stage.clientWidth-16,x0+W)-16,cy-8);
}
// client px -> LOGICAL canvas px (S.W×S.H space; the rect already accounts for zoom/pan)
function clientToCanvas(cx,cy){
  const r=cv.getBoundingClientRect();
  return [(cx-r.left)/r.width*S.W, (cy-r.top)/r.height*S.H];
}
// logical canvas px -> image px (rows,cols in the server's display array).
function canvasToImg(px,py){
  return [py/S.H*S.rows, px/S.W*S.cols];
}
function redraw(){
  ctx.setTransform(S.dpr,0,0,S.dpr,0,0);               // draw in logical S.W×S.H, backing store is dpr×
  ctx.clearRect(0,0,S.W,S.H);
  // CT backdrop: SMOOTH interpolation. (This is reset every frame on purpose — leaving
  // imageSmoothingEnabled=false from a prior overlay draw was what made the grayscale
  // look blocky / "contour-map"-banded. It must be true here.)
  if(drawable(S.backImg)){
    ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality="high";
    ctx.drawImage(S.backImg,0,0,S.W,S.H); }
  // Label overlay: NEAREST so label edges stay crisp (no colour bleed between labels).
  if(drawable(S.overlayImg)){ ctx.globalAlpha=0.85;
    ctx.imageSmoothingEnabled=false; ctx.drawImage(S.overlayImg,0,0,S.W,S.H);
    ctx.globalAlpha=1; ctx.imageSmoothingEnabled=true; }
  if(S.drawing && S.pts.length){                         // live ink preview (canvas-internal px)
    const col=S.erase?"#e88":({1:"#5aa469",2:"#3a8f9c",3:"#e0a33a",4:"#b5916a"}[S.label]);
    ctx.strokeStyle=col; ctx.globalAlpha=S.erase?0.6:0.9;
    ctx.lineCap="round"; ctx.lineJoin="round";
    if(S.tool==="lasso"){
      // A lasso is a REGION, so preview it as one: a hairline loop, closed back to the start
      // (dashed, because that segment is implied rather than drawn), with the interior tinted.
      // Previewing it as a fat brush stroke would misrepresent what is about to be filled.
      ctx.lineWidth=Math.max(1.5, 2/S.mmPerPx*0.5);
      ctx.beginPath(); ctx.moveTo(S.pts[0][0],S.pts[0][1]);
      for(const p of S.pts) ctx.lineTo(p[0],p[1]);
      if(S.pts.length>2){ ctx.closePath(); ctx.globalAlpha=0.18; ctx.fillStyle=col; ctx.fill(); }
      ctx.globalAlpha=0.95; ctx.setLineDash([6,4]); ctx.stroke(); ctx.setLineDash([]);
    } else {
      ctx.lineWidth=Math.max(2, S.brush/S.mmPerPx*2);    // brush diameter in canvas px
      ctx.beginPath(); ctx.moveTo(S.pts[0][0],S.pts[0][1]);
      for(const p of S.pts) ctx.lineTo(p[0],p[1]); ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
}
// Backdrop URL is cacheable (max-age) so Safari reuses the tile when you scroll back to a slice
// - cuts the request storm that made iPad Safari cancel in-flight image loads (the BrokenPipe
// spam). It ALSO carries a per-acq/build `sig=` token (from /meta -> data_sig): (view,idx,slab,
// win,frame,rot) is NOT unique across acquisitions or a rebuild - same URL, same host:port,
// different pixels - so without sig the iPad HTTP cache served a *different pig's* tile for a
// byte-identical URL, which WAS the "scrubbing jumps / not continuous" bug (confirmed: a Safari
// Private tab, with no persistent HTTP cache, does not jump). sig gives every acq/build its own
// URL namespace while keeping scrub-back instant within one build. Overlay carries `g=` per edit.
S.gen=0;
S.gateTick=0;                    // bumped by the band controls so the gate wash re-fetches
function backUrl(){ return `/backdrop?view=${S.view}&idx=${S.idx}&slab=${S.slab}&win=${S.win}&frame=${S.frame}&rot=${S.rot}&sig=${S.sig}`; }
// `slab` makes the overlay ghost the labels across the SAME window the CT backdrop projects, so a
// grow-painted duct (which spans several slices) is visible as coverage, not just on the centre slice.
function ovUrl(){ let u=`/overlay?view=${S.view}&idx=${S.idx}&frame=${S.frame}&g=${S.gen}`+
  `&vis=${[...S.vis].join(",")}&rot=${S.rot}&slab=${S.slab}`;     // only toggled-on labels drawn (display only)
  // "Show gate" tints every pixel the band allows. g=S.gen already busts the cache per edit, and
  // gateTick is bumped by the band sliders so moving Floor repaints the wash immediately.
  if(S.showGate && S.gate) u+=`&gfloor=${S.floor}&gceil=${S.ceiling==null?"":S.ceiling}&gt=${S.gateTick}`;
  return u; }
// ---- slice tile loading (iOS-Safari-robust, cache-keyed, NEVER aborts) ------
// THE BUG this fixes (recurred after the abort-based v15 attempt): scrubbing coronal slices
// showed a "different pig" / the image jumping back-and-forth (150-156 the first time, then
// 130-140), as if two acquisitions were interleaved. Re-verified against the data for BOTH
// ranges: the volume on disk is smooth (adjacent NCC ~0.98), the server slab-MIP is smooth
// (NCC ~0.99, slice i always closer to i+1 than i+2 -> NO interleave/period-2 signature), and
// every index renders a UNIQUE server tile. So it was never anatomy, never mixed acqs, never
// disk read order -> purely a client display fault.
//
// LEADING HYPOTHESIS for the recurrence (NOT instrumented — no browser/sockets in the dev
// sandbox, so this is a reasoned suspicion, not a measured fact): the backdrop is served
// Cache-Control:max-age=86400 (cacheable so scrub-back is instant), and the v15 code fetch()ed
// it and then ABORTED superseded fetches on every slice change. A known iOS-Safari failure mode
// is that aborting an in-flight fetch of a *cacheable* URL can leave a poisoned (empty/partial)
// entry in the HTTP cache for that URL; a later scrub back onto that slice would then return the
// poisoned entry -> createImageBitmap throws -> the catch swallows it -> the canvas keeps the
// previous tile. That WOULD explain stale tiles being specific indices that move to whatever
// range you fling across. Whether this exact mechanism was the culprit is unconfirmed; what IS
// established here is that the data + server render are clean in both affected ranges, so the
// fault is client-side, and that v15's abort-of-a-cacheable-fetch is the most plausible trigger.
//
// FIX (defensive; removes that suspected trigger entirely, no abort anywhere): keep a bounded cache of DECODED tiles keyed
// by the full slice identity (the request URL is that key: view|idx|slab|win|frame for the
// backdrop, +g+vis for the overlay). A slice change never cancels anything; it just asks to
// paint the current keys. If both current tiles are cached we paint instantly; if not we fetch
// (deduped) and paint only once the fetched tile is STILL the current one. Superseded fetches
// complete and populate the cache (making scrub-back instant) but never repaint a slice you've
// left. Nothing is ever aborted, so the suspected cache-poisoning trigger cannot occur; and even
// if the true cause was something else, painting only the still-current tile is stale-slice-safe
// by construction.
//
// UPDATE (2026-07-17, confirmed): the recurrence on the full-res multi-acq data was NOT abort
// poisoning. Root cause: this backdrop URL carried NO acq/build identity yet was served
// Cache-Control:max-age=86400, so across an acq switch or a rebuild at the same host:port the
// browser HTTP cache (which THIS JS cache and cacheClear() sit ABOVE and cannot evict) handed
// back a stale DIFFERENT-acq tile for a byte-identical URL = the "jumping/different pig". Proven
// by a Safari Private tab (no persistent HTTP cache) not jumping. Fix: backUrl now carries a
// per-acq/build `sig=` token (see /meta -> Session.data_sig) so distinct data always gets a
// distinct URL. The never-abort cache below is still correct and kept; it just wasn't the whole story.
function drawable(img){ return !!(img && img.width && img.complete!==false); }  // Image or ImageBitmap
function closeBmp(x){ if(x && typeof x.close==="function"){ try{ x.close(); }catch(_){} } }
const TILE_CAP=40;                                         // ~1.4MB decoded each -> ~57MB ceiling
let tileCache=new Map();                                   // url(key) -> ImageBitmap/Image (LRU by insert order)
let inflight=new Map();                                    // url(key) -> Promise (dedupe concurrent fetches)
function cacheGet(k){ const v=tileCache.get(k);
  if(v){ tileCache.delete(k); tileCache.set(k,v); }        // LRU touch: move to newest
  return v; }
function cachePut(k,bmp){
  if(tileCache.has(k)) closeBmp(tileCache.get(k));
  tileCache.set(k,bmp);
  while(tileCache.size>TILE_CAP){ const oldest=tileCache.keys().next().value;
    closeBmp(tileCache.get(oldest)); tileCache.delete(oldest); } }
function cacheClear(){ for(const b of tileCache.values()) closeBmp(b);
  tileCache.clear(); inflight.clear(); }                   // dims change on view/frame/acq switch
async function decodeBlob(blob){
  if(window.createImageBitmap){ try{ return await createImageBitmap(blob); }catch(_){} }
  return await new Promise((res,rej)=>{ const im=new Image(); const u=URL.createObjectURL(blob);
    im.onload=()=>{ URL.revokeObjectURL(u); res(im); };
    im.onerror=()=>{ URL.revokeObjectURL(u); rej(new Error("decode")); }; im.src=u; }); }
function fetchInto(url){                                   // -> Promise<bitmap>; deduped; NEVER aborted
  const hit=cacheGet(url); if(hit) return Promise.resolve(hit);
  if(inflight.has(url)) return inflight.get(url);
  const p=(async()=>{
    try{
      const r=await fetch(url,{cache:"default"});
      if(!r.ok) throw new Error("http "+r.status);
      const bmp=await decodeBlob(await r.blob());
      cachePut(url,bmp); return bmp;
    } finally { inflight.delete(url); }
  })();
  inflight.set(url,p); return p;
}
// Paint the CURRENT slice from cache. Requires BOTH current layers (backdrop+overlay) so the
// two can never show different indices (the old "new CT under old labels" flicker). Returns
// true iff it painted. Leaves the previous consistent frame untouched until the new pair is ready.
function paintCurrent(){
  const bk=cacheGet(backUrl()), ov=cacheGet(ovUrl());
  if(drawable(bk) && drawable(ov)){ S.backImg=bk; S.overlayImg=ov; redraw(); return true; }
  return false;
}
// One coalesced pump per animation frame: paint if ready, else fetch the missing current tiles
// and re-pump when they land (only if still current). All of refresh{Back,Overlay,Both} funnel
// here — which layer changed doesn't matter, pump always reconciles against the live S.* keys.
let _pumpScheduled=false;
function schedulePump(){ if(!_pumpScheduled){ _pumpScheduled=true; requestAnimationFrame(pump); } }
function pump(){
  _pumpScheduled=false;
  if(paintCurrent()){ prefetchNeighbors(); return; }       // both cached -> instant, then warm neighbors
  const bku=backUrl(), ovu=ovUrl();
  for(const u of [bku, ovu]){
    if(tileCache.has(u)) continue;
    fetchInto(u).then(()=>{ if(u===backUrl()||u===ovUrl()) schedulePump(); }).catch(()=>{});
  }
}
function refreshBack(){ schedulePump(); }
function refreshOverlay(){ schedulePump(); }
function refreshBoth(){ schedulePump(); }
// Central slice setter: clamps, updates the slider + label, and pumps. Every path that changes
// the slice (drag, step buttons, arrow keys) goes through here so the displayed slice and the
// slider value can never disagree. (`prefetch` arg kept for call-site compatibility; neighbor
// prefetch now happens automatically once the current pair is painted.)
function setIdx(v, prefetch){
  v=Math.max(0, Math.min((S.nidx||1)-1, Math.round(v)));
  S.idx=v; const sl=$("idx"); if(+sl.value!==v) sl.value=v;
  $("idxv").textContent=`${S.idx}/${(S.nidx||1)-1}`;
  gateStat();                                 // the band's coverage is per-slab, so per-slice
  schedulePump();
}
// Warm the DECODED-tile cache for nearby slices so stepping / scrub-back paints instantly.
// Backdrops only (the overlay is per-edit and cheap); deduped and bounded by TILE_CAP.
function prefetchNeighbors(){
  for(const d of [1,-1,2,-2,3,-3]){ const j=S.idx+d;
    if(j<0||j>=(S.nidx||1)) continue;
    const u=`/backdrop?view=${S.view}&idx=${j}&slab=${S.slab}&win=${S.win}&frame=${S.frame}&rot=${S.rot}&sig=${S.sig}`;
    if(!tileCache.has(u) && !inflight.has(u)) fetchInto(u).catch(()=>{});
    // ALSO warm the overlay: paintCurrent needs BOTH layers, so an un-prefetched overlay made every
    // slice step wait a round-trip for it before painting. It is tiny (~4KB) and quick to render.
    const ov=`/overlay?view=${S.view}&idx=${j}&frame=${S.frame}&g=${S.gen}`+
      `&vis=${[...S.vis].join(",")}&rot=${S.rot}&slab=${S.slab}`;
    if(!tileCache.has(ov) && !inflight.has(ov)) fetchInto(ov).catch(()=>{});
  }
}

// ---- view geometry from server ----
// First load opens on the server's duct-median slice (default_idx), not slice 0 (the anterior
// body surface — mostly black). Later view switches keep the current slice (unchanged behaviour).
let _firstView=true, _forceDefault=false;   // _forceDefault: next loadView opens on the duct-median (set on acq switch)
async function loadView(){
  const d=await jget(`/meta?view=${S.view}&rot=${S.rot}`);
  S.rows=d.rows; S.cols=d.cols; S.dispW_mm=d.w_mm; S.dispH_mm=d.h_mm; S.marks=d.marks;
  S.nidx=d.nidx; S.mmPerColApprox=d.w_mm/d.cols*S.scale;
  S.sig=d.sig||"";                     // per-acq/build cache token -> distinct data mints a distinct backdrop URL
  const sl=$("idx"); sl.max=d.nidx-1;
  if(_firstView || _forceDefault || S.idx>d.nidx-1) S.idx=d.default_idx;    // open on the duct, not slice 0
  _firstView=false; _forceDefault=false;
  sl.value=S.idx;
  $("idxv").textContent=`${S.idx}/${d.nidx-1}`;
  cacheClear();                     // tile dims/content differ per view+frame+acq -> drop stale decoded tiles
  layout(); refreshBoth(); updateHud();
}
function updateHud(){
  jget("/counts").then(c=>{ const v=c.vox, ml=c.ml;
    const row=(name,id,col)=>`<span style="color:${col}">${name}</span> `+
      `<b>${ml[id].toFixed(3)}</b> mL <span style="color:#778">(${v[id]} vox)</span>`;
    $("hud").innerHTML=
      row("CC",1,"#5aa469")+"<br>"+row("TD",2,"#3a8f9c")+"<br>"+
      row("lymph",3,"#e0a33a")+"<br>"+row("bone",4,"#b5916a")+"<br>"+
      `<span style="color:#6c8">${S.saved||"unsaved"}</span>`; });
}

// ---- pointer drawing: Apple Pencil draws, fingers are ignored here (they pan/pinch on #stage) ----
function isPen(ev){ return ev.pointerType==="pen" ||
  (ev.pointerType!=="touch" && ev.pointerType!=="");     // mouse/trackpad also draw (desktop test)
}
function addPt(ev){ const [x,y]=clientToCanvas(ev.clientX,ev.clientY); S.pts.push([x,y]); }
cv.addEventListener("pointerdown",ev=>{
  if(!isPen(ev)) return;                                 // touch handled by #stage (pan/pinch/scroll)
  cv.setPointerCapture(ev.pointerId); S.drawing=true; S.pts=[]; addPt(ev); ev.preventDefault();
},{passive:false});
cv.addEventListener("pointermove",ev=>{
  if(!S.drawing || !isPen(ev)) return;
  const co=ev.getCoalescedEvents?ev.getCoalescedEvents():[ev];
  for(const e of co) addPt(e);
  redraw(); ev.preventDefault();
},{passive:false});
async function endStroke(ev){
  if(!S.drawing) return; S.drawing=false;
  if(!S.pts.length) return;
  const img=S.pts.map(p=>canvasToImg(p[0],p[1]));        // [[row,col],...] in display space
  const band={floor:S.floor, ceiling:S.ceiling, gate:S.gate};
  const common={view:S.view, idx:S.idx, slab:S.slab, label:S.erase?0:S.label, rot:S.rot, ...band};
  let url, body;
  if(S.tool==="wand"){
    url="/wand"; body={...common, row:img[0][0], col:img[0][1], radius_mm:S.wandR};
  } else if(S.tool==="lasso"){
    url="/lasso"; body={...common, points:img, depth:S.depth};
  } else {
    url="/stroke"; body={...common, points:img, depth:S.depth, radius_mm:S.brush};
  }
  S.pts=[]; redraw();
  const res=await jpost(url,body);
  if(!res || !res.ok) return;
  if(res.changed>0 && S.tool!=="brush"){
    // Lasso and wand are BULK: one gesture can land thousands of voxels, and "did that just fill
    // the whole rib?" needs answering before you notice it three slices later. The brush stays
    // silent — a toast on every stroke would be noise.
    toast(`${S.tool}: ${res.changed.toLocaleString()} voxels${S.erase?" erased":""} — Undo (u) reverses it`);
    S.saved="unsaved"; S.dirty=true; S.gen++; refreshOverlay(); updateHud(); return;
  }
  if(res.changed===0 && !S.erase){
    // Fail loud. A gated tool paints NOTHING when no voxel under it is inside [floor,ceiling]
    // — which is now the normal, correct outcome of a band that is doing its job, and used to
    // be indistinguishable from "the app is broken". Say which control to move.
    toast(res.note ? res.note
         : S.gate ? `nothing in [${S.floor}, ${S.ceiling==null?"open":S.ceiling}] HU here `+
                    `— widen the band, turn the HU gate Off, or press “Show gate” to see it`
                  : "nothing to paint here");
  } else {
    S.saved="unsaved"; S.dirty=true; S.gen++; refreshOverlay(); updateHud();
  }
}
cv.addEventListener("pointerup",endStroke); cv.addEventListener("pointercancel",endStroke);

// ---- controls ----
$("viewsel").addEventListener("click",e=>{ const v=e.target.dataset.v; if(!v)return;
  [...$("viewsel").children].forEach(c=>c.classList.toggle("on",c.dataset.v===v));
  S.view=v; loadView(); });
// 90° on-screen turns (display only): re-fetch rotated dims/marks/tiles via the view-switch path.
// Strokes are sent as-drawn plus `rot`; the server un-rotates them, so ink lands on the same voxel.
$("rotate").onclick=()=>{ S.rot=(S.rot+1)%4; cacheClear(); loadView();
  toast(S.rot?`rotated ${S.rot*90}°`:"rotation reset"); };
function buildLabels(){ const box=$("labs"); box.innerHTML="";
  const defs=[[1,"CC","#5aa469","1"],[2,"TD","#3a8f9c","2"],[3,"lymph","#e0a33a","3"],[4,"bone","#b5916a","4"]];
  for(const [id,nm,col,hot] of defs){ const d=document.createElement("div");
    d.className="lab"+(id===S.label?" on":""); d.dataset.id=id;
    // eye toggles DISPLAY of this label; the rest of the chip selects it for painting
    d.innerHTML=`<span class="eye" data-eye="${id}"></span>`+
      `<span class="sw" style="background:${col}"></span>${nm}<span class="hot">${hot}</span>`;
    d.onclick=(ev)=>{ if(ev.target.dataset.eye){ toggleVis(id); return; }
      S.label=id; S.erase=false; S.vis.add(id);           // selecting a label shows it
      syncLabels(); refreshOverlay(); };
    box.appendChild(d); }
  syncLabels();
}
function toggleVis(id){ if(S.vis.has(id)) S.vis.delete(id); else S.vis.add(id);
  syncLabels(); refreshOverlay(); }
function syncLabels(){ [...$("labs").children].forEach(c=>{ const id=+c.dataset.id;
    c.classList.toggle("on", id===S.label && !S.erase);
    const eye=c.querySelector(".eye"); if(eye){ const shown=S.vis.has(id);
      eye.textContent = shown ? "\u25c9" : "\u25cb";        // ◉ shown / ○ hidden
      eye.classList.toggle("off", !shown); } });
  $("erase").classList.toggle("on",S.erase); }
$("erase").onclick=()=>{ S.erase=!S.erase; syncLabels(); };
// oninput fires continuously during a drag -> update live (no prefetch, to keep the pool free).
$("idx").oninput=e=>{ setIdx(+e.target.value, false); };
// onchange fires once on RELEASE -> authoritative re-render of the final slice + prefetch
// neighbors. This guarantees the slice you land on is the one that ends up on screen, even if
// intermediate drag fetches were aborted (the "wrong/stale slice" fix).
$("idx").onchange=e=>{ setIdx(+e.target.value, true); };
// Exact one-slice stepping (the slider alone jumps several slices per touch on a long volume).
$("idxdn").onclick=()=>setIdx(S.idx-1, true);
$("idxup").onclick=()=>setIdx(S.idx+1, true);
// slab now drives BOTH layers: the CT projection AND the overlay's coverage ghost -> refresh together.
$("slab").oninput=e=>{ S.slab=+e.target.value; $("slabv").textContent=S.slab; gateStat(); refreshBoth(); };
$("brush").oninput=e=>{ S.brush=+e.target.value; $("brushv").textContent=S.brush; };
$("zoom").oninput=e=>{ setZoom(+e.target.value); };
// ---- view transform (the zoom test extracts this block verbatim from PAGE) ----
// Procreate's rule: the pixel under your fingers stays under your fingers. transform-origin is
// the canvas top-left, so a content point u sits at ox+panX+u*zoom — scaling ALONE therefore
// blows the image out from its top-left corner and shoves the duct off-screen. Pinning an
// anchor is the whole fix; pan is derived from the anchor, never accumulated separately.
const ZMIN=1, ZMAX=8, KEEP=80;
// Scale to z about `from`, then land that same anchor on `to`. from!==to => pinch and drag in one
// continuous gesture. The anchor is re-derived from live state on every move, so hitting a zoom or
// pan limit just stops the motion instead of letting the image drift out from under the fingers.
function zoomPan(z, fromX,fromY, toX,toY){
  z=Math.max(ZMIN,Math.min(ZMAX,z));
  const u=(fromX-S.ox-S.panX)/S.zoom, v=(fromY-S.oy-S.panY)/S.zoom;   // anchor, logical canvas px
  S.zoom=z; S.panX=toX-S.ox-u*z; S.panY=toY-S.oy-v*z;
  clampPan(); applyTransform();
  $("zoom").value=z; $("zoomv").textContent=z.toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
}
// Fully zoomed out is the reset view: dead-centre. Rounded, because ox is itself a rounded half
// difference — so this lands on exactly pan=0 and never a half-pixel leftover, which would knock
// the canvas off the device grid and soften the backdrop the retina backing store keeps crisp.
//
// Zoomed in, the canvas is YOURS to place, like Procreate's. Do NOT force it to cover the stage:
// a coronal is far narrower than a landscape stage until well past 1.5x, and a cover rule pins X
// to the centre for that whole range — which silently overrides the anchor (so zoom stops
// following your fingers) AND makes a sideways drag do nothing, so the structure on the right is
// simply unreachable. The only rule is that you can't lose the image off-screen entirely.
function clampAxis(pan, base, size, avail){
  const box=size*S.zoom;
  if(S.zoom<=ZMIN) return Math.round((avail-box)/2)-base;   // fully out = reset, centred
  return Math.max(KEEP-base-box, Math.min(avail-KEEP-base, pan));   // else: at least KEEP px in view
}
function clampPan(){
  S.panX=clampAxis(S.panX,S.ox,S.W,stage.clientWidth);
  S.panY=clampAxis(S.panY,S.oy,S.H,stage.clientHeight);
}
// ---- end view transform ----
function setZoom(z){ const cx=stage.clientWidth/2, cy=stage.clientHeight/2;
  zoomPan(z, cx,cy, cx,cy); }                            // slider: keep the middle of the view put
$("win").onchange=e=>{ S.win=e.target.value; refreshBack(); };
// Switching FRAME or ACQUISITION reloads the seg from disk, discarding unsaved in-memory
// edits. Guard both: if dirty, confirm first; if the user cancels, snap the <select> back.
function confirmDiscard(){ return !S.dirty ||
  confirm("You have unsaved edits. Switching will discard them. Continue?\n(Cancel, then press Save first.)"); }
$("frame").onchange=async e=>{ if(!confirmDiscard()){ e.target.value=S.frame; return; }
  S.frame=+e.target.value; await jpost("/frame",{frame:S.frame});
  S.dirty=false; S.gen++; S.saved="unsaved"; loadView(); };
$("acq").onchange=async e=>{ if(!confirmDiscard()){ e.target.value=S.acq; return; }
  const ai=+e.target.value;
  const r=await jpost("/acq",{acq:ai});                  // server switches acquisition + reloads
  S.acq=ai; S.shape=r.shape; S.frame=r.frame; S.dirty=false;
  fillFrames(r.frames, r.frame, r.edited);                         // frame list can differ between acqs
  S.saved=r.resumed?"resumed sidecar":"unsaved";
  _forceDefault=true;                                               // let loadView pick the duct-median slice
  toast("loaded "+r.acq_name); S.gen++; loadView(); };
$("undo").onclick=async()=>{ const r=await jpost("/undo",{}); toast(r.n?`undo ${r.n}`:"nothing to undo");
  if(r.n){ S.dirty=true; S.saved="unsaved"; } S.gen++; refreshOverlay(); updateHud(); };
$("clear").onclick=async()=>{                            // wipe ALL labels (whole volume); confirm first
  if(!confirm("Clear ALL labels in every slice? (Undo brings them back; nothing is saved until you press Save.)")) return;
  const r=await jpost("/clear",{}); toast(r.n?`cleared ${r.n} vox — press Undo to restore`:"already empty");
  if(r.n){ S.dirty=true; S.saved="unsaved"; } S.gen++; refreshOverlay(); updateHud(); };
$("depthsel").addEventListener("click",e=>{ const d=e.target.dataset.d; if(!d)return;
  [...$("depthsel").children].forEach(c=>c.classList.toggle("on",c.dataset.d===d));
  S.depth=d;                                             // off | snap | grow
  toast(d==="off"?"depth-follow off (flat single slice)":
        d==="snap"?"snap: stroke jumps to the brightest in-band slice in the slab":
                   "grow: paints the in-band structure across the slices it spans"); });
// ---- tool: brush / lasso / wand ----
$("toolsel").addEventListener("click",e=>{ const t=e.target.dataset.t; if(!t)return;
  [...$("toolsel").children].forEach(c=>c.classList.toggle("on",c.dataset.t===t));
  S.tool=t; $("wandbox").style.display=(t==="wand")?"":"none";
  toast(t==="brush"?"brush: trace it":
        t==="lasso"?"lasso: circle it loosely — only in-band voxels inside the loop are filled":
                    "wand: tap the duct — the connected in-band blob within the reach is filled"); });
// ---- the HU gate ----
// CEIL_OPEN is the slider's top stop = "no ceiling"; every value below it is a real ceiling.
// It has now been wrong twice for the same reason -- the stop sat BELOW the duct, so the only
// settable finite ceilings excluded the duct itself and the stroke painted nothing:
//   400  -> a 450 HU contrast duct was already above it.
//   2000 -> the Lipiodol studies run far higher. Measured over the hand masks: duct p99 reaches
//           3368 HU (07/20/22 Angiotensin) and single voxels reach 7850, so 2000 could not keep
//           the duct in band at all on those acquisitions.
// 4000 clears p99 on every acquisition in the set while staying below nothing that matters.
const CEIL_OPEN=4000;
// The three "HU target" presets that used to live here (Bright / Bright-no-bone / Near-water) are
// gone on request: they were three arbitrary points in a two-slider space, they hid the fact that
// the band was not actually restricting anything, and every real frame needed the sliders anyway.
$("gatesel").addEventListener("click",e=>{ const g=e.target.dataset.g; if(!g)return;
  [...$("gatesel").children].forEach(c=>c.classList.toggle("on",c.dataset.g===g));
  S.gate=(g==="1"); S.gateTick++; syncBand(); refreshOverlay();
  toast(S.gate?"HU gate ON — only voxels inside [floor, ceiling] are painted"
              :"HU gate OFF — every voxel you touch is painted"); });
function syncBand(){
  $("floorv").textContent=S.floor;
  $("ceilv").textContent=(S.ceiling==null?"open":S.ceiling);
  $("bandbox").style.opacity=S.gate?1:0.45;
  gateStat();
}
// How much of THIS slab the band actually lets through. The single most useful number here: a
// gate that opens 60% of the field is not a gate, and that is precisely the state that made the
// old band look broken. Debounced — the sliders fire continuously while dragging.
let _gsT=null, _gsSeq=0;
function gateStat(){
  const el=$("gatehint");
  if(!S.gate){ el.textContent="gate off — painting is unrestricted"; el.className="warn"; return; }
  clearTimeout(_gsT);
  const mine=++_gsSeq;                    // debouncing orders the REQUESTS, not the responses:
  el.textContent="measuring the gate…";   // an older one landing last would post a wrong number.
  el.className="";                        // Blank it meanwhile: while the server is busy loading
                                          // a frame the answer can take a second, and a stale
                                          // percentage beside new slider values reads as truth.
  _gsT=setTimeout(async()=>{
    const r=await jget(`/gatestat?view=${S.view}&idx=${S.idx}&slab=${S.slab}`+
                       `&floor=${S.floor}&ceil=${S.ceiling==null?"":S.ceiling}`);
    if(mine!==_gsSeq || !r || r.frac==null) return;
    const pct=(100*r.frac).toFixed(1);
    el.textContent=`gate opens ${pct}% of this slab`+(r.frac>0.25?" — too loose to restrict much":"");
    el.className=(r.frac>0.25)?"warn":"";
  },180);
}
function setFloor(v){ v=Math.max(-200,Math.min(800,v)); S.floor=v; $("floor").value=v;
  S.gateTick++; syncBand(); if(S.showGate) refreshOverlay(); }
function setCeil(v){ v=Math.max(0,Math.min(CEIL_OPEN,v)); $("ceil").value=v;
  S.ceiling=(v>=CEIL_OPEN?null:v); S.gateTick++; syncBand(); if(S.showGate) refreshOverlay(); }
$("floor").oninput=e=>setFloor(+e.target.value);
$("ceil").oninput=e=>setCeil(+e.target.value);
$("floordn").onclick=()=>setFloor(S.floor-5); $("floorup").onclick=()=>setFloor(S.floor+5);
$("ceildn").onclick=()=>setCeil((S.ceiling==null?CEIL_OPEN:S.ceiling)-5);
$("ceilup").onclick=()=>setCeil((S.ceiling==null?CEIL_OPEN:S.ceiling)+5);
$("showgate").onclick=()=>{ S.showGate=!S.showGate;
  $("showgate").classList.toggle("on",S.showGate); S.gen++; refreshOverlay();
  toast(S.showGate?"blue wash = every voxel the band allows":"gate preview off"); };
$("wandr").oninput=e=>{ S.wandR=+e.target.value; $("wandrv").textContent=S.wandR; };
$("redo").onclick=async()=>{ const r=await jpost("/redo",{}); toast(r.n?`redo ${r.n}`:"nothing to redo");
  if(r.n){ S.dirty=true; S.saved="unsaved"; } S.gen++; refreshOverlay(); updateHud(); };
$("save").onclick=async()=>{ const r=await jpost("/save",{});
  // fetch() does NOT reject on 4xx/5xx, so a refused save arrives here as a normal body. Without
  // this guard the HUD read "saved undefined vox" and S.dirty was cleared on a save the server had
  // rejected, quietly disarming the unsaved-work warning.
  if(!r || r.ok===false){ toast(r&&r.error ? r.error : "save FAILED - your edits are still unsaved");
                          S.saved="SAVE FAILED"; updateHud(); return; }
  S.saved="saved "+r.voxels+" vox"; S.dirty=false;
  // diff_seed===0 means the file you just wrote is byte-for-byte the builder's automatic seed. It will
  // still be read downstream as a manual mask, so say so loudly rather than letting it pass as work.
  toast(r.diff_seed===0 ? "SAVED UNCHANGED — identical to the auto-seed, not a hand mask"
                        : "saved "+r.file);
  if(r.edited) fillFrames(S.frames, S.frame, r.edited);   // this frame is now "✎ edited"
  updateHud(); };
document.addEventListener("keydown",e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==="s"){ e.preventDefault(); $("save").click(); return; }
  if((e.metaKey||e.ctrlKey)&&(e.key==="z"||e.key==="Z")){    // ⌘Z undo, ⌘⇧Z redo
    e.preventDefault(); (e.shiftKey?$("redo"):$("undo")).click(); return; }
  if(e.key>="1"&&e.key<="4"){ S.label=+e.key; S.erase=false; S.vis.add(+e.key); syncLabels(); refreshOverlay(); }
  else if(e.key==="e"){ S.erase=!S.erase; syncLabels(); }
  else if(e.key==="u"){ $("undo").click(); }
  else if(e.key==="r"){ $("redo").click(); }
  else if(e.key==="b"||e.key==="l"||e.key==="w"){        // tool: brush / lasso / wand
    $("toolsel").querySelector(`[data-t="${{b:"brush",l:"lasso",w:"wand"}[e.key]}"]`).click(); }
  else if(e.key==="g"){ $("gatesel").querySelector(`[data-g="${S.gate?0:1}"]`).click(); }
  else if(e.key==="h"){ $("showgate").click(); }
  else if(e.key==="["){ $("slab").value=Math.max(0,S.slab-2); $("slab").oninput({target:$("slab")}); }
  else if(e.key==="]"){ $("slab").value=Math.min(24,S.slab+2); $("slab").oninput({target:$("slab")}); }
  else if(e.key==="ArrowDown"||e.key==="ArrowLeft"){ e.preventDefault(); setIdx(S.idx-1, true); }
  else if(e.key==="ArrowUp"||e.key==="ArrowRight"){ e.preventDefault(); setIdx(S.idx+1, true); }
});
// Finger gestures on the stage: TWO fingers = pinch-zoom + pan. ONE finger does NOTHING
// on purpose — it used to scroll slices, which is what changed the slice while you drew
// (a resting palm/finger counts as a one-finger touch). Slices now change ONLY via the
// slider. And while a Pencil stroke is active we ignore touch entirely (palm rejection),
// so resting your hand on the screen can never move or zoom anything.
// NEVER latch a gesture mode. iOS Safari fires touchcancel in the middle of a live pinch (it
// decides to claim the gesture) and can transiently report <2 touches while both fingers are
// still on the glass. A latched mode gets cleared by those and then never re-arms — no new
// touchstart is coming, because you never lifted your fingers — so the pinch dies outright and
// stays dead. e.touches is the truth; read it every move and the gesture cannot get stuck.
let g={n:0, fx:0, fy:0, dist:0};
function tdist(t){ const dx=t[0].clientX-t[1].clientX, dy=t[0].clientY-t[1].clientY; return Math.hypot(dx,dy); }
function tmid(t){ return [(t[0].clientX+t[1].clientX)/2,(t[0].clientY+t[1].clientY)/2]; }
function endGesture(){ g.n=0; }                          // fingers gone: the next pinch re-baselines
stage.addEventListener("touchend",endGesture,{passive:true});
stage.addEventListener("touchcancel",endGesture,{passive:true});
// iOS Safari has IGNORED user-scalable=no since iOS 10, and claims two-finger gestures for its own
// page zoom. touch-action:none does not stop that, and a passive listener cannot either, because
// preventDefault() is a silent no-op in one. These WebKit-only gesture events are the sole lever —
// so they, and the touchmove below, must be non-passive. Desktop never exposed this: the wheel
// handler was already non-passive, which is exactly why the trackpad pinch worked while the iPad
// sat there doing nothing. Safari was eating the gesture before any of our code ran.
for(const gev of ["gesturestart","gesturechange","gestureend"])
  document.addEventListener(gev, e=>e.preventDefault(), {passive:false});
stage.addEventListener("touchmove",e=>{
  if(S.drawing){ g.n=0; return; }                        // palm rejection during a Pencil stroke
  const t=e.touches;
  if(t.length<2){ g.n=0; return; }                       // one finger: intentionally inert
  e.preventDefault();                                    // two fingers are OURS, not Safari's
  const d=tdist(t), m=tmid(t), r=stage.getBoundingClientRect();
  // A finger joined or left (or this is the first frame): re-baseline off the new pair and skip
  // one frame, rather than reading the jumped midpoint against a stale baseline and kicking the
  // canvas sideways. Same reason Konva's demo does `if(!lastCenter){lastCenter=...; return;}`.
  if(g.n!==t.length){ g.n=t.length; g.dist=d; g.fx=m[0]; g.fy=m[1]; return; }
  // Scale by how far the fingers spread, anchored where they WERE, landing where they ARE:
  // pinch and pan are one gesture, and the anatomy under the pinch never leaves it.
  zoomPan(S.zoom*d/Math.max(g.dist,1), g.fx-r.left, g.fy-r.top, m[0]-r.left, m[1]-r.top);
  g.dist=d; g.fx=m[0]; g.fy=m[1];
},{passive:false});
// Add ?debug=1 to the URL to read the live gesture straight off the screen — no cable, no Web
// Inspector. Pinch, and the toast shows how many touches actually reached us and what the
// transform did. NOTHING appearing means the events never arrive and Safari still owns the gesture.
if(new URLSearchParams(location.search).has("debug"))
  stage.addEventListener("touchmove",e=>toast(
    `n=${e.touches.length} z=${S.zoom.toFixed(2)} pan=${S.panX|0},${S.panY|0}`),{passive:true});
// trackpad / wheel (desktop): ctrl+wheel (pinch) = zoom only. Plain wheel does NOT change
// slices — slices change only via the slider, as requested. (preventDefault stops page bounce.)
stage.addEventListener("wheel",e=>{ e.preventDefault();
  if(e.ctrlKey){ const r=stage.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
    zoomPan(S.zoom*(1-e.deltaY*0.01), x,y, x,y); }      // about the cursor, same rule as the pinch
},{passive:false});
window.addEventListener("resize",layout);
// The stage can get its real size AFTER boot's first layout() (flex / iPad safe-area settle late), and
// orientation flips resize it too - cases a window 'resize' can miss. A ResizeObserver on the stage
// re-runs layout the instant the size is known or changes, so the fit is deterministic, not a boot race.
new ResizeObserver(layout).observe(stage);

// edited[i] = that frame has a saved f<N>_edit.npz (YOUR manual seg), which this tool and
// io.context.load_seg both auto-read in preference to the built-in seg. Mark it, so you can see at a
// glance which frames are already yours. (The sidecars are NOT frames — see _resolve_frames.)
function fillFrames(frames, cur, edited){
  S.frames=frames; S.edited=edited||[];              // kept so a save can re-mark without a reload
  const fsel=$("frame"); fsel.innerHTML="";
  frames.forEach((nm,i)=>{ const o=document.createElement("option"); o.value=i;
    o.textContent="frame "+nm+(S.edited[i]?"  ✎ edited":"");
    if(i===cur)o.selected=true; fsel.appendChild(o); });
  fsel.style.display = frames.length<2 ? "none" : "";
}
function fillAcqs(acqs, cur){
  const asel=$("acq"); asel.innerHTML="";
  // group by date with <optgroup> so the dropdown reads date -> Baseline/Angiotensin -> Acq
  let group=null, lastDate=null;
  acqs.forEach((a,i)=>{ if(a.date!==lastDate){ group=document.createElement("optgroup");
      group.label=a.date||"(other)"; asel.appendChild(group); lastDate=a.date; }
    const o=document.createElement("option"); o.value=i; o.textContent=a.label;
    if(i===cur)o.selected=true; group.appendChild(o); });
  asel.style.display = acqs.length<2 ? "none" : "";
}
async function boot(){
  const init=await jget("/init");
  S.shape=init.shape;
  fillAcqs(init.acqs, init.acq);
  fillFrames(init.frames, init.frame, init.edited);
  S.acq=init.acq; S.frame=init.frame; S.slab=init.slab; $("slab").value=S.slab; $("slabv").textContent=S.slab;
  $("brushv").textContent=S.brush; $("zoomv").textContent="1";
  $("wandrv").textContent=S.wandR; $("wandbox").style.display="none";   // wand controls: on demand
  $("floor").value=S.floor; $("ceil").value=(S.ceiling==null?CEIL_OPEN:S.ceiling); syncBand();
  S.saved=init.resumed?"resumed sidecar":"unsaved";
  buildLabels();
  await loadView();
}
boot();
</script></body></html>"""


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    sess: Session = None                       # set on the server instance

    def log_message(self, *a):                 # quiet: one line per stroke is enough
        pass

    def _send(self, code, body, ctype="application/json", cache="no-store"):
        if isinstance(body, (dict, list)):
            # default= is not optional: numpy scalars (np.int64, np.bool_) are NOT instances of
            # int/bool, so a stray one anywhere in the body raises TypeError here -- AFTER the
            # work succeeded but BEFORE send_response, turning a good save into a bare 500.
            body = json.dumps(body, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # iPad Safari cancels superseded image requests mid-flight (drag a slider
            # and the previous /backdrop socket closes before we finish writing). That
            # is expected and harmless — swallow it so it doesn't spam the console, and
            # flag the dead socket so the caller won't try to write a 500 on top of it.
            self._client_gone = True

    def _q(self):
        from urllib.parse import urlparse, parse_qs
        # keep_blank_values so an empty `vis=` (hide ALL labels) is distinct from vis absent
        return {k: v[0] for k, v in
                parse_qs(urlparse(self.path).query, keep_blank_values=True).items()}

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # -- GET --
    def do_GET(self):
        s = self.server.sess
        p = self.path.split("?", 1)[0]
        try:
            if p == "/" or p == "/index.html":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if p == "/init":
                return self._send(200, {
                    "shape": list(s.hu.shape), "voxel": list(s.voxel),
                    "frames": [lbl for lbl, _ in s.frames], "frame": s.fi,
                    "edited": s.frames_edited(),          # which frames already have your saved seg
                    "slab": 8, "resumed": bool(s.resumed),
                    "acqs": [{"label": a["label"], "date": a["date"]} for a in s.acqs],
                    "acq": s.ai, "acq_name": s.root.name})
            if p == "/meta":
                q = self._q(); view = q.get("view", "coronal")
                rot = int(q.get("rot", 0))
                rows, cols = rot_dims(view, s.hu.shape, rot)
                mm_r, mm_c = inplane_mm(view, s.voxel)
                if rot % 2:                              # odd turn swaps the display axes
                    mm_r, mm_c = mm_c, mm_r
                ax = AXIS[view]
                return self._send(200, {
                    "rows": rows, "cols": cols,
                    "h_mm": rows * mm_r, "w_mm": cols * mm_c,
                    "marks": list(rot_marks(view, rot)),
                    "nidx": s.hu.shape[ax], "default_idx": s.idx[view],
                    "sig": s.data_sig()})
            if p == "/counts":
                return self._send(200, s.counts())
            if p == "/gatestat":
                # What fraction of this slab the band actually admits. The number that tells you
                # at a glance whether a threshold is restricting anything -- a "gate" that opens
                # 60% of the field is not one, and reading that off the panel beats discovering
                # it by painting.
                q = self._q()
                ce = q.get("ceil", "")
                lo, hi = band_bounds(float(q.get("floor", 120.0)),
                                     float(ce) if ce != "" else None)
                with s.lock:
                    plane = gate_plane(s.hu, q.get("view", "coronal"), int(q.get("idx", 0)),
                                       int(q.get("slab", 0)), lo, hi)
                return self._send(200, {"frac": float(plane.mean()) if plane.size else 0.0})
            if p == "/backdrop":
                q = self._q()
                lvl, win = WL_PRESETS.get(q.get("win", "contrast"), WL_PRESETS["contrast"])
                jpg = render_backdrop_jpeg(s, q["view"], int(q["idx"]), int(q["slab"]), lvl, win,
                                           rot=int(q.get("rot", 0)))
                # Cacheable (max-age) so re-scrolling to a slice reuses the tile. The client URL
                # carries a per-acq/build `sig=` token (backUrl / Session.data_sig): the HU for a
                # given (view,idx,slab,win,frame) DOES change across an acq switch or a rebuild at
                # the same host:port, so sig keeps those in separate browser-cache namespaces -
                # without it Safari served a stale different-acq tile for a byte-identical URL (the
                # "jumping" bug). sig is ignored when rendering (RAM already holds the right acq);
                # it only varies the cache key. (Overlay carries `g=` + no-store.)
                return self._send(200, jpg, "image/jpeg", cache="max-age=86400")
            if p == "/overlay":
                q = self._q()
                vis = q.get("vis")
                visible = [int(x) for x in vis.split(",") if x != ""] if vis is not None else None
                # gfloor/gceil present -> draw the HU-gate wash. gceil="" means "open top".
                gate = None
                if "gfloor" in q:
                    gc = q.get("gceil", "")
                    gate = (float(q["gfloor"]), float(gc) if gc != "" else None)
                png = render_overlay_png(s, q["view"], int(q["idx"]), visible=visible,
                                         rot=int(q.get("rot", 0)), slab=int(q.get("slab", 0)),
                                         gate=gate)
                return self._send(200, png, "image/png", cache="no-store")
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._client_gone = True                     # client hung up; nothing to report
        except Exception as e:
            if not getattr(self, "_client_gone", False):
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    # -- POST --
    def do_POST(self):
        s = self.server.sess
        p = self.path.split("?", 1)[0]
        try:
            b = self._body()
            if p in ("/stroke", "/lasso", "/wand"):
                # One band for all three tools, un-rotated the same way: the client draws in the
                # rotated display, the paint core only ever sees unrotated display space.
                rot = int(b.get("rot", 0))
                H0, W0 = disp_shape(b["view"], s.hu.shape)      # unrotated display the paint core expects
                ceil = b.get("ceiling", None)
                band = dict(floor=float(b.get("floor", 120.0)),
                            ceiling=(float(ceil) if ceil is not None else None),
                            gate=bool(b.get("gate", True)))
                view, idx, label = b["view"], int(b["idx"]), int(b["label"])
                slab = int(b.get("slab", 0))
                note = ""
                if p == "/wand":
                    r, c = unrotate_rc(float(b["row"]), float(b["col"]), rot, H0, W0)
                    n, note = s.apply_wand(view, idx, r, c, label, slab,
                                           radius_mm=float(b.get("radius_mm", 12.0)), **band)
                else:
                    pts = [unrotate_rc(float(r), float(c), rot, H0, W0) for r, c in b["points"]]
                    if p == "/lasso":
                        n = s.apply_lasso(view, idx, pts, label, slab,
                                          depth=b.get("depth", "off"), **band)
                    else:
                        n = s.apply_stroke(view, idx, pts, float(b["radius_mm"]), label, slab,
                                           depth=b.get("depth", "off"), **band)
                return self._send(200, {"ok": True, "changed": n, "note": note})
            if p == "/undo":
                return self._send(200, {"ok": True, "n": s.undo_last()})
            if p == "/redo":
                return self._send(200, {"ok": True, "n": s.redo_last()})
            if p == "/clear":
                return self._send(200, {"ok": True, "n": s.clear_all()})
            if p == "/frame":
                s.set_frame(int(b["frame"]))
                return self._send(200, {"ok": True, "frame": s.fi})
            if p == "/acq":
                a = s.set_acq(int(b["acq"]))
                return self._send(200, {
                    "ok": True, "acq": s.ai, "acq_name": a["name"],
                    "shape": list(s.hu.shape), "voxel": list(s.voxel),
                    "frames": [lbl for lbl, _ in s.frames], "frame": s.fi,
                    "edited": s.frames_edited(),
                    "resumed": bool(s.resumed)})
            if p == "/save":
                path, vox, n_diff = s.save()
                if path is None:                 # frame/acq switched mid-save; nothing was written
                    return self._send(409, {"ok": False, "error":
                                            "the frame changed while saving - nothing was written, "
                                            "press Save again"})
                # return the refreshed edited-flags so the picker marks the frame you just saved
                return self._send(200, {"ok": True, "file": str(path), "voxels": vox,
                                        "diff_seed": n_diff,
                                        "edited": s.frames_edited()})
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._client_gone = True                     # client hung up; nothing to report
        except Exception as e:
            if not getattr(self, "_client_gone", False):
                self._send(500, {"error": f"{type(e).__name__}: {e}"})


def _lan_ip():
    """Best-effort primary LAN IP (the address the iPad should hit)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packet sent; just picks the egress iface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def serve(path, port=8000, host="0.0.0.0"):
    sess = Session(path)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.sess = sess
    ip = _lan_ip()
    nfr = len(sess.frames)
    print("=" * 60)
    print(f"  CC/TD iPad paint — {sess.root.name}")
    print(f"  volume {sess.hu.shape}  voxel {sess.voxel} mm  "
          f"{nfr} frame{'s' if nfr != 1 else ''}")
    print(f"  sidecar -> {sess.sidecar}")
    print("-" * 60)
    print(f"  On the iPad (same WiFi), open Safari to:")
    print(f"      http://{ip}:{port}/")
    print(f"  (Mac-local test: http://127.0.0.1:{port}/ )")
    print(f"  Ctrl-C to stop.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ipad_paint] stopped.")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Self-test (no server) — the geometry, brush, and back-projection math
# --------------------------------------------------------------------------- #
def _selftest():
    shape = (236, 256, 1401)
    vox = (1.562, 1.562, 0.5)

    # display shapes
    assert disp_shape("sagittal", shape) == (1401, 236)
    assert disp_shape("coronal", shape) == (1401, 256)
    assert disp_shape("axial", shape) == (236, 256)

    # disp<->voxel round-trip, every view
    rng = np.random.default_rng(0)
    for v in ("sagittal", "coronal", "axial"):
        H, W = disp_shape(v, shape)
        idx = 7
        for _ in range(3000):
            r = int(rng.integers(0, H)); c = int(rng.integers(0, W))
            x, y, z = disp_to_voxel(v, r, c, idx, shape)
            assert 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]
            assert voxel_to_disp(v, x, y, z, shape) == (r, c)

    # cranial-up semantics: cranial(max z) top, caudal(0) bottom; anterior(x=0) left
    m = np.zeros(shape, np.int16); m[0, 128, shape[2] - 1] = 7
    d = to_display(m, "sagittal", 128, 0)
    assert tuple(np.argwhere(d == 7)[0]) == (0, 0), "cranial+anterior -> top-left"
    m2 = np.zeros(shape, np.int16); m2[0, 128, 0] = 7
    assert np.argwhere(to_display(m2, "sagittal", 128, 0) == 7)[0][0] == shape[2] - 1, "caudal -> bottom"

    # brush: physical-mm disk, correct anisotropic extents, on the right slice
    seg = np.zeros(shape, np.uint8)
    n = paint_stroke(seg, shape, "sagittal", 136, [(700, 118)], 3.0, vox, 1)
    pv = np.argwhere(seg == 1)
    assert (pv[:, 1] == 136).all(), "brush stays on slice y=136"
    z_ext = np.ptp(pv[:, 2]) + 1; x_ext = np.ptp(pv[:, 0]) + 1
    assert 10 <= z_ext <= 14 and 3 <= x_ext <= 5, (z_ext, x_ext)

    # erase + undo record
    rec = []
    paint_stroke(seg, shape, "sagittal", 136, [(700, 118)], 1.0, vox, 0, record=rec)
    assert rec and (seg == 1).sum() < n, "erase removed voxels and recorded them"

    # gap-free ribbon along a stroke
    seg2 = np.zeros(shape, np.uint8)
    paint_stroke(seg2, shape, "sagittal", 136, [(600, 118), (650, 118), (700, 118)], 1.5, vox, 2)
    zs = np.unique(np.argwhere(seg2 == 2)[:, 2])
    assert (np.diff(zs) <= 1).all(), "TD ribbon gap-free"

    # back-projection snaps to brightest slice (sagittal + coronal), slab-0 no-op
    hu = np.zeros(shape, np.int16); s3 = np.zeros(shape, np.uint8)
    hu[130, 140, 720] = 1000; s3[130, 136, 720] = 1
    assert backproject(hu, s3, "sagittal", 136, 8, 1) == 1
    assert s3[130, 140, 720] == 1 and s3[130, 136, 720] == 0, "sagittal snap"
    hu2 = np.zeros(shape, np.int16); s4 = np.zeros(shape, np.uint8)
    hu2[150, 120, 800] = 1000; s4[145, 120, 800] = 2
    assert backproject(hu2, s4, "coronal", 145, 10, 2) == 1
    assert s4[150, 120, 800] == 2 and s4[145, 120, 800] == 0, "coronal snap"
    assert backproject(hu, s3, "sagittal", 140, 0, 1) == 0, "slab-0 no-op"

    # undo restores exactly
    seg5 = np.zeros(shape, np.uint8); rec5 = []
    paint_stroke(seg5, shape, "sagittal", 100, [(500, 118)], 3.0, vox, 1, record=rec5)
    before = int((seg5 == 1).sum())
    for (x, y, z, old) in reversed(rec5):
        seg5[x, y, z] = old
    assert (seg5 == 1).sum() == 0 and before > 0, "undo restores"

    # depth-aware MIP brush: a bright bar spanning several slices in depth
    hu3 = np.full(shape, -200, np.int16)                  # muscle-ish background
    hu3[128:133, 140, 720] = 600                          # duct: 5 slices deep at (x=128..132)
    # paint on the coronal slab centered a few slices off the bar; the brush is at (y,z)=(140,720)
    r0, c0 = voxel_to_disp("coronal", 130, 140, 720, shape)[:2]
    # SNAP -> single brightest slice, on the bar
    ss = np.zeros(shape, np.uint8)
    paint_stroke_depth(hu3, ss, "coronal", 135, [(r0, c0)], 0.5, vox, 12, 1, grow=False)
    xs_snap = np.unique(np.argwhere(ss == 1)[:, 0])
    assert len(xs_snap) == 1 and 128 <= xs_snap[0] <= 132, ("snap lands on the bar", xs_snap)
    # GROW -> the contiguous bright run (multiple slices)
    sg = np.zeros(shape, np.uint8)
    paint_stroke_depth(hu3, sg, "coronal", 135, [(r0, c0)], 0.5, vox, 12, 1, grow=True)
    xs_grow = np.unique(np.argwhere(sg == 1)[:, 0])
    assert len(xs_grow) >= 3 and set(xs_grow) <= set(range(128, 133)), ("grow spans the bar", xs_grow)
    # FLOOR gate: on a muscle-only column (peak below floor) nothing is painted
    sf = np.zeros(shape, np.uint8)
    rmc = voxel_to_disp("coronal", 130, 60, 300, shape)[:2]
    assert paint_stroke_depth(hu3, sf, "coronal", 130, [rmc], 0.5, vox, 12, 1,
                              floor=120.0, grow=True) == 0, "floor gate skips muscle"

    # HU-BAND (pre-contrast): a near-WATER duct DIMMER than nearby muscle. Snap-to-brightest
    # would grab the muscle; the band [floor,ceiling] with a ceiling below muscle picks the duct.
    hu4 = np.full(shape, -90, np.int16)                   # fat bed
    hu4[125, 145, 730] = 80                               # muscle strip (brighter) at x=125
    hu4[131, 145, 730] = 15                               # near-water duct at x=131
    r1, c1 = voxel_to_disp("coronal", 134, 145, 730, shape)[:2]   # drawn a few slices off
    sb = np.zeros(shape, np.uint8)
    paint_stroke_depth(hu4, sb, "coronal", 134, [(r1, c1)], 0.5, vox, 10, 1,
                       floor=-20.0, ceiling=45.0, grow=False)
    xs_band = np.unique(np.argwhere(sb == 1)[:, 0])
    assert list(xs_band) == [131], ("band picks water duct, not muscle", xs_band)
    sb2 = np.zeros(shape, np.uint8)                       # low floor, NO ceiling -> grabs muscle
    paint_stroke_depth(hu4, sb2, "coronal", 134, [(r1, c1)], 0.5, vox, 10, 1,
                       floor=-20.0, ceiling=None, grow=False)
    assert 125 in np.argwhere(sb2 == 1)[:, 0], "no-ceiling snap grabs the brighter muscle"

    # The same failure with BONE standing in for muscle, and the fix. A 450 HU duct next to a 900 HU
    # rib: open-top snaps to the rib; a ceiling between the two picks the duct. This is only reachable
    # from the UI because the ceiling slider now runs to 2000 -- at the old max of 400 the only
    # settable ceiling was 395, which is below the duct, so the stroke painted nothing at all.
    hu5 = np.full(shape, -60, np.int16)                   # soft tissue
    hu5[130, 150, 740] = 450                              # contrast-filled duct
    hu5[136, 150, 740] = 900                              # rib cortex, same column
    r2, c2 = voxel_to_disp("coronal", 133, 150, 740, shape)[:2]
    sb3 = np.zeros(shape, np.uint8)
    paint_stroke_depth(hu5, sb3, "coronal", 133, [(r2, c2)], 0.5, vox, 8, 1,
                       floor=120.0, ceiling=None, grow=False)
    assert list(np.unique(np.argwhere(sb3 == 1)[:, 0])) == [136], "open top snaps to the rib"
    sb4 = np.zeros(shape, np.uint8)
    paint_stroke_depth(hu5, sb4, "coronal", 133, [(r2, c2)], 0.5, vox, 8, 1,
                       floor=120.0, ceiling=700.0, grow=False)
    assert list(np.unique(np.argwhere(sb4 == 1)[:, 0])) == [130], "ceiling 700 picks the duct"
    sb5 = np.zeros(shape, np.uint8)                       # the old slider maximum
    assert paint_stroke_depth(hu5, sb5, "coronal", 133, [(r2, c2)], 0.5, vox, 8, 1,
                              floor=120.0, ceiling=395.0, grow=False) == 0, \
        "a 395 ceiling excludes the duct too -- the old control could not help"

    # ------------------------------------------------------------------ THE HU GATE ------
    # A small phantom in the painter's own geometry: coronal cuts x, display row = nz-1-z,
    # display col = y. Muscle bed, a bright "duct" bar, and a denser "rib" stripe beside it.
    gsh = (40, 40, 60)
    gvox = (1.0, 1.0, 1.0)
    ghu = np.full(gsh, 50, np.int16)          # muscle everywhere
    ghu[:, 18:23, 28:33] = 400                # duct   : y 18-22, z 28-32
    ghu[:, 26:29, 28:33] = 900                # rib    : y 26-28, z 28-32
    line = [(29, 10), (29, 35)]               # one horizontal stroke across both, at z=30

    # vectorised disp_to_voxel must be the same map as the scalar one, in every view
    for v in ("sagittal", "coronal", "axial"):
        Hh, Ww = disp_shape(v, gsh)
        rs = np.array([0, 1, Hh // 3, Hh - 1]); cs = np.array([Ww - 1, 0, Ww // 2, 3])
        ax_, ay_, az_ = disp_to_voxel_arr(v, rs, cs, 7, gsh)
        for k in range(len(rs)):
            assert (int(ax_[k]), int(ay_[k]), int(az_[k])) == \
                disp_to_voxel(v, int(rs[k]), int(cs[k]), 7, gsh), ("vectorised disp_to_voxel", v)

    # (1) THE BUG: without the volume there is nothing to gate against, so the flat brush paints
    #     everything it touches -- muscle included. This is exactly what the app used to do on
    #     EVERY flat stroke, whatever the threshold said.
    sfree = np.zeros(gsh, np.uint8)
    n_free = paint_stroke(sfree, gsh, "coronal", 20, line, 1.0, gvox, 1)
    assert n_free == 80, n_free
    assert ghu[np.nonzero(sfree == 1)].min() == 50, "ungated brush paints muscle (the old bug)"

    # (2) THE FIX: hand the brush the volume and the band, and only in-band voxels are written.
    sgate = np.zeros(gsh, np.uint8)
    n_gate = paint_stroke(sgate, gsh, "coronal", 20, line, 1.0, gvox, 1, hu=ghu, floor=120.0)
    got = ghu[np.nonzero(sgate == 1)]
    assert n_gate == 24 and set(np.unique(got)) == {400, 900}, (n_gate, np.unique(got))
    assert n_gate < n_free, "the gate must restrict, not merely re-order"

    # (3) the ceiling is what separates a 400 HU duct from 900 HU cortex under the same stroke
    sceil = np.zeros(gsh, np.uint8)
    n_ceil = paint_stroke(sceil, gsh, "coronal", 20, line, 1.0, gvox, 1,
                          hu=ghu, floor=120.0, ceiling=700.0)
    assert set(np.unique(ghu[np.nonzero(sceil == 1)])) == {400}, "ceiling 700 drops the rib"
    assert n_ceil == 15, n_ceil                      # duct only: 3 display rows x 5 columns

    # (4) gate=False is the old free-drawing behaviour, now a choice rather than an accident
    soff = np.zeros(gsh, np.uint8)
    assert paint_stroke(soff, gsh, "coronal", 20, line, 1.0, gvox, 1,
                        hu=ghu, floor=120.0, gate=False) == n_free, "gate off == ungated"

    # (5) erase is NEVER gated: you can always remove a label you can see, whatever the band
    n_er = paint_stroke(sfree, gsh, "coronal", 20, line, 1.0, gvox, 0,
                        hu=ghu, floor=120.0, ceiling=125.0)
    assert n_er == n_free and not (sfree == 1).any(), "erase ignores the band"

    # (6) depth-follow writes only in-band voxels too -- the gate is on the WRITE, not just on
    #     the choice of depth. Without the ceiling the same stroke lands on the rib.
    sdep = np.zeros(gsh, np.uint8)
    paint_stroke_depth(ghu, sdep, "coronal", 20, line, 1.0, gvox, 8, 1,
                       floor=120.0, ceiling=700.0, grow=True)
    assert set(np.unique(ghu[np.nonzero(sdep == 1)])) == {400}, "depth grow stays inside the band"

    # ------------------------------------------------------------------- POLYGON FILL -----
    rr, cc = polygon_pixels([(10, 10), (10, 20), (20, 20), (20, 10)], 40, 40)
    assert rr.size == 121 and rr.min() == 10 and rr.max() == 20 and cc.min() == 10, rr.size
    # concave: a C opening toward high columns. The notch must stay empty.
    cpoly = [(0, 0), (0, 20), (20, 20), (20, 12), (6, 12), (6, 8), (20, 8), (20, 0)]
    rr2, cc2 = polygon_pixels(cpoly, 40, 40)
    inside = set(zip(rr2.tolist(), cc2.tolist()))
    assert (10, 4) in inside and (10, 15) in inside, "the two arms of the C are filled"
    assert (10, 10) not in inside, "the notch of a concave lasso is NOT filled"
    # clipped to the display, and a degenerate lasso still marks what it touched
    rr3, cc3 = polygon_pixels([(-50, -50), (-50, 5), (5, 5), (5, -50)], 40, 40)
    assert rr3.size and rr3.min() >= 0 and cc3.min() >= 0 and rr3.max() <= 5
    assert polygon_pixels([(3, 4)], 40, 40)[0].tolist() == [3], "a single tap marks one pixel"
    assert polygon_pixels([], 40, 40)[0].size == 0

    # ------------------------------------------------------------------------- LASSO -----
    box = [(25, 14), (25, 27), (34, 27), (34, 14)]     # rows 25-34 (z 25-34), cols 14-27
    sl1 = np.zeros(gsh, np.uint8)
    n_l = lasso_fill(ghu, sl1, "coronal", 20, box, gvox, 0, 1,
                     depth="off", floor=120.0, ceiling=700.0)
    assert n_l == 25 and set(np.unique(ghu[np.nonzero(sl1 == 1)])) == {400}, n_l
    sl2 = np.zeros(gsh, np.uint8)                      # gate off -> the whole enclosed rectangle
    assert lasso_fill(ghu, sl2, "coronal", 20, box, gvox, 0, 1,
                      depth="off", gate=False) == 140
    n_le = lasso_fill(ghu, sl2, "coronal", 20, box, gvox, 0, 0, depth="off")   # lasso-erase
    assert n_le == 140 and not sl2.any(), "lasso erase clears everything enclosed"
    sl3 = np.zeros(gsh, np.uint8)                      # depth-follow lasso, drawn off the duct
    lasso_fill(ghu, sl3, "coronal", 12, box, gvox, 10, 1,
               depth="grow", floor=120.0, ceiling=700.0)
    assert set(np.unique(ghu[np.nonzero(sl3 == 1)])) == {400}, "grow lasso stays in the band"

    # --------------------------------------------------------------------------- WAND -----
    whu = np.full(gsh, -100, np.int16)
    whu[20, 20, 10:40] = 300                           # a 30-voxel duct running along z
    whu[30, 30, 20] = 300                              # a disconnected blob at the same HU
    ws = np.zeros(gsh, np.uint8)
    n_w, note = wand_fill(whu, ws, "coronal", 20, 60 - 1 - 25, 20, gvox, 1,
                          floor=120.0, radius_mm=100.0)
    assert note == "" and n_w == 30, (n_w, note)
    assert ws[30, 30, 20] == 0, "the wand fills the CONNECTED component, not every in-band voxel"
    ws2 = np.zeros(gsh, np.uint8)                      # the reach is a hard physical bound
    assert wand_fill(whu, ws2, "coronal", 20, 60 - 1 - 25, 20, gvox, 1,
                     floor=120.0, radius_mm=5.0)[0] == 11
    ws3 = np.zeros(gsh, np.uint8)                      # a tap on background says so, paints nothing
    n0, note0 = wand_fill(whu, ws3, "coronal", 20, 5, 5, gvox, 1, floor=120.0, radius_mm=10.0)
    assert n0 == 0 and note0 and not ws3.any(), (n0, note0)
    n_we, _ = wand_fill(whu, ws, "coronal", 20, 60 - 1 - 25, 20, gvox, 0,   # wand-erase
                        floor=120.0, radius_mm=100.0)
    assert n_we == 30 and not ws.any(), "wand erase clears the connected labelled blob"
    # the numpy flood fill is what runs on a bare numpy+Pillow install, so scipy has to match IT
    wband = (whu >= 120)
    assert np.array_equal(_flood_numpy(wband, (20, 20, 25)),
                          _connected_from(wband, (20, 20, 25))), "numpy flood == scipy labelling"

    # gate_plane must land on the same pixels as to_display, or the wash explains the wrong voxels
    gp = gate_plane(ghu, "coronal", 20, 0, 120.0, 700.0)
    assert gp.shape == disp_shape("coronal", gsh)
    assert gp[60 - 1 - 30, 20] and not gp[60 - 1 - 30, 27] and not gp[60 - 1 - 10, 20], \
        "gate plane: duct in, rib out, muscle out -- in display orientation"
    gpo = gate_plane(ghu, "coronal", 20, 0, -np.inf, np.inf)
    assert gpo.all(), "an open band opens everything"

    # save provenance: a seg identical to the auto-seed must not be stamped as hand-edited
    a = np.zeros((4, 4, 4), np.uint8); a[1, 1, 1] = 1
    assert seed_stamp(a, a.copy()) == {"edited": np.bool_(False), "n_diff_seed": np.int64(0),
                                       "n_seed": np.int64(1)}, "unchanged save is not an edit"
    b = a.copy(); b[2, 2, 2] = 2
    st = seed_stamp(b, a)
    assert bool(st["edited"]) and int(st["n_diff_seed"]) == 1, ("one changed voxel", st)
    assert int(seed_stamp(a, None)["n_diff_seed"]) == -1, "no seed -> human by construction"
    assert int(seed_stamp(a, np.zeros((2, 2, 2), np.uint8))["n_diff_seed"]) == -1, "shape mismatch"
    # an UNREADABLE seed must not be mistaken for "no seed", which would certify it as hand work
    unk = seed_stamp(a, None, seed_ok=False)
    assert int(unk["n_diff_seed"]) == -2 and not bool(unk["edited"]), ("unreadable seed", unk)

    # The /save response goes through json.dumps. numpy scalars are NOT int/bool instances, so a
    # raw stamp value here raised TypeError AFTER the sidecar was written -- the save succeeded on
    # disk and the browser got a bare 500. Assert the actual body shape, not just the stamp dict.
    try:
        json.dumps({"ok": True, "diff_seed": seed_stamp(a, a.copy())["n_diff_seed"]})
        raise AssertionError("expected a raw np.int64 to be rejected by json.dumps")
    except TypeError:
        pass
    json.dumps({"ok": True, "diff_seed": int(seed_stamp(a, a.copy())["n_diff_seed"])})
    # and _send's encoder must survive one slipping through anyway
    json.dumps({"d": np.int64(3), "e": np.bool_(True)},
               default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("selftest OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Browser/iPad CC/TD seg painter (cranial-up)")
    ap.add_argument("input", nargs="?", help="context4d_<tag>/ dir or a single f*.npz")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--selftest", action="store_true", help="run math checks, no server")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return
    if not args.input:
        ap.error("give a context4d_<tag>/ dir or an f*.npz (or --selftest)")
    serve(args.input, port=args.port)


if __name__ == "__main__":
    main()

