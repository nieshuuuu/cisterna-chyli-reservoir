"""Refine a hand-traced lumen mask against the enhancement field (single producer).

Why this exists. The hand tracing is the trusted segmentation (operator decision, 2026-07-27),
but it is drawn slice by slice on a slab-MIP, so its boundary wanders by a voxel or two and in
places runs past the opacified lumen. measure.surface's global half-max is the opposite
failure: it sets one level from the single brightest voxel in the whole patch, which on a duct
whose intensity falls off cranially cuts the faint end away entirely.

The fix is the same FWHM criterion, applied LOCALLY. The cisterna chyli tapers into the TD, so
lumen intensity varies several-fold end to end; a half-max threshold is only meaningful against
a nearby reference. We walk the long axis in short stations, take a robust peak from the traced
voxels in each station, and threshold at half of that — the classical vessel-lumen boundary,
just with a local rather than a global reference.

The trace still bounds the result: the refinement may only move the boundary within ``grow_mm``
of what was traced, and only components touching the traced core survive. So this sharpens a
tracing against the data; it cannot invent a duct somewhere else.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes, label,
                           uniform_filter1d)

from cc_reservoir.measure.roi import dilated_mask_roi, _ellipsoid_struct


@dataclass
class RefinedLumen:
    mask: np.ndarray          # refined binary lumen
    stations_mm: np.ndarray   # station centre along the long axis (mm from caudal end)
    peaks: np.ndarray         # robust local enhancement peak per station (HU)
    levels: np.ndarray        # threshold applied per station (HU)
    kept_frac: float          # refined voxels / traced voxels
    flooded: bool = False     # threshold failed to discriminate; tracing returned unchanged


def long_axis(mask, voxel_mm):
    """(centroid_mm, unit long axis) of a mask, by PCA of its voxel centres."""
    pts = np.argwhere(mask) * np.asarray(voxel_mm, float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    return c, d


def refine_traced_lumen(trace, enh, voxel_mm, grow_mm=0.8, station_mm=3.0, peak_pct=90.0,
                        floor_hu=60.0, smooth_stations=3, level_frac=0.5, max_growth=1.5):
    """Snap ``trace`` to the local half-max boundary of ``enh``.

    ``enh`` must be background-subtracted (measure.patch.enhancement_field), so 0 is tissue
    and the half-max level is simply half the local lumen peak.

    Parameters that matter:
      grow_mm     how far outside the tracing the boundary may move. Keep it ~1 voxel: the
                  tracing locates the duct, the data only refines its edge. Larger values are
                  actively dangerous on a low-contrast frame — the dilation band around a long
                  thin tracing has several times the tracing's own volume, so if the threshold
                  stops discriminating the band floods and the "lumen" comes out bigger than
                  what was traced. See the flooding guard below.
      station_mm  long-axis bin width for the local peak. Too short and the peak is noise;
                  too long and the taper is averaged away.
      peak_pct    percentile of in-trace enhancement used as the local peak. NOT the max —
                  that is the single-hot-voxel failure this function exists to avoid.
      floor_hu    a station whose local peak is below this has no credible lumen signal; the
                  tracing is kept there rather than deleted, and it shows up in ``peaks``.
      level_frac  fraction of the local peak used as the boundary. 0.5 is textbook FWHM and
                  is the right choice when you want the true lumen with partial volume backed
                  out. Lower values sit further out toward the visible edge of the blur — use
                  when the boundary is wanted where the eye reads it, not where the FWHM is.
    """
    trace = trace.astype(bool)
    if not trace.any():
        return RefinedLumen(trace.copy(), np.zeros(0), np.zeros(0), np.zeros(0), 0.0)

    vox = np.asarray(voxel_mm, float)
    band = dilated_mask_roi(trace, grow_mm, voxel_mm)          # the only place the edge may move
    c, d = long_axis(trace, voxel_mm)

    # long-axis coordinate of every voxel in the band, and of the traced voxels
    idx = np.argwhere(band)
    t_band = (idx * vox - c) @ d
    t_trace = (np.argwhere(trace) * vox - c) @ d
    e_trace = enh[trace]

    lo, hi = t_trace.min(), t_trace.max()
    nb = max(int(np.ceil((hi - lo) / station_mm)), 1)
    edges = np.linspace(lo, hi, nb + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    which = np.clip(np.digitize(t_trace, edges) - 1, 0, nb - 1)
    peaks = np.full(nb, np.nan)
    for b in range(nb):
        sel = e_trace[which == b]
        if sel.size >= 5:
            peaks[b] = np.percentile(sel, peak_pct)
    if np.isnan(peaks).all():
        return RefinedLumen(trace.copy(), centres, peaks, peaks, 1.0)
    # carry the nearest usable peak into empty stations, then damp station-to-station noise
    good = ~np.isnan(peaks)
    peaks = np.interp(np.arange(nb), np.flatnonzero(good), peaks[good])
    if smooth_stations > 1:
        peaks = uniform_filter1d(peaks, size=smooth_stations, mode="nearest")

    levels = np.maximum(level_frac * peaks, floor_hu)
    # a station with no credible signal keeps its tracing: threshold it out of the way
    weak = peaks < floor_hu
    levels[weak] = -np.inf

    lvl_band = np.interp(t_band, centres, levels)
    keep = enh[band] >= lvl_band
    out = np.zeros_like(trace)
    out[idx[keep, 0], idx[keep, 1], idx[keep, 2]] = True
    if weak.any():                                   # restore the tracing where signal was absent
        weak_t = np.interp(t_trace, centres, weak.astype(float)) > 0.5
        wi = np.argwhere(trace)[weak_t]
        out[wi[:, 0], wi[:, 1], wi[:, 2]] = True

    # close single-voxel pits, fill interior holes, then drop anything not attached to the
    # traced core — the refinement must not spawn a second object next door.
    out = binary_closing(out, _ellipsoid_struct(0.9, voxel_mm))
    out = binary_fill_holes(out)
    core = binary_erosion(trace, iterations=1)
    if not core.any():
        core = trace
    lab, n = label(out)
    if n:
        keep_ids = np.unique(lab[core & (lab > 0)])
        out = np.isin(lab, keep_ids[keep_ids > 0])
    if not out.any():
        out = trace.copy()

    # Flooding guard, on a RATIO not on equality. A tracing drawn slice by slice on a slab-MIP is
    # routinely a little short of the lumen in places, so modest growth is the refinement doing
    # its job and forbidding it outright throws away good frames (8_31_22/Acq4 f0 is traced at
    # only 22% of the peak frame's extent — growing it is a correction, not an error). What must
    # be caught is the threshold ceasing to discriminate at all: the dilation band around a long
    # thin tracing holds several times the tracing's own volume, so a failed threshold floods it
    # and returns a multiple of what was traced. Beyond max_growth, refuse and say so, rather
    # than let a 4x-too-large volume into a V(t) curve looking perfectly plausible.
    if out.sum() > max_growth * trace.sum():
        return RefinedLumen(trace.copy(), centres, peaks, levels, 1.0, flooded=True)
    return RefinedLumen(out, centres, peaks, levels, float(out.sum()) / float(trace.sum()))
