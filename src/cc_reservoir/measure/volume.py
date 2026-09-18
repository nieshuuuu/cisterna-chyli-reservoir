"""Integrated-HU CC volume (Molloi 2019, eScholarship qt5b23p5h6) — single producer.

V = (I - A*S_BG) / (S_O - S_BG) * v_vox, with
  I    = integrated HU over the object ROI (CC dilated, TD excluded),
  A    = its voxel count,
  S_BG = peri-CC ring background (per frame),
  S_O  = a PARTIAL-VOLUME-FREE lumen reference = top of the eroded CC core (per frame).

S_O must come from the eroded core, not the thin CC's own peak voxel (which is itself
PV-attenuated and overestimates V ~3x). The TD is excluded from the object ROI because
CC-TD contiguity inflates the integral. The volume is only meaningful once the lumen is
opacified (S_O - S_BG large); early/late frames give unstable V and are dropped by
robust_volume. That V is ~flat across opacified frames while S_O swings is what validates
it as a real volume rather than a concentration proxy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

from cc_reservoir.measure.roi import dilated_mask_roi, _ellipsoid_struct

MIN_DENOM_HU = 20.0       # need S_O - S_BG above this for the ratio to be meaningful
OPACIFIED_HU = 200.0      # lumen counts as opacified (stable V) when S_O - S_BG exceeds this


@dataclass
class VolumeROIs:
    obj: np.ndarray       # object ROI (CC dilated, TD excluded)
    A: int                # object voxel count
    ring: np.ndarray      # peri-CC background ring
    core: np.ndarray      # eroded CC core (PV-free-ish S_O source)


def volume_rois(patch, obj_dil_mm=3.0, td_excl_mm=1.5, ring_inner_mm=1.0, ring_outer_mm=4.0):
    """Frame-independent ROIs for the integrated-HU volume (depend only on the masks)."""
    mask, td, vox = patch.mask, patch.td, patch.voxel_mm
    obj = dilated_mask_roi(mask, obj_dil_mm, vox)
    if td.sum():
        obj = obj & ~dilated_mask_roi(td, td_excl_mm, vox)
    inner = binary_dilation(obj, _ellipsoid_struct(ring_inner_mm, vox))
    outer = binary_dilation(obj, _ellipsoid_struct(ring_outer_mm, vox))
    ring = outer & ~inner
    if td.sum():
        ring = ring & ~dilated_mask_roi(td, td_excl_mm, vox)
    core = binary_erosion(mask, iterations=1)
    return VolumeROIs(obj=obj, A=int(obj.sum()), ring=ring, core=core)


def _molloi_volume(integrated_hu, n_vox, s_bg, s_o, vvol):
    """Molloi Eq.2 contrast-filled volume from integrated HU — the formula lives ONLY here.
    nan if the lumen reference is not above background (region/frame not opacified)."""
    if (s_o - s_bg) <= MIN_DENOM_HU:
        return float("nan")
    return (integrated_hu - n_vox * s_bg) / (s_o - s_bg) * vvol


def integrated_hu_volume(patch, frame, rois=None):
    """Integrated-HU volume (mm^3) at one frame; returns (V, S_O, S_BG). V is nan if the
    lumen reference is not above background (frame not opacified)."""
    rois = rois or volume_rois(patch)
    mask, vox = patch.mask, patch.voxel_mm
    v = patch.vols[frame]
    s_bg = float(v[rois.ring].mean())
    s_o = (float(np.mean(np.sort(v[rois.core])[-5:])) if rois.core.sum() >= 5
           else float(np.percentile(v[mask], 98)))
    return _molloi_volume(float(v[rois.obj].sum()), rois.A, s_bg, s_o, float(np.prod(vox))), s_o, s_bg


def integrated_hu_volume_region(region, vol, voxel_mm, ring_inner_mm=1.0, ring_outer_mm=4.0):
    """Integrated-HU contrast volume (mm^3) within a FIXED region (a whole CC or TD lumen
    segment), via the same Molloi formula. S_BG = ring around the region; S_O = PV-free
    eroded-core top. Use this when the anatomical extent is fixed and only the contrast
    inside it changes over time. Returns (V, S_O, S_BG); V is nan if not opacified."""
    ring = (binary_dilation(region, _ellipsoid_struct(ring_outer_mm, voxel_mm))
            & ~binary_dilation(region, _ellipsoid_struct(ring_inner_mm, voxel_mm)))
    core = binary_erosion(region, iterations=1)
    if core.sum() < 5:
        core = region
    s_bg = float(vol[ring].mean()); s_o = float(np.mean(np.sort(vol[core])[-5:]))
    V = _molloi_volume(float(vol[region].sum()), int(region.sum()), s_bg, s_o, float(np.prod(voxel_mm)))
    return V, s_o, s_bg


def volume_series(patch, rois=None):
    """(V, S_O, S_BG) for every frame."""
    rois = rois or volume_rois(patch)
    return [integrated_hu_volume(patch, i, rois) for i in range(len(patch.vols))]


def robust_volume(patch, opacified_hu=OPACIFIED_HU, rois=None):
    """Median integrated-HU volume over opacified frames (S_O - S_BG > opacified_hu).

    Returns (V_median, V_spread, frames_used). V_spread is the half-range across the used
    frames — small spread is the in-acquisition evidence that V is a stable volume.
    """
    rois = rois or volume_rois(patch)
    series = volume_series(patch, rois)
    used = [(i, V) for i, (V, s_o, s_bg) in enumerate(series)
            if np.isfinite(V) and (s_o - s_bg) > opacified_hu]
    if not used:
        return float("nan"), float("nan"), []
    vs = np.array([V for _, V in used])
    return float(np.median(vs)), float((vs.max() - vs.min()) / 2), [i for i, _ in used]
