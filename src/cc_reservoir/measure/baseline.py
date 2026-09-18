# src/cc_reservoir/measure/baseline.py
import numpy as np
from scipy.ndimage import binary_dilation
from .roi import _ellipsoid_struct

def ring_baseline(hu, roi, inner_mm, outer_mm, voxel_mm):
    """Mean HU of a soft-tissue shell around (but excluding) the ROI, per timepoint.
    Cancels per-scan/tissue drift and isolates iodine enhancement."""
    inner = binary_dilation(roi, structure=_ellipsoid_struct(inner_mm, voxel_mm))
    outer = binary_dilation(roi, structure=_ellipsoid_struct(outer_mm, voxel_mm))
    ring = outer & ~inner
    if not np.any(ring):
        return float(np.median(hu[roi])) if np.any(roi) else 0.0
    return float(hu[ring].mean())

def tp1_baseline(hu_series, roi):
    """Mean HU inside the ROI at the first timepoint (pre/early-contrast reference)."""
    _, v0 = hu_series[0]
    return float(v0[roi].mean())
