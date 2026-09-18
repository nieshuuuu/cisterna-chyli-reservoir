# src/cc_reservoir/measure/roi.py
import numpy as np
from scipy.ndimage import binary_dilation

def _ellipsoid_struct(mm, voxel_mm):
    rx, ry, rz = (int(np.ceil(mm / v)) for v in voxel_mm)
    xx, yy, zz = np.ogrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    q = (xx*voxel_mm[0]/mm)**2 + (yy*voxel_mm[1]/mm)**2 + (zz*voxel_mm[2]/mm)**2
    return q <= 1.0

def dilated_mask_roi(mask, mm, voxel_mm):
    """Fixed ROI: the tp1 mask grown by `mm` (physical, anisotropic) to capture
    contrast that spreads/moves between timepoints."""
    return binary_dilation(mask, structure=_ellipsoid_struct(mm, voxel_mm))

def threshold_roi(hu, seed_mask, k_sigma=3.0, min_enhancement_hu=100.0):
    """Enhanced voxels within the seed region, for the current timepoint. Threshold =
    robust tissue level + max(k_sigma*std, min_enhancement_hu). Returns an EMPTY mask when
    nothing in the seed is enhanced (CC absent / moved away / not yet filled) -- an empty
    ROI is a meaningful 'not here' signal, never silently widened to the whole seed."""
    vals = hu[seed_mask]
    tissue = float(np.median(vals))
    thr = tissue + max(k_sigma * float(vals.std()), float(min_enhancement_hu))
    out = np.zeros_like(seed_mask)
    out[seed_mask] = hu[seed_mask] >= thr
    return out
