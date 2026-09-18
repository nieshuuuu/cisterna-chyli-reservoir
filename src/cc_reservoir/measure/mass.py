# src/cc_reservoir/measure/mass.py
import numpy as np

def contrast_mass(hu, roi, baseline_hu, voxel_vol):
    """PV-robust contrast mass = sum of baseline-subtracted enhancement over the ROI,
    times the voxel volume (mm^3). Conserved under PSF blur."""
    return float(np.sum(hu[roi] - baseline_hu) * voxel_vol)

def mean_enhancement(hu, roi, baseline_hu):
    """Mean baseline-subtracted HU over the ROI; 0.0 for an empty ROI (CC not detected)."""
    if not np.any(roi):
        return 0.0
    return float(np.mean(hu[roi] - baseline_hu))
