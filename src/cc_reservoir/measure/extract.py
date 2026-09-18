import numpy as np
from .roi import dilated_mask_roi, threshold_roi
from .baseline import ring_baseline, tp1_baseline
from .mass import contrast_mass, mean_enhancement

METHODS = ("dilated+ring", "dilated+tp1", "threshold+ring", "threshold+tp1")

def extract_curves(hu_series, mask, voxel_mm, dilate_mm=2.0, search_mm=8.0,
                   k_sigma=3.0, min_enhancement_hu=100.0):
    """For each (ROI x baseline) method return {'t','mass','conc'} over the series.
    ROIs: 'dilated' = FIXED tp1-anchored mask grown by dilate_mm (where the CC was at tp1);
          'threshold' = per-timepoint enhanced voxels within a LARGER search region (mask
          grown by search_mm > dilate_mm) so it can capture contrast that moved or spread
          OUT of the fixed ROI. Divergence between the two => motion / fill-spread.
    Baselines: 'ring' = per-timepoint peri-CC tissue shell; 'tp1' = first-frame mean inside
          the FIXED ROI (ONE scalar per acquisition)."""
    voxel_vol = float(np.prod(voxel_mm))
    fixed = dilated_mask_roi(mask, dilate_mm, voxel_mm)
    search = dilated_mask_roi(mask, search_mm, voxel_mm)
    b_tp1 = tp1_baseline(hu_series, fixed)          # single scalar, fixed ROI
    out = {m: {"t": [], "mass": [], "conc": []} for m in METHODS}
    for t, hu in hu_series:
        b_ring = ring_baseline(hu, fixed, inner_mm=1.0, outer_mm=4.0, voxel_mm=voxel_mm)
        rois = {"dilated": fixed, "threshold": threshold_roi(hu, search, k_sigma, min_enhancement_hu)}
        for roi_name, roi in rois.items():
            for base_name, b in (("ring", b_ring), ("tp1", b_tp1)):
                key = f"{roi_name}+{base_name}"
                out[key]["t"].append(float(t))
                out[key]["mass"].append(contrast_mass(hu, roi, b, voxel_vol))
                out[key]["conc"].append(mean_enhancement(hu, roi, b))
    return out

def method_agreement(curves):
    """Diagnostics driven by WHERE the contrast mass is:
    - containment_at_peak = (fixed 'dilated' mass) / (wider-search 'threshold' mass) at the
      threshold method's peak frame. ~1 => CC stays within the fixed ROI; << 1 => most of the
      enhanced mass is OUTSIDE the fixed ROI (motion / fill-spread).
    - ring_vs_tp1_peak_ratio = peak 'dilated+ring' mass / peak 'dilated+tp1' mass; far from 1
      => the tp1 frame is not a clean baseline (residual-contrast contamination)."""
    thr = np.asarray(curves["threshold+ring"]["mass"], float)
    fix = np.asarray(curves["dilated+ring"]["mass"], float)
    kpeak = int(np.argmax(thr)) if thr.size and thr.max() > 0 else 0
    denom = thr[kpeak] if thr[kpeak] > 1e-9 else 1e-9
    containment = float(fix[kpeak] / denom)
    ring_peak = max(np.asarray(curves["dilated+ring"]["mass"], float).max(), 1e-9)
    tp1_peak = max(np.asarray(curves["dilated+tp1"]["mass"], float).max(), 1e-9)
    return {"containment_at_peak": containment,
            "ring_vs_tp1_peak_ratio": float(ring_peak / tp1_peak)}


def qa_pass(agreement):
    """Clean-acquisition criterion (single definition, used by extraction + Phase-2):
    containment_at_peak > 0.7 (CC stayed in the fixed ROI; not motion/fill-spread) AND
    0.7 < ring_vs_tp1_peak_ratio < 1.4 (tp1 frame is a clean, uncontaminated baseline)."""
    return (agreement["containment_at_peak"] > 0.7
            and 0.7 < agreement["ring_vs_tp1_peak_ratio"] < 1.4)
