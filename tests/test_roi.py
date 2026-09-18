# tests/test_roi.py
import numpy as np
from cc_reservoir.measure.roi import dilated_mask_roi, threshold_roi

def test_dilation_grows_mask_anisotropically():
    m = np.zeros((30, 30, 30), bool); m[15, 15, 15] = True
    roi = dilated_mask_roi(m, mm=2.0, voxel_mm=(1.0, 1.0, 2.0))
    # 2 mm radius => 2 voxels in x/y, 1 voxel in z
    assert roi[13, 15, 15] and roi[17, 15, 15] and not roi[12, 15, 15]
    assert roi[15, 15, 14] and not roi[15, 15, 13]
    assert roi.sum() > m.sum()

def test_threshold_roi_selects_bright_and_empty_when_flat():
    hu = np.zeros((20, 20, 20)); hu[8:12, 8:12, 8:12] = 600.0
    seed = np.zeros((20, 20, 20), bool); seed[5:15, 5:15, 5:15] = True
    roi = threshold_roi(hu, seed, k_sigma=3.0)
    assert roi[9, 9, 9] and not roi[2, 2, 2] and roi.sum() == 64
    flat = np.full((20, 20, 20), 45.0)                       # no enhancement
    assert threshold_roi(flat, seed, k_sigma=3.0).sum() == 0  # empty, not whole-seed
