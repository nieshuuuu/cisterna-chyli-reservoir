# tests/test_baseline.py
import numpy as np
from cc_reservoir.measure.baseline import ring_baseline, tp1_baseline

def test_ring_baseline_is_local_tissue_mean():
    hu = np.full((30, 30, 30), 45.0)          # soft tissue
    hu[14:17, 14:17, 14:17] = 800.0           # enhanced CC
    roi = np.zeros((30, 30, 30), bool); roi[14:17, 14:17, 14:17] = True
    b = ring_baseline(hu, roi, inner_mm=1.0, outer_mm=4.0, voxel_mm=(1.0, 1.0, 1.0))
    assert abs(b - 45.0) < 1e-6              # ring sits in tissue, excludes the bright core

def test_tp1_baseline_uses_first_frame_mean_in_roi():
    series = [(0.0, np.full((10, 10, 10), 50.0)), (60.0, np.full((10, 10, 10), 900.0))]
    roi = np.ones((10, 10, 10), bool)
    assert tp1_baseline(series, roi) == 50.0
