# tests/test_mass.py
import numpy as np
from cc_reservoir.measure.mass import contrast_mass, mean_enhancement

def test_contrast_mass_is_sum_enhancement_times_voxel():
    hu = np.full((10, 10, 10), 50.0); hu[4:6, 4:6, 4:6] = 550.0
    roi = np.zeros((10, 10, 10), bool); roi[4:6, 4:6, 4:6] = True   # 8 voxels at 550
    m = contrast_mass(hu, roi, baseline_hu=50.0, voxel_vol=2.0)
    assert np.isclose(m, 8 * (550.0 - 50.0) * 2.0)                  # = 8000 HU*mm^3

def test_mean_enhancement_subtracts_baseline():
    hu = np.full((6, 6, 6), 700.0); roi = np.ones((6, 6, 6), bool)
    assert np.isclose(mean_enhancement(hu, roi, baseline_hu=120.0), 580.0)

def test_mean_enhancement_empty_roi_is_zero():
    import numpy as np
    from cc_reservoir.measure.mass import mean_enhancement
    hu = np.full((5,5,5), 300.0); roi = np.zeros((5,5,5), bool)
    assert mean_enhancement(hu, roi, baseline_hu=50.0) == 0.0
