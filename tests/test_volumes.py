import numpy as np
from scipy.io import savemat
from cc_reservoir.io.volumes import load_hu_volume, crop_to_bbox, mask_bbox

def test_load_hu_is_float(tmp_path):
    arr = (np.random.default_rng(0).integers(-1000, 2000, (512, 512, 8))).astype(np.int16)
    p = tmp_path / "0101.mat"; savemat(p, {"im_stack_01": arr})
    v = load_hu_volume(str(p))
    assert v.dtype == np.float64 and v.shape == (512, 512, 8)
    assert np.allclose(v, arr.astype(float))

def test_bbox_and_crop():
    m = np.zeros((20, 20, 10), bool); m[5:9, 6:8, 2:5] = True
    lo, hi = mask_bbox(m, pad=1)
    assert lo == (4, 5, 1) and hi == (10, 9, 6)   # symmetric pad: both faces +1
    vol = np.arange(20*20*10).reshape(20, 20, 10).astype(float)
    sub = crop_to_bbox(vol, lo, hi)
    assert sub.shape == (6, 4, 5)
