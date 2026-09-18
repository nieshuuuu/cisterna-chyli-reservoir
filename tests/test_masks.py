import numpy as np
from scipy.io import savemat
from cc_reservoir.io.masks import load_cc_mask

def test_returns_bool_mask(tmp_path):
    arr = np.zeros((512, 512, 20), dtype=np.uint8); arr[10:14, 10:14, 2:5] = 1
    p = tmp_path / "CC_dcm_01.mat"; savemat(p, {"cc_stack_01": arr})
    m = load_cc_mask(str(p))
    assert m.dtype == bool and m.shape == (512, 512, 20) and int(m.sum()) == arr.sum()
