import numpy as np
from scipy.io import savemat
from cc_reservoir.io.matio import load_mat_3d

def test_loads_v5_and_keeps_xyz(tmp_path):
    arr = np.zeros((512, 512, 40), dtype=np.uint8); arr[100:110, 100:110, 5:9] = 1
    p = tmp_path / "m.mat"; savemat(p, {"cc_stack_01": arr})
    out = load_mat_3d(str(p))
    assert out.shape == (512, 512, 40)
    assert int(out.sum()) == int(arr.sum())

def test_normalizes_transposed_slice_first(tmp_path):
    # simulate an h5py-style (Z,512,512) layout saved as v5; loader must move slices last
    arr = np.zeros((40, 512, 512), dtype=np.uint8); arr[5:9, 100:110, 100:110] = 1
    p = tmp_path / "m.mat"; savemat(p, {"td_stack_01": arr})
    out = load_mat_3d(str(p))
    assert out.shape == (512, 512, 40)

def test_ignores_meta_keys(tmp_path):
    p = tmp_path / "m.mat"; savemat(p, {"im_stack_01": np.ones((512, 512, 10))})
    assert load_mat_3d(str(p)).shape == (512, 512, 10)
