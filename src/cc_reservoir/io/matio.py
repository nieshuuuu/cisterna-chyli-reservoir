import numpy as np


def load_mat_3d(path):
    """Load the single 3-D array from a MATLAB .mat (v5 via scipy, v7.3 via h5py),
    normalized to (X, Y, Z) with Z the slice axis. SSoT for all .mat loading."""
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        arr = np.asarray(d[[k for k in d if not k.startswith("__")][0]])
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            arr = np.asarray(f[list(f.keys())[0]])
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D array in {path}, got shape {arr.shape}")
    # put the slice axis last: the two large (512-ish) axes lead
    if arr.shape[2] == 512 and arr.shape[0] != 512:
        arr = np.transpose(arr, (2, 1, 0))
    return arr
