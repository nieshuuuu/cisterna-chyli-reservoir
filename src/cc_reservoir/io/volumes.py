import numpy as np
from .matio import load_mat_3d


def load_hu_volume(path):
    """Load a per-timepoint HU volume from MAT/*.mat (already HU: slope=1, intercept=0)."""
    return load_mat_3d(path).astype(float)


def mask_bbox(mask, pad=0):
    """Inclusive-low / exclusive-high bounding box of a boolean mask, padded and clipped."""
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + 1 + pad, mask.shape)  # symmetric pad: extend both faces
    return tuple(int(x) for x in lo), tuple(int(x) for x in hi)


def crop_to_bbox(arr, lo, hi):
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
