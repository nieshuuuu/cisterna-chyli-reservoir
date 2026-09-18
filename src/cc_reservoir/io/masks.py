from .matio import load_mat_3d


def load_cc_mask(path):
    """Load a binary CC mask from SEGMENT_dcm/CC_dcm_01.mat."""
    return load_mat_3d(path) > 0
