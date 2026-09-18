# tests/test_patch.py
import numpy as np

from cc_reservoir.measure.patch import CCPatch, enhancement_field
from cc_reservoir.measure.roi import dilated_mask_roi

VOX = (0.78, 0.78, 1.0)


def _synthetic_patch():
    """Background tissue with an enhanced CC blob and a separate, contiguous TD blob."""
    shape = (30, 30, 40)
    mask = np.zeros(shape, bool); mask[12:18, 12:18, 8:24] = True     # CC: central column
    td = np.zeros(shape, bool); td[14:16, 14:16, 26:34] = True        # TD: thin, just above CC
    bg = np.full(shape, 45.0)
    peak = bg.copy(); peak[mask] = 545.0; peak[td] = 500.0            # both enhanced at peak
    early = bg.copy(); early[mask] = 95.0; early[td] = 80.0           # weakly enhanced earlier
    return CCPatch(mask=mask, td=td, vols=[early, peak], peak_i=1, voxel_mm=VOX, lo=(0, 0, 0), hi=shape)


def test_enhancement_field_excludes_td_and_zeros_outside_search():
    patch = _synthetic_patch()
    enh, search = enhancement_field(patch, search_mm=8.0, td_excl_mm=1.5)

    assert len(enh) == len(patch.vols)                               # one field per timepoint
    assert not np.any(search & dilated_mask_roi(patch.td, 1.5, VOX)) # TD removed from search region
    assert np.all(enh[1][~search] == 0.0)                            # nothing outside the search region
    assert np.all(enh[1][patch.td] == 0.0)                           # TD voxels carry no CC enhancement


def test_enhancement_field_recovers_contrast_above_background_in_cc():
    patch = _synthetic_patch()
    enh, _ = enhancement_field(patch, search_mm=8.0, td_excl_mm=1.5)
    core = patch.mask
    assert enh[1][core].mean() > enh[0][core].mean()                 # peak frame enhances more than early
    assert abs(enh[1][core].mean() - 500.0) < 5.0                    # 545 CC - 45 background
