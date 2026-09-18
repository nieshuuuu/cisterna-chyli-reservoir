# tests/test_volume.py
import numpy as np

from cc_reservoir.measure.patch import CCPatch
from cc_reservoir.measure.volume import (integrated_hu_volume, integrated_hu_volume_region,
                                         robust_volume)

VOX = (1.0, 1.0, 1.0)


def _solid_patch(h_peak=550.0, h_early=120.0, bg=50.0):
    """Sharp uniform cuboid (no PV) of known voxel count on uniform background."""
    shape = (30, 30, 40)
    mask = np.zeros(shape, bool); mask[12:18, 12:18, 10:30] = True   # 6*6*20 = 720 voxels
    td = np.zeros(shape, bool)
    peak = np.full(shape, bg); peak[mask] = h_peak
    early = np.full(shape, bg); early[mask] = h_early                # below the opacified threshold
    return CCPatch(mask=mask, td=td, vols=[early, peak], peak_i=1, voxel_mm=VOX,
                   lo=(0, 0, 0), hi=shape), int(mask.sum())


def test_integrated_hu_recovers_geometric_volume_for_uniform_object():
    patch, n_true = _solid_patch()
    V, s_o, s_bg = integrated_hu_volume(patch, frame=1)
    assert abs(s_o - 550.0) < 1e-6 and abs(s_bg - 50.0) < 1e-6      # PV-free core, clean ring
    assert abs(V - n_true * np.prod(VOX)) / (n_true * np.prod(VOX)) < 0.03   # exact up to ROI edges


def test_integrated_hu_volume_is_background_invariant():
    # V must not depend on background level (the (I - A*S_BG)/(S_O - S_BG) construction cancels it)
    v_lo = integrated_hu_volume(_solid_patch(bg=30.0)[0], frame=1)[0]
    v_hi = integrated_hu_volume(_solid_patch(bg=90.0)[0], frame=1)[0]
    assert abs(v_lo - v_hi) / v_lo < 0.02


def test_region_integrated_hu_recovers_geometric_volume():
    # same Molloi formula, but on an explicit fixed region (the solid cuboid itself)
    patch, n_true = _solid_patch()
    V, s_o, s_bg = integrated_hu_volume_region(patch.mask, patch.vols[1], VOX)
    assert abs(s_o - 550.0) < 1e-6 and abs(s_bg - 50.0) < 1e-6
    assert abs(V - n_true * np.prod(VOX)) / (n_true * np.prod(VOX)) < 0.05


def test_robust_volume_uses_only_opacified_frames():
    patch, n_true = _solid_patch()
    V, spread, used = robust_volume(patch)
    assert used == [1]                                             # early frame (70 HU contrast) excluded
    assert abs(V - n_true) / n_true < 0.03 and spread == 0.0
