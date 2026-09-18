import numpy as np
from PIL import Image

from cc_reservoir.scripts.render_qc_sheet import (CC, TD, axial_levels, duct_box, load_frame,
                                                  render_sheet, slab_covering)


def _phantom():
    """A bright vertical tube: CC on its caudal third, TD above it."""
    hu = np.full((80, 90, 200), 40, np.int16)
    seg = np.zeros(hu.shape, np.uint8)
    hu[38:43, 44:49, 30:180] = 900
    seg[38:43, 44:49, 30:80] = CC
    seg[38:43, 44:49, 80:180] = TD
    return hu, seg


def test_duct_box_contains_every_labelled_voxel():
    _, seg = _phantom()
    lo, hi = duct_box(seg, (5, 5, 5))
    idx = np.argwhere(np.isin(seg, (CC, TD)))
    assert (idx >= lo).all() and (idx < hi).all()


def test_slab_covers_the_requested_range():
    for lo, hi in [(0, 1), (3, 10), (10, 17), (0, 302)]:
        c, h = slab_covering(lo, hi)
        assert c - h <= lo and c + h >= hi - 1


def test_axial_levels_land_on_labelled_slices():
    _, seg = _phantom()
    z_cc, *z_td = axial_levels(seg)
    assert (seg[:, :, z_cc] == CC).any()
    assert all((seg[:, :, z] == TD).any() for z in z_td)


def test_sheet_renders_from_disk(tmp_path):
    hu, seg = _phantom()
    np.savez(tmp_path / "f0.npz", hu=hu)
    np.savez(tmp_path / "f0_edit.npz", seg=seg)
    np.savez(tmp_path / "meta.npz", voxel=np.array([0.781, 0.781, 0.5]))
    out = render_sheet(*load_frame(str(tmp_path), "f0"), "phantom f0", str(tmp_path / "qc.png"))
    assert Image.open(out).size == (2000, 1000)


def test_duct_slab_matches_a_fixed_slab_on_a_straight_duct():
    from cc_reservoir.scripts.render_qc_sheet import duct_slab
    from cc_reservoir.viz.ipad_paint import to_display
    hu, seg = _phantom()
    duct = np.isin(seg, (CC, TD))
    assert np.array_equal(duct_slab(hu, duct, "coronal", 3), to_display(hu, "coronal", 40, 3))
    assert np.array_equal(duct_slab(hu, duct, "sagittal", 3), to_display(hu, "sagittal", 46, 3))


def test_duct_slab_follows_a_duct_that_moves_sideways():
    from cc_reservoir.scripts.render_qc_sheet import duct_slab
    hu = np.zeros((60, 60, 100), np.int16)
    duct = np.zeros(hu.shape, bool)
    for z in range(100):                      # a duct drifting 40 voxels in y over its length
        y = 10 + (40 * z) // 99
        hu[30, y, z] = 1000
        duct[30, y, z] = True
    plane = duct_slab(hu, duct, "sagittal", 2)   # a fixed 5-voxel slab would lose most of it
    assert (plane == 1000).any(axis=1).all()
