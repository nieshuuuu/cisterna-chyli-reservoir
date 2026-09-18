"""Orientation maps must be self-consistent: a labelled voxel lands at the pixel the forward map
predicts, and the inverse map returns it."""
import numpy as np

from cc_reservoir.viz.slice_canvas import slice_2d, voxel_from_pixel, pixel_from_voxel, _AXIS


def test_pixel_voxel_roundtrip_and_placement():
    shape = (12, 14, 16)
    vox = (3, 5, 7)                                  # a distinct voxel
    for view in ("axial", "coronal", "sagittal"):
        col, row = pixel_from_voxel(view, vox, shape)
        back = voxel_from_pixel(view, col, row, vox[_AXIS[view]], shape)
        assert back == vox, f"{view}: {back} != {vox}"

        # a marker at `vox`, sliced at this view's slice index, must appear at (row, col)
        v = np.zeros(shape, np.uint8); v[vox] = 9
        plane = slice_2d(v, view, vox[_AXIS[view]])
        assert plane[row, col] == 9, f"{view}: marker not at predicted pixel"
