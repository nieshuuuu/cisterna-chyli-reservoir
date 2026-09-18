import numpy as np

from cc_reservoir.measure.context import place_into


def test_place_into_offsets_correctly():
    tight = np.zeros((4, 4, 4), bool); tight[1, 1, 1] = True
    out = place_into(tight, lo_tight=(10, 20, 30), lo_regional=(8, 16, 24), shape_regional=(20, 20, 20))
    assert out.shape == (20, 20, 20)
    assert out[10 - 8 + 1, 20 - 16 + 1, 30 - 24 + 1]      # the True voxel lands at the offset
    assert out.sum() == 1
