"""context (downsampled crop) -> raw DICOM grid mapping (measure.context).

The 治本 for reading raw values under an edited seg: build_mip_context now records
crop_lo/crop_hi/downsample/raw_full_shape in meta.npz, so the mask maps back to the raw with NO
rebuild via `raw_index = crop_lo + downsample * context_index`. Pure numpy — no DICOM needed here.
"""
import numpy as np
import pytest

from cc_reservoir.measure.context import (
    upsample_seg_to_crop, context_index_to_raw, place_seg_in_raw, load_context_seg,
)


def test_load_context_seg_prefers_the_manual_edit_sidecar(tmp_path):
    """A saved f<N>_edit.npz must win over the frame's built-in seg — same rule as
    io.context.load_seg / the painter. Otherwise the manual edit is silently ignored and the
    automatic seg gets measured instead."""
    d = tmp_path / "context4d_z"; d.mkdir()
    auto = np.zeros((4, 4, 4), np.uint8); auto[0, 0, 0] = 1
    np.savez_compressed(d / "f2.npz", hu=np.zeros((4, 4, 4), np.int16), seg=auto)
    assert np.array_equal(load_context_seg(str(d), 2), auto)          # no sidecar -> built-in seg
    manual = np.zeros((4, 4, 4), np.uint8); manual[3, 3, 3] = 2
    np.savez_compressed(d / "f2_edit.npz", seg=manual, source="f2.npz")
    assert np.array_equal(load_context_seg(str(d), 2), manual)        # sidecar present -> YOUR seg


def test_upsample_repeats_each_voxel_per_axis():
    seg = np.array([[[1]], [[2]]], np.uint8)              # (2,1,1): two voxels along x
    up = upsample_seg_to_crop(seg, (2, 2, 1))
    assert up.shape == (4, 2, 1)
    assert up[0, 0, 0] == 1 and up[1, 0, 0] == 1          # voxel 0 repeated 2x in x
    assert up[2, 0, 0] == 2 and up[3, 1, 0] == 2          # voxel 1 repeated 2x in x, 2x in y


def test_context_index_to_raw_is_crop_lo_plus_ds_times_index():
    assert context_index_to_raw((5, 7, 9), (100, 200, 0), (2, 2, 1)) == (110, 214, 9)
    assert context_index_to_raw((0, 0, 0), (4, 6, 2), (2, 2, 1)) == (4, 6, 2)


def _meta(lo, hi, ds, full):
    return {"crop_lo": np.array(lo), "crop_hi": np.array(hi),
            "downsample": np.array(ds), "raw_full_shape": np.array(full)}


def test_place_seg_lands_on_the_right_raw_block():
    meta = _meta((4, 6, 2), (12, 14, 7), (2, 2, 1), (20, 20, 10))
    seg = np.zeros((4, 4, 5), np.uint8); seg[1, 2, 3] = 1
    rm = place_seg_in_raw(seg, meta)
    assert rm.shape == (20, 20, 10)
    got = set(map(tuple, np.argwhere(rm == 1)))
    expect = {(x, y, 5) for x in (6, 7) for y in (10, 11)}   # crop_lo+ds*(1,2,3)=(6,10,5), a 2x2 block
    assert got == expect


@pytest.mark.parametrize("seed", range(30))
def test_place_seg_matches_context_index_to_raw(seed):
    """Invariant: every labelled context voxel occupies exactly the ds-sized raw block anchored at
    context_index_to_raw(idx). place_seg_in_raw and context_index_to_raw must never disagree."""
    meta = _meta((3, 5, 1), (3 + 2 * 6, 5 + 2 * 6, 1 + 8), (2, 2, 1), (40, 40, 20))
    rng = np.random.default_rng(seed)
    seg = np.zeros((6, 6, 8), np.uint8)
    i, j, k = (int(rng.integers(0, n)) for n in (6, 6, 8))
    seg[i, j, k] = 2
    rm = place_seg_in_raw(seg, meta)
    c = context_index_to_raw((i, j, k), meta["crop_lo"], meta["downsample"])
    expect = {(c[0] + a, c[1] + b, c[2]) for a in range(2) for b in range(2)}
    assert set(map(tuple, np.argwhere(rm == 2))) == expect


def test_two_labels_map_independently():
    """CC(1) and TD(2) must map back separately (never unioned), matching the pipeline rule."""
    meta = _meta((10, 10, 0), (10 + 2 * 4, 10 + 2 * 4, 5), (2, 2, 1), (30, 30, 5))
    seg = np.zeros((4, 4, 5), np.uint8); seg[0, 0, 1] = 1; seg[3, 3, 4] = 2
    rm = place_seg_in_raw(seg, meta)
    assert set(map(tuple, np.argwhere(rm == 1))) == {(10, 10, 1), (10, 11, 1), (11, 10, 1), (11, 11, 1)}
    assert set(map(tuple, np.argwhere(rm == 2))) == {(16, 16, 4), (16, 17, 4), (17, 16, 4), (17, 17, 4)}


def test_roundtrip_against_the_real_block_reduce():
    """End-to-end: raw -> crop -> block_reduce (EXACTLY what build_mip_context does) -> place_seg_in_raw
    recovers the raw block. Proves place_seg_in_raw inverts the ACTUAL downsample the build applies,
    not just my own context_index_to_raw."""
    from skimage.measure import block_reduce
    full = (40, 44, 12); ds = (2, 2, 1)
    lo = (6, 8, 1); hi = (6 + 2 * 8, 8 + 2 * 9, 1 + 7)          # crop 16x18x6 -> context 8x9x6
    raw = np.zeros(full, np.uint8)
    raw[10:12, 14:16, 3] = 1                                    # a block-aligned duct marker in the crop
    crop = raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    seg_ctx = block_reduce(crop, (ds[0], ds[1], 1), np.max).astype(np.uint8)   # what build stores as seg
    rm = place_seg_in_raw(seg_ctx, _meta(lo, hi, ds, full))
    assert (rm[10:12, 14:16, 3] == 1).all()                    # recovers the original marker voxels
    assert set(map(tuple, np.argwhere(rm == 1))) == {(x, y, 3) for x in (10, 11) for y in (14, 15)}


def test_place_seg_stays_in_bounds_at_odd_crop_edge():
    """An odd crop extent (block_reduce pads it) must clip at crop_hi, never overflow raw_full_shape."""
    meta = _meta((0, 0, 0), (5, 5, 3), (2, 2, 1), (5, 5, 3))   # crop 5 (odd) -> context 3, upsample 6 > 5
    seg = np.zeros((3, 3, 3), np.uint8); seg[2, 2, 2] = 1       # last context voxel, at the padded edge
    rm = place_seg_in_raw(seg, meta)                            # must not raise / overflow
    assert rm.shape == (5, 5, 3) and int(np.argwhere(rm == 1)[:, 0].max()) < 5
