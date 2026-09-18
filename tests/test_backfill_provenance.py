"""Backfill of context->raw provenance into legacy meta.npz (scripts.backfill_provenance).

Runs on the Mac WITHOUT the raw DICOM by mocking _crop_bounds; the real raw read is exercised only
when Shu runs it on PC2. Here we lock the two things that must be right regardless of the raw:
the block_reduce shape formula, and the verify-then-write-preserving logic.
"""
import numpy as np
import pytest

from cc_reservoir.scripts import backfill_provenance as bp
from cc_reservoir.measure.context import place_seg_in_raw


def test_expected_context_shape_matches_block_reduce():
    """expected_context_shape must equal skimage.block_reduce's output shape (build's downsample)."""
    from skimage.measure import block_reduce
    cases = [((0, 0, 0), (40, 48, 30), (2, 2, 1)),
             ((3, 5, 1), (3 + 39, 5 + 47, 1 + 30), (2, 2, 1)),   # odd extents -> ceil, not floor
             ((0, 0, 0), (7, 9, 4), (2, 2, 1))]
    for lo, hi, ds in cases:
        crop = np.zeros((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), np.uint8)
        assert bp.expected_context_shape(lo, hi, ds) == block_reduce(crop, (ds[0], ds[1], 1), np.max).shape


def _make_legacy_ctx(tmp_path, tag, hu_shape):
    """A legacy context dir (no provenance) with some existing meta keys to check preservation."""
    d = tmp_path / f"context4d_{tag}"; d.mkdir()
    np.savez_compressed(d / "f0.npz", hu=np.zeros(hu_shape, np.int16), seg=np.zeros(hu_shape, np.uint8))
    np.savez_compressed(d / "meta.npz", voxel=np.array([1.562, 1.562, 0.5]), peak_i=0,
                        duct_centroid=np.array([1., 2., 3.]), seedless=False,
                        lab_cc=np.ones((2, 2, 2), np.uint8))       # existing keys that must survive
    return d


def test_backfill_writes_provenance_and_preserves_keys(tmp_path, monkeypatch):
    spec = bp.WORKING_SET[0]                                       # 07_20_22_data / Baseline/Acq6
    tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
    hu_shape = (20, 24, 30)
    d = _make_legacy_ctx(tmp_path, tag, hu_shape)
    # crop that block-reduces (2x in-plane) exactly to hu_shape -> verification passes
    monkeypatch.setattr(bp, "_crop_bounds", lambda s, a: ((5, 5, 0), (45, 53, 30), (60, 60, 40)))
    assert bp.backfill(str(tmp_path), archive="IGNORED") is True

    meta = dict(np.load(d / "meta.npz", allow_pickle=True))
    assert tuple(meta["crop_lo"]) == (5, 5, 0) and tuple(meta["crop_hi"]) == (45, 53, 30)
    assert tuple(meta["downsample"]) == (2, 2, 1) and tuple(meta["raw_full_shape"]) == (60, 60, 40)
    assert str(meta["raw_session"]) == spec.session and str(meta["raw_sub"]) == spec.subpath
    # existing keys preserved, untouched
    assert np.array_equal(meta["duct_centroid"], [1., 2., 3.]) and meta["lab_cc"].shape == (2, 2, 2)
    # and the written provenance actually drives place_seg_in_raw
    seg = np.zeros(hu_shape, np.uint8); seg[0, 0, 0] = 1
    rm = place_seg_in_raw(seg, meta)
    assert rm.shape == (60, 60, 40) and set(map(tuple, np.argwhere(rm == 1))) == {(5, 5, 0), (5, 6, 0), (6, 5, 0), (6, 6, 0)}


def test_backfill_from_precomputed_bounds_no_raw(tmp_path):
    """The split path: apply crop bounds from a JSON dict (computed elsewhere, e.g. the Mac) with NO
    raw read here — so the write half can run where only the contexts live (PC2)."""
    spec = bp.WORKING_SET[0]
    tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
    d = _make_legacy_ctx(tmp_path, tag, (20, 24, 30))
    bounds = {tag: {"lo": [5, 5, 0], "hi": [45, 53, 30], "full": [60, 60, 40],
                    "session": spec.session, "sub": spec.subpath}}
    assert bp.backfill(str(tmp_path), bounds=bounds) is True     # no _crop_bounds / raw touched
    meta = dict(np.load(d / "meta.npz", allow_pickle=True))
    assert tuple(meta["crop_lo"]) == (5, 5, 0) and tuple(meta["raw_full_shape"]) == (60, 60, 40)
    assert str(meta["raw_sub"]) == spec.subpath


def test_backfill_refuses_to_write_on_shape_mismatch(tmp_path, monkeypatch):
    """If the recomputed crop doesn't reduce to the stored hu.shape (non-default pads), write NOTHING."""
    spec = bp.WORKING_SET[0]
    tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
    d = _make_legacy_ctx(tmp_path, tag, (20, 24, 30))
    monkeypatch.setattr(bp, "_crop_bounds", lambda s, a: ((0, 0, 0), (99, 99, 99), (200, 200, 200)))
    assert bp.backfill(str(tmp_path), archive="IGNORED") is False   # warn -> non-zero exit
    assert "crop_lo" not in np.load(d / "meta.npz").files            # meta left untouched


def test_backfill_is_idempotent(tmp_path, monkeypatch):
    spec = bp.WORKING_SET[0]
    tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
    d = _make_legacy_ctx(tmp_path, tag, (20, 24, 30))
    monkeypatch.setattr(bp, "_crop_bounds", lambda s, a: ((5, 5, 0), (45, 53, 30), (60, 60, 40)))
    bp.backfill(str(tmp_path), archive="IGNORED")
    # second run: already-present, must not error or change anything
    m1 = dict(np.load(d / "meta.npz", allow_pickle=True))
    assert bp.backfill(str(tmp_path), archive="IGNORED") is True
    m2 = dict(np.load(d / "meta.npz", allow_pickle=True))
    assert tuple(m1["crop_lo"]) == tuple(m2["crop_lo"])
