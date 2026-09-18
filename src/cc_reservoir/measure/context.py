"""Place a tight-patch mask into larger (full-volume) coordinates. The tight CC/TD segmentation
is ROI-bounded, so relocating it — not recomputing — is exact."""
import numpy as np


def place_into(mask_tight, lo_tight, lo_regional, shape_regional):
    """Write a tight-patch mask into a larger array at offset (lo_tight - lo_regional)."""
    out = np.zeros(shape_regional, bool)
    o = tuple(int(a - b) for a, b in zip(lo_tight, lo_regional))
    s = mask_tight.shape
    out[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = mask_tight
    return out


# --------------------------------------------------------------------------- #
# context (downsampled crop) -> raw DICOM grid.
# The context `seg` in cc_contexts/*/f*.npz lives on a per-acq CROP of the raw volume that was then
# in-plane block-reduced. build_mip_context now records the inverse (crop_lo/crop_hi/downsample/
# raw_full_shape/raw_session/raw_sub in meta.npz), so the mask maps back to the raw with NO rebuild:
#     raw_index = crop_lo + downsample * context_index
# Each context voxel covers a `downsample`-sized block of raw. These are PURE (no DICOM read) and tested.
# --------------------------------------------------------------------------- #
def upsample_seg_to_crop(seg, downsample):
    """Undo the in-plane block-reduce: repeat each context voxel `downsample[a]` times per axis, landing
    the label back on the FULL-RES crop grid (the pre-downsample DICOM crop)."""
    ds = tuple(int(d) for d in downsample)
    return np.repeat(np.repeat(np.repeat(seg, ds[0], axis=0), ds[1], axis=1), ds[2], axis=2)


def context_index_to_raw(idx, crop_lo, downsample):
    """A context voxel index -> the lower corner of its block in the raw DICOM grid.
    The context voxel covers raw[c : c+downsample] per axis, where c = crop_lo + downsample*idx."""
    lo = tuple(int(v) for v in crop_lo); ds = tuple(int(v) for v in downsample)
    return tuple(lo[a] + int(idx[a]) * ds[a] for a in range(3))


def place_seg_in_raw(seg, meta):
    """Map a context `seg` (from f*.npz) onto the FULL raw DICOM grid using the provenance in meta.npz
    (crop_lo/crop_hi/downsample/raw_full_shape). PURE — no DICOM read. Returns a raw_full_shape LABEL
    array you index the raw volume with: `raw_hu[place_seg_in_raw(seg, meta) == 1]` for CC, etc."""
    lo = tuple(int(v) for v in meta["crop_lo"]); hi = tuple(int(v) for v in meta["crop_hi"])
    ds = tuple(int(v) for v in meta["downsample"]); full = tuple(int(v) for v in meta["raw_full_shape"])
    up = upsample_seg_to_crop(seg, ds)                            # context -> full-res crop grid
    out = np.zeros(full, seg.dtype)
    ce = tuple(min(hi[a], lo[a] + up.shape[a]) for a in range(3))  # clip the block-reduce edge pad (<ds)
    out[lo[0]:ce[0], lo[1]:ce[1], lo[2]:ce[2]] = up[:ce[0] - lo[0], :ce[1] - lo[1], :ce[2] - lo[2]]
    return out


def load_context_seg(context_dir, frame):
    """The seg for one context frame, PREFERRING the manual-edit sidecar `f<N>_edit.npz` — the same
    rule io.context.load_seg and the iPad painter use. Without this an edited seg would be silently
    ignored and you'd measure the automatic one."""
    import os
    base = os.path.join(context_dir, f"f{int(frame)}.npz")
    side = os.path.join(context_dir, f"f{int(frame)}_edit.npz")
    return np.load(side if os.path.exists(side) else base)["seg"]


def raw_values_under_seg(context_dir, frame=None, archive=None):
    """Read raw-DICOM HU values under a context's seg. RUNS WHERE THE RAW DICOM IS REACHABLE (the Mac
    with the share mounted, or PC2), not on a machine holding only cc_contexts. Uses the manual-edit
    sidecar when present, loads the raw series named by meta.raw_session/raw_sub (default the peak
    frame), maps that seg onto the raw grid, and returns {label_id: hu_values_1d}. Full-resolution
    values, provenance-exact, no rebuild."""
    import os
    from cc_reservoir.io.dicom import load_dicom_series, dicom_series_dirs
    from cc_reservoir.io.workingset import ARCHIVE
    meta = np.load(os.path.join(context_dir, "meta.npz"), allow_pickle=True)
    fi = int(meta["peak_i"]) if frame is None else int(frame)
    seg = load_context_seg(context_dir, fi)                        # your edit wins over the auto seg
    acq = os.path.join(archive or ARCHIVE, str(meta["raw_session"]), str(meta["raw_sub"]))
    raw = load_dicom_series(dicom_series_dirs(acq)[fi])           # full-res raw HU, SAME frame
    raw_mask = place_seg_in_raw(seg, meta)
    return {int(k): raw[raw_mask == k] for k in np.unique(raw_mask) if k != 0}
