"""Single producer for a cropped CC working patch and its enhancement field.

Every CC visualization / volume script needs the same upstream steps: load the CC
and TD masks, load all opacified HU volumes, crop everything to the CC bounding box
(so binary_dilation runs on a small patch, not the full 512^3 volume — see project
memory), pick the peak-opacification frame, and form the background-subtracted
enhancement field inside a CC search region with the contiguous TD excluded.

Keeping that here means the renderers, the mesh builder, the integrated-HU volume,
and the TD-boundary sensitivity all derive from one definition instead of four
divergent copies.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np

from cc_reservoir.io import REAL_VOXEL_MM
from cc_reservoir.io.masks import load_cc_mask
from cc_reservoir.io.volumes import load_hu_volume, mask_bbox, crop_to_bbox
from cc_reservoir.io.workingset import ARCHIVE
from cc_reservoir.measure.roi import dilated_mask_roi
from cc_reservoir.measure.baseline import ring_baseline


@dataclass
class CCPatch:
    mask: np.ndarray          # CC binary mask, cropped to bbox
    td: np.ndarray            # TD binary mask on the same crop (all-False if absent)
    vols: list                # HU volumes per timepoint, cropped to bbox
    peak_i: int               # index of peak-opacification frame (max mean HU in CC)
    voxel_mm: tuple           # (vx, vy, vz) at THIS (pre-downsample) crop grid
    lo: tuple                 # bbox lower corner in the full volume (for traceback)
    hi: tuple                 # bbox upper corner in the full volume
    full_shape: tuple = None  # the raw DICOM volume shape the crop came from (seg->raw provenance)


def _load_acq(spec, archive, bbox_source, pad, voxel_mm):
    """Shared loader: CC/TD masks + opacified HU volumes, cropped to a bbox derived from
    either the CC mask alone ('cc') or the CC|TD union ('union'). The union bbox is needed
    when the TD (which extends cranially well past the CC) must stay in frame."""
    acq = os.path.join(archive, spec.session, spec.subpath)
    mask_full = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
    td_path = os.path.join(acq, "SEGMENT_dcm", "TD_dcm_01.mat")
    td_full = load_cc_mask(td_path) if os.path.exists(td_path) else None
    have_td = td_full is not None and td_full.shape == mask_full.shape

    bbox_mask = (mask_full | td_full) if (bbox_source == "union" and have_td) else mask_full
    lo, hi = mask_bbox(bbox_mask, pad=pad)
    mask = crop_to_bbox(mask_full, lo, hi)
    td = crop_to_bbox(td_full, lo, hi) if have_td else np.zeros_like(mask)

    vols = []
    for m in sorted(glob.glob(os.path.join(acq, "MAT", "*01.mat"))):
        v = load_hu_volume(m)
        if v.shape == mask_full.shape:
            vols.append(crop_to_bbox(v, lo, hi))
    if not vols:
        raise FileNotFoundError(f"no MAT/*01.mat volumes matching mask shape in {acq}")

    peak_i = int(np.argmax([v[mask].mean() for v in vols]))
    return CCPatch(mask=mask, td=td, vols=vols, peak_i=peak_i, voxel_mm=voxel_mm, lo=lo, hi=hi)


def load_cc_patch(spec, archive=ARCHIVE, pad=18, voxel_mm=REAL_VOXEL_MM):
    """Load one acquisition cropped to the CC bbox (CC-centric analyses)."""
    return _load_acq(spec, archive, "cc", pad, voxel_mm)


def load_cc_td_patch(spec, archive=ARCHIVE, pad=16, voxel_mm=REAL_VOXEL_MM):
    """Load one acquisition cropped to the CC|TD union bbox (keeps the whole CC+TD lumen in
    frame for per-timepoint lumen segmentation and the caudal-anchored CC/TD split)."""
    return _load_acq(spec, archive, "union", pad, voxel_mm)


def load_cc_td_patch_dicom(spec, archive=ARCHIVE, xy_pad=60, z_caudal_pad=12, z_cranial_pad=320,
                           voxel_mm=REAL_VOXEL_MM):
    """Load one acquisition from the FULL DICOM (1401 slices) instead of the lab's abdomen-only MAT
    (701 slices, which cuts the duct off at the diaphragm). The lab CC|TD seed lives in the caudal
    (abdomen) part of the volume and LOCATES the duct; the crop keeps its x,y plus a generous margin
    and extends cranially past the diaphragm (z_cranial_pad) so the cut-off cranial CC + the ascending
    TD stay in frame. Frames are cropped on load (one full 1401-slice frame in RAM at a time)."""
    from cc_reservoir.io.dicom import load_dicom_series, dicom_series_dirs
    xy_pad = int(os.environ.get("CC_XY_PAD", xy_pad))                       # widen the crop for a whole-body
    z_caudal_pad = int(os.environ.get("CC_Z_CAUDAL_PAD", z_caudal_pad))      # view; the seg is seed-connected
    z_cranial_pad = int(os.environ.get("CC_Z_CRANIAL_PAD", z_cranial_pad))   # so a wider box doesn't change it
    acq = os.path.join(archive, spec.session, spec.subpath)
    cc = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
    td_path = os.path.join(acq, "SEGMENT_dcm", "TD_dcm_01.mat")
    td_seed = load_cc_mask(td_path) if os.path.exists(td_path) else np.zeros_like(cc)
    if td_seed.shape != cc.shape:                                # lab CC/TD masks can differ in z-extent;
        tz = min(cc.shape[2], td_seed.shape[2])                  # they share x,y and the z-origin, so align on z
        _t = np.zeros_like(cc); _t[:, :, :tz] = td_seed[:, :, :tz]; td_seed = _t
    seed_abd = cc | td_seed                                       # (X, Y, Z_abdomen) lab seed
    series = dicom_series_dirs(acq)
    if not series:
        raise FileNotFoundError(f"no DICOM series in {acq}")
    # place the abdomen seed into the full volume grid (the seed is the caudal slices)
    first = load_dicom_series(series[0])
    full_shape = first.shape
    seed = np.zeros(full_shape, bool); seed[:, :, :seed_abd.shape[2]] = seed_abd
    X = np.where(seed.any(axis=(1, 2)))[0]; Y = np.where(seed.any(axis=(0, 2)))[0]; Z = np.where(seed.any(axis=(0, 1)))[0]
    lo = (max(int(X.min()) - xy_pad, 0), max(int(Y.min()) - xy_pad, 0), max(int(Z.min()) - z_caudal_pad, 0))
    hi = (min(int(X.max()) + xy_pad + 1, full_shape[0]), min(int(Y.max()) + xy_pad + 1, full_shape[1]),
          min(int(Z.max()) + z_cranial_pad + 1, full_shape[2]))
    sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    mask = seed[sl]
    vols = [first[sl].astype(np.float32)]
    del first
    for s in series[1:]:
        v = load_dicom_series(s)
        if v.shape == full_shape:                                # some acqs have timepoints at a different
            vols.append(v[sl].astype(np.float32))                # matrix/z; skip them (else vols is ragged)
        del v
    # peak = the bolus-peak frame by TEMPORAL ENHANCEMENT (frame HU - per-voxel static baseline) in the
    # duct seed, NOT raw HU. The contrast bolus lives in the ENHANCEMENT (E rises ~11->300 HU), not in
    # the >threshold voxel COUNT, which is flat: most bright voxels are static (bone rim, plateaued
    # abdomen) and stay bright every frame, so a raw-HU / count peak is dominated by them and lands
    # arbitrarily (frame 0). Subtracting the per-voxel min cancels those exactly and surfaces the bolus,
    # robust both here and for low-contrast acqs (where a raw-HU peak picks a bone-bright pre-contrast frame).
    vmin = np.minimum.reduce(vols)
    peak_i = int(np.argmax([float((v - vmin)[mask].mean()) for v in vols]))
    return CCPatch(mask=mask, td=np.zeros_like(mask), vols=vols, peak_i=peak_i, voxel_mm=voxel_mm,
                   lo=lo, hi=hi, full_shape=tuple(int(s) for s in full_shape))


def load_cc_td_patch_seedless(spec, archive=ARCHIVE, xy_pad=60, z_caudal_pad=12, z_cranial_pad=320,
                              voxel_mm=REAL_VOXEL_MM):
    """SEEDLESS variant of load_cc_td_patch_dicom — uses NO lab mask, so the duct is located
    from the contrast alone. Loads the dynamic frames into a generous para-vertebral geometric box (central LR
    column, posterior-biased AP, full z — wide enough to contain the duct without any prior), locates
    the duct from the contrast itself via ``seedless_duct_seed`` (intranodal injection opacifies only
    the lymphatics), and sub-crops to it. Returns a CCPatch whose ``mask`` is the seedless duct seed
    (the connectivity anchor for tv_peak_lumen). Same env crop pads as the seeded loader."""
    from cc_reservoir.io.dicom import load_dicom_series, dicom_series_dirs
    from cc_reservoir.measure.lumen import seedless_duct_seed
    xy_pad = int(os.environ.get("CC_XY_PAD", xy_pad))
    z_caudal_pad = int(os.environ.get("CC_Z_CAUDAL_PAD", z_caudal_pad))
    z_cranial_pad = int(os.environ.get("CC_Z_CRANIAL_PAD", z_cranial_pad))
    acq = os.path.join(archive, spec.session, spec.subpath)
    series = dicom_series_dirs(acq)
    if not series:
        raise FileNotFoundError(f"no DICOM series in {acq}")
    first = load_dicom_series(series[0])
    X, Y, Z = first.shape
    # generous seedless pre-crop: the swine duct is para-vertebral (LR-centered, just anterior to the
    # vertebral bodies = posterior half of the AP axis). Wide enough that the duct cannot be clipped;
    # the actual localization is seedless_duct_seed within this box, not this geometric guess.
    g = (slice(int(0.28 * X), int(0.97 * X)),
         slice(max(Y // 2 - 130, 0), min(Y // 2 + 130, Y)),
         slice(0, Z))
    vols_g = [first[g].astype(np.float32)]
    del first
    for s in series[1:]:
        v = load_dicom_series(s)
        if v.shape == (X, Y, Z):                                  # skip a timepoint of a different matrix/z
            vols_g.append(v[g].astype(np.float32))
        del v
    seed_g = seedless_duct_seed(vols_g, voxel_mm)
    if not seed_g.any():
        raise RuntimeError(f"seedless: no duct-like structure found in {acq}")
    idx = np.argwhere(seed_g)
    lo, hi = idx.min(0), idx.max(0)
    sl = (slice(max(int(lo[0]) - xy_pad, 0), min(int(hi[0]) + xy_pad + 1, seed_g.shape[0])),
          slice(max(int(lo[1]) - xy_pad, 0), min(int(hi[1]) + xy_pad + 1, seed_g.shape[1])),
          slice(max(int(lo[2]) - z_caudal_pad, 0), min(int(hi[2]) + z_cranial_pad + 1, seed_g.shape[2])))
    mask = seed_g[sl]
    vols = [v[sl] for v in vols_g]
    vmin = np.minimum.reduce(vols)
    peak_i = int(np.argmax([float((v - vmin)[mask].mean()) if mask.any() else float(v.mean()) for v in vols]))
    go = (g[0].start, g[1].start, g[2].start)              # full-volume corner of the crop (traceback)
    lo_full = tuple(go[k] + sl[k].start for k in range(3))
    hi_full = tuple(go[k] + sl[k].stop for k in range(3))
    return CCPatch(mask=mask, td=np.zeros_like(mask), vols=vols, peak_i=peak_i, voxel_mm=voxel_mm,
                   lo=lo_full, hi=hi_full, full_shape=(int(X), int(Y), int(Z)))


def enhancement_field(patch, search_mm=8.0, td_excl_mm=1.5, fixed_dil_mm=2.0,
                      ring_inner_mm=1.0, ring_outer_mm=4.0):
    """Per-timepoint background-subtracted enhancement inside the CC search region.

    The search region is the CC dilated by ``search_mm`` with the TD (dilated by
    ``td_excl_mm``) removed, so the contiguous duct does not leak into the CC field.
    Background is a ring around a tight CC dilation. Returns (enh_list, search_mask).
    """
    mask, td, vox = patch.mask, patch.td, patch.voxel_mm
    fixed = dilated_mask_roi(mask, fixed_dil_mm, vox)
    search = dilated_mask_roi(mask, search_mm, vox)
    if td.sum():
        search = search & ~dilated_mask_roi(td, td_excl_mm, vox)
    enh = [np.where(search, v - ring_baseline(v, fixed, ring_inner_mm, ring_outer_mm, vox), 0.0)
           for v in patch.vols]
    return enh, search
