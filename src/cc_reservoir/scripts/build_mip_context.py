"""Lightweight whole-body context for the MIP panel: full-body HU + the tv_peak_lumen CC/TD seg per
frame, and NOTHING else. The full build_context_viz also computes a skeleton bone-mask, a 6-frame
temporal baseline, and marching-cubes meshes — all of which blow up on a whole-body crop (the spine
is a giant connected component). The MIP panel reads only hu + seg + voxel/peak, so we skip the rest
and build in ~loading-time. Same SSoT seg (tv_peak_lumen) as build_context_viz's DICOM path.

  CC_ARCHIVE=<p> CC_SESSION=<s> CC_SUB=<a> CC_CTX_OUT=<dir> CC_XY_PAD=.. CC_Z_CAUDAL_PAD=.. \
    CC_Z_CRANIAL_PAD=.. PYTHONPATH=src python3 -m cc_reservoir.scripts.build_mip_context
"""
import os

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import block_reduce

from cc_reservoir.io.workingset import WORKING_SET, ARCHIVE, AcqSpec
from cc_reservoir.io.masks import load_cc_mask
from cc_reservoir.measure.patch import load_cc_td_patch_dicom, load_cc_td_patch_seedless
from cc_reservoir.measure.lumen import (tv_peak_lumen, split_caudal, fill_extension, fill_nodes,
                                        caliber_ratio_boundary_z, caudal_cistern_anchor,
                                        reclaim_axial_cistern)

SESSION = os.environ.get("CC_SESSION", "07_20_22_data")
SUB = os.environ.get("CC_SUB", "Baseline/Acq6")
OUT = os.environ.get("CC_CTX_OUT", "/tmp/cc_out"); os.makedirs(OUT, exist_ok=True)
tag = f"{SESSION}_{SUB.replace('/', '_')}"   # full nesting (2023 acqs nest AcqNN/AcqMM) -> unique tag
# No upper bound. The old 3071 ceiling was the 12-bit CT convention (0..4095 stored + intercept
# -1024) — but this Toshiba writes signed 16-bit HU with slope 1 / intercept 0, and the Lipiodol-filled
# duct genuinely exceeds it (Acq6 f2: 8.3% of the cistern above 3071, raw max 4795), so it was
# truncating the duct's bright core. The -1024 floor stays: sub -1024 is the padding outside the
# circular reconstruction FOV, not tissue.
HU_CLIP = (-1024, None)
# whole-body view: store the body HU at CC_DOWNSAMPLE_XY-reduced in-plane resolution (the seg is
# computed at full res first, then the duct label is MAX-pooled so the thin tube survives) — keeps a
# full-body context ~4x smaller without losing the duct overlay. z stays full (the duct's long axis).
_DS = max(int(os.environ.get("CC_DOWNSAMPLE_XY", "1")), 1)


def _ds(arr, how):
    return block_reduce(arr, (_DS, _DS, 1), how) if _DS > 1 else arr
# use the audited WORKING_SET entry if present (carries flows), else build a bare spec from env so any
# seeded acq (2023 nested, unprocessed 2022) batches without a hard-coded list entry.
_match = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB]
spec = _match[0] if _match else AcqSpec(SESSION, SUB, os.environ.get("CC_CONDITION", "baseline"), 0.0, 0.0)

# SEEDLESS (CC_SEEDLESS=1): locate the duct from the contrast itself, with no lab mask as a seed.
# The lab seg is then loaded only as a comparison OVERLAY below, not as a driver.
SEEDLESS = os.environ.get("CC_SEEDLESS", "0") == "1"
tight = load_cc_td_patch_seedless(spec) if SEEDLESS else load_cc_td_patch_dicom(spec)
vox = tuple(float(v) for v in tight.voxel_mm); pk = int(tight.peak_i); full = tight.vols[pk].shape
print(f"{tag}: crop {full}, peak {pk}, voxel {vox}", flush=True)
_ne = max(1, len(tight.vols) // 3)   # early-frame count for the fill signature: 2 for 6-frame, 1 for the 2-frame 2023 series

# adaptive connectivity threshold = a fraction of the duct's OWN peak contrast (the seed's 80th pct),
# kept LOW so the dim ascending TD survives (0.4 dropped Acq6's TD); the bone is rejected by the mask
# below, not by the threshold, so the threshold no longer has to sit above bone.
_frac = float(os.environ.get("CC_THRESH_FRAC", "0.3")); _tmax = float(os.environ.get("CC_THRESH_MAX", "600"))
T_adapt = float(np.clip(_frac * np.percentile(tight.vols[pk][tight.mask], 80), 280.0, _tmax))
# bone is bright but STATIC (its core doesn't fill); the CC/TD fill over the series (large temporal
# range). Carve out static-bright voxels so the connectivity can't bridge the duct into the spine.
# DILATE the core mask to also catch the moving RIM — respiration sweeps the bone edge a few voxels,
# and that rim has a large temporal range (looks like "filling") so the core mask alone leaks it
# (Acq7/10 residual length). The duct sits a soft-tissue gap anterior to the spine, so a 3-vox dilation
# reaches the rim but not the duct.
_mx = np.maximum.reduce(tight.vols); _mn = np.minimum.reduce(tight.vols)
static_bone = ((_mx - _mn) < 300.0) & (_mx > 400.0)
_bd = int(os.environ.get("CC_BONE_DILATE", "0"))   # 0: the duct hugs the spine, so even a 3-vox dilation
#                          swallowed the TD (Acq6 -> 0) while NOT fixing the bleed; the un-dilated core is right
if _bd > 0:
    static_bone = binary_dilation(static_bone, iterations=_bd)
print(f"  adaptive HU thresh {T_adapt:.0f} | static-bright(bone) {int(static_bone.sum())} vox", flush=True)
master = tv_peak_lumen(tight.vols[pk], tight.mask, hu_thresh=T_adapt, exclude=static_bone)
# extend the TD cranially along the contrast bolus front: the dim ascending TD hugs the spine and looks
# identical to the moving bone rim in any single frame, but the TD FILLS over the series (dark->bright)
# while the rim only oscillates. fill_extension grows up on mean(late)-mean(early), self-limiting where
# the bolus stopped climbing — recovering the +cm the peak-HU connectivity drops at the dim front.
_ext = fill_extension(tight.vols, master, exclude=static_bone, n_early=_ne,
                      fill_thresh=float(os.environ.get("CC_FILL_THRESH", "200")))
master = master | _ext
print(f"  fill-extension +{int(_ext.sum())} vox up the bolus front", flush=True)
# para-aortic / renal-hilar lymph NODES off the duct (intranodal lymphangiography fills nodes upstream
# of the cistern). fill-signature keeps them to genuinely enhancing blobs; a raw-HU connectivity would
# flood the spine/body. Label 3, shown like the duct (only the part opacified at each frame).
nodes_mask = fill_nodes(tight.vols, master, exclude=static_bone, n_early=_ne,
                        fill_thresh=float(os.environ.get("CC_FILL_THRESH", "200")))
print(f"  fill-nodes {int(nodes_mask.sum())} vox (para-aortic/renal lymph nodes)", flush=True)
# reclaim the on-axis cistern that fill_nodes over-grabbed in a confluent mass (angiotensin Acq8/12):
# node-labeled enhancement within CC_AXIS_R_MM of the duct centerline is the CC → merged into the duct
# (only lateral blobs stay nodes). r=14mm pulls in the cistern's side-lobe (Shu: the lower-orange左叶 is CC).
_nm0 = int(nodes_mask.sum())
master, nodes_mask = reclaim_axial_cistern(master, nodes_mask, vox,
                                           r_mm=float(os.environ.get("CC_AXIS_R_MM", "14.0")))
print(f"  reclaim axial cistern: nodes {_nm0}->{int(nodes_mask.sum())}, master ->{int(master.sum())}", flush=True)
# caudal CC anchor: drop the thin feeding lumbar trunk, referencing the CAUDAL region's own peak area.
_zc = np.where(master.any(axis=(0, 1)))[0]; _z_caud = int(_zc.min()) if len(_zc) else 0
cc_caud = caudal_cistern_anchor(master, vox, frac=float(os.environ.get("CC_CISTERN_FRAC", "0.35")))
# CC/TD boundary = the OBJECTIVE Loukas rule when it flags a discrete dilated CC, else the length
# anchor. A geometric "dilated sac" boundary was tried and reverted: it cannot include a tapering
# cistern (Acq7) without fabricating one for a thin caudal (Acq16) — the boundary is inherently
# subjective (literature; Shu), so report it objectively + the disagreement, don't tune a geometry.
zb_anchor = int(np.clip(_z_caud + round(float(os.environ.get("CC_LEN_CM", "8.0")) * 10.0 / vox[2]),
                        0, master.shape[2] - 1))
zb_loukas, _diag = caliber_ratio_boundary_z(master, vox,
                                            ratio=float(os.environ.get("CC_BOUNDARY_RATIO", "2.0")),
                                            z_start=cc_caud)
zb = zb_loukas if _diag["caudal_is_cc"] else zb_anchor
print(f"  cc_caud {cc_caud} | zb {zb} (CC z{cc_caud}-{zb}, {(zb-cc_caud)*vox[2]:.0f}mm) | Loukas zb "
      f"{zb_loukas} (caud {_diag['caudal_mm']:.1f}mm, >2xTD={_diag['caudal_is_cc']}) | anchor zb {zb_anchor}",
      flush=True)


def seg_of(vol):
    """Opacified part of the fixed CC/TD/node regions at this frame — restricted to the master/nodes so
    the per-frame seg can never wander into the spine (a pre-contrast frame comes out empty)."""
    bright = np.clip(vol, *HU_CLIP) > 0.6 * T_adapt
    lumen = master & bright
    cc, td = split_caudal(lumen, zb)
    cc[:, :, :cc_caud] = False
    s = np.zeros(full, np.uint8)
    s[td] = 2; s[cc] = 1; s[nodes_mask & bright] = 3
    return s


tpdir = os.path.join(OUT, f"context4d_{tag}"); os.makedirs(tpdir, exist_ok=True)
for i in range(len(tight.vols)):
    _hu = np.clip(tight.vols[i], *HU_CLIP)
    # int16 storage has no upper clip to save it now: wrap silently and the duct's core reads NEGATIVE.
    assert _hu.max() <= np.iinfo(np.int16).max, f"HU {_hu.max():.0f} overflows int16 storage"
    hu = _ds(_hu.astype(np.int16), np.mean).astype(np.int16)
    seg = _ds(seg_of(tight.vols[i]), np.max).astype(np.uint8)   # max-pool keeps the thin duct label
    np.savez_compressed(os.path.join(tpdir, f"f{i}.npz"), hu=hu, seg=seg)
    print(f"  frame {i}: CC {int((seg == 1).sum())} TD {int((seg == 2).sum())} vox saved", flush=True)
    del hu, seg
duct_centroid = (np.argwhere(master).mean(0) * np.array(vox)).astype(np.float32)
# lab (Vitrea) CC/TD seg, kept ONLY as a static comparison overlay for the MIP panel — NOT a driver
# of our seg. Seeded mode: tight.mask IS the cropped lab seed. Seedless mode: tight.mask is OUR
# seedless seed, so load the lab CC|TD separately and place it on the crop grid (it is abdomen-only,
# so it stops at the diaphragm and does not reach the cranial TD).
def _lab_on_crop(fname):
    """Place one lab (Vitrea) mask on the crop grid — SEPARATE CC and TD, never unioned. The lab MAT
    is abdomen-only (caudal slices) so it stops at the diaphragm."""
    out = np.zeros(full, np.uint8)
    p = os.path.join(ARCHIVE, SESSION, SUB, "SEGMENT_dcm", fname)
    if os.path.exists(p):
        _m = load_cc_mask(p).astype(bool)
        lo, hi = tight.lo, tight.hi
        ze = min(int(hi[2]), _m.shape[2])
        if ze > lo[2]:
            out[:, :, :ze - lo[2]] = _m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:ze]
    return out


lab_cc = _ds(_lab_on_crop("CC_dcm_01.mat"), np.max).astype(np.uint8)
lab_td = _ds(_lab_on_crop("TD_dcm_01.mat"), np.max).astype(np.uint8)
print(f"  lab overlay: CC {int((lab_cc > 0).sum())} vox, TD {int((lab_td > 0).sum())} vox "
      "(separate masks, comparison only)", flush=True)
vox = (vox[0] * _DS, vox[1] * _DS, vox[2])                      # in-plane voxel grows with the downsample
np.savez_compressed(os.path.join(tpdir, "meta.npz"), voxel=np.array(vox),
                    n_frames=len(tight.vols), peak_i=pk, duct_centroid=duct_centroid,
                    boundary_z=zb, boundary_loukas=zb_loukas, boundary_anchor=zb_anchor,
                    cc_caud=cc_caud, td_ref_mm=_diag["td_ref_mm"],
                    cc_caudal_mm=_diag["caudal_mm"], caudal_is_cc=_diag["caudal_is_cc"],
                    seedless=SEEDLESS, lab_cc=lab_cc, lab_td=lab_td,
                    # --- context->raw provenance: seg maps back to the raw DICOM with NO rebuild ---
                    # raw_index = crop_lo + downsample * context_index (see measure.context.place_seg_in_raw).
                    # This is the SSoT that build previously discarded (crop offset + downsample + source).
                    crop_lo=np.array(tight.lo, np.int32), crop_hi=np.array(tight.hi, np.int32),
                    downsample=np.array([_DS, _DS, 1], np.int32),
                    raw_full_shape=np.array(tight.full_shape, np.int32),
                    raw_session=SESSION, raw_sub=SUB)
print(f"saved {tpdir}/  ({len(tight.vols)} frames + meta, no bone/meshes)", flush=True)
