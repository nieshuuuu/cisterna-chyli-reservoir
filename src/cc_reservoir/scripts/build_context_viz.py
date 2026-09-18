"""Build the context-viewer npz for one acquisition (run on Linux).

Layout the user asked for: the full volume at the duct's peak-opacification frame as a Horos-style
grayscale background, with four overlays —
  * iodinated lymph = CC (green) + TD (teal). We load the lab's *previous* manual segmentation
    (SEGMENT_dcm/CC_dcm_01.mat, TD_dcm_01.mat) when present so you can see it first and refine
    from there; otherwise we fall back to our integrated-HU CC/TD pipeline.
  * bone (gray) + kidney (gold) = HU-threshold landmarks that locate the duct anatomically.

  CC_ARCHIVE=<path> CC_SESSION=<s> CC_SUB=<a> CC_CTX_OUT=<dir> \
    PYTHONPATH=src python3 -m cc_reservoir.scripts.build_context_viz
"""
import glob
import os

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

from cc_reservoir.io.workingset import WORKING_SET, ARCHIVE
from cc_reservoir.io.volumes import load_hu_volume
from cc_reservoir.io.masks import load_cc_mask
from cc_reservoir.measure.patch import load_cc_td_patch, load_cc_td_patch_dicom
from cc_reservoir.measure.context import place_into
from cc_reservoir.measure.landmarks import bone_mask, kidney_mask
from cc_reservoir.measure.surface import mesh_from_mask as _mesh_from_mask
from cc_reservoir.measure.volume import integrated_hu_volume_region
from cc_reservoir.measure.lumen import (lumen_roi, static_baseline, fixed_lumen, frame_enhancement,
                                        local_opacified_ref, occupancy_field, tv_lumen, tv_peak_lumen,
                                        split_caudal)

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
OUT = os.environ.get("CC_CTX_OUT", "/tmp/cc_ctx"); os.makedirs(OUT, exist_ok=True)
tag = f"{SESSION}_{os.path.basename(SUB)}"
HU_CLIP = (-1024, 3071)
spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
acq = os.path.join(ARCHIVE, spec.session, spec.subpath)
DICOM = os.environ.get("CC_DICOM", "0") in ("1", "true")
if DICOM:
    tight = load_cc_td_patch_dicom(spec)
    vox = tuple(float(v) for v in tight.voxel_mm); pk = int(tight.peak_i)
    bg = np.clip(tight.vols[pk], *HU_CLIP).astype(np.int16); full = bg.shape
else:
    mats = sorted(glob.glob(os.path.join(acq, "MAT", "*01.mat")))
    tight = load_cc_td_patch(spec)
    vox = tuple(float(v) for v in tight.voxel_mm); pk = int(tight.peak_i)
    bg = load_hu_volume(mats[pk]).astype(np.int16); full = bg.shape


def to_ctx(seg):
    """Place a crop-coordinate seg into the context4d grid: identity in DICOM mode (the context4d IS
    the crop), else into the full-volume grid at the crop offset (the MAT path stores the full grid)."""
    return seg if DICOM else place_into(seg, tight.lo, (0, 0, 0), full)


def crop_lab(lab):
    """Crop the abdomen lab mask (its z is < 701, the MAT extent) to the DICOM crop region [lo:hi]."""
    out = np.zeros(full, bool); z_end = min(tight.hi[2], lab.shape[2])
    if z_end > tight.lo[2]:
        out[:, :, :z_end - tight.lo[2]] = lab[tight.lo[0]:tight.hi[0], tight.lo[1]:tight.hi[1], tight.lo[2]:z_end]
    return out


def mesh_from_mask(mask, **kw):
    """This acq's voxel size bound into the shared mesh builder (measure.surface)."""
    return _mesh_from_mask(mask, vox, **kw)


# --- iodinated lymph: compute BOTH our integrated-HU detection (the viewer's default) and the
# lab hand-mask, so the viewer can toggle between them. The two give different CC volumes
# (hand mask ~90-120 uL, integrated-HU ~265 uL); the viewer shows both side by side. ---
rois = lumen_roi(tight); base = static_baseline(tight)
master = tv_peak_lumen(tight.vols[pk], tight.mask) if DICOM else fixed_lumen(tight, rois, baseline=base)
# CC/TD split. CRANIAL boundary `zb` = caudal-most opacified point + CC_LEN_CM cm (Gomercic & Duras
# 2010 porcine cistern length); the cranial CC/TD boundary has no clean caliber landmark (diameter
# does not discriminate — AJP-Heart review 10.1152/ajpheart.00375.2022), so length is the anchor.
# CAUDAL boundary `cc_caud` = the caliber STEP-UP (the trunk confluence): below it the lumen is thin
# (~3 mm, an ascending lumbar trunk), at it the lumen dilates into the cistern. The pre-confluence
# trunk is NOT the cisterna — drop it so CC = the dilated cistern (post-confluence). The step-up is a
# clear caudal landmark (unlike the cranial boundary), so caliber IS defensible here.
_zc = np.where(master.any(axis=(0, 1)))[0]
_z_caud = int(_zc.min()) if len(_zc) else 0
zb = int(np.clip(_z_caud + round(float(os.environ.get("CC_LEN_CM", "8.0")) * 10.0 / float(vox[2])),
                 0, master.shape[2] - 1))
_area = np.convolve(master.sum(axis=(0, 1)).astype(float), np.ones(5) / 5.0, mode="same")
_pk_area = _area[_z_caud:zb + 1].max() if zb >= _z_caud else 0.0
_above = np.where(_area[:zb + 1] >= float(os.environ.get("CC_CISTERN_FRAC", "0.35")) * _pk_area)[0]
cc_caud = int(_above.min()) if len(_above) else _z_caud      # caudal cistern origin (trunk confluence)


def lymph_seg(enh, vol=None):
    """CC/TD = the lumen split: CC = the dilated cistern [cc_caud, zb] (cc_caud = the caliber step-up /
    trunk confluence; the thin pre-confluence trunk below is dropped), TD = cranial (z > zb). DICOM
    uses the peak-HU connected tube (whole-body, motion-proof); MAT uses the TV-FWHM enhancement lumen."""
    lumen = tv_peak_lumen(vol, tight.mask) if DICOM else tv_lumen(enh, master)
    cc, td = split_caudal(lumen, zb)
    cc[:, :, :cc_caud] = False
    return cc, td


e_pk = frame_enhancement(tight, pk, rois, base)
cc_t, td_t = lymph_seg(e_pk, tight.vols[pk])
cc_new = to_ctx(cc_t)
td_new = to_ctx(td_t)
vcc = integrated_hu_volume_region(cc_t, tight.vols[pk], vox)[0]
vtd = integrated_hu_volume_region(td_t, tight.vols[pk], vox)[0]

seg_dir = os.path.join(acq, "SEGMENT_dcm")
cc_lab = load_cc_mask(os.path.join(seg_dir, "CC_dcm_01.mat"))
tdp = os.path.join(seg_dir, "TD_dcm_01.mat")
td_lab = load_cc_mask(tdp) if os.path.exists(tdp) else None
if DICOM:
    cc_lab = crop_lab(cc_lab)
    td_lab = crop_lab(td_lab) if td_lab is not None else np.zeros(full, bool)
elif td_lab is None:
    td_lab = np.zeros(full, bool)

# --- HU landmarks: bone (clean) + kidney (off by default) ---
# Bone thresholds cleanly. Kidney does NOT here: this is lymphatic, not intravenous, contrast, so
# the kidneys never enhance and sit at liver/muscle HU — the posterior soft tissue is then one
# connected mass (~5.4M vox) that a threshold cannot split. Gated behind CC_KIDNEY=1.
duct = cc_new | td_new | cc_lab | td_lab
zc = np.where(duct.any(axis=(0, 1)))[0]
z_lo, z_hi = (int(zc.min()), int(zc.max())) if len(zc) else (0, full[2] - 1)
bone = bone_mask(bg)
kidney = (kidney_mask(bg, z_lo=z_lo - 30, z_hi=z_hi + 30)
          if os.environ.get("CC_KIDNEY", "0") in ("1", "true") else np.zeros(full, bool))


def make_seg(cc, td):
    """Composite label volume (context first, lymph on top so the duct stays visible)."""
    s = np.zeros(full, np.uint8); s[bone] = 4; s[kidney] = 3; s[cc] = 1; s[td] = 2
    return s


seg = make_seg(cc_new, td_new); seg_lab = make_seg(cc_lab, td_lab)
print(f"{tag}: peak frame {pk} | integrated-HU CC {vcc:.0f} uL  TD {vtd:.0f} uL  (paper CC~302)")
print(f"  new: CC {int(cc_new.sum())} TD {int(td_new.sum())} | lab: CC {int(cc_lab.sum())} "
      f"TD {int(td_lab.sum())} | bone {int(bone.sum())} vox")

cc_v, cc_f = mesh_from_mask(cc_new); td_v, td_f = mesh_from_mask(td_new)
cc_lab_v, cc_lab_f = mesh_from_mask(cc_lab); td_lab_v, td_lab_f = mesh_from_mask(td_lab)


def bone_mesh(down=2):
    """Faint spine surface for the 3D: crop bone to the duct z-band ±40, 2× downsample so the
    marching-cubes is small (context only, not a precise bone segmentation)."""
    zb0, zb1 = max(z_lo - 40, 0), min(z_hi + 41, full[2])
    sm = zoom(bone[:, :, zb0:zb1].astype(np.float32), 1.0 / down, order=0)
    v, f, _, _ = marching_cubes(gaussian_filter(sm, 0.6), 0.5, spacing=tuple(x * down for x in vox))
    v = v.astype(np.float32); v[:, 2] += zb0 * vox[2]    # z was the only cropped axis
    return v, f.astype(np.int32)


bone_v, bone_f = bone_mesh()
duct_centroid = (np.argwhere(cc_new | td_new).mean(0) * np.array(vox)).astype(np.float32)

if os.environ.get("CC_TIMEPOINTS", "0") in ("1", "true"):
    # 4D: one file per frame (full HU + integrated-HU duct seg + meshes) so we never hold all
    # frames in RAM at once, plus a meta.npz of the static arrays (bone, lab, palette). The duct
    # deforms with respiration; pre-contrast frames get an empty duct.
    tpdir = os.path.join(OUT, f"context4d_{tag}"); os.makedirs(tpdir, exist_ok=True)
    vvol = float(np.prod(vox))
    cc_master = master.copy(); cc_master[:, :, zb + 1:] = False; cc_master[:, :, :cc_caud] = False  # CC = cistern [cc_caud, zb]
    td_master = master.copy(); td_master[:, :, :zb + 1] = False   # TD = strictly cranial (z > zb);
    #                                          the thin pre-confluence trunk (z < cc_caud) is in neither
    enh_means = []; vol_rows = []
    for i in range(len(tight.vols)):
        hu_i = np.clip(tight.vols[i] if DICOM else load_hu_volume(mats[i]), *HU_CLIP).astype(np.int16)
        e_i = frame_enhancement(tight, i, rois, base); me = float(e_i[master].mean()); enh_means.append(me)
        so_i = local_opacified_ref(e_i, master)
        Vz = (occupancy_field(e_i, so_i) * master).sum(axis=(0, 1)) * vvol   # per-z PV-corrected ∫occ
        vol_rows.append([i, float(Vz[cc_caud:zb + 1].sum()), float(Vz[zb + 1:].sum()),
                         float(e_i[cc_master].mean()) if cc_master.any() else 0.0,
                         float(e_i[td_master].mean()) if td_master.any() else 0.0])
        if me >= 80:                                  # opacified -> solid single-channel this frame
            cc_i, td_i = lymph_seg(e_i, tight.vols[i])
        else:
            cc_i = np.zeros_like(master); td_i = np.zeros_like(master)
        cc_fi = to_ctx(cc_i); td_fi = to_ctx(td_i)
        cv_i, cf_i = mesh_from_mask(cc_fi); tv_i, tf_i = mesh_from_mask(td_fi)
        np.savez_compressed(os.path.join(tpdir, f"f{i}.npz"), hu=hu_i, seg=make_seg(cc_fi, td_fi),
                            cc_v=cv_i, cc_f=cf_i, td_v=tv_i, td_f=tf_i)
        del hu_i, cc_fi, td_fi                        # free before the next frame
        print(f"  frame {i}: enh {me:4.0f} HU  V_cc {vol_rows[-1][1]:4.0f} V_td {vol_rows[-1][2]:5.0f} uL")
    np.savez_compressed(os.path.join(tpdir, "meta.npz"), voxel=np.array(vox), n_frames=len(tight.vols),
                        peak_i=pk, duct_centroid=duct_centroid, bone_v=bone_v, bone_f=bone_f,
                        seg_lab=seg_lab, cc_lab_v=cc_lab_v, cc_lab_f=cc_lab_f,
                        td_lab_v=td_lab_v, td_lab_f=td_lab_f, enh_means=np.array(enh_means, np.float32),
                        vol_table=np.array(vol_rows, np.float32))
    print(f"saved {tpdir}/  ({len(tight.vols)} frame files + meta.npz)")
else:
    np.savez_compressed(os.path.join(OUT, f"context_{tag}.npz"), hu=np.clip(bg, *HU_CLIP).astype(np.int16),
                        seg=seg, seg_lab=seg_lab, voxel=np.array(vox),
                        cc_v=cc_v, cc_f=cc_f, td_v=td_v, td_f=td_f,
                        cc_lab_v=cc_lab_v, cc_lab_f=cc_lab_f, td_lab_v=td_lab_v, td_lab_f=td_lab_f,
                        bone_v=bone_v, bone_f=bone_f, duct_centroid=duct_centroid, peak_i=pk)
    print(f"saved {OUT}/context_{tag}.npz")
