"""Stage 1 of the PAINTED-mask CC rendering: build surfaces + the volumetric patch needed
for cross-sections, for every acquisition in PAINTED_SET. No OpenGL needed.

Differs from compute_cc_meshes.py only in where the CC/TD masks come from: the iPad
painter's `f<N>_edit.npz` sidecars in a full-res context, instead of the lab's Vitrea
`.mat` seeds. Everything downstream (enhancement field, half-max surface, integrated-HU
volume) is the SAME producer as the lab-mask path — this module only assembles a CCPatch.

The CC/TD masks are the UNION over painted frames (the anatomical extent). The surface
itself is not the mask: it is the sub-voxel half-max iso-surface of the enhancement field
at the peak frame, exactly as in measure.surface.

  CC_CONTEXTS=<dir> CC_PAINTED_MESHES=<out.npz> python -m cc_reservoir.scripts.compute_cc_meshes_painted
"""
import glob
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

from cc_reservoir.io.context import load_seg
from cc_reservoir.io.volumes import mask_bbox, crop_to_bbox
from cc_reservoir.io.workingset import CONTEXTS, PAINTED_SET
from cc_reservoir.measure.patch import CCPatch, enhancement_field
from cc_reservoir.measure.refine import refine_traced_lumen
from cc_reservoir.measure.surface import cc_half_max_surface
from cc_reservoir.measure.volume import robust_volume

OUT = os.environ.get("CC_PAINTED_MESHES", os.path.join(os.environ.get("TEMP", "/tmp"),
                                                       "cc_painted_meshes.npz"))
PAD = int(os.environ.get("CC_PAINT_PAD", 18))
UPSAMPLE = int(os.environ.get("CC_UPSAMPLE", 2))
LBL_CC, LBL_TD = 1, 2                       # viz.labels: 1 CC, 2 TD, 3 lymph, 4 bone


def frame_paths(ctx_dir):
    """`f<N>.npz` frames in timepoint order, excluding the `_edit` seg sidecars."""
    fs = [f for f in glob.glob(os.path.join(ctx_dir, "f*.npz")) if "_edit" not in os.path.basename(f)]
    return sorted(fs, key=lambda f: int(re.search(r"f(\d+)\.npz$", f).group(1)))


def load_painted_patch(ctx_dir, pad=PAD, cc_z_min=None, cc_z_max=None):
    """CCPatch from a painted context, cropped to the CC bbox. Raises if nothing painted.

    ``mask`` is the PEAK FRAME's painted CC, not the union over frames. The painted extent
    moves frame to frame (partly real filling, partly where the operator stopped tracing), so
    a union is an OR of several different tracings, not an anatomical extent — on 8_31_22/Acq6,
    whose paint swings 9455 -> 1669 voxels, the union inflated the integrated-HU volume ~17x
    over its siblings. One coherent tracing at best opacification is also what the downstream
    producers assume (measure.surface / measure.volume were written against the lab's
    single-timepoint mask). The union is still used to LOCATE the peak frame, and every frame's
    painted size is returned so the caller can flag this instability.
    """
    meta = np.load(os.path.join(ctx_dir, "meta.npz"), allow_pickle=True)
    voxel = tuple(float(v) for v in meta["voxel"])
    frames = frame_paths(ctx_dir)
    if not frames:
        raise FileNotFoundError(f"no frames in {ctx_dir}")
    # cc_z_min / cc_z_max are RAW-volume z indices; the context is a crop of the raw volume, so
    # shift them into context coordinates before applying. Only the CC label is clipped — the TD
    # label is left exactly as traced.
    z_lo = None if cc_z_min is None else int(cc_z_min) - int(meta["crop_lo"][2])
    z_hi = None if cc_z_max is None else int(cc_z_max) - int(meta["crop_lo"][2])

    segs, sizes, painted = {}, [], []
    cc_union = td_union = None
    for i, f in enumerate(frames):
        if not os.path.exists(os.path.splitext(f)[0] + "_edit.npz"):
            sizes.append(None)                          # unpainted frame
            continue
        seg = load_seg(f)
        c, t = seg == LBL_CC, seg == LBL_TD
        if z_lo is not None or z_hi is not None:
            c = c.copy()
            if z_lo is not None and z_lo > 0:
                c[:, :, :z_lo] = False
            if z_hi is not None and z_hi + 1 < c.shape[2]:
                c[:, :, z_hi + 1:] = False
        sizes.append(int(c.sum()))
        if not c.any() and not t.any():
            continue                                    # sidecar exists but holds no CC/TD (no-contrast acq)
        segs[i] = (c, t)
        cc_union = c if cc_union is None else (cc_union | c)
        td_union = t if td_union is None else (td_union | t)
        painted.append(i)
    if cc_union is None or not cc_union.any():
        raise ValueError(f"{os.path.basename(ctx_dir)}: no painted CC in any frame")

    lo, hi = mask_bbox(cc_union, pad=pad)
    vols = []
    for f in frames:
        with np.load(f) as d:
            vols.append(crop_to_bbox(d["hu"], lo, hi).astype(np.float32))

    # Peak = max mean TEMPORAL ENHANCEMENT in the CC (frame HU - per-voxel static baseline), not
    # max raw HU — same reasoning as measure.patch.load_cc_td_patch_dicom: static bright voxels
    # (bone rim, plateaued abdomen) dominate a raw-HU peak and land it arbitrarily.
    union_c = crop_to_bbox(cc_union, lo, hi)
    vmin = np.minimum.reduce(vols)
    peak_i = int(np.argmax([float((v - vmin)[union_c].mean()) for v in vols]))
    if peak_i not in segs:                              # peak landed on an unpainted frame
        peak_i = min(segs, key=lambda i: abs(i - peak_i))
    c, t = segs[peak_i]
    patch = CCPatch(mask=crop_to_bbox(c, lo, hi), td=crop_to_bbox(t, lo, hi), vols=vols,
                    peak_i=peak_i, voxel_mm=voxel, lo=lo, hi=hi,
                    full_shape=tuple(int(s) for s in cc_union.shape))
    # Full-frame peak masks too: the patch is cropped to the CC bbox, which amputates the TD
    # (it runs cranially well past the CC). Anything that has to SHOW the TD needs the union box.
    return patch, painted, sizes, (c, t)


def mask_surface(mask, voxel_mm, upsample=UPSAMPLE, sigma=0.8):
    """Marching-cubes surface of a BINARY hand mask, in the same mm frame as the half-max
    surface. This is the painted seg drawn literally — since 2026-07-27 it is also the
    PRIMARY rendered CC surface (the half-max surface is kept only for comparison).

    Anti-aliased before contouring: linear upsample + a light Gaussian, then the 0.5 level.
    A nearest-neighbour upsample of a binary mask contours into a hard voxel staircase, and
    that staircase is an artifact of the raster, not of where the operator traced. Blurring
    a binary field and cutting at 0.5 leaves the boundary where it was (the level set of a
    symmetric blur through a step is the step) while removing the blocking. Keep sigma well
    under the structure radius — the CC is ~4 mm across, so ~1 upsampled voxel is safe.
    """
    if not mask.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    sp = tuple(v / upsample for v in voxel_mm)
    field = gaussian_filter(zoom(mask.astype(np.float32), upsample, order=1), sigma=sigma)
    if field.max() <= 0.5:                      # a 1-voxel-thin trace can blur below the level
        field = field / field.max() * 0.9       # rescale rather than return an empty surface
    v, f, _, _ = marching_cubes(field, level=0.5, spacing=sp)
    return v.astype(np.float32), f.astype(np.int32)


def main():
    blob, rows = {}, []
    for k, spec in enumerate(PAINTED_SET):
        ctx = os.path.join(CONTEXTS, spec.context)
        patch, painted, sizes, (cc_full, td_full) = load_painted_patch(
            ctx, cc_z_min=spec.cc_z_min, cc_z_max=spec.cc_z_max)
        surf = cc_half_max_surface(patch, upsample=UPSAMPLE)
        enh, _ = enhancement_field(patch)
        V, spread, used = robust_volume(patch)

        blob[f"verts_{k}"] = surf.verts.astype(np.float32)
        blob[f"faces_{k}"] = surf.faces.astype(np.int32)
        blob[f"fill_{k}"] = surf.fill_vals.astype(np.float32)
        blob[f"spacing_{k}"] = np.array(surf.spacing, np.float32)
        blob[f"voxel_{k}"] = np.array(patch.voxel_mm, np.float32)
        blob[f"hu_{k}"] = patch.vols[patch.peak_i].astype(np.int16)      # peak-frame CT, for cross-sections
        blob[f"enh_{k}"] = enh[patch.peak_i].astype(np.float32)          # peak-frame enhancement field
        blob[f"mask_{k}"] = patch.mask.astype(np.uint8)
        blob[f"td_{k}"] = patch.td.astype(np.uint8)
        blob[f"peak_i_{k}"] = np.int32(patch.peak_i)
        blob[f"peak_enh_{k}"] = np.float32(surf.peak_enh)

        # The painted masks drawn as surfaces: CC to compare against the half-max surface,
        # TD so the two structures can be shown together (they are contiguous in life, and
        # the CC pipeline deliberately excludes the TD — you can only judge that split by
        # looking at both). Built on the CC|TD UNION box so the TD is not amputated, then
        # translated into the CC patch's mm frame so they can be drawn with everything else.
        # Refine the tracing against the local half-max boundary of the enhancement field,
        # then surface that too. Both are kept: the raw tracing is what the operator drew,
        # the refined mask is that tracing snapped to the data.
        # level_frac 0.25, NOT the textbook 0.5. Operator's call (2026-07-27): the boundary
        # should sit where the edge reads by eye, "as far out as possible while still covering
        # the boundary" — about midway between the raw tracing and the FWHM surface. Measured:
        # 0.5 leaves the boundary at radius 0.63 of the tracing, so midway is radius ~0.82,
        # i.e. ~66% of the traced volume; a sweep puts that at 0.25 (mean 67%). Calling this
        # FWHM would be wrong — it is a quarter-max, chosen for visual boundary coverage, and
        # it will read larger than a true partial-volume-corrected lumen.
        ref = refine_traced_lumen(patch.mask, enh[patch.peak_i], patch.voxel_mm,
                                  level_frac=0.25, floor_hu=40.0)
        rv, rf = mask_surface(ref.mask, patch.voxel_mm)
        blob[f"refined_v_{k}"], blob[f"refined_f_{k}"] = rv, rf
        blob[f"refined_mask_{k}"] = ref.mask.astype(np.uint8)
        blob[f"refined_levels_{k}"] = ref.levels.astype(np.float32)
        blob[f"refined_peaks_{k}"] = ref.peaks.astype(np.float32)
        blob[f"refined_vox_mm3_{k}"] = np.float32(ref.mask.sum() * np.prod(patch.voxel_mm))

        lo_u, hi_u = mask_bbox(cc_full | td_full, pad=PAD)
        shift = (np.asarray(lo_u) - np.asarray(patch.lo)) * np.asarray(patch.voxel_mm)
        mv, mf = mask_surface(crop_to_bbox(cc_full, lo_u, hi_u), patch.voxel_mm)
        tv, tf = mask_surface(crop_to_bbox(td_full, lo_u, hi_u), patch.voxel_mm)
        blob[f"ccmask_v_{k}"] = (mv + shift).astype(np.float32) if len(mv) else mv
        blob[f"ccmask_f_{k}"] = mf
        blob[f"tdmask_v_{k}"] = (tv + shift).astype(np.float32) if len(tv) else tv
        blob[f"tdmask_f_{k}"] = tf
        blob[f"td_vox_{k}"] = np.int32(td_full.sum())
        # Voxel-count volume of the tracing, as the check that the contoured surface did not
        # drift: the anti-aliased 0.5 surface should land within a few percent of it.
        blob[f"cc_vox_mm3_{k}"] = np.float32(cc_full.sum() * np.prod(patch.voxel_mm))

        # Paint stability: how far the per-frame painted CC size swings around the peak frame's.
        # A large swing means the tracings disagree with each other, so the single-frame mask
        # (and everything derived from it) is only as good as that one frame.
        ok = [s for s in sizes if s]
        swing = max(ok) / min(ok) if ok else float("nan")
        blob[f"sizes_{k}"] = np.array([-1 if s is None else s for s in sizes], np.int32)

        rows.append((f"{spec.session.replace('_data','')}/{spec.subpath}", spec.condition,
                     float(spec.probe_flow), float(V), float(spread), len(patch.vols), float(swing)))
        print(f"  [{k}] {rows[-1][0]:<20} painted {painted}  peak f{patch.peak_i} "
              f"(CC {int(cc_full.sum())} vox, TD {int(td_full.sum())} vox)  "
              f"peak_enh {surf.peak_enh:6.0f} HU  V {V:6.1f} +/-{spread:4.1f} mm^3  "
              f"paint-swing {swing:4.1f}x  refined {ref.kept_frac * 100:5.1f}% of trace "
              f"(local level {np.nanmin(ref.levels[np.isfinite(ref.levels)]):.0f}-"
              f"{np.nanmax(ref.levels[np.isfinite(ref.levels)]):.0f} HU)  "
              f"patch {patch.mask.shape}", flush=True)

    np.savez_compressed(
        OUT, n=len(rows),
        labels=np.array([r[0] for r in rows]),
        conditions=np.array([r[1] for r in rows]),
        probe_flows=np.array([r[2] for r in rows]),      # reference standard; there is no ct_flow here
        volumes=np.array([r[3] for r in rows]),
        vol_spreads=np.array([r[4] for r in rows]),
        n_frames=np.array([r[5] for r in rows]),
        paint_swings=np.array([r[6] for r in rows]),
        **blob)
    print(f"saved {OUT}: {len(rows)} painted CC models")


if __name__ == "__main__":
    main()
