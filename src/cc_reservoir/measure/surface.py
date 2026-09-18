"""CC half-max iso-surface for VISUALIZATION (single producer).

Builds the sub-voxel CC surface from the background-subtracted enhancement field: upsample,
then marching-cubes at half the peak enhancement (the FWHM boundary). This is the surface
used in the 3D renders; it is NOT the volume measurement — that is measure.volume
(integrated-HU). Keeping the surface here means the single-acq mesh builder and the
multi-acq montage derive the same geometry instead of duplicating marching-cubes.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

from cc_reservoir.measure.patch import enhancement_field


def mesh_from_mask(mask, vox, pad=3, up=2, sigma_mm=0.6):
    """Smooth surface mesh (verts in mm, in `mask`'s own frame) from a boolean mask: crop to bbox so
    marching-cubes runs small, upsample+blur to round the voxel staircase, offset back to mm.

    Every 3D duct render derives its geometry here, so the auto-seg build and the hand-seg bake
    produce the same surface for the same mask and stay comparable frame to frame.

    `sigma_mm` is a PHYSICAL length, not a voxel count, and it is CAPPED BY THE THINNEST STRUCTURE
    IN THE SCENE -- blurring a mask shrinks it, so past a point the iso-surface stops representing
    the segmentation and starts deleting it. The TD is ~3.5 mm across (thin spots 1.8 mm), so on
    Acq6 frame 3 the mesh loses this fraction of the mask's z-slices:

        sigma 0.4 mm -> TD 0.0%,  CC 0.0%
        sigma 0.6 mm -> TD 0.1%,  CC 0.0%
        sigma 1.2 mm -> TD 18.7%, CC 7.6%     <- a continuous duct renders as a dashed line

    Hence the 0.6 default. If a render looks blocky, smooth the SURFACE (Taubin, volume-preserving)
    rather than raising this: mask blur buys smoothness by eating the duct.
    """
    idx = np.argwhere(mask)
    if not len(idx):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    sp = tuple(v / up for v in vox)
    sigma = tuple(sigma_mm / s for s in sp)          # mm -> upsampled voxels, per axis
    v, f, _, _ = marching_cubes(gaussian_filter(zoom(crop, up, order=1), sigma), 0.5, spacing=sp)
    return (v.astype(np.float32) + lo * np.asarray(vox, np.float32)).astype(np.float32), f.astype(np.int32)


@dataclass
class CCSurface:
    verts: np.ndarray      # (n_v, 3) in mm
    faces: np.ndarray      # (n_f, 3) vertex indices
    fill_vals: np.ndarray  # (n_t, n_f) per-face enhancement at each timepoint
    peak_i: int
    peak_enh: float
    spacing: tuple         # upsampled voxel size (mm)


def cc_half_max_surface(patch, upsample=2):
    vx, vy, vz = patch.voxel_mm
    sp = (vx / upsample, vy / upsample, vz / upsample)
    enh, _ = enhancement_field(patch)
    enh_up = [zoom(e, upsample, order=1) for e in enh]
    peak_i = patch.peak_i
    peak_enh = float(enh_up[peak_i].max())

    verts, faces, _, _ = marching_cubes(enh_up[peak_i], level=0.5 * peak_enh, spacing=sp)

    def facevals(field):
        idx = np.clip(np.round(verts / np.array(sp)).astype(int), 0, np.array(field.shape) - 1)
        return field[idx[:, 0], idx[:, 1], idx[:, 2]][faces].mean(1)

    fill_vals = np.stack([facevals(eu) for eu in enh_up])
    return CCSurface(verts=verts, faces=faces, fill_vals=fill_vals,
                     peak_i=peak_i, peak_enh=peak_enh, spacing=sp)
