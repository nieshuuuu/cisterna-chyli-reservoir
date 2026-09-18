"""Anatomical landmark masks by HU threshold — spatial context for where the CC/TD sit, not
clinical organ boundaries.

Bone is clean: cortical/trabecular bone is the only thing above ~250 HU, so a threshold isolates
it. Kidney is NOT clean here: this is lymphatic (not intravenous) contrast, so the kidneys never
take up contrast and their parenchyma sits at the same HU as liver/muscle. `kidney_mask` is
therefore a deliberately bounded heuristic — posterior soft-tissue blobs near the duct level — and
a location hint only, never a true kidney boundary."""
import numpy as np
from scipy.ndimage import binary_opening, label, generate_binary_structure

_ST = generate_binary_structure(3, 1)


def _remove_small(mask, min_vox):
    """Drop connected components smaller than min_vox voxels (speckle removal)."""
    lab, n = label(mask, _ST)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    good = counts >= min_vox
    good[0] = False                      # background label 0 is never kept
    return good[lab]


def bone_mask(vol, hu=250, min_vox=64):
    """Cortical/trabecular bone: HU >= ~250, morphologically opened to drop speckle, then small
    islands removed. Clean because nothing soft-tissue reaches this HU."""
    return _remove_small(binary_opening(vol >= hu, _ST, 1), min_vox)


def kidney_mask(vol, z_lo=None, z_hi=None, hu=(0, 130), min_vox=1500, keep=2):
    """Rough kidney landmark: the largest posterior soft-tissue blobs within the duct z-band.

    Approximate by construction — lymphatic contrast never enters the kidney, so its HU overlaps
    liver and muscle and cannot be thresholded cleanly. We constrain hard (posterior half only,
    HU window, duct z-band, keep the `keep` largest opened components) so it stays a bounded
    location hint rather than grabbing the whole abdomen."""
    nx, ny, nz = vol.shape
    m = (vol >= hu[0]) & (vol <= hu[1])
    z0 = 0 if z_lo is None else max(int(z_lo), 0)
    z1 = nz if z_hi is None else min(int(z_hi) + 1, nz)
    m[:, :, :z0] = False
    m[:, :, z1:] = False
    m[: nx // 2] = False                 # posterior half only (anterior = low x)
    m = binary_opening(m, _ST, 2)
    lab, n = label(m, _ST)
    if n == 0:
        return m
    counts = np.bincount(lab.ravel()); counts[0] = 0
    order = [i for i in np.argsort(counts)[::-1] if counts[i] >= min_vox][:keep]
    out = np.zeros_like(m)
    for i in order:
        out |= lab == i
    return out
