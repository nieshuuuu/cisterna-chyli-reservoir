import numpy as np
from scipy.ndimage import gaussian_filter

def apply_psf(field, sigma_vox):
    """Blur a concentration field by a Gaussian PSF.

    sigma_vox: scalar or per-axis sequence, in (fine) voxel units.
    Background is 0 (post-baseline), so mode='constant' is physically correct;
    interior mass is conserved when signal does not reach the array border.
    """
    return gaussian_filter(np.asarray(field, float), sigma=sigma_vox,
                           mode="constant", cval=0.0)
