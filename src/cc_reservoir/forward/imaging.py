import numpy as np
from .psf import apply_psf
from .geometry import fine_grid, fine_vox, render_ellipsoid

def block_mean(a, f):
    """Average non-overlapping f x f x f blocks (the detector averaging step)."""
    nx, ny, nz = a.shape
    a = a[: nx // f * f, : ny // f * f, : nz // f * f]
    return a.reshape(nx // f, f, ny // f, f, nz // f, f).mean(axis=(1, 3, 5))

def forward_patch(s, conc, base_semi_axes, grid, noise_hu=0.0, rng=None):
    """THE forward model: (size-scale s, concentration conc, base shape) -> voxel patch (HU).

    Renders an ellipsoid at fine resolution, blurs by the PSF, block-mean
    downsamples to the coarse voxel grid, and optionally adds Gaussian noise.
    """
    xs, ys, zs = fine_grid(grid)
    semi = np.asarray(base_semi_axes, float) * float(s)
    field = render_ellipsoid((0.0, 0.0, 0.0), semi, conc, xs, ys, zs)
    fvx, fvy, fvz = fine_vox(grid)
    sig_mm = grid.psf_fwhm_mm / 2.355  # FWHM -> sigma for a Gaussian: 2*sqrt(2*ln2)
    sigma_vox = (sig_mm / fvx, sig_mm / fvy, sig_mm / fvz)
    coarse = block_mean(apply_psf(field, sigma_vox), grid.f)
    if noise_hu and rng is not None:
        coarse = coarse + rng.normal(0.0, noise_hu, coarse.shape)
    return coarse
