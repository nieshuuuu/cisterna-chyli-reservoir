import numpy as np
from cc_reservoir.forward.geometry import Grid, fine_vox, ellipsoid_volume
from cc_reservoir.forward.imaging import block_mean, forward_patch

def test_block_mean_shape_and_mean():
    a = np.ones((4, 4, 4))
    out = block_mean(a, 2)
    assert out.shape == (2, 2, 2) and np.allclose(out, 1.0)

def test_forward_patch_conserves_mass():
    # mass = sum(patch) * coarse_voxel_volume  ~=  V_ellipsoid * conc
    grid = Grid(n_xy=20, n_z=44, vox=(0.78, 0.78, 1.0), f=2, psf_fwhm_mm=1.0)
    semi = (1.25, 1.25, 9.5); conc = 800.0
    patch = forward_patch(s=1.0, conc=conc, base_semi_axes=semi, grid=grid)
    cvx, cvy, cvz = grid.vox
    mass = patch.sum() * cvx * cvy * cvz
    expected = ellipsoid_volume(semi) * conc
    assert np.isclose(mass, expected, rtol=0.08)

def test_forward_patch_noise_changes_output():
    grid = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)
    semi = (1.25, 1.25, 9.5)
    rng = np.random.default_rng(0)
    clean = forward_patch(1.0, 800.0, semi, grid)
    noisy = forward_patch(1.0, 800.0, semi, grid, noise_hu=15.0, rng=rng)
    assert not np.allclose(clean, noisy)
    assert 10 < np.std(noisy - clean) < 20   # ~15 HU
