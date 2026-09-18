import numpy as np
from cc_reservoir.forward.geometry import (
    Grid, DEFAULT_GRID, fine_vox, coarse_centers, fine_centers, fine_grid,
    render_ellipsoid, ellipsoid_volume,
)

def test_fine_coarse_alignment():
    n, v, f = 5, 0.78, 2
    cc = coarse_centers(n, v)
    fc = fine_centers(n, v, f)
    assert np.allclose(fc.reshape(n, f).mean(axis=1), cc)

def test_ellipsoid_volume_matches_render():
    semi = (3.0, 3.0, 3.0)            # a sphere, r = 3 mm
    fv = 0.2
    ax = (np.arange(80) - 39.5) * fv  # centred grid, ±8 mm
    field = render_ellipsoid((0, 0, 0), semi, 1.0, ax, ax, ax)
    numeric = (field > 0).sum() * fv ** 3
    assert np.isclose(numeric, ellipsoid_volume(semi), rtol=0.05)

def test_default_grid_shapes():
    xs, ys, zs = fine_grid(DEFAULT_GRID)
    assert len(xs) == DEFAULT_GRID.n_xy * DEFAULT_GRID.f
    assert len(zs) == DEFAULT_GRID.n_z * DEFAULT_GRID.f
    assert np.allclose(fine_vox(DEFAULT_GRID), (0.39, 0.39, 0.5))
