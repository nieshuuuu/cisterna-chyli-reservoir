from collections import namedtuple
import numpy as np

Grid = namedtuple("Grid", "n_xy n_z vox f psf_fwhm_mm")
DEFAULT_GRID = Grid(n_xy=20, n_z=44, vox=(0.78, 0.78, 1.0), f=2, psf_fwhm_mm=1.0)

def fine_vox(grid):
    return tuple(v / grid.f for v in grid.vox)

def coarse_centers(n, v):
    """n voxel-centre coordinates (mm), symmetric about 0."""
    return (np.arange(n) - (n - 1) / 2.0) * v

def fine_centers(n, v, f):
    """n*f fine centres tiling the same extent, block-aligned to coarse_centers."""
    cc = coarse_centers(n, v)
    sub = (np.arange(f) - (f - 1) / 2.0) * (v / f)
    return (cc[:, None] + sub[None, :]).ravel()

def fine_grid(grid):
    vx, vy, vz = grid.vox
    xs = fine_centers(grid.n_xy, vx, grid.f)
    ys = fine_centers(grid.n_xy, vy, grid.f)
    zs = fine_centers(grid.n_z, vz, grid.f)
    return xs, ys, zs

def render_ellipsoid(center, semi, conc, xs, ys, zs):
    """Concentration field: `conc` inside the ellipsoid, 0 outside."""
    cx, cy, cz = center
    a, b, c = semi
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    q = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 + ((Z - cz) / c) ** 2
    return np.where(q <= 1.0, float(conc), 0.0)

def ellipsoid_volume(semi):
    a, b, c = semi
    return 4.0 / 3.0 * np.pi * a * b * c
