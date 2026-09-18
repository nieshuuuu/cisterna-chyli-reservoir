import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, zoom
from cc_reservoir.forward.geometry import Grid, fine_grid, fine_vox, coarse_centers, render_ellipsoid, ellipsoid_volume
from cc_reservoir.forward.imaging import block_mean
from cc_reservoir.forward.psf import apply_psf
from cc_reservoir.measure.roi import dilated_mask_roi, _ellipsoid_struct

GRID = Grid(n_xy=44, n_z=64, vox=(0.78, 0.78, 1.0), f=2, psf_fwhm_mm=1.0)
vx, vy, vz = GRID.vox; vvol = vx * vy * vz
S_BG_true, NOISE = 45.0, 15.0

def make_patch(semi, conc, rng):
    xs, ys, zs = fine_grid(GRID)
    fvx, fvy, fvz = fine_vox(GRID); sig = GRID.psf_fwhm_mm / 2.355
    blurred = apply_psf(render_ellipsoid((0, 0, 0), semi, conc, xs, ys, zs), (sig / fvx, sig / fvy, sig / fvz))
    hu = S_BG_true + block_mean(blurred, GRID.f) + rng.normal(0, NOISE, (GRID.n_xy, GRID.n_xy, GRID.n_z))
    cx = coarse_centers(GRID.n_xy, vx); cy = coarse_centers(GRID.n_xy, vy); cz = coarse_centers(GRID.n_z, vz)
    true_bin = render_ellipsoid((0, 0, 0), semi, 1.0, cx, cy, cz) > 0.5
    return hu, true_bin

def inthu_vol(hu, true_bin):
    obj = dilated_mask_roi(true_bin, 3.0, GRID.vox)
    A = int(obj.sum()); I = float(hu[obj].sum())
    ring = binary_dilation(obj, _ellipsoid_struct(4.0, GRID.vox)) & ~binary_dilation(obj, _ellipsoid_struct(1.0, GRID.vox))
    S_BG = float(hu[ring].mean())
    core = binary_erosion(true_bin, iterations=1)
    S_O = float(np.mean(np.sort(hu[core])[-5:])) if core.sum() >= 5 else float(np.sort(hu[true_bin])[-3:].mean())
    return (I - A * S_BG) / (S_O - S_BG) * vvol

def halfmax_vol(hu, true_bin):
    gate = dilated_mask_roi(true_bin, 6.0, GRID.vox)
    bg = np.median(hu[binary_dilation(gate, _ellipsoid_struct(2.0, GRID.vox)) & ~gate])
    e = np.where(gate, hu - bg, 0.0); eu = zoom(e, 2, order=1)
    return float((eu >= 0.5 * eu.max()).sum() * (vvol / 8))

cases = [("thin (real CC-like)", (1.25, 1.25, 9.5)), ("wider CC", (2.0, 2.0, 9.5)),
         ("round/large", (3.5, 3.5, 5.0))]
print(f"{'case':22} {'V_true':>7} | {'int-HU':>16} | {'half-max':>16}   (5 noise reps, conc=600 HU, sigma=15)")
for name, semi in cases:
    Vt = ellipsoid_volume(semi)
    vi, vh = [], []
    for s in range(5):
        rng = np.random.default_rng(s)
        hu, tb = make_patch(semi, 600.0, rng)
        vi.append(inthu_vol(hu, tb)); vh.append(halfmax_vol(hu, tb))
    vi, vh = np.array(vi), np.array(vh)
    print(f"{name:22} {Vt:7.0f} | {vi.mean():6.0f} ({100*(vi.mean()-Vt)/Vt:+4.0f}% +-{100*vi.std()/Vt:2.0f}%) | "
          f"{vh.mean():6.0f} ({100*(vh.mean()-Vt)/Vt:+4.0f}% +-{100*vh.std()/Vt:2.0f}%)")
