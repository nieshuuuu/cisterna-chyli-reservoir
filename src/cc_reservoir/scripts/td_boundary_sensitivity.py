"""How much does the CC integrated-HU volume depend on the (definitional) CC-TD boundary?

The CC and TD are anatomically contiguous (no valve), so where the CC ends and the TD
begins is a choice, and the manual TD mask has no independent ground truth. This sweeps
the three knobs that encode that choice — the superior cut toward the TD, the TD-exclusion
margin, and the CC dilation — and reports the resulting volume spread. If V is robust to
them, the exact boundary does not drive the result.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.td_boundary_sensitivity
"""
import os

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch
from cc_reservoir.measure.roi import dilated_mask_roi, _ellipsoid_struct

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")

spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
patch = load_cc_patch(spec)
mask, td, vox = patch.mask, patch.td, patch.voxel_mm
vvol = vox[0] * vox[1] * vox[2]
v = patch.vols[patch.peak_i]

cc_z = np.argwhere(mask)[:, 2]
zmin, zmax = cc_z.min(), cc_z.max()
td_side_high = (np.argwhere(td)[:, 2].mean() > cc_z.mean()) if td.sum() else True
print(f"CC z-range [{zmin},{zmax}] ({(zmax-zmin)*vox[2]:.1f} mm tall); "
      f"TD on {'high' if td_side_high else 'low'}-z side; TD voxels in patch = {int(td.sum())}")


def inthu_volume(mask_use, td_dil_mm, cc_dil_mm):
    obj = dilated_mask_roi(mask_use, cc_dil_mm, vox)
    if td.sum() and td_dil_mm > 0:
        obj = obj & ~dilated_mask_roi(td, td_dil_mm, vox)
    A = int(obj.sum())
    ring = binary_dilation(obj, _ellipsoid_struct(4.0, vox)) & ~binary_dilation(obj, _ellipsoid_struct(1.0, vox))
    if td.sum():
        ring = ring & ~dilated_mask_roi(td, 1.5, vox)
    core = binary_erosion(mask_use, iterations=1)
    if core.sum() < 5:
        return float("nan")
    s_bg = float(v[ring].mean())
    s_o = float(np.mean(np.sort(v[core])[-5:]))
    return (float(v[obj].sum()) - A * s_bg) / (s_o - s_bg) * vvol if (s_o - s_bg) > 20 else float("nan")


def trim(cut_mm):
    """Remove CC voxels within cut_mm of the TD-side end (move the CC-TD junction inward)."""
    m = mask.copy()
    n = int(round(cut_mm / vox[2]))
    if td_side_high:
        m[:, :, zmax - n + 1:] = False
    else:
        m[:, :, :zmin + n] = False
    return m


base = inthu_volume(mask, 1.5, 3.0)
print("\n(a) superior cut toward TD (CC dil 3 mm, TD excl 1.5 mm):")
for cut in [0, 2, 4, 6, 8]:
    V = inthu_volume(trim(cut), 1.5, 3.0)
    print(f"   trim {cut:>2} mm off TD end:  V = {V:6.1f} mm^3   ({100*(V-base)/base:+5.1f}% vs no-trim)")

print("\n(b) TD-exclusion margin (no trim, CC dil 3 mm):")
for tdm in [0.0, 1.5, 3.0]:
    tag = "no TD exclusion" if tdm == 0 else f"exclude TD+{tdm} mm"
    print(f"   {tag:<18}:  V = {inthu_volume(mask, tdm, 3.0):6.1f} mm^3")

print("\n(c) CC dilation margin (no trim, TD excl 1.5 mm):")
for ccd in [2.0, 3.0, 4.0]:
    print(f"   CC dil {ccd} mm:  V = {inthu_volume(mask, 1.5, ccd):6.1f} mm^3")

vals = [inthu_volume(trim(c), t, d) for c in [0, 2, 4] for t in [0.0, 1.5, 3.0] for d in [2.0, 3.0, 4.0]]
vals = [x for x in vals if np.isfinite(x)]
print(f"\nFULL SPREAD across all boundary choices: {min(vals):.0f}-{max(vals):.0f} mm^3 "
      f"(median {np.median(vals):.0f}, CV {100*np.std(vals)/np.mean(vals):.0f}%)")
