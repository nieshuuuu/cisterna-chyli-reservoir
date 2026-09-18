"""CC volume by the integrated-HU method (Molloi 2019) for one acquisition.

Thin driver over measure.volume (the single producer). Prints the per-frame V(t) so the
~flat-once-opacified pattern is visible, plus the robust (median-over-opacified) volume.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.integrated_hu_volume
"""
import os

import numpy as np

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch
from cc_reservoir.measure.volume import volume_rois, volume_series, robust_volume

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
patch = load_cc_patch(spec)
vvol = float(np.prod(patch.voxel_mm))
rois = volume_rois(patch)

V_med, V_spread, used = robust_volume(patch, rois=rois)
print(f"{SESSION}/{SUB}  A={rois.A} vox  core={int(rois.core.sum())} vox")
print(f"robust integrated-HU volume = {V_med:.0f} +- {V_spread:.0f} mm^3 "
      f"(median over opacified frames {used})")
print(f"  vs tp1-binary {patch.mask.sum()*vvol:.0f} mm^3 (voxelized)")
print("V(t) (should be ~flat across opacified frames while S_O swings):")
for i, (V, s_o, s_bg) in enumerate(volume_series(patch, rois)):
    star = " *" if i in used else "  "
    print(f"  timepoint {i}:{star} S_O={s_o:6.1f}  S_BG={s_bg:5.1f}  V={V:7.1f} mm^3")
