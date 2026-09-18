"""Stage 1 of CC 3D rendering (run where the data lives, e.g. the Linux box): build the
surface meshes + per-timepoint fill values for ONE acquisition and save to an .npz. No
OpenGL needed here, so it runs headless over the SMB mount; stage 2 (render_cc_pyvista.py)
renders on a machine with a GL context.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.compute_cc_meshes
"""
import os

import numpy as np
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch
from cc_reservoir.measure.surface import cc_half_max_surface

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
OUT = os.environ.get("CC_MESHES", "/tmp/cc_meshes.npz")
UPSAMPLE = 2

spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
patch = load_cc_patch(spec)
surf = cc_half_max_surface(patch, upsample=UPSAMPLE)

# old tp1 binary surface, for the prev-vs-new comparison panel
vo, fo, _, _ = marching_cubes(zoom(patch.mask.astype(float), UPSAMPLE, order=0),
                              level=0.5, spacing=surf.spacing)

np.savez(OUT, vn=surf.verts, fn=surf.faces, vo=vo, fo=fo, peak_enh=surf.peak_enh,
         fill_vals=surf.fill_vals, times=np.arange(len(patch.vols)), peak_i=surf.peak_i)
print(f"saved {OUT}: new {len(surf.faces)} faces, old {len(fo)} faces, "
      f"peak_enh {surf.peak_enh:.0f}, n_t {len(patch.vols)}")
