"""CC segmentation on the CT (MPR overlay), run where the data lives (Linux box).

Shows the integrated-HU half-max CC boundary (green) and the TD (blue dashed) on
axial + sagittal CT slices, with the TD drawn separately to make explicit that it is
NOT merged into the CC. The 3D shaded views live in render_cc_pyvista.py; this is the
2D-on-anatomy companion.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.render_cc_mpr
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch, enhancement_field

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
OUT = os.environ.get("CC_MPR_OUT", "/tmp/cc_segmentation_mpr.png")

spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
patch = load_cc_patch(spec)
mask, td, vol = patch.mask, patch.td, patch.vols[patch.peak_i]
enh_list, _ = enhancement_field(patch)
enh = enh_list[patch.peak_i]
peak_enh = float(enh.max())

cz = int(round(np.argwhere(mask)[:, 2].mean()))
cx = int(round(np.argwhere(mask)[:, 0].mean()))


def mpr(ax, ct, e, tdm, title):
    ax.imshow(ct.T, cmap="gray", vmin=-150, vmax=300, origin="lower")
    if e.max() > 0.3 * peak_enh:
        ax.contour(e.T, levels=[0.5 * peak_enh], colors=["#1a8a2a"], linewidths=2.0)   # CC = green
    if tdm.sum() > 0:
        ax.contour(tdm.T.astype(float), levels=[0.5], colors=["#1f6fe0"],
                   linewidths=1.1, linestyles="dashed")                                 # TD = blue dashed
    ax.set_title(title, fontsize=10); ax.axis("off")


fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), dpi=130)
mpr(axs[0], vol[:, :, cz], enh[:, :, cz], td[:, :, cz], f"Axial z={cz}   green = CC,  blue dash = TD (excluded)")
mpr(axs[1], vol[cx, :, :], enh[cx, :, :], td[cx, :, :], f"Sagittal x={cx}")
fig.suptitle("Cisterna chyli accurate segmentation on CT — TD kept separate (not merged into CC)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT, dpi=130, facecolor="white", bbox_inches="tight")
print(f"saved {OUT} | peak_enh {peak_enh:.0f} HU | TD voxels in patch {int(td.sum())}")
