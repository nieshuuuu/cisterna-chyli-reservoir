"""Build CC surfaces for every CLEAN acquisition (one montage .npz), run where the data
lives. Stage 2 render_cc_montage.py renders the grid on a GL machine. Clean = the 1b-A QA
flag (recomputed from source). Each model also carries its robust integrated-HU volume so
the montage can be labeled and ordered by volume.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.compute_cc_meshes_clean
"""
import os

import numpy as np

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch
from cc_reservoir.measure.surface import cc_half_max_surface
from cc_reservoir.measure.volume import robust_volume
from cc_reservoir.measure.extract import extract_curves, method_agreement, qa_pass

OUT = os.environ.get("CC_MONTAGE_MESHES", "/tmp/cc_montage.npz")

blob = {}
labels, conditions, volumes, sessions = [], [], [], []
k = 0
for spec in WORKING_SET:
    patch = load_cc_patch(spec)
    curves = extract_curves(list(enumerate(patch.vols)), patch.mask, patch.voxel_mm)
    if not qa_pass(method_agreement(curves)):
        continue
    V, _, _ = robust_volume(patch)
    if not np.isfinite(V):
        continue
    surf = cc_half_max_surface(patch)
    blob[f"verts_{k}"] = surf.verts.astype(np.float32)
    blob[f"faces_{k}"] = surf.faces.astype(np.int32)
    labels.append(f"{spec.session.replace('_data','')}/{os.path.basename(spec.subpath)}")
    sessions.append(spec.session)
    conditions.append(spec.condition)
    volumes.append(round(V, 1))
    print(f"  model {k}: {labels[-1]:22} {spec.condition:11} V={V:6.0f} mm^3  ({len(surf.faces)} faces)")
    k += 1

np.savez(OUT, n=k, labels=np.array(labels), sessions=np.array(sessions),
         conditions=np.array(conditions), volumes=np.array(volumes), **blob)
print(f"saved {OUT}: {k} clean CC models")
