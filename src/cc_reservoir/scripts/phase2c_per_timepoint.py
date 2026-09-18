"""Phase-2c: per-timepoint CC+TD lumen by the INTEGRATED-HU (densitometry) continuous boundary.

For the opacified frames (pre-contrast frames are dropped — the lymphatic lumen is invisible on
CT without contrast), the duct is delineated by the FWHM of the LOCAL opacified peak along the
reused anatomical extent / centerline: occupancy occ = enh / S_O(z), boundary = occ >= 0.5 —
continuous, sub-voxel, concentration-adaptive. CC/TD split is caudal-anchored along the duct.

Outputs (to CC_PHASE2C_OUT, per acquisition):
  phase2c_<tag>.png    the CENTERLINE figure — per-timepoint segmentation MIP, CC green / TD teal
                       + the reused centerline (red) + CC/TD boundary (orange)
  phase2c_<tag>.csv    integrated-HU V_CC/V_TD + mean E per opacified frame
  tdc_<tag>.csv        time-density curves E_CC(t)/E_TD(t) over ALL frames (transport-model input)
  fixedmesh_<tag>.npz  master structure + per-frame swept-tube surfaces for the 3D render

  CC_ARCHIVE=<path> CC_SESSION=<s> CC_SUB=<a> CC_PHASE2C_OUT=<dir> \
    PYTHONPATH=src python3 -m cc_reservoir.scripts.phase2c_per_timepoint
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter, uniform_filter1d
from skimage.measure import marching_cubes

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_td_patch
from cc_reservoir.measure.lumen import (lumen_roi, cc_td_boundary_z, fixed_lumen, split_caudal,
                                        frame_enhancement, static_baseline, local_opacified_ref,
                                        fwhm_lumen, centerline, swept_tube, equiv_radius)
from cc_reservoir.measure.volume import integrated_hu_volume_region

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
OUT = os.environ.get("CC_PHASE2C_OUT", "/tmp/cc_phase2c"); os.makedirs(OUT, exist_ok=True)
tag = f"{SESSION}_{os.path.basename(SUB)}"
OPAC_MIN_E = 80.0                                             # a frame is usable only if BOTH CC and TD are opacified
                                                             # (mean enh ≥ this in each) — else one region is empty
                                                             # and its integrated-HU volume is nan/negative

spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]
patch = load_cc_td_patch(spec)
vox = patch.voxel_mm; vz = vox[2]
rois = lumen_roi(patch)
base = static_baseline(patch)
enh = [frame_enhancement(patch, i, rois, base) for i in range(len(patch.vols))]
master = fixed_lumen(patch, rois, baseline=base)              # anatomical extent (reused every frame)
zb = cc_td_boundary_z(patch)
cx, cy = centerline(master)                                   # reused centerline (per-z centroid, smoothed)
zc0 = np.where(master.any(axis=(0, 1)))[0]; z0, z1 = int(zc0.min()), int(zc0.max())
zrange = range(z0, z1 + 1)
mcc, mtd = split_caudal(master, zb)                           # fixed CC/TD regions for time-density curves
ecc_m = [float(e[mcc].mean()) for e in enh]
etd_m = [float(e[mtd].mean()) for e in enh]
s_o_all = [local_opacified_ref(e, master) for e in enh]
opac = [i for i in range(len(enh)) if min(ecc_m[i], etd_m[i]) >= OPAC_MIN_E]
print(f"{tag}: opacified frames {opac} (boundary z={zb})")
if not opac:
    raise SystemExit(f"{tag}: no frame with both CC & TD opacified (E ≥ {OPAC_MIN_E}) — skipping")

# per-frame integrated-HU metrics on the FWHM lumen
mets = []
for i in opac:
    seg = fwhm_lumen(enh[i], master, s_o_all[i])
    cc, td = split_caudal(seg, zb)
    mets.append(dict(frame=i, V_cc=integrated_hu_volume_region(cc, enh[i], vox)[0],
                     V_td=integrated_hu_volume_region(td, enh[i], vox)[0],
                     E_cc=float(enh[i][cc].mean()) if cc.any() else 0.0,
                     E_td=float(enh[i][td].mean()) if td.any() else 0.0, seg=seg))

with open(os.path.join(OUT, f"phase2c_{tag}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["frame", "V_cc", "V_td", "E_cc", "E_td"]); w.writeheader()
    for m in mets:
        w.writerow({k: (round(m[k], 1) if isinstance(m[k], float) and np.isfinite(m[k]) else m[k])
                    for k in ["frame", "V_cc", "V_td", "E_cc", "E_td"]})

# time-density curves over ALL frames (concentration in the fixed CC/TD region — measurable even
# pre-segmentation; the transport-model input). Stored here, not solved.
with open(os.path.join(OUT, f"tdc_{tag}.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "E_cc", "E_td"])
    for i in range(len(enh)):
        w.writerow([i, round(ecc_m[i], 1), round(etd_m[i], 1)])

print(f"{'tp':>2} | {'V_CC':>6} {'V_TD':>6} | {'E_CC':>5} {'E_TD':>5}")
for m in mets:
    print(f"{m['frame']:>2} | {m['V_cc']:6.0f} {m['V_td']:6.0f} | {m['E_cc']:5.0f} {m['E_td']:5.0f}")

# CENTERLINE figure: per-opacified-frame MIP of the enhancement (grayscale) with the
# segmentation as CONTOURS (CC green / TD teal) + the reused centerline (red) + CC/TD boundary
# (orange). Contours-on-grayscale (not filled) so the underlying contrast data stays visible.
vmax = max(250.0, float(np.percentile(np.concatenate([enh[i][master] for i in opac]), 95)))
fig, axs = plt.subplots(1, len(opac), figsize=(3.0 * len(opac), 8.2), dpi=110, squeeze=False)
for ax, m in zip(axs[0], mets):
    i = m["frame"]
    cc = m["seg"].copy(); cc[:, :, zb + 1:] = False
    td = m["seg"].copy(); td[:, :, :zb + 1] = False
    ax.imshow(enh[i].max(axis=0).T, cmap="gray", origin="lower", aspect="auto", vmin=0, vmax=vmax)
    ax.contour(cc.max(axis=0).T, [0.5], colors=["#39d353"], linewidths=1.2, alpha=0.6)
    ax.contour(td.max(axis=0).T, [0.5], colors=["#56c2d6"], linewidths=1.2, alpha=0.6)
    ax.plot(cy[z0:z1 + 1], np.arange(z0, z1 + 1), "-", color="red", lw=0.8, alpha=0.7)
    ax.axhline(zb, color="orange", ls=":", lw=1)
    ax.set_title(f"tp{i}\nV_CC {m['V_cc']:.0f}  V_TD {m['V_td']:.0f} mm³")
    ax.set_xlabel("y"); ax.set_ylabel("z")
fig.suptitle(f"{tag} — per-timepoint segmentation (integrated-HU) · green=CC, teal=TD, red=reused centerline, orange=boundary", y=1.01)
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"phase2c_{tag}.png"), dpi=110, facecolor="white", bbox_inches="tight")

# meshes: smooth swept TUBES — a disk per z on the reused centerline, radius = integrated-HU
# caliber (sqrt(area/pi)) smoothed along z → gap-free silky surface (the raw patchy mask
# fragments). Master tube = anatomical structure; each frame's tube deforms (respiration).
f = 2; sp = tuple(c / f for c in vox); bz_mm = zb * vz


def tube_mesh(seg):
    tube = swept_tube(cx, cy, equiv_radius(seg, zrange), zrange, master.shape)
    field = gaussian_filter(zoom(tube.astype(float), f, order=1), sigma=f * 0.8)
    v, fc, _, _ = marching_cubes(field, level=0.5, spacing=sp)
    return v.astype(np.float32), fc.astype(np.int32)


def face_is_cc(verts, faces):
    return verts[faces].mean(axis=1)[:, 2] <= bz_mm           # caudal (low z) = CC


def face_enh(verts, faces, enh_field, region):
    """Per-face contrast (HU) = per-z MEAN enhancement in the lumen at that face's level (axial
    profile) → shows the flow (caudal CC fills first, then ascends), unlike a hot-spot sample."""
    nz = region.shape[2]; ez = np.zeros(nz)
    for z in range(nz):
        mz = region[:, :, z]
        if mz.any():
            ez[z] = float(enh_field[:, :, z][mz].mean())
    ez = uniform_filter1d(ez, 5)
    zi = np.clip(np.round(verts[faces].mean(1)[:, 2] / vz).astype(int), 0, nz - 1)
    return ez[zi].astype(np.float32)


mvn, mfn = tube_mesh(master)
arrs = {"vn": mvn, "fn": mfn, "is_cc": face_is_cc(mvn, mfn)}
for m in mets:
    i = m["frame"]
    wv, wf = tube_mesh(m["seg"])
    arrs[f"wv_{i}"], arrs[f"wf_{i}"] = wv, wf
    arrs[f"wcc_{i}"], arrs[f"wenh_{i}"] = face_is_cc(wv, wf), face_enh(wv, wf, enh[i], m["seg"])

np.savez(os.path.join(OUT, f"fixedmesh_{tag}.npz"),
         opac=np.array(opac, int), Vcc=np.array([m["V_cc"] for m in mets]),
         Vtd=np.array([m["V_td"] for m in mets]), **arrs)
print(f"saved {OUT}/phase2c_{tag}.{{png,csv}} + tdc_{tag}.csv + fixedmesh_{tag}.npz")
