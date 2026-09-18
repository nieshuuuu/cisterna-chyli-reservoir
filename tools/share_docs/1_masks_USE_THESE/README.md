# 1_masks_USE_THESE — the masks to use

CC / TD / lymph-node masks for the lymphangiography CT. **Masks only, no CT.**
Pair each `f<N>_edit.npz` here with the matching `f<N>.npz` in `..\2_CT_volumes_and_painter\data_fullres\`
under the same acquisition folder name — that folder also holds the acquisition's `meta.npz`
(voxel size, and the crop back to the raw DICOM grid).

107 masks · 28 acquisitions · 84 of them hand-painted · 13 MB
Labels: `1 = CC`, `2 = TD`, `3 = lymph node`, `4 = bone`.

```python
import os, numpy as np

ROOT  = r"Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation"
ACQ, FRAME = "context4d_4_19_23_data_Acq01_Acq01", "f1"

CT   = os.path.join(ROOT, "2_CT_volumes_and_painter", "data_fullres")
MASK = os.path.join(ROOT, "1_masks_USE_THESE")

hu   = np.load(os.path.join(CT,   ACQ, FRAME + ".npz"))["hu"]        # int16 (x, y, z)
seg  = np.load(os.path.join(MASK, ACQ, FRAME + "_edit.npz"))["seg"]  # uint8, same shape
meta = np.load(os.path.join(CT,   ACQ, "meta.npz"), allow_pickle=True)

ml = float(np.prod(meta["voxel"])) / 1000.0                          # one voxel in mL
print("CC", (seg == 1).sum() * ml, "mL   TD", (seg == 2).sum() * ml, "mL")
```

## Where this came from

Built 2026-08-26 from the newer painter package plus the older July snapshot
(`..\3_old_masks_superseded`), taking the newer copy except where it had dropped hand work, then
removing connected components under 20 voxels per label. Every file records `merged_from`
(`new` / `old`) and `merge_reason` alongside `seg`.

One frame was recovered from the older archive: `07_20_22_data_Angiotensin_Acq12 f0`
(24,524 voxels, wiped to zero in the newer package). Despeckling removed 2,337 voxels.

## Read this before measuring

- **4 masks are not hand work** — they are the builder's threshold auto-seed saved through
  untouched, and some of it lies on rib cortex. Listed in `quarantine_frames.txt`:
  `8_31_22_Acq6 f1` (whole frame), `Acq6 f2`/`f3` lymph, `09_07_22_Acq1 f0` lymph.
- **13 all-zero masks blank out a non-empty auto seg.** `8_31_22_Acq2` f0–f5 sit on the brightest
  duct in that session and were simply never painted.
- **CC vs TD is not defined the same way everywhere.** In 07_20_22 and 09_07_22 the two labels are
  separate structures ~12 mm apart over 3.6–6.4 cm; in the other sessions they are one tube cut at
  a plane. CC volume and the CC/TD split are not comparable across sessions. CC + TD together is
  fine.
- **07_20_22 masks were upsampled 2×2×1 in-plane** from earlier half-resolution work. Fine for
  volume and extent; not for in-plane diameter or wall work.
- **Boundaries are drawn generously**, 1.3–2.6× the half-maximum isosurface for CC and 1.4–3.9× for
  TD, with a 2.8× spread between sessions. Re-threshold inside the mask if you are comparing
  volumes across sessions.

`sidecar_manifest.csv` classifies every frame. Regenerate it any time with:

```
python -m cc_reservoir.scripts.audit_edit_sidecars <context_root> --csv out.csv
```
