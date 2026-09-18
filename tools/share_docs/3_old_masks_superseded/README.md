# Egor's full-resolution segmentation redo — July 2026

Manual cisterna chyli (CC) / thoracic duct (TD) / lymph segmentations of the pig
dynamic-CT lymphangiography archive, re-painted by Egor at **full CT resolution**
(0.781 × 0.781 × 0.5 mm voxels — no downsampling) with the lab's iPad painter, saved
20–22 July 2026. This folder is the shared archive of those masks so anyone can
**re-analyze the studies with the new segmentations**. Copied from the lab Windows PC
2026-08-07 by Shu Nie (nies1@hs.uci.edu).

The painter itself, plus the matching full-resolution HU volumes these masks sit on, are
next door in **`..\2_CT_volumes_and_painter\`** (its README is the full how-to).

---

## Layout

One folder per acquisition, named exactly like the painter's context volumes:

```
context4d_<session>_<condition>_Acq<N>\
    f<N>_edit.npz    one manual mask per time-frame N
    meta.npz         geometry + raw-DICOM provenance for the acquisition
```

Masks only — the HU backdrops are not duplicated here. The matching `f<N>.npz` HU volume
for every mask is in `..\2_CT_volumes_and_painter\data_fullres\<same folder name>\`.

## Inventory & provenance — 98 masks, 28 acquisitions

| Session | Acquisitions with masks | Files | Saved | Provenance |
|---|---|---|---|---|
| `4_19_23` | all 4 sub-acqs (f0+f1 each) | 8 | Jul 20 | painted manually at full res |
| `4_13_23` | all 6 sub-acqs | 10 | Jul 20–21 | painted manually at full res |
| `03_23_23` | 5 sub-acqs of Acq01–Acq03 | 7 | Jul 21 | painted manually at full res |
| `09_07_22` | Acq1, frames f0–f3 (of 6) | 4 | Jul 21 | painted manually at full res |
| `8_31_22` | Acq1–Acq6, all 6 frames each | 36 | Jul 21–22 | painted manually at full res |
| `07_20_22` | Angiotensin Acq8/9/12 + Baseline Acq6/7 (all 6 frames each); Baseline Acq10 f0–f2 (of 7) | 33 | Jul 16 | Egor's earlier **half-res (2× downsampled)** hand paint, converted onto the full-res grid programmatically (HU under the mask verified to match the original) |

The `07_20_22` row is the one caveat: those six acquisitions were hand-painted at half
resolution earlier the same week and up-converted; everything else was painted directly
on the full-resolution volumes.

Session quality, from the lab's usability review:

- **`4_13_23` / `4_19_23`** — 2023 sessions, the highest-signal data in the archive.
- **2022 sessions** (`07_20_22`, `8_31_22`, `09_07_22`) — dynamic-curve studies,
  normal-caliber duct (2–4 mm).
- **`03_23_23`** — two-volume prospective protocol → no dynamic curve, and Acq01's second
  volume was never acquired. Masks exist, but flow analysis is limited.

## File format

`f<N>_edit.npz` (written by the painter, `np.savez_compressed`):

- `seg` — uint8 label volume covering the **whole context volume**, axes `[x, y, z]` =
  anterior→posterior, left→right, caudal→cranial.
  Labels: **1 = CC, 2 = TD, 3 = lymph, 4 = bone** (0 = background).
- `source` — the frame filename the mask belongs to. Caveat: the 33 converted `07_20_22`
  masks instead carry an `egor_ds2:` provenance tag (e.g. `egor_ds2:f0_edit.npz`), so don't
  match `source` against frame filenames programmatically — use the file's own `f<N>` stem.
- No HU data — pair it with the matching `f<N>.npz`.

`meta.npz` (load with `allow_pickle=True`): `voxel` (mm, `[0.781, 0.781, 0.5]`), `n_frames`,
`peak_i` (peak-contrast frame), `duct_centroid`, CC/TD boundary diagnostics, and the raw-DICOM
provenance: `crop_lo`, `crop_hi`, `downsample` (= `[1,1,1]` here), `raw_full_shape`
(`[512, 512, 1401]` for the 2022 whole-body sessions, `[512, 512, 921]` for `4_13_23` /
`4_19_23`, `[512, 512, 1301–1361]` for `03_23_23`), `raw_session`, `raw_sub`.

## Using the masks

**1) In the painter** — nothing to do: `..\2_CT_volumes_and_painter\data_fullres` already carries
these sidecars, and the painter (like all lab loaders) always prefers an `f<N>_edit.npz`
over the automatic seg inside `f<N>.npz`.

**2) Directly with numpy** — label volumes in three lines:

```python
import numpy as np
folder = r"context4d_4_13_23_data_Acq01_Acq01"
seg  = np.load(folder + r"\f1_edit.npz")["seg"]
meta = np.load(folder + r"\meta.npz", allow_pickle=True)
voxel_ml = float(np.prod(meta["voxel"])) / 1000.0        # mm^3 -> mL
print("TD:", float((seg == 2).sum()) * voxel_ml, "mL")
```

**3) On the raw DICOM grid** — the context volume is a crop of the original scan; `meta`
records exactly where. Per axis: `raw_index = crop_lo + context_index` (downsample is 1).
Or let the package do it:

```python
import sys, numpy as np
sys.path.insert(0, r"..\2_CT_volumes_and_painter\software\src")
from cc_reservoir.measure.context import load_context_seg, place_seg_in_raw

folder = r"context4d_4_13_23_data_Acq01_Acq01"
seg  = load_context_seg(folder, frame=1)     # this folder: loads f1_edit.npz
meta = np.load(folder + r"\meta.npz", allow_pickle=True)
raw_labels = place_seg_in_raw(seg, meta)     # full raw grid (this folder: 512x512x921)
# then: raw_hu[raw_labels == 2] -> TD voxels of the original DICOM volume
```

The raw series each mask belongs to is named in `meta` (`raw_session`, `raw_sub`) and lives
under `Z:\ImageData\Lymph_Studies\<session>_data\` (leading zeros in month names are
inconsistent — `8_31_22_data` but `08_15_22_data` — match loosely).

**Precedence rule, worth repeating:** every `cc_reservoir` loader prefers a `*_edit.npz`
sidecar sitting next to a frame file over that frame's built-in `seg`. To re-analyze with
these masks, keep the sidecar beside the `f<N>.npz` you load; to fall back to the automatic
segmentation, remove or rename the sidecar.

Questions: **Shu Nie — nies1@hs.uci.edu**
