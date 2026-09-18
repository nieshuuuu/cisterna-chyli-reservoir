# Cisterna chyli & thoracic duct segmentation — start here

Hand-drawn segmentations of the **cisterna chyli (CC)**, the **thoracic duct (TD)** and
**lymph nodes** on contrast lymphangiography CT in pigs, plus the CT volumes they belong to
and the tool used to draw them.

28 acquisitions across 6 imaging sessions · 107 time frames · 84 of them carry a hand-drawn mask.

## The three folders

| | what it is | use it for |
|---|---|---|
| **`1_masks_USE_THESE`** | the segmentations, 13 MB | **this is the one you want** |
| **`2_CT_volumes_and_painter`** | the CT volumes (16 GB) + the painting app | reading HU; drawing new masks |
| `3_old_masks_superseded` | July 2026 snapshot of the masks | nothing, unless you are chasing history |

`2_CT_volumes_and_painter\data_fullres` also contains its own `_edit.npz` mask files. **Those are
the pre-merge masks — ignore them** and take masks from `1_masks_USE_THESE`. They are kept there
because the painting app reads and writes them in place.

## Loading a frame

Masks and CT are separate files sharing a stem, in identically-named acquisition folders:
`f3_edit.npz` is the mask, `f3.npz` is the CT.

```python
import os, numpy as np

ROOT = r"Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation"
ACQ, FRAME = "context4d_4_19_23_data_Acq01_Acq01", "f1"

hu   = np.load(os.path.join(ROOT, "2_CT_volumes_and_painter", "data_fullres",
                            ACQ, FRAME + ".npz"))["hu"]                 # int16 (x, y, z)
seg  = np.load(os.path.join(ROOT, "1_masks_USE_THESE",
                            ACQ, FRAME + "_edit.npz"))["seg"]           # uint8, same shape
meta = np.load(os.path.join(ROOT, "2_CT_volumes_and_painter", "data_fullres",
                            ACQ, "meta.npz"), allow_pickle=True)

ml = float(np.prod(meta["voxel"])) / 1000.0                             # one voxel in mL
print("CC", (seg == 1).sum() * ml, "mL   TD", (seg == 2).sum() * ml, "mL")
```

Labels: `1 = CC`, `2 = TD`, `3 = lymph node`, `4 = bone`. `meta.npz` also carries `crop_lo` /
`crop_hi` / `raw_full_shape` if you need to map a mask back onto the original DICOM grid, and
`raw_session` / `raw_sub` naming the series it came from — see `..\DATA_GUIDE_shu_nie.md`.

## Before you measure anything

Not every mask is usable. The full per-acquisition status and the caveats are in
`1_masks_USE_THESE\README.md`; the short version:

- **Four masks are not hand work at all** — they are the tool's automatic threshold seed saved
  through untouched, and some of it sits on rib cortex. Listed in
  `1_masks_USE_THESE\quarantine_frames.txt`.
- **`8_31_22_Acq2` has the brightest duct in its session and no mask at all** — six frames that
  were simply never painted. Queued to be drawn.
- **CC vs TD does not mean the same thing in every session.** In 07_20_22 and 09_07_22 the two
  labels are separate structures ~12 mm apart over several centimetres; elsewhere they are one
  tube cut at a plane. **CC volume and the CC/TD split are not comparable across sessions.**
  CC + TD together is fine everywhere.
- **07_20_22 masks were upsampled from half resolution** in-plane. Fine for volume; not for
  in-plane diameter.

## Drawing new masks

**Double-click `2_CT_volumes_and_painter\Paint.bat`.** That is the whole thing.

It pushes up anything still pending from last time, pulls down the current painter and
everyone else's masks, opens the painter on a fast local copy, and pushes your work back to
the share the moment you press Ctrl-C. Needs Python 3.10+ with `numpy` and `pillow`.

The first run on a machine copies the CT volumes to `C:\Storage\CisternaChyli\painter_work`
(~16 GB, asks first, once). After that only masks move and a sync takes seconds.

You do not have to remember to sync. That used to be three separate steps and it is how
`8_31_22_Acq2` was painted and never reached anybody. If the window is killed rather than
closed cleanly, nothing is lost either — every Save is already on local disk, and the next
run pushes it.

When the same frame changed on both sides since your last sync, **neither copy is touched**
and the frame is listed as a conflict for you to settle by hand. Nothing is ever deleted on
either side, and the CT volumes are never uploaded.

On Mac or Linux, `start_cc_painter.sh` opens the painter but does **not** sync — it prints the
`rsync` command to run afterwards.

Masks arrive in `2_CT_volumes_and_painter\data_fullres`, the painter's working area. They reach
`1_masks_USE_THESE` only when the merge is re-run:

```
python -m cc_reservoir.scripts.merge_mask_archives ^
    --new <this folder>\2_CT_volumes_and_painter\data_fullres ^
    --old <this folder>\3_old_masks_superseded ^
    --out <this folder>\1_masks_USE_THESE --write --despeckle 20
```

The painter's own code in `2_CT_volumes_and_painter\software\` is refreshed from the
`cisterna-chyli-reservoir` repo automatically every 4 hours by the `CisternaChyli-AutoSync`
scheduled task, so a fix committed to the repo reaches everyone without anyone copying files.

Source for the painter and the merge/audit scripts: `cisterna-chyli-reservoir`
(`cc_reservoir.scripts.merge_mask_archives`, `cc_reservoir.scripts.audit_edit_sidecars`).

---
*Reorganised 2026-08-26. Previously these sat as three separate folders at the top of
`Lymph_Studies\`, under names that gave no hint which one to use:*

| old name | now |
|---|---|
| `CC_Masks_merged_2026-08` | `1_masks_USE_THESE` |
| `Shu_Nie_CC_Painter` | `2_CT_volumes_and_painter` |
| `Egor_FullRes_Segmentations` | `3_old_masks_superseded` |
