# CC/TD Context Viewer — design

**Goal:** An interactive QA tool to judge the integrated-HU CC/TD segmentation against anatomy —
three scrollable MPR planes with a segmentation overlay (CC, TD, kidney, bone) plus a live,
rotatable 3D model, in one window, scrollable through timepoints.

**Why:** Gifs/MIPs can't show whether the CC/TD sit in the right anatomical place. Seeing the
segmentation over the grayscale CT in axial/coronal/sagittal planes, next to the spine (and
kidney) as landmarks, plus a 3D model you can spin, is how we catch placement errors.

**Architecture:** Two stages, mirroring the existing render pipeline (data is on Linux/SMB; GL +
GUI run on the Mac). A **producer** on Linux builds one `.npz` per acquisition; a **viewer** on
the Mac loads that npz — no data access in the GUI.

**Tech stack:** Python; producer reuses the existing `cc_reservoir` segmentation + scipy/skimage;
viewer is **PySide6** (as in `~/Developer/hvsmr2_viz`) with an embedded **pyvistaqt `QtInteractor`**
for the live 3D panel.

---

## Components

### Producer (run on Linux)

- **`src/cc_reservoir/measure/landmarks.py`** — pure HU-threshold landmark segmentation.
  - `bone_mask(vol, hu=250)` → boolean: HU ≥ threshold, morphological open + remove small objects.
  - `kidney_mask(vol, lo, hi, ...)` → boolean: HU window + connected components + a location prior
    (lateral to the spine, abdominal z-band). **Best-effort** (see Kidney caveat).
- **`src/cc_reservoir/measure/context.py`** — regional field of view.
  - `load_context_patch(spec, pad_xy=100, pad_z=40)` → regional HU volumes (per timepoint) cropped
    to the CC|TD union bbox padded in-plane by `pad_xy` and in z by `pad_z`, plus `lo` (regional
    crop corner in the full volume) and `voxel_mm`.
  - `place(mask_tight, lo_tight, lo_regional, shape_regional)` → the tight-patch CC/TD mask written
    into a regional-sized array at `lo_tight − lo_regional` (the existing segmentation is ROI-bounded,
    so it is unchanged — only relocated).
- **`src/cc_reservoir/scripts/build_context_viz.py`** — orchestrator → `context_<tag>.npz`:
  - Run the existing integrated-HU pipeline (`fixed_lumen`, `local_opacified_ref`, `fwhm_lumen`,
    `split_caudal`) on the tight patch for each opacified frame → CC/TD masks; place into regional coords.
  - `bone_mask` / `kidney_mask` on the regional peak-frame HU (static anatomy).
  - Meshes: per-frame CC/TD swept tubes (reuse `centerline`/`equiv_radius`/`swept_tube` + marching
    cubes); static kidney/bone surfaces (gaussian-blurred mask → marching cubes).

### Viewer (run on the Mac)

- **`src/cc_reservoir/viz/labels.py`** — palette: 1 CC `#5aa469`, 2 TD `#3a8f9c`, 3 kidney `#e0a33a`,
  4 bone `#c9c9c9`; α≈0.45 for overlay.
- **`src/cc_reservoir/viz/slice_canvas.py`** — adapted from hvsmr2_viz `SliceCanvas`: grayscale HU
  (window/level) + semi-transparent label overlay + cyan crosshair + mouse-wheel slice scroll.
- **`src/cc_reservoir/viz/viewer.py`** — `TriPlanar3DWindow`, entry point `python -m cc_reservoir.viz.viewer <npz>`:
  - 2×2 grid: axial / coronal / sagittal `SliceCanvas` + a `QtInteractor` 3D panel.
  - **Timepoint slider** (bottom): updates the three MPR backgrounds to that frame's HU and the
    CC/TD overlay + 3D CC/TD tubes to that frame; kidney/bone stay (static anatomy). Non-opacified
    frames show HU + landmarks only (no CC/TD — the lumen is invisible pre-contrast).
  - Click in any MPR → linked crosshair across all three.
  - 3D panel: CC/TD opaque colored tubes (focus) + bone/kidney **faint semi-transparent** surfaces
    (context, so they don't occlude the duct); free rotate/zoom.

---

## npz format (`context_<tag>.npz`)

| key | shape / dtype | meaning |
|---|---|---|
| `hu` | `(T, X, Y, Z)` int16 | regional HU per timepoint (MPR background) |
| `seg` | `(T, X, Y, Z)` uint8 | per-frame label: 0 bg, 1 CC, 2 TD (0 for non-opacified frames) |
| `land` | `(X, Y, Z)` uint8 | static landmarks: 0 bg, 3 kidney, 4 bone |
| `cc_v_{t}`,`cc_f_{t}`,`td_v_{t}`,`td_f_{t}` | float32 / int32 | per-frame CC/TD tube meshes (verts mm, faces) |
| `kidney_v`,`kidney_f`,`bone_v`,`bone_f` | float32 / int32 | static landmark surfaces |
| `voxel` | `(3,)` float | mm |
| `opac` | `(k,)` int | opacified frame indices |
| `peak_i` | int | peak frame |

Saved with `np.savez_compressed` (background-dominated arrays compress well; ~50–80 MB/acq).
Regional z is the duct range + `pad_z` (not the full 700-slice scan), keeping it bounded.

---

## Kidney caveat (verify first)

Bone via HU is clean. **Kidney is the weak link:** lymphangiography contrast is in the lymphatics,
not the blood, so the kidney sits at soft-tissue HU and may not threshold cleanly from liver/psoas.
**First implementation step: verify kidney HU separability on one real acq.** If separable → ship
`kidney_mask`; if not → fall back to bone-only context (drop the kidney label) rather than ship a
misleading mask. Bone is the primary landmark either way.

## Testing

- `landmarks`: `bone_mask`/`kidney_mask` recover synthetic high-/mid-HU blobs; reject background.
- `context.place`: a tight mask lands at the correct regional offset; round-trips a known box.
- label-map assembly: priority order (CC/TD over kidney/bone where they overlap) is deterministic.
- Viewer: manual — launch on a built npz, scroll slices + timepoints, confirm overlays register
  with the grayscale and the 3D rotates.

## Dependencies

PySide6 (already present — hvsmr2_viz uses it) + **pyvistaqt** (confirm/`pip install` on the Mac).

## Out of scope (YAGNI)

Editing/correcting the segmentation in the viewer; DICOM-SEG export; multiple acqs at once;
aorta/other landmarks (add later if needed). The viewer is read-only QA.
