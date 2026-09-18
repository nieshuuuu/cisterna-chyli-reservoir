# CC/TD Context Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An interactive QA viewer — 3 scrollable MPR planes with a CC/TD/kidney/bone segmentation overlay + a live rotatable 3D model, scrollable through timepoints — to judge the integrated-HU CC/TD segmentation against anatomy.

**Architecture:** Two stages (data is on Linux/SMB, GL+GUI on the Mac). A **producer** (`build_context_viz.py`, Linux) builds one `context_<tag>.npz` per acquisition; a **viewer** (`viz/viewer.py`, PySide6 + pyvistaqt, Mac) loads that npz. The producer reuses the existing integrated-HU segmentation untouched (it is ROI-bounded) and only adds a regional crop, HU-threshold landmarks, and meshes.

**Tech Stack:** Python, numpy, scipy.ndimage, skimage.measure (marching_cubes); PySide6 (GUI, as in `~/Developer/hvsmr2_viz/viewer_qt.py`), pyvistaqt `QtInteractor` (embedded 3D), pyvista.

## Global Constraints

- Reuse the existing segmentation (`cc_reservoir.measure.lumen` / `.volume`); do NOT reimplement it.
- Voxel size is `REAL_VOXEL_MM = (0.78, 0.78, 1.0)` mm; meshes are in mm (marching_cubes `spacing`).
- Run segmentation/`binary_dilation` only on cropped patches, never the full 512×512×~700 volume (it hangs — project memory).
- Producer runs on `imaging-lab-remote` with `CC_ARCHIVE=/home/molloi-lab/smb_mount/shared_drive/ImageData/Lymph_Studies`; GUI runs on the Mac and only reads the npz.
- Tests: pytest under `tests/`, run with `PYTHONPATH=src python3 -m pytest`.
- Label ids fixed: 1=CC, 2=TD, 3=kidney, 4=bone. Colors: CC `#5aa469`, TD `#3a8f9c`, kidney `#e0a33a`, bone `#c9c9c9`.

---

## File Structure

- `src/cc_reservoir/measure/landmarks.py` — `bone_mask`, `kidney_mask` (pure HU-threshold).
- `src/cc_reservoir/measure/context.py` — `load_context_patch`, `place_into`.
- `src/cc_reservoir/scripts/build_context_viz.py` — producer → npz.
- `src/cc_reservoir/viz/__init__.py`, `viz/labels.py` — palette + label→RGBA.
- `src/cc_reservoir/viz/slice_canvas.py` — `SliceCanvas` (adapted from hvsmr2_viz).
- `src/cc_reservoir/viz/viewer.py` — `TriPlanar3DWindow` + entry point.
- `tests/test_landmarks.py`, `tests/test_context.py`, `tests/test_labels.py`.

---

### Task 1: HU-threshold landmarks

**Files:**
- Create: `src/cc_reservoir/measure/landmarks.py`
- Test: `tests/test_landmarks.py`

**Interfaces:**
- Produces: `bone_mask(vol, hu=250.0, min_vox=64) -> np.ndarray(bool)`; `kidney_mask(vol, lo=20.0, hi=60.0, min_vox=400) -> np.ndarray(bool)`. `vol` is a 3D HU array.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_landmarks.py
import numpy as np
from cc_reservoir.measure.landmarks import bone_mask, kidney_mask

def _vol():
    v = np.full((40, 40, 40), 0.0)          # background ~0 HU
    v[5:12, 5:12, 5:12] = 600.0              # bone-like high-HU cube (343 vox)
    v[25:33, 25:33, 25:33] = 40.0            # kidney-like mid-HU cube (512 vox)
    v[20, 5, 5] = 800.0                      # lone bright speckle (must be removed by min_vox)
    return v

def test_bone_mask_keeps_high_hu_block_drops_speckle():
    m = bone_mask(_vol())
    assert m[5:12, 5:12, 5:12].all()         # the high-HU cube is bone
    assert not m[25:33, 25:33, 25:33].any()  # the 40 HU cube is not bone
    assert not m[20, 5, 5]                    # lone speckle removed (< min_vox)

def test_kidney_mask_keeps_mid_hu_block():
    m = kidney_mask(_vol())
    assert m[25:33, 25:33, 25:33].all()      # the 40 HU cube is kidney-like
    assert not m[5:12, 5:12, 5:12].any()     # bone is excluded (above hi)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_landmarks.py -v`
Expected: FAIL (`ModuleNotFoundError: cc_reservoir.measure.landmarks`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/landmarks.py
"""Anatomical landmarks by HU threshold (context for the CC/TD segmentation QA viewer).

Bone is clean (high HU). Kidney is best-effort: without renal contrast it sits at soft-tissue
HU, so the window + connected-component + size filter is approximate (see the spec's caveat)."""
import numpy as np
from scipy.ndimage import binary_opening, label, generate_binary_structure

_ST = generate_binary_structure(3, 1)


def _remove_small(mask, min_vox):
    lab, n = label(mask)
    if n == 0:
        return mask
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        keep[i] = (lab == i).sum() >= min_vox
    return keep[lab]


def bone_mask(vol, hu=250.0, min_vox=64):
    """Cortical bone: HU >= hu, opened once, small components removed."""
    return _remove_small(binary_opening(vol >= hu, _ST, 1), min_vox)


def kidney_mask(vol, lo=20.0, hi=60.0, min_vox=400):
    """Best-effort kidney: soft-tissue HU window, opened, small components removed."""
    return _remove_small(binary_opening((vol >= lo) & (vol <= hi), _ST, 1), min_vox)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_landmarks.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/landmarks.py tests/test_landmarks.py
git commit -m "feat(landmarks): HU-threshold bone + best-effort kidney masks"
```

---

### Task 2: Regional context patch + placement

**Files:**
- Create: `src/cc_reservoir/measure/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `cc_reservoir.measure.patch._load_acq` pattern; `cc_reservoir.io.volumes.mask_bbox/crop_to_bbox`.
- Produces: `place_into(mask_tight, lo_tight, lo_regional, shape_regional) -> np.ndarray(bool)`; `load_context_patch(spec, archive=ARCHIVE, pad_xy=100, pad_z=40, voxel_mm=REAL_VOXEL_MM) -> ContextPatch` with fields `vols (list[np.ndarray])`, `lo (tuple)`, `voxel_mm`, `shape`.

- [ ] **Step 1: Write the failing test** (placement is the pure, critical piece)

```python
# tests/test_context.py
import numpy as np
from cc_reservoir.measure.context import place_into

def test_place_into_offsets_correctly():
    tight = np.zeros((4, 4, 4), bool); tight[1, 1, 1] = True
    out = place_into(tight, lo_tight=(10, 20, 30), lo_regional=(8, 16, 24), shape_regional=(20, 20, 20))
    assert out.shape == (20, 20, 20)
    assert out[10 - 8 + 1, 20 - 16 + 1, 30 - 24 + 1]      # the True voxel lands at the offset
    assert out.sum() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_context.py -v` — Expected: FAIL (no module).

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/context.py
"""Regional field of view for the context viewer: a generously-padded crop around the CC|TD
union (so the spine and nearby kidney are in frame), plus placement of the tight-patch
segmentation into regional coordinates. The tight segmentation is ROI-bounded, so relocating
it (not recomputing) is exact."""
import glob, os
from dataclasses import dataclass

import numpy as np

from cc_reservoir.io import REAL_VOXEL_MM
from cc_reservoir.io.masks import load_cc_mask
from cc_reservoir.io.volumes import load_hu_volume, crop_to_bbox
from cc_reservoir.io.workingset import ARCHIVE


@dataclass
class ContextPatch:
    vols: list
    lo: tuple
    voxel_mm: tuple
    shape: tuple


def place_into(mask_tight, lo_tight, lo_regional, shape_regional):
    out = np.zeros(shape_regional, bool)
    o = tuple(int(a - b) for a, b in zip(lo_tight, lo_regional))
    s = mask_tight.shape
    out[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = mask_tight
    return out


def _regional_bbox(union, pad_xy, pad_z, shape):
    nz = np.argwhere(union)
    pad = np.array([pad_xy, pad_xy, pad_z])
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + 1 + pad, shape)
    return tuple(int(x) for x in lo), tuple(int(x) for x in hi)


def load_context_patch(spec, archive=ARCHIVE, pad_xy=100, pad_z=40, voxel_mm=REAL_VOXEL_MM):
    acq = os.path.join(archive, spec.session, spec.subpath)
    cc = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
    td_path = os.path.join(acq, "SEGMENT_dcm", "TD_dcm_01.mat")
    td = load_cc_mask(td_path) if os.path.exists(td_path) else np.zeros_like(cc)
    union = cc | (td if td.shape == cc.shape else np.zeros_like(cc))
    lo, hi = _regional_bbox(union, pad_xy, pad_z, cc.shape)
    vols = [crop_to_bbox(load_hu_volume(m), lo, hi)
            for m in sorted(glob.glob(os.path.join(acq, "MAT", "*01.mat")))
            if load_hu_volume(m).shape == cc.shape]
    return ContextPatch(vols=vols, lo=lo, voxel_mm=voxel_mm, shape=vols[0].shape)
```

> Note: the double `load_hu_volume` in the comprehension is wasteful; in Step 3 keep it simple, then in Step 4 confirm the test passes. (Producer Task 3 loads each volume once — see there.)

- [ ] **Step 4: Run test to verify it passes** — `PYTHONPATH=src python3 -m pytest tests/test_context.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/context.py tests/test_context.py
git commit -m "feat(context): regional FOV patch + tight-seg placement"
```

---

### Task 3: Producer — build_context_viz.py (+ verify kidney on real data)

**Files:**
- Create: `src/cc_reservoir/scripts/build_context_viz.py`

**Interfaces:**
- Consumes: `load_context_patch`, `place_into`, `bone_mask`, `kidney_mask`; existing `lumen` functions (`lumen_roi`, `static_baseline`, `fixed_lumen`, `local_opacified_ref`, `fwhm_lumen`, `split_caudal`, `centerline`, `equiv_radius`, `swept_tube`, `cc_td_boundary_z`) and `load_cc_td_patch`.
- Produces: `context_<tag>.npz` with the keys in the spec table.

- [ ] **Step 1: Write the producer**

```python
# src/cc_reservoir/scripts/build_context_viz.py
"""Build the context-viewer npz for one acquisition (run on Linux where the data lives).

  CC_ARCHIVE=<path> CC_SESSION=<s> CC_SUB=<a> CC_CTX_OUT=<dir> \
    PYTHONPATH=src python3 -m cc_reservoir.scripts.build_context_viz
"""
import os
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import marching_cubes

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_td_patch
from cc_reservoir.measure.context import load_context_patch, place_into
from cc_reservoir.measure.landmarks import bone_mask, kidney_mask
from cc_reservoir.measure.lumen import (lumen_roi, static_baseline, fixed_lumen,
                                        local_opacified_ref, fwhm_lumen, split_caudal,
                                        cc_td_boundary_z, centerline, equiv_radius, swept_tube)

SESSION, SUB = os.environ.get("CC_SESSION", "09_07_22_data"), os.environ.get("CC_SUB", "Acq16")
OUT = os.environ.get("CC_CTX_OUT", "/tmp/cc_ctx"); os.makedirs(OUT, exist_ok=True)
tag = f"{SESSION}_{os.path.basename(SUB)}"
OPAC_MIN_E = 80.0
spec = [s for s in WORKING_SET if s.session == SESSION and s.subpath == SUB][0]

# regional FOV (per-timepoint HU) + tight patch (segmentation, reused untouched)
ctx = load_context_patch(spec)
tight = load_cc_td_patch(spec)
vox = tight.voxel_mm
rois = lumen_roi(tight); base = static_baseline(tight)
master = fixed_lumen(tight, rois, baseline=base); zb = cc_td_boundary_z(tight)
enh = [None] * len(tight.vols)
from cc_reservoir.measure.lumen import frame_enhancement
enh = [frame_enhancement(tight, i, rois, base) for i in range(len(tight.vols))]
mcc, mtd = split_caudal(master, zb)
opac = [i for i in range(len(enh)) if min(float(enh[i][mcc].mean()), float(enh[i][mtd].mean())) >= OPAC_MIN_E]
s_o = [local_opacified_ref(e, master) for e in enh]

T = len(ctx.vols); shp = ctx.shape
hu = np.stack(ctx.vols).astype(np.int16)
seg = np.zeros((T,) + shp, np.uint8)
cx, cy = centerline(master)
zc = np.where(master.any(axis=(0, 1)))[0]; zrange = range(int(zc.min()), int(zc.max()) + 1)
f = 2; sp = tuple(c / f for c in vox); arrs = {}


def tube(seg_tight):
    tb = swept_tube(cx, cy, equiv_radius(seg_tight, zrange), zrange, master.shape)
    v, fc, _, _ = marching_cubes(gaussian_filter(zoom(tb.astype(float), f, order=1), f * 0.8), 0.5, spacing=sp)
    return v.astype(np.float32) + np.array(tight.lo) * np.array(vox) - np.array(ctx.lo) * np.array(vox), fc.astype(np.int32)


for i in opac:
    fwhm = fwhm_lumen(enh[i], master, s_o[i])
    cc_t, td_t = split_caudal(fwhm, zb)
    seg[i][place_into(cc_t, tight.lo, ctx.lo, shp)] = 1
    seg[i][place_into(td_t, tight.lo, ctx.lo, shp)] = 2
    arrs[f"cc_v_{i}"], arrs[f"cc_f_{i}"] = tube(cc_t)
    arrs[f"td_v_{i}"], arrs[f"td_f_{i}"] = tube(td_t)

# static landmarks on the regional peak frame
peak = ctx.vols[tight.peak_i]
bone = bone_mask(peak); kidney = kidney_mask(peak)
land = np.zeros(shp, np.uint8); land[kidney] = 3; land[bone] = 4
print(f"{tag}: opac {opac}; bone {int(bone.sum())} vox; kidney {int(kidney.sum())} vox "
      f"(VERIFY kidney looks like a kidney, not liver/psoas — else drop in viewer)")


def surf(mask, decim=0.0):
    if mask.sum() < 200:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v, fc, _, _ = marching_cubes(gaussian_filter(mask.astype(float), 1.0), 0.5, spacing=vox)
    return v.astype(np.float32), fc.astype(np.int32)


arrs["bone_v"], arrs["bone_f"] = surf(bone)
arrs["kidney_v"], arrs["kidney_f"] = surf(kidney)
np.savez_compressed(os.path.join(OUT, f"context_{tag}.npz"), hu=hu, seg=seg, land=land,
                    voxel=np.array(vox), opac=np.array(opac, int), peak_i=tight.peak_i, **arrs)
print(f"saved {OUT}/context_{tag}.npz  shape {shp} T={T}")
```

- [ ] **Step 2: Run on the real data (Acq16) and VERIFY kidney** — this is the spec's kidney gate.

```bash
cd ~/Developer/cisterna-chyli-reservoir && rsync -az src/ imaging-lab-remote:/home/molloi-lab/cc-reservoir/src/
ssh imaging-lab-remote 'cd /home/molloi-lab/cc-reservoir && CC_ARCHIVE=/home/molloi-lab/smb_mount/shared_drive/ImageData/Lymph_Studies CC_SESSION=09_07_22_data CC_SUB=Acq16 CC_CTX_OUT=/tmp/cc_ctx PYTHONPATH=src python3 -m cc_reservoir.scripts.build_context_viz'
```
Expected: prints `opac [2,3,4,5]`, a sizeable `bone` voxel count, and a `kidney` count. **Pull the npz, render a quick MIP of `land` in the controller and eyeball whether the kidney blob is plausibly renal** (lateral, paired, abdominal z). If kidney is clearly liver/psoas/noise, set kidney to empty in the producer (`kidney = np.zeros_like(bone)`) and proceed bone-only — do not ship a misleading kidney.

- [ ] **Step 3: Commit**

```bash
git add src/cc_reservoir/scripts/build_context_viz.py
git commit -m "feat(context-viz): producer npz (regional HU + CC/TD seg + bone/kidney + meshes)"
```

---

### Task 4: Viewer label palette + SliceCanvas

**Files:**
- Create: `src/cc_reservoir/viz/__init__.py` (empty), `src/cc_reservoir/viz/labels.py`, `src/cc_reservoir/viz/slice_canvas.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Produces: `LABEL_RGBA: np.ndarray(5, 4) float [0,1]` (rows = label id 0..4, α baked); `overlay_rgba(gray_u8, label_slice, alpha=0.45) -> np.ndarray(H,W,4) uint8`. `SliceCanvas(QWidget)` with `set_data(hu_vol, label_vol)`, `set_index(i)`, wheel-scroll, crosshair signal.

- [ ] **Step 1: Write the failing test** (the pure compositing, not the Qt widget)

```python
# tests/test_labels.py
import numpy as np
from cc_reservoir.viz.labels import LABEL_RGBA, overlay_rgba

def test_palette_has_five_rows_background_transparent():
    assert LABEL_RGBA.shape == (5, 4)
    assert LABEL_RGBA[0, 3] == 0.0           # background fully transparent

def test_overlay_blends_label_over_gray():
    gray = np.full((4, 4), 100, np.uint8)
    lab = np.zeros((4, 4), np.uint8); lab[0, 0] = 1          # CC
    out = overlay_rgba(gray, lab, alpha=0.5)
    assert out.shape == (4, 4, 4) and out.dtype == np.uint8
    assert (out[1, 1, :3] == 100).all()      # un-labelled pixel stays gray
    assert not (out[0, 0, :3] == 100).all()  # labelled pixel tinted toward CC green
```

- [ ] **Step 2: Run to verify it fails** — `PYTHONPATH=src python3 -m pytest tests/test_labels.py -v` → FAIL.

- [ ] **Step 3: Implement `labels.py`**

```python
# src/cc_reservoir/viz/labels.py
"""Label palette + grayscale↔label compositing for the MPR overlay. Label ids: 1 CC, 2 TD,
3 kidney, 4 bone (0 = background)."""
import numpy as np

_HEX = ["#00000000", "#5aa469", "#3a8f9c", "#e0a33a", "#c9c9c9"]   # 0 bg, 1 CC, 2 TD, 3 kidney, 4 bone


def _rgba(h):
    h = h.lstrip("#")
    if len(h) == 8:
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4, 6)]
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]


LABEL_RGBA = np.array([_rgba(h) for h in _HEX], float)
LABEL_NAME = {1: "CC", 2: "TD", 3: "kidney", 4: "bone"}


def overlay_rgba(gray_u8, label_slice, alpha=0.45):
    """RGBA image: grayscale base with labels alpha-blended on top (per-label α scaled by alpha)."""
    h, w = gray_u8.shape
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = gray_u8
    out[..., 3] = 255
    for lid in range(1, 5):
        m = label_slice == lid
        if not m.any():
            continue
        col = (LABEL_RGBA[lid, :3] * 255)
        a = alpha
        out[m, :3] = (out[m, :3] * (1 - a) + col * a).astype(np.uint8)
    return out
```

- [ ] **Step 4: Run to verify it passes** — `PYTHONPATH=src python3 -m pytest tests/test_labels.py -v` → PASS.

- [ ] **Step 5: Implement `slice_canvas.py`** — adapt `~/Developer/hvsmr2_viz/viewer_qt.py`'s `SliceCanvas` (read it first). Keep its window/level (`apply_wl`), wheel-scroll accumulation, and `QImage` painting. Changes: (a) take `(hu_vol, label_vol)` 3D arrays + a `view` axis; (b) build the per-slice display via `overlay_rgba(apply_wl(hu_slice), label_slice)`; (c) emit a `voxelClicked(i,j,k)` signal for crosshair linking. One class, ~150 lines.

- [ ] **Step 6: Commit**

```bash
git add src/cc_reservoir/viz/__init__.py src/cc_reservoir/viz/labels.py src/cc_reservoir/viz/slice_canvas.py tests/test_labels.py
git commit -m "feat(viz): label palette + overlay compositing + adapted SliceCanvas"
```

---

### Task 5: Viewer window (2×2 MPR + embedded 3D + timepoint slider)

**Files:**
- Create: `src/cc_reservoir/viz/viewer.py`

**Interfaces:**
- Consumes: `SliceCanvas`, `LABEL_RGBA`, the `context_<tag>.npz` keys.
- Produces: `python -m cc_reservoir.viz.viewer <npz>` launches the window.

- [ ] **Step 1: Confirm pyvistaqt is available (dependency gate)**

```bash
python3 -c "import pyvistaqt, PySide6; print('ok', pyvistaqt.__version__)"
```
If missing: `python3 -m pip install pyvistaqt`. Record the version. (If pyvistaqt cannot embed on this Mac/PyVista combo, fall back to a separate `pv.Plotter(off_screen=False).show()` window launched from the same process — the MPR window stays the same.)

- [ ] **Step 2: Implement `viewer.py`**

`TriPlanar3DWindow(QMainWindow)`:
- Load npz; build a 2×2 `QGridLayout`: three `SliceCanvas` (axial=z, coronal=y, sagittal=x) at (0,0),(0,1),(1,0) and a `QtInteractor` at (1,1).
- A bottom `QSlider` over timepoints `0..T-1`. On change: for each canvas `set_data(hu[t], seg[t] | land_broadcast)` where the displayed label volume is `np.where(seg[t] > 0, seg[t], land)` (CC/TD take priority over static landmarks); refresh the 3D CC/TD actors from `cc_v_{t}/cc_f_{t}`, `td_v_{t}/td_f_{t}`.
- 3D panel setup once: add bone (`bone_v/f`) and kidney (`kidney_v/f`) as faint `opacity=0.18` surfaces (gray / gold), add CC (green) + TD (teal) opaque actors for the initial frame, `enable_anti_aliasing`, reset camera. Per-frame: remove old CC/TD actors by name, add the new ones (PyVista `add_mesh(..., name="cc")` replaces by name).
- Crosshair: each canvas `voxelClicked` → set all three indices + draw crosshair (reuse hvsmr2_viz's pattern).

Provide the full file in implementation (≈220 lines). Key 3D refresh:

```python
def _set_frame(self, t):
    lab = np.where(self.seg[t] > 0, self.seg[t], self.land)
    for cv in self.canvases:
        cv.set_label_volume(lab)
    for tag, cmap_col in (("cc", "#5aa469"), ("td", "#3a8f9c")):
        v, f = self.d.get(f"{tag}_v_{t}"), self.d.get(f"{tag}_f_{t}")
        if v is not None and len(v):
            faces = np.hstack([np.full((len(f), 1), 3, int), f]).ravel()
            self.plotter.add_mesh(pv.PolyData(v, faces), color=cmap_col, name=tag,
                                  smooth_shading=True, specular=0.4)
        else:
            self.plotter.remove_actor(tag, render=False)
    self.plotter.render()
```

- [ ] **Step 3: Manual verification** — build npz for Acq16 (Task 3), pull to Mac, run:

```bash
CC_CTX_NPZ=/tmp/context_09_07_22_data_Acq16.npz python3 -m cc_reservoir.viz.viewer /tmp/context_09_07_22_data_Acq16.npz
```
Confirm: three MPR planes show the CT with CC(green)/TD(teal) overlay + bone(gray)/kidney(gold); mouse-wheel scrolls slices; the timepoint slider changes the HU + CC/TD (bone/kidney stay); the 3D panel shows the tubes + faint bone/kidney and rotates with the mouse. Screenshot for the record.

- [ ] **Step 4: Commit**

```bash
git add src/cc_reservoir/viz/viewer.py
git commit -m "feat(viz): triplanar + embedded 3D context viewer"
```

---

### Task 6: End-to-end on all clean acqs + docs

**Files:**
- Modify: `docs/cc_transport_model.md` (or a short `docs/context_viewer.md`) — how to build + launch.

- [ ] **Step 1:** Run the producer for the 8 clean acqs (loop, as in the phase2c batch), pull the npzs to the Mac.
- [ ] **Step 2:** Launch the viewer on 2–3 acqs (Acq16 + one 07_20_22 + Acq5); confirm overlays register and the segmentation looks anatomically placed (CC anterior to spine, TD ascending in the mediastinum). Note any acq where CC/TD is clearly misplaced — that's the QA payoff.
- [ ] **Step 3:** Write a short usage doc (build command, launch command, controls) and commit.

```bash
git add docs/context_viewer.md && git commit -m "docs: context viewer usage"
```

---

## Self-Review

**Spec coverage:** 3 MPR + overlay (Task 4/5), embedded interactive 3D (Task 5), regional FOV (Task 2), timepoint scroll (Task 5), bone+kidney HU (Task 1), kidney verify-first gate (Task 3 Step 2), npz format (Task 3), reuse existing seg (Task 3), tests (Tasks 1,2,4), manual GUI verify (Task 5/6). All covered.

**Placeholder scan:** no TBD/"handle errors"; code shown for every testable step; GUI steps give the structure + key code + a manual-verify command (GUI can't be unit-tested). The `load_context_patch` double-load is flagged for the implementer.

**Type consistency:** label ids 1–4 consistent across `landmarks`/producer/`labels`/`viewer`; `place_into` signature matches its caller in Task 3; mesh keys `cc_v_{t}` etc. consistent between producer (Task 3) and viewer (Task 5); `ContextPatch` fields (`vols`,`lo`,`voxel_mm`,`shape`) used consistently.
