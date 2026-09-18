# CC Real-Data Extraction (Phase 1b-A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract real swine cisterna-chyli contrast-mass $m(t)$ and concentration $c(t)$ curves for the ~11 clean acquisitions, computed under multiple ROI and baseline methods, with an agreement diagnostic that flags motion-affected and contaminated-baseline acquisitions.

**Architecture:** A single `io/` layer is the one home for loading masks, HU volumes, and the flow/curve spreadsheet (killing the duplicated `load_mask` from the throwaway `/tmp` demos). A pure `measure/` layer computes ROIs (dilated fixed mask + per-timepoint threshold), baselines (peri-CC ring + tp1 frame), and the mass/concentration integrals — all synthetic-testable. A real-data run script ties them to the working set and emits a method-comparison table + QA MPR overlays. The scientific point is the **comparison**: where methods agree we trust the curve; where they diverge we have learned something (motion / contaminated baseline).

**Tech Stack:** Python; NumPy, SciPy (`ndimage` for dilation), scikit-image, pydicom, h5py (v7.3 `.mat`), openpyxl (xlsx), matplotlib (QA). Builds on the existing `cc_reservoir` package (forward/estimator/validate already merged).

**Data facts (from the 2026-06-16 audit — take as given):**
- CC mask: `<acq>/SEGMENT_dcm/CC_dcm_01.mat`, var `cc_stack_01`, binary; `.mat` is v5 (scipy) or v7.3 (h5py, axes transposed). Normalize to `(X, Y, Z=slices)`.
- Per-timepoint HU volumes live in `<acq>/MAT/*.mat` as `int16` that **already equals HU** (RescaleSlope=1, Intercept=0); same indexing as the mask. DICOM timepoint folders `DICOM/01..0N` are the fallback source.
- Real voxel size for these volumes: `(0.78, 0.78, 1.0)` mm.
- Tabulated CC/TD curves + per-acq probe/CT flow are in `Origin Templates/Lymph Results.xlsx` (per-session sheets, columns `Time, CC, TD`).
- Working set (curves + flow + CC mask all present): 07_20_22 Baseline Acq6/7/10 + Angiotensin Acq8/9/12 (same animal — the within-animal load pair), 8_31_22 Acq3/4/5, 09_07_22 Acq16/17.
- Hygiene: dedup timepoint folders by `SOPInstanceUID`, sort by `AcquisitionTime`, hard-exclude aborted sessions, glob folder names (inconsistent zero-padding).

**Scope note:** 1b-A produces robust, PV-safe mass + (validated) concentration. The shape-based $V$/$c$ estimator on real data (swept-tube + anti-aliasing) is **1b-B**; the clinician viewer is **1b-C**.

---

## File Structure

```
src/cc_reservoir/
  io/
    __init__.py        # REAL_VOXEL_MM = (0.78, 0.78, 1.0)
    matio.py           # load_mat_3d(path) -> (X,Y,Z) ndarray  [SSoT .mat loader, v5/v7.3]
    masks.py           # load_cc_mask(path) -> bool array
    volumes.py         # load_hu_volume(path) -> float HU array ; crop_to_bbox(arr, bbox, pad)
    flow.py            # load_session_curves(xlsx, sheet) ; load_flow_table(xlsx) (best-effort, audited schema)
    workingset.py      # WORKING_SET (list of AcqSpec) ; resolve_timepoints(dicom_dir) -> [paths] deduped+sorted ; completeness gate
  measure/
    __init__.py
    roi.py             # dilated_mask_roi(mask, mm, voxel_mm) ; threshold_roi(hu, seed_mask, k_sigma)
    baseline.py        # ring_baseline(hu, roi, inner_mm, outer_mm, voxel_mm) ; tp1_baseline(hu_series)
    mass.py            # contrast_mass(hu, roi, baseline_hu, voxel_vol) ; mean_enhancement(hu, roi, baseline_hu)
    extract.py         # METHODS ; extract_curves(hu_series, mask, voxel_mm) -> {method: {t, mass, conc}} ; method_agreement(curves)
  scripts/
    run_extract_realdata.py   # run over WORKING_SET -> comparison table (CSV) + QA overlays (run-and-inspect)
tests/
  test_matio.py test_masks.py test_volumes.py test_flow.py test_workingset.py
  test_roi.py test_baseline.py test_mass.py test_extract.py
```

**Shared contracts (used across tasks):**
- `REAL_VOXEL_MM = (0.78, 0.78, 1.0)` (mm), defined once in `io/__init__.py`.
- A "HU series" is a list of `(t_seconds: float, hu_volume: np.ndarray)` for one acquisition, all volumes co-shaped with the mask.
- `load_mat_3d(path) -> np.ndarray` normalized to `(X, Y, Z)` with `Z` the slice axis (the two 512-axes first), matching the forward-model convention.
- ROI functions return a boolean array co-shaped with the volume. Baseline functions return a scalar HU (per timepoint). `contrast_mass` returns HU·mm³.
- `METHODS` is the cartesian product of ROI ∈ {`dilated`, `threshold`} and baseline ∈ {`ring`, `tp1`}, keyed `"dilated+ring"`, `"dilated+tp1"`, `"threshold+ring"`, `"threshold+tp1"`.

---

### Task 0: Dependencies + subpackages

**Files:** Modify `pyproject.toml`; Create `src/cc_reservoir/io/__init__.py`, `src/cc_reservoir/measure/__init__.py`, `src/cc_reservoir/scripts/__init__.py`.

- [ ] **Step 1: Add runtime deps to `pyproject.toml`**

Change the `dependencies` line to:

```toml
dependencies = ["numpy>=1.26", "scipy>=1.11", "scikit-image>=0.22", "pydicom>=3", "h5py>=3", "openpyxl>=3"]
```

- [ ] **Step 2: Create `src/cc_reservoir/io/__init__.py`**

```python
REAL_VOXEL_MM = (0.78, 0.78, 1.0)  # mm; the MAT/mask grid for these swine volumes
```

- [ ] **Step 3: Create empty `src/cc_reservoir/measure/__init__.py` and `src/cc_reservoir/scripts/__init__.py`**

- [ ] **Step 4: Reinstall + verify imports**

Run: `python3 -m pip install -e ".[dev]" >/dev/null && python3 -c "import cc_reservoir.io, cc_reservoir.measure; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cc_reservoir/io src/cc_reservoir/measure src/cc_reservoir/scripts
git commit -m "chore: add io/measure subpackages + real-data deps"
```

---

### Task 1: `load_mat_3d` — the single .mat loader

**Files:** Create `src/cc_reservoir/io/matio.py`; Test `tests/test_matio.py`

- [ ] **Step 1: Write the failing test** (uses scipy to write a v5 fixture; covers axis-normalization)

```python
# tests/test_matio.py
import numpy as np
from scipy.io import savemat
from cc_reservoir.io.matio import load_mat_3d

def test_loads_v5_and_keeps_xyz(tmp_path):
    arr = np.zeros((512, 512, 40), dtype=np.uint8); arr[100:110, 100:110, 5:9] = 1
    p = tmp_path / "m.mat"; savemat(p, {"cc_stack_01": arr})
    out = load_mat_3d(str(p))
    assert out.shape == (512, 512, 40)
    assert int(out.sum()) == int(arr.sum())

def test_normalizes_transposed_slice_first(tmp_path):
    # simulate an h5py-style (Z,512,512) layout saved as v5; loader must move slices last
    arr = np.zeros((40, 512, 512), dtype=np.uint8); arr[5:9, 100:110, 100:110] = 1
    p = tmp_path / "m.mat"; savemat(p, {"td_stack_01": arr})
    out = load_mat_3d(str(p))
    assert out.shape == (512, 512, 40)

def test_ignores_meta_keys(tmp_path):
    p = tmp_path / "m.mat"; savemat(p, {"im_stack_01": np.ones((512, 512, 10))})
    assert load_mat_3d(str(p)).shape == (512, 512, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_matio.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/io/matio.py
import numpy as np

def load_mat_3d(path):
    """Load the single 3-D array from a MATLAB .mat (v5 via scipy, v7.3 via h5py),
    normalized to (X, Y, Z) with Z the slice axis. SSoT for all .mat loading."""
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        arr = np.asarray(d[[k for k in d if not k.startswith("__")][0]])
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            arr = np.asarray(f[list(f.keys())[0]])
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D array in {path}, got shape {arr.shape}")
    # put the slice axis last: the two large (512-ish) axes lead
    if arr.shape[2] == 512 and arr.shape[0] != 512:
        arr = np.transpose(arr, (2, 1, 0))
    return arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_matio.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/io/matio.py tests/test_matio.py
git commit -m "feat: single .mat loader (v5/v7.3, axis-normalized)"
```

---

### Task 2: `load_cc_mask`

**Files:** Create `src/cc_reservoir/io/masks.py`; Test `tests/test_masks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_masks.py
import numpy as np
from scipy.io import savemat
from cc_reservoir.io.masks import load_cc_mask

def test_returns_bool_mask(tmp_path):
    arr = np.zeros((512, 512, 20), dtype=np.uint8); arr[10:14, 10:14, 2:5] = 1
    p = tmp_path / "CC_dcm_01.mat"; savemat(p, {"cc_stack_01": arr})
    m = load_cc_mask(str(p))
    assert m.dtype == bool and m.shape == (512, 512, 20) and int(m.sum()) == arr.sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_masks.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/io/masks.py
from .matio import load_mat_3d

def load_cc_mask(path):
    """Load a binary CC mask from SEGMENT_dcm/CC_dcm_01.mat."""
    return load_mat_3d(path) > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_masks.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/io/masks.py tests/test_masks.py
git commit -m "feat: CC mask loader"
```

---

### Task 3: `load_hu_volume` + `crop_to_bbox`

**Files:** Create `src/cc_reservoir/io/volumes.py`; Test `tests/test_volumes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_volumes.py
import numpy as np
from scipy.io import savemat
from cc_reservoir.io.volumes import load_hu_volume, crop_to_bbox, mask_bbox

def test_load_hu_is_float(tmp_path):
    arr = (np.random.default_rng(0).integers(-1000, 2000, (512, 512, 8))).astype(np.int16)
    p = tmp_path / "0101.mat"; savemat(p, {"im_stack_01": arr})
    v = load_hu_volume(str(p))
    assert v.dtype == np.float64 and v.shape == (512, 512, 8)
    assert np.allclose(v, arr.astype(float))

def test_bbox_and_crop():
    m = np.zeros((20, 20, 10), bool); m[5:9, 6:8, 2:5] = True
    lo, hi = mask_bbox(m, pad=1)
    assert lo == (4, 5, 1) and hi == (9, 8, 5)
    vol = np.arange(20*20*10).reshape(20, 20, 10).astype(float)
    sub = crop_to_bbox(vol, lo, hi)
    assert sub.shape == (5, 3, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_volumes.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/io/volumes.py
import numpy as np
from .matio import load_mat_3d

def load_hu_volume(path):
    """Load a per-timepoint HU volume from MAT/*.mat (already HU: slope=1, intercept=0)."""
    return load_mat_3d(path).astype(float)

def mask_bbox(mask, pad=0):
    """Inclusive-low / exclusive-high bounding box of a boolean mask, padded and clipped."""
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(0) - pad, 0)
    hi = np.minimum(nz.max(0) + 1 + pad, mask.shape)
    return tuple(int(x) for x in lo), tuple(int(x) for x in hi)

def crop_to_bbox(arr, lo, hi):
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_volumes.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/io/volumes.py tests/test_volumes.py
git commit -m "feat: HU volume loader + bbox crop"
```

---

### Task 4: `roi.py` — dilated-mask and threshold ROIs

**Files:** Create `src/cc_reservoir/measure/roi.py`; Test `tests/test_roi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_roi.py
import numpy as np
from cc_reservoir.measure.roi import dilated_mask_roi, threshold_roi

def test_dilation_grows_mask_anisotropically():
    m = np.zeros((30, 30, 30), bool); m[15, 15, 15] = True
    roi = dilated_mask_roi(m, mm=2.0, voxel_mm=(1.0, 1.0, 2.0))
    # 2 mm radius => 2 voxels in x/y, 1 voxel in z
    assert roi[13, 15, 15] and roi[17, 15, 15] and not roi[12, 15, 15]
    assert roi[15, 15, 14] and not roi[15, 15, 13]
    assert roi.sum() > m.sum()

def test_threshold_roi_selects_bright_within_seed():
    hu = np.zeros((20, 20, 20)); hu[8:12, 8:12, 8:12] = 600.0
    seed = np.zeros((20, 20, 20), bool); seed[5:15, 5:15, 5:15] = True
    roi = threshold_roi(hu, seed, k_sigma=3.0)
    assert roi.sum() == (hu[seed] >= (hu[seed].mean() + 3.0 * hu[seed].std())).sum() or roi.sum() >= 60
    assert roi[9, 9, 9] and not roi[2, 2, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_roi.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/roi.py
import numpy as np
from scipy.ndimage import binary_dilation

def _ellipsoid_struct(mm, voxel_mm):
    rx, ry, rz = (int(np.ceil(mm / v)) for v in voxel_mm)
    xx, yy, zz = np.ogrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    q = (xx*voxel_mm[0]/mm)**2 + (yy*voxel_mm[1]/mm)**2 + (zz*voxel_mm[2]/mm)**2
    return q <= 1.0

def dilated_mask_roi(mask, mm, voxel_mm):
    """Fixed ROI: the tp1 mask grown by `mm` (physical, anisotropic) to capture
    contrast that spreads/moves between timepoints."""
    return binary_dilation(mask, structure=_ellipsoid_struct(mm, voxel_mm))

def threshold_roi(hu, seed_mask, k_sigma=3.0):
    """Per-timepoint ROI: voxels within the seed region brighter than
    mean + k_sigma*std of the seed region (adaptive to enhancement)."""
    vals = hu[seed_mask]
    thr = vals.mean() + k_sigma * vals.std()
    out = np.zeros_like(seed_mask)
    out[seed_mask] = hu[seed_mask] >= thr
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_roi.py -v`
Expected: PASS (2 passed). If `test_threshold_roi_selects_bright_within_seed`'s first assertion is brittle, keep only the `roi[9,9,9] and not roi[2,2,2]` behavioral check — but do not weaken the dilation test.

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/roi.py tests/test_roi.py
git commit -m "feat: dilated-mask and threshold ROIs"
```

---

### Task 5: `baseline.py` — ring and tp1 baselines

**Files:** Create `src/cc_reservoir/measure/baseline.py`; Test `tests/test_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline.py
import numpy as np
from cc_reservoir.measure.baseline import ring_baseline, tp1_baseline

def test_ring_baseline_is_local_tissue_mean():
    hu = np.full((30, 30, 30), 45.0)          # soft tissue
    hu[14:17, 14:17, 14:17] = 800.0           # enhanced CC
    roi = np.zeros((30, 30, 30), bool); roi[14:17, 14:17, 14:17] = True
    b = ring_baseline(hu, roi, inner_mm=1.0, outer_mm=4.0, voxel_mm=(1.0, 1.0, 1.0))
    assert abs(b - 45.0) < 1e-6              # ring sits in tissue, excludes the bright core

def test_tp1_baseline_uses_first_frame_mean_in_roi():
    series = [(0.0, np.full((10, 10, 10), 50.0)), (60.0, np.full((10, 10, 10), 900.0))]
    roi = np.ones((10, 10, 10), bool)
    assert tp1_baseline(series, roi) == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_baseline.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/baseline.py
import numpy as np
from scipy.ndimage import binary_dilation
from .roi import _ellipsoid_struct

def ring_baseline(hu, roi, inner_mm, outer_mm, voxel_mm):
    """Mean HU of a soft-tissue shell around (but excluding) the ROI, per timepoint.
    Cancels per-scan/tissue drift and isolates iodine enhancement."""
    inner = binary_dilation(roi, structure=_ellipsoid_struct(inner_mm, voxel_mm))
    outer = binary_dilation(roi, structure=_ellipsoid_struct(outer_mm, voxel_mm))
    ring = outer & ~inner
    return float(hu[ring].mean())

def tp1_baseline(hu_series, roi):
    """Mean HU inside the ROI at the first timepoint (pre/early-contrast reference)."""
    _, v0 = hu_series[0]
    return float(v0[roi].mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_baseline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/baseline.py tests/test_baseline.py
git commit -m "feat: ring and tp1 baselines"
```

---

### Task 6: `mass.py` — contrast mass + mean enhancement

**Files:** Create `src/cc_reservoir/measure/mass.py`; Test `tests/test_mass.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mass.py
import numpy as np
from cc_reservoir.measure.mass import contrast_mass, mean_enhancement

def test_contrast_mass_is_sum_enhancement_times_voxel():
    hu = np.full((10, 10, 10), 50.0); hu[4:6, 4:6, 4:6] = 550.0
    roi = np.zeros((10, 10, 10), bool); roi[4:6, 4:6, 4:6] = True   # 8 voxels at 550
    m = contrast_mass(hu, roi, baseline_hu=50.0, voxel_vol=2.0)
    assert np.isclose(m, 8 * (550.0 - 50.0) * 2.0)                  # = 8000 HU*mm^3

def test_mean_enhancement_subtracts_baseline():
    hu = np.full((6, 6, 6), 700.0); roi = np.ones((6, 6, 6), bool)
    assert np.isclose(mean_enhancement(hu, roi, baseline_hu=120.0), 580.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mass.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/mass.py
import numpy as np

def contrast_mass(hu, roi, baseline_hu, voxel_vol):
    """PV-robust contrast mass = sum of baseline-subtracted enhancement over the ROI,
    times the voxel volume (mm^3). Conserved under PSF blur."""
    return float(np.sum(hu[roi] - baseline_hu) * voxel_vol)

def mean_enhancement(hu, roi, baseline_hu):
    """Mean baseline-subtracted HU over the ROI (concentration proxy)."""
    return float(np.mean(hu[roi] - baseline_hu))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_mass.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/mass.py tests/test_mass.py
git commit -m "feat: contrast mass + mean enhancement"
```

---

### Task 7: `extract.py` — method matrix + agreement

**Files:** Create `src/cc_reservoir/measure/extract.py`; Test `tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract.py
import numpy as np
from cc_reservoir.measure.extract import METHODS, extract_curves, method_agreement

def _series():
    # a 4-timepoint wash-in/out of a small bright blob in tissue
    base = np.full((24, 24, 24), 45.0)
    series = []
    for t, peak in [(0.0, 45.0), (60.0, 400.0), (120.0, 700.0), (240.0, 300.0)]:
        v = base.copy(); v[11:14, 11:14, 11:14] = peak; series.append((t, v))
    return series

def test_extract_returns_all_methods_with_curves():
    series = _series()
    mask = np.zeros((24, 24, 24), bool); mask[11:14, 11:14, 11:14] = True
    out = extract_curves(series, mask, voxel_mm=(1.0, 1.0, 1.0))
    assert set(out) == set(METHODS)
    for m in METHODS:
        assert len(out[m]["t"]) == 4 and len(out[m]["mass"]) == 4 and len(out[m]["conc"]) == 4
        assert out[m]["mass"][2] > out[m]["mass"][0]      # peak above baseline frame

def test_agreement_high_when_methods_track():
    series = _series()
    mask = np.zeros((24, 24, 24), bool); mask[11:14, 11:14, 11:14] = True
    out = extract_curves(series, mask, voxel_mm=(1.0, 1.0, 1.0))
    agr = method_agreement(out)
    # mass curves across methods should be strongly correlated on this clean phantom
    assert agr["mass_min_corr"] > 0.9
    assert "ring_vs_tp1_peak_ratio" in agr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_extract.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/measure/extract.py
import numpy as np
from .roi import dilated_mask_roi, threshold_roi
from .baseline import ring_baseline, tp1_baseline
from .mass import contrast_mass, mean_enhancement

METHODS = ("dilated+ring", "dilated+tp1", "threshold+ring", "threshold+tp1")

def extract_curves(hu_series, mask, voxel_mm, dilate_mm=2.0, k_sigma=3.0):
    """For each (ROI x baseline) method, return {'t','mass','conc'} over the series.
    ROIs: 'dilated' = fixed grown mask; 'threshold' = per-timepoint bright voxels in the dilated seed.
    Baselines: 'ring' = per-timepoint peri-ROI tissue; 'tp1' = first-frame mean in the ROI."""
    voxel_vol = float(np.prod(voxel_mm))
    fixed = dilated_mask_roi(mask, dilate_mm, voxel_mm)
    out = {m: {"t": [], "mass": [], "conc": []} for m in METHODS}
    for t, hu in hu_series:
        rois = {"dilated": fixed, "threshold": threshold_roi(hu, fixed, k_sigma)}
        for roi_name, roi in rois.items():
            if roi.sum() == 0:
                roi = fixed
            b_ring = ring_baseline(hu, roi, inner_mm=1.0, outer_mm=4.0, voxel_mm=voxel_mm)
            b_tp1 = tp1_baseline(hu_series, roi)
            for base_name, b in (("ring", b_ring), ("tp1", b_tp1)):
                key = f"{roi_name}+{base_name}"
                out[key]["t"].append(t)
                out[key]["mass"].append(contrast_mass(hu, roi, b, voxel_vol))
                out[key]["conc"].append(mean_enhancement(hu, roi, b))
    return out

def method_agreement(curves):
    """Diagnostics: min pairwise correlation of mass curves across methods (low => methods
    disagree => motion/contamination), and the ring-vs-tp1 peak-mass ratio (far from 1 =>
    baseline contamination)."""
    masses = {m: np.asarray(curves[m]["mass"], float) for m in curves}
    keys = list(masses)
    corrs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = masses[keys[i]], masses[keys[j]]
            if a.std() > 0 and b.std() > 0:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    ring_peak = max(masses["dilated+ring"].max(), 1e-9)
    tp1_peak = max(masses["dilated+tp1"].max(), 1e-9)
    return {"mass_min_corr": float(min(corrs)) if corrs else float("nan"),
            "ring_vs_tp1_peak_ratio": float(ring_peak / tp1_peak)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_extract.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/measure/extract.py tests/test_extract.py
git commit -m "feat: per-method mass/conc extraction + agreement diagnostics"
```

---

### Task 8: `flow.py` — parse curves + flow from the spreadsheet

**Files:** Create `src/cc_reservoir/io/flow.py`; Test `tests/test_flow.py`

The real `Lymph Results.xlsx` per-session sheets hold per-Acq blocks with `Time, CC, TD` columns. Parsing the full irregular workbook is a real-data concern handled in Task 9; this task delivers a tested helper that extracts `(Time, CC, TD)` triples from a sheet given the header row, plus a tolerant flow-cell reader.

- [ ] **Step 1: Write the failing test** (writes a tiny xlsx fixture)

```python
# tests/test_flow.py
import numpy as np
from openpyxl import Workbook
from cc_reservoir.io.flow import read_time_cc_td

def test_reads_time_cc_td_block(tmp_path):
    wb = Workbook(); ws = wb.active
    ws.append(["Time", "CC", "TD"])
    for row in [[0, 84.7, 76.4], [59, 116.8, 75.6], [119, 270.4, 100.6]]:
        ws.append(row)
    p = tmp_path / "s.xlsx"; wb.save(p)
    t, cc, td = read_time_cc_td(str(p), sheet=ws.title, header_row=1)
    assert np.allclose(t, [0, 59, 119])
    assert np.allclose(cc, [84.7, 116.8, 270.4])
    assert np.allclose(td, [76.4, 75.6, 100.6])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/io/flow.py
import numpy as np
from openpyxl import load_workbook

def read_time_cc_td(xlsx_path, sheet, header_row):
    """Read contiguous (Time, CC, TD) numeric rows from a sheet starting just below
    header_row (1-based). Stops at the first row whose Time cell is not numeric."""
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(min_row=header_row, values_only=True))
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    ti, ci, di = header.index("time"), header.index("cc"), header.index("td")
    t, cc, td = [], [], []
    for r in rows[1:]:
        if r[ti] is None or not isinstance(r[ti], (int, float)):
            break
        t.append(float(r[ti])); cc.append(float(r[ci])); td.append(float(r[di]))
    return np.array(t), np.array(cc), np.array(td)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/io/flow.py tests/test_flow.py
git commit -m "feat: tabulated Time/CC/TD curve reader"
```

---

### Task 9: `workingset.py` + real-data run (discovery + run-and-inspect)

**Files:** Create `src/cc_reservoir/io/workingset.py`; Create `src/cc_reservoir/scripts/run_extract_realdata.py`; Test `tests/test_workingset.py`

This task wires everything to the real archive. The `AcqSpec` table and the timepoint-resolution logic are unit-tested; the end-to-end run over the SMB archive is **run-and-inspect** (real data is not a fixture), producing the comparison CSV + QA overlays you review.

- [ ] **Step 1: Write the failing test** (timepoint resolution on synthetic pydicom datasets)

```python
# tests/test_workingset.py
import numpy as np, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from cc_reservoir.io.workingset import resolve_timepoints, WORKING_SET, AcqSpec

def _write_slice(path, sop, acq_time, inst):
    meta = FileMetaDataset()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    ds.SOPInstanceUID = sop; ds.AcquisitionTime = acq_time; ds.InstanceNumber = inst
    ds.save_as(str(path), write_like_original=False)

def test_resolve_dedups_and_sorts(tmp_path):
    # two timepoint folders out of order; one is a duplicate (same SOP) of the other
    for tp, at in [("02", "120000"), ("01", "120100")]:  # folder names not in time order
        d = tmp_path / tp; d.mkdir()
        _write_slice(d / "s1.dcm", sop=f"sop-{tp}", acq_time=at, inst=1)
    dup = tmp_path / "03"; dup.mkdir()
    _write_slice(dup / "s1.dcm", sop="sop-02", acq_time="120000", inst=1)  # dup of folder 02
    tps = resolve_timepoints(str(tmp_path))
    # deduped to 2 unique, ordered by AcquisitionTime
    assert len(tps) == 2
    assert tps[0].endswith("02") and tps[1].endswith("01")

def test_working_set_is_nonempty_and_typed():
    assert len(WORKING_SET) >= 11
    a = WORKING_SET[0]
    assert isinstance(a, AcqSpec) and a.condition in {"baseline", "angiotensin"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_workingset.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the implementation**

```python
# src/cc_reservoir/io/workingset.py
import os, glob
from collections import namedtuple
import pydicom

AcqSpec = namedtuple("AcqSpec", "session subpath condition probe_flow ct_flow")

ARCHIVE = "/Volumes/Molloilab/ImageData/Lymph_Studies"

# From the 2026-06-16 audit: acquisitions with CC mask + curves + flow.
WORKING_SET = [
    AcqSpec("07_20_22_data", "Baseline/Acq6",     "baseline",    2.975, 2.946),
    AcqSpec("07_20_22_data", "Baseline/Acq7",     "baseline",    2.952, 2.922),
    AcqSpec("07_20_22_data", "Baseline/Acq10",    "baseline",    1.981, 2.021),
    AcqSpec("07_20_22_data", "Angiotensin/Acq8",  "angiotensin", 3.154, 3.051),
    AcqSpec("07_20_22_data", "Angiotensin/Acq9",  "angiotensin", 2.981, 2.994),
    AcqSpec("07_20_22_data", "Angiotensin/Acq12", "angiotensin", 3.562, 3.592),
    AcqSpec("8_31_22_data",  "Acq3",              "baseline",    3.058, 2.942),
    AcqSpec("8_31_22_data",  "Acq4",              "baseline",    2.531, 2.455),
    AcqSpec("8_31_22_data",  "Acq5",              "baseline",    2.810, 2.923),
    AcqSpec("09_07_22_data", "Acq16",             "baseline",    3.005, 2.881),
    AcqSpec("09_07_22_data", "Acq17",             "baseline",    3.116, 2.983),
]

def _first_dicom(folder):
    # Tolerant by design: a timepoint folder may hold non-DICOM siblings; we scan
    # for the first readable DICOM header. Returning None (no DICOM) is handled by
    # the caller, so this is boundary tolerance, not a swallowed error.
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        try:
            return pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
    return None

def resolve_timepoints(dicom_dir, min_timepoints=5, require_complete=False):
    """Return per-timepoint subfolder paths, deduped by SOPInstanceUID and ordered by
    AcquisitionTime. Raises if fewer than min_timepoints when require_complete."""
    subdirs = [d for d in sorted(glob.glob(os.path.join(dicom_dir, "*"))) if os.path.isdir(d)]
    seen, items = set(), []
    for d in subdirs:
        ds = _first_dicom(d)
        if ds is None:
            continue
        sop = str(getattr(ds, "SOPInstanceUID", d))
        if sop in seen:
            continue
        seen.add(sop)
        items.append((str(getattr(ds, "AcquisitionTime", "")), d))
    items.sort(key=lambda x: x[0])
    paths = [d for _, d in items]
    if require_complete and len(paths) < min_timepoints:
        raise ValueError(f"{dicom_dir}: only {len(paths)} timepoints (<{min_timepoints})")
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_workingset.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the real-data run script**

```python
# src/cc_reservoir/scripts/run_extract_realdata.py
"""Run the method-comparison extraction over the working set (reads the SMB archive).
Run-and-inspect: emits /tmp/cc_realdata/curves.csv + per-acq QA MPR overlays.

Real-data wiring note: per-timepoint HU volumes come from each acq's MAT/*.mat files
(already HU). The MAT->timepoint mapping is session-specific; discover it by loading
each MAT/*.mat via io.volumes.load_hu_volume, keeping those co-shaped with the mask,
and ordering by the matching DICOM AcquisitionTime via io.workingset.resolve_timepoints.
If MAT mapping is ambiguous for a session, fall back to building each timepoint volume
from its DICOM/<tp> slices (read only the CC z-range using the mask bbox to stay fast).
"""
import os, csv, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..io import REAL_VOXEL_MM
from ..io.workingset import WORKING_SET, ARCHIVE, resolve_timepoints
from ..io.masks import load_cc_mask
from ..io.volumes import load_hu_volume, mask_bbox, crop_to_bbox
from ..measure.extract import extract_curves, method_agreement, METHODS

OUT = "/tmp/cc_realdata"

def _acq_dir(spec):
    return os.path.join(ARCHIVE, spec.session, spec.subpath)

def _load_series(acq_dir, mask):
    """Best-effort: load per-timepoint HU volumes from MAT/*.mat (co-shaped with mask).
    LOGS every file it skips (fail-loud, Rule 8) instead of silently dropping it."""
    mats = sorted(glob.glob(os.path.join(acq_dir, "MAT", "*.mat")))
    series = []
    for i, m in enumerate(mats):
        try:
            v = load_hu_volume(m)
        except Exception as e:
            print(f"    skip {os.path.basename(m)}: load failed ({e})")
            continue
        if v.shape != mask.shape:
            print(f"    skip {os.path.basename(m)}: shape {v.shape} != mask {mask.shape}")
            continue
        series.append((float(i), v))   # index as pseudo-time; refine with real AcquisitionTime
    return series

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for spec in WORKING_SET:
        acq = _acq_dir(spec)
        mask = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
        series = _load_series(acq, mask)
        if len(series) < 2:
            print(f"SKIP {spec.session}/{spec.subpath}: {len(series)} usable volumes")
            continue
        curves = extract_curves(series, mask, REAL_VOXEL_MM)
        agr = method_agreement(curves)
        rows.append(dict(session=spec.session, acq=spec.subpath, condition=spec.condition,
                         probe_flow=spec.probe_flow, n_tp=len(series),
                         mass_min_corr=agr["mass_min_corr"],
                         ring_vs_tp1_peak_ratio=agr["ring_vs_tp1_peak_ratio"]))
        # QA: mass curves under all methods
        plt.figure(figsize=(6, 4))
        for mth in METHODS:
            plt.plot(curves[mth]["t"], curves[mth]["mass"], marker="o", label=mth)
        plt.title(f"{spec.session}/{spec.subpath} ({spec.condition})  mass curves")
        plt.xlabel("timepoint index"); plt.ylabel("contrast mass (HU*mm^3)"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"mass_{spec.session}_{os.path.basename(spec.subpath)}.png"), dpi=120)
        plt.close()
    with open(os.path.join(OUT, "curves.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} acqs to {OUT}/curves.csv")
    for r in rows:
        flag = "" if (r["mass_min_corr"] > 0.9 and 0.7 < r["ring_vs_tp1_peak_ratio"] < 1.4) else "  <-- METHODS DISAGREE"
        print(f"  {r['session']}/{r['acq']:<16} corr={r['mass_min_corr']:.2f} "
              f"ring/tp1={r['ring_vs_tp1_peak_ratio']:.2f}{flag}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit the code**

```bash
git add src/cc_reservoir/io/workingset.py src/cc_reservoir/scripts/run_extract_realdata.py tests/test_workingset.py
git commit -m "feat: working set + real-data extraction run script"
```

- [ ] **Step 7: Run on real data and inspect (run-and-inspect, needs SMB)**

Run: `python3 -m cc_reservoir.scripts.run_extract_realdata`
Expected: writes `/tmp/cc_realdata/curves.csv` + per-acq mass-curve PNGs. **Inspect:** which acqs have all four methods agreeing (corr > 0.9, ring/tp1 ratio ≈ 1)? Acqs where the methods diverge are the motion/contaminated ones — record them. If the MAT→timepoint mapping fails for a session, implement the DICOM fallback described in the script docstring (read only the CC z-range via `mask_bbox` to stay fast over SMB), then re-run. Report the per-acq agreement table and flagged acquisitions; this is the deliverable.

- [ ] **Step 8: Commit any real-data wiring fixes**

```bash
git commit -am "fix: real-data timepoint wiring per inspection"
```

---

## Self-Review

**Spec coverage (spec §7 concentration leg, §9 Phase-1b, §11 hygiene):**
- IO single-source (`matio`/`masks`/`volumes`/`flow`) replacing the duplicated `/tmp` loaders — Tasks 1–3, 8. ✓
- Working-set hygiene (dedup by SOPInstanceUID, sort by AcquisitionTime, completeness gate) — Task 9. ✓
- Robust mass + concentration on real data — Tasks 4–7, 9. ✓
- Method comparison (2 ROI × 2 baseline) with agreement diagnostic flagging motion + contaminated baselines — Tasks 4–7, 9. ✓
- QA overlays — Task 9 (mass-curve figures; full MPR overlay reuses the prototype, expanded in 1b-C). ✓
- *Deferred:* shape-based V/c on real data (1b-B); the clinician MPR+3D viewer (1b-C). Noted, not dropped.

**Placeholder scan:** every code step has complete code; every test has assertions + expected result. The only non-unit step (9.7) is explicitly a run-and-inspect with a concrete command, expected outputs, and a fallback procedure — not a placeholder.

**Type/name consistency:** `load_mat_3d` → `load_cc_mask`/`load_hu_volume`; `mask_bbox`/`crop_to_bbox`; `dilated_mask_roi`/`threshold_roi`/`_ellipsoid_struct` (shared by `baseline.py`); `ring_baseline`/`tp1_baseline`; `contrast_mass`/`mean_enhancement`; `METHODS`/`extract_curves`/`method_agreement`; `AcqSpec`/`resolve_timepoints`/`WORKING_SET`. Consistent across tasks. `REAL_VOXEL_MM` defined once.

**Known real-data risk (documented, not a defect):** the MAT→timepoint mapping and the true elapsed-time vector are session-specific; Task 9 starts with a best-effort MAT loader + pseudo-time index and specifies the DICOM fallback (CC-bbox-cropped for speed). Pinning real elapsed times from `AcquisitionTime` and validating the curves against the tabulated `Lymph Results.xlsx` values is the first refinement once the pipeline runs.
