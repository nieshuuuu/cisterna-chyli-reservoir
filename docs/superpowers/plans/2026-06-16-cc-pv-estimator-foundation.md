# CC PV-Aware Estimator Foundation — Implementation Plan (Phase 0 + 1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a partial-volume-aware forward+inverse model for the cisterna chyli and a synthetic-validation gate that quantifies whether volume `V` and concentration `c` are separable from a blurred, noisy, ~5-voxel CT object.

**Architecture:** One pure forward model (ellipsoid concentration field → Gaussian PSF blur → block-mean downsample to voxels → optional noise) is the single source of truth. The estimator inverts it (MAP fit of size-scale `s` and concentration `c` via least-squares, with a Laplace covariance for separability). A gate runs the forward→inverse loop over a morphology library at realistic noise/resolution and reports recovery accuracy + the `s`–`c` correlation. Everything is synthetic — no real data, no SMB.

**Tech Stack:** Python, NumPy, SciPy (`ndimage.gaussian_filter`, `optimize.least_squares`), pytest.

**Why an ellipsoid (YAGNI):** the feasibility question is about the *resolution/partial-volume/noise regime*, not shape fidelity. A thin ellipsoid (≈ the CC's ~2.5 mm × 19 mm aspect) exhibits the same `V`-vs-`c` degeneracy as a swept tube, with far simpler, testable geometry. Swept-tube refinement is deferred to a later plan.

---

## File Structure

```
src/cc_reservoir/
  __init__.py
  forward/
    __init__.py
    psf.py          # apply_psf(field, sigma_vox)
    geometry.py     # Grid, coarse_centers, fine_centers, fine_grid, render_ellipsoid, ellipsoid_volume
    imaging.py      # block_mean, forward_patch  (THE forward model: params -> voxel patch)
    morphology.py   # MORPHOLOGIES, sample_cc(kind, rng) -> (semi_axes_mm, conc_hu)
  estimator/
    __init__.py
    fit.py          # fit_patch(observed, base_semi_axes, grid, init) -> {s, conc, cov, success}
    recovery.py     # vcmass(s, conc, base_semi_axes) -> (V, c, mass); separability(cov)
  validate/
    __init__.py
    gate.py         # run_gate(...) -> rows; summarize(rows) -> verdict; __main__ prints it
tests/
  test_psf.py  test_geometry.py  test_imaging.py  test_morphology.py
  test_fit.py  test_recovery.py  test_gate.py
```

**Shared types/contracts (defined in Task 2, used everywhere):**
- `Grid = namedtuple("Grid", "n_xy n_z vox f psf_fwhm_mm")`; `DEFAULT_GRID = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)`
- `fine_vox(grid) -> tuple[float,float,float]` = `(vx/f, vy/f, vz/f)`
- A "patch" is a 3-D `np.ndarray` of shape `(n_xy, n_xy, n_z)` in HU.
- `semi_axes` are millimetre semi-axes `(a, b, c)`; `s` is a dimensionless size scale applied to all three.
- `conc` is concentration-as-enhancement in HU.

---

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/cc_reservoir/__init__.py`, `src/cc_reservoir/forward/__init__.py`, `src/cc_reservoir/estimator/__init__.py`, `src/cc_reservoir/validate/__init__.py`, `tests/__init__.py`, `pytest.ini`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "cc-reservoir"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "scipy>=1.11"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create empty package files**

Create these as empty files: `src/cc_reservoir/__init__.py`, `src/cc_reservoir/forward/__init__.py`, `src/cc_reservoir/estimator/__init__.py`, `src/cc_reservoir/validate/__init__.py`, `tests/__init__.py`.

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 4: Install editable + verify pytest runs**

Run: `python3 -m pip install -e ".[dev]" && python3 -m pytest`
Expected: pytest runs and reports "no tests ran" (exit 0 or 5).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pytest.ini src tests
git commit -m "chore: scaffold cc-reservoir package"
```

---

### Task 1: PSF blur (mass-conserving)

**Files:**
- Create: `src/cc_reservoir/forward/psf.py`
- Test: `tests/test_psf.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_psf.py
import numpy as np
from cc_reservoir.forward.psf import apply_psf

def test_psf_conserves_interior_mass():
    field = np.zeros((40, 40, 40))
    field[18:22, 18:22, 18:22] = 7.0          # centred blob, wide margin
    blurred = apply_psf(field, sigma_vox=2.0)
    assert np.isclose(field.sum(), blurred.sum(), rtol=1e-6)

def test_psf_spreads_signal():
    field = np.zeros((21, 21, 21)); field[10, 10, 10] = 1.0
    blurred = apply_psf(field, sigma_vox=1.5)
    assert blurred[10, 10, 10] < 1.0 and blurred[10, 11, 10] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_psf.py -v`
Expected: FAIL — `ModuleNotFoundError: cc_reservoir.forward.psf`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/forward/psf.py
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_psf(field, sigma_vox):
    """Blur a concentration field by a Gaussian PSF.

    sigma_vox: scalar or per-axis sequence, in (fine) voxel units.
    Background is 0 (post-baseline), so mode='constant' is physically correct;
    interior mass is conserved when signal does not reach the array border.
    """
    return gaussian_filter(np.asarray(field, float), sigma=sigma_vox,
                           mode="constant", cval=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_psf.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/forward/psf.py tests/test_psf.py
git commit -m "feat: mass-conserving Gaussian PSF"
```

---

### Task 2: Geometry — grids and ellipsoid rendering

**Files:**
- Create: `src/cc_reservoir/forward/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry.py
import numpy as np
from cc_reservoir.forward.geometry import (
    Grid, DEFAULT_GRID, fine_vox, coarse_centers, fine_centers, fine_grid,
    render_ellipsoid, ellipsoid_volume,
)

def test_fine_coarse_alignment():
    n, v, f = 5, 0.78, 2
    cc = coarse_centers(n, v)
    fc = fine_centers(n, v, f)
    assert np.allclose(fc.reshape(n, f).mean(axis=1), cc)

def test_ellipsoid_volume_matches_render():
    semi = (3.0, 3.0, 3.0)            # a sphere, r = 3 mm
    fv = 0.2
    ax = (np.arange(80) - 39.5) * fv  # centred grid, ±8 mm
    field = render_ellipsoid((0, 0, 0), semi, 1.0, ax, ax, ax)
    numeric = (field > 0).sum() * fv ** 3
    assert np.isclose(numeric, ellipsoid_volume(semi), rtol=0.05)

def test_default_grid_shapes():
    xs, ys, zs = fine_grid(DEFAULT_GRID)
    assert len(xs) == DEFAULT_GRID.n_xy * DEFAULT_GRID.f
    assert len(zs) == DEFAULT_GRID.n_z * DEFAULT_GRID.f
    assert np.allclose(fine_vox(DEFAULT_GRID), (0.39, 0.39, 0.5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_geometry.py -v`
Expected: FAIL — module/attributes missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/forward/geometry.py
from collections import namedtuple
import numpy as np

Grid = namedtuple("Grid", "n_xy n_z vox f psf_fwhm_mm")
DEFAULT_GRID = Grid(n_xy=20, n_z=44, vox=(0.78, 0.78, 1.0), f=2, psf_fwhm_mm=1.0)

def fine_vox(grid):
    return tuple(v / grid.f for v in grid.vox)

def coarse_centers(n, v):
    """n voxel-centre coordinates (mm), symmetric about 0."""
    return (np.arange(n) - (n - 1) / 2.0) * v

def fine_centers(n, v, f):
    """n*f fine centres tiling the same extent, block-aligned to coarse_centers."""
    cc = coarse_centers(n, v)
    sub = (np.arange(f) - (f - 1) / 2.0) * (v / f)
    return (cc[:, None] + sub[None, :]).ravel()

def fine_grid(grid):
    vx, vy, vz = grid.vox
    xs = fine_centers(grid.n_xy, vx, grid.f)
    ys = fine_centers(grid.n_xy, vy, grid.f)
    zs = fine_centers(grid.n_z, vz, grid.f)
    return xs, ys, zs

def render_ellipsoid(center, semi, conc, xs, ys, zs):
    """Concentration field: `conc` inside the ellipsoid, 0 outside."""
    cx, cy, cz = center
    a, b, c = semi
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    q = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 + ((Z - cz) / c) ** 2
    return np.where(q <= 1.0, float(conc), 0.0)

def ellipsoid_volume(semi):
    a, b, c = semi
    return 4.0 / 3.0 * np.pi * a * b * c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_geometry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/forward/geometry.py tests/test_geometry.py
git commit -m "feat: grids + ellipsoid rendering with analytic volume"
```

---

### Task 3: Imaging — block-mean downsample + the forward model

**Files:**
- Create: `src/cc_reservoir/forward/imaging.py`
- Test: `tests/test_imaging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imaging.py
import numpy as np
from cc_reservoir.forward.geometry import Grid, fine_vox, ellipsoid_volume
from cc_reservoir.forward.imaging import block_mean, forward_patch

def test_block_mean_shape_and_mean():
    a = np.ones((4, 4, 4))
    out = block_mean(a, 2)
    assert out.shape == (2, 2, 2) and np.allclose(out, 1.0)

def test_forward_patch_conserves_mass():
    # mass = sum(patch) * coarse_voxel_volume  ~=  V_ellipsoid * conc
    grid = Grid(n_xy=20, n_z=44, vox=(0.78, 0.78, 1.0), f=2, psf_fwhm_mm=1.0)
    semi = (1.25, 1.25, 9.5); conc = 800.0
    patch = forward_patch(s=1.0, conc=conc, base_semi_axes=semi, grid=grid)
    cvx, cvy, cvz = grid.vox
    mass = patch.sum() * cvx * cvy * cvz
    expected = ellipsoid_volume(semi) * conc
    assert np.isclose(mass, expected, rtol=0.08)

def test_forward_patch_noise_changes_output():
    grid = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)
    semi = (1.25, 1.25, 9.5)
    rng = np.random.default_rng(0)
    clean = forward_patch(1.0, 800.0, semi, grid)
    noisy = forward_patch(1.0, 800.0, semi, grid, noise_hu=15.0, rng=rng)
    assert not np.allclose(clean, noisy)
    assert 10 < np.std(noisy - clean) < 20   # ~15 HU
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_imaging.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/forward/imaging.py
import numpy as np
from .psf import apply_psf
from .geometry import fine_grid, fine_vox, render_ellipsoid

def block_mean(a, f):
    """Average non-overlapping f x f x f blocks (the detector averaging step)."""
    nx, ny, nz = a.shape
    a = a[: nx // f * f, : ny // f * f, : nz // f * f]
    return a.reshape(nx // f, f, ny // f, f, nz // f, f).mean(axis=(1, 3, 5))

def forward_patch(s, conc, base_semi_axes, grid, noise_hu=0.0, rng=None):
    """THE forward model: (size-scale s, concentration conc, base shape) -> voxel patch (HU).

    Renders an ellipsoid at fine resolution, blurs by the PSF, block-mean
    downsamples to the coarse voxel grid, and optionally adds Gaussian noise.
    """
    xs, ys, zs = fine_grid(grid)
    semi = np.asarray(base_semi_axes, float) * float(s)
    field = render_ellipsoid((0.0, 0.0, 0.0), semi, conc, xs, ys, zs)
    fvx, fvy, fvz = fine_vox(grid)
    sig_mm = grid.psf_fwhm_mm / 2.355
    sigma_vox = (sig_mm / fvx, sig_mm / fvy, sig_mm / fvz)
    coarse = block_mean(apply_psf(field, sigma_vox), grid.f)
    if noise_hu and rng is not None:
        coarse = coarse + rng.normal(0.0, noise_hu, coarse.shape)
    return coarse
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_imaging.py -v`
Expected: PASS (3 passed). If `test_forward_patch_conserves_mass` is slightly outside `rtol=0.08`, the cause is PSF leakage at the z-border — widen `grid.n_z` to 48 in the test and implementation default, do **not** loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/forward/imaging.py tests/test_imaging.py
git commit -m "feat: forward model (render -> PSF -> downsample -> noise)"
```

---

### Task 4: Morphology library

**Files:**
- Create: `src/cc_reservoir/forward/morphology.py`
- Test: `tests/test_morphology.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_morphology.py
import numpy as np
from cc_reservoir.forward.morphology import MORPHOLOGIES, sample_cc

def test_all_kinds_return_valid_shapes():
    rng = np.random.default_rng(1)
    for kind in MORPHOLOGIES:
        semi, conc = sample_cc(kind, rng)
        assert len(semi) == 3 and all(x > 0 for x in semi)
        assert 200.0 <= conc <= 1400.0          # plausible CC enhancement (HU)
        assert 0.6 <= semi[0] <= 4.0            # thin in-plane semi-axis (mm)
        assert 5.0 <= semi[2] <= 22.0           # elongated long semi-axis (mm)

def test_kinds_have_distinct_aspect():
    rng = np.random.default_rng(2)
    aspects = {k: sample_cc(k, rng)[0][2] / sample_cc(k, rng)[0][0] for k in MORPHOLOGIES}
    assert max(aspects.values()) > 2 * min(aspects.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_morphology.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/forward/morphology.py
import numpy as np

# Aspect families spanning the Moazzam morphology range (swine-scaled).
MORPHOLOGIES = ("saccular", "fusiform", "tubular", "thin")

_ASPECT = {            # (in-plane semi-axis range mm, long semi-axis range mm)
    "saccular": ((2.2, 4.0), (5.0, 9.0)),     # rounder, shorter
    "fusiform": ((1.6, 3.0), (8.0, 14.0)),
    "tubular":  ((1.0, 2.0), (10.0, 18.0)),
    "thin":     ((0.6, 1.3), (12.0, 22.0)),   # worst partial volume
}

def sample_cc(kind, rng):
    """Return (semi_axes_mm=(a, a, c), conc_hu) for a morphology family."""
    (a_lo, a_hi), (c_lo, c_hi) = _ASPECT[kind]
    a = float(rng.uniform(a_lo, a_hi))
    c = float(rng.uniform(c_lo, c_hi))
    conc = float(rng.uniform(400.0, 1100.0))
    return (a, a, c), conc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_morphology.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/forward/morphology.py tests/test_morphology.py
git commit -m "feat: CC morphology library (aspect families)"
```

---

### Task 5: Estimator — MAP fit of (size-scale, concentration)

**Files:**
- Create: `src/cc_reservoir/estimator/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit.py
import numpy as np
from cc_reservoir.forward.geometry import Grid
from cc_reservoir.forward.imaging import forward_patch
from cc_reservoir.estimator.fit import fit_patch

GRID = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)

def test_recovers_params_noiseless():
    semi = (1.25, 1.25, 9.5)
    obs = forward_patch(s=1.3, conc=700.0, base_semi_axes=semi, grid=GRID)  # no noise
    out = fit_patch(obs, base_semi_axes=semi, grid=GRID, init=(1.0, 500.0))
    assert out["success"]
    assert np.isclose(out["s"], 1.3, rtol=0.02)
    assert np.isclose(out["conc"], 700.0, rtol=0.02)

def test_cov_is_2x2():
    semi = (1.25, 1.25, 9.5)
    obs = forward_patch(1.0, 700.0, semi, GRID)
    out = fit_patch(obs, base_semi_axes=semi, grid=GRID, init=(1.1, 600.0))
    assert out["cov"].shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fit.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/estimator/fit.py
import numpy as np
from scipy.optimize import least_squares
from ..forward.imaging import forward_patch

def fit_patch(observed, base_semi_axes, grid, init=(1.0, 500.0)):
    """MAP fit of (size-scale s, concentration conc) to an observed voxel patch,
    using the same forward model (SSoT). Laplace covariance from the Jacobian."""
    observed = np.asarray(observed, float)

    def resid(theta):
        s, conc = theta
        pred = forward_patch(max(s, 1e-3), conc, base_semi_axes, grid)
        return (pred - observed).ravel()

    res = least_squares(resid, x0=list(init), method="lm")
    n, p = res.fun.size, 2
    dof = max(n - p, 1)
    sigma2 = 2.0 * res.cost / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * sigma2
    except np.linalg.LinAlgError:
        # Singular J^T J means s and conc are perfectly degenerate here: NaN is the
        # honest signal (it surfaces via separability -> nanmedian), not a swallowed error.
        cov = np.full((p, p), np.nan)
    return {"s": float(res.x[0]), "conc": float(res.x[1]),
            "cov": cov, "success": bool(res.success)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_fit.py -v`
Expected: PASS (2 passed). The noiseless fit should be near-exact; if `success` is False, increase `least_squares` `max_nfev` to 200.

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/estimator/fit.py tests/test_fit.py
git commit -m "feat: MAP fit of size-scale and concentration"
```

---

### Task 6: Recovery metrics + separability

**Files:**
- Create: `src/cc_reservoir/estimator/recovery.py`
- Test: `tests/test_recovery.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recovery.py
import numpy as np
from cc_reservoir.forward.geometry import Grid, ellipsoid_volume
from cc_reservoir.forward.imaging import forward_patch
from cc_reservoir.estimator.fit import fit_patch
from cc_reservoir.estimator.recovery import vcmass, separability

GRID = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)

def test_vcmass_matches_analytic():
    semi = (1.25, 1.25, 9.5)
    V, c, m = vcmass(s=2.0, conc=600.0, base_semi_axes=semi)
    assert np.isclose(V, ellipsoid_volume((2.5, 2.5, 19.0)))
    assert np.isclose(m, V * 600.0)

def test_noisy_recovery_within_tolerance():
    semi = (1.25, 1.25, 9.5); conc = 700.0
    rng = np.random.default_rng(3)
    Vt, _, Mt = vcmass(1.0, conc, semi)
    Verr, Merr = [], []
    for _ in range(8):
        obs = forward_patch(1.0, conc, semi, GRID, noise_hu=15.0, rng=rng)
        out = fit_patch(obs, semi, GRID, init=(1.2, 560.0))
        Ve, _, Me = vcmass(out["s"], out["conc"], semi)
        Verr.append(abs(Ve - Vt) / Vt); Merr.append(abs(Me - Mt) / Mt)
    # mass is more robust to partial volume than volume (the plan's core claim)
    assert np.median(Merr) <= np.median(Verr) + 0.05

def test_separability_range():
    cov = np.array([[1.0, -0.95], [-0.95, 1.0]])
    assert np.isclose(separability(cov), -0.95)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_recovery.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/estimator/recovery.py
import numpy as np
from ..forward.geometry import ellipsoid_volume

def vcmass(s, conc, base_semi_axes):
    """Return (volume mm^3, concentration HU, contrast mass HU*mm^3)."""
    semi = tuple(x * float(s) for x in base_semi_axes)
    V = ellipsoid_volume(semi)
    return V, float(conc), V * float(conc)

def separability(cov):
    """Correlation between the s and conc estimates. Near +/-1 => degenerate
    (size and concentration trade off; V and c not individually identifiable)."""
    sd = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sd)) or np.any(sd == 0):
        return float("nan")
    return float(cov[0, 1] / (sd[0] * sd[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_recovery.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cc_reservoir/estimator/recovery.py tests/test_recovery.py
git commit -m "feat: V/c/mass recovery + separability metric"
```

---

### Task 7: The synthetic-validation gate

**Files:**
- Create: `src/cc_reservoir/validate/gate.py`
- Test: `tests/test_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
import numpy as np
from cc_reservoir.validate.gate import run_gate, summarize

def test_gate_runs_and_reports():
    rows = run_gate(n_per_kind=2, noise_hu=15.0, seed=0)
    assert len(rows) == 2 * 4
    for r in rows:
        assert set(r) >= {"kind", "conc", "V_err", "c_err", "m_err", "rho_sc"}
    rep = summarize(rows)
    assert {"median_abs_V_err", "median_abs_c_err", "median_abs_m_err",
            "median_abs_rho", "verdict"} <= set(rep)
    assert rep["verdict"] in {"V_c_separable", "mass_only"}

def test_mass_more_robust_than_volume_on_average():
    rows = run_gate(n_per_kind=6, noise_hu=15.0, seed=1)
    rep = summarize(rows)
    assert rep["median_abs_m_err"] <= rep["median_abs_V_err"] + 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gate.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cc_reservoir/validate/gate.py
import numpy as np
from ..forward.geometry import DEFAULT_GRID
from ..forward.imaging import forward_patch
from ..forward.morphology import MORPHOLOGIES, sample_cc
from ..estimator.fit import fit_patch
from ..estimator.recovery import vcmass, separability

def run_gate(n_per_kind=15, noise_hu=15.0, grid=DEFAULT_GRID, seed=0):
    """Forward->inverse over the morphology library at realistic noise.
    s_true = 1; the fit knows the base shape and recovers (s, conc)."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in MORPHOLOGIES:
        for _ in range(n_per_kind):
            semi, conc = sample_cc(kind, rng)
            obs = forward_patch(1.0, conc, semi, grid, noise_hu=noise_hu, rng=rng)
            out = fit_patch(obs, semi, grid, init=(1.2, conc * 0.8))
            Vt, _, Mt = vcmass(1.0, conc, semi)
            Ve, _, Me = vcmass(out["s"], out["conc"], semi)
            rows.append(dict(kind=kind, conc=conc,
                             V_err=(Ve - Vt) / Vt,
                             c_err=(out["conc"] - conc) / conc,
                             m_err=(Me - Mt) / Mt,
                             rho_sc=separability(out["cov"])))
    return rows

def summarize(rows):
    aV = np.median([abs(r["V_err"]) for r in rows])
    ac = np.median([abs(r["c_err"]) for r in rows])
    am = np.median([abs(r["m_err"]) for r in rows])
    rho = np.nanmedian([abs(r["rho_sc"]) for r in rows])
    separable = (aV < 0.25) and (ac < 0.25) and (rho < 0.9)
    return {"median_abs_V_err": float(aV), "median_abs_c_err": float(ac),
            "median_abs_m_err": float(am), "median_abs_rho": float(rho),
            "verdict": "V_c_separable" if separable else "mass_only"}

if __name__ == "__main__":
    rep = summarize(run_gate())
    print("=== CC PV-estimator synthetic-validation gate ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("\nVERDICT:", rep["verdict"],
          "\n  V_c_separable -> Phase 2 may report V(t) and c(t) separately."
          "\n  mass_only     -> report contrast mass m(t); V/c not identifiable at this resolution.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the gate and record the verdict**

Run: `python3 -m cc_reservoir.validate.gate`
Expected: prints the four medians and a `VERDICT`. **This number is the deliverable** — it decides whether Phase 2 can report `V(t)`/`c(t)` separately or must fall back to mass. Paste the output into the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/cc_reservoir/validate/gate.py tests/test_gate.py
git commit -m "feat: synthetic-validation gate + V-vs-c separability verdict"
```

---

## Self-Review

**Spec coverage (against §6, §12 of the design):**
- Forward model (geometry→PSF→noise→voxels): Tasks 1–3. ✓
- PSF mass-conservation test (DoD): Task 1. ✓
- Morphology-library generator (DoD): Task 4. ✓
- PV-aware estimator with size/shape parameters: Tasks 5–6. ✓
- Synthetic-validation gate + V-vs-c separability verdict (DoD): Task 7. ✓
- Plain-language separability statement (DoD): Task 7 Step 5 prints it. ✓
- *Deferred to Phase 1b (next plan):* real-data IO (`io/`), application to the ~11-acq working set, shape priors beyond "known base shape", swept-tube geometry, and clinician visualization. Noted, not silently dropped.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every test has assertions and an expected result. ✓

**Type/name consistency:** `Grid`/`DEFAULT_GRID`, `fine_vox`, `coarse_centers`/`fine_centers`/`fine_grid`, `render_ellipsoid`/`ellipsoid_volume`, `block_mean`/`forward_patch`, `sample_cc`/`MORPHOLOGIES`, `fit_patch` (returns `{s, conc, cov, success}`), `vcmass`/`separability`, `run_gate`/`summarize` are used identically across tasks. Patch shape `(n_xy, n_xy, n_z)` consistent. ✓

**Known approximation (documented, not a defect):** the gate gives the fit the *true* base shape and only estimates scale + concentration — an optimistic bound on separability. If the verdict is `mass_only` even here, the real (shape-uncertain) problem is at least as hard. A shape-mismatch stress test belongs in Phase 1b.
