# The Cisterna Chyli as a Lymphatic Capacitor

**A partial-volume-aware measurement of the swine cisterna chyli, in service of testing
whether it is a compliant lymphatic reservoir.**

- Date: 2026-06-16 (revised same day after a 4-agent data audit — see Revision history)
- Status: design finalized, pre-plan
- Executable scope this cycle: **Phase 0 + Phase 1** (the PV-aware estimator foundation)
- Stack: **Python**

---

## 1. One-line summary

Build a partial-volume-aware, shape-prior estimator that recovers the cisterna chyli's
volume, concentration, and contrast mass over time from dynamic contrast-enhanced CT —
validated on synthetic truth *before* it touches real data — so that the reservoir
question (compliant capacitor vs. passive widening) rests on solid measurement, and so
the same forward model later drives a 4D lymphatic phantom.

---

## 2. Background

The Molloi Lab quantified **thoracic-duct (TD)** lymphatic flow with dynamic
contrast-enhanced CT lymphangiography in 11 Yorkshire swine (Molloi et al., *Radiology*
2023; 309(3):e230959), validated against a Transonic US flow probe, with angiotensin-II to
elevate flow. Data: `/Volumes/Molloilab/ImageData/Lymph_Studies` (see `DATA_GUIDE_shu_nie.md`).

This project turns to the **cisterna chyli (CC)** — the dilated sac at the caudal end of
the TD, formed by the confluence of the lumbar and intestinal trunks. The CC is a genuinely
ambiguous object: present in only ~50–55% of subjects (Moazzam 2022), wildly variable in
morphology (tubular/fusiform/saccular/beaded…), near-water at baseline, with only scant
smooth muscle and rudimentary valves — i.e., mostly **passive and compliant**. Moazzam's
review ends by questioning whether the CC is a discrete functional entity at all.

The literature points to a **compliant reservoir**: Niggemann 2010 found CC cross-section
swings ≈4× (14.7→62.9 mm²) supine→standing, the signature of a passive elastic buffer
absorbing the mismatch between variable inflow (gut chyle, lower-limb lymph) and TD outflow.
The lab's own protocol diagram labels the CC as the input compartment feeding the TD.

---

## 3. Data reality (established by a 4-agent audit, 2026-06-16)

The original plan assumed a clean per-timepoint CC volume time series. The audit overturned
that. What is actually true:

| What the model needs | Status | Evidence |
|---|---|---|
| CC + TD concentration curves `c(t)` | ✅ tabulated & validated | `Origin Templates/Lymph Results.xlsx` per-session sheets (`Time, CC, TD`); agree <1 HU with DICOM-derived values; ~30+ acqs |
| Per-acquisition flow `Q` | ✅ tabulated | probe **and** CT scalar (the raw Biopac waveforms are gone, the scalars are not); ~2.0–3.6 mL/min, baseline vs angiotensin |
| Image noise | ✅ known | σ_HU ≈ 15 HU (muscle); CC CNR 41–68 at peak |
| **CC volume over time `V(t)`** | ❌ **absent** | CC segmented at **one timepoint only**; `cc_stack_02` empty in every acquisition |
| Absolute iodine concentration | ⚠️ biased low | ~96% of CC voxels are partial-volume → full-mask mean under-reads true conc by 21–45% |
| Clean pre-contrast baseline | ⚠️ partial | some same-session later acqs contaminated by residual contrast (screen tp1: ~40 HU clean vs ~330 contaminated) |
| Temporal sampling / completeness | ⚠️ coarse & patchy | mostly 6 pts over ~6 min; ~18% duplicate-timepoint folders; aborted sessions (06_16_22); scrambled folder order (09_07_22) |

The "611±347 HU (57% CV)" that first looked like noise is **temporal signal** (45→1127→342 HU);
within a single peak frame the spatial CV is ~16%.

**Working set (clean, usable):** `07_20_22` (Baseline + Angiotensin, same animal),
`8_31_22` (incl. Stress), `09_07_22`. **~11 acquisitions** have curves + flow + a CC mask
ready; more have curves + flow. **Not** `03_23_23` (a 2-timepoint design, not a curve).

---

## 4. The scientific claim, and the physics that constrains it

- **H₁ (reservoir):** the CC is a compliant capacitor (compliance `C` ≠ 0) that stores
  fluid volume when inflow transiently exceeds outflow.
- **H₀ (null):** a passive pass-through — rigid, negligible storage.

**The physics that reframes the whole project.** The signal we trust most (concentration)
measures **residence time** `τ = V/Q`. With `V ≈ 0.1 mL`, `Q ≈ 3 mL/min`, `τ ≈ 2–6 s` —
far below the 1-min sampling. So the CC equilibrates with its inflow within a breath; the
minute-scale curve is the *delivery* of contrast to the CC, not storage within it.

But the **reservoir claim is about compliance** `C = dV/dP` — an *orthogonal* property. A
balloon with high flow through it flushes dye fast (low residence) yet still inflates under
pressure (high compliance). **Therefore the reservoir question requires `V`, not just `c`** —
and `V` requires sub-voxel estimation of a ~5-voxel structure. Hence the foundation-first
design below.

---

## 5. Architecture (Single Source of Truth)

One **shared forward model** is the engine for everything:

```
            ┌──────── PURE FORWARD MODEL (single source of truth) ────────┐
            │  true CC: geometry G (volume V) + iodine concentration c     │
            │    → rasterize → ⊗ PSF (scanner blur) → resample to voxels   │
            │    → + Gaussian noise (σ≈15 HU)  →  simulated CT voxels       │
            └──────────────────────────────────────────────────────────────┘
              │ used to VALIDATE        │ used to CALIBRATE      │ used to GENERATE
              ▼                         ▼                        ▼
      PV-AWARE ESTIMATOR          0-D RESERVOIR MODEL        4D PHANTOM (Phase 3)
      (inverse problem)           (Phase 2 science)          synthetic dynamic CT
      recover V(t),c(t),mass      distensibility/buffering    same forward model,
      + uncertainty               /residence + null test      run generatively
```

The estimator's validator, the reservoir science, and the phantom are all derived from the
*one* forward model (SSoT rules #1, #2). The forward model is pure and deterministic; I/O,
DICOM, fitting, and rendering live in shells (rule #6).

**Notation:** `V` volume, `Q` flow, `P` pressure, `R` resistance, **`C` compliance**
(uppercase constant), **`c(t)` concentration** (lowercase, ∝ HU). Subscripts `_CC`, `_TD`, `_in`.

---

## 6. The PV-aware CC estimator (the foundation — Phase 1)

The CC is 2–5 voxels wide, ~96% partial-volume. No binary mask can beat the scanner PSF, so
we **model** the structure rather than threshold it.

### 6.1 Prior knowledge of CC geometry (what makes the sub-voxel problem well-posed)
1. **Parametric shape model:** a low-dimensional swept tube (centerline + radius profile) or
   a deformable template from Moazzam's morphology types, **convolved with the PSF** and fit
   to the blurred voxels. A handful of parameters, well-constrained by ~150 voxels. (This is
   "fix geometry at the model level, not the voxel mask.")
2. **Size/location priors:** Moazzam/Plutecki diameter/length/volume ranges, T12–L2 /
   retrocrural position → Bayesian parameter priors.
3. **TD-connectivity anchor:** the CC is the dilated caudal continuation of the better-
   segmented TD; anchor its centerline/entry to the TD segmentation we already have.
4. **Morphology library:** generate synthetic CCs spanning the morphology *types* for
   validation, so the estimator is tested against real anatomical variability.

### 6.2 Robustness ladder (be explicit about what's recoverable)
- **Contrast mass** `m(t) = Σ(HU−baseline)·v_voxel` is **PV-robust** (blur conserves the
  integral). Trust it.
- **`V` and `c` separately** are the hard part: `m = V·c`, and the PSF smears "bigger" and
  "brighter" into nearly the same image. Whether they separate at this resolution is an
  **empirical question answered on synthetic truth** (§6.3) before any real `V(t)` is trusted.

### 6.3 The synthetic-validation gate (non-negotiable, rule #11)
Generate synthetic CCs (morphology library, prescribed `V, c`) → forward model → run the
estimator → quantify recovery accuracy and the `V`-vs-`c` separability at the observed σ≈15 HU,
the 21–45% PV bias, 6-point sampling, and realistic volume uncertainty. **If `V(t)` is not
recoverable, we learn it here and fall back to mass + concentration — before betting the
science on it.**

### 6.4 Estimator output
Per acquisition, per timepoint: `V_CC(t)`, `c_CC(t)`, `m_CC(t)` **with posterior credible
intervals**, plus a stated separability verdict.

---

## 7. The reservoir model and metrics (Phase 2 science, designed now)

A 0-D compartmental network (lumbar+intestinal trunks → CC → TD → venous angle), fed by the
estimator's outputs, fit with Bayesian uncertainty. The reservoir question is answered on
**two legs**:

- **Concentration leg (data-ready):** indicator-dilution on `c_CC(t)` vs `c_TD(t)` — pooling,
  lead/lag, flow-dependent washout (baseline vs angiotensin). Measures residence (expected
  fast) and delivery.
- **Compliance leg (needs estimator `V`):** **cross-sectional distensibility** — `V_CC` at a
  consistent peak frame vs flow-load, **within the same animal** (07_20_22, 8_31_22 have
  baseline + load). Does the CC distend under elevated flow? This is the flow-analog of
  Niggemann's postural test, and the direct test of compliance.

Metrics: distensibility `C`, buffering (only if within-acq `V(t)` proves recoverable),
residence `τ` (report the bound — likely sampling-limited). **Pre-registered null test:** is
`C` distinguishable from a rigid duct? State the verdict plainly either way.

---

## 8. Clinician-facing visualization (threaded through every phase)

- **QA (Phase 1):** per-timepoint overlay — observed voxels vs fitted shape vs residual — to
  verify the estimator isn't hallucinating geometry.
- **Payoff (Phase 1b/2):** the CC filling with contrast over time, in 3D + MPR, on real data.
- **Production clinician viewer:** **MPR-primary (axial/sag/cor) with the segmentation
  overlaid on CT**, linked to a 3D surface, opacity/toggle, standard window/level; runs in a
  **browser** (no install); **DICOM-SEG export** to the clinician's PACS/3D Slicer; and an
  **uncertainty shell** (translucent band) so a ~5-voxel estimate never looks more precise
  than it is. Dev/QA uses napari/PyVista; clinicians use the web viewer / Slicer.

A self-contained interactive viewer and an MPR overlay have already been prototyped on real
data (CC at the base of the TD; the tp1 mask sits at the edge of peak enhancement, motivating
per-timepoint estimation).

---

## 9. Phasing

- **Phase 0 — forward model + synthetic harness.** Pure CC-geometry→PSF→noise→voxels model;
  morphology-library generator. *(this cycle)*
- **Phase 1a — estimator + validation gate.** Build the PV-aware estimator; run §6.3; report
  achievable `V`/`c`/mass accuracy and separability. *(this cycle)*
- **Phase 1b — apply to the working set.** ~11 clean acqs → `V_CC(t), c_CC(t), m_CC(t)` with
  uncertainty; QA renders. *(this cycle)*
- **Phase 2 — reservoir science.** 0-D model + the two legs + null test → the three metrics
  with credible intervals. *(next cycle, gated on Phase 1)*
- **Phase 3 — 4D phantom.** Forward model run generatively on the XCAT lymphatic substrate.

---

## 10. Stack and repo layout

Python, pure core isolated from I/O.

```
cisterna-chyli-reservoir/
  pyproject.toml, README.md
  docs/superpowers/specs/2026-06-16-cisterna-chyli-reservoir-design.md
  src/cc_reservoir/
    forward/       # PURE: CC geometry, PSF, noise → simulated voxels (the SSoT engine)
    estimator/     # PV-aware inverse problem + shape priors + synthetic-validation gate
    io/            # DICOM (pydicom/SimpleITK), .mat masks/volumes (scipy/h5py), xlsx flow (openpyxl)
    reservoir/     # 0-D compartmental model + metrics + null test (Phase 2)
    viz/           # MPR overlays, 3D render, interactive viewer, DICOM-SEG export
  tests/           # invariants: PSF mass-conservation, synthetic param recovery, mask dedup/ordering
  notebooks/  data/ (gitignored; pointers to SMB, never copies)
```
Libs: `numpy, scipy, scikit-image, diffrax/jax, numpyro, pydicom, SimpleITK, h5py, openpyxl,
pandas, matplotlib, pyvista/vtk`.

---

## 11. Risks and open questions

1. **`V`-vs-`c` separability** on a 5-voxel structure — the central risk; resolved by the
   §6.3 synthetic gate before trusting `V(t)`.
2. **Residence is sampling-limited** (`τ`≈2–6 s vs 1-min sampling) → concentration leg
   reports bounds; compliance leg carries the reservoir claim.
3. **Data hygiene:** dedup timepoint folders by SOPInstanceUID; order by `AcquisitionTime`;
   hard-exclude aborted sessions (assert ≥5 non-empty timepoints); screen tp1 for baseline
   contamination; glob folder names (inconsistent zero-padding).
4. **No pressure** → compliance is pinned via the flow step (within-animal baseline vs
   angiotensin), not direct ΔV/ΔP.
5. **Anchoring concentration:** the TD is also only a few voxels wide, so it is *not* a clean
   absolute-concentration reference; the PSF-aware shape model is the primary degeneracy-breaker.

---

## 12. Definition of done (this cycle: Phase 0 + 1a, executed 2026-06-16)

- [x] Pure forward model (geometry→PSF→noise→voxels) with a test proving **PSF mass
      conservation** and a morphology-library generator.
- [x] PV-aware estimator (size-scale + concentration) with a Laplace covariance.
      *Full shape/TD-connectivity priors and swept-tube geometry deferred to Phase 1b.*
- [x] **Synthetic-validation gate executed**: per-morphology `V`/`c`/mass recovery + the
      `V`-vs-`c` separability verdict at 15 HU noise / DEFAULT_GRID resolution.
- [x] A plain-language Phase-2 statement (printed by the gate; recorded below).
- [ ] Estimator applied to the ~11-acq working set with real-data IO + QA overlays —
      **Phase 1b (next plan)**; this cycle was synthetic-only by design (de-risk before SMB).

### Phase 1 outcome — the verdict

Gate at 15 HU noise, DEFAULT_GRID, *true base shape known* (an optimistic bound):

| morphology | median \|V_err\| | median \|c_err\| | verdict |
|---|---|---|---|
| saccular / fusiform / tubular (in-plane semi-axis ≳ 1.3 mm) | 0.3–1.7% | 0.1–0.4% | **V_c_separable** |
| **thin** (semi-axis 0.6–1.3 mm — the CC-relevant regime) | **49%** | **24%** | **mass_only** |

**Consequence for Phase 2:** report `V(t)` and `c(t)` separately ONLY for rounder CCs; for
thin CCs (the common case) report **contrast mass `m(t)`**, which stays robust (~12% error)
where `V`/`c` individually do not. The Laplace separability `rho_sc` is **not** a sufficient
diagnostic — it stays ≈0.5 even on the degenerate thin family because the fit lands in a
*wrong* mass-preserving basin, not a flat one; only the empirical recovery error detects it.
And this is the *optimistic* bound (shape known): real shape uncertainty + small-`s` aliasing
(both Phase 1b) can only widen the `mass_only` regime.

---

## 13. References

- Molloi S, et al. Dynamic Contrast-enhanced CT Lymphangiography to Quantify Thoracic Duct
  Lymphatic Flow. *Radiology* 2023; 309(3):e230959.
- Moazzam S, et al. The cisterna chyli: a systematic review… *Am J Physiol Heart Circ Physiol*
  2022; 323:H1010.
- Plutecki D, et al. Anatomy of the Thoracic Duct and Cisterna Chyli: a meta-analysis.
  *J Clin Med* 2024; 13:4285.
- Niggemann P, et al. Postural Effect on the Size of the Cisterna Chyli. *Lymphat Res Biol*
  2010; 8(4):193.
- Fedrigo R, et al. 4D-XCAT lymphatic phantom (2022) — Phase-3 geometry substrate.
- Local: `/Volumes/Molloilab/ImageData/Lymph_Studies/DATA_GUIDE_shu_nie.md`.

---

## Revision history

- **2026-06-16 (initial):** 0-D reservoir fit to assumed `V(t)` + `c(t)`.
- **2026-06-16 (revised):** A 4-agent data audit found there is **no `V(t)`** (CC segmented at
  one timepoint), but **clean tabulated `c(t)` + flow**, severe partial volume, and a patchy
  time series. Reframed around the residence-vs-compliance distinction and a **foundation-first
  PV-aware estimator validated on synthetic truth**, with a shared forward model that also
  powers the future phantom and a clinician-facing MPR+3D visualization plan.
