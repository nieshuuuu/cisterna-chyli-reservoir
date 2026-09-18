# Cisterna Chyli transport model — project north star

**This is the modeling goal the whole project serves. Every measurement step exists to feed
it. Do not diverge from this.** (Set 2026-06-17 from Shu's derivation.)

## Anatomy / flow picture

Lymph carrying contrast flows **upward** (caudal → cranial):

```
        TD (thoracic duct, narrow, length L, volume V_L)
         ↑  Q2  = flow OUT of CC into TD
       [ CC ]   cisterna chyli — the wide reservoir sac, volume V_CC
         ↑  Q1  = flow INTO CC (from the lumbar/intestinal trunks below)
        inlet duct(s)
```

- **Q1(t)** — volumetric inflow into the CC [mL/s] (from the inlet duct below).
- **Q2(t)** — volumetric outflow from the CC into the TD [mL/s].
- **E** — the contrast indicator (enhancement / concentration); the unknown we solve for.
  Written **E(t, v)** in the TD because the bolus is transported along the duct (v = bolus
  velocity / position along L), so enhancement there is a function of both time and place.
- **V_CC** — CC volume. **V_L** — TD volume. **L** — TD length.

## The two mass-balance (indicator-dilution) equations

$$\int_0^T Q_1\, E \, dt = V_{CC}$$

$$\int_0^T Q_2\, E(t, v)\, dt = V_L$$

Contrast delivered into a compartment over [0, T] equals that compartment's (contrast)
volume. **We want to solve for E** (the transport/enhancement function), given everything
else measured from the animal data.

## What the data pipeline must produce (this is why we measure what we measure)

| Symbol | Source | Status |
|---|---|---|
| **V_CC(t)** | per-timepoint 3D CC segmentation (CC = the WIDE segment by diameter) | Phase-2b (rebuild) |
| **V_L, L** | per-timepoint 3D TD segmentation (TD = the NARROW segment by diameter) | Phase-2b (rebuild) |
| **radius/diameter(arc-length)** | centerline + distance transform — also DEFINES the CC/TD split | Phase-2b (rebuild) |
| **E(t)** (and E(t,v) along TD) | CT enhancement curve in CC and along the TD | from contrast dynamics |
| **Q1, Q2** | probe/CT flow (Q1 = inflow) and the CC→TD partition (Q2) | flow table + balance |

## Method commitments (from 2026-06-17 review of the segmentation)

1. **Fixed anatomical extent; per-frame volume & enhancement.** The duct does not change
   LENGTH frame to frame — segmenting per frame at a hard threshold made the extent appear to
   grow as contrast arrived, which is an artifact. Segment the lumen **once** (fixed extent),
   then measure the integrated-HU (PV-corrected) volume and mean enhancement E **inside that
   extent, per frame**. Respiration/filling variation belongs in V_integratedHU(t) and E(t),
   not in a spuriously growing mask. (This supersedes the earlier "per-timepoint
   re-segmentation" idea and the Phase-2 "median over frames" geometric volume.)
2. **CC vs TD is anchored ANATOMICALLY at the caudal end** (DICOM: higher MAT z = cranial).
   CC = the caudal sac up to `boundary_z` (cranial extent of the lab CC mask); TD = everything
   ascending cranially, including a cranial terminal ampulla. Diameter is reported per region
   but is NOT the splitter — in Acq16 the ampulla is wider than the CC, so "diameter(CC) >
   diameter(TD)" does not hold and cannot define the split.
   *Candidate objective rule to test against this anatomical split (literature, 2026-06-28 —
   see [`literature/cc_2026-06_new_papers_digest.md`](literature/cc_2026-06_new_papers_digest.md)):*
   **Loukas 2007 defines CC = the most inferior point where lumen diameter exceeds 200% (2×) of
   the mean TD diameter** — the first *objective* CC/TD criterion, and fixation-robust (a ratio).
   Honest caveat for our data: in Acq16 the caudal "CC" (d≈3.5 mm) is *narrower* than the cranial
   ampulla (d≈4.7 mm), so the 200%-rule would not flag the caudal sac as a CC at all — it would
   pick the ampulla. So the rule and the caudal-anchored split can disagree; compute both and
   report the disagreement rather than tuning one to match. (Second candidate: Moazzam's MRI
   "≥5 mm fluid collection" threshold.)
3. **Visualize CC and TD together.** They are one connected lumen; render the combined
   CC+TD contrast filling (4D), not the CC alone.
4. **Enhancement is TEMPORAL (DSA-style), not spatial.** Enhancement = frame HU − per-voxel
   static baseline (min over time); a voxel is lumen only if it FILLED with contrast. A
   spatial ring-baseline does NOT cancel a static bright structure sitting in the ROI — the
   spine (~420 HU, unchanging) survived the threshold and segmented as a false SECOND tube.
   Temporal subtraction cancels bone exactly; the absolute threshold (≥200 HU) then sits above
   the residual motion halo (~100–150 HU) and below the duct (dimmest cross-section ≈280 HU).

## Acq16 measurements (2026-06-17, fixed-extent + integrated-HU — `phase2c_per_timepoint`)

Segmentation: ONE fixed anatomical extent from the **peak-frame temporal enhancement**
(peak HU − per-voxel min over time; cancels bone/static), absolute threshold **≥200 HU**
(above the motion halo, below the duct), largest connected component. Caudal-anchored split at
the lab-CC cranial extent (z≈36; CC = caudal sac, TD = ascending duct + cranial ampulla).
Length is constant by construction: **CC L=29 mm (d=3.5 mm), TD L=126 mm (d=4.7 mm)**. Per
frame, the integrated-HU (Molloi PV-corrected) volume and mean temporal enhancement E are
measured INSIDE the fixed extent.

| tp | **V_CC** (mm³) | **V_TD** (mm³) | **E_CC** (HU) | **E_TD** (HU) |
|----|------|------|------|------|
| 0 | 116 | 673 | 11 | 19 |
| 1 | 144 | 655 | 49 | 28 |
| 2 | 190 | 976 | 161 | 124 |
| 3 | 190 | 1068 | 252 | 222 |
| 4 (peak) | 199 | 1257 | 308 | 317 |
| 5 | 196 | 1214 | 216 | 277 |

Findings that constrain the model:
- **CC fills first and plateaus** (~190 mm³ by tp2, the short caudal reservoir is fully
  opacified early); **TD integrated-HU volume keeps rising** 673→1257 mm³ as the bolus ascends
  and opacifies progressively more of the long duct. This is reservoir-then-conduit behavior.
- **E(t) is a clean indicator-dilution bolus**: rises to a peak at tp4 (E_CC 308, E_TD 317)
  then washes out at tp5. E_CC slightly LEADS E_TD early (tp1: 49 vs 28; tp2: 161 vs 124) —
  consistent with the caudal CC filling before the ascending TD, i.e. supporting the assumed
  inlet→CC→TD (caudal→cranial) direction. (The earlier "TD-first / cranial→caudal" reading was
  an artifact of spine contamination; it disappears with temporal subtraction.) Still confirm
  direction against the flow table.
- The **caudal CC is modest** (d≈3.5 mm), narrower than the cranial TD ampulla (d≈4.7 mm), so
  "diameter(CC) > diameter(TD)" does NOT hold in Acq16 — diameter cannot define the split here.
- Two earlier approaches are superseded: a fixed manual mask with a half-max threshold (both
  steps under-count a partial-volume-blurred duct) and the spatial-baseline "two tubes" (spine
  leaked in as a false second duct). The single-duct lumen is CC≈200 + TD≈1257 ≈ 1460 mm³ (PV-corrected) at peak.

Still needed to close the model: Q1/Q2 from the flow table + the dV/dt balance (E(t) is now in
hand, columns above; cross-check against the lab's `TDCs.mat`/`time_vector_autoGenerated.mat`).

## Literature volume anchors (2026-06-28 batch — for sanity-checking V_CC)

The project fills a real gap: **no porcine CC volume has ever been reported.** The closest
measured anchors (full provenance + corrections in
[`literature/cc_2026-06_new_papers_digest.md`](literature/cc_2026-06_new_papers_digest.md)):

| Anchor | Volume | Source |
|---|---|---|
| Human CC (contrast-CT, n=484) | **302 µL** (= 0.30 mL); +37% in >70 yr | Feuerlein, via Moazzam 2022 |
| Dog CC (CTLa + CAD, n=23) | **1.82 mL** (right 1.46 / left 0.49); 0.07 mL/kg | Carvajal 2020 |
| Dog CC **+ TD** fill (in vivo) | **2.8 mL** to mid-thorax (~0.094 mL/kg, ~30 kg) | Weisse 2015 |
| Our Acq16 V_CC plateau | **≈0.20 mL** (200 mm³); V_TD ≈ 1.26 mL | this project |

Reading: our caudal V_CC (0.20 mL) sits just *below* the human 302 µL and well below the dog
1.82 mL — but the pig CC is anatomically *longer* than the human (6–11 cm vs ~2–3 cm), so 0.20 mL
likely reflects the **short caudal-anchored split**, not a small organ. **V_CC is boundary-rule
dependent** — this is the same caliber/extent question as commitment #2, so report V_CC as a
*range over the candidate boundary rules*, not a single tuned number. Caveat (Carvajal,
verifier-confirmed): **contrast injection can artificially dilate/globularize the CC**, so a
distension signal must be separated from injection-induced passive widening.

## Outflow resistance for Q2 — a pressure-based forward model (Patel/Kassab 2025)

The mass-balance equations give Q2 only *indirectly* (Q2 = Q1 − dV_CC/dt, or via ∫Q2·E dt = V_L). A
swine-specific **forward** model now lets us also predict Q2 from pressure, as an independent
cross-check (full detail + verification: [`literature/cc_2026-06_new_papers_digest.md`](literature/cc_2026-06_new_papers_digest.md)
§8):

$$Q_2 = \frac{\Delta p}{R_{vessel} + R_{valve}}, \qquad R_{valve} \gg R_{vessel}\ (\text{1–2 orders})$$

- **R_vessel = 128 µL / (π D⁴)** — Poiseuille; D = TD inner diameter (from segmentation), L = TD
  length, µ ≈ lymph viscosity ~1 cP. We already produce D and L per frame, so R_vessel is computable.
- **R_valve** is sigmoidal in Δp across the valve and collapses to its **low floor R_vl ≈ 0.01–0.15
  cmH₂O/(mL/min)** once that gradient exceeds **~0.84 cmH₂O**. The in-vivo CC→jugular gradient is
  ~8 mmHg ≈ 11 cmH₂O (≫ 0.84), so **in normal forward flow the first TD valve is essentially fully
  open** — the CC→TD junction is *not* a high-resistance gate in the physiological regime (it gates
  only against reflux). This is the answer to "is there functional gating at the CC–TD transition?":
  the resistance lives in the valve, but the valve is open downstream of ~0.84 cmH₂O.
- **The valve, not the conduit, sets the outflow resistance** — so a Q2 model can largely ignore
  R_vessel and use R_valve(Δp).
- Units: Patel/Kassab use **cmH₂O/(mL/min)**; 1 mmHg ≈ 1.36 cmH₂O.

Use: with a CC/TD pressure (measured, or the ~15 mmHg CC operating point from Lu/Kassab 2020) this
predicts Q2 *independently* of the indicator-dilution Q2 — a closure check on the mass balance. It is
a **passive, static** relation (no respiration/active-pump term), so it sets the passive baseline,
not the full dynamic Q2. Open data + code on Zenodo (Patel/Kassab 2025).

## Open questions to resolve as the data comes in

- Units/normalization of E so that ∫Q·E dt has units of volume (E as volume-fraction of
  contrast vs HU concentration — fix the calibration).
- Whether Q2 is measured or inferred from V_CC mass balance (Q2 = Q1 − dV_CC/dt by
  conservation, if CC is the only branch) — now also predictable from the pressure-based forward
  model above (Q2 = Δp/(R_vessel + R_valve)); **reconcile the three routes** as the data lands.
- E(t,v) transport along the TD: advection speed v from the bolus front vs L.
