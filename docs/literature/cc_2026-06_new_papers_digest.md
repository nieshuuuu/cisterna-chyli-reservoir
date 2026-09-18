# New CC/TD papers — integrated digest (2026-06-28 batch)

*Scope: the 22 papers added to `paper/` on 2026-06-25→28 (11 Chinese journal articles + 11
English), read and integrated for the swine cisterna-chyli (CC) reservoir project. Each paper was
read by one agent and every numeric/translation claim independently re-checked against the PDF by a
second agent (44 agents total, all "high" confidence). **Numbers below are post-verification: the
mistranscriptions the checkers caught have been corrected here (see "Corrections applied").** This
digest complements `cc_td_boundary_literature.md` (boundary) and `cc_peri_adipose_literature.md`
(peri-CC fat); it does not repeat what those already cover.*

---

## Bottom line — the 6 things actually worth acting on

1. **The porcine CC-volume gap is now bracketed by real animal data.** Carvajal 2020 reports the
   only segmented CC volume in any animal: **median 1.82 mL in 23 chylothorax dogs** (right 1.46 /
   left 0.49 mL), **1.12–1.30 mL in 3 healthy cadavers**, **~0.04–0.07 mL/kg**. Weisse 2015 gives an
   in-vivo **CC+TD fill volume of 2.8 mL** (~30 kg dogs, ~0.094 mL/kg). The human anchor Shu had
   quoted from memory — **302 µL** — is now *sourced* (Feuerlein, n=484, via Moazzam 2022). Our
   measured ~240–865 µL spread sits squarely inside dog-vs-human.

2. **An *objective* CC/TD boundary criterion exists** and resolves the long-running "the boundary is
   subjective" problem with a testable rule: **Loukas 2007 — CC = the most inferior point where lumen
   diameter exceeds 200% (2×) of the mean TD diameter.** Fixation-robust (it is a ratio). A second
   candidate: **Moazzam — MRI fluid collection ≥5 mm**. Both are encodable in the estimator and worth
   testing against our current caudal-length split.

3. **The "compliant reservoir vs passive widening" hypothesis now has strong literature backing —
   and a measured pressure axis.** CC cross-section **+428% erect vs supine** and **+179% max
   diameter in CKD** (Moazzam); the duct "**dilates to several times normal width depending on lymph
   flow**" (Tubbs/Smith); CC wall histology = **loose fibromuscular meshwork, scant smooth muscle,
   elastic fibers, sclerosing under chronic load** (passive-distensible, strain-stiffening). And the
   P-axis: **swine CC operating pressure 15.3 ± 3.0 mmHg** (Lu/Kassab) — the absolute number the
   brief lacked. Caveat that *cuts the other way*: swine TD flow is pulsatile, **out of phase with
   venous pressure, and LACKS the intrinsic-contractility peaks seen in sheep** → the swine reservoir
   is **extrinsically (respiration/venous) driven, passively compliant** — though "not actively
   pumped" is too strong; intrinsic contractility is coupled in, just less prominent than in sheep
   (see §3, and §8 for the *passive* pressure–flow model that backs the passive baseline).

4. **Concrete time constants for the transport model (E(t), Q1, Q2).** Porcine DCMRL (Dori, 5 pigs):
   **TD first opacifies at 244 s (201–387 s), enhancement persists >60 min**. Intranodal small-animal
   peak **~5 min**, washout over **15–50 min**, the **5–10 min window** is the universal sweet spot.
   Dog CT-lymphangiography TD opacifies in **2–13 min**. And the validating premise: **IV contrast
   does NOT fill the CC within 5 min** (it is a lymphatic compartment) — only delayed >5–10 min.

5. **The "rope-ladder"/reticular CC forms in our segmentations are normal anatomy, not artifacts.**
   Human CC is **58% lobulated/saccular vs 42% fusiform** (Loukas); **~1/3 are plexiform networks**
   (Wu Bi 32.4%); documented forms include rope-of-pearls, parallel/duplicated tubes, up to **5
   sacculations** on one duct, a literal **triple-CC** case, a **reticular CC + right-sided TD**, an
   **intrathoracic CC in 6/8 dogs**, and a **non-dilated "CC" that is just a trunk confluence**. So a
   porcine "CC" may be a *plexus*, the volume estimator must handle multi-branch geometry, and
   "CC present" is inherently definition-dependent.

6. **A validated passive pressure–flow + valve-resistance model for the swine TD now exists — the Q2
   outflow element the transport model needs (§8).** Patel/Kassab 2025 (ex vivo, n=5 pigs): TD outflow
   is **Q₂ = Δp / (R_vessel + R_valve)** with **R_valve 1–2 orders of magnitude larger than R_vessel**
   (the valve controls the flow); the valve opens to near-max at a tiny **Δp ≈ 0.84 cmH₂O**, so in the
   physiological regime (~9–10 cmH₂O) it sits at its *low* resistance. **This is the paper my earlier
   web-built brief mislabeled as "valve spacing"** — its real content is exactly this pressure–flow
   modeling, with open data + code on Zenodo. It is explicitly the *passive* relation (active = future
   work), backing the passive baseline from §3.

---

## 1. CC volume — the gap-(a) headline (Carvajal, Weisse, Moazzam/Feuerlein)

**Carvajal 2020 (Vet Radiol Ultrasound; dog) — `Vet Radiology Ultrasound - 2020 - Carvajal ...`.**
The single most valuable new paper.
- **Median total CC volume = 1.82 mL (IQR 1.15–2.95)** in 23 dogs with idiopathic chylothorax,
  by CT-lymphangiography (CTLa) + CAD 3-D segmentation.
- **Strongly right-dominant: 1.46 mL right vs 0.49 mL left of the aorta (P = .014)**; dorsal+right in
  18/23 (78%). → a laterality prior for the PV-aware estimator.
- **CC:body-weight ≈ 0.07 mL/kg** (diseased, IQR 0.035–0.10); **0.04–0.05 mL/kg** in 3 healthy
  cadavers (total CC 1.25 / 1.12 / 1.30 mL).
- **Method validation: CT+CAD recovered 95–97% of a known 2.0 mL injected volume (3–5% error).** A
  concrete accuracy benchmark for *our* volume estimator.
- **Portable porcine ground-truth recipe:** right paracostal approach → TD clip-ligated in caudal
  thorax → 25-g puncture caudal to ligation → **retrograde inject 2.0 mL of 1:1 Lipiodol:NBCA glue
  → postmortem CT**. This is a physical-truth model we can replicate in pigs (the project wanted
  synthetic truth; here is a cadaver truth).
- **Operational CC definition used:** contrast structure **dorsal to aorta, cranial to the converging
  lumbar trunks, caudal to a tapered TD**; ventral mesenteric/hepatic plexus excluded. CC = dorsal
  saccular dilation (between celiac artery and left renal hilus) + a variable plexiform ventral part.
- Dimensions: length 150 mm (IQR 100–270), height 5.5 mm, max width 13.3 mm; **Ao:CC 0.57** (vs
  0.32–0.38 healthy → disease dilation). Pre-contrast CC **18.7 HU** → post-CTLa **median 987 HU**.
- Protocol: intranodal (popliteal) **Omnipaque 300, 1 mL/kg at 2 mL/min**, 0.5 mm slices, 120 kVp.
- **CENTRAL CAUTION (verify-confirmed):** authors warn **contrast injection may artificially dilate
  / globularize the CC** — exactly our standing worry that injection-induced passive widening can
  masquerade as compliance. They do *not* claim "first animal CC volume" (that framing was the
  reader's; corrected).

**Weisse 2015 (Vet Image-Guided Interventions; dog/cat) — glue embolization chapter.**
- Experimental dogs (mean **29.75 kg**): **mean 2.8 mL fills CC + TD up to mid-thoracic level**
  (~0.094 mL/kg). This is a *combined trunk* fill, not isolated CC → an order-of-magnitude **upper
  bound** on V_CC(t)+V_TD(t). Large inter-animal variability emphasized.
- Lymphangiogram/embolic injection **2–4 mL over 3–5 s** (~0.4–1.3 mL/s) — a flow-rate/capacity
  bracket for Q1.

**Moazzam 2022 (= the AJP-Heart human systematic review already in the brief; human).**
- **Human CC volume = 302 µL (Feuerlein, contrast-CT, n=484), +37% in >70 yr vs <40 yr.** This is
  the source for Shu's remembered "302 µL." Volume is age/state-dependent, not fixed.
- Weighted-average human CC by method: cadaver 13.8 × 24.3 mm; **CT 6.2 mm × 13.8 mm length**; MRI
  5.7 × 28.9 mm; living ~10 × 20 mm. CT diameters run *smaller* than cadaveric → a PV/threshold
  caution since we use CT.

**Porcine caliber corroboration (for converting a measured shape into a volume sanity-check):**
- **42155 2023 (pig, in vivo lipiodol/CT): CC diameter "relatively large (4–6 mm)."**
- **Lu/Kassab (`2297906`, pig cast): CC max 11.4–15 mm (mean 13.1 ± 2.5), CC min 2.0–3.0 mm; largest
  diameter is at the CC (opposite to humans).** Cast may over-distend → treat 11–15 mm as an upper
  bound, 4–6 mm (in-vivo) as the working width.

---

## 2. CC/TD boundary — an objective rule now exists (Loukas 2007, Tubbs 2016, Moazzam 2022)

- **Loukas 2007 (120 human cadavers): CC = the most inferior point at which lumen diameter > 200%
  of the mean TD diameter** (TD measured at T8, T4, termination). Authors state this is the *first
  objective CC definition*. **Fixation-robust** (a ratio; formalin shrinks CC and TD proportionally).
  This complements the porcine 2.5–7× step-down already in the brief, and is directly encodable as a
  caliber-ratio boundary in the estimator.
- **Moazzam alternative: MRI fluid collection ≥5 mm** with bile/CSF signal → CC present in 15%
  (vs Loukas's 200%-rule → 83.3%). Both confirm **absolute diameter does not discriminate**; the two
  rules are candidate objective boundaries to test against our current caudal-length split.
- **Position prior refined:** Moazzam — **midline is actually most common (70–76%)**, right 7–20%,
  left 14–17% (contrary to the classic right-of-midline default). Loukas — right of aorta 75%.
- **Chinese landmark anchors** (where the CC sits relative to arteries): Ji Rongming —
  **CC 47 ± 15 mm from celiac-trunk origin, 39 ± 14 mm from SMA origin**, mid-width measured at the
  CC midpoint. Wu Bi/Guan Qing — CC = the trunk-confluence point; TD = the merged ascending trunks,
  deviating left at T5. **TD-continuity remains the safest single discriminator** (echoed by every
  source).

---

## 3. Compliance / reservoir physiology — gap (b) partially filled

**Distension evidence (favours capacitor C = dV/dP):**
- **CC cross-sectional area +428% erect vs supine** (postural/gravity filling, Moazzam). Huge passive
  volume swing → the reservoir is highly position/load-sensitive; we must distinguish this from active
  compliance.
- **CC max diameter +179% in CKD vs controls**, correlating with severity; also enlarges in portal
  hypertension, cirrhosis, malignancy (Moazzam). (Correction: it does **not** enlarge in
  immune-mediated pathology — the reader had that backwards.)
- The duct **"dilates to several times its normal width depending on lymph flow"** (Tubbs/Smith 2013).
- TD distends with rising intrathoracic pressure / disease; CC larger in malignancy than benign
  (Lu/Kassab).

**Wall structure (favours passive, strain-stiffening compliance):**
- **CC wall (only human histology, Lenz & Huth, n=3): endothelium-lined loose fibromuscular meshwork,
  few nuclei, occasional elastic fibers, scant smooth-muscle remnants; sclerosed → dense collagen
  under chronic congestion.** A low-muscle distensible wall that stiffens with collagen = the same
  strain-stiffening picture as the canine TD (Deng 1999) already in the peri-adipose digest.
- **Swine TD wall ~106 ± 32 µm, media 7 ± 3 µm, only 1–2 SMC layers, lymphangion 2.6 cm** (Lu/Kassab)
  — thin, weakly muscular = distensible conduit.

**Pressure axis (new absolute numbers, Lu/Kassab `2297906`):**
- **CC mean pressure 15.3 ± 3.0 mmHg; lymphovenous junction 7.6 ± 3.7 mmHg** (gradient 8.1, CV 0.25).
  The brief had only the gradient — now we have the **~15 mmHg CC operating point** for the P-axis.

**Drive mechanism (state carefully — the strong "not a pump" version is NOT supported):**
- **Swine TD flow pulses at ≈ respiratory/venous frequency, out of phase with venous pressure, and
  LACKS the distinct intrinsic-contractility spectral peaks that SHEEP show** (TDQ–TDP r 0.858;
  TDQ–JVP r 0.586; max-amplitude offset ~0.15 Hz). **But Lu/Kassab's own wording (p.410):** venous/
  outlet pressure "plays an important role... **coupled with an intrinsic mechanism (lymphatic wall
  contractility)**," and "the smooth muscle serves as the pump." So the defensible reading is
  *extrinsic (respiration + venous-outlet) drive is more **prominent** in pigs than the intrinsic
  pump — unlike sheep — NOT that the pump is absent.* Physics behind it: an autonomous lymphangion
  pump would beat at its **own** frequency (a spectral peak unrelated to respiration); pig flow
  instead sits at respiratory/venous frequency and is phase-locked to the venous outlet → the
  pacemaker is external. Anatomy agrees: wall ~106 µm, **one** ~7 µm smooth-muscle layer, "fragile"
  duct = weak contractile machinery. **Caveats:** (1) pigs were anesthetized — anesthesia suppresses
  intrinsic lymphatic pumping, so extrinsic dominance is partly expected and may not hold awake;
  (2) n=6; (3) the imaging literature *restates* (does not newly observe) an intrinsic contractile
  tone, and the citation chain funnels to ONE old human source:
  **rhythmic CC contraction** — 吴苾 ("X线淋巴造影透视可观察到乳糜池有轻微节律性收缩", **ref [4]**)
  → that [4] **= Pinto 2004 (293457)** → which cites it to **its ref (1) = Rosenberger A, Abrams HL.
  "Radiology of the thoracic duct." Am J Roentgenol Radium Ther Nucl Med 1971;111:807–820** — the
  actual primary (human X-ray lymphography, 1971). And Pinto's *own* MR found the CC remarkably
  *stable* over months (i.e. their data contradicts the cited claim). **Segmental TD contraction** —
  吴苾 (own explanation for discontinuous TD) and 张晓杰 (cited to **Chen et al [14]**; exact citation
  not recoverable from the OCR text layer — read the ref-list page as an image to confirm). NB 张晓杰
  reports **TD-segmental only, NOT CC rhythmic.** So as a counterweight to Lu/Kassab's swine
  spectral finding this is weak: two old **human** sources (Rosenberger & Abrams 1971; Chen et al),
  neither swine, neither a fresh observation, one self-contradicted. The swine-specific anchor stays
  Lu/Kassab's own "coupled with an intrinsic mechanism."
- **Why this still helps the project:** a weakly-muscular, externally-driven duct favours reading
  V_CC changes as **passive compliance C = dV/dP** (an active pump would change volume by contracting,
  confounding the read), and predicts **respiration should modulate V_CC(t)/E(t)** — a testable
  signature in the dynamic CT.

**External mechanical pressure (touches the unfilled peri-CC-fat gap):**
- Weisse — outflow obstruction → "**lymphatic hypertension within the cisterna chyli**."
- Fransson — **abdominal pressure impedes lymph flow to the CC/TD** (sternal positioning with a
  pendulous abdomen is chosen to *avoid* compression); laparoscopic CC ablation done at **6 mmHg**
  insufflation. A physiologic external-pressure scale (6–12 mmHg) around the retroperitoneal CC.
- Kang Bo — outflow obstruction → mediastinal TD **dilates, tortuous, reticular** (pressure → form).

---

## 4. Dynamics & timing — time constants for E(t), Q1, Q2

| Quantity | Value | Source |
|---|---|---|
| **Porcine TD first opacification** | **244 s (201–387 s)** after intranodal start | Dori, 5 pigs DCMRL (cited in Pan Haipeng) |
| **Porcine TD enhancement duration** | **>60 min** | same |
| Dog CT-lymph TD opacification | **2–13 min** | Fransson |
| Small-animal (rabbit) intranodal peak | **~5 min**; decline 15–50 min; window **5–10 min** | Pan Haipeng (40 rabbits) |
| IV contrast → CC | **no enhancement <5 min**, delayed >5–10 min | 293457, Wu Bi |
| Intranodal lipiodol inflow rate | **0.1–0.5 mL/min** (Q1 bracket) | 42155 (pig) |
| Human duct throughput | ~100 mL/h (~1.7 mL/min) | 293457 |
| Swine TD flow / CC pressure | 0.7 ± 0.49 mL/min / 15.3 mmHg | Lu/Kassab |

**Method caveat that matters for our CE-CT (42155, pig):** *water-soluble iodinated contrast shunts
rapidly from lymphatics into venous capillaries and fails to opacify the central lymphatics* —
lipiodol (oily) is retained; **interstitial pedal lipiodol also failed to reach the CC** (only limbs).
The **intranodal/mesenteric** route is what reliably fills the CC. Our data clearly *did* opacify, so
the lab's route is right — but this is the physiological reason a single bolus of water-soluble CT
contrast can wash out fast, relevant to how we read E(t) wash-in/wash-out.

---

## 5. Variant catalog — validates the reticular / multi-channel CC

**Human imaging/cadaver morphology distributions (priors for classifying our segmentations):**
- Loukas: **58% lobulated/saccular, 42% fusiform** (no tube-only majority).
- Wu Bi (100 human MR): **single-tube 43.7%, bifurcation 23.9%, network/plexiform 32.4%**; mean
  length 4.5 cm, single-tube diam ~3.3 mm; plexiform CCs are the *longest* (5.2 cm).
- 293457 (30 human MR): thin tube 30%, focal round/oval 27%, focal plexus 17%, parallel/converging
  10%, sausage 7%, tortuous 7%, thick tube 3%.
- Yu Dexin (142 human MR): tubular/fusiform dominant + double-tube, inverted-V, reticular, bundle,
  dumbbell, bird-claw; one CC made of **3 lymph trunks**.
- Moazzam menu: rope-of-pearls/beaded, plexiform, parallel/duplicated tubes, multiple sacculations.
- Tubbs/Anson-McVay: **up to 5 sacculations** on one duct; twin (duplicated) CC; bilateral CC at L1.

**Striking case reports (the Chinese batch — direct analogues of our "rope-ladder" forms):**
- **He Shangkuan 1989 — a TRIPLE cisterna chyli** in one cadaver: upper pool 2.0 × 1.3 cm, lower
  2.3 × 0.7 cm, left 2.0 × 1.0 cm; double TD (left branch 0.25 × 4.5 cm, right 0.3 × 2.0 cm) merging
  2 cm above the upper pool. CC→TD caliber step-down ~13 mm pool → ~2.5–3 mm TD.
- **Zhou Jinge 2016 — reticular (net-like) CC + right-sided TD** draining to the *right* venous angle;
  terminal TD dilations 5.6–5.8 mm wide × 9.5–12 mm (a second reservoir-like widening at the TD end).
- **Guan Qing 2012 — a high (intrathoracic, anterior to T10) CC with NO obvious dilatation** — a real
  passive non-dilated trunk-confluence "CC." Tributary calibers ~1 mm.
- **Ji Rongming 2004 — discrete CC pool in only 22% (7/32) cadavers; 78% are an organized confluence
  with no pool**; plexiform ("丛状") TD origin in 4%. TD origin 2.8 ± 0.7 mm at T11–L2.

**Animal variants:**
- Carvajal — **intrathoracic CC in 6/8 VATS dogs** (not previously reported); TD branching 1–3,
  highly variable per dog.
- 42155 — **"small / not-well-developed CC" in some pigs** (unsuitable for trans-CC access) = porcine
  CC size genuinely varies between animals (so Acq-to-Acq CC-size differences are real biology).
- Fransson — CC is a **"bipartite saclike"** structure that can wrap around / lie *ventral* to the
  aorta; CC→TD can present as a "large dilated cisterna entering the chest" (boundary blur).

**Take for the estimator:** "CC present" is definition-dependent (1.2%–100% across studies by
threshold); the volume estimator must handle plexiform/multi-lobed geometry, not assume a single
ellipsoidal sac.

---

## 6. Imaging protocols (porcine-first, then comparison modalities)

**Porcine (most directly reusable):**
- **42155 2023:** lipiodol intranodal/mesenteric lymphangiography + **post-lymph CT 120 kV, B30f
  iterative, 3-D recon**; CC at **L2/L3 (renal-hilum level)**; manual lipiodol **0.1–0.5 mL/min**.
  MR option: gadoxetate 0.025 mmol/kg, 3D-TWIST dynamic **~2–4 s temporal res, 9 dynamics, voxel
  1.2 mm iso** (if an MR E(t) arm is ever wanted).
- **Lu/Kassab `2297906`:** bilateral intranodal inguinal lipiodol + transabdominal CC cannulation
  (2.8F Cantata) + Omnipaque, 6-landmark pressure pullback; fluoroscopy res ~0.6 mm.

**Comparison / human MR-lymphangiography sequences (heavy-T2 = the CC gold standard, detection
~90–93% vs CT ~14–16%):**
- Zhang Xiaojie 2024: 3.0T 3D heavy-T2, **TR 2500–3000 / TE 550–600 ms, ETL 85–105, voxel 1.0 mm
  iso**; + a full human pedal-DLG→CTL workflow (lipiodol 8–20 mL at 1–2 mL/h, CT 20 min–2 h post).
- Yu Dexin 2006: FS-T2WI (TR 4000–8000 / TE 70–110) **94.4%** CC detection vs 3D heavy-T2 MRL
  (TR 2000–4500 / **TE 550–750**) 87.8%, both ≫ pedal lymphangiography ~50%.
- Wu Bi 2007: TSE-3D heavy-T2 hydrography best (TR/TE 1870/764, 1 mm), CC detection **71%** vs CT
  1.7%.
- 293457 (Pinto 2004): 1.5T HASTE **TR 1060 / TE 116, ETL 256, 6 mm**, MIP improves conspicuity; CC
  scored only if ≥5 mm (a PV detection floor tied to slice thickness).

**Cat/dog CT-lymphangiography route atlas (Kang Bo 2024, Table 1):** 7 routes with cat-specific
doses/timing — e.g. **popliteal iohexol 350, 1.5 mL/kg, CT 3/5/7 min**; **percutaneous-hepatic
iopamidol 370 mgI/mL, 1–1.5 mL/kg → CC+TD imaged in 64% (7/11) cats**. Useful for timing windows.

**Animal-model opacification (no imaging):** Chen Yujuan 2021 — green-dye retrograde **mesenteric
lymph-node** injection fills CC via the intestinal trunk (dog gross-anatomy); Fransson — popliteal/
mesenteric **iohexol 60 mgI/kg → TD in 2–13 min**, pause-and-refill "turgid then relax" pressurization.

---

## 7. Method backbone — the integrated-HU estimator is the right foundation

**s10554 2019 (`A phantom based evaluation of vessel lumen area quantification for coronary CT
angiography`; Molloi, UCI).** Not a lymphatic paper — it is the *math* the project's PV-aware
estimator descends from:
- **CSA = (I − A·S_BG)/(S_O − S_BG)** — integrated-HU is conserved under partial-volume blurring,
  independent of spatial resolution; depends only on lumen-vs-background HU contrast. (This is the
  2-D form of the 3-D ∫occ / integrated-HU volume the project already uses.)
- Three-ROI recipe: central PV-free calibration core (S_O), object ROI capturing the full PV spread
  (I), ring ROI at **1.2× extent** for background (S_BG). Registration-free pre-contrast subtraction
  for confounders (calcified spine/nodes near the CC).
- **Performance floor: CV < 10% down to ~2 mm² cross-section; SNR-limited below** → a direct caution
  that the ~2 mm TD is near the estimator's resolution floor while the larger CC is comfortable.
- >2× better than manual ROIs; phantom validated to slope ~0.98–0.99.

Plus **Carvajal's CT+CAD validation (3–5% error vs known 2.0 mL)** = an independent confirmation that
threshold-based CT volume of an opacified lymphatic is accurate to a few percent.

**Fluid-property priors:** chyle specific gravity **1.030–1.032**, protein **~2.5 g/dL** (Fransson,
Kang) → CC lumen near-water density on unopacified CT (consistent with the ~4–18 HU pre-contrast we
and Carvajal see).

---

## 8. Passive pressure–flow & valve-resistance model — Patel/Kassab 2025 (the Q2 element)

*Added 2026-06-29 after the file rename surfaced it. This is `bioengineering-12-00401` — the paper the
original web-built brief MISLABELED as "valve spacing, translational model." Its real content is the
TD pressure–flow model. (The "31 mm valve spacing" actually came from Lu/Kassab 2020 = `2297906`,
which this paper cites as ref [29].) Read in full + numbers cross-checked against the paper's own
figures/table.*

**Patel B, Lu X, …, Kassab GS. "Pressure–Flow Relation of Porcine Thoracic Duct Segment."
*Bioengineering* 2025, 12, 401. doi:10.3390/bioengineering12040401.** Ex-vivo bench, **n=5 Yorkshire
swine (55 ± 7 kg)**. A single-valve TD segment (~31 mm, stretched 30% to **L = 40.3 mm**, since in-situ
length runs ~30% longer than ex-vivo) is cannulated, bathed in 0.9% saline at 37 °C, and driven by a
reservoir head **Δp = 0–10 cmH₂O in 1-cmH₂O steps**; volumetric flow Q and outer diameter are measured.

**The model (lumped electrical-circuit analogy) — this is the reusable part:**
- **Δp_total = (R_conn + R_vessel + R_valve)·Q** — series resistances; Q constant through the segment.
- **R_vessel = 128 µL / (π D⁴)** (Poiseuille), with inner **D = D_out − t** and wall **t = 0.11 ± 0.04 mm**.
- **R_valve = R_vl + R_vh · 1/(1 + e^{ s·Δp_valve })** — sigmoidal: **R_vl** = open-valve floor (high Δp),
  **R_vl + R_vh** = closed-valve max (large negative Δp), **s** = transition slope.
- Rig constant **R_conn = 0.393 cmH₂O/(mL/min)** (from a rigid phantom; phantom fit R²=0.992).
- Units throughout: resistance in **cmH₂O/(mL/min)**.

**Key results:**
- **The VALVE dominates flow resistance: R_valve is 1–2 orders of magnitude larger than R_vessel**
  (ratio ~330× at low Δp, min ~40× at Δp 4–6, ~70× at 10). R_vessel ≈ 0.001–0.010, R_valve ≈ 0.1–3.0.
- Both resistances **drop steeply then plateau** as Δp rises → the TD **stiffens fast** (diameter nearly
  constant above ~8 cmH₂O — a strain-stiffening P–D, matching the canine wall in the peri-adipose
  digest) and **the valve opens fast**.
- **The valve reaches near-maximal opening at Δp_valve ≈ 0.84 ± 0.42 cmH₂O.** Since the in-vivo TD
  gradient is ~9–10 cmH₂O (Lu 2020), in normal forward flow the valve sits at its **low** resistance
  R_vl (≈ 0.01–0.15 across segments).
- Fit quality: valve-resistance model **R² = 0.971 ± 0.027**; flow validation **R² = 0.985 ± 0.010**
  (per-segment 0.926–0.999).
- Per-segment params (Table 1): **R_vl 0.011–0.154** (stable, physiologically meaningful); **R_vh
  9.1×10² – 6.4×10¹⁰** and **s 14.5–1919** (span many orders — poorly constrained; the authors flag
  the exponential low-pressure regime makes R_vh, s sensitive, so trust R_vl + the 0.84 cmH₂O threshold,
  not the raw R_vh/s).

**Why it matters for the project:**
1. **Q₂ (CC→TD outflow) now has a forward model:** Q₂ = Δp / (R_vessel + R_valve). The transport model
   currently gets Q₂ from mass balance (Q₂ = Q₁ − dV_CC/dt); this gives an *independent*, pressure-based
   Q₂ and the resistance the duct imposes — a cross-check / closure for the Q₂ column.
2. **The first valve dominates the CC→TD resistance** — directly answers the boundary digest's open
   question #1 (is there functional gating at the CC–TD transition?): the resistance is mostly the
   *valve*, not the conduit.
3. **A passive P–D stiffening datum for the TD** (diameter ~constant above 8 cmH₂O) — partially fills
   gap (b) on the *duct* side (CC itself still unmeasured).
4. **Explicitly the PASSIVE relation**; the paper frames active (intrinsic contractility) as separate,
   future work — exactly our active-vs-passive distinction, and it endorses measuring the *passive*
   baseline first.
5. **Open data (`bench_data.xlsx`) + code (`main.ipynb`) on Zenodo** — directly reusable to seed the
   transport model's Q₂ element.

**Caveats (the authors' own):** ex-vivo, **static/passive only** (no periodic/dynamic flow), **no
external loading** (no respiration term), **no retrograde/adverse-gradient** valve resistance, single
valve, large inter-segment variation, R_vh/s poorly constrained. So it is a *passive static baseline*,
not the full dynamic duct.

## 9. Secondary — Chen et al. 2020 (TD anatomy + imaging review)

**Chen L, …, Kang M. "Application of imaging technique in thoracic duct anatomy." *Ann Palliat Med*
2020;9(3):1249–1256. doi:10.21037/apm.2020.03.10** (Fujian Medical Univ; human review). Lower-priority
— mostly TD **termination** variants + an imaging-technique survey, much of it already covered:
- CC morphology classes restated: **single / double / triple tubular / plexus**, seen in **53%
  lymphangiography, 50% autopsy, 15% abdominal MRI**; most CC are fusiform or cystic; **when CC is
  absent the TD forms from a confluent plexus**. (Consistent with §5; the 53% ≈ the Rosenberger &
  Abrams lymphangiography rate.)
- TD trunk variation: Davis 1915 nine-types; right-main 63%, left+right coexisting 27%; left/right-main
  39–47%. Termination: left jugular **92–95%**, right 2–3%, bilateral 1–1.5%; single TD 68–87.5%, 2
  ducts 8–25%, 3 ducts 4–7%; ~20% branch+reanastomose before termination; Japanese A–D termination
  subtypes (left jugular angle 38% / internal jugular 27% / external jugular 28% / complex 7%).
- Imaging survey: intraoperative fluorescence imaging (indocyanine-green-type dye), I-123 BMIPP
  lymphoscintigraphy, MR/CT lymphangiography. **No new quantitative data for the project's gaps** —
  keep as a variant/imaging-landscape reference only.

## Corrections applied during verification (do NOT propagate the reader's originals)

| Paper | Reader said | Correct value (from PDF) |
|---|---|---|
| Carvajal | "first-ever animal CC volume" | paper makes **no such novelty claim**; 1.82 mL itself is correct |
| Carvajal | Table 2 cross-tab (laterality × level) | laterality & vertebral level are **independent columns**; T11-L3 = 1/23, T13-L2 = 1/23 (not 3/23, 2/23) |
| Tubbs | Van Pernis "0% / no definable CC" | Van Pernis saw CC in **ALL 1081 cadavers** (≈100%) — opposite |
| Tubbs | Loukas Type III = left lumbar + intestinal | Type III = **right** lumbar + intestinal (Type I is left) |
| Moazzam | plexiform 17–58% | plexiform **17–19%**; the 58% is **saccular** (Loukas) |
| Moazzam | CT weighted-avg length 13.1 mm | **13.8 mm** (13.1 was the single Feuerlein row) |
| Moazzam | CC enlarges in autoimmune hepatitis | it does **NOT** in immune-mediated pathology |
| Wu Bi | HASTE slice gap 8 mm | **3 mm** (thickness 8 mm is right) |
| Wu Bi | CC at L2 = 6 cases | **38 cases** |
| Wu Bi | TD most-shown = origin | **terminal** segment (before left venous angle) |
| Pan Haipeng | rabbit TD ~1–2 mm | the ~1–2 mm caliber is the **lumbar trunk**, not the TD |
| Pan Haipeng | ~53% = INL CC detection | ~53% = **(pedal) lymphangiography** CC detection in general |
| Kang Bo | iopamidol 1370 mg/mL | **370 mgI/mL** |
| Kang Bo | dog perianal CT "1.5 min", "peak ~10 min" | scans at **1 and 5 min**; enhancement **weakens after 10 min** |
| Kang Bo | cat doses "1.5 mL", "ALT" | **1.5 mL/kg**; generic **liver-enzyme** rise (not ALT-specific) |
| Ji Rongming | "lower than pooled human ~55%" | the ~55% comparison is **not in the paper** (external); the 22% figure itself is correct |

---

## Not useful / set aside
- **Zhang Li 2016** (下肢淋巴显像/lower-limb lymphoscintigraphy for chylous reflux lymphedema) — off-topic, no CC/TD data.
- **s-0040-1713448** (Complications during Lymphangiography) — method/safety context only; no new anatomy/volume.
- **Chen Yujuan 2021** (dog dye method) — confirms inflow topology (intestinal + bilateral lumbar trunks → CC; TD = single outflow) but no quantitative data.
- **Moazzam (= AJP-Heart) and `2297906` (= Lu/Kassab)** are already cited in the brief — kept here only for the *new* details (302 µL, CC pressure 15.3 mmHg, per-pig caliber, extrinsic-drive).

---

## References (this batch)

- **Carvajal JL, et al.** Anatomic and volumetric characterization of the cisterna chyli using CT
  lymphangiography and CAD software in dogs with idiopathic chylothorax. *Vet Radiol Ultrasound* 2020.
- **Weisse C.** Cisterna Chyli and Thoracic Duct Glue Embolization. In: *Veterinary Image-Guided
  Interventions*, 2015, ch.50.
- **Fransson BA.** Minimally Invasive Chylothorax Treatment. In: *Small Animal Laparoscopy and
  Thoracoscopy*, 2015, ch.36.
- **Loukas M, et al.** Cisterna chyli: a detailed anatomic investigation. *Clinical Anatomy* 2007.
- **Tubbs RS, et al.** Thoracic duct, cisterna chyli, and right lymphatic duct. In: *Bergman's
  Comprehensive Encyclopedia of Human Anatomic Variation*, 2016, ch.75.
- **Moazzam Z, et al.** The cisterna chyli: a systematic review of definition, prevalence, and
  anatomy. 2022 *(= AJP-Heart 2022 review, doi:10.1152/ajpheart.00375.2022; already in brief)*.
- **`2297906`** = file `2020 Morphometry and Lymph Dynamics of Swine Thoracic Duct.pdf` — Lu/Kassab,
  *Lymphat Res Biol* 2020 (doi:10.1089/lrb.2019.0069; already in brief).
- **`293457`** = file `2004 Cisterna Chyli at Routine Abdominal MR Imaging - A Normal Anatomic Structure in the Retrocrural Space.pdf` — Pinto PS, et al. *RadioGraphics* 2004.
- **`42155`** = file `2023 Standardizing lymphangiography and lymphatic interventions - a preclinical in vivo approach with detailed procedural steps.pdf` — German landrace pigs, 2023.
- **`s-0040`** = file `2020 Complications during Lymphangiography and Lymphatic Interventions.pdf` — 2020.
- **`s10554`** = file `2019 A phantom based evaluation of vessel lumen area quantification for coronary CT angiography.pdf` — integrated-HU method (Molloi), 2019.
- **Yu Dexin 于德新, et al.** 3.0T MRI T2WI vs heavy-T2 detection of the cisterna chyli. *中国医学影像技术* 2006.
- **Zhang Xiaojie 张晓杰, et al.** CT-lymphangiography vs non-enhanced MR-lymphangiography for chylothorax. 2024.
- **Wu Bi 吴苾, et al.** Non-enhanced MR lymphography of the cisterna chyli and thoracic duct. 2007.
- **Pan Haipeng 潘海鹏.** Preliminary experimental MR imaging of the thoracic duct (40 rabbits + review). 2017.
- **Ji Rongming 纪荣明, et al.** Anatomic study of chyle leakage due to abdominal operation (32 cadavers). 2004.
- **Kang Bo 康博, Guo Ruize, Zhang Di.** CT-lymphangiography in feline chylothorax. 2024.
- **Chen Yujuan 陈玉娟.** A simple method to display whole-body lymph trunks in animals (dog). 2021.
- **He Shangkuan 何尚宽.** A case of three cisterna chyli. 1989.
- **Guan Qing 关清.** A case of cisterna chyli with positional variation. 2012.
- **Zhou Jinge 周锦鸽.** A case of right thoracic duct with reticular cisterna chyli. 2016.
- **Zhang Li 张丽.** Lower-limb lymphoscintigraphy in chylous reflux lymphedema. 2016. *(off-topic)*
- **Patel B, Lu X, …, Kassab GS.** Pressure–Flow Relation of Porcine Thoracic Duct Segment.
  *Bioengineering* 2025;12(4):401. doi:10.3390/bioengineering12040401. *(file: `2025 Pressure-Flow
  Relation of Porcine Thoracic Duct Segment.pdf`; = `bioengineering-12-00401`, previously mislabeled
  "valve spacing" — see §8; data + code on Zenodo.)*
- **Chen L, …, Kang M.** Application of imaging technique in thoracic duct anatomy. *Ann Palliat Med*
  2020;9(3):1249–1256. doi:10.21037/apm.2020.03.10. *(file: `2020 Application of imaging technique in
  thoracic duct anatomy.pdf`; TD variation + imaging review — see §9.)*

*Provenance: 22 PDFs, one reader + one independent adversarial verifier each (44 agents, all "high"
confidence). All corrections in the table above are verifier-confirmed against the source PDFs.*
