# The CC/TD boundary — literature review

*Compiled 2026-06-23. Scope: how the peer-reviewed literature defines the cisterna chyli (CC) /
thoracic duct (TD) boundary — anatomy, morphology, imaging, valves, prevalence/variability, and
reservoir function — with porcine anatomy foregrounded for the swine reservoir-modeling project.
All numeric claims below survived 3-vote adversarial verification against the cited primary
sources. DOIs / PMC IDs are the authoritative anchors; verify author/year against the DOI before
citing in the manuscript.*

---

## Bottom line for this project

1. **There is no consensus morphological threshold separating CC from TD.** The boundary is a
   *subjective caliber/shape judgment*, not an objective criterion — stated explicitly in the
   AJP-Heart systematic review. This directly backs your method commitment to anchor the CC/TD
   split **anatomically (caudal-end)** rather than by a diameter rule
   ([docs/cc_transport_model.md](../docs/cc_transport_model.md), commitment #2).

2. **Diameter genuinely does not discriminate.** Pooled human CC diameter is 2–32 mm; TD caliber
   is ~2–5 mm (mean termination 3.79 mm). The low end of the CC range *overlaps* the TD, so even
   in humans these are a bounding envelope, not a separator. Your Acq16 case (CC d≈3.5 mm < cranial
   ampulla d≈4.7 mm) is not anomalous — it sits inside this known overlap.

3. **The CC-as-reservoir / TD-as-outflow framing is literature-supported.** The AJP-Heart review
   calls the CC a *"primary reservoir for collecting and storing lymph."* Swine measurements give a
   directional CC→jugular flow (0.7 ± 0.49 mL/min) under a roughly constant 8.1 mmHg gradient.

4. **The single load-bearing parameter for your model is *not* in the literature.** No reviewed
   source reports porcine CC **volume** or its **pressure–volume (compliance)** relation — only
   linear dimensions and duct flow/pressure. This is the gap your measurement pipeline fills
   (and ties to your standing note that *residence ≠ compliance*).

---

## 1. Anatomical definition & location of the CC→TD transition

The CC is the **dilated, saccular abdominal origin of the TD**, in the retrocrural space to the
**right of and behind the abdominal aorta**, lateral to the **right crus of the diaphragm**. The TD
ascends from its **cranial/superior aspect** through the aortic hiatus into the posterior
mediastinum. So the **CC–TD junction is the proximal (caudal) end of the duct**, and the boundary
is a caliber step-down from the wide saccular CC to the narrow tubular TD.

- **Human vertebral level:** classically **L1–L2** (single most frequent, pooled 19.08%),
  concentrating in the thoracolumbar transition **T12–L2**; full distribution spans **T9/T10 → L2/L3**.
  A systematic review widens the structural range to **T10–L3**. The commonly cited "T11–L2 junction
  window" is therefore the central tendency of a wide, variable distribution (I² up to ~97%).
- *(Kiyonaga/Mori, Br J Radiol 2012, PMC3587101; MDPI meta-analysis, J Clin Med 2024;13:4285;
  RadioGraphics, doi:10.1148/rg.243035086; AJP-Heart review, doi:10.1152/ajpheart.00375.2022.)*

## 2. Morphological criteria distinguishing CC from TD

The distinction is by **shape and caliber, not a sharp demarcation**: CC = wide saccular/dilated;
TD = narrow tubular outflow. **CC shape is highly variable** — tubular (thick/thin/converging/
tortuous), saccular/sausage-shaped/fusiform, and plexiform forms all occur, with no single dominant
morphology (literature lists 6+: bulbous, saccular, tubular, fusiform, plexiform, deltaic, inverted-V,
Y-shaped, beaded).

- One MDCT study (Kiyonaga/Mori, n=40 visualized) binned CC into **three types** — straight thin
  tube (5–9 mm; **75% of cases**), straight thick tube (≥10 mm), focal round/oval. **Attribute this
  scheme to that single study**; it is *not* a field-wide consensus taxonomy.
- **No agreed-upon objective criterion** exists for what degree of dilatation "counts" as a CC; the
  call is subjective. *(AJP-Heart review, doi:10.1152/ajpheart.00375.2022.)*

## 3. Imaging characterization & reported dimensions

- **Routine protocols under-detect:** CC evident on **~15%** of routine abdominal MR (30/200
  consecutive; Pinto/Sirlin, RadioGraphics 2004) and **~16%** on contrast-enhanced CT (Feuerlein,
  Eur Radiol 2009, n=3000).
- **Dedicated lymphatic imaging** (thin-slice MDCT, T2 MRI) reaches **>90%**. Overall detectability
  spans **1.7–98%**, driven by method *and* the subjective criteria — the variability is itself the
  boundary's defining feature.
- **Human morphometric envelope:** CC **2–32 mm** max diameter, **13–80 mm** max length (pooled mean
  length **18.25 mm**, 95% CI 14.55–21.94); **mean TD termination diameter 3.79 mm** by CT (normal TD
  caliber ~2–5 mm). Treat as an envelope, not a discriminant (CC lower bound overlaps TD).

## 4. Valves / sphincter at the junction

- The (swine) TD carries **9–13 intraluminal anti-reflux valves** along its full length
  (**~31 mm** average ex-vivo spacing). These prevent retrograde flow.
- **No discrete functional sphincter** is documented at the CC–TD junction. The porcine origin is a
  **single-trunk caliber continuation**, not a valved sphincteric gate. *(Note: this rests partly on
  absence of contrary evidence — a slightly weaker basis than a positive statement.)*
- *(Lu/Kassab, Lymphatic Research and Biology 2020, doi:10.1089/lrb.2019.0069, PMID 32202948;
  Patel/Kassab, Bioengineering 2025;12:401, PMID 40281761.)*

## 5. Prevalence & anatomical variability

- CC present in **only ~55% of humans** (pooled 55.49%, 95% CI **26.79–82.53%**, n=1447, I²~99%).
  **When absent, the TD arises from a less distinct plexus** of lymphatic channels (or dilated
  lumbar trunks / multiple sacculations) — i.e. *no discrete sac to bound at all*.
- **Statistical caveat:** these pooled point estimates blend incompatible detection methods; with
  I²≈99% they are method-blended, near-uninterpretable as a true biological rate. **Report with the
  wide CIs, not as clean values.** *(MDPI meta-analysis 2024; AJP-Heart review 2022.)*

## 6. Functional / physiological characterization (reservoir → outflow)

- **AJP-Heart review (explicit):** the CC *"functions as a primary reservoir for collecting and
  storing lymph fluid enriched with chyle."*
- **Swine in vivo (Lu/Kassab 2020, n=9):** lymph flows **CC → lymphovenous (jugular) junction** under
  a relatively constant **8.1 mmHg** average gradient along the TD; **flow 0.7 ± 0.49 mL/min**. Flow
  is pulsatile (valve/contraction pump, ~25% pressure rise at ~5/min) superimposed on the net
  downstream gradient.
- **Proposed respiratory cycle (hypothesis):** CC fills during **expiration** (abdominal pressure
  falls); increased **intrathoracic pressure** empties it cephalad past central venous pressure
  (Kelly 2022, via Moazzam review 2026). Treat as provisional — the respiratory-phase literature is
  *"widely discordant."*

---

## Porcine anatomy (most load-bearing for the swine model)

| Quantity | Value | Source |
|---|---|---|
| CC origin level | **L3**, at renal-artery origins; retroaortic, under diaphragmatic crura | Gomercic/Duras 2010 |
| CC dimensions | **6–11 cm** long × **~0.5 cm** (5 mm) wide | Gomercic/Duras 2010 |
| CC→TD transition | single-trunk **direct continuation**, at **13th/14th dorsal intercostal arteries** | Gomercic/Duras 2010 |
| TD diameter at origin | **~0.2 cm** (2 mm) → **~2.5× step-down** from CC | Gomercic/Duras 2010 |
| CC width (alt. series) | **11.4–15 mm** vs TD **2–4.3 mm** → **~3–7× step-down** | Lu/Kassab 2020 |
| TD valves | **9–13** over full length, ~31 mm spacing | Lu/Kassab 2020; Patel/Kassab 2025 |
| Flow / gradient | **0.7 ± 0.49 mL/min**; **8.1 mmHg** CC→jugular | Lu/Kassab 2020 |
| Surgical access | CC–TD junction in cranial sub-lumbar region, near aortic hiatus (retract left crus) | BMC Vet Res 2015 |
| Translational validity | porcine TD anatomically & physiologically similar to human | Patel/Kassab 2025; Riquet 2000 |

**Cross-species caveat:** pig landmarks (**"L3"**, 13th/14th dorsal intercostal arteries; pigs have
~13–15 thoracic vertebrae) are **NOT homologous** to human T11–L2/L1–L2. Use **porcine numbers
directly** for swine geometry — do not translate human values.

**Reconciliation with your Acq16:** the porcine CC width range (5–15 mm) is *larger* than your
measured CC d≈3.5 mm. Two readings worth checking: (a) your caudal-anchored CC segment may be
under-segmented relative to the dissection-defined CC, or (b) the wider cranial structure you label
the "TD ampulla" (d≈4.7 mm) may correspond to a more dilated CC/transition region than the caudal
tip. Either way, the literature confirms diameter alone cannot adjudicate the split — your anatomical
anchor is the defensible choice.

---

## Open questions (literature gaps relevant to the model)

1. **No direct measurement at the porcine CC–TD junction** (vs. duct as a whole) — does the
   single-trunk continuation imply negligible resistance / no functional gating at the transition?
2. **No measured porcine CC volume or compliance (P–V relation)** — the capacitance parameter that a
   reservoir model actually needs. Sources give linear dimensions + duct flow/pressure only.
3. **Respiratory fill–empty dynamics under mechanical vs. spontaneous ventilation** — the Kelly 2022
   mechanism is hypothesized for spontaneous breathing; some measurements were under positive-pressure
   ventilation (where CVP rose more than TD pressure).
4. **No cross-species scaling map** from human CC/TD morphometry onto porcine landmarks, so human
   imaging literature cannot yet be cleanly reused to constrain the swine geometry.

---

## References

1. **Kiyonaga M, Mori H, et al.** Thoracic duct and cisterna chyli: evaluation with MDCT.
   *Br J Radiol* 2012. PMC3587101. https://pmc.ncbi.nlm.nih.gov/articles/PMC3587101/
2. **Meta-analysis of the cisterna chyli** (prevalence, level, morphometry). *J Clin Med*
   2024;13(15):4285. PMC11313251. https://www.mdpi.com/2077-0383/13/15/4285
3. **Pinto PS, Sirlin CB, et al.** Cisterna chyli at routine abdominal MR imaging. *RadioGraphics*
   2004;24:809–817. doi:10.1148/rg.243035086. PubMed 15143230.
   https://pubs.rsna.org/doi/10.1148/rg.243035086
4. **Systematic review of the thoracic duct / cisterna chyli** (49 human studies).
   *Am J Physiol Heart Circ Physiol* 2022. doi:10.1152/ajpheart.00375.2022.
   https://journals.physiology.org/doi/full/10.1152/ajpheart.00375.2022
5. **Moazzam et al.** Thoracic duct review. *Physiological Reports* 2026. doi:10.14814/phy2.70742.
   https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.70742
6. **Gomercic T, Duras M, et al.** The cisterna chyli and thoracic duct in pigs (*Sus scrofa
   domestica*). *Veterinarni Medicina* 2010;55(1):30–34.
   http://vetmed.agriculturejournals.cz/artkey/vet-201001-0003_the-cisterna-chyli-and-thoracic-duct-in-pigs-sus-scrofa-domestica.php
7. **Lu/Kassab et al.** Biomechanics of the swine thoracic duct (flow, pressure, valves).
   *Lymphatic Research and Biology* 2020. doi:10.1089/lrb.2019.0069. PubMed 32202948.
   https://www.liebertpub.com/doi/10.1089/lrb.2019.0069
8. **Patel B, Lu X, …, Kassab GS.** Pressure–flow relation of porcine thoracic duct segment — passive
   vessel + sigmoidal valve-resistance model (the ~31 mm valve spacing it quotes is from ref 7 =
   Lu/Kassab 2020, NOT measured here; earlier "valve spacing, translational model" label was wrong —
   see `cc_2026-06_new_papers_digest.md` §8). *Bioengineering* 2025;12(4):401. PMID 40281761.
   https://www.mdpi.com/2306-5354/12/4/401
9. **Large White pig TD cannulation** study (surgical access to CC–TD junction). *BMC Vet Res*
   2015. PMC4429499. https://pmc.ncbi.nlm.nih.gov/articles/PMC4429499/
10. **Feuerlein S, et al.** Cisterna chyli on contrast-enhanced CT (n=3000, 16.1%). *Eur Radiol*
    2009. *(corroborating, secondary)*
11. **Riquet M, et al.** Lymphatics of heart and lungs in the pig. *Surg Radiol Anat* 2000;22:47–50.
    *(porcine translational support)*

*Provenance: synthesized via a fan-out web-search + adversarial-verification harness (14 sources,
69 candidate claims, 25 verified, all 25 confirmed 3-0/2-1). Two findings were 2-1 votes (the
T10–L3 range and the swine-as-model claim) but remain well-supported. Author names/years are as the
verification attributed them — anchor citations on the DOI/PMC ID.*
