"""Phase-2 distensibility: does the CC volume rise with lymph flow (compliant-reservoir
hypothesis), or is it a passive widening?

For every acquisition: robust integrated-HU volume (measure.volume) + the 1b-A QA flag
(recomputed from source via measure.extract, not read from a CSV). On the CLEAN acqs:
  - within-animal 07_20_22 (the valid test): baseline vs angiotensin volume, since
    angiotensin raises flow and animal/CC size is held fixed;
  - pooled V-vs-flow across all 3 animals, shown only with the size-confound caveat
    (different animals have different CC sizes, which dwarfs a flow effect).

Outputs phase2_volumes.csv + phase2_volume_vs_flow.png.

  CC_ARCHIVE=<path> PYTHONPATH=src python3 -m cc_reservoir.scripts.phase2_distensibility
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cc_reservoir.io.workingset import WORKING_SET
from cc_reservoir.measure.patch import load_cc_patch
from cc_reservoir.measure.volume import robust_volume
from cc_reservoir.measure.extract import extract_curves, method_agreement, qa_pass

OUT = os.environ.get("CC_PHASE2_OUT", "/tmp/cc_phase2")
os.makedirs(OUT, exist_ok=True)

rows = []
for spec in WORKING_SET:
    patch = load_cc_patch(spec)
    V, V_spread, used = robust_volume(patch)
    curves = extract_curves(list(enumerate(patch.vols)), patch.mask, patch.voxel_mm)
    agr = method_agreement(curves)
    rows.append(dict(session=spec.session, acq=spec.subpath, condition=spec.condition,
                     probe_flow=spec.probe_flow, ct_flow=spec.ct_flow,
                     volume_mm3=V, volume_spread_mm3=V_spread, n_opacified=len(used),
                     containment=agr["containment_at_peak"],
                     ring_vs_tp1=agr["ring_vs_tp1_peak_ratio"], clean=qa_pass(agr)))

with open(os.path.join(OUT, "phase2_volumes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

clean = [r for r in rows if r["clean"] and np.isfinite(r["volume_mm3"])]
print(f"{len(clean)}/{len(rows)} clean acqs with a finite robust volume\n")
print(f"{'acq':28} {'cond':11} {'probe':>6} {'V (mm^3)':>12} {'n_op':>5}")
for r in rows:
    tag = "" if (r["clean"] and np.isfinite(r["volume_mm3"])) else "  (excluded)"
    print(f"  {r['session'][:7]+'/'+os.path.basename(r['acq']):26} {r['condition']:11} "
          f"{r['probe_flow']:6.2f} {r['volume_mm3']:8.0f}+-{r['volume_spread_mm3']:<3.0f} "
          f"{r['n_opacified']:5d}{tag}")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---- within-animal 07_20_22: the valid test ----
animal = [r for r in clean if r["session"] == "07_20_22_data"]
base = [r for r in animal if r["condition"] == "baseline"]
angio = [r for r in animal if r["condition"] == "angiotensin"]
Vb = np.array([r["volume_mm3"] for r in base]); Va = np.array([r["volume_mm3"] for r in angio])
Fb = np.array([r["probe_flow"] for r in base]); Fa = np.array([r["probe_flow"] for r in angio])
print("\n=== within-animal 07_20_22 (CC size fixed) ===")
print(f"  baseline   n={len(Vb)}  flow {Fb.mean():.2f}  V = {Vb.mean():.0f} +- {Vb.std(ddof=0):.0f} mm^3  {np.round(Vb).tolist()}")
print(f"  angiotensin n={len(Va)} flow {Fa.mean():.2f}  V = {Va.mean():.0f} +- {Va.std(ddof=0):.0f} mm^3  {np.round(Va).tolist()}")
if len(Vb) and len(Va):
    dV = Va.mean() - Vb.mean()
    print(f"  delta V (angio - baseline) = {dV:+.0f} mm^3 ({100*dV/Vb.mean():+.0f}%); "
          f"flow {Fb.mean():.2f} -> {Fa.mean():.2f}")
print(f"  within-animal Pearson r(V, probe_flow) = {pearson([r['probe_flow'] for r in animal], [r['volume_mm3'] for r in animal]):.2f}  (n={len(animal)})")

# ---- pooled, with caveat ----
print("\n=== pooled across 3 animals (size-confounded) ===")
print(f"  Pearson r(V, probe_flow) = {pearson([r['probe_flow'] for r in clean], [r['volume_mm3'] for r in clean]):.2f}  (n={len(clean)})")
print("  NOTE: pooling mixes per-animal CC size with any flow effect; treat as descriptive only.")

# ---- figure ----
COND = {"baseline": ("#1f6fe0", "o"), "angiotensin": ("#d8392b", "^")}
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=130)

ax = axs[0]
for r in animal:
    c, m = COND[r["condition"]]
    ax.scatter(r["probe_flow"], r["volume_mm3"], s=130, c=c, marker=m, edgecolor="k",
               linewidth=0.6, zorder=3, label=r["condition"])
for cond, V, F in (("baseline", Vb, Fb), ("angiotensin", Va, Fa)):
    if len(V):
        ax.scatter([], [], c=COND[cond][0], marker=COND[cond][1], edgecolor="k", label=cond)
        ax.hlines(V.mean(), F.min() - 0.05, F.max() + 0.05, color=COND[cond][0], lw=2, alpha=0.6)
handles, labels = ax.get_legend_handles_labels()
seen = dict(zip(labels, handles))
ax.legend(seen.values(), seen.keys(), fontsize=9, loc="best")
ax.set_title("Within-animal 07_20_22 (CC size fixed)\nthe valid compliant-reservoir test", fontsize=11)
ax.set_xlabel("probe lymph flow (mL/min)"); ax.set_ylabel("CC volume (integrated-HU, mm³)")
ax.grid(alpha=0.25)

ax = axs[1]
animals = sorted({r["session"] for r in clean})
amark = {a: mk for a, mk in zip(animals, ["o", "s", "D"])}
for r in clean:
    c = COND[r["condition"]][0]
    ax.scatter(r["probe_flow"], r["volume_mm3"], s=110, c=c, marker=amark[r["session"]],
               edgecolor="k", linewidth=0.6, zorder=3)
for a, mk in amark.items():
    ax.scatter([], [], c="gray", marker=mk, edgecolor="k", label=a.replace("_data", ""))
for cond, (c, _) in COND.items():
    ax.scatter([], [], c=c, marker="o", edgecolor="k", label=cond)
ax.legend(fontsize=8, loc="best")
ax.set_title("Pooled, 3 animals (size-confounded — descriptive only)", fontsize=11)
ax.set_xlabel("probe lymph flow (mL/min)"); ax.set_ylabel("CC volume (integrated-HU, mm³)")
ax.grid(alpha=0.25)

fig.suptitle("Phase-2 distensibility: does CC volume rise with lymph flow?", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, "phase2_volume_vs_flow.png"), dpi=130, facecolor="white", bbox_inches="tight")
print(f"\nsaved {OUT}/phase2_volumes.csv + phase2_volume_vs_flow.png")
