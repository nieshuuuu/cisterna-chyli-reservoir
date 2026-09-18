"""CC lumen volume at EVERY painted timepoint, not just the peak frame.

The mesh/render pipeline reports one frame per acquisition (the enhancement peak), and those
peaks do not fall on the same timepoint index across acquisitions — so a volume comparison
between acquisitions silently compares different phases of bolus filling. Since the painter
produced a mask for most frames, V(t) is directly computable, which is also the quantity the
project's design doc listed as ABSENT (CC segmented at one timepoint only).

Note on why this is not fixed by picking a common timepoint (operator, 2026-07-29): the animal
is BREATHING. Two acquisitions sampled at the same elapsed second are still at uncontrolled and
different points of the respiratory cycle, and the diaphragm carries the cisterna with it. There
is no common phase to align to here, so the comparable quantity is not V at any single instant —
it is each acquisition's DISTRIBUTION of V over its own timepoints (mean, and the range it
spans). Two acquisitions agree when their ranges overlap, not when their peak frames match.

Each frame is refined independently, against ITS OWN enhancement field, with the same
0.25 x local-peak rule. That fraction is scale-free, so a lumen of fixed size whose contrast
is merely washing in and out should give a flat V(t); departures from flat are distension.

Frames whose local peak never clears ``floor_hu`` carry no usable lumen signal — the refinement
falls back to the raw tracing there, so they are reported but flagged, not silently plotted as
if measured.

  CC_CONTEXTS=<dir> CC_RENDER_OUT=<dir> python -m cc_reservoir.scripts.cc_volume_timeseries
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from cc_reservoir.io.context import load_seg
from cc_reservoir.io.volumes import crop_to_bbox, mask_bbox
from cc_reservoir.io.workingset import ARCHIVE, CONTEXTS, PAINTED_SET
from cc_reservoir.measure.patch import CCPatch, enhancement_field
from cc_reservoir.measure.refine import refine_traced_lumen
from cc_reservoir.scripts.compute_cc_meshes_painted import frame_paths

OUT = os.environ.get("CC_RENDER_OUT", os.path.join(os.environ.get("TEMP", "/tmp"), "cc_painted"))
LEVEL_FRAC, FLOOR_HU, PAD = 0.25, 40.0, 18
COVER_MIN = 0.70          # a frame tracing less of the duct's length than this is incomplete
LBL_CC = 1
os.makedirs(OUT, exist_ok=True)


def timepoint_seconds(session, subpath):
    """Elapsed seconds of each DICOM timepoint from the first, in acquisition order."""
    dcm = os.path.join(ARCHIVE, session, subpath, "DICOM")
    ts = []
    for tp in sorted(os.listdir(dcm)):
        d = os.path.join(dcm, tp)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*")))[:5]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                continue
            t = str(getattr(ds, "AcquisitionTime", "") or "")
            if t:
                ts.append(int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:]))
            break
    ts.sort()
    return [t - ts[0] for t in ts] if ts else []


def series(spec):
    ctx = os.path.join(CONTEXTS, spec.context)
    meta = np.load(os.path.join(ctx, "meta.npz"), allow_pickle=True)
    voxel = tuple(float(v) for v in meta["voxel"])
    vvol = float(np.prod(voxel))
    z_lo = None if spec.cc_z_min is None else int(spec.cc_z_min) - int(meta["crop_lo"][2])
    z_hi = None if spec.cc_z_max is None else int(spec.cc_z_max) - int(meta["crop_lo"][2])
    frames = frame_paths(ctx)

    masks = {}
    for i, f in enumerate(frames):
        if not os.path.exists(os.path.splitext(f)[0] + "_edit.npz"):
            continue
        c = load_seg(f) == LBL_CC
        if z_lo is not None or z_hi is not None:
            c = c.copy()
            if z_lo is not None and z_lo > 0:
                c[:, :, :z_lo] = False
            if z_hi is not None and z_hi + 1 < c.shape[2]:
                c[:, :, z_hi + 1:] = False
        if c.any():
            masks[i] = c
    if not masks:
        return None

    union = np.zeros_like(next(iter(masks.values())))
    for c in masks.values():
        union |= c
    lo, hi = mask_bbox(union, pad=PAD)
    vols = []
    for f in frames:
        with np.load(f) as d:
            vols.append(crop_to_bbox(d["hu"], lo, hi).astype(np.float32))

    # Coverage: how much of the duct's length this frame's tracing spans, against the longest
    # tracing in the acquisition. A frame traced over half the duct is not a measurement of a
    # small CC, it is half a measurement — and the refinement cannot rescue it, because it is
    # bounded by the tracing it is given. Distinct from FLOODED (threshold failed) and from
    # LOW SIGNAL (no bolus): this is an incomplete tracing, and only repainting fixes it.
    # Reference is the MEDIAN span, not the max. An over-extended frame (CC label carried down
    # the lumbar trunks or up the TD) would otherwise set the bar and make the well-traced
    # frames look deficient — on 8_31_22/Acq4 that flagged the two cleanest frames.
    zspan = {}
    for i, c in masks.items():
        zs = np.where(c.any(axis=(0, 1)))[0]
        zspan[i] = int(zs.max() - zs.min() + 1)
    zref = float(np.median(list(zspan.values())))

    rows = []
    for i, c in sorted(masks.items()):
        m = crop_to_bbox(c, lo, hi)
        # enhancement_field needs a patch; TD is not excluded here because the CC crop for
        # these acquisitions barely contains it and the refinement is bounded by the tracing.
        p = CCPatch(mask=m, td=np.zeros_like(m), vols=vols, peak_i=i, voxel_mm=voxel,
                    lo=lo, hi=hi)
        enh, _ = enhancement_field(p)
        e = enh[i]
        r = refine_traced_lumen(m, e, voxel, level_frac=LEVEL_FRAC, floor_hu=FLOOR_HU)
        cover = zspan[i] / zref
        partial = cover < COVER_MIN
        usable = bool(np.isfinite(r.levels).any() and (r.peaks > FLOOR_HU).any()
                      and not r.flooded and not partial)
        rows.append(dict(frame=i, trace_mm3=m.sum() * vvol, lumen_mm3=r.mask.sum() * vvol,
                         mean_enh=float(e[m].mean()), peak_enh=float(np.percentile(e[m], 90)),
                         usable=usable, flooded=r.flooded, cover=cover, partial=partial))
    return rows, timepoint_seconds(spec.session, spec.subpath)


results = {}
print(f"{'acq':<8}{'frame':>6}{'t (s)':>8}{'trace mm3':>11}{'lumen mm3':>11}"
      f"{'mean enh':>10}{'p90 enh':>9}{'cover':>7}  flag")
for spec in PAINTED_SET:
    got = series(spec)
    if got is None:
        continue
    rows, secs = got
    results[spec.subpath] = (rows, secs, spec.probe_flow)
    for r in rows:
        t = secs[r["frame"]] if r["frame"] < len(secs) else float("nan")
        print(f"{spec.subpath:<8}{r['frame']:>6}{t:>8.0f}{r['trace_mm3']:>11.0f}"
              f"{r['lumen_mm3']:>11.0f}{r['mean_enh']:>10.0f}{r['peak_enh']:>9.0f}"
              f"{r['cover']:>7.2f}  "
              f"{'' if r['usable'] else ('FLOODED -> tracing kept' if r['flooded'] else ('PARTIAL TRACING - repaint' if r['partial'] else 'LOW SIGNAL'))}")

fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), dpi=140)
cols = plt.cm.viridis(np.linspace(0.08, 0.82, len(results)))
for k, (acq, (rows, secs, probe)) in enumerate(results.items()):
    t = [secs[r["frame"]] if r["frame"] < len(secs) else np.nan for r in rows]
    ok = [r["usable"] for r in rows]
    lum = [r["lumen_mm3"] for r in rows]
    tr = [r["trace_mm3"] for r in rows]
    enh = [r["mean_enh"] for r in rows]
    axes[0].plot(t, lum, "-o", color=cols[k], ms=4, lw=1.7, label=f"{acq}  probe {probe:.2f}")
    axes[0].plot([x for x, o in zip(t, ok) if not o], [y for y, o in zip(lum, ok) if not o],
                 "x", color="crimson", ms=9, mew=2)
    axes[1].plot(t, tr, "-o", color=cols[k], ms=4, lw=1.7)
    axes[2].plot(t, enh, "-o", color=cols[k], ms=4, lw=1.7)
axes[0].set_title("refined lumen volume V(t)")
axes[1].set_title("raw tracing volume (operator variability)")
axes[2].set_title("mean enhancement in the tracing")
axes[0].set_ylabel("mm$^3$"); axes[1].set_ylabel("mm$^3$"); axes[2].set_ylabel("HU")
for a in axes[:3]:
    a.set_xlabel("seconds from first timepoint of the acquisition")
    a.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=8, frameon=False)

# Panel 4 — the comparable quantity. Each acquisition as mean + full range over its own
# timepoints, because respiration makes any single instant non-comparable between acquisitions.
ax = axes[3]
stats = []
for k, (acq, (rows, secs, probe)) in enumerate(results.items()):
    v = np.array([r["lumen_mm3"] for r in rows if r["usable"]])
    if not v.size:
        continue
    stats.append((acq, v.mean(), v.min(), v.max(), v.size, probe, cols[k]))
for i, (acq, mu, lo, hi, n, probe, c) in enumerate(stats):
    ax.vlines(i, lo, hi, color=c, lw=8, alpha=0.35)
    ax.plot([i], [mu], "o", color=c, ms=9)
    ax.text(i, hi + 12, f"n={n}", ha="center", fontsize=8, color=(0.35, 0.35, 0.35))
ax.set_xticks(range(len(stats)))
ax.set_xticklabels([f"{s[0]}\nprobe {s[5]:.2f}" for s in stats], fontsize=8.5)
ax.set_ylabel("mm$^3$")
ax.set_title("per-acquisition mean and range")
ax.grid(alpha=0.25, lw=0.5, axis="y")
allv = np.concatenate([np.array([s[2], s[3]]) for s in stats])
ax.set_ylim(allv.min() - 60, allv.max() + 60)

overlap_lo = max(s[2] for s in stats)
overlap_hi = min(s[3] for s in stats)
verdict = (f"all four ranges overlap over {overlap_lo:.0f}-{overlap_hi:.0f} mm3"
           if overlap_hi > overlap_lo else "ranges do NOT all overlap")
fig.suptitle("8_31_22 Acq3-6 (one pig): CC volume at EVERY painted timepoint. "
             "Red x = local peak below the signal floor, refinement fell back to the tracing.\n"
             "The pig is breathing, so no two acquisitions share a respiratory phase at any "
             f"given second — compare the per-acquisition mean and range (right), not one "
             f"instant.   {verdict}.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(f"{OUT}/cc_volume_timeseries.png", facecolor="white")
plt.close(fig)
print(f"\nsaved {OUT}/cc_volume_timeseries.png")

print("\nper-acquisition volume distribution (usable frames only) — THE comparable quantity:")
print(f"{'acq':<8}{'n':>3}{'V mean':>9}{'V min':>8}{'V max':>8}{'range':>8}{'CV %':>7}{'probe':>8}")
mus = []
for acq, (rows, secs, probe) in results.items():
    v = np.array([r["lumen_mm3"] for r in rows if r["usable"]])
    if v.size:
        mus.append(v.mean())
        print(f"{acq:<8}{v.size:>3}{v.mean():>9.0f}{v.min():>8.0f}{v.max():>8.0f}"
              f"{v.max() - v.min():>8.0f}{100 * v.std() / v.mean():>7.1f}{probe:>8.3f}")
mus = np.array(mus)
print(f"\nbetween-acquisition spread of the MEANS: {mus.mean():.0f} +/- {mus.std():.0f} mm3 "
      f"(CV {100 * mus.std() / mus.mean():.1f}%)")
