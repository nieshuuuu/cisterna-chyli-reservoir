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
from ..measure.extract import extract_curves, method_agreement, qa_pass, METHODS

OUT = "/tmp/cc_realdata"

def _acq_dir(spec):
    return os.path.join(ARCHIVE, spec.session, spec.subpath)

def _load_series(acq_dir, full_shape, lo, hi):
    """Load per-timepoint HU volumes from MAT/*01.mat and CROP each to the CC bbox
    (lo,hi) immediately. Cropping is essential: the measure/ dilations on a full
    512x512x701 volume take minutes and ~9 GB; on the cropped patch they are instant.
    LOGS every file it skips (fail-loud, Rule 8) instead of silently dropping it."""
    # MAT convention (verified): 0N01.mat = timepoint N's HU volume (co-shaped with the
    # mask); 0N02.mat is a second 700-slice recon. Load only the *01.mat timepoint volumes.
    mats = sorted(glob.glob(os.path.join(acq_dir, "MAT", "*01.mat")))
    series = []
    for i, m in enumerate(mats):
        try:
            v = load_hu_volume(m)
        except Exception as e:
            print(f"    skip {os.path.basename(m)}: load failed ({e})")
            continue
        if v.shape != full_shape:
            print(f"    skip {os.path.basename(m)}: shape {v.shape} != mask {full_shape}")
            continue
        series.append((float(i), crop_to_bbox(v, lo, hi)))   # keep only the CC patch
    return series

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for spec in WORKING_SET:
        acq = _acq_dir(spec)
        mask_full = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
        lo, hi = mask_bbox(mask_full, pad=20)   # generous: covers the 8mm search dilation + ring
        mask = crop_to_bbox(mask_full, lo, hi)
        series = _load_series(acq, mask_full.shape, lo, hi)
        if len(series) < 2:
            print(f"SKIP {spec.session}/{spec.subpath}: {len(series)} usable volumes")
            continue
        curves = extract_curves(series, mask, REAL_VOXEL_MM)
        agr = method_agreement(curves)
        rows.append(dict(session=spec.session, acq=spec.subpath, condition=spec.condition,
                         probe_flow=spec.probe_flow, n_tp=len(series),
                         peak_mass=max(curves["dilated+ring"]["mass"]),
                         containment_at_peak=agr["containment_at_peak"],
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
        flag = "" if qa_pass(r) else "  <-- CHECK (motion/contamination)"
        print(f"  {r['session']}/{r['acq']:<16} containment={r['containment_at_peak']:.2f} "
              f"ring/tp1={r['ring_vs_tp1_peak_ratio']:.2f}{flag}")

if __name__ == "__main__":
    main()
