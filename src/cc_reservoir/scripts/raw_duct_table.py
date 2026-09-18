"""ONE per-acq table: duct volume + raw CT number per frame, straight off the full-res DICOM.

Replaces the volume.csv / tdc.csv pair. Those two split one table in half and, worse, their HU
columns were not CT numbers at all: they held TEMPORAL ENHANCEMENT (frame HU minus a per-voxel
min-over-frames baseline) measured on the CONTEXT grid -- which is in-plane 2x2 MEAN-pooled and, in
every context built before 2026-07-15, clipped at 3071 while the raw duct core reaches ~4800. So the
number was the wrong quantity, off the wrong grid, with its bright tail cut off.

This reads the raw series named by meta.raw_session/raw_sub and measures under the hand seg mapped
back through the recorded crop provenance (measure.context) -- full resolution, unclipped, no rebuild.

PER-FRAME SEG, not a union over frames: CC and TD bump into each other, so the CC/TD boundary a
painter draws moves frame to frame. Unioning over frames puts those boundary voxels in BOTH masters
(double-counting them) and dilutes early-frame HU with voxels that are not opacified yet. Each frame
is measured inside exactly the seg that was painted on it.

RUNS WHERE THE RAW DICOM IS (pc2: C:\\Users\\nies1\\Documents\\Lymph_Studies), not on the render box.

  CC_ARCHIVE=<archive> python -m cc_reservoir.scripts.raw_duct_table <context4d_dir> <out_dir>
  python -m cc_reservoir.scripts.raw_duct_table --selftest
"""
import csv
import os
import sys

import numpy as np

from cc_reservoir.io import REAL_VOXEL_MM
from cc_reservoir.measure.context import load_context_seg, place_seg_in_raw

CC, TD = 1, 2
LABELS = (("cc", CC), ("td", TD))


def duct_stats(raw, raw_mask, label, voxel_mm=REAL_VOXEL_MM):
    """Volume (uL) + raw CT number stats inside one label. Pure: arrays in, numbers out.

    Both mean and median are reported. They disagree a lot here and the gap is the point: the duct's
    HU distribution is heavily right-skewed (a bright Lipiodol core) and its boundary voxels are
    partial-volume rim that sits far darker than the lumen. The mean chases both tails; the median is
    what a reader would call the duct's CT number.
    """
    m = raw_mask == label
    n = int(m.sum())
    if not n:
        return dict(n=0, V_uL=0.0, mean=float("nan"), median=float("nan"))
    v = raw[m]
    # each raw voxel is one sample of the same physical block the painter marked, so volume is just
    # the count -- the seg carries no sub-voxel boundary to correct toward.
    return dict(n=n, V_uL=n * float(np.prod(voxel_mm)), mean=float(v.mean()), median=float(np.median(v)))


def acq_table(context_dir, archive=None):
    """Per-frame rows for one acquisition, read off the raw DICOM under that frame's hand seg."""
    from cc_reservoir.io.dicom import load_dicom_series, dicom_series_dirs
    from cc_reservoir.io.workingset import ARCHIVE
    meta = np.load(os.path.join(context_dir, "meta.npz"), allow_pickle=True)
    acq = os.path.join(archive or ARCHIVE, str(meta["raw_session"]), str(meta["raw_sub"]))
    dirs = dicom_series_dirs(acq)
    nF = int(meta["n_frames"])
    if len(dirs) < nF:
        raise SystemExit(f"[raw_duct_table] {acq}: {len(dirs)} DICOM series for {nF} context frames")

    rows = []
    for i in range(nF):
        seg = load_context_seg(context_dir, i)            # the hand-edit sidecar wins
        raw = load_dicom_series(dirs[i])                  # full-res, unclipped, SAME frame
        raw_mask = place_seg_in_raw(seg, meta)            # provenance-exact, no rebuild
        st = {tag: duct_stats(raw, raw_mask, lid) for tag, lid in LABELS}
        rows.append(dict(
            frame=i,
            V_cc_uL=round(st["cc"]["V_uL"], 1), V_td_uL=round(st["td"]["V_uL"], 1),
            HU_cc_mean=round(st["cc"]["mean"], 1), HU_cc_median=round(st["cc"]["median"], 1),
            HU_td_mean=round(st["td"]["mean"], 1), HU_td_median=round(st["td"]["median"], 1)))
        print(f"  f{i}: CC {st['cc']['n']:6d} vox  V {st['cc']['V_uL']:7.1f} uL  "
              f"HU median {st['cc']['median']:7.1f} mean {st['cc']['mean']:7.1f}  |  "
              f"TD {st['td']['n']:6d} vox  V {st['td']['V_uL']:7.1f} uL  "
              f"HU median {st['td']['median']:7.1f} mean {st['td']['mean']:7.1f}")
        del raw, raw_mask
    return rows


FIELDS = ["frame", "V_cc_uL", "V_td_uL", "HU_cc_mean", "HU_cc_median", "HU_td_mean", "HU_td_median"]


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _selftest():
    """A known cylinder: volume is the voxel count, median ignores the skew the mean chases."""
    vox = (0.781, 0.781, 0.5)
    raw = np.zeros((10, 10, 10), np.float32)
    mask = np.zeros((10, 10, 10), np.uint8)
    mask[2:5, 2:5, 2:8] = CC                              # 3*3*6 = 54 voxels
    raw[mask == CC] = 800.0
    raw[2, 2, 2] = 4795.0                                 # one bright core voxel (the Lipiodol tail)
    st = duct_stats(raw, mask, CC, vox)
    assert st["n"] == 54, st["n"]
    assert abs(st["V_uL"] - 54 * float(np.prod(vox))) < 1e-6, st["V_uL"]
    assert st["median"] == 800.0, f"median must ignore the bright tail, got {st['median']}"
    assert st["mean"] > 800.0, "mean must be dragged up by the tail"
    empty = duct_stats(raw, mask, TD, vox)
    assert empty["n"] == 0 and empty["V_uL"] == 0.0 and np.isnan(empty["median"]), empty
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        ctx, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
        os.makedirs(out, exist_ok=True)
        tag = os.path.basename(ctx.rstrip("/\\")).replace("context4d_", "")
        print(f"[raw_duct_table] {tag}")
        r = acq_table(ctx)
        p = os.path.join(out, "duct.csv")
        write_csv(r, p)
        print(f"[raw_duct_table] wrote {p}")
