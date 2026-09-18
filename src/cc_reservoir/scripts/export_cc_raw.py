"""Ship the CC and the raw CT around it off the raw-data host, on the RAW grid (run where the DICOM is).

The 3D render needs two things the context grid cannot give: the duct's true calibre (the CC is only
~6 raw voxels across in-plane -- the context 2x2 mean-pools that away) and its true CT number (every
context built before 2026-07-15 clips at 3071, while the CC core reaches ~4800). So crop the raw
around the CC instead of reading the context: the cistern is ~10 cm x ~1 cm, so its raw crop is a
few MB per frame even though the full series is 1.4 GB.

The seg is mapped onto the raw grid through the recorded crop provenance (measure.context), so each
context voxel becomes its 2x2 block -- the mask is still only known to that precision, but it now
lands on, and can be coloured by, full-resolution unclipped HU.

ONE bbox over all frames: the camera and the crop must not move as the duct fills.

Keeps the labels (1 CC, 2 TD) rather than a boolean: the render colours by CC/TD identity, and the
skill's rule is that the two never get unioned.

  CC_ARCHIVE=<archive> python -m cc_reservoir.scripts.export_cc_raw <context4d_dir> <out.npz>
"""
import os
import sys

import numpy as np

from cc_reservoir.io import REAL_VOXEL_MM
from cc_reservoir.measure.context import load_context_seg, place_seg_in_raw

PAD = np.array([24, 24, 30])          # raw voxels (~19 x 19 x 15 mm) of CT around the duct
DUCT = (1, 2)                         # 1 CC, 2 TD -- landmark labels (kidney) are dropped


def export(context_dir, out_npz):
    from cc_reservoir.io.dicom import load_dicom_series, dicom_series_dirs
    from cc_reservoir.io.workingset import ARCHIVE
    meta = np.load(os.path.join(context_dir, "meta.npz"), allow_pickle=True)
    nF = int(meta["n_frames"])
    acq = os.path.join(os.environ.get("CC_ARCHIVE") or ARCHIVE,
                       str(meta["raw_session"]), str(meta["raw_sub"]))
    dirs = dicom_series_dirs(acq)

    segs = []
    for i in range(nF):
        s = place_seg_in_raw(load_context_seg(context_dir, i), meta)
        segs.append(np.where(np.isin(s, DUCT), s, 0).astype(np.uint8))
    union = np.zeros(segs[0].shape, bool)
    for s in segs:
        union |= s > 0
    if not union.any():
        raise SystemExit("[export_cc_raw] no CC/TD in any frame")
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - PAD, 0)
    hi = np.minimum(idx.max(0) + PAD + 1, np.array(union.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    print(f"[export_cc_raw] duct bbox {idx.min(0)}..{idx.max(0)} -> crop "
          f"{tuple(int(b - a) for a, b in zip(lo, hi))} (raw voxels)")

    hu, seg = [], []
    for i in range(nF):
        raw = load_dicom_series(dirs[i])                  # full-res, unclipped
        hu.append(np.round(raw[sl]).astype(np.int16))     # slope 1 / intercept 0 -> HU are integers
        seg.append(segs[i][sl])
        msg = []
        for name, lid in (("CC", 1), ("TD", 2)):
            m = segs[i][sl] == lid
            msg.append(f"{name} {int(m.sum()):6d} vox  HU median "
                       f"{np.median(raw[sl][m]) if m.any() else float('nan'):7.1f}")
        print(f"  f{i}: " + " | ".join(msg))
        del raw
    np.savez_compressed(out_npz, hu=np.stack(hu), seg=np.stack(seg),
                        voxel=np.asarray(REAL_VOXEL_MM, float),
                        tag=os.path.basename(context_dir.rstrip("/\\")).replace("context4d_", ""),
                        raw_session=meta["raw_session"], raw_sub=meta["raw_sub"])
    print(f"[export_cc_raw] wrote {out_npz}  hu{np.stack(hu).shape}  "
          f"{os.path.getsize(out_npz) / 1e6:.0f} MB")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    export(a[0], a[1])
