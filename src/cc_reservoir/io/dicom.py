"""Load a full DICOM series as an HU volume — the SOURCE of these swine dynamic scans.

The lab's MAT (`im_stack`) is only the caudal HALF (701 of 1401 slices = the abdomen); it cuts the
duct off at the diaphragm, dropping the cranial CC + the whole TD (thorax). The DICOM series 01..06
ARE the 6 dynamic frames at FULL coverage (1401 slices, 0.5 mm), and their HU is identical to the MAT
(verified MAT[:,:,k]==DICOM[k]). So we read the DICOM directly and keep the whole duct in frame.
"""
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydicom


def _read_slice(path, retries=4):
    """Read one DICOM's (sort_key, pixel_array). sort_key = ImagePositionPatient z (true spatial order),
    falling back to InstanceNumber. The filenames are NOT always zero-padded (2023 series run '1.dcm' ..
    '1301.dcm'), so a lexicographic filename sort scrambles the volume — sort by geometry instead.
    Retries on the transient SMB timeouts these shares throw."""
    for k in range(retries):
        try:
            ds = pydicom.dcmread(path)
            ipp = getattr(ds, "ImagePositionPatient", None)
            key = float(ipp[2]) if ipp is not None else float(getattr(ds, "InstanceNumber", 0))
            return key, ds.pixel_array
        except (OSError, TimeoutError):
            if k == retries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def load_dicom_series(series_dir, max_workers=32):
    """One DICOM series -> (X, Y, Z) HU volume, Z = slice axis caudal->cranial (sorted by spatial
    ImagePositionPatient z, NOT filename — 2023 filenames aren't zero-padded so a string sort scrambles
    them). Same orientation as load_mat_3d (the abdomen MAT is this volume's caudal slices). Reads in
    parallel (cost is per-file SMB latency, not CPU) with per-slice retry so an SMB hiccup doesn't kill it."""
    dcms = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dcms:
        raise FileNotFoundError(f"no DICOM in {series_dir}")
    ds0 = pydicom.dcmread(dcms[0], stop_before_pixels=True)
    slope = float(getattr(ds0, "RescaleSlope", 1)); intercept = float(getattr(ds0, "RescaleIntercept", 0))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        slices = list(ex.map(_read_slice, dcms))
    slices.sort(key=lambda s: s[0])                  # spatial z order, never the (possibly unpadded) filename
    arrs = [s[1] for s in slices]
    return np.stack(arrs, axis=2).astype(np.float32) * slope + intercept


def dicom_series_dirs(acq_dir):
    """The dynamic-frame DICOM series under an acquisition (DICOM/01, /02, ...), sorted = time order.
    Only subdirs that actually hold .dcm are returned: the archive has empty placeholder series dirs
    and note-named ones ("02- there is no V2") that would otherwise blow up load_dicom_series."""
    base = os.path.join(acq_dir, "DICOM")
    return [os.path.join(base, d) for d in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, d)) and glob.glob(os.path.join(base, d, "*.dcm"))]
