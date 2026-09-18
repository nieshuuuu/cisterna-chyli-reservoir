"""Backfill context->raw provenance into LEGACY meta.npz (contexts built before build_mip_context
started recording it). For each WORKING_SET acquisition it recomputes crop_lo/crop_hi/raw_full_shape
from the raw DICOM + lab seed (deterministic — the same bbox+pad logic as load_cc_td_patch_dicom),
VERIFIES the recomputed crop against the stored hu.shape, and only then writes the provenance keys
into meta.npz (all existing keys preserved atomically; f*.npz untouched). Idempotent: acqs that
already carry crop_lo are skipped.

MUST run WHERE THE RAW DICOM IS MOUNTED — i.e. an INTERACTIVE PC2 session (with Z: live), or the
Linux box. An SSH session on Windows does NOT inherit the interactive login's mapped drives.

  # interactive PC2 terminal (Z: reachable):
  set CC_ARCHIVE=Z:\\ImageData\\Lymph_Studies
  set PYTHONPATH=C:\\Users\\nies1\\cc_paint\\src
  python -m cc_reservoir.scripts.backfill_provenance C:\\Users\\nies1\\cc_paint\\data

If an acq reports a shape MISMATCH, it was built with non-default crop pads; set the matching
CC_XY_PAD / CC_Z_CAUDAL_PAD / CC_Z_CRANIAL_PAD and re-run (only the mismatched ones re-do).
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pydicom

# Allow running as a bare script (python …\scripts\backfill_provenance.py) with NO PYTHONPATH: put the
# package root (…/src) on sys.path. Harmless under `python -m` (already importable). Robust on Windows
# where `set PYTHONPATH=…` is shell-dependent (cmd vs PowerShell).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cc_reservoir.io import REAL_VOXEL_MM
from cc_reservoir.io.masks import load_cc_mask
from cc_reservoir.io.dicom import dicom_series_dirs
from cc_reservoir.io.workingset import WORKING_SET, ARCHIVE


def _crop_bounds(spec, archive):
    """(lo, hi, full_shape) exactly as load_cc_td_patch_dicom computes them, but WITHOUT loading HU
    pixels — full_shape comes from DICOM headers (Rows, Columns, #slices). Same env pads."""
    acq = os.path.join(archive, spec.session, spec.subpath)
    cc = load_cc_mask(os.path.join(acq, "SEGMENT_dcm", "CC_dcm_01.mat"))
    td_path = os.path.join(acq, "SEGMENT_dcm", "TD_dcm_01.mat")
    td = load_cc_mask(td_path) if os.path.exists(td_path) else np.zeros_like(cc)
    seed_abd = cc | td
    series = dicom_series_dirs(acq)
    if not series:
        raise FileNotFoundError(f"no DICOM series in {acq}")
    dcms = glob.glob(os.path.join(series[0], "*.dcm"))
    h = pydicom.dcmread(dcms[0], stop_before_pixels=True)
    full_shape = (int(h.Rows), int(h.Columns), len(dcms))          # matches load_dicom_series stacking
    xy = int(os.environ.get("CC_XY_PAD", 60))
    zc = int(os.environ.get("CC_Z_CAUDAL_PAD", 12))
    zr = int(os.environ.get("CC_Z_CRANIAL_PAD", 320))
    seed = np.zeros(full_shape, bool)
    seed[:, :, :seed_abd.shape[2]] = seed_abd                      # lab seed = the caudal (abdomen) slices
    X = np.where(seed.any(axis=(1, 2)))[0]
    Y = np.where(seed.any(axis=(0, 2)))[0]
    Z = np.where(seed.any(axis=(0, 1)))[0]
    lo = (max(int(X.min()) - xy, 0), max(int(Y.min()) - xy, 0), max(int(Z.min()) - zc, 0))
    hi = (min(int(X.max()) + xy + 1, full_shape[0]), min(int(Y.max()) + xy + 1, full_shape[1]),
          min(int(Z.max()) + zr + 1, full_shape[2]))
    return lo, hi, full_shape


def expected_context_shape(lo, hi, ds):
    """Shape after block_reduce((hi-lo), (ds0,ds1,1)) — block_reduce pads, so in-plane is ceil()."""
    return (math.ceil((hi[0] - lo[0]) / ds[0]), math.ceil((hi[1] - lo[1]) / ds[1]), hi[2] - lo[2])


def _atomic_savez(path, **arrays):
    tmp = os.path.join(os.path.dirname(path), "_meta_backfill_tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)                                          # never leave a half-written meta.npz


def dump_bounds(out_path, archive=ARCHIVE):
    """Compute crop bounds for every WORKING_SET acq from the raw and write them to JSON — WITHOUT
    touching any context. Run this WHERE THE RAW IS (e.g. the Mac with the share mounted); then apply
    with `backfill(..., bounds=<loaded json>)` where the CONTEXTS are (PC2). Splits the raw read from
    the meta write so the two can live on different machines."""
    res = {}
    for spec in WORKING_SET:
        tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
        try:
            lo, hi, full = _crop_bounds(spec, archive)
            res[tag] = {"lo": list(lo), "hi": list(hi), "full": list(full),
                        "session": spec.session, "sub": spec.subpath}
            print(f"[computed] {tag}: lo={lo} hi={hi} full={full}")
        except Exception as e:
            print(f"[ERR] {tag}: {type(e).__name__}: {e}")
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"\nwrote {len(res)}/{len(WORKING_SET)} bounds -> {out_path}")
    return res


def backfill(ctx_root, archive=ARCHIVE, bounds=None):
    """Write provenance into legacy meta.npz. If `bounds` (a {tag: {lo,hi,full,session,sub}} dict from
    dump_bounds) is given, use it and DON'T read the raw — so this half can run where only the contexts
    live. Otherwise recompute the crop from the raw here."""
    ok = present = warn = missing = err = 0
    for spec in WORKING_SET:
        tag = f"{spec.session}_{spec.subpath.replace('/', '_')}"
        d = os.path.join(ctx_root, f"context4d_{tag}")
        meta_p = os.path.join(d, "meta.npz")
        if not os.path.exists(meta_p):
            print(f"[skip] {tag}: no context dir under {ctx_root}"); missing += 1; continue
        meta = dict(np.load(meta_p, allow_pickle=True))
        if "crop_lo" in meta:
            print(f"[have] {tag}: provenance already present"); present += 1; continue
        fi = int(meta["peak_i"])
        hu_shape = tuple(int(x) for x in np.load(os.path.join(d, f"f{fi}.npz"))["hu"].shape)
        ds_i = int(round(float(meta["voxel"][0]) / REAL_VOXEL_MM[0]))
        ds = (ds_i, ds_i, 1)
        if bounds is not None:                                    # precomputed elsewhere (no raw here)
            if tag not in bounds:
                print(f"[skip] {tag}: not in --bounds json"); missing += 1; continue
            b = bounds[tag]; lo, hi, full = tuple(b["lo"]), tuple(b["hi"]), tuple(b["full"])
        else:
            try:                                                  # raw read here (per-acq isolation, so
                lo, hi, full = _crop_bounds(spec, archive)        # one flaky share read won't abort all)
            except Exception as e:
                print(f"[ERR]  {tag}: raw read failed - {type(e).__name__}: {e}"); err += 1; continue
        exp = expected_context_shape(lo, hi, ds)
        if exp != hu_shape:
            print(f"[WARN] {tag}: recomputed crop -> {exp} != stored hu {hu_shape} "
                  f"(ds={ds}, lo={lo}, hi={hi}); non-default pads — NOT writing. Set CC_*_PAD and re-run.")
            warn += 1; continue
        meta.update(crop_lo=np.array(lo, np.int32), crop_hi=np.array(hi, np.int32),
                    downsample=np.array(ds, np.int32), raw_full_shape=np.array(full, np.int32),
                    raw_session=spec.session, raw_sub=spec.subpath)
        _atomic_savez(meta_p, **meta)
        print(f"[ok]   {tag}: crop_lo={lo} ds={ds} raw_full={full}  (verified against hu {hu_shape})")
        ok += 1
    print(f"\ndone: {ok} written, {present} already-present, {warn} shape-mismatch, "
          f"{err} raw-read-error, {missing} missing")
    return warn == 0 and err == 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Backfill context->raw provenance into legacy meta.npz")
    ap.add_argument("ctx_root", nargs="?", help="the cc_contexts dir holding context4d_* subdirs")
    ap.add_argument("--archive", default=None,
                    help="raw DICOM root (default: $CC_ARCHIVE or the built-in mount path)")
    ap.add_argument("--dump-bounds", metavar="JSON",
                    help="compute crop bounds from the raw and write JSON; don't touch contexts "
                         "(run where the raw is mounted, e.g. the Mac)")
    ap.add_argument("--bounds", metavar="JSON",
                    help="apply precomputed crop bounds from JSON instead of reading the raw "
                         "(run where the contexts are, e.g. PC2)")
    a = ap.parse_args()
    if a.dump_bounds:
        dump_bounds(a.dump_bounds, archive=a.archive or ARCHIVE)
        sys.exit(0)
    if not a.ctx_root:
        ap.error("ctx_root is required (unless --dump-bounds)")
    b = json.load(open(a.bounds)) if a.bounds else None
    sys.exit(0 if backfill(a.ctx_root, archive=a.archive or ARCHIVE, bounds=b) else 1)
