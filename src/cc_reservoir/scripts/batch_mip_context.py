"""Batch the whole-body MIP context build over EVERY seeded acquisition in the archive — the
all-dates extension of the single-acq build (build_mip_context).

Enumerates each acq dir that carries the lab volumetric seed (``SEGMENT_dcm/CC_dcm_01.mat``),
covering all three nesting schemes in one recursive glob: flat ``Acq#/``, ``Baseline|Angiotensin/Acq#/``,
and the double-nested 2023 ``AcqNN/AcqMM/``. Acqs with only a CC_MIP overlay (no volumetric mask) are
skipped — the seg pipeline is seed-driven and cannot locate the duct without the mask.

Each acq is built in a SUBPROCESS so its ~6x full-DICOM RAM is reclaimed before the next (the archive
is ~600 GB; one acq in memory at a time). Resumable: an acq whose ``context4d_<tag>/meta.npz`` already
exists is skipped. All ``context4d_*`` land in one out dir, so the MIP panel's acq dropdown lists every
date. Per-acq failures are logged and the batch continues, with a summary at the end.

  CC_ARCHIVE=<p> CC_CTX_OUT=<dir> PYTHONPATH=src \
    python3 -m cc_reservoir.scripts.batch_mip_context [--limit N] [--only SUBSTR] [--list]
"""
import glob
import os
import subprocess
import sys
import time

from cc_reservoir.io.workingset import ARCHIVE

OUT = os.environ.get("CC_CTX_OUT", "/tmp/cc_out_all")


def seeded_acqs(archive):
    """(session, subpath) for every acq dir holding SEGMENT_dcm/CC_dcm_01.mat AND a DICOM/ series.
    One recursive glob over <session>/**/SEGMENT_dcm/CC_dcm_01.mat handles all nesting depths."""
    out = []
    pat = os.path.join(archive, "*", "**", "SEGMENT_dcm", "CC_dcm_01.mat")
    for seed in sorted(glob.glob(pat, recursive=True)):
        acq = os.path.dirname(os.path.dirname(seed))            # .../<acq>/SEGMENT_dcm/CC_dcm_01.mat
        if not os.path.isdir(os.path.join(acq, "DICOM")):       # build needs the dynamic DICOM frames
            continue
        rel = os.path.relpath(acq, archive)
        session = rel.split(os.sep)[0]
        subpath = os.path.relpath(acq, os.path.join(archive, session))
        out.append((session, subpath))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    argv = sys.argv
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    if "--workingset" in argv:                                 # the already-analyzed acqs (SSoT list)
        from cc_reservoir.io.workingset import WORKING_SET
        acqs = [(s.session, s.subpath) for s in WORKING_SET]
    else:
        acqs = seeded_acqs(ARCHIVE)
    if only:
        acqs = [a for a in acqs if only in os.path.join(*a)]
    if limit:
        acqs = acqs[:limit]

    if "--list" in argv:                                        # dry run: just print what would build
        for session, sub in acqs:
            print(f"{session}/{sub}")
        print(f"\n{len(acqs)} seeded acqs", flush=True)
        return

    print(f"{len(acqs)} seeded acqs -> {OUT}", flush=True)
    ok, fail = [], []
    for i, (session, sub) in enumerate(acqs, 1):
        tag = f"{session}_{sub.replace('/', '_')}"
        if os.path.exists(os.path.join(OUT, f"context4d_{tag}", "meta.npz")):
            print(f"[{i}/{len(acqs)}] {tag}: exists, skip", flush=True)
            ok.append(tag)
            continue
        print(f"[{i}/{len(acqs)}] {tag}: building...", flush=True)
        env = {**os.environ, "CC_SESSION": session, "CC_SUB": sub, "CC_CTX_OUT": OUT, "PYTHONPATH": "src"}
        t0 = time.time()
        # one retry: the Mac SMB mount throws transient per-file timeouts under sustained DICOM reads;
        # a fresh subprocess often gets past a slice that timed out (a fully dead mount still fails -> logged).
        for attempt in (1, 2):
            r = subprocess.run([sys.executable, "-m", "cc_reservoir.scripts.build_mip_context"], env=env)
            if r.returncode == 0 or attempt == 2:
                break
            print(f"    rc={r.returncode}, retrying once...", flush=True)
        if r.returncode == 0:
            ok.append(tag)
            print(f"    done in {time.time() - t0:.0f}s", flush=True)
        else:
            fail.append(tag)
            print(f"    FAILED rc={r.returncode} (continuing)", flush=True)
    print(f"\nDONE: {len(ok)} ok, {len(fail)} failed", flush=True)
    if fail:
        print("failed: " + ", ".join(fail), flush=True)


if __name__ == "__main__":
    main()
