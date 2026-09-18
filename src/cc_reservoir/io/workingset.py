import os, glob
from collections import namedtuple
import pydicom

AcqSpec = namedtuple("AcqSpec", "session subpath condition probe_flow ct_flow")

ARCHIVE = os.environ.get("CC_ARCHIVE", "/Volumes/Molloilab/ImageData/Lymph_Studies")

# From the 2026-06-16 audit: acquisitions with CC mask + curves + flow.
WORKING_SET = [
    AcqSpec("07_20_22_data", "Baseline/Acq6",     "baseline",    2.975, 2.946),
    AcqSpec("07_20_22_data", "Baseline/Acq7",     "baseline",    2.952, 2.922),
    AcqSpec("07_20_22_data", "Baseline/Acq10",    "baseline",    1.981, 2.021),
    AcqSpec("07_20_22_data", "Angiotensin/Acq8",  "angiotensin", 3.154, 3.051),
    AcqSpec("07_20_22_data", "Angiotensin/Acq9",  "angiotensin", 2.981, 2.994),
    AcqSpec("07_20_22_data", "Angiotensin/Acq12", "angiotensin", 3.562, 3.592),
    AcqSpec("8_31_22_data",  "Acq3",              "baseline",    3.058, 2.942),
    AcqSpec("8_31_22_data",  "Acq4",              "baseline",    2.531, 2.455),
    AcqSpec("8_31_22_data",  "Acq5",              "baseline",    2.810, 2.923),
    AcqSpec("09_07_22_data", "Acq16",             "baseline",    3.005, 2.881),
    AcqSpec("09_07_22_data", "Acq17",             "baseline",    3.116, 2.983),
]

# --- Hand-painted set -------------------------------------------------------------------
# Acquisitions whose CC/TD masks come from the iPad painter (`f<N>_edit.npz` sidecars in
# cc_contexts_fullres), not from the lab's Vitrea `.mat` seeds. 8_31_22 Acq6 has NO lab mask
# at all, so it exists only here.
#
# NO ct_flow FIELD, DELIBERATELY. The probe (Transonic T208) reading is the reference standard,
# and it is the only flow value carried by this set. Anything needing a flow value takes probe_flow.
#
# cc_z_min: raw-volume z index below which painted CC voxels are dropped, i.e. a caudal cutoff
# imposed after the fact to make the CC/lumbar-trunk boundary consistent across acquisitions.
# None = take the tracing as drawn.
#
# 8_31_22 Acq3-6 are ONE pig, one anaesthesia, 15:29-16:59 — same table position, so raw z is
# directly comparable between them. Measured cranial (CC/TD) ends agree: z 547/560/548/547.
# The caudal ends do not: 486/489/476, but Acq5 at 461, i.e. 11.5 mm below the other three's
# mean of 484. That extra caudal segment is the whole of Acq5's excess volume, and it is a
# tracing-convention difference rather than anatomy — the CC's caudal limit (where the lumbar
# trunks converge) has no crisp edge, so it is easy to carry the tracing further down.
# Operator's call 2026-07-27: bring Acq5 into line with the other three.
# cc_z_max: the mirror image, a CRANIAL cutoff. On 8_31_22 Acq6 the CC label was carried up the
# thoracic duct on three of five frames — f1 sits entirely at raw z 883-1015 in 297 fragments with
# ZERO overlap with the CC, and f2/f3 hold the CC (80-88% overlap with f4) plus that same duct
# segment. Acq6's own clean frames put the CC/TD boundary at z 547/546, so 547 keeps each frame's
# CC and drops what belongs to the TD. This is a label-extent correction, not a data rejection:
# f2/f3 become usable, and only f1 — which contains no CC voxels at all — stays out.
PaintedSpec = namedtuple("PaintedSpec",
                         "context session subpath condition probe_flow cc_z_min cc_z_max",
                         defaults=(None, None))

PAINTED_SET = [
    PaintedSpec("context4d_8_31_22_data_Acq3", "8_31_22_data", "Acq3", "baseline", 3.058),
    PaintedSpec("context4d_8_31_22_data_Acq4", "8_31_22_data", "Acq4", "baseline", 2.531,
                cc_z_min=485),              # f2/f3 carry the label down to z 372 (lumbar trunks);
                                            # the clean frames f1/f4/f5 start at 485/489/491
    PaintedSpec("context4d_8_31_22_data_Acq5", "8_31_22_data", "Acq5", "baseline", 2.810,
                cc_z_min=484),              # mean caudal end of Acq3/4/6
    PaintedSpec("context4d_8_31_22_data_Acq6", "8_31_22_data", "Acq6", "baseline", 2.935,
                cc_z_min=476,               # f2/f3 also run ~8 mm caudal of the clean f4/f5
                cc_z_max=547),              # CC/TD boundary from Acq6's own clean frames f4/f5
]

CONTEXTS = os.environ.get("CC_CONTEXTS", r"C:\Storage\CisternaChyli\cc_contexts_fullres")


def _first_dicom(folder):
    # Tolerant by design: a timepoint folder may hold non-DICOM siblings; we scan
    # for the first readable DICOM header. Returning None (no DICOM) is handled by
    # the caller, so this is boundary tolerance, not a swallowed error.
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        try:
            return pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
    return None

def resolve_timepoints(dicom_dir, min_timepoints=5, require_complete=False):
    """Return per-timepoint subfolder paths, deduped by SOPInstanceUID and ordered by
    AcquisitionTime. Raises if fewer than min_timepoints when require_complete."""
    subdirs = [d for d in sorted(glob.glob(os.path.join(dicom_dir, "*"))) if os.path.isdir(d)]
    seen, items = set(), []
    for d in subdirs:
        ds = _first_dicom(d)
        if ds is None:
            continue
        sop = str(getattr(ds, "SOPInstanceUID", d))
        if sop in seen:
            continue
        seen.add(sop)
        items.append((str(getattr(ds, "AcquisitionTime", "")), d))
    items.sort(key=lambda x: x[0])
    paths = [d for _, d in items]
    if require_complete and len(paths) < min_timepoints:
        raise ValueError(f"{dicom_dir}: only {len(paths)} timepoints (<{min_timepoints})")
    return paths
