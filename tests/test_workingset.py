import numpy as np, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, UID
from cc_reservoir.io.workingset import resolve_timepoints, WORKING_SET, AcqSpec

_CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage

def _write_slice(path, sop, acq_time, inst):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(_CT_SOP_CLASS)
    meta.MediaStorageSOPInstanceUID = UID("1.2.3.4.5." + str(abs(hash(sop)) % 10**12))
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    ds.SOPClassUID = UID(_CT_SOP_CLASS)
    ds.SOPInstanceUID = UID("1.2.3.4.5." + str(abs(hash(sop)) % 10**12))
    ds.AcquisitionTime = acq_time; ds.InstanceNumber = inst
    # Store sop as a plain string tag (private tag not needed; we use SOPInstanceUID for dedup)
    # We need a stable sop string per folder for the test; use SeriesDescription as proxy
    ds.SeriesDescription = sop
    ds.save_as(str(path), enforce_file_format=True)

def test_resolve_dedups_and_sorts(tmp_path):
    # two timepoint folders out of order; one is a duplicate (same SOP) of the other
    _sop_map = {}
    for tp, at in [("02", "120000"), ("01", "120100")]:  # folder names not in time order
        d = tmp_path / tp; d.mkdir()
        uid = UID("1.2.3.4.5." + str(abs(hash(f"sop-{tp}")) % 10**12))
        _sop_map[tp] = uid
        # Write a DICOM with a unique SOPInstanceUID per folder
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = UID(_CT_SOP_CLASS)
        meta.MediaStorageSOPInstanceUID = uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(d / "s1.dcm"), {}, file_meta=meta, preamble=b"\0"*128)
        ds.SOPClassUID = UID(_CT_SOP_CLASS)
        ds.SOPInstanceUID = uid
        ds.AcquisitionTime = at; ds.InstanceNumber = 1
        ds.save_as(str(d / "s1.dcm"), enforce_file_format=True)
    # Write a duplicate of folder 02 into folder 03 (same SOPInstanceUID)
    dup = tmp_path / "03"; dup.mkdir()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(_CT_SOP_CLASS)
    meta.MediaStorageSOPInstanceUID = _sop_map["02"]
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(dup / "s1.dcm"), {}, file_meta=meta, preamble=b"\0"*128)
    ds.SOPClassUID = UID(_CT_SOP_CLASS)
    ds.SOPInstanceUID = _sop_map["02"]  # same UID as folder 02 → duplicate
    ds.AcquisitionTime = "120000"; ds.InstanceNumber = 1
    ds.save_as(str(dup / "s1.dcm"), enforce_file_format=True)
    tps = resolve_timepoints(str(tmp_path))
    # deduped to 2 unique, ordered by AcquisitionTime
    assert len(tps) == 2
    assert tps[0].endswith("02") and tps[1].endswith("01")

def test_working_set_is_nonempty_and_typed():
    assert len(WORKING_SET) >= 11
    a = WORKING_SET[0]
    assert isinstance(a, AcqSpec) and a.condition in {"baseline", "angiotensin"}
