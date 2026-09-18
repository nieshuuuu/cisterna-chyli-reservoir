"""Invariants for the HU landmark masks (measure/landmarks.py)."""
import numpy as np

from cc_reservoir.measure.landmarks import bone_mask, kidney_mask


def test_bone_keeps_dense_drops_soft():
    """Bone mask selects a dense (>=250 HU) blob and rejects soft tissue at the same location."""
    vol = np.full((40, 40, 40), 40, np.int16)          # soft-tissue background
    vol[10:25, 10:25, 10:25] = 600                      # a dense block (bone-like)
    m = bone_mask(vol)
    assert m[17, 17, 17]                                # inside the dense block -> bone
    assert not m[2, 2, 2]                               # soft tissue -> not bone


def test_bone_removes_speckle():
    """A single dense voxel is below min_vox and must be dropped (no speckle survives)."""
    vol = np.full((30, 30, 30), 40, np.int16)
    vol[15, 15, 15] = 800
    assert bone_mask(vol).sum() == 0


def test_kidney_is_posterior_and_bounded():
    """Kidney heuristic keeps only posterior (high-x) blobs and never the anterior duplicate."""
    vol = np.full((40, 40, 40), -200, np.int16)         # mostly air/fat (outside HU window)
    vol[28:36, 12:22, 14:30] = 60                       # posterior soft-tissue blob (kidney-like)
    vol[4:12, 12:22, 14:30] = 60                        # anterior blob (must be rejected)
    m = kidney_mask(vol, z_lo=14, z_hi=30, min_vox=200)
    assert m[31, 17, 20]                                # posterior blob kept
    assert not m[7, 17, 20]                             # anterior blob dropped
    assert m.sum() > 0


def test_kidney_respects_z_band():
    """A blob entirely outside the duct z-band is excluded."""
    vol = np.full((40, 40, 40), -200, np.int16)
    vol[28:36, 12:22, 2:8] = 60                         # posterior blob, but low z
    m = kidney_mask(vol, z_lo=20, z_hi=35, min_vox=200)
    assert m.sum() == 0
