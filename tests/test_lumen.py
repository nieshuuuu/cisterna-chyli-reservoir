# tests/test_lumen.py
import numpy as np

from cc_reservoir.measure.patch import CCPatch
from cc_reservoir.measure.lumen import (cc_td_boundary_z, segment_lumen, split_caudal,
                                        caliber, frame_metrics, fixed_lumen, fixed_split,
                                        fixed_frame_metrics, static_baseline, frame_enhancement,
                                        local_opacified_ref, occupancy_field, fwhm_lumen,
                                        centerline, equiv_radius, swept_tube,
                                        caliber_ratio_boundary_z, caudal_cistern_anchor,
                                        seedless_duct_seed, reclaim_axial_cistern)

VOX = (1.0, 1.0, 1.0)


def _disk(shape, cz, cx, cy, z, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _synthetic():
    """Single connected enhanced lumen: fat caudal segment (z<=16, r=3) + thin cranial
    segment (z>16, r=1.5), on uniform background. CC mask marks the caudal segment."""
    shape = (30, 30, 60)
    lum = np.zeros(shape, bool)
    for z in range(5, 56):
        r = 3.0 if z <= 16 else 1.5
        lum[..., z] |= _disk(shape, None, 15, 15, z, r)
    cc_mask = np.zeros(shape, bool); cc_mask[..., 5:17] = lum[..., 5:17]   # lab CC = caudal segment
    td_mask = np.zeros(shape, bool); td_mask[..., 17:56] = lum[..., 17:56]  # lab TD = cranial segment
    bg, hi = 50.0, 450.0
    peak = np.full(shape, bg); peak[lum] = hi
    early = np.full(shape, bg)                                             # no contrast yet
    return CCPatch(mask=cc_mask, td=td_mask, vols=[early, peak], peak_i=1,
                   voxel_mm=VOX, lo=(0, 0, 0), hi=shape), lum


def _synthetic_with_static_bone():
    """Same enhanced lumen, plus a STATIC bright bar (bone) inside the search ROI: bright in
    BOTH frames, so it never fills with contrast. A spatial ring-baseline keeps it (false
    second tube); temporal subtraction (peak - per-voxel-min) must cancel it."""
    patch, lum = _synthetic()
    bone = np.zeros(lum.shape, bool); bone[18:22, 18:22, 5:56] = True   # parallel bar, within 10mm
    bone_hu = 800.0
    for v in patch.vols:                                                # static: bright in every frame
        v[bone] = bone_hu
    return patch, lum, bone


def test_temporal_enhancement_cancels_static_structure():
    # the bug: spine/bone (bright but non-filling) leaked into the lumen. peak-min must kill it.
    patch, _, bone = _synthetic_with_static_bone()
    enh = frame_enhancement(patch, patch.peak_i)
    assert enh[bone].max() < 1e-6                                       # static bar contributes zero enhancement


def test_fixed_lumen_excludes_static_bright_structure():
    patch, lum, bone = _synthetic_with_static_bone()
    fx = fixed_lumen(patch)
    assert not np.any(fx & bone)                                        # no bone voxels segmented
    assert abs(int(fx.sum()) - int(lum.sum())) / lum.sum() < 0.15       # still recovers the duct


def test_boundary_is_caudal_extent_of_cc_mask():
    patch, _ = _synthetic()
    assert cc_td_boundary_z(patch) == 16          # cranial edge of the CC mask


def test_segment_and_split_partition_the_lumen():
    patch, lum = _synthetic()
    seg, _ = segment_lumen(patch, frame=1)
    assert abs(int(seg.sum()) - int(lum.sum())) / lum.sum() < 0.10   # recovers the enhanced lumen
    cc, td = split_caudal(seg, cc_td_boundary_z(patch))
    assert not np.any(cc & td)                                        # disjoint
    assert int((cc | td).sum()) == int(seg.sum())                    # cover the whole lumen


def test_caudal_segment_is_wider_and_volumes_add_up():
    patch, _ = _synthetic()
    m = frame_metrics(patch, frame=1, boundary_z=cc_td_boundary_z(patch))
    assert m["d_cc"] > m["d_td"]                                      # fat caudal vs thin cranial
    assert abs((m["V_cc"] + m["V_td"]) - m["V_total"]) < 1e-6
    assert m["V_cc"] > 0 and m["V_td"] > 0


def test_fixed_lumen_is_constant_extent_and_split_partitions():
    patch, lum = _synthetic()
    fx = fixed_lumen(patch)                                           # peak-frame temporal extent
    assert abs(int(fx.sum()) - int(lum.sum())) / lum.sum() < 0.12
    cc, td = fixed_split(patch, fx)
    assert not np.any(cc & td)
    assert int((cc | td).sum()) == int(fx.sum())


def test_integrated_hu_boundary_is_continuous_and_recovers_tube():
    patch, lum = _synthetic()
    enh = frame_enhancement(patch, patch.peak_i)
    master = fixed_lumen(patch)
    s_o = local_opacified_ref(enh, master)
    assert s_o.max() > 300                                            # local peak ~ the 400 HU enhancement
    occ = occupancy_field(enh, s_o)
    assert occ.min() >= 0.0 and occ.max() <= 1.0                      # occupancy is a 0..1 field
    seg = fwhm_lumen(enh, master, s_o)
    zc = np.where(seg.any(axis=(0, 1)))[0]
    gaps = sum(1 for z in range(zc.min(), zc.max() + 1) if not seg[:, :, z].any())
    assert gaps == 0                                                  # continuous: no empty cross-section
    assert abs(int(seg.sum()) - int(lum.sum())) / lum.sum() < 0.30    # recovers the tube


def test_swept_tube_is_gap_free_and_follows_caliber():
    patch, lum = _synthetic()
    master = fixed_lumen(patch)
    zc = np.where(master.any(axis=(0, 1)))[0]; zr = range(int(zc.min()), int(zc.max()) + 1)
    cx, cy = centerline(master)
    r = equiv_radius(master, zr)
    assert r[zr.start + 2] > r[zr.stop - 2]                           # caudal (r=3) wider than cranial (r=1.5)
    tube = swept_tube(cx, cy, r, zr, master.shape)
    zt = np.where(tube.any(axis=(0, 1)))[0]
    gaps = sum(1 for z in range(zt.min(), zt.max() + 1) if not tube[:, :, z].any())
    assert gaps == 0 and tube.sum() > 0                              # continuous, non-empty


def _tube(radii_by_z, shape=(30, 30, 60)):
    """Solid tube centered at (15,15): a disk of the given radius at each listed z."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    lum = np.zeros(shape, bool)
    for z, r in radii_by_z.items():
        if r > 0:
            lum[:, :, z] = (xx - 15) ** 2 + (yy - 15) ** 2 <= r ** 2
    return lum


def test_caliber_ratio_boundary_splits_at_caliber_step():
    # wide caudal sac (r=4 -> d~8mm, z5..16) stepping down to a narrow TD (r=1.5 -> d~3mm).
    lum = _tube({z: (4.0 if z <= 16 else 1.5) for z in range(5, 56)})
    zb, diag = caliber_ratio_boundary_z(lum, (1.0, 1.0, 1.0), ratio=2.0)
    assert diag["caudal_is_cc"]                       # caudal d ~8 > 2x TD d ~3
    assert 2.0 < diag["td_ref_mm"] < 4.0             # TD reference ~3 mm
    assert 13 <= zb <= 18                             # boundary near the step at z16 (smoothing slack)


def test_caliber_ratio_boundary_flags_narrow_caudal_cranial_ampulla():
    # the documented Acq16 disagreement: caudal "CC" (r=1.7 -> d~3.4mm) is NARROWER than a cranial
    # ampulla (r=2.35 -> d~4.7mm). The 200% rule must NOT flag the caudal sac as a CC.
    radii = {z: 1.2 for z in range(5, 56)}           # baseline TD ~2.4 mm
    for z in range(5, 13):
        radii[z] = 1.7                               # modest caudal sac
    for z in range(48, 53):
        radii[z] = 2.35                              # wider cranial ampulla
    _, diag = caliber_ratio_boundary_z(_tube(radii), (1.0, 1.0, 1.0), ratio=2.0)
    assert not diag["caudal_is_cc"]                   # caudal 3.4 mm is not > 2x the ~2.4 mm TD


def test_caudal_anchor_not_fooled_by_cranial_ampulla():
    # the Angio/Acq8,Acq12 failure: a cranial TD ampulla wider than the caudal cistern. A global-peak
    # area fraction trims the real caudal cistern and mislocates the CC onto the ampulla; the caudal
    # anchor must reference the CAUDAL region and land at the cistern near the caudal end.
    radii = {}
    for z in range(5, 11):
        radii[z] = 1.0                                  # thin feeding lumbar trunk (caudal tip)
    for z in range(11, 26):
        radii[z] = 3.0                                  # caudal CISTERN
    for z in range(26, 81):
        radii[z] = 1.2                                  # thin TD
    for z in range(81, 92):
        radii[z] = 5.0                                  # big CRANIAL ampulla (wider than the cistern)
    master = _tube(radii, shape=(40, 40, 100))
    cc_caud = caudal_cistern_anchor(master, (1.0, 1.0, 1.0))
    assert 10 <= cc_caud <= 16                          # lands at the caudal cistern, NOT up near z81


def test_seedless_duct_seed_zones_para_vertebral_excludes_lateral_blob():
    # anatomy: spine = posterior (high-x) central bone column; duct = long tube just ANTERIOR (lower x)
    # of it, LR-central, that FILLS; a lateral filling blob (kidney) sits off-midline. The zoned
    # seedless seed must pick the duct and EXCLUDE the lateral kidney (the v1 failure was grabbing it).
    shape = (40, 40, 100)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    spine = np.zeros(shape, bool); spine[30:35, 16:25, :] = True          # posterior central column
    duct = np.zeros(shape, bool)
    for z in range(5, 96):                                                # anterior (x~22), LR-center
        duct[:, :, z] = (xx - 22) ** 2 + (yy - 20) ** 2 <= 4
    kidney = np.zeros(shape, bool)
    for z in range(40, 70):                                               # lateral (y~36), filling blob
        kidney[:, :, z] = (xx - 25) ** 2 + (yy - 36) ** 2 <= 9
    vols = []
    for i in range(4):
        v = np.full(shape, -100.0)
        v[duct] = 250.0 + i * 350.0                                       # duct FILLS
        v[kidney] = 250.0 + i * 300.0                                     # kidney also fills (not bone)
        v[spine] = 950.0                                                  # static bright bone
        vols.append(v)
    seed = seedless_duct_seed(vols, (1.0, 1.0, 1.0), min_zextent=40, ant_mm=15.0, lat_mm=10.0)
    assert seed[duct].sum() > 0.5 * duct.sum()                            # recovers the duct
    assert seed[spine].sum() == 0                                         # static bone excluded
    assert seed[kidney].sum() == 0                                        # lateral kidney zoned OUT


def test_reclaim_axial_cistern_merges_on_axis_keeps_lateral():
    # the angiotensin CC/node conflation fix: node-labeled enhancement ON the duct axis is the CC
    # (reclaim into the duct); a LATERAL blob stays a node.
    shape = (40, 40, 60); yy, xx = np.ogrid[:shape[0], :shape[1]]
    master = np.zeros(shape, bool)
    for z in range(5, 55):
        master[:, :, z] = (xx - 20) ** 2 + (yy - 20) ** 2 <= 2          # thin duct axis at (20,20)
    on_axis = np.zeros(shape, bool); lateral = np.zeros(shape, bool)
    for z in range(20, 30):
        on_axis[:, :, z] = (xx - 22) ** 2 + (yy - 20) ** 2 <= 9         # cistern hugging the axis (~2mm off)
    for z in range(20, 28):
        lateral[:, :, z] = (xx - 20) ** 2 + (yy - 34) ** 2 <= 6         # node 14mm lateral of the axis
    nodes = on_axis | lateral
    m2, n2 = reclaim_axial_cistern(master, nodes, (1.0, 1.0, 1.0), r_mm=9.0)
    assert (m2 & on_axis).sum() > 0.5 * on_axis.sum()                   # on-axis cistern -> duct
    assert (n2 & lateral).sum() > 0.5 * lateral.sum()                   # lateral blob stays a node
    assert (m2 & lateral).sum() == 0                                    # lateral NOT merged into the duct


def test_fixed_frame_metrics_volume_stable_enhancement_rises():
    patch, _ = _synthetic()
    fx = fixed_lumen(patch); cc, td = fixed_split(patch, fx)
    m0 = fixed_frame_metrics(patch, 0, cc, td)                        # no contrast yet
    m1 = fixed_frame_metrics(patch, 1, cc, td)                        # opacified
    assert m1["E_cc"] > m0["E_cc"]                                    # enhancement washes in
    assert np.isfinite(m1["V_cc"]) and m1["V_cc"] > 0                 # volume measurable when opacified
