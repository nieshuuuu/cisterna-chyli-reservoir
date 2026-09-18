"""Per-timepoint CC+TD lumen segmentation and caudal-anchored CC/TD split — single producer.

Why this exists (2026-06-17 review of Acq16):
- Enhancement is TEMPORAL (DSA-style): a voxel's HU at a frame minus its per-voxel min over
  the whole dynamic series (its pre-arrival/static value). A voxel belongs to the lumen only
  if it FILLED with contrast. A spatial ring-baseline does NOT cancel static bright structures
  sitting inside the ROI — the spine (~420 HU, unchanging) survived the threshold and segmented
  as a false SECOND tube. Subtracting the per-voxel static baseline cancels bone exactly.
- The threshold is ABSOLUTE (``ABS_HU``), sitting above the residual motion-induced halo
  (~100-150 HU) and below the duct, whose dimmest cross-section still peaks ~280 HU. A
  half-max / FWHM threshold relative to a global peak would UNDER-segment diluted segments.
- Respiration changes the CC frame to frame, so each timepoint is segmented INDEPENDENTLY and
  V(t)/caliber(t) are reported per frame (the variation is signal, not noise).
- CC vs TD is anchored ANATOMICALLY at the caudal end (DICOM: higher MAT z = cranial). The
  CC is the caudal segment up to ``boundary_z``; the TD is everything ascending cranially
  (including a cranial terminal ampulla, which in some animals is wider than the CC itself).
  Boundary defaults to the cranial extent of the lab CC mask (caudal-anchored).
"""
import numpy as np
from scipy.ndimage import (label, distance_transform_edt, binary_closing, binary_erosion,
                           binary_fill_holes, generate_binary_structure, uniform_filter1d,
                           center_of_mass)
from skimage.restoration import denoise_tv_chambolle

from cc_reservoir.measure.roi import dilated_mask_roi
from cc_reservoir.measure.volume import integrated_hu_volume_region

ABS_HU = 200.0          # absolute enhancement threshold (HU): above the motion halo, below the duct
_ST = generate_binary_structure(3, 2)


def _largest_cc(binary):
    lab, nl = label(binary)
    if nl <= 1:
        return binary
    return lab == (1 + np.argmax([(lab == i).sum() for i in range(1, nl + 1)]))


def static_baseline(patch):
    """Per-voxel pre-contrast/static HU = min over all frames. A voxel's lowest value over the
    bolus passage is its pre-arrival tissue value; for a non-filling structure (bone) the min
    equals its constant value, so temporal subtraction cancels it exactly."""
    return np.stack(patch.vols).min(axis=0)


def lumen_roi(patch, search_mm=10.0, fixed_dil_mm=2.0):
    """(search, fixed) ROIs around the CC|TD union for segmentation + background."""
    union = patch.mask | patch.td
    return (dilated_mask_roi(union, search_mm, patch.voxel_mm),
            dilated_mask_roi(union, fixed_dil_mm, patch.voxel_mm))


def segment_lumen(patch, frame, rois=None, abs_hu=ABS_HU, baseline=None):
    """Contrast lumen at one frame: largest connected component of (temporal enhancement >=
    abs_hu) within the search ROI, lightly closed. Returns (lumen, enh)."""
    rois = rois or lumen_roi(patch)
    enh = frame_enhancement(patch, frame, rois, baseline)
    return _largest_cc(binary_closing(enh >= abs_hu, _ST, 1)), enh


def cc_td_boundary_z(patch):
    """Caudal-anchored CC/TD boundary z-index = cranial extent of the lab CC mask.
    caudal = LOW z, cranial = HIGH z (DICOM-confirmed). CC is z<=boundary, TD is z>boundary."""
    cc_z = np.where(patch.mask.any(axis=(0, 1)))[0]
    return int(cc_z.max()) if cc_z.size else patch.mask.shape[2] // 2


def split_caudal(lumen, boundary_z):
    """Split a lumen mask into (cc_mask, td_mask): CC = caudal (z<=boundary), TD = cranial."""
    cc = lumen.copy(); cc[:, :, boundary_z + 1:] = False
    td = lumen.copy(); td[:, :, :boundary_z + 1] = False
    return cc, td


def caliber(mask, voxel_mm):
    """Maximal inscribed diameter (mm) = 2 x max EDT — orientation-robust caliber."""
    if mask.sum() == 0:
        return 0.0
    return float(2 * distance_transform_edt(mask, sampling=voxel_mm).max())


def per_z_caliber(mask, voxel_mm, smooth_z=5):
    """Per-z maximal inscribed lumen diameter (mm) = 2 x max in-slice EDT, smoothed along z.
    The caliber PROFILE along the duct (caudal CC = wide, cranial TD = narrow); orientation-robust
    because the inscribed-disk radius does not care how the cross-section is tilted in the slice."""
    nz = mask.shape[2]
    d = np.zeros(nz)
    for z in range(nz):
        m = mask[:, :, z]
        if m.any():
            d[z] = 2.0 * distance_transform_edt(m, sampling=voxel_mm[:2]).max()
    return uniform_filter1d(d, smooth_z) if smooth_z > 1 else d


def caudal_cistern_anchor(master, voxel_mm, frac=0.35, caud_frac=0.33, max_cm=8.0, smooth_z=5):
    """Caudal CC anchor (the cranial extent of the thin feeding lumbar trunk to drop) that is ROBUST
    to a cranial TD ampulla. The CC is the caudal abdominal origin of the duct; the only thing to trim
    below it is the thin ascending lumbar trunk that feeds the cistern (a caliber step-up marks the
    confluence). Reference the CAUDAL region's OWN peak area (the caudal ``caud_frac`` of the duct,
    capped at ``max_cm``), NOT the global maximum: when a cranial ampulla is wider than the caudal
    cistern, a global-peak fraction raises the bar so high that the real caudal cistern falls below it
    and gets trimmed away, mislocating the CC onto the ampulla (the Angio/Acq8, Acq12 failure)."""
    zc = np.where(master.any(axis=(0, 1)))[0]
    if not zc.size:
        return 0
    z_caud, z_cran = int(zc.min()), int(zc.max())
    area = uniform_filter1d(master.sum(axis=(0, 1)).astype(float), smooth_z)
    z_hi = min(z_cran, z_caud + min(int(caud_frac * (z_cran - z_caud)),
                                    int(round(max_cm * 10.0 / voxel_mm[2]))))
    caud_peak = float(area[z_caud:z_hi + 1].max()) if z_hi >= z_caud else float(area[z_caud])
    above = np.where(area[:z_hi + 1] >= frac * caud_peak)[0]
    above = above[above >= z_caud]
    return int(above.min()) if above.size else z_caud


def caliber_ratio_boundary_z(mask, voxel_mm, ratio=2.0, z_start=None, td_window_frac=0.4, smooth_z=5):
    """Objective Loukas-2007 CC/TD boundary by caliber RATIO — the first *objective* CC criterion
    (Clinical Anatomy 2007) and fixation-robust because it is a ratio, not an absolute diameter:
    the CC is the contiguous CAUDAL run of slices whose lumen diameter exceeds ``ratio`` x the mean
    TD diameter. Drop-in replacement for the arbitrary fixed-length caudal anchor.

    Returns ``(boundary_z, diag)``. ``boundary_z`` is the cranial extent of the caudal CC run
    (CC = z <= boundary_z). The TD reference diameter is the median caliber over the cranial
    ``td_window_frac`` of the duct's z-extent (unambiguously TD — well above any plausible CC).
    ``z_start`` starts the caudal run above the noise tip (pass the pipeline's ``cc_caud``).

    ``diag`` carries td_ref_mm, the caudal caliber, the CC length in mm, and ``caudal_is_cc``:
    when the caudal end does NOT exceed the ratio (the documented Acq16 case — a cranial ampulla
    wider than the caudal sac), the rule flags NO caudal CC and falls back to ``z_start``. That
    disagreement with the length-anchor is the finding to report, not something to tune away."""
    zc = np.where(mask.any(axis=(0, 1)))[0]
    if not zc.size:
        return 0, dict(td_ref_mm=0.0, caudal_mm=0.0, cc_len_mm=0.0, caudal_is_cc=False)
    z_caud, z_cran = int(zc.min()), int(zc.max())
    z0 = z_caud if z_start is None else int(np.clip(z_start, z_caud, z_cran))
    d = per_z_caliber(mask, voxel_mm, smooth_z)
    # TD reference = median caliber over the cranial window (clearly TD, narrow tube)
    w0 = max(int(z_cran - td_window_frac * (z_cran - z_caud)), z0 + 1)
    win = d[w0:z_cran + 1]
    td_ref = float(np.median(win[win > 0])) if (win > 0).any() else 0.0
    thr = ratio * td_ref
    # representative caudal caliber = PEAK over the caudal quarter (the single smoothed tip slice is
    # edge-deflated by the z-smoothing; the sac's true width is its local peak, not its endpoint)
    caud_hi = z0 + max(3, int(0.25 * (z_cran - z0)))
    caud = d[z0:caud_hi + 1]
    caudal_mm = float(caud.max()) if caud.size else 0.0
    caudal_is_cc = bool(td_ref > 0 and caudal_mm > thr)
    if caudal_is_cc:                                    # boundary = cranial top of the wide run rooted at the sac
        z = z0 + int(np.argmax(caud))
        while z + 1 <= z_cran and d[z + 1] > thr:
            z += 1
        boundary = z
    else:
        boundary = z0
    return boundary, dict(td_ref_mm=td_ref, caudal_mm=caudal_mm,
                          cc_len_mm=float((boundary - z0) * voxel_mm[2]), caudal_is_cc=caudal_is_cc)


def frame_enhancement(patch, frame, rois=None, baseline=None):
    """TEMPORAL (DSA-style) enhancement at one frame: frame HU minus the per-voxel static
    baseline (min over time), within the search ROI. Cancels bone and any non-filling
    structure — a spatial ring-baseline would leave static bright structures (spine) inside
    the ROI, which then segment as a false second tube. ``baseline`` may be precomputed and
    passed in to avoid recomputing the per-voxel min on every frame."""
    search, _ = rois or lumen_roi(patch)
    base = baseline if baseline is not None else static_baseline(patch)
    return np.where(search, patch.vols[frame] - base, 0.0)


def fixed_lumen(patch, rois=None, abs_hu=ABS_HU, baseline=None):
    """The FIXED anatomical lumen extent at the PEAK frame, tightened to the TV-FWHM boundary.
    The envelope (largest connected enh >= abs_hu, lightly closed) bounds the duct; tv_lumen then
    pulls the boundary in to the half-max of the TV-denoised enhancement, so the displayed extent
    hugs the opacified lumen instead of the partial-volume halo (the old hard-threshold extent
    over-segmented that bloom). Constant across timepoints: the duct does not change length, only
    the contrast inside it does. Peak frame = most fully opacified (both CC and TD), clean single
    component; max-over-time would also capture per-frame noise spikes."""
    rois = rois or lumen_roi(patch)
    base = baseline if baseline is not None else static_baseline(patch)
    enh_peak = frame_enhancement(patch, patch.peak_i, rois, base)
    envelope = _largest_cc(binary_closing(enh_peak >= abs_hu, _ST, 1))
    return tv_lumen(enh_peak, envelope)


def fixed_split(patch, fixed=None, boundary_z=None):
    """Caudal-anchored split of the fixed lumen into (cc_mask, td_mask)."""
    fixed = fixed if fixed is not None else fixed_lumen(patch)
    zb = boundary_z if boundary_z is not None else cc_td_boundary_z(patch)
    return split_caudal(fixed, zb)


def fixed_frame_metrics(patch, frame, cc_fix, td_fix, rois=None, baseline=None):
    """Integrated-HU contrast volume + mean temporal enhancement E in the FIXED CC and TD at
    one frame. V is the contrast-filled (PV-corrected) volume; it rises as the fixed shape
    fills and is concentration-independent once opacified. Returns a dict."""
    enh = frame_enhancement(patch, frame, rois, baseline)
    v, vox = patch.vols[frame], patch.voxel_mm
    Vcc, so_cc, sb_cc = integrated_hu_volume_region(cc_fix, v, vox)
    Vtd, so_td, sb_td = integrated_hu_volume_region(td_fix, v, vox)
    return dict(frame=frame, V_cc=Vcc, V_td=Vtd,
                E_cc=float(enh[cc_fix].mean()) if cc_fix.any() else 0.0,
                E_td=float(enh[td_fix].mean()) if td_fix.any() else 0.0,
                s_o_cc=so_cc, s_o_td=so_td)


def local_opacified_ref(enh, master, smooth_z=3, top_k=3):
    """Per-z opacified-lumen peak S_O(z) for the integrated-HU / FWHM boundary: the top-k mean
    of the PV-free (eroded) master core at each axial level, smoothed along z. Concentration
    varies along the duct, so the boundary uses the LOCAL peak, not one global value (a global
    half-max under-segments the dilute segments and over-segments the concentrated ones)."""
    nz = master.shape[2]
    so = np.zeros(nz)
    for z in range(nz):
        m = master[:, :, z]
        if not m.any():
            continue
        core = binary_erosion(m)
        core = core if core.sum() >= 1 else m
        so[z] = float(np.sort(enh[:, :, z][core])[-top_k:].mean())
    return np.maximum(uniform_filter1d(so, smooth_z), 1e-3)


def occupancy_field(enh, s_o):
    """Integrated-HU occupancy = enhancement normalised by the local opacified peak S_O(z),
    clipped to [0, 1]. occ=1 is fully-opacified lumen; the occ=0.5 iso-level is the continuous
    (sub-voxel, concentration-adaptive) boundary; V = integral of occ is the PV-corrected volume."""
    return np.clip(enh / s_o[None, None, :], 0.0, 1.0)


def fwhm_lumen(enh, master, s_o, min_peak=150.0):
    """Per-frame lumen as the FWHM of the local peak (occ >= 0.5) within the anatomical extent,
    made gap-free along z (an un-opacified cross-section falls back to the anatomical extent)
    and reduced to the single duct component — a continuous, concentration-adaptive boundary."""
    seg = (occupancy_field(enh, s_o) >= 0.5) & master
    for z in range(master.shape[2]):
        if master[:, :, z].any() and (s_o[z] < min_peak or not seg[:, :, z].any()):
            seg[:, :, z] = master[:, :, z]
    seg = _largest_cc(binary_closing(seg, _ST, 1))
    for z in range(seg.shape[2]):                    # a lumen cross-section is solid: fill interior
        if seg[:, :, z].any():                       # voids (e.g. a calcification/void the FWHM cut out)
            seg[:, :, z] = binary_fill_holes(seg[:, :, z])
    return binary_fill_holes(seg)                    # + close any 3D-enclosed cavity


def solid_lumen(enh, master, abs_hu=ABS_HU):
    """Per-frame contrast-filled lumen = the opacified (enh >= abs_hu) part of the anatomical
    extent, made SOLID (closed, per-slice hole-filled, single component). Unlike fwhm_lumen, it
    does not carve out the dilute interior of a distended duct — a low-concentration but real lumen
    (occ << 1) stays one solid object instead of breaking into specks/notches."""
    seg = _largest_cc(binary_closing((enh >= abs_hu) & master, _ST, 1))
    for z in range(seg.shape[2]):
        if seg[:, :, z].any():
            seg[:, :, z] = binary_fill_holes(seg[:, :, z])
    return binary_fill_holes(seg)


def single_channel(seg):
    """Reduce a lumen mask to ONE channel: per axial slice keep only the largest cross-section
    component (drop a secondary / bifurcated branch that otherwise reads as a fork gap in the MIP),
    then keep the largest 3D component. Keeps the real (non-circular) shape — so it stays anterior
    to the spine and never balloons into bone the way a swept circular tube would."""
    out = np.zeros_like(seg)
    for z in range(seg.shape[2]):
        s = seg[:, :, z]
        if not s.any():
            continue
        lab, n = label(s)
        if n <= 1:
            out[:, :, z] = s
        else:
            counts = np.bincount(lab.ravel()); counts[0] = 0
            out[:, :, z] = lab == int(counts.argmax())
    return _largest_cc(out)


def _tv_denoise(enh, weight=40.0, max_hu=900.0):
    """Edge-preserving total-variation (Chambolle) denoise of the clipped enhancement: removes the
    speckle that makes a hard threshold jagged and a raw half-max notch, without blurring the wall."""
    return denoise_tv_chambolle(np.clip(enh, 0.0, max_hu), weight=weight)


def tv_lumen(enh, master, min_peak=150.0, tv_weight=40.0):
    """Lumen boundary = FWHM (occ >= 0.5) of the TV-DENOISED enhancement within the anatomical
    extent, single-channel + per-slice hole-filled. TV denoising is what lets the half-max boundary
    hug the true opacified lumen instead of ballooning into the partial-volume halo: a hard
    enh>=ABS_HU over-segments that bloom, and a raw FWHM notches on the speckle. Concentration-
    adaptive via per-z S_O; only a genuinely un-opacified slice (no local peak) falls back to the
    extent. TV runs on the duct bounding box only — the volume is mostly empty/bone."""
    enh_tv = np.asarray(enh, np.float32).copy()
    if master.any():
        sl = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in np.where(master))
        enh_tv[sl] = _tv_denoise(enh_tv[sl], tv_weight)
    s_o = local_opacified_ref(enh_tv, master)
    seg = (occupancy_field(enh_tv, s_o) >= 0.5) & master
    for z in range(master.shape[2]):
        if master[:, :, z].any() and s_o[z] < min_peak and not seg[:, :, z].any():
            seg[:, :, z] = master[:, :, z]
    seg = single_channel(seg)
    for z in range(seg.shape[2]):
        if seg[:, :, z].any():
            seg[:, :, z] = binary_fill_holes(seg[:, :, z])
    return binary_fill_holes(seg)


def spine_centerline(bone, smooth_z=15, y_band_frac=0.28, post_pct=60):
    """Per-z (x, y) of the vertebral column from a static-bone mask, smoothed along z. The duct runs
    just ANTERIOR (lower x) and slightly RIGHT of this column. Vertebral bodies are the POSTERIOR
    (high-x), LR-CENTRAL, z-persistent bone; ribs are lateral, so we first keep only a central LR band
    (drops ribs) then take the posterior (high-x) bone centroid per slice (the vertebral body, not the
    transverse/spinous processes). NaN slices are interpolated; an all-empty axis falls back to centre."""
    nx, ny, nz = bone.shape
    yc = ny / 2.0
    yband = np.abs(np.arange(ny) - yc) < (y_band_frac * ny)
    sx = np.full(nz, np.nan)
    sy = np.full(nz, np.nan)
    for z in range(nz):
        xs, ys = np.where(bone[:, :, z] & yband[None, :])
        if xs.size < 5:
            continue
        sel = xs >= np.percentile(xs, post_pct)            # posterior (vertebral body) bone
        sx[z] = float(xs[sel].mean()); sy[z] = float(ys[sel].mean())

    def _fill(a, fallback):
        idx = np.where(~np.isnan(a))[0]
        if not idx.size:
            return np.full(nz, fallback)
        return uniform_filter1d(np.interp(np.arange(nz), idx, a[idx]), smooth_z)

    return _fill(sx, nx / 2.0), _fill(sy, ny / 2.0)


def seedless_duct_seed(vols, voxel_mm, hu_bright=700.0, fill_thresh=150.0, min_zextent=40,
                       ant_mm=45.0, post_mm=4.0, lat_mm=28.0):
    """Locate the CC+TD duct WITHOUT a lab mask — the seedless replacement for the lab connectivity
    seed, with an ANATOMICAL para-vertebral prior. Physics: intranodal injection opacifies ONLY the
    lymphatics (arteries/veins do not fill) and the iodinated lymph SATURATES (~1500 HU) far above
    cortical bone (~380 HU). A voxel is a duct candidate if it is bright (> ``hu_bright``) at some
    frame OR clearly FILLS (mean(late)-mean(early) > ``fill_thresh``), is NOT static bone, AND lies in
    the para-vertebral ZONE — within ``ant_mm`` ANTERIOR (and ``post_mm`` posterior) of the vertebral
    column and ``lat_mm`` either side of it. The zone is the key fix over v1: it drops the lateral
    kidneys and anterior bladder / vascular-intravasation pools that an unzoned brightest-blob pick
    grabbed (caudal caliber came out 20-31 mm, a duct is ~5 mm). The duct is then the most ELONGATED
    (largest z-extent) connected component in the zone. Returns the duct-chain anchor; empty if nothing
    clears ``min_zextent`` (caller raises)."""
    mx = np.maximum.reduce(vols)
    mn = np.minimum.reduce(vols)
    static_bone = ((mx - mn) < 300.0) & (mx > 400.0)
    n = len(vols)
    ne = max(1, n // 3)
    fill = np.mean(vols[ne:], axis=0) - np.mean(vols[:ne], axis=0)
    sx, sy = spine_centerline(static_bone)
    nx, ny, nz = mx.shape
    zone = np.zeros(mx.shape, bool)
    ax_a = int(round(ant_mm / voxel_mm[0])); ax_p = int(round(post_mm / voxel_mm[0]))
    lat = int(round(lat_mm / voxel_mm[1]))
    for z in range(nz):                                     # para-vertebral box anterior of the column
        x0 = max(int(sx[z]) - ax_a, 0); x1 = min(int(sx[z]) + ax_p, nx)
        y0 = max(int(sy[z]) - lat, 0); y1 = min(int(sy[z]) + lat, ny)
        zone[x0:x1, y0:y1, z] = True
    cand = ((mx > hu_bright) | (fill > fill_thresh)) & ~static_bone & zone
    # pick the most ELONGATED (largest z-extent) in-zone component = the para-vertebral duct. (A
    # caudal-reach preference + closing was tried and REGRESSED — it grabbed caudal blobs/fragments,
    # caud caliber back to 20-29mm; max-z-extent is the better seedless selector. The remaining
    # failure = a dim caudal CC below hu_bright that never becomes a candidate; that needs a real
    # tubularity/multi-scale step, not a selection tweak.)
    lab, nl = label(cand, generate_binary_structure(3, 1))
    if nl == 0:
        return np.zeros(mx.shape, bool)
    zext = np.zeros(nl + 1)
    for i in range(1, nl + 1):
        zs = np.where((lab == i).any(axis=(0, 1)))[0]
        zext[i] = (int(zs.max()) - int(zs.min())) if zs.size else 0
    best = int(zext.argmax())
    return (lab == best) if zext[best] >= min_zextent else np.zeros(mx.shape, bool)


def hu_connected(vol, seed, hu_thresh=550.0, hu_clip=(-1024.0, 3071.0), xy_pad=70, exclude=None):
    """Connectivity extent: the >hu_thresh component(s) of one frame touching ``seed`` (the lab duct
    mask). This LOCATES the contrast-filled duct and drops the spine — equally bright (the iodinated
    lymph saturates >1000 HU, cortical bone ~400) but NOT connected to the duct, so connectivity, not
    intensity, is what excludes it. ``exclude`` (a static-bright bone mask) is carved out BEFORE the
    connectivity so the duct can't bridge to the spine when threshold alone can't separate them.
    The boundary itself is set by the TV-FWHM, not this threshold. Computed ONLY within the seed's x,y
    column extended cranially to the volume top (the duct stays within ~xy_pad of the seed over its
    course) — so a whole-body crop's full skeleton, a giant connected component, is never labeled."""
    idx = np.argwhere(seed)
    if not len(idx):
        return np.zeros(vol.shape, bool)
    lo = idx.min(0); hi = idx.max(0)
    xs = slice(max(int(lo[0]) - xy_pad, 0), min(int(hi[0]) + xy_pad + 1, vol.shape[0]))
    ys = slice(max(int(lo[1]) - xy_pad, 0), min(int(hi[1]) + xy_pad + 1, vol.shape[1]))
    zs = slice(max(int(lo[2]) - 8, 0), vol.shape[2])           # caudal seed up to the volume top (the ascending TD)
    bright = np.clip(vol[xs, ys, zs], *hu_clip) > hu_thresh
    if exclude is not None:
        bright &= ~exclude[xs, ys, zs]
    lab, _ = label(bright, generate_binary_structure(3, 1))
    keep = np.unique(lab[seed[xs, ys, zs] & (lab > 0)]); keep = keep[keep > 0]
    out = np.zeros(vol.shape, bool)
    out[xs, ys, zs] = np.isin(lab, keep)
    return out


def tv_peak_lumen(vol, seed, thin_vox=40, hu_thresh=550.0, tv_weight=0.1, max_hu=3071.0, exclude=None):
    """Whole-body CC+TD lumen on the PEAK frame — the way to segment when respiration destroys the
    temporal enhancement in the thorax (the TD vanishes from the subtraction but is plainly bright in
    the raw frame). (1) connectivity extent from the lab seed (drops the spine); (2) TV-denoise the
    frame; (3) tighten the WIDE opacified slices (the cistern) to the TV-FWHM half-max so the wall
    hugs the bright lumen, not the partial-volume halo a hard HU cut grabs; (4) leave the THIN slices
    (the thoracic duct) as the extent — the half-max over-tightens a sub-voxel tube and snaps its
    z-continuity. No raw HU level sets the boundary; the adaptive half-max of the TV-denoised frame
    does, exactly as the enhancement path's tv_lumen, just on the peak frame instead of the subtraction."""
    extent = hu_connected(vol, seed, hu_thresh, exclude=exclude)
    if not extent.any():
        return extent
    tv = np.asarray(vol, np.float32).copy()
    sl = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in np.where(extent))
    tv[sl] = denoise_tv_chambolle(np.clip(tv[sl], 0.0, max_hu) / max_hu, weight=tv_weight) * max_hu
    s_o = local_opacified_ref(tv, extent)
    fwhm = (occupancy_field(tv, s_o) >= 0.5) & extent
    area = extent.sum(axis=(0, 1))
    seg = extent.copy()
    for z in range(extent.shape[2]):                 # tighten only wide+opacified slices (cistern); thin TD stays extent
        if area[z] > thin_vox and fwhm[:, :, z].any():
            seg[:, :, z] = fwhm[:, :, z]
    for z in range(seg.shape[2]):
        if seg[:, :, z].any():
            seg[:, :, z] = binary_fill_holes(seg[:, :, z])
    return seg                                       # already the seed-connected extent (hu_connected), only tightened —
    #                          no further connectivity filter: FWHM-tightening can snap the wide->thin
    #                          neck, and a seed-touching filter would then drop the whole thoracic TD


def fill_extension(vols, master, exclude=None, n_early=2, fill_thresh=200.0, pad_below=6):
    """Extend the master TD cranially along the contrast BOLUS FRONT. A dim ascending TD and the
    moving bone rim look identical in any single frame (both bright, both temporally varying, sitting
    a voxel apart), so a peak-HU connectivity stops at the clear part and a brightest-voxel tracker
    rides the rim to the top. They differ in SIGN of change: the TD FILLS (dark early -> bright late as
    the bolus climbs) while the rim only OSCILLATES with respiration. Region-grow UPWARD only (the
    abdominal lymph nodes hang off the cistern BELOW, so cranial-only growth never reaches them) on the
    directional fill signature mean(late)-mean(early), keeping the component that touches the master's
    cranial end. It self-limits at the bolus front — where the contrast stopped climbing during the
    series — which is the physical TD reach for THIS acquisition. Do NOT morphologically bridge the
    stop: above the front the fill is rim/noise speckle, and a >=3-vox dilation floods it into the body
    (100x the voxels). The boundary is temporal, not a tunable HU level."""
    vols = list(vols)
    fill = np.mean(vols[n_early:], axis=0) - np.mean(vols[:n_early], axis=0)
    zc = np.where(master.any(axis=(0, 1)))[0]
    if not zc.size:
        return np.zeros(master.shape, bool)
    ztop = int(zc.max()); z0 = max(ztop - pad_below, 0)
    cand = fill > fill_thresh
    if exclude is not None:
        cand &= ~exclude
    cand[:, :, :z0] = False                                       # cranial-only: no abdominal nodes
    seed = np.zeros(master.shape, bool)
    seed[:, :, z0:ztop + 1] = master[:, :, z0:ztop + 1]          # anchor to the master's cranial end
    lab, _ = label(cand | seed, generate_binary_structure(3, 1))
    keep = np.unique(lab[seed & (lab > 0)]); keep = keep[keep > 0]
    return np.isin(lab, keep) & cand


def fill_nodes(vols, master, exclude=None, n_early=2, fill_thresh=200.0):
    """Contrast-filled lymph NODES hanging off the duct. Intranodal lymphangiography fills the nodes
    UPSTREAM of the cistern (nodes -> trunks -> CC -> TD), so the para-aortic / renal-hilar nodes
    opacify as bean-shaped blobs lateral to the duct's column. Segment them as the fill>thresh
    component(s) CONNECTED to the duct master (the lymphatic chain) minus the master itself. The FILL
    signature (mean(late)-mean(early)), not raw HU, is what makes this clean: a raw-HU connectivity to
    the master floods the whole spine/body (~325k vox on Acq6 — the nodes touch the cistern which
    touches the bone rim), whereas the nodes genuinely ENHANCE, so the fill signature keeps them (~12k
    vox, bounded to the para-aortic/renal region) and drops the static rim. No xy-restriction (unlike
    hu_connected) — the nodes sit lateral to the duct, outside its column."""
    vols = list(vols)
    fill = np.mean(vols[n_early:], axis=0) - np.mean(vols[:n_early], axis=0)
    cand = fill > fill_thresh
    if exclude is not None:
        cand &= ~exclude
    lab, _ = label(cand | master, generate_binary_structure(3, 1))
    keep = np.unique(lab[master & (lab > 0)]); keep = keep[keep > 0]
    return np.isin(lab, keep) & ~master & cand


def reclaim_axial_cistern(master, nodes, voxel_mm, r_mm=9.0):
    """Reclaim the dilated cistern that ``fill_nodes`` over-grabbed in a confluent para-aortic mass
    (the angiotensin Acq8/Acq12 failure: the node label swallowed the CC). Enhancement labeled 'node'
    but lying ON the duct AXIS — within ``r_mm`` of the master's per-z centerline — is the CC, not a
    node, so merge it into the duct; only the genuinely LATERAL blobs stay nodes. The central tube vs
    lateral blob distinction is the principled CC/node separator (nodes hang off the duct's side)."""
    if not master.any() or not nodes.any():
        return master, nodes
    cx, cy = centerline(master)
    xs, ys, zs = np.where(nodes)
    dx = (xs - cx[zs]) * voxel_mm[0]
    dy = (ys - cy[zs]) * voxel_mm[1]
    near = (dx * dx + dy * dy) <= (r_mm * r_mm)
    reclaim = np.zeros_like(nodes)
    reclaim[xs[near], ys[near], zs[near]] = True
    return master | reclaim, nodes & ~reclaim


def centerline(mask, smooth_z=5):
    """The duct path as the smoothed per-z (x, y) centroid of ``mask`` over its z-extent; empty
    slices are linearly interpolated. Built once from the contrast (master) extent and reused
    across timepoints — the centerline does not move appreciably between respiratory phases."""
    nz = mask.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in range(nz):
        if mask[:, :, z].any():
            cx[z], cy[z] = center_of_mass(mask[:, :, z])

    def fill(a):
        idx = np.where(~np.isnan(a))[0]
        return uniform_filter1d(np.interp(np.arange(nz), idx, a[idx]), smooth_z) if idx.size else a

    return fill(cx), fill(cy)


def swept_tube(cx, cy, radius_vox, z_range, shape):
    """Reconstruct a SMOOTH continuous tube: at each z a disk centred on the (reused) centerline
    with the per-z ``radius_vox`` (the equivalent radius of the integrated-HU cross-section,
    smoothed along z). Circular cross-sections guarantee a gap-free, silky surface for the 3D —
    the lumpy/patchy raw mask is what fragments. Volume/caliber still vary per timepoint."""
    nx, ny = shape[:2]
    yy, xx = np.ogrid[:nx, :ny]
    tube = np.zeros((nx, ny, shape[2]), bool)
    for z in z_range:
        if radius_vox[z] > 0.3:
            tube[:, :, z] = (xx - cx[z]) ** 2 + (yy - cy[z]) ** 2 <= radius_vox[z] ** 2
    return tube


def equiv_radius(seg, z_range, smooth_z=7):
    """Per-z equivalent radius (voxels) of a cross-section area, sqrt(area/pi), smoothed along z."""
    nz = seg.shape[2]
    r = np.zeros(nz)
    for z in z_range:
        a = int(seg[:, :, z].sum())
        r[z] = np.sqrt(a / np.pi) if a > 0 else 0.0
    return uniform_filter1d(r, smooth_z)


def frame_metrics(patch, frame, boundary_z, rois=None, abs_hu=ABS_HU):
    """Per-frame PER-TIMEPOINT-segmented metrics (the extent grows with filling — kept only
    for diagnostics; prefer the fixed-extent fixed_frame_metrics for volumes). Returns a dict
    with V_total, V_cc, V_td, L_td, d_cc, d_td and the masks."""
    lumen, enh = segment_lumen(patch, frame, rois, abs_hu)
    cc, td = split_caudal(lumen, boundary_z)
    vvol = float(np.prod(patch.voxel_mm)); vz = patch.voxel_mm[2]
    tdz = np.where(td.any(axis=(0, 1)))[0]
    return dict(frame=frame, V_total=lumen.sum() * vvol, V_cc=cc.sum() * vvol,
                V_td=td.sum() * vvol, L_td=(tdz.max() - tdz.min()) * vz if tdz.size else 0.0,
                d_cc=caliber(cc, patch.voxel_mm), d_td=caliber(td, patch.voxel_mm),
                cc_mask=cc, td_mask=td, enh=enh)
