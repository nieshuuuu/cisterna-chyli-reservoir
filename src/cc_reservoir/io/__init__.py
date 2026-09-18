REAL_VOXEL_MM = (0.781, 0.781, 0.5)  # mm; VERIFIED against DICOM (PixelSpacing 0.781; the MAT grid is
# CONSECUTIVE DICOM slices at 0.5 mm — confirmed MAT[:,:,k]==DICOM[k] corr 1.0 for all 3 sessions).
# The z was previously hardcoded 1.0 (WRONG) → every volume was 2x too big. (2026-06-24)
