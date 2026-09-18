# Superseded — use `..\1_masks_USE_THESE` instead

These are the masks as they stood in July 2026 (files dated 07-16 to 07-22), from when this folder
was called `Egor_FullRes_Segmentations`. They have been merged with the newer painter package into
`..\1_masks_USE_THESE`.

**Read masks from there, not from here.**

## Why this folder is kept rather than deleted

It is 18 MB, and for 13 frames it is the only surviving copy of an earlier, larger mask. The newer
painter package lost content in those frames — 22% to 91% fewer voxels — and
`07_20_22_Angiotensin_Acq12 f0` was wiped to zero entirely (that one has already been recovered
into the merged folder). Most of those reductions look like correct trimming of pre-bolus frames,
but they have not been reviewed one by one, so the pre-loss version stays available.

| acquisition | frame | here | merged | lost |
|---|---|---|---|---|
| 07_20_22_Baseline_Acq10 | f0 | 25,172 | 2,176 | 91% |
| 03_23_23_Acq02_Acq01 | f0 | 10,537 | 1,258 | 88% |
| 07_20_22_Angiotensin_Acq8 | f0 | 26,528 | 5,676 | 79% |
| 07_20_22_Angiotensin_Acq12 | f1 | 14,068 | 3,192 | 77% |
| 07_20_22_Baseline_Acq10 | f1 | 10,940 | 3,875 | 65% |
| 07_20_22_Angiotensin_Acq12 | f2 | 28,660 | 10,560 | 63% |
| 07_20_22_Baseline_Acq7 | f0 | 20,920 | 7,868 | 62% |
| 07_20_22_Angiotensin_Acq8 | f1 | 16,956 | 7,119 | 58% |
| 03_23_23_Acq02_Acq01 | f1 | 19,240 | 8,207 | 57% |
| 07_20_22_Angiotensin_Acq12 | f3 | 33,888 | 15,826 | 53% |
| 07_20_22_Baseline_Acq6 | f0 | 21,640 | 13,871 | 36% |
| 07_20_22_Angiotensin_Acq12 | f4 | 33,364 | 25,799 | 23% |
| 07_20_22_Angiotensin_Acq8 | f2 | 23,928 | 18,640 | 22% |

Note the 07_20_22 masks here are the raw 2×2×1 in-plane upsample of earlier half-resolution work
(they carry `source="egor_ds2:..."`), before any full-resolution editing.

Once someone has checked those 13 frames and confirmed the newer version is right, this folder can
go.
