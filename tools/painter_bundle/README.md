# CC / TD iPad Painter — full-resolution package

Shu Nie (nies1@hs.uci.edu), Molloi Lab, UCI — packaged 2026-08-07.

This is the lab's segmentation painter for the cisterna chyli (CC) and thoracic duct (TD)
in the pig dynamic-CT lymphangiography archive (`Z:\ImageData\Lymph_Studies\`, the data
behind Molloi et al., *Radiology* 2023;309(3):e230959), together with the full-resolution
CT volumes for **every acquisition that currently has a segmentation** (28 acquisitions,
~15 GB). You paint on an iPad with an Apple Pencil in Safari; a small Python web server
runs on any computer.

**Start here: double-click `Paint.bat`.** It syncs your masks up, pulls down the current
painter and everyone else's work, opens the painter on a fast local copy, and pushes your
masks back when you press Ctrl-C. The first run copies the CT locally (~16 GB, once, asks
first). `Start_CC_Painter.bat` is now just a shortcut to it.

The `_edit.npz` mask files in `data_fullres` are the painter's **working copies** -- it reads
and writes them in place. They are not the delivered masks: for analysis use
`..\1_masks_USE_THESE`, which reconciles these against the July 2026 snapshot in
`..\3_old_masks_superseded` and drops what turned out not to be hand work. After a painting
session, re-run `cc_reservoir.scripts.merge_mask_archives` to fold the new work in.

---

## What's in this folder

| Item | What it is |
|---|---|
| `Start_CC_Painter.bat` | Double-click launcher (Windows) |
| `start_cc_painter.sh` | Launcher for Mac / Linux: `sh start_cc_painter.sh` |
| `software\src\cc_reservoir\` | The Python package. The painter is a single file — `viz\ipad_paint.py` (the whole web UI is inlined in it). `measure\context.py` has the re-analysis helpers (section 7). |
| `data_fullres\` | 28 acquisitions of full-resolution 4-D context volumes, including the current segmentations (`f*_edit.npz`) |
| `README.md` | this guide |

**Full resolution** = the native DICOM grid, voxel 0.781 × 0.781 × 0.5 mm; `downsample = [1,1,1]`
in every `meta.npz` (no downsampling anywhere).

Software snapshot: `MolloiLab/cisterna-chyli-reservoir` (private GitHub repo), local branch
`painted-cc-render` as of 2026-07-29, which includes lab improvements not yet pushed to GitHub.

---

## 1. Set up (once)

1. **Copy this whole folder to a local drive** (~15 GB). It *does* run straight off `Z:`,
   but every time-frame is a ~100–290 MB file read on demand — first load of each frame over
   the network is slow; from a local disk it's fast, and frames are cached/prefetched in RAM
   after that (up to ~8 frames, ~1 GB each — a machine with 16 GB+ RAM is comfortable).
2. Install **Python 3.10+** (python.org). On Windows tick *"Add python.exe to PATH"*.
3. Install the only two dependencies:

```bash
pip install numpy pillow
```

An iPad + Apple Pencil gives the intended experience, but a mouse/trackpad works too
(good for QC, small fixes, or just viewing).

## 2. Launch

- **Windows:** double-click `Start_CC_Painter.bat`.
- **Mac / Linux:** `sh start_cc_painter.sh`

What the launchers actually run (use this directly to pick another start folder or port):

```
cd <this folder>
set PYTHONPATH=software\src        (Mac/Linux:  export PYTHONPATH="$PWD/software/src")
python -m cc_reservoir.viz.ipad_paint "data_fullres\context4d_4_13_23_data_Acq01_Acq01" --port 8778
```

- The folder you pass is only the **starting** acquisition. On startup the painter scans all
  sibling `context4d_*` folders and lists **all of them in the Acquisition dropdown**, grouped
  by session date and ordered on the true acquisition timeline.
- Wait for the `http://<this-computer's-IP>:8778/` line, then open that URL **in Safari on an
  iPad on the same Wi-Fi**. On the computer itself: `http://127.0.0.1:8778/`.
- First run on Windows: **allow the firewall prompt** (Private networks), or the iPad won't reach it.
- Keep the console window open while painting — closing it stops the server.
- Extras: `--port NNNN` if 8778 is taken; add `?debug=1` to the URL for a touch-gesture debug
  readout; `--selftest` runs the internal checks with no server.

## 3. Painting — how the UI works

**Input model (Procreate-style):**

- **The Pencil draws. Fingers never draw.** A resting palm can't scribble or change slices.
- **Two fingers = pinch-zoom + pan** (the anatomy under your fingers stays under them).
  One finger on the image does nothing, by design.
- A mouse/trackpad also draws (desktop use); Ctrl+wheel zooms about the cursor.
- Slices change **only** via the scrub bar under the image (slider, exact −/+ buttons)
  or the arrow keys (←/→ or ↑/↓) — never by accidental touch.

**Views:** Cor / Sag / Ax buttons (coronal default), cranial-up radiological convention with
Cr/Cd/A/P/R/L edge marks. "⟳ Rotate 90°" turns the *display* only — strokes still land on the
correct voxels.

**Right sidebar:**

- Four **label chips** — CC (green), TD (teal), lymph (gold), bone (tan) — tap to select the
  paint label. Each chip has an eye toggle controlling *display only* (default: only CC shown;
  selecting a label auto-shows it).
- **Erase** (erases any label under the brush), **Undo / Redo**, **Clear all** (whole volume,
  asks first, one Undo restores it).
- **Slab ±** (0–24 slices, default 8): the CT backdrop becomes a rolling slab-MIP and labels
  anywhere inside the slab appear as a dim "ghost" (solid = on this exact slice). Slab 0 = plain
  single slice.
- **Brush** 1–12 mm (default 3) — a true physical-mm circle, anisotropy-corrected.
- **Zoom** slider 1–8×; **window presets**: contrast (default), soft tissue, bone.
- **Frame dropdown** — frames with a saved edit are marked "✎ edited"; other frames are
  prefetched in the background so switching is instant.
- **Save** button and a HUD with live per-label volume (mL), voxel counts, saved/unsaved state.

**Left sidebar (usually set once):**

- **Tool — Brush / Lasso / Wand** (keys `b` / `l` / `w`):
  - *Brush*: trace it. What it always was.
  - *Lasso*: draw a loose loop around the duct and let go. Everything the loop encloses is
    filled — **subject to the HU gate**, so on a well-set band you circle roughly and only the
    duct inside the loop is labelled. This is the fast way to cover a long tortuous run.
  - *Wand*: one tap. Fills the connected in-band structure around the tapped voxel, out to
    **Wand reach** (3–25 mm). Where the threshold separates the duct cleanly, one tap replaces
    a whole stroke. With **Erase** on, both tools remove instead: lasso-erase wipes everything
    enclosed, wand-erase removes the connected labelled blob you tap.
  - Lasso and wand report how many voxels they landed; **Undo (`u`) reverses the whole gesture**.
- **MIP depth-follow — Off / Snap / Grow (default Grow).** The other key trick:
  - *Off*: paint flat on the current slice only.
  - *Snap*: each stroke pixel is back-projected to the brightest slice within ±slab whose HU
    falls in the band — you paint on the MIP, the mask lands on the duct.
  - *Grow*: Snap, plus fills the contiguous in-band run of slices, so a duct several slices
    thick is captured in one stroke.
- **HU gate — On / Off** (key `g`), with **Floor** and **Ceiling** sliders (± buttons step 5 HU;
  the ceiling's top stop reads "open" = no upper limit).
  - **On: a voxel is painted only if floor ≤ HU ≤ ceiling.** That holds for every tool and every
    depth mode. Erase is never gated, so you can always remove a label you can see.
  - **Off: nothing is restricted** — free drawing, for the places a threshold cannot help.
  - **Show gate** (key `h`) washes every voxel the band allows in blue. Use it to set the band:
    if the blue covers half the picture, the band is not restricting anything.
  - The line under the sliders reports **what percentage of the current slab the band admits**.
    Under ~10% is a band doing real work; above ~25% it warns you that it is too loose.
  - Rough starting points: post-contrast duct ≈ floor 120, ceiling 700 (drops rib cortex);
    Lipiodol runs need a much *higher* ceiling — the duct itself reaches 2000–3400 HU on some
    acquisitions, so put the ceiling above the duct, not below it; pre-contrast f0, where chyle
    is *dimmer* than muscle, needs a low band such as −20…45.
  - The three fixed "HU target" presets (Bright / Bright-no-bone / Near-water) were removed:
    they were three arbitrary points in a two-slider space, no acquisition actually matched one,
    and they concealed the fact that the band was restricting nothing.
- If a stroke finds **no in-band voxel it paints nothing** and says so in a toast — that is the
  gate working, not a fault. Widen the band, press **Show gate** to see where it lets you paint,
  or switch the HU gate Off.

## 4. Saving — read this before you paint

- **There is no autosave.** The blue **Save** button (or Cmd/Ctrl+S) writes the current frame's
  mask to `f<N>_edit.npz` *next to* the frame file. Saving is **per time-frame**.
- Switching frame or acquisition with unsaved strokes **discards them** — a confirm dialog
  warns you first; hit Cancel, Save, then switch.
- On load, an `f<N>_edit.npz` sidecar **always takes precedence** over the seg baked into
  `f<N>.npz`. Every lab tool applies the same rule, so your saved edits are automatically what
  gets measured. To revert a frame to the automatic segmentation, delete its sidecar file.
- Undo/Redo are unlimited within the current frame session and reset when you switch frames.

## 5. Keyboard shortcuts (desktop / external keyboard)

| Key | Action |
|---|---|
| `1 2 3 4` | select label CC / TD / lymph / bone |
| `b` / `l` / `w` | tool: brush / lasso / wand |
| `g` | HU gate on / off |
| `h` | show / hide the gate preview wash |
| `e` | toggle erase |
| `u` / `r` (also Cmd/Ctrl+Z / Cmd/Ctrl+Shift+Z) | undo / redo |
| Cmd/Ctrl+S | save |
| `[` / `]` | slab −2 / +2 |
| `←` / `→` (also `↓` / `↑`) | slice −1 / +1 |

## 6. What the data is

Each `data_fullres\context4d_<session>_<condition>_Acq<N>\` folder is one CT acquisition:

| File | Contents |
|---|---|
| `f<N>.npz` | one time-frame: `hu` int16 volume `[x,y,z]` + the builder's automatic `seg` uint8 |
| `f<N>_edit.npz` | manual segmentation sidecar: `seg` uint8 + `source` (no HU — always wins over `f<N>.npz`'s seg) |
| `meta.npz` | voxel size (mm), frame count, peak-contrast frame index, duct centroid, CC/TD boundary diagnostics, and the raw-DICOM provenance: `crop_lo`, `crop_hi`, `downsample`, `raw_full_shape`, `raw_session`, `raw_sub` (load with `allow_pickle=True`) |

Axes: x = anterior→posterior, y = left→right, z = caudal→cranial.
Labels: **1 = CC, 2 = TD, 3 = lymph, 4 = bone.**

Included sessions (only acquisitions that already have segmentations):

| Session | Acquisitions | Notes |
|---|---|---|
| `4_13_23` | all 6 sub-acqs | 2023 session, highest-signal data |
| `4_19_23` | all 4 sub-acqs | 2023 session, highest-signal data |
| `03_23_23` | 5 sub-acqs (Acq01–Acq03) | two-volume protocol, no dynamic curve; Acq01 has no 2nd volume — limited use |
| `07_20_22` | Angiotensin Acq8/9/12, Baseline Acq6/7/10 | 2022 dynamic sessions, normal-caliber duct |
| `8_31_22` | Acq1–Acq6 | 2022 dynamic session |
| `09_07_22` | Acq1 | 2022 dynamic session |

The other 60 acquisitions of the full build (63.9 GB total) have no segmentations yet and live
on the lab Windows PC at `C:\Storage\CisternaChyli\cc_contexts_fullres` — ask Shu Nie, or
rebuild any acquisition from the raw DICOM with
`software\src\cc_reservoir\scripts\build_mip_context.py`.

## 7. Re-analysis without the painter

Label volumes straight from a mask file:

```python
import numpy as np

folder = r"data_fullres\context4d_4_13_23_data_Acq01_Acq01"
seg  = np.load(folder + r"\f1_edit.npz")["seg"]           # uint8 [x,y,z]
meta = np.load(folder + r"\meta.npz", allow_pickle=True)
voxel_ml = float(np.prod(meta["voxel"])) / 1000.0         # mm^3 -> mL
for lab, name in [(1, "CC"), (2, "TD"), (3, "lymph"), (4, "bone")]:
    print(name, round(float((seg == lab).sum()) * voxel_ml, 3), "mL")
```

Same thing with the sidecar-precedence rule handled for you, then mapped onto the **raw
DICOM grid** (this example: 512 × 512 × 921 — see `meta["raw_full_shape"]`; the 2022
whole-body sessions are 512 × 512 × 1401):

```python
import sys, numpy as np
sys.path.insert(0, r"software\src")
from cc_reservoir.measure.context import load_context_seg, place_seg_in_raw

folder = r"data_fullres\context4d_4_13_23_data_Acq01_Acq01"
seg  = load_context_seg(folder, frame=1)      # automatically prefers f1_edit.npz
meta = np.load(folder + r"\meta.npz", allow_pickle=True)
raw_labels = place_seg_in_raw(seg, meta)      # label array on the full raw grid
# raw_hu[raw_labels == 2] -> TD voxel HUs of the original DICOM volume
```

The mapping is exact and simple: `raw_index = crop_lo + context_index` per axis (downsample
is 1 here). Which raw series a mask belongs to is recorded in `meta` (`raw_session`,
`raw_sub`); the raw DICOM lives under `Z:\ImageData\Lymph_Studies\<session>_data\` (note:
the leading zero in month names is inconsistent — `8_31_22_data` but `08_15_22_data` — so
match loosely). `measure.context.raw_values_under_seg(context_dir, frame, archive)` runs the
whole chain when the raw archive is reachable.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| iPad can't open the URL | Same Wi-Fi as the computer? Windows firewall allowed (Private networks)? Type `http://<computer-IP>:8778/` manually. |
| First load of each frame is very slow | You're running off `Z:` — copy this folder to a local disk. |
| Stroke paints nothing + a toast appears | The HU gate found no voxel inside [Floor, Ceiling] under the stroke. Press **Show gate** (`h`) to see where it does let you paint, widen the band, or turn the HU gate Off (`g`). |
| The threshold does not seem to restrict anything | Read the "gate opens N% of this slab" line. At floor 120 with an open ceiling that is ~74% on a thorax — nearly everything is in band, so the gate has nothing to refuse. Raise the Floor and set a real Ceiling until the number drops. |
| Port already in use | launch with `--port 8899` (any free port) |
| `ModuleNotFoundError: cc_reservoir` | `PYTHONPATH` must point at `software\src` — use the launchers, or set it as in section 2. |
| `No module named numpy` / `PIL` | `pip install numpy pillow` |
| Want a quick sanity check | run the launch command with `--selftest` appended — internal checks run, no server starts. |

Questions: **Shu Nie — nies1@hs.uci.edu**
