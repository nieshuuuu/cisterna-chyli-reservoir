"""Manual correction for CC/TD segmentation, in napari, sagittal-first.

Backdrop = HU as an adjustable rolling slab-MIP (slider, +/-slab along the L-R
scroll axis) so the thin, tortuous duct reads continuously. The paintable layer
is the TRUE per-slice seg (never MIP'd), so every brush stroke lands on a
definite slice. Dial slab THICK to see / locate / rough-paint; dial to 0 for
depth-exact boundary work.

Why not paint on the MIP itself: a MIP collapses the scroll axis, so a stroke
would be depth-ambiguous. Instead we MIP only the backdrop and keep the label
per-slice; `b` recovers depth by snapping a stroke to the brightest slab-slice.

Controls
  paint / erase / pick ...... napari native (Labels layer: 1=CC 2=TD 3=kidney 4=bone)
  slab slider ............... backdrop MIP thickness, 0 = single true slice
  b ......................... back-project: snap the current label's voxels on
                              THIS slice to the brightest slice within +/-slab
                              (recovers depth after thick-slab painting; no-op at slab 0)
  Ctrl-S / Save button ...... write sidecar  <name>_edit.npz   (never touches source `seg`)

Run:   python -m cc_reservoir.viz.edit_seg  /path/f4.npz  [--out LOCAL_DIR]
       python -m cc_reservoir.viz.edit_seg  --selftest        (no GUI, checks the math)
Needs: pip install "napari[pyside6]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from cc_reservoir.io.context import load_seg

# Source arrays are indexed [x, y, z]: x anterior->posterior, y left-right, z
# caudal->cranial. We reproduce viewer.py's sagittal EXACTLY (the only in-repo view
# with drawn A/P/Cd/Cr labels): scroll y; rows = x (anterior at top), cols = z
# (caudal left, cranial right) -- the "lying" orientation. napari scrolls axis 0 and
# shows the last two axes, so reorder [x,y,z] -> [y, x, z]; no flips, save round-trips
# trivially. (Patient left/right is NOT in the data -- no DICOM ImageOrientationPatient
# anywhere -- so only A-P/Cd-Cr are meaningful; napari's transpose gives standing.)


def to_view(a: np.ndarray) -> np.ndarray:
    return a.transpose(1, 0, 2)                     # [x,y,z] -> [y, x, z] (== viewer.py sagittal)


def from_view(a: np.ndarray) -> np.ndarray:
    return a.transpose(1, 0, 2)                     # self-inverse (swaps axes 0,1 back)


def rolling_mip(vol_v: np.ndarray, slab: int) -> np.ndarray:
    """Max over +/-slab along the scroll axis (0). slab<=0 -> identity (single slice)."""
    if slab <= 0:
        return vol_v
    # ponytail: recomputes the whole slab-MIP per slider change. Fine for +/-24 on
    # these volumes; if it lags, compute on slider-release or use a running max.
    return maximum_filter1d(vol_v, size=2 * slab + 1, axis=0, mode="nearest")


def backproject_ops(hu_v, seg_v, y0, slab, label):
    """Ops to relocate `label` voxels on slice y0 to their brightest slab-slice.

    Returns (clear_idx, set_idx) as (y, x, z) index tuples, or None if nothing to
    do. Pure: the GUI applies the ops via data_setitem so napari records undo.
    A voxel already on its brightest slice maps to itself (cleared then re-set).
    """
    if slab <= 0:
        return None
    ny = hu_v.shape[0]
    lo, hi = max(0, y0 - slab), min(ny, y0 + slab + 1)
    mask2d = seg_v[y0] == label
    if not mask2d.any():
        return None
    xs, zs = np.nonzero(mask2d)
    window = hu_v[lo:hi][:, xs, zs]              # (win, N) HU along the slab
    best = (lo + np.argmax(window, axis=0)).astype(np.intp)   # brightest slice per voxel
    y0arr = np.full(xs.shape, y0, dtype=np.intp)
    return (y0arr, xs, zs), (best, xs, zs)


def _try_set_colors(lbl):
    """Match the viz/labels.py palette (CC green, TD teal, kidney gold, bone tan). Cosmetic."""
    rgba = {1: (0.353, 0.643, 0.412, 1.0), 2: (0.227, 0.561, 0.612, 1.0),
            3: (0.878, 0.639, 0.227, 1.0), 4: (0.710, 0.569, 0.416, 1.0)}
    try:                                          # napari >= 0.5 (current; 0.4's lbl.color is a silent no-op here)
        from napari.utils.colormaps import DirectLabelColormap
        lbl.colormap = DirectLabelColormap(color_dict={**rgba, None: (0, 0, 0, 0)})
        return
    except Exception:
        pass
    try:                                          # napari <= 0.4.x
        lbl.color = rgba
        lbl.color_mode = "direct"
    except Exception as e:
        print(f"[edit_seg] label color match skipped ({e}); using napari defaults")


def _voxel(src):
    """(vx, vy, vz) mm spacing: from the npz's own 'voxel' key, else a sibling meta.npz, else None."""
    with np.load(src) as d:
        if "voxel" in d.files:
            return tuple(float(v) for v in d["voxel"])
    meta = src.with_name("meta.npz")                  # 4D per-frame f<i>.npz keep voxel in meta.npz
    if meta.exists():
        with np.load(meta) as m:
            if "voxel" in m.files:
                return tuple(float(v) for v in m["voxel"])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="napari CC/TD seg editor (sagittal, slab-MIP)")
    ap.add_argument("npz", help="context npz with keys hu, seg (e.g. f4.npz)")
    ap.add_argument("--out", default=None,
                    help="dir for the sidecar (default: next to source; use a LOCAL dir if source is on SMB)")
    args = ap.parse_args(argv)

    src = Path(args.npz)
    d = np.load(src)
    if "hu" not in d or "seg" not in d:
        sys.exit(f"[edit_seg] {src} lacks hu/seg keys; found {list(d.files)}")
    hu_v = np.ascontiguousarray(to_view(d["hu"]))
    seg_v = np.ascontiguousarray(to_view(load_seg(src)))   # resume from a prior _edit.npz if present
    out_dir = Path(args.out) if args.out else src.parent
    sidecar = out_dir / f"{src.stem}_edit.npz"
    vox = _voxel(src)                                  # (vx, vy, vz) mm, None if unknown
    scale_kw = {"scale": (vox[1], vox[0], vox[2])} if vox else {}   # view order (Y, X, Z); fixes aspect

    import napari
    from magicgui import magicgui

    state = {"slab": 8}
    viewer = napari.Viewer(title=f"edit {src.name} - sagittal")
    lo, hi = np.percentile(hu_v, [1, 99])
    clim = (float(lo), float(hi) if hi > lo else float(lo) + 1.0)
    img = viewer.add_image(rolling_mip(hu_v, state["slab"]), name="HU slab-MIP",
                           colormap="gray", contrast_limits=clim, **scale_kw)
    lbl = viewer.add_labels(seg_v, name="seg", **scale_kw)
    lbl.selected_label = 1
    lbl.mode = "paint"
    lbl.brush_size = 3
    try:
        lbl.n_edit_dimensions = 2                 # brush paints the shown slice only
    except Exception:
        pass
    _try_set_colors(lbl)

    slab_max = min(int(hu_v.shape[0] // 2), 24)    # ~matches the PySide6 viewer's max slab

    @magicgui(auto_call=True,
              slab={"widget_type": "Slider", "min": 0, "max": slab_max,
                    "label": "MIP slab (+/-slices)"})
    def slab_ctl(slab: int = state["slab"]):
        state["slab"] = int(slab)
        img.data = rolling_mip(hu_v, state["slab"])
    viewer.window.add_dock_widget(slab_ctl, area="right", name="slab")

    def do_save():
        out_dir.mkdir(parents=True, exist_ok=True)
        seg_out = from_view(np.asarray(lbl.data)).astype(np.uint8)
        np.savez_compressed(sidecar, seg=seg_out, source=src.name)
        print(f"[edit_seg] saved {sidecar}  ({int((seg_out > 0).sum())} labeled voxels)")

    @magicgui(call_button="Save sidecar (Ctrl-S)")
    def save_ctl():
        do_save()
    viewer.window.add_dock_widget(save_ctl, area="right", name="save")

    @viewer.bind_key("Control-S", overwrite=True)
    def _save_key(_v):
        do_save()

    @viewer.bind_key("b", overwrite=True)
    def _snap(_v):
        y0 = int(viewer.dims.current_step[0])
        label = int(lbl.selected_label)
        ops = backproject_ops(hu_v, np.asarray(lbl.data), y0, state["slab"], label)
        if ops is None:
            print(f"[edit_seg] back-project: nothing to snap (label {label}, y={y0}, slab={state['slab']})")
            return
        clear_idx, set_idx = ops
        try:
            lbl.data_setitem(clear_idx, 0)        # via data_setitem -> napari undo works
            lbl.data_setitem(set_idx, label)
        except AttributeError:                    # very old napari: direct write
            arr = np.asarray(lbl.data)
            arr[clear_idx] = 0
            arr[set_idx] = label
            lbl.refresh()
        print(f"[edit_seg] back-projected label {label} at y={y0} within +/-{state['slab']} "
              f"({len(set_idx[0])} voxels)")

    print(__doc__)
    napari.run()


def _selftest():
    rng = np.random.default_rng(0)
    a = rng.integers(-200, 800, size=(5, 6, 7)).astype(np.int16)
    assert np.array_equal(from_view(to_view(a)), a), "view round-trip"
    marker = np.zeros((3, 4, 5), np.int16)
    marker[0, 1, 4] = 7                                    # x0 = anterior, max z = cranial
    assert to_view(marker)[1, 0, 4] == 7, "anterior (x0) at top row, cranial (max z) at right col"
    v = to_view(a)
    assert np.array_equal(rolling_mip(v, 0), v), "slab 0 is identity"
    assert rolling_mip(v, 1)[2, 3, 4] == v[1:4, 3, 4].max(), "slab MIP window"

    hu = np.zeros((5, 4, 4), np.int16)
    seg = np.zeros((5, 4, 4), np.uint8)
    hu[3, 1, 2] = 1000          # brightest slice for column (x=1, z=2) is y=3
    seg[2, 1, 2] = 1            # label sits on center slice y0=2
    ops = backproject_ops(hu, seg, y0=2, slab=2, label=1)
    assert ops is not None
    (cy, cx, cz), (sy, sx, sz) = ops
    s = seg.copy()
    s[cy, cx, cz] = 0
    s[sy, sx, sz] = 1
    assert s[3, 1, 2] == 1 and s[2, 1, 2] == 0, "back-projection snaps to argmax slice"
    assert backproject_ops(hu, seg, y0=2, slab=0, label=1) is None, "slab 0 no-op"

    import os
    td = "/tmp/cc_edit_seg_test"
    os.makedirs(td, exist_ok=True)
    fp = Path(td) / "f0.npz"
    np.savez_compressed(fp, hu=np.zeros((2, 2, 2), np.int16), seg=np.zeros((2, 2, 2), np.uint8))
    np.savez_compressed(Path(td) / "meta.npz", voxel=np.array([1.5, 1.5, 0.5]))
    assert _voxel(fp) == (1.5, 1.5, 0.5), "voxel read from sibling meta.npz"
    print("selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        main()
