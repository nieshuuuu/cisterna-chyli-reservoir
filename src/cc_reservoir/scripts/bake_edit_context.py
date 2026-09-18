"""Bake a context4d_<tag>/ from a HAND seg, so the deliverables show the hand seg and not the auto one.

Why this exists: `io.context.load_seg` lets a hand edit (`f<N>_edit.npz`) override the OVERLAY only.
Everything else render_context_assets draws -- the 3D meshes (rotate/fill/structure) -- was baked
from the producer's auto seg at build time. Point the renderer at a raw context and the 3D still
shows the OLD segmentation. This re-derives the meshes from the hand seg, so the edited seg is the
single source for every product.

Reads the crop that export_edit_crop.py pulled off pc2 (hu + hand seg, all frames, duct bbox +
margin) and writes the context4d layout render_context_assets already consumes -- so the renderer
runs unchanged.

CC and TD only: no bone/kidney landmark labels, no spine mesh. The grayscale CT under the MPR is
the anatomical context.

The V/HU table is NOT computed here. It comes from `raw_duct_table.py`, which reads the full-res
unclipped DICOM on the machine that holds it; this crop's `hu` is the context grid (2x2 mean-pooled,
and clipped at 3071 in any context built before 2026-07-15), so measuring CT numbers off it would
report the wrong quantity off the wrong grid. Pass that table in with --table and it rides along in
meta for the renderer to caption and write out.

  python -m cc_reservoir.scripts.bake_edit_context <edit_crop_*.npz> <out_parent_dir> [--table duct.csv]
  python -m cc_reservoir.scripts.bake_edit_context --selftest
"""
import csv
import os
import sys

import numpy as np

from cc_reservoir.measure.surface import mesh_from_mask

CC, TD = 1, 2


def load_table(path):
    """duct.csv (raw_duct_table.py) -> (fields, rows as float32). Frame order = row order."""
    with open(path, newline="") as fh:
        r = list(csv.DictReader(fh))
    if not r:
        raise SystemExit(f"[bake] {path} is empty")
    fields = list(r[0].keys())
    return fields, np.array([[float(row[f]) for f in fields] for row in r], np.float32)


def duct_masters(seg):
    """Per-frame CC/TD masks + the union extent, for camera framing only.

    The union is NOT a measurement extent: CC and TD bump, so the painted CC/TD boundary drifts
    between frames and a union puts those voxels in both masters. Measurement uses each frame's own
    seg (see raw_duct_table). This is only used to pick one camera box for the whole series.
    """
    return (seg == CC).any(0) | (seg == TD).any(0)


def bake(crop_npz, out_parent, table_path=None):
    d = np.load(crop_npz, allow_pickle=True)
    hu, seg = d["hu"], d["seg"]
    vox = tuple(float(v) for v in d["voxel"])
    tag = str(d["tag"])
    F = len(hu)

    master = duct_masters(seg)
    if not master.any():
        raise SystemExit("[bake] hand seg has no CC/TD voxels")

    extra = {}
    peak = int(d["auto_peak_i"])
    if table_path:
        fields, rows = load_table(table_path)
        if len(rows) != F:
            raise SystemExit(f"[bake] {table_path} has {len(rows)} frames, crop has {F}")
        extra = dict(raw_table=rows, raw_fields=np.array(fields))
        # peak = the frame the CISTERN is actually most opacified in, by raw CT number. Median, not
        # mean: the duct's HU is right-skewed and its rim is partial-volume dark.
        peak = int(np.argmax(rows[:, fields.index("HU_cc_median")]))

    out = os.path.join(out_parent, "context4d_" + tag)
    os.makedirs(out, exist_ok=True)
    for i in range(F):
        # CC/TD only -- drop any landmark labels the editor painted (kidney), keep the CT as context
        s = np.where(seg[i] <= TD, seg[i], 0).astype(np.uint8)
        cv, cf = mesh_from_mask(s == CC, vox)
        tv, tf = mesh_from_mask(s == TD, vox)
        np.savez_compressed(os.path.join(out, f"f{i}.npz"), hu=hu[i].astype(np.int16), seg=s,
                            cc_v=cv, cc_f=cf, td_v=tv, td_f=tf)
        print(f"  f{i}: CC {int((s == CC).sum()):5d} vox  TD {int((s == TD).sum()):5d} vox")

    # bone_v/bone_f empty: render_context_assets guards on `.size`, so no spine is drawn anywhere.
    np.savez_compressed(
        os.path.join(out, "meta.npz"), voxel=np.array(vox), n_frames=F, peak_i=peak,
        duct_centroid=(np.argwhere(master).mean(0) * np.asarray(vox)).astype(np.float32),
        bone_v=np.zeros((0, 3), np.float32), bone_f=np.zeros((0, 3), np.int32),
        raw_session=d["raw_session"], raw_sub=d["raw_sub"], seg_source="hand_edit_sidecar", **extra)
    print(f"[bake] {tag}: peak frame {peak} (auto seg said {int(d['auto_peak_i'])}) -> {out}")
    return out


def _selftest():
    vox = (1.0, 1.0, 1.0)
    seg = np.zeros((3, 12, 12, 12), np.uint8)
    seg[:, 4:8, 4:8, 0:6] = CC
    seg[:, 4:8, 4:8, 6:12] = TD
    seg[0, 4:8, 4:8, 6] = CC                       # boundary drift: f0 calls this slice CC, f1/f2 TD
    m = duct_masters(seg)
    assert m.sum() == 4 * 4 * 12, "union extent covers both labels"
    # the drift voxels are exactly why measurement must not use a union of per-frame masters
    cc_u, td_u = (seg == CC).any(0), (seg == TD).any(0)
    assert (cc_u & td_u).sum() == 16, "union masters overlap on the drifted slice"
    for i in range(3):
        assert not ((seg[i] == CC) & (seg[i] == TD)).any(), "a single frame never double-labels"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        a = [x for x in sys.argv[1:] if not x.startswith("--")]
        t = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else None
        bake(a[0], a[1] if len(a) > 1 else ".", t)
