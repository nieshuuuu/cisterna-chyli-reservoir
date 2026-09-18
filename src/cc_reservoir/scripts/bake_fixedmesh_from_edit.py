"""Build the fixedmesh npz that render_cc_td_4d.py already reads, from a HAND seg + raw CT.

render_cc_td_4d is the producer of the archive's fill.gif look -- CC green / TD teal, brightening to
pale where contrast washes in, per-timepoint deforming mesh on a fixed camera. It consumes the npz
phase2c_per_timepoint emits from the AUTO seg. This emits the same keys from the hand seg + the raw
DICOM crop (export_cc_raw), so the renderer runs unchanged and the two are directly comparable.

Keys it must provide: opac, Vcc, Vtd, vn/fn/is_cc (master), and per opacified frame i:
wv_i/wf_i (mesh), wcc_i (per-FACE CC flag), wenh_i (per-FACE contrast).

wenh_i is the LUMEN's local raw CT number, not HU sampled at the surface: the iso-surface sits on the
partial-volume rim, so sampling there paints a full duct dark. See render_cc_fill.lumen_hu_field.

The MASTER is resolved by majority vote per voxel, not a union: CC and TD bump, so the painted
boundary drifts between frames, and a plain union would put the drifted voxels in BOTH labels.

  python -m cc_reservoir.scripts.bake_fixedmesh_from_edit <cc_raw_*.npz> <out.npz>
  python -m cc_reservoir.scripts.bake_fixedmesh_from_edit --selftest
"""
import os
import sys

import numpy as np

from cc_reservoir.measure.surface import mesh_from_mask
from cc_reservoir.scripts.render_cc_fill import lumen_hu_field

CC, TD = 1, 2


def master_labels(seg):
    """Per-voxel CC/TD by majority vote over frames (0 where the duct never is). Disjoint by
    construction, so the master mesh never double-labels the drifted CC/TD boundary."""
    ncc = (seg == CC).sum(0)
    ntd = (seg == TD).sum(0)
    out = np.zeros(seg.shape[1:], np.uint8)
    out[(ncc > 0) | (ntd > 0)] = TD
    out[ncc >= ntd] = CC                       # ties -> CC (the caudal reservoir owns its boundary)
    out[(ncc == 0) & (ntd == 0)] = 0
    return out


def _merge(cc_vf, td_vf):
    """(verts, faces, is_cc_per_face) for the CC and TD meshes concatenated into one surface."""
    (cv, cf), (tv, tf) = cc_vf, td_vf
    v = np.vstack([cv, tv]) if len(tv) else cv
    f = np.vstack([cf, tf + len(cv)]) if len(tf) else cf
    is_cc = np.concatenate([np.ones(len(cf), bool), np.zeros(len(tf), bool)])
    return v.astype(np.float32), f.astype(np.int32), is_cc


def _face_vals(field, verts, faces, vox):
    """Per-FACE value = the field at the face's centroid (render_cc_td_4d colours by cell_data)."""
    c = verts[faces].mean(1)
    idx = np.clip(np.round(c / np.array(vox)).astype(int), 0, np.array(field.shape) - 1)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)


def bake(src, out_npz):
    d = np.load(src, allow_pickle=True)
    hu, seg = d["hu"], d["seg"]
    vox = tuple(float(v) for v in d["voxel"])
    nF = len(hu)
    vvol = float(np.prod(vox))

    m = master_labels(seg)
    vn, fn, is_cc = _merge(mesh_from_mask(m == CC, vox),
                           mesh_from_mask(m == TD, vox))

    out = dict(vn=vn, fn=fn, is_cc=is_cc, voxel=np.asarray(vox), tag=str(d["tag"]))
    # every frame is kept: the hand seg only marks what the painter could SEE, so a frame with little
    # duct is bolus timing, not a failed frame -- and dropping it would hide the arrival.
    opac, Vcc, Vtd = [], [], []
    for i in range(nF):
        v, f, wcc = _merge(mesh_from_mask(seg[i] == CC, vox),
                           mesh_from_mask(seg[i] == TD, vox))
        if not len(f):
            print(f"  f{i}: no duct, skipped")
            continue
        field = lumen_hu_field(hu[i], seg[i] > 0, vox)
        out[f"wv_{i}"] = v; out[f"wf_{i}"] = f; out[f"wcc_{i}"] = wcc
        out[f"wenh_{i}"] = _face_vals(field, v, f, vox)
        opac.append(i)
        Vcc.append(int((seg[i] == CC).sum()) * vvol)
        Vtd.append(int((seg[i] == TD).sum()) * vvol)
        print(f"  f{i}: CC {int((seg[i] == CC).sum()):6d} vox ({Vcc[-1]:6.0f} uL)  "
              f"TD {int((seg[i] == TD).sum()):6d} vox ({Vtd[-1]:6.0f} uL)  "
              f"faces {len(f):6d}  lumen HU p50 {np.percentile(out[f'wenh_{i}'], 50):6.0f}")
    out["opac"] = np.array(opac, int)
    out["Vcc"] = np.array(Vcc, np.float32)
    out["Vtd"] = np.array(Vtd, np.float32)
    np.savez_compressed(out_npz, **out)
    print(f"[bake_fixedmesh] {d['tag']}: {len(opac)} frames -> {out_npz} "
          f"({os.path.getsize(out_npz) / 1e6:.0f} MB)")


def _selftest():
    seg = np.zeros((3, 10, 10, 12), np.uint8)
    seg[:, 3:7, 3:7, 0:6] = CC
    seg[:, 3:7, 3:7, 6:12] = TD
    seg[0, 3:7, 3:7, 6] = CC                    # drift: f0 calls this slice CC, f1/f2 call it TD
    m = master_labels(seg)
    assert not ((m == CC) & (m == TD)).any(), "master is disjoint"
    assert m[5, 5, 6] == TD, "majority (2 of 3 frames say TD) wins the drifted slice"
    assert m[5, 5, 0] == CC and m[5, 5, 11] == TD, "undisputed voxels keep their label"
    assert (m > 0).sum() == (seg > 0).any(0).sum(), "master covers the whole extent"
    v, f, is_cc = _merge((np.zeros((4, 3), np.float32), np.array([[0, 1, 2]], np.int32)),
                         (np.ones((5, 3), np.float32), np.array([[0, 1, 2], [1, 2, 3]], np.int32)))
    assert len(v) == 9 and len(f) == 3, "meshes concatenate"
    assert is_cc.tolist() == [True, False, False], "per-FACE flag follows the source mesh"
    assert f[1].tolist() == [4, 5, 6], "TD faces are re-indexed past the CC verts"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        bake(sys.argv[1], sys.argv[2])
