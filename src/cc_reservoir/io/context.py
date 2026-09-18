"""Load the `seg` label map from a context npz, preferring a hand-edit sidecar.

The editor (viz/edit_seg.py) writes corrected masks to `<stem>_edit.npz` next to
the source npz. Everything that DISPLAYS the auto `seg` reads it through here, so a
hand correction shows up without touching the producer's original file — SSoT: the
sidecar is the authoritative seg when it exists, the producer's `seg` otherwise.

Scope: this affects the seg OVERLAY only. Products the producer derives from the
auto seg (the 3D meshes cc_v/cc_f/td_v/td_f, and meta.vol_table volumes) are baked
at production time and are NOT recomputed here; making those reflect edits means
re-deriving them from the edited seg (see the note where this is called).
"""
import os

import numpy as np


def edit_sidecar(npz_path):
    """Path of the hand-edit sidecar for a context npz: `<stem>_edit.npz`."""
    return os.path.splitext(str(npz_path))[0] + "_edit.npz"


def load_seg(npz_path, on_unedited="sidecar"):
    """`seg` (uint8) from a context npz, from the `_edit.npz` sidecar if it exists.

    `on_unedited` decides what to do with a sidecar the painter recorded as NOT hand-edited
    (`edited=False`, i.e. it just echoes the producer's own auto seg):
      "sidecar" (default) -- use it anyway. Preserves the historical behaviour exactly, so results
                             computed before the flag existed still reproduce.
      "base"              -- fall back to the producer's seg.
      "error"             -- raise. Use this in measurement code that must only ever see human work.
    Sidecars written before the flag existed carry no `edited` key; they are treated as edited here,
    because there is nothing in the file to say otherwise. Use `sidecar_info` / the
    `audit_edit_sidecars` script to classify those by content instead of by trust."""
    if on_unedited not in ("sidecar", "base", "error"):
        raise ValueError(f"on_unedited must be sidecar|base|error, got {on_unedited!r}")
    src = edit_sidecar(npz_path)
    if not os.path.exists(src):
        src = str(npz_path)
    elif on_unedited != "sidecar":
        # with_empty=False: this only needs the provenance keys, and the emptiness scan would
        # inflate the whole ~356 MB mask a second time on every read.
        info = sidecar_info(npz_path, with_empty=False)
        if info["edited"] is False:
            if on_unedited == "error":
                raise ValueError(
                    f"{src} is an unedited sidecar (edited=False, n_diff_seed="
                    f"{info['n_diff_seed']}): it is the auto seg, not a hand mask")
            src = str(npz_path)
    with np.load(src) as d:
        return d["seg"].astype(np.uint8)


def sidecar_info(npz_path, with_empty=True):
    """What the `_edit.npz` sidecar next to `npz_path` actually is.

    Returns {exists, edited, n_diff_seed, n_seed, empty, source, legacy}. `edited` is None for a
    legacy sidecar written before the painter stamped provenance -- unknown, not False. `empty` is
    True when the sidecar's seg has no non-zero voxel, which reads downstream as "this frame has no
    duct" even when the producer's own seg is full.

    The provenance keys are tiny, but `empty` requires inflating the whole mask (~356 MB on a
    full-res frame). Pass `with_empty=False` to skip that scan; `empty` is then left None, which
    already means "unknown" in the returned dict."""
    src = edit_sidecar(npz_path)
    out = {"exists": False, "edited": None, "n_diff_seed": None, "n_seed": None,
           "empty": None, "source": None, "legacy": None}
    if not os.path.exists(src):
        return out
    out["exists"] = True
    with np.load(src, allow_pickle=True) as d:
        files = list(d.files)
        out["legacy"] = "edited" not in files
        if "edited" in files:
            out["edited"] = bool(d["edited"])
        if "n_diff_seed" in files:
            out["n_diff_seed"] = int(d["n_diff_seed"])
        if "n_seed" in files:
            out["n_seed"] = int(d["n_seed"])
        if "source" in files:
            out["source"] = str(d["source"])
        if with_empty:
            # .any() straight off the uint8 array -- `(seg > 0).any()` first materialises a
            # full-size bool temporary, doubling peak memory for an identical answer.
            out["empty"] = not bool(d["seg"].any())
    return out


# meta.npz grew keys over time: most acquisitions carry the 20-key schema (duct_centroid, boundary_z,
# lab_cc, lab_td, ...), a few carry an older 11-key one with `ct_only` instead. Readers that index
# meta directly either raise KeyError or silently miss a key depending on which they hit.
META_DEFAULTS = {
    "n_frames": 1, "peak_i": 0, "downsample": (1, 1, 1),
    "seedless": False, "ct_only": False,
    "duct_centroid": None, "boundary_z": None, "boundary_loukas": None, "boundary_anchor": None,
    "cc_caud": None, "td_ref_mm": None, "cc_caudal_mm": None, "caudal_is_cc": None,
    "lab_cc": None, "lab_td": None, "raw_session": "", "raw_sub": "",
}


def load_meta(path):
    """`meta.npz` as a plain dict, with the keys older schemas omit filled in from META_DEFAULTS.

    Accepts the meta.npz path or the directory holding it. Required geometry keys (voxel, crop_lo,
    crop_hi, raw_full_shape) are NOT defaulted -- a meta without them is broken, not merely old."""
    p = str(path)
    if os.path.isdir(p):
        p = os.path.join(p, "meta.npz")
    with np.load(p, allow_pickle=True) as d:
        out = {}
        for k in d.files:
            v = d[k]                      # read ONCE: d[k] decompresses on every subscript
            if v.ndim == 0:
                out[k] = v.item()
            elif v.size <= 16:
                # the small geometry vectors (voxel, crop_lo/hi, downsample, raw_full_shape...).
                # tuple, not list, so they compare equal to META_DEFAULTS.
                out[k] = tuple(v.tolist())
            else:
                # lab_cc / lab_td are whole label VOLUMES. .tolist() on those builds hundreds of
                # millions of Python ints -- gigabytes of RAM and seconds of CPU, for a key most
                # callers never touch. Hand back the ndarray.
                out[k] = v
    for k, v in META_DEFAULTS.items():
        out.setdefault(k, v)
    out["_schema_keys"] = sorted(k for k in out if not k.startswith("_"))
    return out


def _selftest(tmp="/tmp/cc_ctx_context_test"):
    os.makedirs(tmp, exist_ok=True)
    base = os.path.join(tmp, "f0.npz")
    np.savez_compressed(base, hu=np.zeros((2, 2, 2), np.int16), seg=np.ones((2, 2, 2), np.uint8))
    assert (load_seg(base) == 1).all(), "base seg used when no sidecar"
    np.savez_compressed(edit_sidecar(base), seg=np.full((2, 2, 2), 2, np.uint8))
    assert (load_seg(base) == 2).all(), "sidecar preferred when present"
    assert load_seg(base).dtype == np.uint8, "uint8"

    # a legacy sidecar carries no provenance: unknown, and treated as edited
    i = sidecar_info(base)
    assert i["exists"] and i["legacy"] and i["edited"] is None, i
    assert (load_seg(base, on_unedited="error") == 2).all(), "legacy sidecar is not refused"

    # an UNEDITED sidecar echoes the auto seg; the policy decides whether it counts
    np.savez_compressed(edit_sidecar(base), seg=np.ones((2, 2, 2), np.uint8),
                        source="f0.npz", edited=np.bool_(False),
                        n_diff_seed=np.int64(0), n_seed=np.int64(8))
    i = sidecar_info(base)
    assert i["edited"] is False and i["n_diff_seed"] == 0 and not i["legacy"], i
    assert (load_seg(base) == 1).all(), "default policy still uses the sidecar"
    assert (load_seg(base, on_unedited="base") == 1).all(), "falls back to the base seg"
    try:
        load_seg(base, on_unedited="error")
        raise AssertionError("on_unedited='error' must refuse an unedited sidecar")
    except ValueError:
        pass

    # an all-zero sidecar is flagged, so a blanked frame is not mistaken for "no duct here"
    np.savez_compressed(edit_sidecar(base), seg=np.zeros((2, 2, 2), np.uint8),
                        source="f0.npz", edited=np.bool_(True),
                        n_diff_seed=np.int64(8), n_seed=np.int64(8))
    assert sidecar_info(base)["empty"] is True, "empty sidecar flagged"

    # meta defaulting: an old-schema meta must not KeyError on the new keys
    np.savez_compressed(os.path.join(tmp, "meta.npz"), voxel=np.array([0.8, 0.8, 0.5]),
                        crop_lo=np.zeros(3, np.int32), crop_hi=np.full(3, 2, np.int32),
                        raw_full_shape=np.full(3, 2, np.int32), n_frames=np.int64(1),
                        ct_only=np.bool_(True))
    m = load_meta(tmp)
    assert m["boundary_z"] is None and m["ct_only"] is True and m["downsample"] == (1, 1, 1), m
    assert m["voxel"] == (0.8, 0.8, 0.5), m["voxel"]

    # a big array member (lab_cc/lab_td are whole label volumes) must stay an ndarray, not be
    # exploded into nested Python lists
    np.savez_compressed(os.path.join(tmp, "meta.npz"), voxel=np.array([0.8, 0.8, 0.5]),
                        crop_lo=np.zeros(3, np.int32), crop_hi=np.full(3, 2, np.int32),
                        raw_full_shape=np.full(3, 2, np.int32),
                        downsample=np.ones(3, np.int32),
                        lab_cc=np.zeros((8, 8, 8), np.uint8))
    m = load_meta(tmp)
    assert isinstance(m["lab_cc"], np.ndarray) and m["lab_cc"].shape == (8, 8, 8), type(m["lab_cc"])
    assert m["downsample"] == (1, 1, 1), m["downsample"]

    # the emptiness scan must be skippable, and skipping it must not change any other field
    np.savez_compressed(edit_sidecar(base), seg=np.zeros((2, 2, 2), np.uint8),
                        source="f0.npz", edited=np.bool_(True),
                        n_diff_seed=np.int64(8), n_seed=np.int64(8))
    full, lean = sidecar_info(base), sidecar_info(base, with_empty=False)
    assert full["empty"] is True and lean["empty"] is None, (full, lean)
    assert {k: v for k, v in full.items() if k != "empty"} == \
           {k: v for k, v in lean.items() if k != "empty"}, "with_empty changed something else"
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
