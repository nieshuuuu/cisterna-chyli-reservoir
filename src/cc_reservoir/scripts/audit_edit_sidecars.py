"""Classify every `f<N>_edit.npz` in a context tree as real hand work or the auto-seed echoed back.

WHY THIS EXISTS
    The painter initialises its editable seg from the producer's automatic seed (the `seg` array
    shipped inside every `f<N>.npz`) and writes whatever is in memory on save. A frame that was
    opened and saved without a stroke therefore lands on disk as a `_edit.npz` that
    `io.context.load_seg`, `measure.context.load_context_seg` and the mesh scripts all read as manual
    ground truth. `viz.ipad_paint.save` now stamps `edited` / `n_diff_seed` so new saves say which
    they are; sidecars written before that stamp existed carry no such key, and the only way to tell
    is to compare them against the seed. That is what this does.

CLASSES
    unedited        seg is identical to the producer's seed -- no human voxel at all. Includes the
                    all-zero-over-all-zero case: that is still the seed echoed back
    seed_superset   every seed voxel is still there and more was added -- the seed was never pruned,
                    so whatever the threshold got wrong is still in the mask
    edited          seed voxels were erased -- somebody actually worked on it
    empty           all-zero sidecar that is NOT merely the seed echoed back. Downstream this reads
                    as "no duct in this frame", including when the producer's own seed is full, so
                    it is called out separately
    no_seed         the frame npz carries no `seg` to compare against; human by construction

USAGE
    python -m cc_reservoir.scripts.audit_edit_sidecars <context_root> [--json out.json] [--csv out.csv]
    python -m cc_reservoir.scripts.audit_edit_sidecars <root> --quarantine-list bad_frames.txt

`<context_root>` is a directory of `context4d_*` folders, or a single such folder.
Read-only: this never writes into the context tree.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

LABEL_NAME = {1: "CC", 2: "TD", 3: "lymph", 4: "bone"}
#     classes that must not be fed to a measurement as if a person had drawn them
SUSPECT = ("unedited", "seed_superset")


def context_dirs(root):
    root = os.path.abspath(root)
    if any(f.startswith("f") and f.endswith("_edit.npz") for f in os.listdir(root)):
        return [root]
    out = [os.path.join(root, d) for d in sorted(os.listdir(root))
           if os.path.isdir(os.path.join(root, d))]
    return [d for d in out if any(f.endswith("_edit.npz") for f in os.listdir(d))]


def classify(edit_seg, seed):
    """Class of a whole sidecar, plus the per-label breakdown.

    Identity is tested FIRST. An all-zero sidecar over an all-zero seed is still the seed echoed
    back, and returning "empty" for it hid those frames from both SUSPECT and the quarantine
    list -- exactly the frames this script exists to catch."""
    if (seed is not None and seed.shape == edit_seg.shape and np.array_equal(edit_seg, seed)):
        return "unedited", {}
    if not edit_seg.any():
        return "empty", {}
    if seed is None or seed.shape != edit_seg.shape:
        return "no_seed", {}
    per = {}
    erased_any = False
    for lid, name in LABEL_NAME.items():
        me, ms = edit_seg == lid, seed == lid
        if not (me.any() or ms.any()):
            continue
        added = int((me & ~ms).sum())
        erased = int((ms & ~me).sum())
        erased_any = erased_any or erased > 0
        per[name] = {"n": int(me.sum()), "n_seed": int(ms.sum()),
                     "added": added, "erased": erased,
                     "seed_kept": round(float((me & ms).sum()) / ms.sum(), 4) if ms.any() else None}
    return ("edited" if erased_any else "seed_superset"), per


def audit_one(cdir):
    rows = []
    for f in sorted(os.listdir(cdir)):
        if not f.endswith("_edit.npz"):
            continue
        frame = f.split("_")[0]
        side = os.path.join(cdir, f)
        base = os.path.join(cdir, frame + ".npz")
        with np.load(side, allow_pickle=True) as d:
            seg = d["seg"].astype(np.uint8)
            stamped = "edited" in d.files
            stamp = bool(d["edited"]) if stamped else None
            source = str(d["source"]) if "source" in d.files else None
        seed = None
        if os.path.exists(base):
            with np.load(base) as d:
                if "seg" in d.files:
                    seed = d["seg"].astype(np.uint8)
        cls, per = classify(seg, seed)
        rows.append({
            "acq": os.path.basename(cdir).replace("context4d_", ""),
            "frame": frame,
            "class": cls,
            "n_vox": int((seg > 0).sum()),
            "n_seed_vox": None if seed is None else int((seed > 0).sum()),
            "seed_empty_too": None if seed is None else not bool(seed.any()),
            "stamped": stamped,
            "stamp_edited": stamp,
            "source": source,
            "labels": per,
            "path": side,
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--json", help="write the full per-frame manifest here")
    ap.add_argument("--csv", help="write a flat one-row-per-frame table here")
    ap.add_argument("--quarantine-list",
                    help="write acq/frame lines for every sidecar that is not real hand work")
    a = ap.parse_args(argv)

    dirs = context_dirs(a.root)
    if not dirs:
        print(f"no context folders with _edit.npz under {a.root}", file=sys.stderr)
        return 1
    rows = []
    for i, d in enumerate(dirs, 1):
        rows.extend(audit_one(d))
        print(f"[{i}/{len(dirs)}] {os.path.basename(d)}", file=sys.stderr, flush=True)

    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print()
    print(f"{len(rows)} sidecars in {len(dirs)} acquisitions")
    for k in ("edited", "seed_superset", "unedited", "empty", "no_seed"):
        if k in counts:
            print(f"   {k:<14} {counts[k]:>4}")

    bad = [r for r in rows if r["class"] in SUSPECT]
    if bad:
        print()
        print("NOT hand work -- do not read these as manual masks:")
        print(f"   {'acq':<34} {'frm':<4} {'class':<14} {'voxels':>9}  labels")
        for r in bad:
            lab = ", ".join(
                f"{k} {v['n']}" + ("" if v["erased"] else " (seed untouched)")
                for k, v in r["labels"].items()) or "whole frame = seed"
            print(f"   {r['acq']:<34} {r['frame']:<4} {r['class']:<14} {r['n_vox']:>9,}  {lab}")

    blanked = [r for r in rows if r["class"] == "empty" and r["seed_empty_too"] is False]
    if blanked:
        print()
        print("all-zero sidecars that blank out a NON-empty auto seg:")
        for r in blanked:
            print(f"   {r['acq']:<34} {r['frame']:<4} seed had {r['n_seed_vox']:,} voxels")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nmanifest -> {a.json}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["acq", "frame", "class", "n_vox", "n_seed_vox", "stamped", "source"])
            for r in rows:
                w.writerow([r["acq"], r["frame"], r["class"], r["n_vox"],
                            r["n_seed_vox"], r["stamped"], r["source"]])
        print(f"table -> {a.csv}")
    if a.quarantine_list:
        with open(a.quarantine_list, "w") as fh:
            for r in bad + blanked:
                fh.write(f"{r['acq']}\t{r['frame']}\t{r['class']}\n")
        print(f"quarantine list -> {a.quarantine_list}  ({len(bad) + len(blanked)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
