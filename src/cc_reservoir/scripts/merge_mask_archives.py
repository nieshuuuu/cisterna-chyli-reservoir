"""Merge two generations of `f<N>_edit.npz` mask archives, keeping the best copy of each frame.

WHY THIS EXISTS
    Under Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\, the painter's
    working masks (`2_CT_volumes_and_painter/data_fullres`) are uniformly newer than the July 2026
    snapshot beside them (`3_old_masks_superseded`) -- but newer is not better everywhere. Some
    frames were legitimately corrected downward, some were wiped, and at least one wipe destroyed
    hand work that only survives in the older copy. Picking one archive wholesale loses data either
    way; this picks per frame and says why. Its output is `1_masks_USE_THESE`.

RULE
    Take NEW, except where NEW dropped content that OLD still has AND that content was human:
      * NEW empty, OLD not      -> recover from OLD, unless OLD was itself just the producer's
                                   auto-seed echoed back (then the wipe was correct)
      * NEW lost more than      -> flagged for review; NEW is still taken unless --recover-losses
        --loss-frac of OLD
    "Was human" is decided by comparing OLD against the auto-seed in the frame's own `f<N>.npz`,
    the same test as scripts/audit_edit_sidecars.py.

USAGE
    python -m cc_reservoir.scripts.merge_mask_archives --new <data_fullres> --old <snapshot> \
        --seeds <dir with f<N>.npz> [--out <dir> --write] [--despeckle 20] [--loss-frac 0.10]

Dry run by default: nothing is written without --write, and neither input is ever modified.
"""
import argparse
import json
import os
import sys

import numpy as np

LABEL_NAME = {1: "CC", 2: "TD", 3: "lymph", 4: "bone"}


def sidecars(root):
    """{(acq_dir_name, 'fN'): path} for every mask sidecar under `root`."""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith("_edit.npz"):
                out[(d, f.split("_")[0])] = os.path.join(p, f)
    return out


def load_seg(path):
    with np.load(path, allow_pickle=True) as d:
        src = str(d["source"]) if "source" in d.files else None
        return d["seg"].astype(np.uint8), src


def load_seed(seeds_root, acq, frame):
    p = os.path.join(seeds_root, acq, frame + ".npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as d:
        return d["seg"].astype(np.uint8) if "seg" in d.files else None


def is_human(seg, seed):
    """False ONLY when the mask is the producer's auto-seed unchanged.

    This gates whether a wipe in the newer archive is treated as a correct cleanup or as lost
    work, so it must not be stricter than that. Requiring that a SEED voxel changed classed a
    seed-superset -- the seed plus real hand-drawn additions -- as "only the auto-seed", and the
    wipe branch then deleted those additions while logging that nothing human was lost. Same test
    as viz.ipad_paint.seed_stamp: did anything at all change?"""
    if seed is None or seed.shape != seg.shape:
        return True                               # nothing to compare against: human by construction
    return bool((seg != seed).any())


def despeckle(seg, min_size):
    """Drop connected components smaller than `min_size` voxels, per label."""
    from scipy import ndimage
    out = seg.copy()
    removed = 0
    st = np.ones((3, 3, 3), bool)
    for lid in LABEL_NAME:
        m = seg == lid
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=st)
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        # sizes[0] is the background; drop it BY ID, not by slicing off the first element of the
        # filtered list -- background is only in that list when it happens to be small, so `[1:]`
        # silently spared one real speck (the single stray label-4 voxel, for one).
        small_ids = np.nonzero(sizes < min_size)[0]
        small_ids = small_ids[small_ids != 0]
        if not small_ids.size:
            continue
        small = np.isin(lab, small_ids) & m
        removed += int(small.sum())
        out[small] = 0
    return out, removed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True, help="the newer archive (wins by default)")
    ap.add_argument("--old", required=True, help="the older archive (fallback for lost content)")
    ap.add_argument("--seeds", help="tree holding the f<N>.npz frames; defaults to --new")
    ap.add_argument("--out", help="destination for the merged archive")
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--despeckle", type=int, default=0, metavar="N",
                    help="drop connected components under N voxels per label")
    ap.add_argument("--loss-frac", type=float, default=0.10,
                    help="flag a frame when NEW holds less than (1-frac) of OLD")
    ap.add_argument("--recover-losses", action="store_true",
                    help="also take OLD for flagged partial losses, not just for wipes")
    ap.add_argument("--json", help="write the decision table here")
    a = ap.parse_args(argv)
    seeds_root = a.seeds or a.new
    if a.write and not a.out:
        ap.error("--write needs --out")

    new, old = sidecars(a.new), sidecars(a.old)
    keys = sorted(set(new) | set(old))
    if not keys:
        print("no sidecars found in either archive", file=sys.stderr)
        return 1

    rows = []
    for acq, frame in keys:
        r = {"acq": acq.replace("context4d_", ""), "dir": acq, "frame": frame,
             "n_new": None, "n_old": None, "take": "new", "why": "", "old_human": None}
        sn = so = None
        if (acq, frame) in new:
            sn, src_n = load_seg(new[(acq, frame)])
            r["n_new"] = int((sn > 0).sum())
            r["source_new"] = src_n
        if (acq, frame) in old:
            so, src_o = load_seg(old[(acq, frame)])
            r["n_old"] = int((so > 0).sum())
            r["source_old"] = src_o

        if sn is None:
            r["take"], r["why"] = "old", "only in the old archive"
        elif so is None:
            r["take"], r["why"] = "new", "only in the new archive"
        elif r["n_new"] == 0 and r["n_old"] > 0:
            seed = load_seed(seeds_root, acq, frame)
            r["old_human"] = is_human(so, seed)
            if r["old_human"]:
                r["take"], r["why"] = "old", "NEW is empty and OLD is hand work -- recovered"
            else:
                r["take"], r["why"] = "new", "NEW is empty and OLD was only the auto-seed -- wipe kept"
        elif r["n_old"] and r["n_new"] < (1.0 - a.loss_frac) * r["n_old"]:
            seed = load_seed(seeds_root, acq, frame)
            r["old_human"] = is_human(so, seed)
            lost = 1.0 - r["n_new"] / r["n_old"]
            if a.recover_losses and r["old_human"]:
                r["take"], r["why"] = "old", f"NEW lost {lost:.0%} of OLD -- recovered"
            else:
                r["take"], r["why"] = "new", f"REVIEW: NEW lost {lost:.0%} of OLD"
        else:
            r["why"] = "newer"
        rows.append(r)

        if a.write:
            src = new[(acq, frame)] if r["take"] == "new" else old[(acq, frame)]
            seg = sn if r["take"] == "new" else so
            if a.despeckle:
                seg, nrem = despeckle(seg, a.despeckle)
                r["despeckled"] = nrem
            dst_dir = os.path.join(a.out, acq)
            os.makedirs(dst_dir, exist_ok=True)
            np.savez_compressed(os.path.join(dst_dir, frame + "_edit.npz"),
                                seg=seg, source=os.path.basename(src),
                                merged_from=r["take"], merge_reason=r["why"])

    # Write the decision table BEFORE printing anything. The output tree is already on disk by
    # now, and a formatting bug in the summary below must never be what destroys the only record
    # of which archive each frame came from.
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"decision table -> {a.json}")

    def n(v):
        """Counts are None for a frame that exists in only one archive."""
        return f"{v:>7,}" if v is not None else f"{'-':>7}"

    takes_old = [r for r in rows if r["take"] == "old"]
    review = [r for r in rows if r["why"].startswith("REVIEW")]
    print(f"{len(rows)} frames: {len(rows) - len(takes_old)} from NEW, {len(takes_old)} from OLD")
    if takes_old:
        print("\nrecovered from the older archive:")
        for r in takes_old:
            print(f"   {r['acq']:<34} {r['frame']:<4} new={n(r['n_new'])} old={n(r['n_old'])}"
                  f"   {r['why']}")
    if review:
        print("\nflagged, NEW kept -- look at these before you trust the volumes:")
        for r in review:
            print(f"   {r['acq']:<34} {r['frame']:<4} new={n(r['n_new'])} old={n(r['n_old'])}"
                  f"   old_is_hand_work={r['old_human']}   {r['why']}")
    if a.despeckle and a.write:
        tot = sum(r.get("despeckled", 0) for r in rows)
        print(f"\ndespeckle(<{a.despeckle} vox): removed {tot:,} voxels")
    if not a.write:
        print("\nDRY RUN -- nothing written. Add --out <dir> --write to materialise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
