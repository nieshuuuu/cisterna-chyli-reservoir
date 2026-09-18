"""CC alone, over time, surface coloured by the lumen's raw CT number. Renders where a GL context is.

Reads the raw CC crop (export_cc_raw.py): full-resolution 0.781 x 0.781 x 0.5 mm HU, unclipped, with
the hand seg on the same grid. Per frame it rebuilds the CC surface, so the duct's motion is real
geometry rather than a fixed mesh recoloured.

COLOUR = the LUMEN's local raw CT number, not the CT number at the surface itself. The iso-surface
sits on the partial-volume rim, which is far darker than the lumen it bounds, so sampling HU there
would paint a full duct dark and report the segmentation's edge instead of its contrast. See
`lumen_hu_field`.

CAMERA is derived, not guessed (see `best_view`). Fixed across frames, and so is the colour scale --
either one retracking per frame would turn filling into swimming.

  DISPLAY=:0 python -m cc_reservoir.scripts.render_cc_fill <cc_raw_*.npz> <out_dir> [--clim LO HI]
  python -m cc_reservoir.scripts.render_cc_fill --selftest
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

CMAP = "inferno"          # dark -> white with HU: reads as "brighter = more contrast", like the
#                           teal->white fill the earlier renders used, but perceptually uniform.
TAUBIN_ITER = 90          # volume-preserving, so more iterations cost calibre nothing


def lumen_hu_field(hu, seg, vox, sigma_mm=1.5):
    """Local raw CT number OF THE LUMEN at every point (normalized convolution).

    Gaussian-weighted mean of HU over the SEGMENTED voxels only, so the partial-volume rim and the
    surrounding fat never enter the average. This matters: the iso-surface sits exactly on that rim,
    so sampling raw HU at a vertex reports the segmentation's dark edge, not the contrast it
    encloses -- a full duct would render dark. Dividing by the same-blurred mask renormalises away
    the fraction of the window that is outside the duct.

    A local field, not a per-z profile: the CC is a flattened sac and its contrast is not uniform
    across a cross-section, so a z-profile paints it in horizontal bands and hides where the
    contrast actually sits.
    """
    s = tuple(sigma_mm / v for v in vox)
    m = seg.astype(np.float32)
    w = gaussian_filter(m, s)
    f = gaussian_filter(np.where(seg, hu, 0).astype(np.float32), s)
    return f / np.maximum(w, 1e-6)


def _rot(v, axis, deg):
    """Rodrigues: rotate v about a unit axis."""
    a = np.deg2rad(deg); k = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k @ v) * (1 - np.cos(a))


def best_view(verts, az_deg=38, el_deg=14):
    """(centre, view_direction, up) that shows the most of the duct.

    SVD gives its own axes: the 1st is length, the 3rd is the direction it is thinnest in. Looking
    straight down that 3rd axis maximises projected area -- but the CC is a flattened sac (~86 x 18 x
    10 mm), so face-on it reads as a featureless plate with no depth. Swinging off that face by
    `az_deg` about the long axis, plus a little elevation, brings the second and third axes into view
    together: the classic 3/4 view, where the sac looks like a solid instead of a silhouette.
    """
    c = verts.mean(0)
    _, _, vt = np.linalg.svd(verts - c, full_matrices=False)
    up, view = vt[0], vt[2]
    view = _rot(view, up, az_deg)                    # swing around the long axis -> depth
    right = np.cross(view, up); right /= np.linalg.norm(right)
    view = _rot(view, right, el_deg)                 # tip slightly -> see along its length too
    return c, view / np.linalg.norm(view), up


def frame_extent(verts, centre, view, up):
    """(half_height, half_width) of the point cloud in the camera's own frame, for a tight fit."""
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    d = verts - centre
    return float(np.abs(d @ up).max()), float(np.abs(d @ right).max())


def _selftest():
    vox = (1.0, 1.0, 1.0)
    hu = np.full((16, 16, 16), -100, np.int16)        # fat around the duct: must never tint the lumen
    seg = np.zeros((16, 16, 16), bool)
    seg[6:11, 6:11, 3:13] = True
    hu[seg] = 900
    f = lumen_hu_field(hu, seg, vox, sigma_mm=1.5)
    # a point ON the duct's surface must report the LUMEN (900), not the rim/fat it touches
    assert abs(f[6, 8, 8] - 900) < 25, f"surface must read the lumen, got {f[6, 8, 8]:.0f}"
    assert abs(f[8, 8, 8] - 900) < 1, f"interior must read the lumen, got {f[8, 8, 8]:.0f}"
    hu2 = hu.copy(); hu2[seg] = 900; hu2[8, 8, 3:13] = 4000     # bright core -> field must rise
    f2 = lumen_hu_field(hu2, seg, vox, sigma_mm=1.5)
    assert f2[8, 8, 8] > f[8, 8, 8], "a brighter core must raise the local lumen CT number"
    # a flat sheet: face-on is its normal (z); the 3/4 view must swing OFF that, keeping up = long axis
    pts = np.random.RandomState(0).rand(500, 3) * np.array([10.0, 4.0, 0.2])
    _, view, up = best_view(pts)
    assert abs(abs(up[0]) - 1) < 0.1, f"up should be the long axis, got {up}"
    assert abs(view[2]) < 0.99, f"view must not stay face-on (that reads as a flat plate), got {view}"
    assert abs(np.linalg.norm(view) - 1) < 1e-6, "view is a unit vector"
    print("selftest OK")


def main():
    import pyvista as pv
    from PIL import Image, ImageDraw
    from cc_reservoir.measure.surface import mesh_from_mask

    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = np.load(src, allow_pickle=True)
    hu, seg = d["hu"], d["seg"].astype(bool)
    vox = tuple(float(v) for v in d["voxel"])
    tag = str(d["tag"])
    nF = len(hu)
    vvol = float(np.prod(vox))

    mesh, scal, vols = [], [], []
    for i in range(nF):
        v, f = mesh_from_mask(seg[i], vox)   # sigma is capped by the thinnest duct: measure.surface
        field = lumen_hu_field(hu[i], seg[i], vox)
        idx = np.clip(np.round(v / np.array(vox)).astype(int), 0, np.array(seg[i].shape) - 1)
        s = field[idx[:, 0], idx[:, 1], idx[:, 2]]
        pd = pv.PolyData(v.astype(float), np.hstack([np.full((len(f), 1), 3, int), f]).ravel())
        pd.point_data["HU"] = s
        try:
            pd = pd.smooth_taubin(n_iter=TAUBIN_ITER, pass_band=0.05)   # volume-preserving
        except Exception:
            pass
        mesh.append(pd); scal.append(s); vols.append(int(seg[i].sum()) * vvol)

    allv = np.vstack([m.points for m in mesh])
    centre, view, up = best_view(allv)
    hh, hw = frame_extent(allv, centre, view, up)
    if "--clim" in sys.argv:
        k = sys.argv.index("--clim"); clim = (float(sys.argv[k + 1]), float(sys.argv[k + 2]))
    else:
        alls = np.concatenate(scal)
        clim = (float(np.floor(alls.min() / 50) * 50), float(np.ceil(np.percentile(alls, 99) / 50) * 50))
    print(f"[render_cc_fill] {tag}: clim {clim}  half-height {hh:.1f} mm  half-width {hw:.1f} mm")

    H = 1300
    W = max(int(H * (hw + 4) / (hh + 4)), 240)        # frame the duct's own aspect; room for the bar
    imgs = []
    for i in range(nF):
        p = pv.Plotter(off_screen=True, window_size=(W, H), lighting="three lights")
        p.set_background("white")
        for fn in (lambda: p.enable_anti_aliasing("ssaa"), p.enable_ssao):
            try:
                fn()
            except Exception:
                pass
        p.add_mesh(mesh[i], scalars="HU", cmap=CMAP, clim=clim, smooth_shading=True,
                   specular=0.35, specular_power=14, ambient=0.3, diffuse=0.8,
                   scalar_bar_args=dict(title="CC lumen CT number (HU)", vertical=False,
                                        position_x=0.12, position_y=0.02, width=0.76, height=0.05,
                                        title_font_size=16, label_font_size=13, color="black",
                                        n_labels=5, fmt="%.0f"))
        p.enable_parallel_projection()
        p.camera.focal_point = tuple(centre)
        p.camera.position = tuple(centre + view * (hh * 6 + 100))
        p.camera.up = tuple(up)
        p.camera.parallel_scale = hh * 1.06           # identical every frame: filling, not swimming
        imgs.append(Image.fromarray(p.screenshot(return_img=True)).convert("RGB")); p.close()

    med = [float(np.nanmedian(np.where(seg[i], hu[i], np.nan))) for i in range(nF)]
    for i, im in enumerate(imgs):
        ImageDraw.Draw(im).text((8, 8), f"frame {i}    CC median {med[i]:.0f} HU    V {vols[i]:.0f} uL",
                                fill=(0, 0, 0))
    imgs[0].save(os.path.join(out, "cc_fill.gif"), save_all=True, append_images=imgs[1:],
                 duration=700, loop=0)
    imgs[int(np.argmax(med))].save(os.path.join(out, "cc_peak.png"))
    print(f"[render_cc_fill] saved {out}/cc_fill.gif ({W}x{H}, {nF} frames) + cc_peak.png")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
