"""Geometry / brush / back-projection math for the iPad paint server.

Mirrors viz.ipad_paint._selftest but as pytest cases so it runs with the suite:
    PYTHONPATH=src python3 -m pytest tests/test_ipad_paint.py -v
"""
import numpy as np
import pytest

from cc_reservoir.viz.ipad_paint import (
    AXIS, MARKS, disp_shape, to_display, disp_to_voxel, voxel_to_disp,
    inplane_mm, paint_stroke, paint_stroke_depth, backproject,
    disp_to_voxel_arr, polygon_pixels, lasso_fill, wand_fill, gate_plane,
    band_bounds, render_overlay_png, GATE_RGB, _connected_from, _flood_numpy,
)

SHAPE = (236, 256, 1401)          # a real acquisition shape
VOX = (1.562, 1.562, 0.5)         # anisotropic: z is finer than x/y


def test_disp_shapes():
    assert disp_shape("sagittal", SHAPE) == (1401, 236)   # rows=z, cols=x
    assert disp_shape("coronal", SHAPE) == (1401, 256)    # rows=z, cols=y
    assert disp_shape("axial", SHAPE) == (236, 256)       # rows=x, cols=y


@pytest.mark.parametrize("view", ["sagittal", "coronal", "axial"])
def test_disp_voxel_roundtrip(view):
    H, W = disp_shape(view, SHAPE)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        r = int(rng.integers(0, H)); c = int(rng.integers(0, W)); idx = 11
        x, y, z = disp_to_voxel(view, r, c, idx, SHAPE)
        assert 0 <= x < SHAPE[0] and 0 <= y < SHAPE[1] and 0 <= z < SHAPE[2]
        assert voxel_to_disp(view, x, y, z, SHAPE) == (r, c)


def test_cranial_up_semantics():
    # cranial (max z), anterior (x=0) -> TOP-LEFT of the sagittal display
    m = np.zeros(SHAPE, np.int16); m[0, 128, SHAPE[2] - 1] = 7
    assert tuple(np.argwhere(to_display(m, "sagittal", 128, 0) == 7)[0]) == (0, 0)
    # caudal (z=0) -> BOTTOM row
    m2 = np.zeros(SHAPE, np.int16); m2[0, 128, 0] = 7
    assert np.argwhere(to_display(m2, "sagittal", 128, 0) == 7)[0][0] == SHAPE[2] - 1
    # marks read head-up
    assert MARKS["sagittal"] == ("Cr", "Cd", "A", "P")


def test_slab_mip_identity_and_window():
    vol = np.arange(np.prod((5, 4, 6))).reshape(5, 4, 6).astype(np.int16)
    assert np.array_equal(to_display(vol, "axial", 3, 0), to_display(vol, "axial", 3, 0))
    # slab>0 is a max over the window along the cut axis
    d0 = to_display(vol, "sagittal", 2, 0)
    d1 = to_display(vol, "sagittal", 2, 1)
    assert (d1 >= d0).all()


def test_brush_is_physical_mm_disk_on_one_slice():
    seg = np.zeros(SHAPE, np.uint8)
    n = paint_stroke(seg, SHAPE, "sagittal", 136, [(700, 118)], 3.0, VOX, 1)
    pv = np.argwhere(seg == 1)
    assert n > 0 and (pv[:, 1] == 136).all()             # all on slice y=136
    # 3 mm radius: z (0.5 mm/vox) ~12 vox, x (1.562 mm/vox) ~4 vox
    assert 10 <= np.ptp(pv[:, 2]) + 1 <= 14
    assert 3 <= np.ptp(pv[:, 0]) + 1 <= 5


def test_stroke_is_gap_free():
    seg = np.zeros(SHAPE, np.uint8)
    paint_stroke(seg, SHAPE, "sagittal", 136, [(600, 118), (650, 118), (700, 118)], 1.5, VOX, 2)
    zs = np.unique(np.argwhere(seg == 2)[:, 2])
    assert (np.diff(zs) <= 1).all()


def test_erase_and_undo_record():
    seg = np.zeros(SHAPE, np.uint8)
    paint_stroke(seg, SHAPE, "sagittal", 100, [(500, 118)], 3.0, VOX, 1)
    before = int((seg == 1).sum())
    rec = []
    paint_stroke(seg, SHAPE, "sagittal", 100, [(500, 118)], 3.0, VOX, 0, record=rec)
    assert (seg == 1).sum() < before and rec
    for (x, y, z, old) in reversed(rec):                 # undo restores exactly
        seg[x, y, z] = old
    assert int((seg == 1).sum()) == before


@pytest.mark.parametrize("view,cut", [("sagittal", 1), ("coronal", 0)])
def test_backproject_snaps_to_brightest(view, cut):
    hu = np.zeros(SHAPE, np.int16); seg = np.zeros(SHAPE, np.uint8)
    # place a bright voxel one slice off the painted slice along the cut axis
    painted = [0, 0, 0]; bright = [0, 0, 0]
    painted[0], painted[1], painted[2] = 130, 136, 720
    bright[:] = painted[:]
    bright[cut] = painted[cut] + 4
    hu[tuple(bright)] = 1000
    seg[tuple(painted)] = 1
    moved = backproject(hu, seg, view, painted[cut], slab=8, label=1)
    assert moved == 1
    assert seg[tuple(bright)] == 1 and seg[tuple(painted)] == 0


def test_backproject_slab0_is_noop():
    hu = np.zeros(SHAPE, np.int16); seg = np.zeros(SHAPE, np.uint8)
    seg[130, 136, 720] = 1
    assert backproject(hu, seg, "sagittal", 136, 0, 1) == 0


# --------------------------------------------------------------------------- #
# HTTP handler: a client that hangs up mid-response (iPad Safari cancels
# superseded image requests) must NOT raise or emit a traceback, and must not
# then try to write a 500 on the dead socket.
# --------------------------------------------------------------------------- #
import io
import json

from cc_reservoir.viz import ipad_paint as ip


def _make_session(tmp_path):
    """A tiny on-disk context so Session loads without the real 166 MB volumes."""
    hu = np.zeros((6, 6, 8), np.int16)
    hu[2, 3, 4] = 1000
    seg = np.zeros((6, 6, 8), np.uint8)
    seg[2, 3, 4] = 1
    p = tmp_path / "f0.npz"
    np.savez_compressed(p, hu=hu, seg=seg, voxel=np.array([1.562, 1.562, 0.5]))
    return ip.Session(str(p))


class _BrokenW(io.BytesIO):
    def write(self, b):
        raise BrokenPipeError(32, "Broken pipe")


def _cap(sess, method, path, body=None, broken=False):
    class Srv:
        pass
    srv = Srv(); srv.sess = sess

    class Cap(ip.Handler):
        def __init__(self):
            self.server = srv; self.command = method; self.path = path
            self.client_address = ("127.0.0.1", 0)
            self.rfile = io.BytesIO(json.dumps(body).encode() if body is not None else b"")
            self.wfile = _BrokenW() if broken else io.BytesIO()
            self.headers = {"Content-Length": str(len(self.rfile.getvalue()))}
            self.requestline = f"{method} {path}"; self.request_version = "HTTP/1.1"
            self._status = None; self._hdrs = {}

        def send_response(self, code, msg=None):
            self._status = code

        def send_header(self, k, v):
            self._hdrs[k] = v

        def end_headers(self):
            pass

    c = Cap()
    (c.do_GET if method == "GET" else c.do_POST)()
    return c


def test_parse_acq_handles_mixed_date_and_condition_formats():
    """The folder names use inconsistent date formats and only some carry a condition label."""
    from cc_reservoir.viz.ipad_paint import _parse_acq
    assert _parse_acq("context4d_07_20_22_data_Baseline_Acq10") == ("07/20/22", "Baseline", "Acq10")
    assert _parse_acq("context4d_8_31_22_data_Acq3") == ("8/31/22", "", "Acq3")
    assert _parse_acq("context4d_09_07_22_data_Acq16") == ("09/07/22", "", "Acq16")
    assert _parse_acq("context4d_07_20_22_data_Angiotensin_Acq8") == ("07/20/22", "Angiotensin", "Acq8")


def test_discover_acqs_sorts_chronologically_and_numerically(tmp_path):
    """Acquisitions must sort by real date (not string) and by numeric Acq (not string):
    Aug before Sep despite '8_31' < '09_07' lexically; Acq8/9 before Acq12."""
    from cc_reservoir.viz.ipad_paint import _discover_acqs
    names = ["context4d_09_07_22_data_Acq16", "context4d_8_31_22_data_Acq3",
             "context4d_07_20_22_data_Angiotensin_Acq12", "context4d_07_20_22_data_Angiotensin_Acq8"]
    for nm in names:
        d = tmp_path / nm; d.mkdir()
        np.savez_compressed(d / "f0.npz", hu=np.zeros((4, 4, 4), np.int16),
                            seg=np.zeros((4, 4, 4), np.uint8), voxel=np.array([1.0, 1.0, 1.0]))
    acqs, cur = _discover_acqs(str(tmp_path / "context4d_8_31_22_data_Acq3"))
    order = [a["name"] for a in acqs]
    assert order == [
        "context4d_07_20_22_data_Angiotensin_Acq8",   # July, Acq8 (numeric) before Acq12
        "context4d_07_20_22_data_Angiotensin_Acq12",
        "context4d_8_31_22_data_Acq3",                 # Aug before Sep (real date, not string)
        "context4d_09_07_22_data_Acq16",
    ], order
    assert acqs[cur]["name"] == "context4d_8_31_22_data_Acq3"   # current index points at the opened dir


def test_acq_switch_swaps_volume_and_frames(tmp_path):
    """POST /acq reloads a DIFFERENT acquisition: volume shape, frame list, and seg all swap."""
    # two acquisitions with distinct shapes/frame-counts
    a = tmp_path / "context4d_07_20_22_data_Baseline_Acq10"; a.mkdir()
    for fi in range(3):
        np.savez_compressed(a / f"f{fi}.npz", hu=np.zeros((6, 6, 8), np.int16),
                            seg=np.zeros((6, 6, 8), np.uint8), voxel=np.array([1.562, 1.562, 0.5]))
    b = tmp_path / "context4d_8_31_22_data_Acq3"; b.mkdir()
    for fi in range(2):
        np.savez_compressed(b / f"f{fi}.npz", hu=np.zeros((5, 7, 9), np.int16),
                            seg=np.zeros((5, 7, 9), np.uint8), voxel=np.array([1.0, 1.0, 1.0]))
    sess = ip.Session(str(a))
    init = json.loads(_cap(sess, "GET", "/init").wfile.getvalue())
    assert init["acq_name"] == "context4d_07_20_22_data_Baseline_Acq10"
    assert len(init["acqs"]) == 2 and list(init["shape"]) == [6, 6, 8]
    # switch to the OTHER acquisition (index of Acq3)
    ai = next(i for i, x in enumerate(init["acqs"]) if x["label"] == "Acq3")
    r = json.loads(_cap(sess, "POST", "/acq", {"acq": ai}).wfile.getvalue())
    assert r["ok"] and r["acq_name"] == "context4d_8_31_22_data_Acq3"
    assert list(r["shape"]) == [5, 7, 9] and len(r["frames"]) == 2   # volume + frames swapped
    assert tuple(sess.hu.shape) == (5, 7, 9)                         # in-RAM session actually changed


def test_edit_sidecars_are_not_listed_as_frames(tmp_path):
    """Saving writes f<N>_edit.npz beside the frames. The frame glob must NOT sweep those up: they
    hold only `seg` (no `hu`), so they showed up as bogus 'f0_edit' picker entries, and clicking one
    hit the missing-hu guard — which raised SystemExit and silently killed the request thread."""
    d = tmp_path / "context4d_x"; d.mkdir()
    for i in (0, 1, 2):
        np.savez_compressed(d / f"f{i}.npz", hu=np.zeros((4, 4, 4), np.int16),
                            seg=np.zeros((4, 4, 4), np.uint8), voxel=np.array([1.0, 1.0, 1.0]))
    for i in (0, 2):                                          # exactly what a Save leaves behind
        np.savez_compressed(d / f"f{i}_edit.npz", seg=np.zeros((4, 4, 4), np.uint8), source=f"f{i}.npz")
    assert [lbl for lbl, _ in ip._resolve_frames(str(d))] == ["f0", "f1", "f2"]


def test_init_marks_which_frames_are_edited(tmp_path):
    """The picker must show which frames already carry YOUR saved seg — the sidecar that both this
    tool and io.context.load_seg auto-read in preference to the frame's built-in seg."""
    d = tmp_path / "context4d_y"; d.mkdir()
    for i in (0, 1):
        np.savez_compressed(d / f"f{i}.npz", hu=np.zeros((4, 4, 4), np.int16),
                            seg=np.zeros((4, 4, 4), np.uint8), voxel=np.array([1.0, 1.0, 1.0]))
    np.savez_compressed(d / "f1_edit.npz", seg=np.ones((4, 4, 4), np.uint8), source="f1.npz")
    init = json.loads(_cap(ip.Session(str(d)), "GET", "/init").wfile.getvalue())
    assert init["frames"] == ["f0", "f1"] and init["edited"] == [False, True]


def test_save_marks_the_frame_edited_and_page_shows_it(tmp_path):
    """/save returns refreshed flags so the picker marks the frame immediately, and the sidecar it
    just wrote is not then mistaken for a frame."""
    sess = _make_session(tmp_path)
    assert json.loads(_cap(sess, "GET", "/init").wfile.getvalue())["edited"] == [False]
    r = json.loads(_cap(sess, "POST", "/save", {}).wfile.getvalue())
    assert r["ok"] and r["edited"] == [True]
    assert [lbl for lbl, _ in ip._resolve_frames(str(tmp_path))] == ["f0"]   # sidecar not a frame
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert "✎ edited" in page and "fillFrames(S.frames, S.frame, r.edited)" in page


def test_backdrop_cache_header(tmp_path):
    sess = _make_session(tmp_path)
    c = _cap(sess, "GET", "/backdrop?view=sagittal&idx=4&slab=2&win=contrast&frame=0")
    assert c._status == 200
    assert c._hdrs.get("Cache-Control") == "max-age=86400"      # cacheable -> no refetch on re-scroll
    assert c._hdrs.get("Content-Type") == "image/jpeg"          # backdrop is JPEG (fast to scrub)
    assert c.wfile.getvalue()[:2] == b"\xff\xd8"                # JPEG SOI marker


def test_overlay_cache_header_no_store(tmp_path):
    sess = _make_session(tmp_path)
    c = _cap(sess, "GET", "/overlay?view=sagittal&idx=4&frame=0&g=0")
    assert c._status == 200 and c._hdrs.get("Cache-Control") == "no-store"


def test_client_disconnect_is_swallowed(tmp_path):
    sess = _make_session(tmp_path)
    for path in ("/backdrop?view=sagittal&idx=4&slab=2&win=contrast&frame=0",
                 "/overlay?view=sagittal&idx=4&frame=0&g=1"):
        c = _cap(sess, "GET", path, broken=True)          # must not raise BrokenPipeError
        assert getattr(c, "_client_gone", False) is True


def test_real_error_still_500_on_live_socket(tmp_path):
    sess = _make_session(tmp_path)
    c = _cap(sess, "GET", "/backdrop?view=BOGUS&idx=0&slab=0&win=contrast&frame=0")
    assert c._status == 500


# --------------------------------------------------------------------------- #
# v4/v5: mL volume readout, O(1) incremental counts, coronal-default view,
# native-resolution JPEG backdrop (no server upsampling -> no ringing).
# --------------------------------------------------------------------------- #
def test_counts_reports_ml_and_voxels(tmp_path):
    sess = _make_session(tmp_path)
    c = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())
    assert set(c) == {"vox", "ml", "voxvol_ml"}
    # 1.562*1.562*0.5 mm^3 per voxel, in mL
    assert abs(c["voxvol_ml"] - (1.562 * 1.562 * 0.5) / 1000.0) < 1e-9
    assert c["vox"]["1"] == 1                                    # the one CC voxel we seeded
    assert abs(c["ml"]["1"] - c["vox"]["1"] * c["voxvol_ml"]) < 1e-4


def test_incremental_counts_match_full_recompute(tmp_path):
    sess = _make_session(tmp_path)

    def full():
        return {k: int((sess.seg == k).sum()) for k in (1, 2, 3, 4)}

    meta = json.loads(_cap(sess, "GET", "/meta?view=sagittal").wfile.getvalue())
    idx, H, W = meta["default_idx"], meta["rows"], meta["cols"]
    body = {"view": "sagittal", "idx": idx, "slab": 1, "radius_mm": 2.0, "label": 2,
            "points": [[H * 0.5, W * 0.5]]}
    _cap(sess, "POST", "/stroke", body)
    assert {int(k): v for k, v in sess._count.items()} == full()          # after paint
    _cap(sess, "POST", "/undo", {})
    assert {int(k): v for k, v in sess._count.items()} == full()          # after undo


def test_meta_marks_are_radiological_not_flipped(tmp_path):
    """Flip was removed; /meta returns the fixed radiological marks with no flip param
    honoured (a stray flip=1 must NOT swap L/R)."""
    sess = _make_session(tmp_path)
    m = json.loads(_cap(sess, "GET", "/meta?view=coronal").wfile.getvalue())["marks"]
    assert m == ["Cr", "Cd", "R", "L"]                      # cranial-up, R-on-left (radiological)
    m_flip = json.loads(_cap(sess, "GET", "/meta?view=coronal&flip=1").wfile.getvalue())["marks"]
    assert m_flip == m                                       # flip param is ignored now


def test_stroke_ignores_flip_param(tmp_path):
    """A leftover flip:true in a stroke body must not mirror the write — the column is
    used as-is now that flip is gone."""
    sess = _make_session(tmp_path)
    meta = json.loads(_cap(sess, "GET", "/meta?view=coronal").wfile.getvalue())
    idx, rows, cols = meta["default_idx"], meta["rows"], meta["cols"]
    _cap(sess, "POST", "/stroke", {"view": "coronal", "idx": idx, "slab": 0, "radius_mm": 1.0,
                                   "label": 1, "points": [[rows // 2, 0]], "flip": True,
                                   "gate": False})   # this is about the column, not the HU band
    ys = sorted(set(int(v) for v in np.argwhere(sess.seg == 1)[:, 1]))
    # painted at display col 0; the write must stay at the LOW-column end, not be mirrored
    # to the high end (cols-1). Flip is gone, so the column is taken literally.
    assert ys and min(ys) == 0 and max(ys) < cols - 1


def test_backdrop_is_native_resolution_jpeg(tmp_path):
    """v5: no server-side upsampling (that Lanczos step caused ringing). The backdrop
    comes back at NATIVE display resolution as JPEG; the browser does the one stretch."""
    sess = _make_session(tmp_path)                          # voxel (1.562,1.562,0.5)
    from PIL import Image
    b = _cap(sess, "GET", "/backdrop?view=sagittal&idx=4&slab=0&win=contrast&frame=0").wfile.getvalue()
    im = Image.open(io.BytesIO(b))
    rows, cols = ip.disp_shape("sagittal", sess.hu.shape)   # native (z=8, x=6)
    assert im.format == "JPEG"
    assert (im.height, im.width) == (rows, cols)            # NOT upsampled server-side


def test_redo_reapplies_undone_edit(tmp_path):
    """v10: Redo re-applies the most recently undone stroke exactly, and a NEW edit
    invalidates the redo stack (standard editor semantics)."""
    sess = _make_session(tmp_path)
    c0 = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]
    # paint something
    view, idx = "coronal", sess.idx["coronal"]
    rows, cols = ip.disp_shape(view, sess.hu.shape)
    body = {"view": view, "idx": idx, "slab": 0, "radius_mm": 3.0, "label": 1,
            "points": [[rows // 2, cols // 2]], "depth": "off", "gate": False}
    _cap(sess, "POST", "/stroke", body)
    c1 = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]
    assert c1["1"] > c0["1"]                                # CC grew
    # undo -> back to c0
    u = json.loads(_cap(sess, "POST", "/undo", {}).wfile.getvalue())
    assert u["n"] > 0
    assert json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"] == c0
    # redo -> back to c1
    r = json.loads(_cap(sess, "POST", "/redo", {}).wfile.getvalue())
    assert r["ok"] and r["n"] == u["n"]
    assert json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"] == c1
    # a new edit clears the redo stack -> redo now a no-op
    _cap(sess, "POST", "/stroke", body)
    assert json.loads(_cap(sess, "POST", "/redo", {}).wfile.getvalue())["n"] == 0


def test_overlay_visibility_is_display_only(tmp_path):
    """v10: the ?vis= param hides labels in the rendered overlay WITHOUT touching the seg.
    Hiding all -> fully transparent; hiding one -> fewer opaque px; seg/counts unchanged."""
    from PIL import Image
    sess = _make_session(tmp_path)
    seg_before = sess.seg.copy()
    counts_before = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]

    def opaque(qs):
        png = _cap(sess, "GET", "/overlay?" + qs).wfile.getvalue()
        return int((np.array(Image.open(io.BytesIO(png)))[:, :, 3] > 0).sum())

    view, idx = "coronal", sess.idx["coronal"]
    base = f"view={view}&idx={idx}&frame=0&g=1"
    all_px = opaque(base + "&vis=1,2,3,4")
    cc_px = opaque(base + "&vis=1")
    none_px = opaque(base + "&vis=")
    assert none_px == 0                                    # hide everything -> transparent
    assert 0 < cc_px <= all_px                             # one label -> subset
    assert np.array_equal(sess.seg, seg_before)            # seg untouched
    assert json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"] == counts_before


def _make_multislice_session(tmp_path):
    """seg labelled on TWO different coronal slices (x=2 and x=4) — the grow-across-depth case."""
    hu = np.zeros((8, 6, 8), np.int16)
    seg = np.zeros((8, 6, 8), np.uint8)
    seg[2, 3, 4] = 1                                   # on the slice we will view
    seg[4, 3, 5] = 1                                   # 2 slices away in x, a different display pixel
    p = tmp_path / "f0.npz"
    np.savez_compressed(p, hu=hu, seg=seg, voxel=np.array([1.562, 1.562, 0.5]))
    return ip.Session(str(p))


def test_overlay_ghosts_slab_coverage(tmp_path):
    """Depth-follow paints the duct across several slices, so a single-slice overlay made a correctly
    painted duct look unpainted. The overlay must ghost the labels across the SAME ±slab the CT
    projects, while keeping THIS slice's label solid — that's how you judge 'did I cover it all'."""
    from PIL import Image
    sess = _make_multislice_session(tmp_path)

    def alpha(slab):
        png = _cap(sess, "GET", f"/overlay?view=coronal&idx=2&frame=0&g=1&slab={slab}").wfile.getvalue()
        return np.array(Image.open(io.BytesIO(png)))[:, :, 3]

    a0, a2 = alpha(0), alpha(2)
    assert (a0 == 255).sum() == 1 and (a0 > 0).sum() == 1        # slab 0: ONLY this slice, solid, no ghost
    assert (a2 == 255).sum() == 1                                # slab 2: this slice still solid...
    assert (a2 == ip.GHOST_ALPHA).sum() == 1                     # ...plus the off-slice label as a ghost
    assert (a2 > 0).sum() > (a0 > 0).sum()                       # more coverage visible on the slab


def test_page_overlay_follows_the_slab(tmp_path):
    """The overlay URL carries slab (ghost tracks the CT projection) and the slab slider refreshes
    BOTH layers — otherwise the ghost would go stale when you change slab thickness."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    ov = page.split("function ovUrl(", 1)[1].split("; }", 1)[0]
    assert "slab=${S.slab}" in ov
    handler = page.split('$("slab").oninput=', 1)[1].split("\n", 1)[0]
    assert "refreshBoth()" in handler and "refreshBack()" not in handler


def test_clear_all_wipes_everything_and_is_undoable(tmp_path):
    """v7: 'Clear all' zeroes every label in the whole volume as ONE undo entry; a single
    Undo restores it exactly, and the count cache stays consistent."""
    sess = _make_session(tmp_path)
    c0 = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]
    total0 = sum(c0.values())
    assert total0 > 0                                       # seed has at least one labelled voxel
    r = json.loads(_cap(sess, "POST", "/clear", {}).wfile.getvalue())
    assert r["ok"] and r["n"] == total0                    # cleared exactly the labelled voxels
    c1 = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]
    assert all(v == 0 for v in c1.values())                # every label now empty
    assert int((sess.seg > 0).sum()) == 0                  # and the array really is blank
    u = json.loads(_cap(sess, "POST", "/undo", {}).wfile.getvalue())
    assert u["n"] == total0                                 # one undo, whole volume back
    c2 = json.loads(_cap(sess, "GET", "/counts").wfile.getvalue())["vox"]
    assert c2 == c0                                         # exactly restored
    # count cache == full recompute (no drift)
    assert c2 == {str(k): int((sess.seg == k).sum()) for k in (1, 2, 3, 4)}


def _make_depth_session(tmp_path):
    """A session whose HU has a bright bar 5 slices deep in x at (y=3,z=4), muscle elsewhere.
    The bar has a DISTINCT brightest slice at x=6 (900 HU) so 'snap' has an unambiguous target."""
    import numpy as np
    shape = (12, 8, 10)
    hu = np.full(shape, -100, np.int16)
    hu[4:9, 3, 4] = 600                                     # duct-bright bar, x=4..8 (5 slices deep)
    hu[6, 3, 4] = 900                                       # distinct brightest slice at x=6
    seg = np.zeros(shape, np.uint8)
    d = tmp_path / "ctx"; d.mkdir()
    np.savez_compressed(d / "f0.npz", hu=hu, seg=seg, voxel=np.array([1.562, 1.562, 0.5]))
    return ip.Session(str(d))


def test_depth_snap_relocates_to_brightest_slice(tmp_path):
    """MIP depth-follow 'snap': a stroke drawn on an OFF-duct slice must RELOCATE onto the
    single brightest slice in the slab. The bar is x=4..8 (peak x=6); we draw at idx=1 (off
    the bar) with a slab wide enough to reach it, and assert the label lands at x=6, not x=1."""
    sess = _make_depth_session(tmp_path)
    r, c = ip.voxel_to_disp("coronal", 1, 3, 4, sess.hu.shape)[:2]   # display px; x arg is irrelevant
    body = {"view": "coronal", "idx": 1, "slab": 8, "radius_mm": 0.5, "label": 1,
            "points": [[r, c]], "depth": "snap", "floor": 120.0}
    j = json.loads(_cap(sess, "POST", "/stroke", body).wfile.getvalue())
    assert j["ok"] and j["changed"] >= 1
    xs = sorted(set(int(v) for v in np.argwhere(sess.seg == 1)[:, 0]))
    assert xs == [6]                                       # relocated onto the brightest slice
    assert 1 not in xs                                     # did NOT stay on the drawn slice (idx=1)


def test_depth_grow_spans_multiple_slices(tmp_path):
    """MIP depth-follow 'grow': paints the contiguous bright run across the slices it spans."""
    sess = _make_depth_session(tmp_path)
    r, c = ip.voxel_to_disp("coronal", 6, 3, 4, sess.hu.shape)[:2]
    body = {"view": "coronal", "idx": 6, "slab": 6, "radius_mm": 0.5, "label": 2,
            "points": [[r, c]], "depth": "grow", "floor": 120.0}
    _cap(sess, "POST", "/stroke", body)
    xs = sorted(set(int(v) for v in np.argwhere(sess.seg == 2)[:, 0]))
    assert len(xs) >= 3 and set(xs) <= set(range(4, 9))    # multiple slices, all on the bar


def test_depth_floor_gate_skips_muscle(tmp_path):
    """On a column with no bright (duct) voxel, depth-follow paints nothing."""
    sess = _make_depth_session(tmp_path)
    r, c = ip.voxel_to_disp("coronal", 6, 6, 8, sess.hu.shape)[:2]   # far from the bar -> muscle
    body = {"view": "coronal", "idx": 6, "slab": 6, "radius_mm": 0.5, "label": 1,
            "points": [[r, c]], "depth": "grow", "floor": 120.0}
    j = json.loads(_cap(sess, "POST", "/stroke", body).wfile.getvalue())
    assert j["changed"] == 0 and int((sess.seg > 0).sum()) == 0


def test_depth_band_picks_near_water_duct_over_muscle(tmp_path):
    """v10 pre-contrast (f0) case: the chyle-filled duct is near WATER (~15 HU), DIMMER than
    the muscle beside it (~80 HU). Snap-to-brightest would grab the muscle; the HU band
    [floor,ceiling] with a ceiling below muscle must pick the water-density duct instead.
    Also asserts that a low floor with NO ceiling regresses to grabbing the muscle."""
    import numpy as np
    shape = (16, 8, 10)
    hu = np.full(shape, -90, np.int16)                     # fat bed
    hu[5, 3, 4] = 80                                       # muscle (brighter) at x=5
    hu[11, 3, 4] = 15                                      # near-water duct at x=11
    seg = np.zeros(shape, np.uint8)
    d = tmp_path / "ctx"; d.mkdir()
    np.savez_compressed(d / "f0.npz", hu=hu, seg=seg, voxel=np.array([1.562, 1.562, 0.5]))
    sess = ip.Session(str(d))
    r, c = ip.voxel_to_disp("coronal", 13, 3, 4, sess.hu.shape)[:2]   # drawn a few slices off
    # BAND [-20,45] -> lands on the water duct at x=11
    body = {"view": "coronal", "idx": 13, "slab": 10, "radius_mm": 0.5, "label": 1,
            "points": [[r, c]], "depth": "snap", "floor": -20.0, "ceiling": 45.0}
    _cap(sess, "POST", "/stroke", body)
    xs = sorted(set(int(v) for v in np.argwhere(sess.seg == 1)[:, 0]))
    assert xs == [11], ("band picks the near-water duct, not muscle", xs)
    # low floor, NO ceiling -> regresses to grabbing the brighter muscle at x=5
    sess2 = ip.Session(str(d))
    body2 = dict(body); body2.pop("ceiling")
    _cap(sess2, "POST", "/stroke", body2)
    xs2 = sorted(set(int(v) for v in np.argwhere(sess2.seg == 1)[:, 0]))
    assert 5 in xs2, ("no-ceiling snap grabs the muscle (why the ceiling exists)", xs2)


def test_depth_off_is_flat_single_slice(tmp_path):
    """depth='off' paints THIS slice only -- but it is not exempt from the HU band.

    The exemption was the bug: the flat path never read floor/ceiling, so a threshold set in
    the UI restricted nothing at all and the only way to stay on the duct was to trace it
    voxel by voxel. Flat still means one slice; it no longer means unconditional."""
    sess = _make_depth_session(tmp_path)
    r, c = ip.voxel_to_disp("coronal", 6, 6, 8, sess.hu.shape)[:2]   # muscle column (-100 HU)
    base = {"view": "coronal", "idx": 6, "slab": 6, "radius_mm": 0.5, "label": 1,
            "points": [[r, c]], "depth": "off"}
    # gate off -> flat on slice x=6, whatever the HU is there
    j = json.loads(_cap(sess, "POST", "/stroke", dict(base, gate=False)).wfile.getvalue())
    xs = sorted(set(int(v) for v in np.argwhere(sess.seg == 1)[:, 0]))
    assert j["changed"] >= 1 and xs == [6]
    _cap(sess, "POST", "/undo", {})
    # gate on with the default 120 HU floor -> muscle at -100 is refused, and said so
    j = json.loads(_cap(sess, "POST", "/stroke", base).wfile.getvalue())
    assert j["ok"] and j["changed"] == 0 and not sess.seg.any()
    # ...and the same stroke on the 600 HU bar lands, still on one slice
    r2, c2 = ip.voxel_to_disp("coronal", 6, 3, 4, sess.hu.shape)[:2]
    j = json.loads(_cap(sess, "POST", "/stroke", dict(base, points=[[r2, c2]])).wfile.getvalue())
    xs = sorted(set(int(v) for v in np.argwhere(sess.seg == 1)[:, 0]))
    assert j["changed"] >= 1 and xs == [6]


def test_page_does_not_scroll_slices_on_touch_or_wheel(tmp_path):
    """v6: a resting palm/finger must NOT change the slice. The served page must have no
    one-finger 'scroll' gesture and no plain-wheel slice change; slices move only via the
    slider. It must also reject touch while a Pencil stroke is active (palm rejection)."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert 'g.mode="scroll"' not in page and 'mode==="scroll"' not in page   # no 1-finger scroll
    # Palm rejection: touchmove must bail the moment a Pencil stroke is live, and a gesture must
    # not arm while one is. Anchored to the handler + the guard's position, NOT to the exact
    # condition — the touchmove guard also filters finger count, and pinning its wording made this
    # test fail on a refactor that kept the guarantee intact.
    tmove = page.split('addEventListener("touchmove"', 1)[1].split("},{passive:", 1)[0]
    assert tmove.split("=>{", 1)[1].lstrip().startswith("if(S.drawing"), "touchmove must bail first"
    # touchmove is now the ONLY place a gesture can act, so its S.drawing guard is the whole of
    # palm rejection — there is no longer a touchstart arming step to guard separately.
    assert "g.n=0; return; }" in tmove                      # ...and it drops the gesture, not just skips
    # plain wheel must not touch S.idx (only ctrl+wheel zoom remains)
    wheel = page.split('addEventListener("wheel"', 1)[1].split("},{passive:false});", 1)[0]
    assert "S.idx" not in wheel


def test_pinch_gesture_cannot_latch_itself_off(tmp_path):
    """Regression: the pinch was DEAD on a real iPad while every synthetic test passed.

    A latched `g.mode="zoom"` was cleared by iOS Safari's mid-pinch touchcancel (Safari claiming
    the gesture) and by transient <2-touch reports — and then never re-armed, because re-arming
    needed a fresh touchstart and your fingers had never left the glass. A clean synthetic pinch
    never fires either interruption, so the unit tests all passed on a feature that did not work.

    The handler must therefore derive from the live e.touches every move and hold no gesture
    latch at all, so no event sequence can wedge it off.
    """
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert "g.mode" not in page, "a latched gesture mode is exactly what wedged the pinch off"
    tmove = page.split('addEventListener("touchmove"', 1)[1].split("},{passive:", 1)[0]
    assert "e.touches" in tmove, "the live touch list must drive the gesture, not stored state"
    # touchcancel must only reset the baseline; it must never be able to disable the gesture
    assert "touchcancel" in page and "endGesture" in page


def test_page_scrub_is_atomic_and_selects_are_dark(tmp_path):
    """v13 bug fixes: (1) scrubbing must swap CT + overlay in ONE redraw (no strobe) via a
    single coalesced refresh that decodes before painting; (2) native <select> must render
    with dark colour-scheme so the value text isn't dark-on-dark on iOS Safari."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    # (1) atomic scrub: the slice slider refreshes both layers together, not separately
    assert "refreshBoth()" in page
    assert "refreshBack(); refreshOverlay()" not in page          # the old two-step (strobe) path
    # decode-before-paint: v16 decodes via createImageBitmap (fully-decoded bitmap) with an
    # <img>+object-URL fallback, and paintCurrent() only swaps when BOTH layers are ready.
    assert "createImageBitmap" in page
    assert "drawable(bk) && drawable(ov)" in page                 # never paint a half-pair
    # idx slider drives the atomic path: oninput -> setIdx(...) -> schedulePump() -> paintCurrent()
    # which swaps CT+overlay together (verified via the setIdx body). The handler routes through setIdx.
    assert '$("idx").oninput=e=>{ setIdx(' in page
    setidx_body = page.split("function setIdx(", 1)[1].split("\n}", 1)[0]
    assert "schedulePump()" in setidx_body                        # single coalesced atomic refresh
    # (2) dark selects: color-scheme + explicit option/optgroup colours
    assert "color-scheme:dark" in page
    assert "option{ background" in page and "optgroup{ background" in page


def test_backdrop_is_deterministic_and_tracks_idx(tmp_path):
    """OBSERVED symptom (proven, not the mechanism): the user's 4 coronal screenshots at
    sliders 150/152/153/155 contained only 2 distinct images (150==153, 152==155, but
    150-vs-152 = 0.42) while the TRUE data at those slices is smooth/monotonic (all adjacent
    ~0.97-0.99) — i.e. the canvas displayed a STALE slice for some slider positions. The exact
    client-side cause (superseded/aborted fetches vs Safari connection limits) was NOT tested
    from this sandbox and remains a hypothesis; the fix is defensive regardless. This test locks
    the one thing the fix DOES rely on and that IS testable here: the server backdrop is a pure
    function of idx — same idx -> identical bytes (revisits/prefetch are cache-safe), adjacent
    idx -> different bytes (the final slice's authoritative render can't be confused with a
    neighbour). Give each coronal slice a unique bright marker and verify both halves."""
    hu = np.zeros((10, 6, 8), np.int16)
    for x in range(10):
        hu[x, 3, x % 8] = 400 + 20 * x          # a distinct bright voxel per coronal slice
    seg = np.zeros((10, 6, 8), np.uint8)
    p = tmp_path / "f0.npz"
    np.savez_compressed(p, hu=hu, seg=seg, voxel=np.array([1.562, 1.562, 0.5]))
    sess = ip.Session(str(p))

    def backdrop(idx):
        c = _cap(sess, "GET", f"/backdrop?view=coronal&idx={idx}&slab=0&win=contrast&frame=0")
        return c.wfile.getvalue()

    a1 = backdrop(4)
    a2 = backdrop(4)
    assert a1 == a2 and len(a1) > 0                       # same idx -> byte-identical (cache-safe)
    assert backdrop(3) != a1 and backdrop(5) != a1        # neighbours differ -> no stale-tile mixup


# --------------------------------------------------------------------------- #
# On-screen 90° rotation (display-only ergonomics) + depth-follow default.
# The invariant: rotation must NOT change which voxel a stroke lands on — it is
# purely a view turn, un-rotated server-side before the paint core sees it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rot", [0, 1, 2, 3])
@pytest.mark.parametrize("view", ["sagittal", "coronal", "axial"])
def test_unrotate_rc_is_exact_inverse_of_rot90(view, rot):
    """unrotate_rc(pixel-in-rotated-image) == original unrotated pixel, checked against
    np.rot90 ground truth (a one-hot marker), for every view and every quarter-turn."""
    from cc_reservoir.viz.ipad_paint import disp_shape, unrotate_rc
    H0, W0 = disp_shape(view, SHAPE)
    rng = np.random.default_rng(rot)
    for _ in range(500):
        r0 = int(rng.integers(0, H0)); c0 = int(rng.integers(0, W0))
        m = np.zeros((H0, W0), np.int32); m[r0, c0] = 1
        i, j = (int(t) for t in np.argwhere(np.rot90(m, rot) == 1)[0])   # rotated pixel of (r0,c0)
        assert unrotate_rc(i, j, rot, H0, W0) == (r0, c0)


@pytest.mark.parametrize("rot", [1, 2, 3])
def test_rotated_stroke_labels_the_same_voxels(tmp_path, rot):
    """THE invariant: painting at the rotated-display pixel of a voxel (with rot=k) labels
    exactly the same voxels as painting at its unrotated pixel (rot=0). If this fails, the
    rotation button would corrupt the segmentation (ink on the wrong voxel)."""
    view = "coronal"
    base_sess = _make_session(tmp_path)
    H0, W0 = ip.disp_shape(view, base_sess.hu.shape)
    x, y, z = 2, 3, 4
    r0, c0 = ip.voxel_to_disp(view, x, y, z, base_sess.hu.shape)
    m = np.zeros((H0, W0), np.int32); m[r0, c0] = 1
    i, j = (int(t) for t in np.argwhere(np.rot90(m, rot) == 1)[0])

    def painted(points, r):
        s = _make_session(tmp_path)                     # fresh seg each time
        body = {"view": view, "idx": x, "slab": 0, "radius_mm": 2.0, "label": 1,
                "points": points, "depth": "off", "rot": r}
        _cap(s, "POST", "/stroke", body)
        return set(map(tuple, np.argwhere(s.seg == 1)))

    base = painted([[r0, c0]], 0)
    rotd = painted([[i, j]], rot)
    assert len(base) > 0 and rotd == base, (rot, len(base), len(rotd))


def test_meta_rot_swaps_dims_and_rotates_marks(tmp_path):
    """/meta?rot=1 reports the swapped (rows,cols) + aspect and turned edge labels; rot=2
    keeps dims but flips top/bottom and left/right."""
    sess = _make_session(tmp_path)
    m0 = json.loads(_cap(sess, "GET", "/meta?view=coronal&rot=0").wfile.getvalue())
    m1 = json.loads(_cap(sess, "GET", "/meta?view=coronal&rot=1").wfile.getvalue())
    assert (m1["rows"], m1["cols"]) == (m0["cols"], m0["rows"])     # 90° swaps axes
    assert (round(m1["w_mm"], 6), round(m1["h_mm"], 6)) == (round(m0["h_mm"], 6), round(m0["w_mm"], 6))
    assert m1["marks"] != m0["marks"]
    m2 = json.loads(_cap(sess, "GET", "/meta?view=coronal&rot=2").wfile.getvalue())
    assert (m2["rows"], m2["cols"]) == (m0["rows"], m0["cols"])     # 180° keeps dims
    assert m2["marks"] == [m0["marks"][1], m0["marks"][0], m0["marks"][3], m0["marks"][2]]


def test_backdrop_and_overlay_rot_swap_dims(tmp_path):
    """A 90° turn swaps the rendered tile's height/width (backdrop JPEG + overlay PNG)."""
    from PIL import Image
    sess = _make_session(tmp_path)
    def img(path):
        return Image.open(io.BytesIO(_cap(sess, "GET", path).wfile.getvalue()))
    b0 = img("/backdrop?view=coronal&idx=4&slab=0&win=contrast&frame=0&rot=0")
    b1 = img("/backdrop?view=coronal&idx=4&slab=0&win=contrast&frame=0&rot=1")
    assert (b1.height, b1.width) == (b0.width, b0.height)
    o0 = img("/overlay?view=coronal&idx=4&frame=0&g=0&rot=0")
    o1 = img("/overlay?view=coronal&idx=4&frame=0&g=0&rot=1")
    assert (o1.height, o1.width) == (o0.width, o0.height)


def test_page_defaults_to_grow_depth_and_has_rotate(tmp_path):
    """The fix for 'ink lands on the wrong slice': depth-follow is ON (grow) by default so a
    stroke on the slab-MIP back-projects to the true duct slice, not the centre slice. Plus
    the rotate button, the stroke carrying `rot`, and the fail-loud 0-voxel toast."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert 'depth:"grow"' in page                                    # default state
    assert '<div data-d="grow" class="on">Grow</div>' in page        # chip shown active
    assert '<div data-d="off">Off</div>' in page                     # off no longer default-on
    assert 'id="rotate"' in page and "S.rot=(S.rot+1)%4" in page      # rotate button + handler
    assert "rot:S.rot" in page                                       # stroke carries rotation
    assert "res.changed===0" in page                                 # fail loud on a gated depth stroke


def test_page_opens_on_duct_slice_not_zero(tmp_path):
    """First load must jump to the server's duct-median slice (default_idx), not slice 0 (the
    anterior body surface, mostly black). A _firstView latch does it once; later view switches
    keep the current slice."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert "let _firstView=true" in page
    lv = page.split("async function loadView(", 1)[1].split("\n}", 1)[0]
    # matched as separate clauses, not as one literal: _forceDefault was later added between
    # them and froze this test red for no behavioural reason.
    assert "_firstView" in lv and "S.idx>d.nidx-1" in lv and "S.idx=d.default_idx" in lv
    assert "_firstView=false;" in lv                       # latch clears after the first load


def test_page_has_slice_stepper_and_cache_and_prefetch(tmp_path):
    """Scrub controls + the v16 cache-keyed loader that fixed the RECURRING stale-slice bug.

    History: v14/v15 tried to fix the stale slice by ABORTING superseded tile fetches, and it
    still recurred (150-156, then 130-140). ESTABLISHED here: the data + server render are clean
    in both ranges (see test_backdrop_is_deterministic_and_tracks_idx), so the fault is
    client-side. LEADING HYPOTHESIS (NOT instrumented — no browser in the dev sandbox, so this
    is a suspicion, not a verified fact): the backdrop is served Cache-Control:max-age, and on
    iOS Safari aborting an in-flight fetch of a *cacheable* URL can poison that URL's HTTP cache
    entry, so a later scrub-back returns the poisoned entry -> stale tile. v16 removes ALL
    aborting (eliminating that suspected trigger) and instead keeps a bounded decoded-tile cache
    keyed by the request URL (the full slice identity), painting a slice only when its CURRENT
    tile is ready — stale-slice-safe by construction regardless of the true cause. This locks in:
      (1) a central setIdx() clamps + syncs slider/label so shown slice == slider;
      (2) NO abort machinery survives (the suspected trigger, removed);
      (3) a URL-keyed decoded-tile cache + deduped fetches + neighbour prefetch;
      (4) -/+ step buttons and arrow keys move exactly one slice."""
    sess = _make_session(tmp_path)
    page = _cap(sess, "GET", "/").wfile.getvalue().decode("utf-8")
    assert "function setIdx(" in page
    # (2) the abort approach must be fully gone — the suspected (untested) cache-poisoning trigger
    assert "AbortController" not in page and ".abort(" not in page
    assert "abortInflight" not in page and "_inflight" not in page
    # (3) URL-keyed decoded-tile cache, dedupe, and prefetch
    assert "tileCache" in page and "function fetchInto(" in page
    assert "inflight" in page                                    # concurrent-fetch dedupe map
    assert "prefetchNeighbors(" in page                          # warm cache for neighbours
    assert "cacheClear();" in page                               # dropped on view/frame/acq switch
    # (1)/(4) stepper + authoritative release + keyboard single-step
    assert 'id="idxdn"' in page and 'id="idxup"' in page
    assert '$("idxdn").onclick=()=>setIdx(S.idx-1' in page
    assert '$("idxup").onclick=()=>setIdx(S.idx+1' in page
    assert '$("idx").onchange=e=>{ setIdx(+e.target.value, true); }' in page  # release = authoritative
    assert "ArrowUp" in page and "ArrowDown" in page             # keyboard single-step


# --------------------------------------------------------------------------- #
# View transform: pinch-zoom anchoring (Procreate's rule)
#
# The math lives in JS inside PAGE, so these run the SHIPPED block under node —
# sliced out of PAGE, never copied here, so the test cannot drift from the code.
# They assert the geometric invariant, not the wording of the source.
# --------------------------------------------------------------------------- #
import json
import shutil
import subprocess

MARK_A = "// ---- view transform (the zoom test extracts this block verbatim from PAGE) ----"
MARK_B = "// ---- end view transform ----"

# Stage 800x800; a portrait CT fits height-wise -> W=400,H=800 at ox=200,oy=0.
# So the image is NARROWER than the stage until zoom 2, and pan=0 is centred only at zoom 1.
BASE = dict(W=400, H=800, zoom=1, panX=0, panY=0, ox=200, oy=0)
STAGE_W = STAGE_H = 800
SLACK = 40

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _view_transform_js():
    from cc_reservoir.viz import ipad_paint as ip
    assert MARK_A in ip.PAGE and MARK_B in ip.PAGE, "view-transform markers moved — fix this slice"
    return ip.PAGE.split(MARK_A, 1)[1].split(MARK_B, 1)[0]


def _pinch(state, z, frm, to=None):
    """Apply zoomPan to `state` and report where the anchor under `frm` ended up."""
    to = to if to is not None else frm
    js = f"""
{_view_transform_js()}
const S = Object.assign({json.dumps(BASE)}, {json.dumps(state)});
const stage = {{clientWidth:{STAGE_W}, clientHeight:{STAGE_H}}};
function applyTransform(){{}}                       // DOM shell, stubbed: this test is the pure math
const $ = () => ({{}});
const toContent = (x,y) => [(x-S.ox-S.panX)/S.zoom, (y-S.oy-S.panY)/S.zoom];
const toStage   = (u,v) => [S.ox+S.panX+u*S.zoom, S.oy+S.panY+v*S.zoom];
const anchor = toContent({frm[0]}, {frm[1]});       // content point under the fingers BEFORE
zoomPan({z}, {frm[0]}, {frm[1]}, {to[0]}, {to[1]});
const landed = toStage(anchor[0], anchor[1]);       // ...and where it sits AFTER
console.log(JSON.stringify({{
  zoom:S.zoom, panX:S.panX, panY:S.panY, landedX:landed[0], landedY:landed[1],
  left:S.ox+S.panX, right:S.ox+S.panX+S.W*S.zoom,
  top:S.oy+S.panY, bottom:S.oy+S.panY+S.H*S.zoom }}));
"""
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# zoom 3 is centred at panX=-400,panY=-800 and is bigger than the stage on BOTH axes,
# so the pan clamp cannot legitimately fight the anchor.
CENTRED_Z3 = dict(zoom=3, panX=-400, panY=-800)


@needs_node
@pytest.mark.parametrize("focal", [(400, 400), (300, 250), (520, 610)])
@pytest.mark.parametrize("k", [1.2, 0.85, 1.9])
def test_pinch_keeps_the_point_under_your_fingers(focal, k):
    """THE invariant. Zooming used to blow the image out from the canvas top-left, so the duct
    you were pinching slid away and you had to chase it with a pan. It must stay put."""
    r = _pinch(CENTRED_Z3, 3 * k, focal)
    moved = np.hypot(r["landedX"] - focal[0], r["landedY"] - focal[1])
    assert moved < 1e-9, f"anchor drifted {moved:.3f}px at k={k}"


@needs_node
def test_pinch_and_drag_are_one_gesture():
    """Spread AND move in the same gesture: the anchor tracks to the new focal point."""
    r = _pinch(CENTRED_Z3, 3 * 1.3, (400, 400), (460, 330))
    assert np.hypot(r["landedX"] - 460, r["landedY"] - 330) < 1e-9


@needs_node
def test_fully_zoomed_out_is_exactly_centred():
    """The old code special-cased `if(z===1){panX=0;panY=0}`. That is now derived from the
    clamp — one source of truth for 'where does the image sit', not two."""
    r = _pinch(dict(zoom=4, panX=-500, panY=-900), 1, (400, 400))
    assert (r["panX"], r["panY"]) == (0, 0)


@needs_node
@pytest.mark.parametrize("z", [1.2, 2.5, 4, 8])
def test_zoomed_in_you_can_slide_the_image_anywhere(z):
    """The real workflow: spread to zoom, then slide two fingers left to bring the structure on
    the RIGHT into view.

    This replaces a 'the image must cover the stage' rule that was actively wrong here. BASE is a
    portrait image 400 wide in an 800 stage — as a coronal is against a landscape iPad — so it is
    narrower than the stage until zoom 2, and 'must cover' pinned X to the centre for that whole
    range. That silently overrode the anchor AND made the sideways drag a no-op, which is exactly
    'zooms from the corner, and I can't move it'. A drag must actually drag.
    """
    # constant zoom, focal dragged 400px left => a pure two-finger pan
    r = _pinch(dict(zoom=z, panX=0, panY=0), z, (600, 400), (200, 400))
    assert r["panX"] < -100, f"a 400px leftward drag only moved pan to {r['panX']} (clamp ate it)"
    # ...but the image can never be lost off-screen entirely
    assert r["right"] > 0 and r["left"] < STAGE_W, f"image lost: [{r['left']},{r['right']}]"


@needs_node
def test_past_the_zoom_ceiling_scale_stops_but_pan_still_follows():
    """Keep pinching past 8x and the image must not drift or freeze — scale pins, pan tracks."""
    start = dict(zoom=8, panX=-1000, panY=-2000)
    r = _pinch(start, 16, (400, 400), (450, 400))
    assert r["zoom"] == 8
    assert abs(r["panX"] - (start["panX"] + 50)) < 1e-9, "pan stopped following the fingers"


@needs_node
def test_a_long_pinch_with_drag_never_drifts():
    """A real pinch arrives as ~60 small touchmoves with the midpoint moving the whole time.

    This is the failure mode of incremental focal-zoom: Konva's widely-copied form derives its
    anchor from the NEW midpoint but then ALSO adds the midpoint's motion as a separate pan term,
    double-counting it. That leaves a residual of d*(1-z1/z0) per frame — invisible in one frame,
    but it integrates into visible slip over a gesture. We anchor on where the fingers WERE and
    land on where they ARE, with no additive pan term, so the invariant must hold on EVERY frame.
    """
    js = f"""
{_view_transform_js()}
const S = Object.assign({json.dumps(BASE)}, {json.dumps(dict(zoom=2, panX=-200, panY=-400))});
const stage = {{clientWidth:{STAGE_W}, clientHeight:{STAGE_H}}};
function applyTransform(){{}}
const $ = () => ({{}});
const toContent = (x,y) => [(x-S.ox-S.panX)/S.zoom, (y-S.oy-S.panY)/S.zoom];
const toStage   = (u,v) => [S.ox+S.panX+u*S.zoom, S.oy+S.panY+v*S.zoom];

let fx=380, fy=300, dist=120, worst=0;
for(let i=0;i<60;i++){{
  const nfx=fx+2.5, nfy=fy+1.7, ndist=dist*1.02;      // fingers spread AND drift together
  const a = toContent(fx,fy);                          // content under the fingers now
  zoomPan(S.zoom*ndist/dist, fx,fy, nfx,nfy);
  const l = toStage(a[0],a[1]);                        // ...must be under them after
  // only meaningful while the clamp is not legitimately overriding the anchor
  const free = (S.W*S.zoom > stage.clientWidth+2*40) && (S.H*S.zoom > stage.clientHeight+2*40);
  if(free) worst = Math.max(worst, Math.hypot(l[0]-nfx, l[1]-nfy));
  fx=nfx; fy=nfy; dist=ndist;
}}
console.log(JSON.stringify({{worst, zoom:S.zoom}}));
"""
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["zoom"] > 3, "the simulated pinch should actually have zoomed"
    assert out["worst"] < 1e-9, f"anchor slipped {out['worst']:.6f}px over a 60-frame pinch"


# --------------------------------------------------------------------------- #
# The HU gate, the lasso and the wand.
#
# Regression cover for the bug that [floor, ceiling] restricted nothing: it was only ever read
# when choosing WHICH slice in the slab to snap to, so flat painting ignored it outright and
# depth painting let a stroke through wherever the +/-slab column happened to contain one
# in-band voxel -- which, on a thorax, is nearly everywhere.
#
# Geometry of the phantom below: coronal cuts x, display row = nz-1-z, display col = y.
# --------------------------------------------------------------------------- #
GSH = (40, 40, 60)
GVOX = (1.0, 1.0, 1.0)
LINE = [(29, 10), (29, 35)]       # one horizontal stroke at z=30, across duct and rib


@pytest.fixture
def phantom():
    hu = np.full(GSH, 50, np.int16)           # muscle bed
    hu[:, 18:23, 28:33] = 400                 # duct : y 18-22, z 28-32
    hu[:, 26:29, 28:33] = 900                 # rib  : y 26-28, z 28-32
    return hu


@pytest.mark.parametrize("view", ["sagittal", "coronal", "axial"])
def test_disp_to_voxel_arr_matches_scalar(view):
    H, W = disp_shape(view, GSH)
    rows = np.array([0, 1, H // 3, H - 1])
    cols = np.array([W - 1, 0, W // 2, 3])
    xs, ys, zs = disp_to_voxel_arr(view, rows, cols, 7, GSH)
    for k in range(rows.size):
        assert (int(xs[k]), int(ys[k]), int(zs[k])) == \
            disp_to_voxel(view, int(rows[k]), int(cols[k]), 7, GSH)


def test_flat_brush_without_the_volume_paints_everything(phantom):
    """The old behaviour, kept honest: no volume -> no gate -> muscle gets painted."""
    seg = np.zeros(GSH, np.uint8)
    n = paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 1)
    assert n == 80
    assert phantom[np.nonzero(seg == 1)].min() == 50


def test_flat_brush_honours_the_band(phantom):
    """THE FIX: a flat stroke writes only voxels inside [floor, ceiling]."""
    seg = np.zeros(GSH, np.uint8)
    n = paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 1, hu=phantom, floor=120.0)
    assert n == 24 < 80
    assert set(np.unique(phantom[np.nonzero(seg == 1)])) == {400, 900}


def test_ceiling_separates_duct_from_cortex(phantom):
    seg = np.zeros(GSH, np.uint8)
    n = paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 1,
                     hu=phantom, floor=120.0, ceiling=700.0)
    assert n == 15
    assert set(np.unique(phantom[np.nonzero(seg == 1)])) == {400}


def test_gate_off_is_the_old_free_drawing(phantom):
    seg = np.zeros(GSH, np.uint8)
    assert paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 1,
                        hu=phantom, floor=120.0, gate=False) == 80


def test_erase_is_never_gated(phantom):
    """You must always be able to remove a label you can see, whatever the band is set to."""
    seg = np.zeros(GSH, np.uint8)
    n = paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 1)
    assert paint_stroke(seg, GSH, "coronal", 20, LINE, 1.0, GVOX, 0,
                        hu=phantom, floor=120.0, ceiling=125.0) == n
    assert not seg.any()


def test_depth_follow_writes_only_in_band_voxels(phantom):
    seg = np.zeros(GSH, np.uint8)
    paint_stroke_depth(phantom, seg, "coronal", 20, LINE, 1.0, GVOX, 8, 1,
                       floor=120.0, ceiling=700.0, grow=True)
    assert set(np.unique(phantom[np.nonzero(seg == 1)])) == {400}


def test_polygon_fill_square_and_clipping():
    rows, cols = polygon_pixels([(10, 10), (10, 20), (20, 20), (20, 10)], 40, 40)
    assert rows.size == 121                                  # 11 x 11, outline included
    assert rows.min() == 10 and rows.max() == 20 and cols.min() == 10 and cols.max() == 20
    rows, cols = polygon_pixels([(-50, -50), (-50, 5), (5, 5), (5, -50)], 40, 40)
    assert rows.size and rows.min() >= 0 and cols.min() >= 0 and rows.max() <= 5
    assert polygon_pixels([(3, 4)], 40, 40)[0].tolist() == [3]   # a lone tap still marks a pixel
    assert polygon_pixels([], 40, 40)[0].size == 0


def test_polygon_fill_is_concave_aware():
    """A C-shaped lasso must not fill its own notch (even-odd, not bounding box)."""
    poly = [(0, 0), (0, 20), (20, 20), (20, 12), (6, 12), (6, 8), (20, 8), (20, 0)]
    rows, cols = polygon_pixels(poly, 40, 40)
    inside = set(zip(rows.tolist(), cols.tolist()))
    assert (10, 4) in inside and (10, 15) in inside
    assert (10, 10) not in inside


def test_lasso_fills_only_the_in_band_voxels_it_encloses(phantom):
    box = [(25, 14), (25, 27), (34, 27), (34, 14)]           # 10 rows x 14 cols enclosed
    seg = np.zeros(GSH, np.uint8)
    n = lasso_fill(phantom, seg, "coronal", 20, box, GVOX, 0, 1,
                   depth="off", floor=120.0, ceiling=700.0)
    assert n == 25                                            # the duct's share of that rectangle
    assert set(np.unique(phantom[np.nonzero(seg == 1)])) == {400}
    free = np.zeros(GSH, np.uint8)
    assert lasso_fill(phantom, free, "coronal", 20, box, GVOX, 0, 1,
                      depth="off", gate=False) == 140


def test_lasso_erase_clears_everything_enclosed(phantom):
    box = [(25, 14), (25, 27), (34, 27), (34, 14)]
    seg = np.zeros(GSH, np.uint8)
    lasso_fill(phantom, seg, "coronal", 20, box, GVOX, 0, 1, depth="off", gate=False)
    assert lasso_fill(phantom, seg, "coronal", 20, box, GVOX, 0, 0, depth="off") == 140
    assert not seg.any()


def test_lasso_with_depth_follow_stays_in_the_band(phantom):
    box = [(25, 14), (25, 27), (34, 27), (34, 14)]
    seg = np.zeros(GSH, np.uint8)
    lasso_fill(phantom, seg, "coronal", 12, box, GVOX, 10, 1,
               depth="grow", floor=120.0, ceiling=700.0)
    assert set(np.unique(phantom[np.nonzero(seg == 1)])) == {400}


@pytest.fixture
def tube():
    hu = np.full(GSH, -100, np.int16)
    hu[20, 20, 10:40] = 300                                   # a 30-voxel duct along z
    hu[30, 30, 20] = 300                                      # disconnected, same HU
    return hu


def test_wand_fills_the_connected_component_only(tube):
    seg = np.zeros(GSH, np.uint8)
    n, note = wand_fill(tube, seg, "coronal", 20, 60 - 1 - 25, 20, GVOX, 1,
                        floor=120.0, radius_mm=100.0)
    assert note == "" and n == 30
    assert seg[30, 30, 20] == 0


def test_wand_reach_is_a_hard_physical_bound(tube):
    seg = np.zeros(GSH, np.uint8)
    assert wand_fill(tube, seg, "coronal", 20, 60 - 1 - 25, 20, GVOX, 1,
                     floor=120.0, radius_mm=5.0)[0] == 11     # +/-5 mm of a 1 mm/voxel tube


def test_wand_on_background_reports_instead_of_painting(tube):
    seg = np.zeros(GSH, np.uint8)
    n, note = wand_fill(tube, seg, "coronal", 20, 5, 5, GVOX, 1, floor=120.0, radius_mm=10.0)
    assert n == 0 and note and not seg.any()


def test_wand_erase_removes_the_connected_label_blob(tube):
    seg = np.zeros(GSH, np.uint8)
    wand_fill(tube, seg, "coronal", 20, 60 - 1 - 25, 20, GVOX, 1, floor=120.0, radius_mm=100.0)
    n, _ = wand_fill(tube, seg, "coronal", 20, 60 - 1 - 25, 20, GVOX, 0,
                     floor=120.0, radius_mm=100.0)
    assert n == 30 and not seg.any()


def test_numpy_flood_matches_scipy_labelling(tube):
    """The numpy fallback is what runs on a bare numpy+Pillow install, so it is the reference."""
    band = tube >= 120
    assert np.array_equal(_flood_numpy(band, (20, 20, 25)), _connected_from(band, (20, 20, 25)))


def test_gate_plane_is_in_display_orientation(phantom):
    """The wash has to explain the pixels it covers, so it must use to_display's transform."""
    gp = gate_plane(phantom, "coronal", 20, 0, 120.0, 700.0)
    assert gp.shape == disp_shape("coronal", GSH)
    assert gp[60 - 1 - 30, 20]                                # duct
    assert not gp[60 - 1 - 30, 27]                            # rib, above the ceiling
    assert not gp[60 - 1 - 10, 20]                            # muscle, below the floor
    assert gate_plane(phantom, "coronal", 20, 0, -np.inf, np.inf).all()


def test_gate_plane_slab_is_any_not_max(phantom):
    """A column whose BRIGHTEST voxel is above the ceiling can still hold an in-band one, and
    that column is paintable -- a max-based preview would tell you the opposite."""
    hu = np.full(GSH, -100, np.int16)
    hu[20, 20, 30] = 400                                      # in band
    hu[24, 20, 30] = 2000                                     # out of band, brighter
    plane = gate_plane(hu, "coronal", 22, 4, 120.0, 700.0)
    assert plane[60 - 1 - 30, 20]


def test_gate_wash_turns_with_the_display(tmp_path):
    """The wash has to rotate with the tile it explains. A wash that stays put while the CT
    turns points at the wrong voxels, which is worse than no preview at all."""
    from PIL import Image
    sess = _make_session(tmp_path)                       # hu[2,3,4] = 1000, everything else 0
    gate = (500.0, None)                                 # only that one voxel is in band

    def blue(rot):
        png = render_overlay_png(sess, "coronal", 2, visible=[], rot=rot, slab=0, gate=gate)
        a = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
        m = (a[:, :, 3] > 0)
        assert tuple(a[m][0][:3]) == GATE_RGB
        return a[:, :, 3] > 0

    m0 = blue(0)
    assert m0.sum() == 1                                 # exactly the one in-band voxel
    for rot in (1, 2, 3):
        assert np.array_equal(blue(rot), np.rot90(m0, rot)), f"wash out of step at rot={rot}"


def test_gate_wash_is_off_by_default_and_for_an_open_band(tmp_path):
    """No `gate` argument, or a band with no limits, must add nothing to the overlay."""
    from PIL import Image
    sess = _make_session(tmp_path)

    def opaque(**kw):
        png = render_overlay_png(sess, "coronal", 2, visible=[], slab=0, **kw)
        return int((np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3] > 0).sum())

    assert opaque() == 0
    assert opaque(gate=(None, None)) == 0
    assert opaque(gate=(500.0, None)) == 1
