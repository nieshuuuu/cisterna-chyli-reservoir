"""Interactive CC/TD context viewer: 3 scrollable MPR planes with the iodinated-lymph overlay —
CC (green) + TD (teal) — plus bone (gray) and kidney (gold) HU landmarks, and an embedded live 3D
model of the duct.

Two inputs:
  * a single context_<tag>.npz  (peak frame only)
  * a context4d_<tag>/ directory (meta.npz + f0.npz..fN.npz, one full HU + duct seg per frame);
    a timepoint slider / ',' '.' keys scrub the duct deforming with respiration.

  python -m cc_reservoir.viz.viewer /path/context_<tag>.npz
  python -m cc_reservoir.viz.viewer /path/context4d_<tag>/
  python -m cc_reservoir.viz.viewer <input> --shot out.png      # headless verification grab
"""
import os
import sys

import PySide6
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                      os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins", "platforms"))
import numpy as np

from cc_reservoir.io.context import load_seg
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from cc_reservoir.viz.slice_canvas import SliceCanvas, _AXIS
from cc_reservoir.viz.labels import LABEL_RGBA

SLAB_CYCLE = [0, 8, 20]                                # 'm' cycles MIP slab thickness (±slices)
WL_PRESETS = {"v": (400.0, 500.0), "a": (40.0, 400.0), "b": (300.0, 1500.0), "r": (80.0, 900.0)}


def _poly(v, f):
    return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3, int), f]).ravel())


class TriPlanarWindow(QtWidgets.QMainWindow):
    def __init__(self, path):
        super().__init__()
        if os.path.isdir(path):                       # 4D: meta.npz + per-frame f{i}.npz
            self.tp = True; self.tpdir = path; self._fr = {}
            self.meta = np.load(os.path.join(path, "meta.npz"))
            self.n_frames = int(self.meta["n_frames"]); self.frame = int(self.meta["peak_i"])
        else:
            self.tp = False; self.meta = np.load(path); self.n_frames = 1
            self.frame = int(self.meta["peak_i"]) if "peak_i" in self.meta.files else 0
        self.active = "new"                           # 'o' toggles new(integrated-HU) vs lab hand-mask
        self.show_bone = True                         # 'k' hides bone when it distracts
        self.vox = tuple(float(v) for v in self.meta["voxel"])
        self.shape = self._F("hu").shape
        self.setWindowTitle(f"CC/TD context viewer — {os.path.basename(path.rstrip('/'))}")

        self.canvases = {v: SliceCanvas(v, self.vox) for v in ("axial", "coronal", "sagittal")}
        self.plotter = QtInteractor(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.canvases["axial"], 0, 0)
        grid.addWidget(self.canvases["coronal"], 0, 1)
        grid.addWidget(self.canvases["sagittal"], 1, 0)
        grid.addWidget(self.plotter.interactor, 1, 1)
        gridw = QtWidgets.QWidget(); gridw.setLayout(grid)
        central = QtWidgets.QWidget(); vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.addWidget(gridw, 1)
        if self.tp:                                   # timepoint scrubber
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setRange(0, self.n_frames - 1); self.slider.setValue(self.frame)
            self.slider.valueChanged.connect(self._set_frame)
            vbox.addWidget(self.slider)
        self.setCentralWidget(central)
        QtWidgets.QApplication.instance().installEventFilter(self)   # keys app-wide (VTK eats them)
        self._slab_i = 0                              # open at single slice (true CT plane); 'm' to MIP

        hu0 = self._F("hu")
        for cv in self.canvases.values():
            cv.set_volumes(hu0, self._disp_seg(), level=80.0, window=900.0)
            cv.set_slab(SLAB_CYCLE[self._slab_i])
            cv.voxelClicked.connect(self._on_click)
            cv.wlChanged.connect(self._on_wl)

        self.plotter.set_background("white")
        for fn in (lambda: self.plotter.enable_anti_aliasing("ssaa"), self.plotter.enable_ssao):
            try:                                      # silky/shiny: SSAA edges + SSAO depth
                fn()
            except Exception:
                pass
        bv, bf = self._S("bone_v"), self._S("bone_f")
        if bv is not None and len(bv):                # bone is static -> add once
            self.plotter.add_mesh(_poly(bv.astype(float), bf), color=tuple(LABEL_RGBA[4, :3]),
                                  opacity=0.13, name="bone", smooth_shading=True, specular=0.3)
        self._rebuild_duct()
        try:                                          # click a point on the 3D duct -> jump the slices
            self.plotter.enable_point_picking(callback=self._pick3d, show_message=False,
                                              left_clicking=True, show_point=True, point_size=14,
                                              color="red", use_picker="cell")
        except Exception:
            pass
        self._status()
        self._frame_duct()
        c = (np.asarray(self.meta["duct_centroid"]) / np.array(self.vox)).astype(int)   # open at the duct
        self._on_click((int(c[0]), int(c[1]), int(c[2])))

    # -- data access (static = meta; per-frame = f{i}.npz, or meta for the peak npz) --
    def _S(self, key):
        return self.meta[key] if key in self.meta.files else None

    def _frame_npz(self):
        if self.frame not in self._fr:
            self._fr[self.frame] = np.load(os.path.join(self.tpdir, f"f{self.frame}.npz"))
        return self._fr[self.frame]

    def _F(self, key):
        src = self._frame_npz() if self.tp else self.meta
        return src[key] if key in src.files else None

    def _has_lab(self):
        return "seg_lab" in self.meta.files

    def _seg(self):
        """Current-frame seg, preferring the _edit.npz hand-edit sidecar (io.context.load_seg)."""
        if not self.tp:
            return self._F("seg")                     # single-frame context: sidecar not wired here
        if not hasattr(self, "_seg_ovr"):
            self._seg_ovr = {}
        if self.frame not in self._seg_ovr:
            self._seg_ovr[self.frame] = load_seg(os.path.join(self.tpdir, f"f{self.frame}.npz"))
        return self._seg_ovr[self.frame]

    def _cur_seg(self):
        if self.active == "off":                          # CT only — no overlay (inspect raw HU)
            return np.zeros_like(self._F("seg"))
        if self.active == "lab" and self._has_lab():
            return self.meta["seg_lab"]
        return self._seg()

    def _disp_seg(self):
        """Active seg for this frame, bone dropped when hidden so it can't clutter the duct."""
        s = self._cur_seg()
        if self.show_bone:
            return s
        s2 = s.copy(); s2[s2 == 4] = 0; return s2

    def _duct_meshes(self):
        """(tag, label-id, verts, faces) for CC/TD of the active seg at the current frame."""
        if self.active == "lab":
            return [("cc", 1, self._S("cc_lab_v"), self._S("cc_lab_f")),
                    ("td", 2, self._S("td_lab_v"), self._S("td_lab_f"))]
        return [("cc", 1, self._F("cc_v"), self._F("cc_f")),
                ("td", 2, self._F("td_v"), self._F("td_f"))]

    # -- updates --
    def _rebuild_duct(self):
        for tag, lid, v, f in self._duct_meshes():
            self.plotter.remove_actor(tag, render=False)
            if v is not None and len(v):
                self.plotter.add_mesh(_poly(v.astype(float), f), color=tuple(LABEL_RGBA[lid, :3]),
                                      opacity=1.0, name=tag, smooth_shading=True, specular=0.6,
                                      specular_power=18, ambient=0.25, diffuse=0.8)
        self.plotter.render()

    def _set_seg(self, which):
        """Switch CC/TD between 'new' (integrated-HU) and 'lab' (hand-mask); camera untouched."""
        if which == "lab" and not self._has_lab():
            return
        self.active = which
        for cv in self.canvases.values():
            cv.set_label(self._disp_seg())
        self._rebuild_duct(); self._status()

    def _set_frame(self, i):
        """Scrub to timepoint i: reload that frame's HU + duct seg + 3D mesh (slices keep position)."""
        self.frame = int(i)
        hu = self._F("hu")
        for cv in self.canvases.values():
            cv.set_volumes(hu, self._disp_seg())      # keeps idx / level / window
        self._rebuild_duct(); self._status()

    def _toggle_bone(self):
        self.show_bone = not self.show_bone
        for cv in self.canvases.values():
            cv.set_label(self._disp_seg())
        a = self.plotter.actors.get("bone")
        if a is not None:
            a.SetVisibility(self.show_bone); self.plotter.render()

    def _status(self):
        name = {"new": "integrated-HU (new)", "lab": "lab hand-mask (original)",
                "off": "OFF — CT only"}[self.active]
        fr = ""
        if self.tp:
            em = self._S("enh_means")
            e = f"  enh {em[self.frame]:.0f}HU" if em is not None else ""
            fr = f"   frame {self.frame}/{self.n_frames - 1}{e}"
        self.statusBar().showMessage(f"seg: {name}{fr}    o=new/lab/off  k=bone  m=MIP  v/a/b=window"
                                     + ("  ,/.=timepoint" if self.tp else ""))

    def _frame_duct(self, margin=15.0):
        """Frame the 3D on the CC/TD duct, side-on with cranial up."""
        vs = [v for _, _, v, _ in self._duct_meshes() if v is not None and len(v)]
        self.plotter.camera_position = "yz"
        if vs:
            a = np.vstack(vs); lo, hi = a.min(0) - margin, a.max(0) + margin
            self.plotter.reset_camera(bounds=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
        else:
            self.plotter.reset_camera()
        self.plotter.camera.azimuth = 20; self.plotter.camera.elevation = 8
        self.plotter.render()

    def _on_click(self, voxel):
        for cv in self.canvases.values():
            cv.set_idx(voxel[_AXIS[cv.view]])         # each plane scrolls to its own slice axis
            cv.set_crosshair(voxel)

    def _pick3d(self, point, *_):
        """A click on a 3D mesh (verts in mm) -> jump the MPR slices to that voxel."""
        if point is None:
            return
        p = np.asarray(point, float)
        if p.size < 3:
            return
        v = p[:3] / np.array(self.vox)
        if (v < 0).any() or (v >= np.array(self.shape)).any():
            return
        self._on_click(tuple(int(round(x)) for x in v))

    def _on_wl(self, level, window):
        for cv in self.canvases.values():
            cv.set_window_level(level, window)

    def _step_frame(self, d):
        if self.tp:
            self.slider.setValue(int(np.clip(self.frame + d, 0, self.n_frames - 1)))

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.KeyPress:
            k = ev.text().lower()
            if k in ("o", "m", "k", ",", ".") or k in WL_PRESETS:
                self._key(k); return True
        return super().eventFilter(obj, ev)

    def _key(self, k):
        if k == "o":                                      # cycle new -> lab -> off (CT only) -> new
            order = ["new", "lab", "off"] if self._has_lab() else ["new", "off"]
            i = order.index(self.active) if self.active in order else 0
            self._set_seg(order[(i + 1) % len(order)])
        elif k == "k":
            self._toggle_bone()
        elif k == ",":
            self._step_frame(-1)
        elif k == ".":
            self._step_frame(+1)
        elif k == "m":
            self._slab_i = (self._slab_i + 1) % len(SLAB_CYCLE)
            for cv in self.canvases.values():
                cv.set_slab(SLAB_CYCLE[self._slab_i])
        elif k in WL_PRESETS:
            self._on_wl(*WL_PRESETS[k])

    def shot(self, path):
        from PIL import Image
        QtWidgets.QApplication.processEvents()
        self.resize(1100, 1100); self.show(); QtWidgets.QApplication.processEvents()
        self._frame_duct(); QtWidgets.QApplication.processEvents()
        tiles = []
        for v in ("axial", "coronal", "sagittal"):
            self.canvases[v].grab().save(f"/tmp/_mpr_{v}.png")
            tiles.append(Image.open(f"/tmp/_mpr_{v}.png").convert("RGB"))
        tiles.append(Image.fromarray(self.plotter.screenshot(return_img=True)).convert("RGB"))
        cell = (480, 480); tiles = [t.resize(cell) for t in tiles]
        c = Image.new("RGB", (cell[0] * 2, cell[1] * 2), "white")
        for k, t in enumerate(tiles):
            c.paste(t, ((k % 2) * cell[0], (k // 2) * cell[1]))
        c.save(path)


def main():
    path = sys.argv[1]
    shot = sys.argv[sys.argv.index("--shot") + 1] if "--shot" in sys.argv else None
    app = QtWidgets.QApplication(sys.argv)
    win = TriPlanarWindow(path)
    if shot:
        win.shot(shot); print(f"saved {shot}"); return
    win.resize(1300, 1240); win.show()
    QtCore.QTimer.singleShot(80, win._frame_duct)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
