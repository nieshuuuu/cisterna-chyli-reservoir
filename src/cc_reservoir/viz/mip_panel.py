"""Whole-body MIP panel — the coronal max-intensity-projection view (spine + ribs + the whole duct in
one image) that reads CC/TD far more clearly than a cropped single-slice MPR. A thick slab is the
point: it projects the entire winding duct into one plane instead of catching it in fragments.

Controls: an acq dropdown (switch animal/condition), a slab-DEPTH slider (slide the projection window
anterior↔posterior), a slab-THICKNESS slider, a TIMEPOINT slider, and a CC/TD overlay toggle ('s' or
the checkbox). 'c'/'g' switch coronal/sagittal; ',' '.' step the timepoint; 'a'/'b'/'w' window presets.

  python -m cc_reservoir.viz.mip_panel /path/context4d_<tag>/        # one acq
  python -m cc_reservoir.viz.mip_panel /path/cc_out/                 # a folder of context4d_* -> dropdown
  python -m cc_reservoir.viz.mip_panel <input> --shot out.png        # headless verification grab
"""
import glob
import os
import sys

import PySide6
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                      os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins", "platforms"))
import numpy as np

from cc_reservoir.io.context import load_seg
from cc_reservoir.viz.labels import LAB_CC_RGB, LAB_TD_RGB
from PySide6 import QtCore, QtGui, QtWidgets

CC_RGB = np.array([57, 211, 83], np.float32)          # CC green (locked: CC is always green)
TD_RGB = np.array([22, 179, 179], np.float32)         # TD teal
NODE_RGB = np.array([255, 140, 0], np.float32)        # para-aortic/renal lymph nodes (label 3) orange
# The lab's REAL clinical seg is the Vitrea screenshots (Data Analysis/Vitrea Images/) — no volumetric
# export exists, so we can't overlay it as a mask; show its VOLUMES as the reference target instead.
VITREA_REF = "Vitrea clinical ref (exemplar): TD 2.92 ml (HU 379) · CC 0.157 ml / 157 µL (HU 611)"
WL_PRESETS = {"a": (40.0, 400.0), "b": (300.0, 1500.0), "w": (275.0, 850.0)}   # soft / bone+contrast / reach.png
_PROJ = {"coronal": 0, "sagittal": 1}                 # MIP axis: coronal projects x(AP), sagittal projects y(LR)


def _wl(arr, level, window):
    lo = level - window / 2.0
    return (np.clip((arr - lo) / max(window, 1e-6), 0.0, 1.0) * 255).astype(np.uint8)


def _ctx_tag(c):
    return os.path.basename(c.rstrip("/")).replace("context4d_", "").replace("context_", "").replace(".npz", "")


def _find_contexts(path):
    """A single context (a context4d dir with meta.npz, or a context_*.npz file), or a PARENT folder
    holding several context4d_* dirs -> the acq dropdown lists them."""
    if os.path.isfile(path):
        return [path]
    if os.path.exists(os.path.join(path, "meta.npz")):
        return [path]
    subs = sorted(glob.glob(os.path.join(path, "context4d_*")))
    return subs if subs else [path]


class MipPanel(QtWidgets.QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.contexts = _find_contexts(path)
        # view state that PERSISTS across acq switches
        self.proj = "coronal"; self.show_seg = True; self.show_lab = False; self.level, self.window = 275.0, 850.0
        self.slab = 45; self.ds = 1                    # ds>1 downsamples the MIP (full-res is ~21 ms, so 1)
        self._fr = {}                                  # decompressed-frame cache for the CURRENT context

        self.img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img.setMinimumSize(400, 600); self.img.setStyleSheet("background:black;")
        self.img.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.acq_combo = QtWidgets.QComboBox(); self.acq_combo.addItems([_ctx_tag(c) for c in self.contexts])
        self.s_depth = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_slab = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.s_slab.setRange(1, 160); self.s_slab.setValue(self.slab)
        self.s_tp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.cb_seg = QtWidgets.QCheckBox("our seg (s)"); self.cb_seg.setChecked(True)
        self.cb_lab = QtWidgets.QCheckBox("rough .mat seed (l)"); self.cb_lab.setChecked(False)
        self.lbl = QtWidgets.QLabel()
        self.ref = QtWidgets.QLabel(VITREA_REF); self.ref.setStyleSheet("color:#888;")   # Vitrea target caption
        self.acq_combo.currentIndexChanged.connect(self._on_acq)
        self.s_depth.valueChanged.connect(self._on_depth)
        self.s_slab.valueChanged.connect(self._on_slab)
        self.s_tp.valueChanged.connect(self._on_tp)
        self.cb_seg.toggled.connect(self._on_seg)
        self.cb_lab.toggled.connect(self._on_lab)

        for _s in (self.s_depth, self.s_slab, self.s_tp):       # let sliders fill the panel width
            _s.setMinimumWidth(420)
            _s.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        if len(self.contexts) > 1:
            form.addRow("acq", self.acq_combo)
        form.addRow("depth (AP/LR)", self.s_depth)
        form.addRow("slab ±", self.s_slab)
        form.addRow("timepoint", self.s_tp)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.cb_seg); row.addWidget(self.cb_lab)
        row.addStretch(1); row.addWidget(self.lbl)
        form.addRow(row)
        form.addRow(self.ref)
        ctl = QtWidgets.QWidget(); ctl.setLayout(form)
        central = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(central)
        v.addWidget(self.img, 1); v.addWidget(ctl)
        self.setCentralWidget(central)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._rtimer = QtCore.QTimer(self); self._rtimer.setSingleShot(True); self._rtimer.timeout.connect(self._update)
        self._load_context(self.contexts[0])

    # -- context (re)load: everything that changes when the acq changes --
    def _load_context(self, ctxdir):
        self._fr = {}
        if os.path.isdir(ctxdir):                      # 4D: meta.npz + per-frame f{i}.npz
            self.tp = True; self.tpdir = ctxdir
            self.meta = np.load(os.path.join(ctxdir, "meta.npz"))
            self.n_frames = int(self.meta["n_frames"]); self.frame = int(self.meta["peak_i"])
        else:
            self.tp = False; self.tpdir = None; self.meta = np.load(ctxdir); self.n_frames = 1
            self.frame = int(self.meta["peak_i"]) if "peak_i" in self.meta.files else 0
        self.vox = tuple(float(v) for v in self.meta["voxel"])
        self.shape = self._F("hu").shape
        seg0 = self._F("seg"); ax = _PROJ[self.proj]
        duct = np.argwhere((seg0 >= 1) & (seg0 <= 2)) if seg0 is not None else np.zeros((0, 3))
        self.center = int(duct[:, ax].mean()) if len(duct) else self.shape[ax] // 2
        self.tag = _ctx_tag(ctxdir)
        self.setWindowTitle(f"whole-body MIP — {self.tag}")
        self.s_tp.blockSignals(True); self.s_tp.setRange(0, max(self.n_frames - 1, 0))
        self.s_tp.setValue(self.frame); self.s_tp.blockSignals(False); self.s_tp.setEnabled(self.tp)
        self._reset_depth_range()
        self._update()
        if self.tp:
            QtCore.QTimer.singleShot(250, self._preload)

    # -- data --
    def _frame_npz(self):
        if self.frame not in self._fr:
            self._fr[self.frame] = self._load_frame(self.frame)
        return self._fr[self.frame]

    @staticmethod
    def _decompress(path):
        with np.load(path) as d:                          # NpzFile re-decompresses on EVERY key access, so
            return {k: d[k] for k in d.files}             # pull the arrays out ONCE into a plain dict and cache that

    def _load_frame(self, i):
        path = os.path.join(self.tpdir, f"f{i}.npz")
        d = self._decompress(path)
        d["seg"] = load_seg(path)                      # prefer hand-edit sidecar (io.context)
        return d

    def _preload(self):
        """Warm the remaining frames of the CURRENT context in the background (one per tick) so
        timepoint scrubbing is instant; the panel still opens immediately on the peak frame."""
        if not self.tp:
            return
        todo = [i for i in range(self.n_frames) if i not in self._fr]
        if not todo:
            return
        self._fr[todo[0]] = self._load_frame(todo[0])
        QtCore.QTimer.singleShot(15, self._preload)

    def _F(self, key):
        src = self._frame_npz() if self.tp else self.meta
        keys = src.files if hasattr(src, "files") else src      # NpzFile (.files) or cached dict
        return src[key] if key in keys else None

    def _reset_depth_range(self):
        ax = _PROJ[self.proj]
        self.s_depth.blockSignals(True)
        self.s_depth.setRange(0, self.shape[ax] - 1); self.s_depth.setValue(int(np.clip(self.center, 0, self.shape[ax] - 1)))
        self.s_depth.blockSignals(False)

    # -- render --
    def _render_np(self):
        hu = self._F("hu")
        ax = _PROJ[self.proj]
        c0 = max(self.center - self.slab, 0); c1 = min(self.center + self.slab + 1, self.shape[ax])
        d = self.ds                                         # downsample the in-plane axes (display is scaled
        sl = [slice(None, None, d)] * 3; sl[ax] = slice(c0, c1)   # anyway); the projected (slab) axis stays full
        mip = hu[tuple(sl)].max(axis=ax).T
        rgb = np.repeat(_wl(mip.astype(np.float32), self.level, self.window)[:, :, None], 3, 2).astype(np.float32)
        # OUR seg first, tracking its projected footprint; then the lab masks (CC + TD as SEPARATE
        # colours) are drawn ONLY where our seg is absent — they never overlap ours on screen.
        ours = np.zeros(rgb.shape[:2], bool)
        if self.show_seg:
            seg = self._F("seg")
            if seg is not None:
                for lid, col in ((1, CC_RGB), (2, TD_RGB), (3, NODE_RGB)):
                    m = (seg[tuple(sl)] == lid).max(axis=ax).T
                    if m.any():
                        rgb[m] = 0.35 * rgb[m] + 0.65 * col; ours |= m
        if self.show_lab:
            for key, col in (("lab_cc", LAB_CC_RGB), ("lab_td", LAB_TD_RGB)):
                arr = self.meta[key] if key in self.meta.files else None
                if arr is not None:
                    m = ((arr[tuple(sl)] > 0).max(axis=ax).T) & ~ours    # don't overlap our seg
                    if m.any():
                        rgb[m] = 0.35 * rgb[m] + 0.65 * col
        return np.ascontiguousarray(rgb[::-1].astype(np.uint8))      # flip z -> cranial up

    def _update(self):
        rgb = self._render_np()
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        other = 1 if self.proj == "coronal" else 0          # physical aspect: rows=z, cols=in-plane
        ph, pw = h * self.vox[2], w * self.vox[other]
        qimg = qimg.scaled(int(pw * 2.0), int(ph * 2.0), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                           QtCore.Qt.TransformationMode.SmoothTransformation)
        self._pix = QtGui.QPixmap.fromImage(qimg)
        self._fit()
        emz = self.meta["enh_means"] if "enh_means" in self.meta.files else None
        e = f"   enh {emz[self.frame]:.0f} HU" if emz is not None else ""
        self.lbl.setText(f"{self.tag}  {self.proj}  frame {self.frame}/{self.n_frames - 1}{e}   "
                         f"depth {self.center} ±{self.slab}    s=our l=.mat c/g=proj ,/.=frame a/b/w=window")

    def _fit(self):
        if hasattr(self, "_pix"):
            self.img.setPixmap(self._pix.scaled(self.img.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._fit(); super().resizeEvent(e)

    # -- handlers --
    def _schedule(self):
        self._rtimer.start(10)                            # coalesce rapid slider ticks -> one render

    def _on_acq(self, idx):
        if 0 <= idx < len(self.contexts):
            self._load_context(self.contexts[idx])

    def _on_depth(self, v): self.center = int(v); self._schedule()
    def _on_slab(self, v): self.slab = int(v); self._schedule()
    def _on_tp(self, v): self.frame = int(v); self._schedule()
    def _on_seg(self, on): self.show_seg = bool(on); self._update()
    def _on_lab(self, on): self.show_lab = bool(on); self._update()

    def _set_proj(self, p):
        if p == self.proj:
            return
        self.proj = p
        seg0 = self._F("seg"); ax = _PROJ[p]
        duct = np.argwhere((seg0 >= 1) & (seg0 <= 2)) if seg0 is not None else np.zeros((0, 3))
        self.center = int(duct[:, ax].mean()) if len(duct) else self.shape[ax] // 2
        self._reset_depth_range(); self._update()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.KeyPress:
            k = ev.text().lower()
            if k == "s":
                self.cb_seg.toggle(); return True
            if k == "l":
                self.cb_lab.toggle(); return True
            if k == "c":
                self._set_proj("coronal"); return True
            if k == "g":
                self._set_proj("sagittal"); return True
            if k == "," and self.tp:
                self.s_tp.setValue(max(self.frame - 1, 0)); return True
            if k == "." and self.tp:
                self.s_tp.setValue(min(self.frame + 1, self.n_frames - 1)); return True
            if k in WL_PRESETS:
                self.level, self.window = WL_PRESETS[k]; self._update(); return True
        return super().eventFilter(obj, ev)

    def shot(self, path):
        self.resize(700, 1100); self.show()
        QtWidgets.QApplication.processEvents(); self._update(); QtWidgets.QApplication.processEvents()
        self._pix.save(path)


def main():
    path = sys.argv[1]
    shot = sys.argv[sys.argv.index("--shot") + 1] if "--shot" in sys.argv else None
    app = QtWidgets.QApplication(sys.argv)
    win = MipPanel(path)
    if "--lab" in sys.argv:                              # start with the .mat-seed-vs-ours comparison on
        win.show_lab = True; win.cb_lab.setChecked(True)
    if shot:
        win.shot(shot); print(f"saved {shot}"); return
    win.resize(760, 1180); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
