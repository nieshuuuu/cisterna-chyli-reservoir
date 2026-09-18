"""A single MPR slice widget (axial/coronal/sagittal) for the context viewer — grayscale CT with
a semi-transparent CC/TD/kidney/bone label overlay, mouse-wheel slice scroll, right-drag
window/level, and left-click crosshair linking. Adapted from ~/Developer/hvsmr2_viz SliceCanvas.

Volume convention: array axes (x, y, z) with x = anterior(low)->posterior(high), y = left-right,
z = caudal->cranial(high). Axial shows anterior up; coronal/sagittal show cranial up. The same
`slice_2d` drives both HU and label so the overlay always registers."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from cc_reservoir.viz.labels import overlay_rgba

# Data axes: x = anterior(low)->posterior(high), y = left-right, z = caudal->cranial(high).
# Slice axis per view: axial cuts z, coronal cuts x (AP), sagittal cuts y (LR).
_AXIS = {"axial": 2, "coronal": 0, "sagittal": 1}


# Edge labels per view (top, bottom, left, right). Quadruped: cranio-caudal (z) lies horizontal.
_MARKS = {"axial": ("A", "P", "R", "L"), "coronal": ("R", "L", "Cd", "Cr"),
          "sagittal": ("A", "P", "Cd", "Cr")}


def slice_2d(vol, view, idx, slab=0):
    """2D display plane. slab=0 → single slice; slab>0 → max-intensity projection over ±slab
    slices (Horos-style MIP). Plane axes map straight through: axial=(x,y) anterior-up; coronal=
    (y,z) and sagittal=(x,z) lie down with caudal→cranial left→right."""
    ax = _AXIS[view]
    sl = [slice(None)] * 3
    if slab > 0:
        sl[ax] = slice(max(idx - slab, 0), min(idx + slab + 1, vol.shape[ax]))
        return vol[tuple(sl)].max(axis=ax)
    sl[ax] = idx
    return vol[tuple(sl)]


def plane_vox(view, vox):
    """(row_mm, col_mm): the physical size of one pixel of this view's `slice_2d` output.

    The grid is anisotropic (z is ~3x finer than in-plane), so anything DISPLAYING a slice has to
    scale by these or the anatomy is stretched along z. Plane axes map straight through from
    slice_2d: axial=(x,y), coronal=(y,z), sagittal=(x,z)."""
    sx, sy, sz = (float(v) for v in vox)
    return {"axial": (sx, sy), "coronal": (sy, sz), "sagittal": (sx, sz)}[view]


def voxel_from_pixel(view, col, row, idx, shape):
    """Inverse of slice_2d: (col,row) in the display array + this view's idx -> (x,y,z) voxel."""
    if view == "axial":           # array (x,y): row->x, col->y ; z=idx
        return (int(row), int(col), int(idx))
    if view == "coronal":         # array (y,z): row->y, col->z ; x=idx
        return (int(idx), int(row), int(col))
    return (int(row), int(idx), int(col))            # sagittal (x,z): row->x, col->z ; y=idx


def pixel_from_voxel(view, voxel, shape):
    """Forward map (x,y,z) -> (col, row) in this view's display array (for the crosshair)."""
    x, y, z = voxel
    if view == "axial":
        return (y, x)
    if view == "coronal":
        return (z, y)
    return (z, x)                 # sagittal


def _wl(arr, level, window):
    lo = level - window / 2.0
    return (np.clip((arr - lo) / max(window, 1e-6), 0.0, 1.0) * 255).astype(np.uint8)


class SliceCanvas(QtWidgets.QWidget):
    voxelClicked = QtCore.Signal(tuple)
    wlChanged = QtCore.Signal(float, float)

    def __init__(self, view, voxel_mm=(1.0, 1.0, 1.0), parent=None):
        super().__init__(parent)
        self.view = view
        self.vox = tuple(float(v) for v in voxel_mm)
        self.hu = None
        self.lab = None
        self.idx = 0
        self.level = 60.0
        self.window = 600.0
        self.slab = 0
        self.crosshair = None
        self._acc = 0.0
        self._qimg = None
        self._wl0 = None
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    # -- data --
    def set_volumes(self, hu, lab, level=None, window=None):
        first = self.hu is None
        self.hu, self.lab = hu, lab
        if level is not None:
            self.level = float(level)
        if window is not None:
            self.window = float(window)
        if first:
            self.idx = hu.shape[_AXIS[self.view]] // 2
        self._refresh(); self.update()

    def set_label(self, lab):
        self.lab = lab; self._refresh(); self.update()

    def set_window_level(self, level, window):
        self.level, self.window = float(level), max(float(window), 1e-3)
        self._refresh(); self.update()

    def set_slab(self, n):
        self.slab = int(max(n, 0)); self._refresh(); self.update()

    def set_crosshair(self, voxel):
        self.crosshair = voxel; self.update()

    def max_idx(self):
        return 0 if self.hu is None else self.hu.shape[_AXIS[self.view]] - 1

    def set_idx(self, idx):
        if self.hu is None:
            return
        idx = int(np.clip(idx, 0, self.max_idx()))
        if idx != self.idx:
            self.idx = idx; self._refresh(); self.update()

    # -- render --
    def _refresh(self):
        if self.hu is None:
            self._qimg = None; return
        g = _wl(slice_2d(self.hu, self.view, self.idx, self.slab), self.level, self.window)
        # The overlay follows the SAME slab as the CT (single slice by default) so it shows the true
        # plane, not a projection — a thick overlay slab projects the duct's meander and reads as a
        # second tube. Axial cuts ACROSS the duct, so cap its slab (a thick axial slab scatters dots).
        lab_slab = min(self.slab, 4) if self.view == "axial" else self.slab
        lab2d = slice_2d(self.lab, self.view, self.idx, lab_slab).astype(np.uint8) \
            if self.lab is not None else np.zeros_like(g)
        rgba = np.ascontiguousarray(overlay_rgba(g, lab2d))
        self._qimg = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], 4 * rgba.shape[1],
                                  QtGui.QImage.Format.Format_RGBA8888).copy()

    def _inplane_mm(self):
        rows, cols = slice_2d(self.hu, self.view, 0).shape       # (wmm, hmm) = (cols, rows) in mm
        r_mm, c_mm = plane_vox(self.view, self.vox)
        return cols * c_mm, rows * r_mm

    def _rect(self):
        wmm, hmm = self._inplane_mm()
        ww, wh = self.width(), self.height()
        s = min(ww / wmm, wh / hmm) if wmm and hmm else 1.0
        dw, dh = int(wmm * s), int(hmm * s)
        return QtCore.QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.GlobalColor.black)
        if self._qimg is None:
            p.setPen(QtCore.Qt.GlobalColor.white)
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "no data"); return
        rect = self._rect()
        p.drawImage(rect, self._qimg)
        if self.crosshair is not None:
            col, row = pixel_from_voxel(self.view, self.crosshair, self.hu.shape)
            w = self._qimg.width(); h = self._qimg.height()
            wx = rect.x() + (col + 0.5) * rect.width() / w
            wy = rect.y() + (row + 0.5) * rect.height() / h
            pen = QtGui.QPen(QtGui.QColor(0, 220, 220, 200), 1, QtCore.Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(rect.x(), int(wy), rect.right(), int(wy))
            p.drawLine(int(wx), rect.y(), int(wx), rect.bottom())
        p.setPen(QtGui.QColor(220, 220, 220))
        p.drawText(8, 18, f"{self.view}  {self.idx}/{self.max_idx()}" + (f"   MIP±{self.slab}" if self.slab else ""))
        t, b, l, r = _MARKS[self.view]      # anatomical edge labels (A/P, R/L, Cr/Cd)
        p.setPen(QtGui.QColor(120, 220, 255))
        cx, cy = rect.x() + rect.width() // 2, rect.y() + rect.height() // 2
        p.drawText(cx - 4, rect.y() + 28, t); p.drawText(cx - 4, rect.bottom() - 6, b)
        p.drawText(rect.x() + 4, cy, l); p.drawText(rect.right() - 18, cy, r)

    def _pos_to_voxel(self, pos):
        rect = self._rect()
        if self._qimg is None or not rect.contains(int(pos.x()), int(pos.y())):
            return None
        col = int((pos.x() - rect.x()) / rect.width() * self._qimg.width())
        row = int((pos.y() - rect.y()) / rect.height() * self._qimg.height())
        return voxel_from_pixel(self.view, col, row, self.idx, self.hu.shape)

    # -- input --
    def wheelEvent(self, e):
        px = e.pixelDelta().y()
        thr = 12.0 if px != 0 else 120.0
        self._acc += px if px != 0 else e.angleDelta().y()
        steps = 0
        while self._acc >= thr:
            self._acc -= thr; steps += 1
        while self._acc <= -thr:
            self._acc += thr; steps -= 1
        if steps:
            self.set_idx(self.idx + steps)
        e.accept()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            v = self._pos_to_voxel(e.position())
            if v is not None:
                self.voxelClicked.emit(v)
        elif e.button() == QtCore.Qt.MouseButton.RightButton:
            self._wl0 = (e.position().x(), e.position().y(), self.level, self.window)
        e.accept()

    def mouseMoveEvent(self, e):
        if self._wl0 is not None and (e.buttons() & QtCore.Qt.MouseButton.RightButton):
            x0, y0, l0, w0 = self._wl0
            sc = max(self.window, 1.0) * 0.004 + 0.6
            self.wlChanged.emit(l0 - (e.position().y() - y0) * sc,
                                max(w0 + (e.position().x() - x0) * sc, 1.0))
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.RightButton:
            self._wl0 = None
        e.accept()
