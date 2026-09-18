"""Label palette + grayscale<->label compositing for the MPR overlay.
Label ids: 1 CC (green), 2 TD (teal) = iodinated lymph; 3 kidney, 4 bone = HU landmark context."""
import numpy as np

#                 0 bg        1 CC      2 TD      3 kidney  4 bone
_HEX = ["#00000000", "#5aa469", "#3a8f9c", "#e0a33a", "#b5916a"]   # bone = tan, tells it from teal duct
_ALPHA = [0.0, 0.90, 0.90, 0.30, 0.18]                # CC/TD near-solid; bone very faint (toggle with 'k')
LABEL_NAME = {1: "CC", 2: "TD", 3: "kidney", 4: "bone"}
_COLOR_WORD = {1: "green", 2: "teal", 3: "gold", 4: "tan"}   # plain-English _HEX, for figure legends


def legend(label_ids):
    """'CC green / TD teal' for the labels actually drawn — a caption naming a colour that isn't in
    the panel is worse than no caption, so callers pass what's in the seg rather than a fixed string."""
    return " / ".join(f"{LABEL_NAME[i]} {_COLOR_WORD[i]}" for i in sorted(label_ids) if i in LABEL_NAME)

# The rough `.mat` seed shipped with the data (SEGMENT_dcm/{CC,TD}_dcm_01.mat). Deliberately NOT
# green/teal: those are locked to OUR seg, and a comparison overlay must never blur the provenance.
# (This is also not the lab's real clinical seg — that exists only as Vitrea screenshots.)
LAB_CC_RGB = np.array([245, 225, 40], np.float32)     # seed CC — yellow
LAB_TD_RGB = np.array([229, 46, 226], np.float32)     # seed TD — magenta


def _rgba(h):
    h = h.lstrip("#")
    if len(h) == 8:
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4, 6)]
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1.0]


LABEL_RGBA = np.array([_rgba(h) for h in _HEX], float)
N_LABELS = len(_HEX)


def overlay_rgba(gray_u8, label_slice):
    """RGBA image (H,W,4 uint8): grayscale base with each label alpha-blended on top (per-label alpha)."""
    h, w = gray_u8.shape
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = gray_u8
    out[..., 3] = 255
    for lid in range(1, N_LABELS):
        m = label_slice == lid
        if not m.any():
            continue
        a = _ALPHA[lid]
        out[m, :3] = (out[m, :3] * (1 - a) + LABEL_RGBA[lid, :3] * 255 * a).astype(np.uint8)
    return out
