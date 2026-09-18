import numpy as np

from cc_reservoir.viz.labels import LABEL_RGBA, overlay_rgba, N_LABELS


def test_palette_background_transparent_and_has_all_labels():
    assert LABEL_RGBA.shape == (N_LABELS, 4) and N_LABELS == 5   # bg, CC, TD, kidney, bone
    assert LABEL_RGBA[0, 3] == 0.0           # background fully transparent


def test_overlay_blends_label_over_gray():
    gray = np.full((4, 4), 100, np.uint8)
    lab = np.zeros((4, 4), np.uint8); lab[0, 0] = 1          # CC at one pixel
    out = overlay_rgba(gray, lab)
    assert out.shape == (4, 4, 4) and out.dtype == np.uint8
    assert (out[1, 1, :3] == 100).all()      # un-labelled pixel stays gray
    assert not (out[0, 0, :3] == 100).all()  # labelled pixel tinted toward CC green
