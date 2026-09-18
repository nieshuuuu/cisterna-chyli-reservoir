import numpy as np
from cc_reservoir.measure.extract import METHODS, extract_curves, method_agreement

def _blob(center, peak):
    v = np.full((30, 30, 30), 45.0)
    cx, cy, cz = center
    v[cx-1:cx+2, cy-1:cy+2, cz-1:cz+2] = peak
    return v

def test_extract_returns_all_methods_with_curves():
    mask = np.zeros((30, 30, 30), bool); mask[14:17, 14:17, 14:17] = True
    series = [(0., _blob((15,15,15),45.)), (1., _blob((15,15,15),400.)),
              (2., _blob((15,15,15),700.)), (3., _blob((15,15,15),300.))]
    out = extract_curves(series, mask, (1.,1.,1.))
    assert set(out) == set(METHODS)
    for m in METHODS:
        assert len(out[m]["t"]) == 4 and len(out[m]["mass"]) == 4 and len(out[m]["conc"]) == 4
    assert out["dilated+ring"]["mass"][2] > out["dilated+ring"]["mass"][0]

def test_containment_high_when_stationary():
    mask = np.zeros((30, 30, 30), bool); mask[14:17, 14:17, 14:17] = True
    stay = [(float(t), _blob((15,15,15), p)) for t, p in [(0,45.),(1,400.),(2,700.),(3,300.)]]
    a = method_agreement(extract_curves(stay, mask, (1.,1.,1.)))
    assert a["containment_at_peak"] > 0.8            # CC stays inside the fixed ROI

def test_containment_flags_motion():
    mask = np.zeros((30, 30, 30), bool); mask[14:17, 14:17, 14:17] = True
    # CC drifts out of the fixed ROI (dilate 2mm ~ z<=18) into the search region (8mm ~ z<=24)
    move = [(0., _blob((15,15,15),400.)), (1., _blob((15,15,18),600.)),
            (2., _blob((15,15,21),700.)), (3., _blob((15,15,21),700.))]
    a = method_agreement(extract_curves(move, mask, (1.,1.,1.)))
    assert a["containment_at_peak"] < 0.6            # mass moved OUTSIDE the fixed ROI -> flagged
