import numpy as np

# Aspect families spanning the Moazzam morphology range (swine-scaled).
MORPHOLOGIES = ("saccular", "fusiform", "tubular", "thin")

_ASPECT = {            # (in-plane semi-axis range mm, long semi-axis range mm)
    "saccular": ((2.2, 4.0), (5.0, 9.0)),     # rounder, shorter
    "fusiform": ((1.6, 3.0), (8.0, 14.0)),
    "tubular":  ((1.0, 2.0), (10.0, 18.0)),
    "thin":     ((0.6, 1.3), (12.0, 22.0)),   # worst partial volume
}

def sample_cc(kind, rng):
    """Return (semi_axes_mm=(a, a, c), conc_hu) for a morphology family."""
    (a_lo, a_hi), (c_lo, c_hi) = _ASPECT[kind]
    a = float(rng.uniform(a_lo, a_hi))
    c = float(rng.uniform(c_lo, c_hi))
    conc = float(rng.uniform(400.0, 1100.0))
    return (a, a, c), conc
