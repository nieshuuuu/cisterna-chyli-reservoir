import numpy as np
from cc_reservoir.forward.morphology import MORPHOLOGIES, sample_cc

def test_all_kinds_return_valid_shapes():
    rng = np.random.default_rng(1)
    for kind in MORPHOLOGIES:
        semi, conc = sample_cc(kind, rng)
        assert len(semi) == 3 and all(x > 0 for x in semi)
        assert 400.0 <= conc <= 1100.0          # implementation's CC enhancement range (HU)
        assert 0.6 <= semi[0] <= 4.0            # thin in-plane semi-axis (mm)
        assert 5.0 <= semi[2] <= 22.0           # elongated long semi-axis (mm)

def test_kinds_have_distinct_aspect():
    rng = np.random.default_rng(2)
    aspects = {k: sample_cc(k, rng)[0][2] / sample_cc(k, rng)[0][0] for k in MORPHOLOGIES}
    assert max(aspects.values()) > 2 * min(aspects.values())
