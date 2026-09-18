import numpy as np
from cc_reservoir.forward.psf import apply_psf

def test_psf_conserves_interior_mass():
    field = np.zeros((40, 40, 40))
    field[18:22, 18:22, 18:22] = 7.0          # centred blob, wide margin
    blurred = apply_psf(field, sigma_vox=2.0)
    assert np.isclose(field.sum(), blurred.sum(), rtol=1e-6)

def test_psf_spreads_signal():
    field = np.zeros((21, 21, 21)); field[10, 10, 10] = 1.0
    blurred = apply_psf(field, sigma_vox=1.5)
    assert blurred[10, 10, 10] < 1.0 and blurred[10, 11, 10] > 0.0
