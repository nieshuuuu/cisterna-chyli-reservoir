import numpy as np
from cc_reservoir.forward.geometry import Grid
from cc_reservoir.forward.imaging import forward_patch
from cc_reservoir.estimator.fit import fit_patch

GRID = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)

def test_recovers_params_noiseless():
    semi = (1.25, 1.25, 9.5)
    obs = forward_patch(s=1.3, conc=700.0, base_semi_axes=semi, grid=GRID)  # no noise
    out = fit_patch(obs, base_semi_axes=semi, grid=GRID, init=(1.0, 500.0))
    assert out["success"]
    assert np.isclose(out["s"], 1.3, rtol=0.02)
    assert np.isclose(out["conc"], 700.0, rtol=0.02)

def test_cov_is_2x2():
    semi = (1.25, 1.25, 9.5)
    obs = forward_patch(1.0, 700.0, semi, GRID)
    out = fit_patch(obs, base_semi_axes=semi, grid=GRID, init=(1.1, 600.0))
    assert out["cov"].shape == (2, 2)
