import numpy as np
from cc_reservoir.forward.geometry import Grid, ellipsoid_volume
from cc_reservoir.forward.imaging import forward_patch
from cc_reservoir.estimator.fit import fit_patch
from cc_reservoir.estimator.recovery import vcmass, separability

GRID = Grid(20, 44, (0.78, 0.78, 1.0), 2, 1.0)

def test_vcmass_matches_analytic():
    semi = (1.25, 1.25, 9.5)
    V, c, m = vcmass(s=2.0, conc=600.0, base_semi_axes=semi)
    assert np.isclose(V, ellipsoid_volume((2.5, 2.5, 19.0)))
    assert np.isclose(m, V * 600.0)

def test_noisy_recovery_within_tolerance():
    semi = (1.25, 1.25, 9.5); conc = 700.0
    rng = np.random.default_rng(3)
    Vt, _, Mt = vcmass(1.0, conc, semi)
    Verr, Merr = [], []
    for _ in range(8):
        obs = forward_patch(1.0, conc, semi, GRID, noise_hu=15.0, rng=rng)
        out = fit_patch(obs, semi, GRID, init=(1.2, 560.0))
        Ve, _, Me = vcmass(out["s"], out["conc"], semi)
        Verr.append(abs(Ve - Vt) / Vt); Merr.append(abs(Me - Mt) / Mt)
    # mass is more robust to partial volume than volume (the plan's core claim)
    assert np.median(Merr) <= np.median(Verr) + 0.05

def test_separability_range():
    cov = np.array([[1.0, -0.95], [-0.95, 1.0]])
    assert np.isclose(separability(cov), -0.95)
