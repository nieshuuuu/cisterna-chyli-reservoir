import numpy as np
from scipy.optimize import least_squares
from ..forward.imaging import forward_patch

def fit_patch(observed, base_semi_axes, grid, init=(1.0, 500.0)):
    """MAP fit of (size-scale s, concentration conc) to an observed voxel patch,
    using the same forward model (SSoT). Laplace covariance from the Jacobian."""
    observed = np.asarray(observed, float)

    def resid(theta):
        s, conc = theta
        pred = forward_patch(max(s, 1e-3), conc, base_semi_axes, grid)
        return (pred - observed).ravel()

    # diff_step=0.05 is needed because render_ellipsoid is piecewise-constant:
    # its gradient wrt s is zero at the default finite-diff eps (~1e-8), so LM
    # stalls after 2 evaluations without this larger step.
    # Caveat: accuracy degrades for very small objects (s < ~0.9), where few
    # fine-voxels lie on the boundary and the FD gradient is alias-noisy.
    # Phase 1b fix = supersampled / soft (anti-aliased) rendering.
    res = least_squares(resid, x0=list(init), method="lm", max_nfev=200, diff_step=0.05)
    n, p = res.fun.size, 2
    dof = max(n - p, 1)
    sigma2 = 2.0 * res.cost / dof  # SciPy least_squares cost = 0.5*Σr², so 2*cost = Σr²
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * sigma2
    except np.linalg.LinAlgError:
        # Singular J^T J means s and conc are perfectly degenerate here: NaN is the
        # honest signal (it surfaces via separability -> nanmedian), not a swallowed error.
        cov = np.full((p, p), np.nan)
    return {"s": float(res.x[0]), "conc": float(res.x[1]),
            "cov": cov, "success": bool(res.success)}
