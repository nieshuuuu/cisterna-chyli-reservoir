import numpy as np
from ..forward.geometry import ellipsoid_volume

def vcmass(s, conc, base_semi_axes):
    """Return (volume mm^3, concentration HU, contrast mass HU*mm^3)."""
    semi = tuple(x * float(s) for x in base_semi_axes)
    V = ellipsoid_volume(semi)
    return V, float(conc), V * float(conc)

def separability(cov):
    """Correlation between the s and conc estimates. Near +/-1 => degenerate
    (size and concentration trade off; V and c not individually identifiable)."""
    sd = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sd)) or np.any(sd == 0):
        return float("nan")
    return float(cov[0, 1] / (sd[0] * sd[1]))
