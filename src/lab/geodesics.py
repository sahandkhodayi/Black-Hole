from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RayTrajectory:
    """A light-ray trajectory in dimensionless Schwarzschild units."""

    phi: FloatArray
    u: FloatArray

    @property
    def radius(self) -> FloatArray:
        """Return radius in units of the Schwarzschild radius."""
        radius = np.full_like(self.u, np.inf)
        np.divide(1.0, self.u, out=radius, where=self.u > 0)
        return radius


def _rhs(
    phi: float,
    state: FloatArray,
) -> tuple[float, float]:
    """First-order form of the Schwarzschild photon equation."""
    del phi

    u, du_dphi = state
    d2u_dphi2 = 1.5 * u**2 - u

    return du_dphi, d2u_dphi2


def simulate_ray(
    closest_approach: float = 10.0,
    *,
    max_angle: float = 8.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> RayTrajectory:
    """Integrate a scattering ray from its point of closest approach."""
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5.")

    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive.")

    if not (0 < rtol < 1 and 0 < atol < 1):
        raise ValueError("rtol and atol must be between 0 and 1.")

    u0 = 1.0 / closest_approach

    solution = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(u0, 0.0),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=0.02,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # The equation is symmetric around the turning point at phi = 0.
    phi_out = solution.t
    u_out = solution.y[0]

    phi = np.concatenate((-phi_out[:0:-1], phi_out))
    u = np.concatenate((u_out[:0:-1], u_out))

    return RayTrajectory(phi=phi, u=u)