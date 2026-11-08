"""Schwarzschild photon scattering equations and a compact trajectory API.

Units are normalized so the Schwarzschild radius r_s = 1. The independent
variable is phi, the polar angle in radians. We use u = 1/r and v = du/dphi.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]

SCHWARZSCHILD_RADIUS = 1.0
PHOTON_SPHERE_RADIUS = 1.5


@dataclass(frozen=True)
class RayTrajectory:
    """A mirrored photon trajectory sampled by the DOP853 solver."""

    phi: FloatArray
    u: FloatArray
    escaped: bool = False
    escape_angle: float | None = None
    nfev: int = 0

    @property
    def radius(self) -> FloatArray:
        """Convert inverse radius u=1/r back to r, safely handling u <= 0."""
        radius = np.full_like(self.u, np.inf, dtype=np.float64)
        np.divide(1.0, self.u, out=radius, where=self.u > 0.0)
        return radius

    @property
    def deflection_radian(self) -> float | None:
        """Total scattering deflection in radians, or None if unresolved."""
        if not self.escaped or self.escape_angle is None:
            return None
        return float(2.0 * self.escape_angle - np.pi)

    @property
    def deflection_degrees(self) -> float | None:
        """Total scattering deflection in degrees."""
        angle = self.deflection_radian
        return None if angle is None else float(np.degrees(angle))

    @property
    def deflection_angle(self) -> float | None:
        """Compatibility alias for deflection_degrees."""
        return self.deflection_degrees


def _rhs(phi: float, state: FloatArray) -> tuple[float, float]:
    """First-order form of u'' = 3/2*u**2 - u.

    phi is accepted because solve_ivp expects rhs(phi, state). The
    differential equation is autonomous, so the value itself is unused.
    """
    del phi
    u, v = state
    return float(v), float(1.5 * u**2 - u)


def _escape_event(phi: float, state: FloatArray) -> float:
    """Detect an outward crossing of u=0, the asymptotic escape boundary."""
    del phi
    u, _v = state
    return float(u)


_escape_event.terminal = True
_escape_event.direction = -1


def _validate_simulation_inputs(
    closest_approach: float,
    max_angle: float,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[float, float, float, float, float]:
    """Validate public solver settings once for both simulation entry points."""
    try:
        r0 = float(closest_approach)
        angle_limit = float(max_angle)
        relative_tolerance = float(rtol)
        absolute_tolerance = float(atol)
        step_limit = float(max_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("All solver settings must be numeric") from exc

    if not np.isfinite(r0) or r0 <= PHOTON_SPHERE_RADIUS:
        raise ValueError("closest_approach must be finite and > 1.5")
    if not np.isfinite(angle_limit) or angle_limit <= 0.0:
        raise ValueError("max_angle must be finite and positive")
    if not (0.0 < relative_tolerance < 1.0):
        raise ValueError("rtol must be finite and between 0 and 1")
    if not (0.0 < absolute_tolerance < 1.0):
        raise ValueError("atol must be finite and between 0 and 1")
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("max_step must be finite and positive")

    return r0, angle_limit, relative_tolerance, absolute_tolerance, step_limit


def _solve_raw(
    closest_approach: float,
    *,
    max_angle: float,
    rtol: float,
    atol: float,
    max_step: float,
):
    """Shared DOP853 call so the raw and plotted APIs use identical settings."""
    r0, angle_limit, relative_tolerance, absolute_tolerance, step_limit = (
        _validate_simulation_inputs(
            closest_approach, max_angle, rtol, atol, max_step
        )
    )
    return solve_ivp(
        _rhs,
        (0.0, angle_limit),
        y0=(1.0 / r0, 0.0),
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=step_limit,
        events=_escape_event,
    )


def simulate_ray(
    closest_approach: float = 10.0,
    *,
    max_angle: float = 8.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.02,
) -> RayTrajectory:
    """Simulate one scattering ray and mirror it around closest approach.

    The returned phi and u arrays represent both sides of the orbit. The
    escape event is detected on the positive-phi half and represents r tending
    to infinity, not a physical point at a finite radius.
    """
    solution = _solve_raw(
        closest_approach,
        max_angle=max_angle,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 failed: {solution.message}")

    escaped = bool(len(solution.t_events[0]) > 0)
    phi_out = np.asarray(solution.t, dtype=np.float64)
    u_out = np.asarray(solution.y[0], dtype=np.float64)

    # The initial closest-approach sample is shared by both halves, so reverse
    # all later samples on the incoming side to avoid duplicating phi=0.
    phi = np.concatenate((-phi_out[:0:-1], phi_out))
    u = np.concatenate((u_out[:0:-1], u_out))
    escape_angle = float(solution.t_events[0][0]) if escaped else None

    return RayTrajectory(
        phi=phi,
        u=u,
        escaped=escaped,
        escape_angle=escape_angle,
        nfev=int(solution.nfev),
    )


def simulate_ray_raw(
    closest_approach: float = 10.0,
    *,
    max_angle: float = 8.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.02,
):
    """Return SciPy's OdeResult directly for solver diagnostics and plots."""
    result = _solve_raw(
        closest_approach,
        max_angle=max_angle,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not result.success:
        raise RuntimeError(f"DOP853 failed: {result.message}")
    return result
