"""High-accuracy DOP853 reference solutions for Schwarzschild photon rays.

The independent variable is the polar angle phi (radians), not physical time.
The state returned by solve_ivp is [u, v], where u = 1/r and v = du/dphi.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .geodesics import _escape_event, _rhs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceRay:
    """One dense-output reference trajectory from closest approach to escape."""

    solution: Any
    closest_approach: float

    @property
    def escaped(self) -> bool:
        """Whether the outward u=0 event was detected before max_angle."""
        return bool(self.solution.t_events and len(self.solution.t_events[0]) > 0)

    @property
    def escape_angle(self) -> float | None:
        """Angular coordinate phi at the detected escape event, in radians."""
        events = self.solution.t_events[0]
        return float(events[0]) if len(events) else None

    @property
    def deflection_radian(self) -> float | None:
        """Total scattering deflection, using symmetry about closest approach."""
        angle = self.escape_angle
        if angle is None:
            return None
        return float(2.0 * angle - np.pi)

    @property
    def rhs_evaluations(self) -> int:
        """Number of right-hand-side evaluations performed by DOP853."""
        return int(self.solution.nfev)

    def state_at(self, phi: ArrayLike) -> FloatArray:
        """Evaluate [u, v] at one or more angles using dense interpolation.

        Dense output is a continuous interpolant across the interval integrated
        by solve_ivp. It is useful when a dataset needs a fixed angular grid,
        even though DOP853 internally chooses adaptive, nonuniform steps.
        """
        interpolant = self.solution.sol
        if interpolant is None:
            raise RuntimeError(
                "Reference was not created with dense_output=True"
            )

        values = np.asarray(phi, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Requested angles must all be finite")

        # solution.t stores the adaptive solver's accepted angular nodes.
        # Its first and last values define the interval covered by the
        # dense interpolant. Do not silently extrapolate outside that interval.
        start = float(self.solution.t[0])
        end = float(self.solution.t[-1])
        if np.any(values < start) or np.any(values > end):
            raise ValueError(
                f"Requested angle must be inside [{start}, {end}] radians"
            )

        # A scalar angle returns shape (2,); N angles return shape (2, N).
        # Row 0 is u and row 1 is v.
        return np.asarray(interpolant(values), dtype=np.float64)


def _as_positive_finite(name: str, value: float) -> float:
    """Convert a numeric setting to float and reject invalid solver settings."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc

    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def build_reference(
    closest_approach: float,
    *,
    max_angle: float = 20.0,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_step: float = 0.01,
) -> ReferenceRay:
    """Integrate one scattering ray from its turning point toward escape.

    We use Schwarzschild-radius units (r_s = 1), so the photon sphere is at
    r = 1.5. This initial-value setup describes escaping/scattering rays only.
    """
    try:
        r0 = float(closest_approach)
    except (TypeError, ValueError) as exc:
        raise ValueError("closest_approach must be finite and > 1.5") from exc

    if not np.isfinite(r0) or r0 <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")

    angle_limit = _as_positive_finite("max_angle", max_angle)
    relative_tolerance = _as_positive_finite("rtol", rtol)
    absolute_tolerance = _as_positive_finite("atol", atol)
    step_limit = _as_positive_finite("max_step", max_step)

    if relative_tolerance >= 1.0 or absolute_tolerance >= 1.0:
        raise ValueError("rtol and atol must be smaller than 1")

    result = solve_ivp(
        _rhs,
        (0.0, angle_limit),
        y0=(1.0 / r0, 0.0),
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=step_limit,
        events=_escape_event,
        dense_output=True,
    )

    if not result.success:
        raise RuntimeError(f"DOP853 failed: {result.message}")

    return ReferenceRay(solution=result, closest_approach=r0)
