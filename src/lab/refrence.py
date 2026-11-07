from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .geodesics import _rhs, _escape_event


@dataclass(frozen=True)
class ReferenceRay:
    solution: object
    closest_approach: float

    @property
    def escaped(self):
        return len(self.solution.t_events[0]) > 0

    @property
    def escape_angle(self):
        events = self.solution.t_events[0]
        return float(events[0]) if len(events) else None

    @property
    def deflection_radian(self):
        angle = self.escape_angle
        if angle is None:
            return None
        return 2.0 * angle - np.pi

    @property
    def rhs_evaluations(self):
        return int(self.solution.nfev)

    def state_at(self, phi):
        if self.solution.sol is None:
            raise RuntimeError("Reference was not created with dense_output=True")

        values = np.asarray(phi, dtype=float)

        if np.any(values < self.solution.t[0]) or np.any(values > self.solution.t[-1]):
            raise ValueError("Requested angle is outside the integrated interval")

        return np.asarray(self.solution.sol(values), dtype=np.float64)


def build_reference(
    closest_approach,
    *,
    max_angle=20.0,
    rtol=1e-12,
    atol=1e-14,
    max_step=0.01,
):
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")

    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive")

    result = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(1.0 / closest_approach, 0.0),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=_escape_event,
        dense_output=True,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return ReferenceRay(
        solution=result,
        closest_approach=float(closest_approach),
    )