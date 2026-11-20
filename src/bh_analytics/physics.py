"""Small, JSON-safe physics payloads for the classroom browser explorer."""
from __future__ import annotations

import math

import numpy as np

from lab.geodesics import _validate_simulation_inputs, simulate_ray_raw


def ray_payload(closest_approach: float, *, max_samples: int = 900) -> dict:
    """Integrate one escaping Schwarzschild ray and return its plotted state.

    The independent variable is phi. The returned orbit is mirrored about
    closest approach so the viewer shows both the incoming and outgoing ray.
    Radius is clipped only for drawing the asymptotic ends of the path; u and
    v retain the numerical solver values.
    """
    r0, _, _, _, _ = _validate_simulation_inputs(
        closest_approach, 8.0, 1e-9, 1e-11, 0.02
    )
    if r0 > 12.0:
        raise ValueError("Closest approach must be between 1.500... and 12 Schwarzschild radii.")
    if max_samples < 50:
        raise ValueError("max_samples must be at least 50.")

    solution = simulate_ray_raw(
        r0,
        max_angle=8.0,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 could not trace this ray: {solution.message}")

    positive_phi = np.asarray(solution.t, dtype=np.float64)
    positive_u = np.asarray(solution.y[0], dtype=np.float64)
    positive_v = np.asarray(solution.y[1], dtype=np.float64)

    phi = np.concatenate((-positive_phi[:0:-1], positive_phi))
    u = np.concatenate((positive_u[:0:-1], positive_u))
    v = np.concatenate((-positive_v[:0:-1], positive_v))

    if len(phi) > max_samples:
        indexes = np.linspace(0, len(phi) - 1, max_samples, dtype=np.int64)
        phi, u, v = phi[indexes], u[indexes], v[indexes]

    radius = np.full_like(u, 20.0)
    np.divide(1.0, u, out=radius, where=u > 1e-5)
    radius = np.clip(radius, 0.0, 20.0)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)

    escaped = bool(len(solution.t_events[0]) > 0)
    escape_angle = float(solution.t_events[0][0]) if escaped else None
    deflection = 2.0 * escape_angle - math.pi if escape_angle is not None else None

    return {
        "closest_approach": r0,
        "escaped": escaped,
        "escape_angle": escape_angle,
        "deflection_degrees": math.degrees(deflection) if deflection is not None else None,
        "samples": len(phi),
        "nfev": int(solution.nfev),
        "phi": phi.tolist(),
        "u": u.tolist(),
        "v": v.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
    }
