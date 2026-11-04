import numpy as np


def rk4_step(rhs, phi, state, step_size):
    """Advance a first-order ODE by one RK4 step."""

    state = np.asarray(state, dtype=float)

    if state.ndim != 1:
        raise ValueError("state must be a one-dimensional array")

    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")

    k1 = np.asarray(rhs(phi, state), dtype=float)
    k2 = np.asarray(
        rhs(phi + step_size / 2, state + step_size * k1 / 2),
        dtype=float,
    )
    k3 = np.asarray(
        rhs(phi + step_size / 2, state + step_size * k2 / 2),
        dtype=float,
    )
    k4 = np.asarray(
        rhs(phi + step_size, state + step_size * k3),
        dtype=float,
    )

    if any(k.shape != state.shape for k in (k1, k2, k3, k4)):
        raise ValueError("rhs must return derivatives with the same shape as state")

    return state + (step_size / 6) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )