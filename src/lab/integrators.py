"""Small fixed-step numerical integrators used for method comparisons."""

import numpy as np


def rk4_step(rhs, phi, state, step_size):
    """Advance one first-order ODE by a single classical fourth-order RK step.

    The four derivative estimates sample the beginning, two midpoints, and
    end of the step. Their weighted average gives RK4 its fourth-order local
    accuracy for sufficiently smooth problems.
    """
    state = np.asarray(state, dtype=np.float64)

    if state.ndim != 1:
        raise ValueError("state must be a one-dimensional array")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values")
    if not np.isfinite(phi):
        raise ValueError("phi must be finite")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")

    # k1: slope at the beginning of the interval.
    k1 = np.asarray(rhs(phi, state), dtype=np.float64)
    if k1.shape != state.shape:
        raise ValueError("rhs must return derivatives with the same shape as state")

    # k2 and k3: two midpoint slope estimates, each using the previous slope.
    half_step = step_size / 2.0
    midpoint = phi + half_step
    k2 = np.asarray(rhs(midpoint, state + half_step * k1), dtype=np.float64)
    if k2.shape != state.shape:
        raise ValueError("rhs must return derivatives with the same shape as state")
    k3 = np.asarray(rhs(midpoint, state + half_step * k2), dtype=np.float64)
    if k3.shape != state.shape:
        raise ValueError("rhs must return derivatives with the same shape as state")

    # k4: slope at the end, estimated from the third midpoint slope.
    k4 = np.asarray(
        rhs(phi + step_size, state + step_size * k3),
        dtype=np.float64,
    )
    if k4.shape != state.shape:
        raise ValueError("rhs must return derivatives with the same shape as state")

    # The classical RK4 weights are 1, 2, 2, 1.
    result = state + (step_size / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("RK4 produced a non-finite state")
    return result
