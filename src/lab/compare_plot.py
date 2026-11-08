"""Compare fixed-step RK4 against the adaptive DOP853 reference."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .geodesics import _rhs, simulate_ray_raw
from .integrators import rk4_step


def simulate_rk4(closest_approach=5.0, step_size=0.01, max_angle=8.0):
    """Integrate one photon trajectory using fixed-step RK4.

    This is a numerical-method comparison, not part of the RL task. The simple
    endpoint policy stops at the first step that reaches or crosses u=0.
    """
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive")

    phi = 0.0
    state = np.array([1.0 / closest_approach, 0.0], dtype=np.float64)
    phi_values = [phi]
    state_values = [state.copy()]

    while phi < max_angle:
        step = min(step_size, max_angle - phi)
        next_state = rk4_step(_rhs, phi, state, step)
        next_phi = phi + step

        # Stop at escape. Do not convert a non-positive inverse radius to r.
        if next_state[0] <= 0.0:
            break

        phi, state = next_phi, next_state
        phi_values.append(phi)
        state_values.append(state.copy())

    states = np.asarray(state_values)
    return np.asarray(phi_values), states[:, 0]


def to_cartesian(phi, u):
    """Convert only positive inverse-radius values to Cartesian coordinates."""
    values = np.asarray(u, dtype=np.float64)
    valid = values > 0.0
    radius = np.full_like(values, np.inf)
    np.divide(1.0, values, out=radius, where=valid)
    return radius * np.cos(phi), radius * np.sin(phi)


def main():
    closest_approach = 5.0
    step_size = 0.01
    rk_phi, rk_u = simulate_rk4(
        closest_approach=closest_approach,
        step_size=step_size,
    )
    reference = simulate_ray_raw(closest_approach=closest_approach)
    dop_phi = reference.t
    dop_u = reference.y[0]

    common_end = min(rk_phi[-1], dop_phi[-1])
    common_phi = np.linspace(0.0, common_end, 1000)
    rk_common = np.interp(common_phi, rk_phi, rk_u)
    dop_common = np.interp(common_phi, dop_phi, dop_u)
    max_error = np.max(np.abs(rk_common - dop_common))

    print("========== SOLVER COMPARISON ==========")
    print(f"Closest approach: {closest_approach} r_s")
    print(f"RK4 step size: {step_size}")
    print(f"RK4 stored steps: {len(rk_phi) - 1}")
    print(f"DOP853 RHS evaluations: {reference.nfev}")
    print(f"Maximum |u_RK4 - u_DOP853|: {max_error:.10e}")
    print(f"DOP853 success: {reference.success}")
    print(f"DOP853 message: {reference.message}")

    rk_x, rk_y = to_cartesian(rk_phi, rk_u)
    dop_x, dop_y = to_cartesian(dop_phi, dop_u)
    rk_x = np.concatenate((rk_x[::-1], rk_x))
    rk_y = np.concatenate((-rk_y[::-1], rk_y))
    dop_x = np.concatenate((dop_x[::-1], dop_x))
    dop_y = np.concatenate((-dop_y[::-1], dop_y))

    max_plot_radius = 30.0
    rk_mask = np.hypot(rk_x, rk_y) <= max_plot_radius
    dop_mask = np.hypot(dop_x, dop_y) <= max_plot_radius

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(dop_x[dop_mask], dop_y[dop_mask],
            label="DOP853 reference", linewidth=2.5)
    ax.plot(rk_x[rk_mask], rk_y[rk_mask], "--",
            label=f"RK4 (h={step_size})", linewidth=1.8)
    ax.add_patch(Circle((0, 0), 1.0, color="black", label="Event horizon"))
    ax.add_patch(Circle(
        (0, 0), 1.5, fill=False, linestyle=":", color="gray",
        label="Photon sphere",
    ))

    ax.set_title("Schwarzschild photon: RK4 vs DOP853")
    ax.set_xlabel("x / r_s")
    ax.set_ylabel("y / r_s")
    ax.set_aspect("equal")
    ax.set_xlim(-max_plot_radius, max_plot_radius)
    ax.set_ylim(-max_plot_radius, max_plot_radius)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
