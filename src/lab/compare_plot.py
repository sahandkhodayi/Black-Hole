import numpy as np
import matplotlib.pyplot as plt

from geodesics import _rhs, simulate_ray_raw
from integrators import rk4_step


def simulate_rk4(closest_approach=5.0, step_size=0.01, max_angle=8.0):
    """Integrate one photon trajectory using fixed-step RK4."""

    if closest_approach <= 1.5:
        raise ValueError("closest_approach must be > 1.5")

    phi = 0.0
    state = np.array([1.0 / closest_approach, 0.0])

    phi_values = [phi]
    u_values = [state[0]]

    while phi < max_angle:
        h = min(step_size, max_angle - phi)

        state = rk4_step(_rhs, phi, state, h)
        phi += h

        # The photon has reached or crossed the escape boundary.
        if state[0] <= 0:
            break

        phi_values.append(phi)
        u_values.append(state[0])

    return np.array(phi_values), np.array(u_values)


def to_cartesian(phi, u):
    """Convert inverse-radius coordinates to Cartesian coordinates."""

    radius = 1.0 / u

    x = radius * np.cos(phi)
    y = radius * np.sin(phi)

    return x, y


def main():
    closest_approach = 5.0
    step_size = 0.01

    # Our fixed-step RK4 solution.
    rk_phi, rk_u = simulate_rk4(
        closest_approach=closest_approach,
        step_size=step_size,
    )

    # SciPy's adaptive DOP853 reference solution.
    reference = simulate_ray_raw(
        closest_approach=closest_approach,
    )

    dop_phi = reference.t
    dop_u = reference.y[0]

    # Compare only where both solutions have positive u.
    common_end = min(rk_phi[-1], dop_phi[-1])
    common_phi = np.linspace(0, common_end, 1000)

    rk_common = np.interp(common_phi, rk_phi, rk_u)
    dop_common = np.interp(common_phi, dop_phi, dop_u)

    max_error = np.max(np.abs(rk_common - dop_common))

    print("========== SOLVER COMPARISON ==========")
    print(f"Closest approach: {closest_approach} rs")
    print(f"RK4 step size: {step_size}")
    print(f"RK4 steps: {len(rk_phi) - 1}")
    print(f"DOP853 RHS evaluations: {reference.nfev}")
    print(f"Maximum |u_RK4 - u_DOP853|: {max_error:.10e}")
    print(f"DOP853 success: {reference.success}")
    print(f"DOP853 message: {reference.message}")

    # Convert both trajectories to Cartesian coordinates.
    rk_x, rk_y = to_cartesian(rk_phi, rk_u)
    dop_x, dop_y = to_cartesian(dop_phi, dop_u)

    # Mirror the trajectories around the x-axis.
    rk_x = np.concatenate((rk_x[::-1], rk_x))
    rk_y = np.concatenate((-rk_y[::-1], rk_y))

    dop_x = np.concatenate((dop_x[::-1], dop_x))
    dop_y = np.concatenate((-dop_y[::-1], dop_y))

    # Avoid extremely distant points dominating the plot.
    max_plot_radius = 30.0

    rk_mask = np.hypot(rk_x, rk_y) <= max_plot_radius
    dop_mask = np.hypot(dop_x, dop_y) <= max_plot_radius

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(
        dop_x[dop_mask],
        dop_y[dop_mask],
        label="DOP853 reference",
        linewidth=2.5,
    )

    ax.plot(
        rk_x[rk_mask],
        rk_y[rk_mask],
        "--",
        label=f"RK4 (h={step_size})",
        linewidth=1.8,
    )

    # Event horizon: r = 1 in our normalized units.
    black_hole = plt.Circle(
        (0, 0),
        1.0,
        color="black",
        label="Event horizon",
    )

    ax.add_patch(black_hole)

    ax.set_title("Schwarzschild Photon: RK4 vs DOP853")
    ax.set_xlabel("x / Schwarzschild radius")
    ax.set_ylabel("y / Schwarzschild radius")

    ax.set_aspect("equal")
    ax.set_xlim(-max_plot_radius, max_plot_radius)
    ax.set_ylim(-max_plot_radius, max_plot_radius)

    ax.grid(True)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()