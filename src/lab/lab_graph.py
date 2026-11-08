"""Visual diagnostics for a raw SciPy DOP853 solution."""

import matplotlib.pyplot as plt
import numpy as np

from .geodesics import simulate_ray_raw


def inspect_solution(solution):
    """Plot u(phi), v(phi), phase space, and DOP853's adaptive step sizes."""
    phi = solution.t
    u = solution.y[0]
    v = solution.y[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(phi, u)
    axes[0, 0].set(title="Inverse radius: u(phi)",
                   xlabel="phi (radians)", ylabel="u = 1/r")
    axes[0, 0].grid()

    axes[0, 1].plot(phi, v)
    axes[0, 1].set(title="Derivative: v(phi)",
                   xlabel="phi (radians)", ylabel="v = du/dphi")
    axes[0, 1].grid()

    axes[1, 0].plot(u, v)
    axes[1, 0].set(title="Phase space", xlabel="u", ylabel="v")
    axes[1, 0].grid()

    step_sizes = np.diff(phi)
    axes[1, 1].plot(phi[1:], step_sizes)
    axes[1, 1].set(title="DOP853 adaptive step sizes",
                   xlabel="phi (radians)", ylabel="Delta phi")
    axes[1, 1].grid()

    fig.tight_layout()
    plt.show()

    print("\n--- SOLVER INFORMATION ---")
    print("Success:", solution.success)
    print("Message:", solution.message)
    print("Number of accepted points:", len(solution.t))
    print("RHS evaluations:", solution.nfev)
    print("Final phi:", solution.t[-1])
    print("Final u:", solution.y[0, -1])
    print("Final v:", solution.y[1, -1])
    print("Event times:", solution.t_events)
    if step_sizes.size:
        print("Step sizes:", step_sizes.min(), "to", step_sizes.max())


if __name__ == "__main__":
    raw_solution = simulate_ray_raw(closest_approach=5.0)
    inspect_solution(raw_solution)
