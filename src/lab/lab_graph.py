import numpy as np
import matplotlib.pyplot as plt

from geodesics import simulate_ray_raw


def inspect_solution(solution):
    phi = solution.t
    u = solution.y[0]
    v = solution.y[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Inverse radius against angular coordinate
    axes[0, 0].plot(phi, u)
    axes[0, 0].set_title("Inverse radius: u(phi)")
    axes[0, 0].set_xlabel("phi (radians)")
    axes[0, 0].set_ylabel("u = 1/r")
    axes[0, 0].grid()

    # 2. Derivative against angular coordinate
    axes[0, 1].plot(phi, v)
    axes[0, 1].set_title("Derivative: v(phi)")
    axes[0, 1].set_xlabel("phi (radians)")
    axes[0, 1].set_ylabel("v = du/dphi")
    axes[0, 1].grid()

    # 3. Phase-space trajectory
    axes[1, 0].plot(u, v)
    axes[1, 0].set_title("Phase space")
    axes[1, 0].set_xlabel("u")
    axes[1, 0].set_ylabel("v")
    axes[1, 0].grid()

    # 4. Numerical step sizes
    step_sizes = np.diff(phi)
    axes[1, 1].plot(phi[1:], step_sizes)
    axes[1, 1].set_title("Solver step sizes")
    axes[1, 1].set_xlabel("phi (radians)")
    axes[1, 1].set_ylabel("Delta phi")
    axes[1, 1].grid()

    plt.tight_layout()
    plt.show()

    print("\n--- SOLVER INFORMATION ---")
    print("Success:", solution.success)
    print("Message:", solution.message)
    print("Number of points:", len(solution.t))
    print("RHS evaluations:", solution.nfev)
    print("Final phi:", solution.t[-1])
    print("Final u:", solution.y[0, -1])
    print("Final v:.", solution.y[1, -1])
    print("Event times:", solution.t_events)
    print("Step sizes:", step_sizes.min(), "to", step_sizes.max())


if __name__ == "__main__":
    solution = simulate_ray_raw(closest_approach=5.0)
    inspect_solution(solution)