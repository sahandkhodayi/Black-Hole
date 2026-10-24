from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab.geodesics import RayTrajectory


def plot_trajectory(
    trajectory: RayTrajectory,
    output_path: str | Path,
) -> None:
    """Plot the trajectory and Schwarzschild event horizon."""
    radius = trajectory.radius
    phi = trajectory.phi

    finite = np.isfinite(radius) & (radius <= 50.0)
    x = radius[finite] * np.cos(phi[finite])
    y = radius[finite] * np.sin(phi[finite])

    fig, ax = plt.subplots(figsize=(8, 8))

    horizon = plt.Circle(
        (0, 0),
        1.0,
        color="black",
        label="Event horizon: r = r_s",
    )
    ax.add_patch(horizon)

    ax.plot(x, y, label="Photon trajectory")
    ax.scatter([0], [0], color="black", s=12)

    ax.set(
        title="Schwarzschild black hole: photon trajectory",
        xlabel="x / r_s",
        ylabel="y / r_s",
        aspect="equal",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)