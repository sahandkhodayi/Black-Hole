import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from lab.geodesics import simulate_ray


def plot_black_hole(closest_approaches=(3.0, 5.0, 10.0)):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Black hole: event horizon, radius = 1
    horizon = Circle(
        (0, 0),
        radius=1.0,
        color="black",
        zorder=5,
        label="Event horizon",
    )
    ax.add_patch(horizon)

    # Photon sphere: radius = 1.5
    photon_sphere = Circle(
        (0, 0),
        radius=1.5,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label="Photon sphere",
    )
    ax.add_patch(photon_sphere)

    for distance in closest_approaches:
        trajectory = simulate_ray(closest_approach=distance)

        r = trajectory.radius
        phi = trajectory.phi

        # Ignore infinite radii near escape and keep the view readable.
        visible = np.isfinite(r) & (r <= 30)

        x = r[visible] * np.cos(phi[visible])
        y = r[visible] * np.sin(phi[visible])

        ax.plot(x, y, linewidth=1.8, label=f"Closest approach: {distance} rs")

    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("x / rs")
    ax.set_ylabel("y / rs")
    ax.set_title("Light Trajectories Around a Schwarzschild Black Hole")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_black_hole()