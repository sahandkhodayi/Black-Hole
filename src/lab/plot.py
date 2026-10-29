
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from lab.geodesics import simulate_ray


def plot_rays(closest_approaches=(3.0, 5.0, 10.0), output="outputs/rays.png"):
    """Run example ray simulations and save a trajectory figure."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.add_patch(
        Circle((0, 0), 1.0, color="black", label="Event horizon")
    )
    ax.add_patch(
        Circle(
            (0, 0), 1.5, fill=False, linestyle="--",
            color="gray", label="Photon sphere"
        )
    )

    for r_min in closest_approaches:
        ray = simulate_ray(closest_approach=float(r_min))

        if not ray.escaped:
            print(f"Unresolved at max_angle: r_min={r_min:g}")
            continue

        radius = np.asarray(ray.radius)
        phi = np.asarray(ray.phi)

        # Avoid infinite radii and points too far from the black hole.
        visible = np.isfinite(radius) & (radius <= 30)

        x = radius[visible] * np.cos(phi[visible])
        y = radius[visible] * np.sin(phi[visible])

        deflection = ray.deflection_angle
        deflection_text = (
            f"{deflection:.6f} rad"
            if deflection is not None
            else "N/A"
        )

        ax.plot(x, y, label=f"r_min = {r_min:g} r_s")

        print(
            f"r_min={r_min:g} | deflection={deflection_text} "
            f"| nfev={ray.nfev}"
        )

    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("x / r_s")
    ax.set_ylabel("y / r_s")
    ax.set_title("Schwarzschild light-ray trajectories")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    plot_rays()
