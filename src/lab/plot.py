"""Plot mirrored Schwarzschild photon-scattering trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .geodesics import simulate_ray


def plot_rays(closest_approaches=(3.0, 5.0, 10.0), output="outputs/rays.png"):
    """Simulate selected closest approaches and save their spatial curves."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Circle((0, 0), 1.0, color="black", label="Event horizon"))
    ax.add_patch(Circle(
        (0, 0), 1.5, fill=False, linestyle="--",
        color="gray", label="Photon sphere",
    ))

    for r0 in closest_approaches:
        ray = simulate_ray(closest_approach=float(r0))
        if not ray.escaped:
            print(f"Unresolved at max_angle: r0={r0:g}")
            continue

        radius = ray.radius
        visible = np.isfinite(radius) & (radius <= 30.0)
        x = radius[visible] * np.cos(ray.phi[visible])
        y = radius[visible] * np.sin(ray.phi[visible])

        angle = ray.deflection_degrees
        angle_text = f"{angle:.4f} deg" if angle is not None else "N/A"
        ax.plot(x, y, label=f"r0={r0:g} r_s; alpha={angle_text}")
        print(
            f"r0={r0:g} r_s | deflection={angle_text} "
            f"| RHS evaluations={ray.nfev}"
        )

    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("x / r_s")
    ax.set_ylabel("y / r_s")
    ax.set_title("Schwarzschild photon trajectories")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_rays()
