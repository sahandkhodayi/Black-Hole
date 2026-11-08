"""Inspect a packed reference dataset and render a small trajectory gallery.

Run from the repository root:
    python src/inspect_data.py
    python src/inspect_data.py --show
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


REQUIRED_ARRAYS = {
    "case_id", "r0", "phi", "u", "v", "offsets",
    "escape_angle", "deflection_radian",
}


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load numeric arrays only; pickle-backed objects are deliberately disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_dataset(data: dict[str, np.ndarray]) -> None:
    """Check packed-array dimensions and each ray's physical endpoints."""
    ray_count = len(data["r0"])
    packed_count = len(data["phi"])

    for name in ("u", "v"):
        if len(data[name]) != packed_count:
            raise ValueError(f"{name!r} must have the same length as phi")

    for name in ("case_id", "escape_angle", "deflection_radian"):
        if len(data[name]) != ray_count:
            raise ValueError(f"{name!r} must contain one value per ray")

    offsets = data["offsets"]
    if len(offsets) != ray_count + 1:
        raise ValueError("offsets must have one more element than the ray count")
    if offsets[0] != 0 or offsets[-1] != packed_count:
        raise ValueError("offsets must start at 0 and end at the packed sample count")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every stored trajectory must contain at least one sample")

    for name in ("r0", "phi", "u", "v", "escape_angle", "deflection_radian"):
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"{name!r} contains NaN or infinite values")

    for index in range(ray_count):
        start, end = offsets[index:index + 2]
        phi = data["phi"][start:end]
        u = data["u"][start:end]
        v = data["v"][start:end]

        if np.any(np.diff(phi) <= 0):
            raise ValueError(f"phi must be strictly increasing for ray {index}")
        if not np.isclose(phi[0], 0.0, atol=1e-12):
            raise ValueError(f"ray {index} does not start at phi=0")
        if not np.isclose(phi[-1], data["escape_angle"][index], atol=1e-10):
            raise ValueError(f"ray {index} does not end at its escape angle")
        if not np.isclose(u[0], 1.0 / data["r0"][index], rtol=1e-9, atol=1e-12):
            raise ValueError(f"ray {index} has an incorrect initial inverse radius")
        if not np.isclose(v[0], 0.0, atol=1e-12):
            raise ValueError(f"ray {index} has an incorrect initial derivative")
        if abs(u[-1]) > 1e-9:
            raise ValueError(f"ray {index} does not end at the u=0 escape event")


def plot_gallery(
    data: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    max_curves: int = 12,
    show: bool = False,
) -> Path:
    """Plot representative mirrored orbits and their u(phi), v(phi) histories."""
    if max_curves < 1:
        raise ValueError("max_curves must be at least 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(data["r0"])
    selected_positions = np.linspace(
        0, len(order) - 1, min(max_curves, len(order)), dtype=int
    )
    selected = order[selected_positions]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    orbit_ax, u_ax, v_ax = axes
    orbit_ax.add_patch(Circle((0, 0), 1.0, color="black", label="Event horizon"))
    orbit_ax.add_patch(Circle(
        (0, 0), 1.5, fill=False, linestyle="--", color="gray",
        label="Photon sphere",
    ))

    for index in selected:
        start, end = data["offsets"][index:index + 2]
        phi = data["phi"][start:end]
        u = data["u"][start:end]
        v = data["v"][start:end]
        label = f"r0={data['r0'][index]:.2f} r_s"

        # Mirror the simulated half around the closest-approach axis, without
        # duplicating the phi=0 point.
        full_phi = np.concatenate((-phi[:0:-1], phi))
        full_u = np.concatenate((u[:0:-1], u))
        positive = full_u > 0.0
        radius = np.full_like(full_u, np.inf)
        np.divide(1.0, full_u, out=radius, where=positive)

        # The plotted window focuses on bending near the hole rather than
        # allowing the asymptotically distant end to dominate the axes.
        visible = positive & (radius <= 30.0)
        orbit_ax.plot(
            radius[visible] * np.cos(full_phi[visible]),
            radius[visible] * np.sin(full_phi[visible]),
            label=label,
        )
        u_ax.plot(phi, u, label=label)
        v_ax.plot(phi, v, label=label)

    orbit_ax.set(title="Photon trajectories", xlabel="x / r_s", ylabel="y / r_s")
    orbit_ax.set_aspect("equal")
    orbit_ax.set_xlim(-15, 15)
    orbit_ax.set_ylim(-15, 15)
    orbit_ax.legend(fontsize=8)

    u_ax.set(title="Inverse radius u(phi)", xlabel="phi (radians)", ylabel="u = 1/r")
    u_ax.grid(alpha=0.25)
    u_ax.legend(fontsize=7)

    v_ax.set(title="Radial derivative v(phi)", xlabel="phi (radians)", ylabel="v = du/dphi")
    v_ax.grid(alpha=0.25)
    v_ax.legend(fontsize=7)

    fig.suptitle("DOP853 reference trajectory dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/photon_trajectories.npz")
    parser.add_argument("--manifest", default="data/trajectory_manifest.json")
    parser.add_argument("--output", default="outputs/reference_trajectory_gallery.png")
    parser.add_argument("--max-curves", type=int, default=12)
    parser.add_argument("--show", action="store_true",
                        help="Also open the figure in an interactive window.")
    args = parser.parse_args()

    data = load_dataset(args.data)
    validate_dataset(data)

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("num_escaped", -1)) != len(data["r0"]):
            raise ValueError("Manifest num_escaped does not match the NPZ dataset")
        print("Manifest schema:", manifest.get("schema_version", "unknown"))

    print("Dataset:", args.data)
    print("Trajectories:", len(data["r0"]))
    print("Packed samples:", len(data["phi"]))
    print(f"r0 range: {data['r0'].min():.6g} to {data['r0'].max():.6g} r_s")
    print(
        "Deflection range: "
        f"{np.degrees(data['deflection_radian']).min():.6g} to "
        f"{np.degrees(data['deflection_radian']).max():.6g} degrees"
    )

    output = plot_gallery(
        data, args.output, max_curves=args.max_curves, show=args.show
    )
    print("Saved trajectory gallery:", output)


if __name__ == "__main__":
    main()
