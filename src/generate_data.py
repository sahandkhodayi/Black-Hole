"""Command-line entry point for generating the reference photon dataset.

Run from the repository root:
    python -m pip install -e ".[dev]"
    python src/generate_data.py --count 30
"""

import argparse

import numpy as np

from lab.trajectory import generate_trajectory_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DOP853 reference trajectories for Schwarzschild photons."
    )
    parser.add_argument("--count", type=int, default=120,
                        help="Number of closest-approach values to sample.")
    parser.add_argument("--r-min", type=float, default=2.2,
                        help="Smallest closest approach in Schwarzschild radii; must exceed 1.5.")
    parser.add_argument("--r-max", type=float, default=12.0,
                        help="Largest closest approach in Schwarzschild radii.")
    parser.add_argument("--delta-phi", type=float, default=0.01,
                        help="Uniform angular sampling interval in radians.")
    parser.add_argument("--max-angle", type=float, default=20.0,
                        help="Maximum integrated angle before a ray is marked unresolved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used to shuffle the evenly spaced radius values.")
    parser.add_argument("--output", default="data/photon_trajectories.npz",
                        help="Destination NPZ file.")
    parser.add_argument("--manifest", default="data/trajectory_manifest.json",
                        help="Destination JSON manifest.")
    args = parser.parse_args()

    if args.count < 2:
        parser.error("--count must be at least 2")
    if not np.isfinite(args.r_min) or args.r_min <= 1.5:
        parser.error("--r-min must be finite and greater than 1.5")
    if not np.isfinite(args.r_max) or args.r_max <= args.r_min:
        parser.error("--r-max must be finite and greater than --r-min")
    if not np.isfinite(args.delta_phi) or args.delta_phi <= 0:
        parser.error("--delta-phi must be finite and positive")
    if not np.isfinite(args.max_angle) or args.max_angle <= 0:
        parser.error("--max-angle must be finite and positive")

    # Cover the complete radius interval, then shuffle its order. A fixed
    # seed keeps the case ordering reproducible.
    rng = np.random.default_rng(args.seed)
    r0_values = np.linspace(args.r_min, args.r_max, args.count)
    rng.shuffle(r0_values)

    manifest = generate_trajectory_dataset(
        r0_values,
        output_path=args.output,
        manifest_path=args.manifest,
        delta_phi=args.delta_phi,
        max_angle=args.max_angle,
        seed=args.seed,
    )

    print(f"Requested trajectories: {manifest['num_requested']}")
    print(f"Successfully escaped:   {manifest['num_escaped']}")
    print(f"Unresolved cases:       {len(manifest['unresolved'])}")
    print(f"Dataset:                {manifest['data_file']}")
    print(f"Manifest:               {args.manifest}")


if __name__ == "__main__":
    main()
