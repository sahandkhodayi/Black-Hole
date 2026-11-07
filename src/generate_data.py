import argparse

import numpy as np

from lab.trajectory import generate_trajectory_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference Schwarzschild photon trajectories."
    )
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--r-min", type=float, default=2.2)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--delta-phi", type=float, default=0.01)
    parser.add_argument("--max-angle", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count < 2:
        parser.error("--count must be at least 2")
    if args.r_min <= 1.5:
        parser.error("--r-min must be greater than 1.5")
    if args.r_max <= args.r_min:
        parser.error("--r-max must be greater than --r-min")

    # A fixed seed makes the parameter ordering reproducible.
    rng = np.random.default_rng(args.seed)
    r0_values = np.linspace(args.r_min, args.r_max, args.count)
    rng.shuffle(r0_values)

    manifest = generate_trajectory_dataset(
        r0_values,
        delta_phi=args.delta_phi,
        max_angle=args.max_angle,
    )

    print("Generated trajectories:", manifest["num_escaped"])
    print("Unresolved cases:", len(manifest["unresolved"]))
    print("Dataset:", manifest["data_file"])


if __name__ == "__main__":
    main()