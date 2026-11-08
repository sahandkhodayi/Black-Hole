"""Create reproducible whole-trajectory splits from a generated NPZ dataset.

Run from the repository root:
    python src/make_splits.py --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lab.splits import make_ray_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/photon_trajectories.npz")
    parser.add_argument("--output", default="data/split_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    # Version-2 datasets preserve their original case IDs. For older v1
    # archives, the stable array position is used as the case ID.
    with np.load(Path(args.data), allow_pickle=False) as archive:
        if "case_id" in archive.files:
            case_ids = np.asarray(archive["case_id"], dtype=np.int64)
        else:
            case_ids = np.arange(len(archive["r0"]), dtype=np.int64)

    manifest = make_ray_splits(
        case_ids,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Split counts:", manifest["counts"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
