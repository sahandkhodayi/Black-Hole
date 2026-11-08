"""Test data inspection compatibility with legacy and current NPZ layouts."""

import numpy as np

from inspect_data import load_dataset, validate_dataset
from lab.trajectory import generate_trajectory_dataset


def test_legacy_v1_archive_gets_stable_synthetic_case_ids(tmp_path):
    current_path = tmp_path / "current.npz"
    legacy_path = tmp_path / "legacy.npz"
    generate_trajectory_dataset(
        [3.0, 5.0, 10.0],
        output_path=current_path,
        manifest_path=tmp_path / "manifest.json",
        delta_phi=0.04,
    )

    # Remove only the fields introduced in schema v2 to imitate the tracked
    # schema-v1 archive that predates explicit case IDs.
    with np.load(current_path, allow_pickle=False) as current:
        legacy = {
            name: np.asarray(current[name])
            for name in current.files
            if name not in {"schema_version", "case_id"}
        }
    np.savez_compressed(legacy_path, **legacy)

    loaded = load_dataset(legacy_path)
    validate_dataset(loaded)
    assert loaded["case_id"].tolist() == [0, 1, 2]
