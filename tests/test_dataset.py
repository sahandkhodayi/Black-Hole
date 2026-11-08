"""Tests for packed reference-dataset generation and integrity."""

import json

import numpy as np
import pytest

from lab.reference import build_reference
from lab.trajectory import generate_trajectory_dataset, sample_reference


def test_sample_reference_has_uniform_grid_and_exact_escape_endpoint():
    reference = build_reference(5.0)
    phi, u, v = sample_reference(reference, delta_phi=0.07)

    assert phi[0] == pytest.approx(0.0)
    assert phi[-1] == pytest.approx(reference.escape_angle)
    assert np.all(np.diff(phi) > 0.0)
    assert np.allclose(np.diff(phi)[:-1], 0.07, atol=1e-12)
    assert u[0] == pytest.approx(1.0 / 5.0)
    assert v[0] == pytest.approx(0.0)
    assert abs(u[-1]) < 1e-9


def test_dataset_is_packed_and_manifest_matches_npz(tmp_path):
    dataset_path = tmp_path / "nested" / "rays.npz"
    manifest_path = tmp_path / "nested" / "manifest.json"

    manifest = generate_trajectory_dataset(
        [3.0, 5.0, 10.0],
        output_path=dataset_path,
        manifest_path=manifest_path,
        delta_phi=0.03,
    )

    assert dataset_path.exists()
    assert manifest_path.exists()
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored_manifest["num_escaped"] == 3
    assert manifest["num_requested"] == 3

    with np.load(dataset_path, allow_pickle=False) as archive:
        assert int(archive["schema_version"]) == 2
        assert len(archive["r0"]) == 3
        assert len(archive["case_id"]) == 3
        assert len(archive["offsets"]) == 4
        assert archive["offsets"][0] == 0
        assert archive["offsets"][-1] == len(archive["phi"])
        assert np.all(np.diff(archive["offsets"]) > 0)
        assert len(archive["phi"]) == len(archive["u"]) == len(archive["v"])

        for i, r0 in enumerate(archive["r0"]):
            start, end = archive["offsets"][i:i + 2]
            phi = archive["phi"][start:end]
            u = archive["u"][start:end]
            v = archive["v"][start:end]
            assert np.all(np.diff(phi) > 0)
            assert phi[0] == pytest.approx(0.0)
            assert phi[-1] == pytest.approx(archive["escape_angle"][i])
            assert u[0] == pytest.approx(1.0 / r0)
            assert v[0] == pytest.approx(0.0)
            assert abs(u[-1]) < 1e-9


def test_invalid_dataset_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_trajectory_dataset([], output_path=tmp_path / "x.npz")
    with pytest.raises(ValueError):
        generate_trajectory_dataset([3.0, 1.5], output_path=tmp_path / "x.npz")
