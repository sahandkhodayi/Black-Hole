"""Tests that complete photon trajectories remain isolated across data splits."""

import numpy as np
import pytest

from lab.splits import make_ray_splits


def test_split_is_reproducible_and_has_no_case_overlap():
    ids = np.arange(100, 140)
    first = make_ray_splits(ids, seed=17)
    second = make_ray_splits(ids, seed=17)

    assert first == second
    train = set(first["train"])
    validation = set(first["validation"])
    test = set(first["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(ids.tolist())
    assert first["counts"]["total"] == len(ids)


def test_smallest_supported_dataset_populates_all_splits():
    result = make_ray_splits([4, 9, 20], seed=1)
    assert result["counts"] == {
        "train": 1, "validation": 1, "test": 1, "total": 3
    }


@pytest.mark.parametrize("ids", [[], [1, 1, 2], [1, 2], [1.5, 2.5, 3.5]])
def test_invalid_case_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        make_ray_splits(ids)


@pytest.mark.parametrize(
    "train,validation",
    [(0.0, 0.2), (0.7, 0.3), (0.8, 0.3), (np.nan, 0.1)],
)
def test_invalid_split_fractions_are_rejected(train, validation):
    with pytest.raises(ValueError):
        make_ray_splits(range(10), train_fraction=train,
                        validation_fraction=validation)
