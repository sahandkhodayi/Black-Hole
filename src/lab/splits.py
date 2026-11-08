"""Deterministic train/validation/test splits at the whole-ray level.

A trajectory is one independent simulation case. Its individual phi samples
must stay together in one split; splitting points from the same ray across
train and test would leak almost identical neighboring states across splits.
"""

from collections.abc import Iterable

import numpy as np


def make_ray_splits(
    case_ids: Iterable[int],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Partition unique trajectory IDs and return a JSON-ready split manifest."""
    ids = np.asarray(list(case_ids))
    if ids.ndim != 1 or ids.size < 3:
        raise ValueError("At least three one-dimensional case IDs are required")
    if not np.issubdtype(ids.dtype, np.integer):
        # Reject fractional IDs instead of silently truncating them.
        try:
            numeric = ids.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("case_ids must be integers") from exc
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("case_ids must be finite integers")
        ids = numeric.astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)

    if len(np.unique(ids)) != len(ids):
        raise ValueError("case_ids must be unique")

    try:
        train_fraction = float(train_fraction)
        validation_fraction = float(validation_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("Split fractions must be numeric") from exc

    test_fraction = 1.0 - train_fraction - validation_fraction
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("Train, validation, and test fractions must all be positive")
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1")

    # Largest-remainder allocation gives integer counts close to the requested
    # proportions. For small datasets, guarantee at least one ray per split.
    raw_counts = fractions * len(ids)
    counts = np.floor(raw_counts).astype(int)
    remainders = raw_counts - counts
    for index in np.argsort(-remainders)[: len(ids) - int(counts.sum())]:
        counts[index] += 1

    for empty_index in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError("Not enough cases to populate all three splits")
        counts[donor] -= 1
        counts[empty_index] += 1

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_train, n_validation, n_test = (int(value) for value in counts)

    train_ids = shuffled[:n_train]
    validation_ids = shuffled[n_train:n_train + n_validation]
    test_ids = shuffled[n_train + n_validation:]

    # Defensive assertions: no trajectory is lost or assigned twice.
    combined = np.concatenate((train_ids, validation_ids, test_ids))
    if len(combined) != len(ids) or set(combined.tolist()) != set(ids.tolist()):
        raise RuntimeError("Internal split error: case IDs were lost or duplicated")

    return {
        "schema_version": 1,
        "seed": int(seed),
        "fractions": {
            "train": float(fractions[0]),
            "validation": float(fractions[1]),
            "test": float(fractions[2]),
        },
        "counts": {
            "train": n_train,
            "validation": n_validation,
            "test": n_test,
            "total": len(ids),
        },
        "train": train_ids.astype(int).tolist(),
        "validation": validation_ids.astype(int).tolist(),
        "test": test_ids.astype(int).tolist(),
        "split_unit": "complete trajectory case_id; never individual phi samples",
    }
