"""Shared data loading, scaling, and autoregressive rollout helpers.

The feature/action contract intentionally matches PhotonTrajectoryEnv:
    observation = [r0/12, phi/3, u/0.5, v/0.2]
    action[0] -> u_next = 0.25 * (action[0] + 1)
    action[1] -> v_next = 0.2 * action[1]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

R_SCALE = 12.0
PHI_SCALE = 3.0
U_SCALE = 0.5
V_SCALE = 0.2
OBS_DIM = 4
ACTION_DIM = 2


@dataclass(frozen=True)
class Ray:
    case_id: int
    r0: float
    phi: np.ndarray
    u: np.ndarray
    v: np.ndarray


def read_split_ids(splits_path: str | Path, split: str) -> list[int]:
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if split not in manifest:
        raise ValueError(f"Split {split!r} is not present in {path}")
    ids = [int(x) for x in manifest[split]]
    if not ids:
        raise ValueError(f"Split {split!r} is empty in {path}")
    return ids


def load_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trajectory dataset not found: {path}\n"
            "Generate it with the project's data-generation command first."
        )
    required = {"case_id", "r0", "phi", "u", "v", "offsets"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in required}
    if len(data["offsets"]) != len(data["case_id"]) + 1:
        raise ValueError("Dataset offsets must have len(case_id) + 1 elements")
    if data["offsets"][0] != 0 or data["offsets"][-1] != len(data["phi"]):
        raise ValueError("Dataset offsets do not cover the packed trajectory arrays")
    if not (len(data["phi"]) == len(data["u"]) == len(data["v"])):
        raise ValueError("Packed phi, u, and v arrays must have equal lengths")
    if not all(np.all(np.isfinite(data[k])) for k in ("r0", "phi", "u", "v")):
        raise ValueError("Dataset contains non-finite trajectory values")
    return data


def extract_ray(data: dict[str, np.ndarray], case_id: int) -> Ray:
    positions = np.flatnonzero(data["case_id"] == int(case_id))
    if positions.size != 1:
        raise KeyError(f"case_id {case_id} is not present exactly once in the dataset")
    pos = int(positions[0])
    start, end = map(int, data["offsets"][pos : pos + 2])
    if end - start < 2:
        raise ValueError(f"case_id {case_id} has fewer than two samples")
    return Ray(
        case_id=int(case_id),
        r0=float(data["r0"][pos]),
        phi=np.asarray(data["phi"][start:end], dtype=np.float64),
        u=np.asarray(data["u"][start:end], dtype=np.float64),
        v=np.asarray(data["v"][start:end], dtype=np.float64),
    )


def observation(r0: float, phi: float, u: float, v: float) -> np.ndarray:
    """Return the exact four-feature observation used by the PPO environment."""
    return np.clip(
        np.asarray(
            [r0 / R_SCALE, phi / PHI_SCALE, u / U_SCALE, v / V_SCALE],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


def action_to_state(action: Iterable[float]) -> tuple[float, float]:
    """Decode the action using the exact mapping in rl.environment."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("An action must be a finite vector with shape (2,)")
    values = np.clip(values, -1.0, 1.0)
    next_u = float((values[0] + 1.0) * 0.5 * U_SCALE)
    next_v = float(values[1] * V_SCALE)
    return next_u, next_v


def state_to_action(u: float, v: float) -> np.ndarray:
    """Encode a reference next state into the environment's two action values."""
    a0 = 2.0 * (float(u) / U_SCALE) - 1.0
    a1 = float(v) / V_SCALE
    return np.clip(np.asarray([a0, a1], dtype=np.float32), -1.0, 1.0)


def build_supervised_examples(
    data: dict[str, np.ndarray], case_ids: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-step training pairs, keeping complete rays in their split.

    Each input is the current reference state. Each target is the next state,
    encoded as the two continuous actions used by PhotonTrajectoryEnv.
    At evaluation, predictions are fed back autoregressively, matching PPO.
    """
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for case_id in case_ids:
        ray = extract_ray(data, int(case_id))
        obs = np.column_stack(
            (
                np.full(len(ray.phi) - 1, ray.r0 / R_SCALE),
                ray.phi[:-1] / PHI_SCALE,
                ray.u[:-1] / U_SCALE,
                ray.v[:-1] / V_SCALE,
            )
        )
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        targets = np.column_stack(
            (
                2.0 * (ray.u[1:] / U_SCALE) - 1.0,
                ray.v[1:] / V_SCALE,
            )
        )
        targets = np.clip(targets, -1.0, 1.0).astype(np.float32)
        feature_parts.append(obs)
        target_parts.append(targets)
    if not feature_parts:
        raise ValueError("Cannot build examples from an empty case-id list")
    return np.concatenate(feature_parts), np.concatenate(target_parts)


def autoregressive_rollout(
    ray: Ray,
    action_predictor: Callable[[np.ndarray], Iterable[float]],
) -> dict[str, np.ndarray]:
    """Roll a policy over a ray; its own output becomes its next input.

    The reference phi grid determines the evaluation sample locations, just as
    the current environment does. Reference future u/v values are not passed to
    the predictor. They are used only after the rollout for scoring.
    """
    count = len(ray.phi)
    pred_u = np.empty(count, dtype=np.float64)
    pred_v = np.empty(count, dtype=np.float64)
    pred_u[0] = float(ray.u[0])
    pred_v[0] = float(ray.v[0])
    for i in range(count - 1):
        obs = observation(ray.r0, float(ray.phi[i]), pred_u[i], pred_v[i])
        action = action_predictor(obs)
        pred_u[i + 1], pred_v[i + 1] = action_to_state(action)
    return {
        "phi": ray.phi.copy(),
        "reference_u": ray.u.copy(),
        "reference_v": ray.v.copy(),
        "predicted_u": pred_u,
        "predicted_v": pred_v,
    }


def calculate_metrics(ray: Ray, result: dict[str, np.ndarray]) -> dict[str, float | int]:
    du = result["predicted_u"] - ray.u
    dv = result["predicted_v"] - ray.v
    return {
        "case_id": ray.case_id,
        "r0": ray.r0,
        "samples": len(ray.phi),
        "rmse_u": float(np.sqrt(np.mean(du**2))),
        "rmse_v": float(np.sqrt(np.mean(dv**2))),
        "mae_u": float(np.mean(np.abs(du))),
        "mae_v": float(np.mean(np.abs(dv))),
        "normalized_rmse": float(
            np.sqrt(0.5 * np.mean((du / U_SCALE) ** 2 + (dv / V_SCALE) ** 2))
        ),
    }