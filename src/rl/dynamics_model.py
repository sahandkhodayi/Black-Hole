"""Supervised neural surrogate for one-step Schwarzschild photon dynamics."""
from pathlib import Path

import numpy as np
import torch
from torch import nn


R_SCALE = 12.0
PHI_SCALE = 3.0
U_SCALE = 0.5
V_SCALE = 0.2


class PhotonDynamicsMLP(nn.Module):
    """Map the current ray state and angular step to the next (u, v)."""

    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(self, features):
        return self.network(features)


def load_packed_dataset(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"case_id", "r0", "phi", "u", "v", "offsets"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Dataset missing arrays: {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in required}

    n_cases = len(data["case_id"])
    if len(data["r0"]) != n_cases or len(data["offsets"]) != n_cases + 1:
        raise ValueError("Inconsistent packed trajectory metadata")
    if not (len(data["phi"]) == len(data["u"]) == len(data["v"])):
        raise ValueError("phi, u, and v arrays must have equal lengths")
    if int(data["offsets"][0]) != 0 or int(data["offsets"][-1]) != len(data["phi"]):
        raise ValueError("Invalid trajectory offsets")
    if np.any(np.diff(data["offsets"]) < 2):
        raise ValueError("Every trajectory must contain at least two samples")
    if not all(np.all(np.isfinite(data[key])) for key in ("r0", "phi", "u", "v")):
        raise ValueError("Dataset contains non-finite values")
    return data


def make_transition_arrays(data, case_ids):
    """Create leakage-safe one-step examples for the requested whole rays."""
    positions = {int(case_id): i for i, case_id in enumerate(data["case_id"])}
    absent = [int(case_id) for case_id in case_ids if int(case_id) not in positions]
    if absent:
        raise ValueError(f"Case IDs not found in dataset: {absent}")

    all_steps = []
    for start, end in zip(data["offsets"][:-1], data["offsets"][1:]):
        steps = np.diff(data["phi"][int(start):int(end)])
        if np.any(steps <= 0):
            raise ValueError("Each trajectory must have strictly increasing phi")
        all_steps.extend(steps.tolist())
    if not all_steps:
        raise ValueError("No trajectory transitions found")
    step_scale = float(max(all_steps))

    features, targets = [], []
    for case_id in case_ids:
        pos = positions[int(case_id)]
        start, end = map(int, data["offsets"][pos:pos + 2])
        phi = data["phi"][start:end]
        u = data["u"][start:end]
        v = data["v"][start:end]
        delta = np.diff(phi)
        x = np.column_stack((
            np.full(len(delta), float(data["r0"][pos]) / R_SCALE),
            phi[:-1] / PHI_SCALE,
            u[:-1] / U_SCALE,
            v[:-1] / V_SCALE,
            delta / step_scale,
        ))
        y = np.column_stack((u[1:] / U_SCALE, v[1:] / V_SCALE))
        features.append(x)
        targets.append(y)

    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
        step_scale,
    )


def predict_next(model, state, *, r0, phi, delta_phi, step_scale, device):
    """Predict one normalized transition; clamp only to physical output bounds."""
    x = np.asarray([[
        r0 / R_SCALE,
        phi / PHI_SCALE,
        state[0] / U_SCALE,
        state[1] / V_SCALE,
        delta_phi / step_scale,
    ]], dtype=np.float32)
    with torch.no_grad():
        y = model(torch.as_tensor(x, device=device)).cpu().numpy()[0]
    return np.asarray([
        np.clip(y[0], 0.0, 1.0) * U_SCALE,
        np.clip(y[1], -1.0, 1.0) * V_SCALE,
    ], dtype=float)


def save_checkpoint(path, model, *, step_scale, width, best_epoch):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "step_scale": float(step_scale),
        "width": int(width),
        "best_epoch": int(best_epoch),
        "input_schema": ["r0/12", "phi/3", "u/0.5", "v/0.2", "delta_phi/step_scale"],
        "target_schema": ["next_u/0.5", "next_v/0.2"],
    }, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = PhotonDynamicsMLP(width=int(payload["width"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, float(payload["step_scale"]), payload
