"""Teacher-guided environment for learning photon ODE step-size selection.

Unlike PhotonTrajectoryEnv, the action is a step size, not a predicted state.
The environment advances the geodesic with RK4 and uses the DOP853 trajectory
only to calculate reward. This is an adaptive-integration research baseline.
"""
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def geodesic_rhs(u: float, v: float):
    return v, 1.5 * u * u - u


def rk4_step(u: float, v: float, h: float):
    k1u, k1v = geodesic_rhs(u, v)
    k2u, k2v = geodesic_rhs(u + 0.5 * h * k1u, v + 0.5 * h * k1v)
    k3u, k3v = geodesic_rhs(u + 0.5 * h * k2u, v + 0.5 * h * k2v)
    k4u, k4v = geodesic_rhs(u + h * k3u, v + h * k3v)
    next_u = u + h * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
    next_v = v + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    return float(next_u), float(next_v)


class AdaptivePhotonEnv(gym.Env):
    """PPO chooses h_phi in [min_step, max_step] at every integration step."""
    metadata = {"render_modes": []}

    def __init__(self, dataset_path, case_ids, *, min_step=0.002, max_step=0.04,
                 error_tolerance=2e-3, randomize_cases=True):
        super().__init__()
        if not 0 < min_step < max_step:
            raise ValueError("Require 0 < min_step < max_step")
        with np.load(Path(dataset_path), allow_pickle=False) as archive:
            required = {"case_id", "r0", "phi", "u", "v", "offsets"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Dataset missing arrays: {sorted(missing)}")
            self.data = {key: np.asarray(archive[key]) for key in required}
        self.positions = {int(cid): i for i, cid in enumerate(self.data["case_id"])}
        self.case_ids = np.asarray(case_ids, dtype=np.int64)
        if self.case_ids.ndim != 1 or not self.case_ids.size:
            raise ValueError("case_ids must be a non-empty one-dimensional list")
        absent = [int(cid) for cid in self.case_ids if int(cid) not in self.positions]
        if absent:
            raise ValueError(f"Unknown case IDs: {absent}")
        self.min_step, self.max_step = float(min_step), float(max_step)
        self.error_tolerance = float(error_tolerance)
        self.randomize_cases = bool(randomize_cases)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self._cursor = 0
        self._done = True

    def _load(self, case_id):
        pos = self.positions[int(case_id)]
        start, end = map(int, self.data["offsets"][pos:pos + 2])
        return (float(self.data["r0"][pos]),
                self.data["phi"][start:end].astype(np.float64, copy=True),
                self.data["u"][start:end].astype(np.float64, copy=True),
                self.data["v"][start:end].astype(np.float64, copy=True))

    def _reference(self, phi):
        return (float(np.interp(phi, self.phi, self.ref_u)),
                float(np.interp(phi, self.phi, self.ref_v)))

    def _obs(self):
        return np.clip(np.asarray([
            self.r0 / 12.0, self.phi_now / max(float(self.phi[-1]), 1e-8),
            self.u / 0.5, self.v / 0.2,
            self.previous_step / self.max_step,
        ], dtype=np.float32), -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "case_id" in options:
            case_id = int(options["case_id"])
            if case_id not in self.positions or case_id not in self.case_ids:
                raise ValueError("case_id is outside this environment split")
        elif self.randomize_cases:
            case_id = int(self.np_random.choice(self.case_ids))
        else:
            case_id = int(self.case_ids[self._cursor % len(self.case_ids)])
            self._cursor += 1
        self.case_id = case_id
        self.r0, self.phi, self.ref_u, self.ref_v = self._load(case_id)
        self.phi_now = float(self.phi[0])
        self.u, self.v = float(self.ref_u[0]), float(self.ref_v[0])
        self.previous_step = self.min_step
        self.steps = 0
        self._done = False
        return self._obs(), {"case_id": case_id, "r0": self.r0, "phi_end": float(self.phi[-1])}

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode finished; call reset() before step().")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (1,)")
        normalized = float(np.clip(action[0], -1.0, 1.0))
        step_size = self.min_step + 0.5 * (normalized + 1.0) * (self.max_step - self.min_step)
        step_size = min(step_size, float(self.phi[-1]) - self.phi_now)
        next_u, next_v = rk4_step(self.u, self.v, step_size)
        next_phi = self.phi_now + step_size
        target_u, target_v = self._reference(next_phi)
        error = ((next_u - target_u) / 0.5) ** 2 + ((next_v - target_v) / 0.2) ** 2
        # Penalize local error and excessive step count. Reward is bounded above
        # and finite; the policy must trade accuracy against integration work.
        reward = -float(error) - 0.015 * (self.min_step / max(step_size, self.min_step))
        self.u, self.v, self.phi_now = next_u, next_v, next_phi
        self.previous_step = step_size
        self.steps += 1
        terminated = self.phi_now >= float(self.phi[-1]) - 1e-10
        self._done = bool(terminated)
        info = {"case_id": self.case_id, "phi": self.phi_now,
                "step_size": step_size, "reference_u": target_u,
                "reference_v": target_v, "predicted_u": next_u,
                "predicted_v": next_v, "normalized_error": error,
                "steps": self.steps}
        return self._obs(), reward, terminated, False, info
