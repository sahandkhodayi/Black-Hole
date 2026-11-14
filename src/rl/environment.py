"""Teacher-guided sequential photon trajectory environment."""
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PhotonTrajectoryEnv(gym.Env):
    """Predict the next (u, v) state; reference targets shape reward only."""
    metadata = {"render_modes": []}

    def __init__(self, dataset_path, case_ids, *, randomize_cases=True):
        super().__init__()
        with np.load(Path(dataset_path), allow_pickle=False) as archive:
            required = {"case_id", "r0", "phi", "u", "v", "offsets"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Dataset missing arrays: {sorted(missing)}")
            self.data = {key: np.asarray(archive[key]) for key in required}
        ids = np.asarray(case_ids, dtype=np.int64)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("case_ids must be a non-empty 1D list")
        self.positions = {int(cid): i for i, cid in enumerate(self.data["case_id"])}
        absent = [int(cid) for cid in ids if int(cid) not in self.positions]
        if absent:
            raise ValueError(f"Case IDs not found in dataset: {absent}")
        self.case_ids = ids
        self.randomize_cases = bool(randomize_cases)
        self._next_case = 0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.u_scale, self.v_scale = 0.5, 0.2
        self.r_scale, self.phi_scale = 12.0, 3.0
        self._done = True

    def _load_ray(self, case_id):
        pos = self.positions[int(case_id)]
        start, end = self.data["offsets"][pos:pos + 2]
        return (float(self.data["r0"][pos]),
                self.data["phi"][start:end].astype(float, copy=True),
                self.data["u"][start:end].astype(float, copy=True),
                self.data["v"][start:end].astype(float, copy=True))

    def _obs(self):
        obs = np.asarray([self.r0 / self.r_scale,
                          self.phi[self.index] / self.phi_scale,
                          self.u / self.u_scale,
                          self.v / self.v_scale], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "case_id" in options:
            case_id = int(options["case_id"])
            if case_id not in self.positions or case_id not in self.case_ids:
                raise ValueError(f"case_id {case_id} is outside this split")
        elif self.randomize_cases:
            case_id = int(self.np_random.choice(self.case_ids))
        else:
            case_id = int(self.case_ids[self._next_case % len(self.case_ids)])
            self._next_case += 1
        self.case_id = case_id
        self.r0, self.phi, self.target_u, self.target_v = self._load_ray(case_id)
        self.index = 0
        self.u, self.v = float(self.target_u[0]), float(self.target_v[0])
        self._done = False
        return self._obs(), {"case_id": case_id, "r0": self.r0,
                             "reference_samples": len(self.phi)}

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode finished; call reset() before step().")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        next_index = min(self.index + 1, len(self.phi) - 1)
        proposed_u = float((action[0] + 1.0) * 0.5 * self.u_scale)
        proposed_v = float(action[1] * self.v_scale)
        target_u, target_v = float(self.target_u[next_index]), float(self.target_v[next_index])
        eu = (proposed_u - target_u) / self.u_scale
        ev = (proposed_v - target_v) / self.v_scale

        # A bounded, non-negative reward avoids the large negative episode
        # returns produced by the previous quadratic reward after policy drift.
        # It remains smooth and strictly prefers smaller normalized state error.
        squared_error = eu * eu + ev * ev
        reward = float(1.0 / (1.0 + squared_error))

        self.u, self.v, self.index = proposed_u, proposed_v, next_index
        terminated = self.index == len(self.phi) - 1
        self._done = bool(terminated)
        info = {"case_id": self.case_id, "phi": float(self.phi[self.index]),
                "reference_u": target_u, "reference_v": target_v,
                "predicted_u": proposed_u, "predicted_v": proposed_v,
                "u_abs_error": abs(proposed_u - target_u),
                "v_abs_error": abs(proposed_v - target_v),
                "normalized_squared_error": squared_error}
        return self._obs(), reward, terminated, False, info
