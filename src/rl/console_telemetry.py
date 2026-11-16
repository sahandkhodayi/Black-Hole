"""Low-overhead JSONL telemetry bridge for the native Rust research console.

The callback leaves the physics environment, PPO model, reward, and optimizer
untouched. It only samples completed training rollouts and writes compact
telemetry records for the Rust frontend.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ConsoleTelemetryCallback(BaseCallback):
    """Stream training metrics and validation trajectories to a JSONL file."""

    def __init__(self, run_dir, eval_env, validation_ids, *, eval_freq=4096,
                 control_poll_freq=64, max_plot_points=320, verbose=0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.eval_env = eval_env
        self.validation_ids = [int(value) for value in validation_ids]
        self.eval_freq = max(1, int(eval_freq))
        self.control_poll_freq = max(1, int(control_poll_freq))
        self.max_plot_points = max(32, int(max_plot_points))
        self.events_path = self.run_dir / "live_events.jsonl"
        self.control_path = self.run_dir / "console_control.json"
        self.started = None
        self.last_eval_step = 0
        self.last_control_poll = -self.control_poll_freq
        self.stop_requested = False
        self.paused = False
        self._events_handle = None
        self.latest_episode_reward = None
        self.action_samples = deque(maxlen=2048)

    def _emit(self, event):
        event.setdefault("timesteps", int(self.num_timesteps))
        event.setdefault("elapsed_seconds", round(time.monotonic() - self.started, 3))
        self._events_handle.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        self._events_handle.flush()

    @staticmethod
    def _sample(values, limit):
        values = np.asarray(values, dtype=float)
        if len(values) <= limit:
            return values.tolist()
        indices = np.linspace(0, len(values) - 1, limit, dtype=int)
        return values[indices].tolist()

    def _validation_payload(self):
        env = self.eval_env.unwrapped
        errors_u, errors_v = [], []
        first_ray = None
        for case_id in self.validation_ids:
            obs, _ = env.reset(options={"case_id": case_id})
            ref_u = [float(env.target_u[0])]
            pred_u = [float(env.u)]
            ref_v = [float(env.target_v[0])]
            pred_v = [float(env.v)]
            angles = [float(env.phi[0])]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ref_u.append(float(info["reference_u"]))
                pred_u.append(float(info["predicted_u"]))
                ref_v.append(float(info["reference_v"]))
                pred_v.append(float(info["predicted_v"]))
                angles.append(float(info["phi"]))
            errors_u.append(float(np.sqrt(np.mean((np.asarray(pred_u) - ref_u) ** 2))))
            errors_v.append(float(np.sqrt(np.mean((np.asarray(pred_v) - ref_v) ** 2))))
            if first_ray is None:
                first_ray = {
                    "case_id": int(case_id),
                    "r0": float(env.r0),
                    "phi": self._sample(angles, self.max_plot_points),
                    "reference_u": self._sample(ref_u, self.max_plot_points),
                    "predicted_u": self._sample(pred_u, self.max_plot_points),
                    "reference_v": self._sample(ref_v, self.max_plot_points),
                    "predicted_v": self._sample(pred_v, self.max_plot_points),
                }
        return float(np.mean(errors_u)), float(np.mean(errors_v)), first_ray

    def _read_control(self):
        try:
            state = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        self.paused = bool(state.get("paused", False))
        if bool(state.get("stop", False)):
            self.stop_requested = True

    def _on_training_start(self):
        self.started = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Start a fresh event stream for each run; retain the rest of the run
        # directory and all existing checkpoint semantics.
        self._events_handle = self.events_path.open("w", encoding="utf-8", buffering=1)
        self._emit({"type": "status", "status": "running"})

    def _action_histogram(self, column):
        if not self.action_samples:
            return []
        values = np.asarray(self.action_samples, dtype=float)[:, column]
        counts, edges = np.histogram(values, bins=24, range=(-1.0, 1.0))
        centers = (edges[:-1] + edges[1:]) * 0.5
        return [[float(x), int(y)] for x, y in zip(centers, counts)]

    def _on_step(self):
        actions = self.locals.get("actions")
        if actions is not None:
            self.action_samples.extend(
                row.copy() for row in np.asarray(actions).reshape(-1, 2)
            )
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode and "r" in episode:
                self.latest_episode_reward = float(episode["r"])
        if self.num_timesteps - self.last_control_poll >= self.control_poll_freq:
            self.last_control_poll = self.num_timesteps
            self._read_control()
        while self.paused and not self.stop_requested:
            time.sleep(0.1)
            self._read_control()
        return not self.stop_requested

    def _on_rollout_end(self):
        values = getattr(self.model.logger, "name_to_value", {})
        policy_loss = values.get("train/policy_gradient_loss")
        value_loss = values.get("train/value_loss")
        event = {
            "type": "metrics",
            "episode_reward": self.latest_episode_reward,
            "policy_loss": float(policy_loss) if policy_loss is not None and np.isfinite(policy_loss) else None,
            "value_loss": float(value_loss) if value_loss is not None and np.isfinite(value_loss) else None,
            "action_hist_u": self._action_histogram(0),
            "action_hist_v": self._action_histogram(1),
        }
        self.latest_episode_reward = None
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            rmse_u, rmse_v, ray = self._validation_payload()
            event.update({
                "validation_rmse_u": rmse_u,
                "validation_rmse_v": rmse_v,
                "ray": ray,
            })
            self.last_eval_step = self.num_timesteps
        self._emit(event)

    def _on_training_end(self):
        try:
            self._emit({
                "type": "status",
                "status": "stopped" if self.stop_requested else "finished",
                "stop_requested": self.stop_requested,
            })
        finally:
            if self._events_handle is not None:
                self._events_handle.close()
                self._events_handle = None
