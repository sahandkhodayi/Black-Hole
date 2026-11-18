"""JSONL telemetry and process controls for the standalone Qt dashboard.

The callback runs in the training process. It never imports Qt or Matplotlib,
so the GUI event loop remains independent from PPO optimization.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DashboardTelemetryCallback(BaseCallback):
    """Publish compact training snapshots and accept pause/resume/stop commands."""

    def __init__(self, telemetry_path, control_path, eval_env, validation_ids,
                 *, interval_seconds=60.0, verbose=0):
        super().__init__(verbose)
        self.telemetry_path = Path(telemetry_path)
        self.control_path = Path(control_path)
        self.eval_env = eval_env
        self.validation_ids = [int(value) for value in validation_ids]
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._last_report = 0.0
        self._last_control_poll = 0.0
        self._paused = False
        self._stopped = False

    def _emit(self, event, **payload):
        record = {"event": event, "time": time.time(),
                  "timesteps": int(self.num_timesteps), **payload}
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()

    def _read_control(self):
        try:
            command = self.control_path.read_text(encoding="utf-8").strip().lower()
        except FileNotFoundError:
            return
        if command == "pause":
            self._paused = True
        elif command == "resume":
            self._paused = False
        elif command == "stop":
            self._stopped = True

    @staticmethod
    def _downsample(values, limit=500):
        values = np.asarray(values, dtype=float)
        if len(values) <= limit:
            return values.tolist()
        indices = np.linspace(0, len(values) - 1, limit, dtype=int)
        return values[indices].tolist()

    def _evaluate(self):
        env = self.eval_env.unwrapped
        errors_u, errors_v = [], []
        first_ray = None
        for case_id in self.validation_ids:
            obs, _ = env.reset(options={"case_id": case_id})
            ref_u = [float(env.target_u[0])]
            pred_u = [float(env.u)]
            ref_v = [float(env.target_v[0])]
            pred_v = [float(env.v)]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ref_u.append(float(info["reference_u"]))
                pred_u.append(float(info["predicted_u"]))
                ref_v.append(float(info["reference_v"]))
                pred_v.append(float(info["predicted_v"]))
            errors_u.append(float(np.sqrt(np.mean((np.asarray(pred_u) - ref_u) ** 2))))
            errors_v.append(float(np.sqrt(np.mean((np.asarray(pred_v) - ref_v) ** 2))))
            if first_ray is None:
                phi = np.asarray(env.phi[:len(ref_u)], dtype=float)
                first_ray = {
                    "case_id": int(case_id),
                    "r0": float(env.r0),
                    "phi": self._downsample(phi),
                    "reference_u": self._downsample(ref_u),
                    "predicted_u": self._downsample(pred_u),
                    "reference_v": self._downsample(ref_v),
                    "predicted_v": self._downsample(pred_v),
                }
        return float(np.mean(errors_u)), float(np.mean(errors_v)), first_ray

    def _on_training_start(self):
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_path.write_text("", encoding="utf-8")
        self._last_report = time.monotonic() - self.interval_seconds
        self._emit("status", status="running")

    def _on_step(self):
        now = time.monotonic()
        if now - self._last_control_poll >= 0.25:
            self._read_control()
            self._last_control_poll = now
        if self._stopped:
            self._emit("status", status="stopping")
            return False
        while self._paused and not self._stopped:
            time.sleep(0.1)
            self._read_control()
        if self._stopped:
            self._emit("status", status="stopping")
            return False
        return True

    def _on_rollout_end(self):
        values = getattr(self.model.logger, "name_to_value", {})
        losses = {
            "policy_loss": self._finite(values.get("train/policy_gradient_loss")),
            "value_loss": self._finite(values.get("train/value_loss")),
            "entropy_loss": self._finite(values.get("train/entropy_loss")),
            "approx_kl": self._finite(values.get("train/approx_kl")),
            "explained_variance": self._finite(values.get("train/explained_variance")),
        }
        self._emit("metrics", **losses)
        now = time.monotonic()
        if now - self._last_report < self.interval_seconds:
            return
        self._last_report = now
        try:
            rmse_u, rmse_v, ray = self._evaluate()
            self._emit("validation", rmse_u=rmse_u, rmse_v=rmse_v, ray=ray)
        except Exception as exc:  # keep a failed diagnostic from killing PPO
            self._emit("warning", message=f"Validation snapshot failed: {exc}")

    @staticmethod
    def _finite(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    def _on_training_end(self):
        self._emit("status", status="finished")
