"""Local process manager for browser-launched PPO training runs."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bh_analytics.core import read_csv


class TrainingManager:
    """Start, monitor, and safely stop one local PPO training process."""

    poll_interval_seconds = 60

    def __init__(self, outputs_root: str | Path, project_root: str | Path | None = None):
        self.outputs_root = Path(outputs_root).expanduser().resolve()
        self.project_root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._run_name: str | None = None
        self._run_dir: Path | None = None
        self._stop_file: Path | None = None
        self._started_at: float | None = None
        self._created_at: str | None = None

    @staticmethod
    def normalize_request(payload: Any) -> tuple[int, int]:
        """Validate the small, public set of options accepted by the web UI."""
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")

        minutes = payload.get("minutes", 5)
        seed = payload.get("seed", 42)
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
            raise ValueError("Training length must be a whole number from 1 to 120 minutes.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**31) <= seed < 2**31:
            raise ValueError("Seed must be a whole number between -2147483648 and 2147483647.")
        return minutes * 60, seed

    def _check_inputs(self) -> None:
        required = (
            self.project_root / "data" / "photon_trajectories_v2.npz",
            self.project_root / "data" / "split_manifest_v2.json",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            names = ", ".join(path.name for path in missing)
            raise RuntimeError(
                f"Training data is missing ({names}). Run the dataset and split setup first."
            )

        missing_packages = []
        for package in ("stable_baselines3", "gymnasium", "torch"):
            try:
                available = importlib.util.find_spec(package) is not None
            except (ImportError, ValueError):
                available = False
            if not available:
                missing_packages.append(package)
        if missing_packages:
            raise RuntimeError(
                'PPO training dependencies are missing. Install them with '
                '`uv pip install -e ".[rl]"` or `python -m pip install -e ".[rl]"`.'
            )

    def start(self, payload: Any) -> dict[str, Any]:
        seconds, seed = self.normalize_request(payload)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A training run is already active. Stop it before starting another.")

            self._close_log_if_finished()
            self._check_inputs()
            self.outputs_root.mkdir(parents=True, exist_ok=True)
            self._run_name = "web_ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._run_dir = self.outputs_root / "rl_runs" / self._run_name
            self._run_dir.mkdir(parents=True, exist_ok=False)
            self._stop_file = self._run_dir / "STOP"
            log_path = self._run_dir / "training.log"

            command = [
                sys.executable, "-m", "rl.train",
                "--seconds", str(seconds),
                "--seed", str(seed),
                "--run-name", self._run_name,
                "--output-root", str(self.outputs_root),
                "--live-interval-seconds", str(self.poll_interval_seconds),
                "--web-monitor",
                "--control-file", str(self._stop_file),
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")

            try:
                self._log_handle = log_path.open("w", encoding="utf-8")
                self._process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                if self._log_handle is not None:
                    self._log_handle.close()
                    self._log_handle = None
                raise RuntimeError(f"Could not start the training process: {exc}") from exc

            self._started_at = time.monotonic()
            self._created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return self._snapshot_locked(status_override="starting")

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return self._snapshot_locked()
            assert self._stop_file is not None
            self._stop_file.write_text("stop requested by the dashboard\n", encoding="utf-8")
            return self._snapshot_locked(status_override="stopping")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _close_log_if_finished(self) -> int | None:
        if self._process is None:
            return None
        return_code = self._process.poll()
        if return_code is not None and self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
        return return_code

    def _snapshot_locked(self, status_override: str | None = None) -> dict[str, Any]:
        if self._process is None or self._run_dir is None or self._run_name is None:
            return {
                "status": "idle",
                "active": False,
                "poll_interval_seconds": self.poll_interval_seconds,
                "history": [],
                "latest": None,
            }

        return_code = self._close_log_if_finished()
        if status_override:
            state = status_override
        elif return_code is None:
            state = "stopping" if self._stop_file and self._stop_file.exists() else "running"
        else:
            state = "completed" if return_code == 0 else "failed"

        history = read_csv(self._run_dir / "live_history.csv")[-240:]
        config_path = self._run_dir / "config.json"
        config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                import json
                value = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    config = value
            except (OSError, UnicodeError, ValueError):
                pass

        elapsed = time.monotonic() - self._started_at if self._started_at is not None else 0.0
        visual = None
        visual_path = self._run_dir / "live_visual.json"
        if visual_path.is_file():
            try:
                loaded_visual = json.loads(visual_path.read_text(encoding="utf-8"))
                if isinstance(loaded_visual, dict):
                    visual = loaded_visual
            except (OSError, UnicodeError, ValueError):
                visual = None

        result: dict[str, Any] = {
            "status": state,
            "active": return_code is None,
            "run_name": self._run_name,
            "experiment_id": f"rl/{self._run_name}",
            "created_at": self._created_at,
            "elapsed_seconds": round(elapsed, 1),
            "requested_seconds": config.get("seconds_requested"),
            "timesteps": config.get("timesteps_actual"),
            "return_code": return_code,
            "poll_interval_seconds": self.poll_interval_seconds,
            "history": history,
            "latest": history[-1] if history else None,
            "visual": visual,
        }
        if state == "failed":
            log_path = self._run_dir / "training.log"
            try:
                result["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            except OSError:
                result["log_tail"] = []
        return result
