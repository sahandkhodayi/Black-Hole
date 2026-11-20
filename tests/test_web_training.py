from __future__ import annotations

import csv
import json

import pytest

from bh_analytics.training import TrainingManager


def test_normalize_request_uses_minutes_and_seed():
    assert TrainingManager.normalize_request({"minutes": 5, "seed": 7}) == (300, 7)
    assert TrainingManager.normalize_request({}) == (300, 42)


@pytest.mark.parametrize(
    "payload",
    [
        {"minutes": 0},
        {"minutes": 121},
        {"minutes": 1.5},
        {"minutes": True},
        {"seed": 1.5},
        {"seed": 2**31},
        [],
    ],
)
def test_normalize_request_rejects_invalid_options(payload):
    with pytest.raises(ValueError):
        TrainingManager.normalize_request(payload)


def test_status_is_idle_before_a_training_run(tmp_path):
    manager = TrainingManager(tmp_path / "outputs", project_root=tmp_path)
    status = manager.status()
    assert status["status"] == "idle"
    assert status["active"] is False
    assert status["history"] == []


def test_start_stop_and_status_use_a_safe_control_file(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "data" / "photon_trajectories_v2.npz").write_bytes(b"dataset placeholder")
    (project / "data" / "split_manifest_v2.json").write_text(
        '{"train": [1], "validation": [2]}', encoding="utf-8"
    )

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = None

        def poll(self):
            return self.returncode

    launched = []

    def fake_popen(command, **kwargs):
        process = FakeProcess(command, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr("bh_analytics.training.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("bh_analytics.training.subprocess.Popen", fake_popen)

    manager = TrainingManager(project / "outputs", project_root=project)
    started = manager.start({"minutes": 1, "seed": 9})

    assert started["status"] == "starting"
    assert started["active"] is True
    assert "--web-monitor" in launched[0].command
    assert "--output-root" in launched[0].command
    assert str(project / "outputs") in launched[0].command
    assert "--live-interval-seconds" in launched[0].command
    assert "60" in launched[0].command

    run_dir = project / "outputs" / "rl_runs" / started["run_name"]
    history_path = run_dir / "live_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timesteps", "elapsed_seconds", "reward_mean", "validation_rmse_u"],
        )
        writer.writeheader()
        writer.writerow({
            "timesteps": 512,
            "elapsed_seconds": 61.0,
            "reward_mean": 0.7,
            "validation_rmse_u": 0.03,
        })

    stopping = manager.stop()
    assert stopping["status"] == "stopping"
    assert (run_dir / "STOP").is_file()

    visual_payload = {
        "case_id": 22,
        "closest_approach": 5.0,
        "phi": [-1.0, 0.0, 1.0],
        "reference_u": [0.1, 0.2, 0.1],
        "predicted_u": [0.12, 0.2, 0.13],
    }
    (run_dir / "live_visual.json").write_text(
        json.dumps(visual_payload), encoding="utf-8"
    )

    running_snapshot = manager.status()
    assert running_snapshot["latest"]["timesteps"] == 512
    assert running_snapshot["latest"]["reward_mean"] == 0.7
    assert running_snapshot["visual"] == visual_payload

    launched[0].returncode = 0
    finished = manager.status()
    assert finished["status"] == "completed"
    assert finished["active"] is False
    assert finished["return_code"] == 0
