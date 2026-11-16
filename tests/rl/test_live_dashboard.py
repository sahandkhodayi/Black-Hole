from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
from rl.live_dashboard import LiveTrainingDashboard


def test_dashboard_pause_toggle_without_gui(tmp_path: Path):
    dashboard = LiveTrainingDashboard(tmp_path, None, [], enabled=False)
    assert dashboard.paused is False
    dashboard._toggle_pause()
    assert dashboard.paused is True
    dashboard._toggle_pause()
    assert dashboard.paused is False


def test_dashboard_stop_request(tmp_path: Path):
    dashboard = LiveTrainingDashboard(tmp_path, None, [], enabled=False)
    dashboard._request_stop()
    assert dashboard.stop_requested is True
