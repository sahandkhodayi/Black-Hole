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


def test_orbit_coordinates_use_inverse_radius(tmp_path: Path):
    phi = __import__("numpy").array([0.0, __import__("numpy").pi / 2])
    u = __import__("numpy").array([0.2, 0.2])
    x, y = LiveTrainingDashboard._orbit_xy(phi, u)
    assert x[0] == pytest.approx(5.0)
    assert y[0] == pytest.approx(0.0, abs=1e-7)
    assert x[1] == pytest.approx(0.0, abs=1e-7)
    assert y[1] == pytest.approx(5.0)
