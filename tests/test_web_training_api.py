import pytest
from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bh_analytics.dashboard import make_handler


class FakeTrainingManager:
    def __init__(self, _outputs_root):
        self.started_with = None

    def status(self):
        return {"status": "idle", "active": False, "history": []}

    def start(self, payload):
        self.started_with = payload
        return {"status": "starting", "active": True, "run_name": "fake_run"}

    def stop(self):
        return {"status": "stopping", "active": True}


def _serve(monkeypatch, tmp_path, *, allow_training=True):
    manager = FakeTrainingManager(tmp_path)
    monkeypatch.setattr("bh_analytics.dashboard.TrainingManager", lambda _root: manager)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(tmp_path, allow_training=allow_training),
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, manager, f"http://127.0.0.1:{server.server_port}"


def _json_get(url):
    with urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def _json_post(url, payload=None, *, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    data = json.dumps(payload).encode("utf-8") if payload is not None else b"{}"
    request = Request(url, data=data, headers=headers, method="POST")
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())



def test_physics_ray_api_returns_actual_solver_trajectory(monkeypatch, tmp_path):
    server, thread, _manager, base = _serve(monkeypatch, tmp_path)
    try:
        status, body = _json_get(base + "/api/physics/ray?r0=5.0")
        assert status == 200
        assert body["escaped"] is True
        assert body["closest_approach"] == 5.0
        assert body["samples"] > 100
        assert len(body["x"]) == len(body["y"]) == body["samples"]
        middle = min(range(len(body["phi"])), key=lambda i: abs(body["phi"][i]))
        assert body["u"][middle] == pytest.approx(0.2, abs=1e-8)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_physics_ray_api_rejects_captured_or_out_of_range_paths(monkeypatch, tmp_path):
    server, thread, _manager, base = _serve(monkeypatch, tmp_path)
    try:
        for radius in ("1.5", "12.1", "nan", "not-a-number"):
            try:
                _json_get(base + "/api/physics/ray?r0=" + radius)
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError(f"invalid closest approach should be rejected: {radius}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

def test_training_api_exposes_status_start_and_stop(monkeypatch, tmp_path):
    server, thread, manager, base = _serve(monkeypatch, tmp_path)
    try:
        status, body = _json_get(base + "/api/training/status")
        assert status == 200
        assert body["status"] == "idle"

        status, body = _json_post(base + "/api/training/start", {"minutes": 5, "seed": 12})
        assert status == 202
        assert body["status"] == "starting"
        assert manager.started_with == {"minutes": 5, "seed": 12}

        status, body = _json_post(base + "/api/training/stop")
        assert status == 202
        assert body["status"] == "stopping"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_training_api_rejects_cross_origin_and_non_local_training(monkeypatch, tmp_path):
    server, thread, _manager, base = _serve(monkeypatch, tmp_path)
    try:
        try:
            _json_post(
                base + "/api/training/start",
                {"minutes": 5, "seed": 12},
                origin="https://untrusted.example",
            )
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("cross-origin training request should be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    server, thread, _manager, base = _serve(monkeypatch, tmp_path, allow_training=False)
    try:
        try:
            _json_get(base + "/api/training/status")
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("training should be disabled for non-local bindings")

        try:
            _json_post(base + "/api/training/start", {"minutes": 5, "seed": 12})
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("training should be disabled for non-local bindings")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
