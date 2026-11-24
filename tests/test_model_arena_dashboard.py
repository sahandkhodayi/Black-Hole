import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from bh_analytics.dashboard import make_handler


def test_model_arena_routes_serve_saved_files_without_training(tmp_path):
    arena = tmp_path / "presentation" / "model_arena"
    arena.mkdir(parents=True)
    (arena / "summary.json").write_text(json.dumps({
        "split": "test", "test_case_ids": [3],
        "models": [{"model": "MLP", "mean_normalized_rmse": 0.01}],
    }), encoding="utf-8")
    (arena / "predictions.csv").write_text(
        "case_id,r0,phi,model,reference_u,reference_v,predicted_u,predicted_v,absolute_error_u,absolute_error_v\n"
        "3,5,0,DOP853,0.2,0,0.2,0,0,0\n", encoding="utf-8")
    (arena / "metrics_by_ray.csv").write_text(
        "model,case_id,normalized_rmse\nMLP,3,0.01\n", encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/arena") as response:
            assert response.status == 200
            assert b"MODEL ARENA" in response.read()
        with urlopen(base + "/assets/model-arena.js") as response:
            assert response.status == 200
            assert b"saved results only" in response.read()
        with urlopen(base + "/api/model-arena") as response:
            payload = json.loads(response.read())
        assert payload["summary"]["test_case_ids"] == [3]
        assert payload["predictions"][0]["model"] == "DOP853"
        assert payload["metrics"][0]["model"] == "MLP"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_observatory_home_and_assets_are_served_without_training(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            page = response.read()
            assert response.status == 200
            assert b"PHOTON DYNAMICS OBSERVATORY" in page
            assert b"LIVE TRAINING" not in page
            assert b"Your experiments" not in page
        with urlopen(base + "/assets/observatory.css") as response:
            assert response.status == 200
            assert b".space-scene" in response.read()
        with urlopen(base + "/assets/observatory.js") as response:
            assert response.status == 200
            assert b"/api/physics/simulate" in response.read()
        try:
            urlopen(base + "/api/training/status")
            raise AssertionError("Browser training should be disabled in presentation mode.")
        except Exception as exc:
            assert getattr(exc, "code", None) == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
