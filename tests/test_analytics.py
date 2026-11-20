import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from bh_analytics.core import discover_experiments, get_experiment, read_csv, resolve_artifact
from bh_analytics.dashboard import make_handler
from bh_analytics.report import generate_report, render_report


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    mlp = root / "mlp_runs/demo_mlp"
    put(mlp / "config.json", json.dumps({"epochs_completed": 2, "best_epoch": 2}))
    put(mlp / "history.csv", "epoch,train_mse,validation_mse,validation_rollout_normalized_rmse\n1,.3,.4,.5\n2,.2,.25,.3\n")
    put(mlp / "best_model.pt", "checkpoint")
    rl = root / "rl_runs/demo_ppo"
    put(rl / "config.json", json.dumps({"algorithm": "PPO", "timesteps_actual": 1200}))
    put(rl / "evaluation_test/summary.json", json.dumps({"num_rays": 2, "mean_rmse_u": .1}))
    put(rl / "evaluation_test/metrics.csv", "case_id,rmse_u\n1,.1\n2,.11\n")
    put(rl / "final_model.zip", "checkpoint")
    pair = root / "presentation/paired/comparison"
    put(pair / "summary.json", json.dumps({"num_test_rays": 1, "mlp": {"mean_rmse_u": .01}, "ppo": {"mean_rmse_u": .1}}))
    put(pair / "metrics_by_ray.csv", "model,case_id,rmse_u\nMLP,1,.01\nPPO,1,.1\n")
    put(pair / "analysis.md", "Measured test comparison")
    return root


def test_discovers_and_normalizes_all_existing_run_kinds(outputs):
    runs = {x["id"]: x for x in discover_experiments(outputs)}
    assert set(runs) == {"mlp/demo_mlp", "rl/demo_ppo", "presentation/paired"}
    assert runs["mlp/demo_mlp"]["summary"]["best_validation_rollout_normalized_rmse"] == .3
    assert runs["rl/demo_ppo"]["summary"]["num_rays"] == 2  # test split is preferred
    assert runs["presentation/paired"]["comparison"]["mlp"]["mean_rmse_u"] == .01
    assert "Measured test" in runs["presentation/paired"]["analysis_markdown"]
    assert get_experiment(outputs, "missing") is None


def test_csv_coercion_and_path_traversal_protection(outputs, tmp_path):
    path = tmp_path / "data.csv"
    put(path, "id,score,flag,missing\n1,.25,true,NaN\n")
    assert read_csv(path) == [{"id": 1, "score": .25, "flag": True, "missing": None}]
    put(tmp_path / "secret.csv", "secret")
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact(outputs, "../secret.csv")


def test_standalone_report_contains_saved_analysis_and_rejects_unknown_id(outputs):
    item = get_experiment(outputs, "presentation/paired")
    assert "Measured test comparison" in render_report(item, outputs)
    report = generate_report(outputs, "presentation/paired")
    assert report.is_file() and report.parent == outputs / "analytics/reports"
    with pytest.raises(KeyError, match="Unknown experiment"):
        generate_report(outputs, "../unknown")


def test_dashboard_http_api_report_assets_and_bad_paths(outputs):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(outputs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert b"Experiment analytics" in response.read()
            assert response.headers["Content-Security-Policy"]
        with urlopen(base + "/api/experiments") as response:
            assert len(json.loads(response.read())["experiments"]) == 3
        with urlopen(base + "/api/experiment?id=mlp%2Fdemo_mlp") as response:
            assert json.loads(response.read())["kind"] == "mlp"
        with urlopen(base + "/assets/dashboard-ui.js") as response:
            assert b"renderEvaluation" in response.read()
        request = Request(base + "/api/report", data=b'{"id":"presentation/paired"}',
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request) as response:
            report_url = json.loads(response.read())["url"]
            assert response.status == 201
        with urlopen(base + report_url) as response:
            assert b"Measured test comparison" in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/artifact?path=..%2Fsecret.csv")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/report", data=b"[]", method="POST"))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
