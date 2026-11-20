"""Tests for readable, partial-data-friendly Black Hole Lab results."""
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

    mlp = root / "mlp_runs" / "demo_mlp"
    put(mlp / "config.json", json.dumps({
        "epochs_completed": 2, "best_epoch": 2,
        "best_validation_rollout_normalized_rmse": 0.03,
    }))
    put(mlp / "history.csv", (
        "epoch,train_mse,validation_mse,validation_rollout_normalized_rmse\n"
        "1,.3,.4,.5\n2,.2,.25,.3\n"
    ))
    put(mlp / "best_model.pt", "checkpoint")

    rl = root / "rl_runs" / "demo_ppo"
    put(rl / "config.json", json.dumps({"algorithm": "PPO", "timesteps_actual": 1200}))
    put(rl / "evaluation_test" / "summary.json", json.dumps({
        "num_rays": 2, "mean_rmse_u": 0.1, "mean_rmse_v": 0.2,
    }))
    put(rl / "final_model.zip", "checkpoint")

    pair = root / "presentation" / "paired" / "comparison"
    put(pair / "summary.json", json.dumps({
        "num_test_rays": 2,
        "test_case_ids": [4, 27],
        "mlp": {"mean_normalized_rmse": 0.04, "mean_rmse_u": 0.01, "mean_rmse_v": 0.02},
        "ppo": {"mean_normalized_rmse": 0.28, "mean_rmse_u": 0.1, "mean_rmse_v": 0.2},
    }))
    put(pair / "metrics_by_ray.csv", (
        "model,case_id,normalized_rmse,rmse_u,rmse_v\n"
        "MLP,4,.02,.01,.02\nMLP,27,.06,.01,.02\n"
        "PPO,4,.2,.1,.2\nPPO,27,.36,.1,.2\n"
    ))
    put(pair / "analysis.md", "A measured comparison of both models.")
    return root


def test_discovery_opens_most_useful_comparison_first(outputs):
    runs = discover_experiments(outputs)
    assert runs[0]["id"] == "presentation/paired"
    assert {run["id"] for run in runs} == {
        "presentation/paired", "rl/demo_ppo", "mlp/demo_mlp"
    }
    assert runs[0]["summary"]["num_test_rays"] == 2
    assert runs[0]["comparison"]["mlp"]["mean_normalized_rmse"] == 0.04


def test_comparison_still_works_without_per_ray_csv(outputs):
    path = outputs / "presentation" / "paired" / "comparison" / "metrics_by_ray.csv"
    path.unlink()
    run = get_experiment(outputs, "presentation/paired")
    assert run is not None
    assert run["summary"]["num_test_rays"] == 2
    assert run["comparison"]["mlp"]["mean_normalized_rmse"] == 0.04
    report = render_report(run, outputs)
    assert "MLP" in report and "PPO" in report
    assert "per-ray metrics CSV" not in report


def test_comparison_can_rebuild_headline_from_per_ray_csv_when_summary_missing(outputs):
    path = outputs / "presentation" / "paired" / "comparison" / "summary.json"
    path.unlink()
    run = get_experiment(outputs, "presentation/paired")
    assert run is not None
    assert run["summary"]["num_test_rays"] == 2
    assert run["comparison"]["mlp"]["mean_normalized_rmse"] == pytest.approx(0.04)
    assert run["comparison"]["ppo"]["mean_normalized_rmse"] == pytest.approx(0.28)


def test_csv_is_optional_and_path_traversal_is_blocked(outputs, tmp_path):
    path = tmp_path / "values.csv"
    put(path, "id,score,flag,missing\n1,.25,true,NaN\n")
    assert read_csv(path) == [{"id": 1, "score": .25, "flag": True, "missing": None}]
    put(tmp_path / "secret.csv", "secret")
    with pytest.raises(ValueError, match="escapes"):
        resolve_artifact(outputs, "../secret.csv")


def test_report_is_plain_language_and_does_not_show_config_dump(outputs):
    run = get_experiment(outputs, "presentation/paired")
    rendered = render_report(run, outputs)
    assert "Which model followed the paths more accurately?" in rendered
    assert "not prove" in rendered
    assert "Experiment configuration" not in rendered
    assert '"mean_normalized_rmse"' not in rendered
    report = generate_report(outputs, "presentation/paired")
    assert report.is_file()
    assert report.parent == outputs / "analytics" / "reports"


def test_dashboard_http_api_assets_reports_and_bad_paths(outputs):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(outputs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            page = response.read()
            assert b"See how gravity bends" in page
            assert b"Trace a photon" in page
            assert b'id="physics-orbit"' in page and b'id="u-plot"' in page and b'id="v-plot"' in page
            assert b"Reproducibility configuration" not in page
            assert response.headers["Content-Security-Policy"]

        with urlopen(base + "/api/experiments") as response:
            items = json.loads(response.read())["experiments"]
            assert items[0]["kind"] == "presentation"

        with urlopen(base + "/api/experiment?id=presentation%2Fpaired") as response:
            item = json.loads(response.read())
            assert item["summary"]["num_test_rays"] == 2

        with urlopen(base + "/assets/dashboard-ui.js") as response:
            script = response.read()
            assert b"renderSavedExperiment" in script
            assert b"drawOrbitCanvas" in script
            assert b"renderTrainingSnapshot" in script
            assert b"Reproducibility configuration" not in script

        request = Request(
            base + "/api/report",
            data=b'{"id":"presentation/paired"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            report_url = json.loads(response.read())["url"]
            assert response.status == 201

        with urlopen(base + report_url) as response:
            report = response.read()
            assert b"Which model followed the paths more accurately?" in report
            assert b"Experiment configuration" not in report

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


def test_unknown_experiment_is_rejected(outputs):
    with pytest.raises(KeyError, match="Unknown experiment"):
        generate_report(outputs, "../unknown")
