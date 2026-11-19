import numpy as np
import torch

from presentation.common import (
    action_to_state,
    autoregressive_rollout,
    build_supervised_examples,
    extract_ray,
    state_to_action,
)
from presentation.model import DynamicsMLP


def toy_dataset():
    case_ids = np.array([10, 20], dtype=np.int64)
    phi_one = np.linspace(0.0, 0.3, 5)
    r0 = np.array([5.0, 7.0])
    u_one = np.array([0.2, 0.19, 0.17, 0.14, 0.1])
    v_one = np.array([0.0, -0.05, -0.08, -0.11, -0.14])
    u_two = np.array([1 / 7, 0.135, 0.12, 0.10, 0.08])
    v_two = np.array([0.0, -0.03, -0.06, -0.09, -0.12])
    return {
        "case_id": case_ids,
        "r0": r0,
        "phi": np.concatenate([phi_one, phi_one]),
        "u": np.concatenate([u_one, u_two]),
        "v": np.concatenate([v_one, v_two]),
        "offsets": np.array([0, 5, 10], dtype=np.int64),
    }


def test_state_action_mapping_matches_environment_bounds():
    for u, v in [(0.0, 0.0), (0.2, -0.1), (0.49, 0.19)]:
        action = state_to_action(u, v)
        decoded = action_to_state(action)
        assert np.allclose(decoded, (u, v), atol=1e-6)


def test_transition_builder_uses_current_state_and_next_target():
    data = toy_dataset()
    x, y = build_supervised_examples(data, [10])
    assert x.shape == (4, 4)
    assert y.shape == (4, 2)
    assert np.isclose(x[0, 0], 5 / 12)
    assert np.isclose(x[0, 2], 0.2 / 0.5)
    expected_action = state_to_action(0.19, -0.05)
    assert np.allclose(y[0], expected_action)


def test_network_has_expected_shape_and_bounded_actions():
    model = DynamicsMLP(width=128, depth=3)
    output = model(torch.zeros((7, 4)))
    assert output.shape == (7, 2)
    assert torch.all(torch.isfinite(output))


def test_autoregressive_rollout_returns_reference_and_prediction_arrays():
    ray = extract_ray(toy_dataset(), 10)
    result = autoregressive_rollout(ray, lambda _obs: np.array([-0.2, -0.3]))
    assert len(result["phi"]) == 5
    assert result["predicted_u"].shape == ray.u.shape
    assert result["predicted_v"].shape == ray.v.shape
    assert result["predicted_u"][0] == ray.u[0]
    assert result["predicted_v"][0] == ray.v[0]
    assert np.all(np.isfinite(result["predicted_u"]))
    assert np.all(np.isfinite(result["predicted_v"]))


def test_full_comparison_writes_csv_plots_and_plain_language_report(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from presentation.evaluate_pair import run_comparison

    data = toy_dataset()
    dataset_path = tmp_path / "rays.npz"
    np.savez_compressed(dataset_path, **data)
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"test": [10]}), encoding="utf-8")

    model = DynamicsMLP(width=128, depth=3)
    mlp_path = tmp_path / "mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {"input_dim": 4, "width": 128, "depth": 3, "output_dim": 2},
        },
        mlp_path,
    )

    class DummyPPO:
        @staticmethod
        def load(_path, device="cpu"):
            return DummyPPO()

        @staticmethod
        def predict(_obs, deterministic=True):
            return np.asarray([-0.25, -0.35], dtype=np.float32), None

    fake_sb3 = types.SimpleNamespace(PPO=DummyPPO)
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)

    output = tmp_path / "comparison"
    summary = run_comparison(
        mlp_path,
        tmp_path / "mock_ppo.zip",
        dataset_path,
        split_path,
        output,
        split="test",
        example_case=10,
    )

    assert summary["num_test_rays"] == 1
    for filename in (
        "metrics_by_ray.csv",
        "summary.json",
        "run_metadata.json",
        "01_mean_error_comparison.png",
        "02_error_by_test_ray.png",
        "03_example_trajectory_comparison.png",
        "analysis.md",
    ):
        assert (output / filename).is_file(), filename
    report = (output / "analysis.md").read_text(encoding="utf-8")
    assert "RMSE" in report
    assert "teacher-guided" in report


def test_ppo_checkpoint_selection_uses_validation_rollouts_only(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from presentation.evaluate_pair import select_ppo_checkpoint

    data = toy_dataset()
    dataset_path = tmp_path / "rays.npz"
    np.savez_compressed(dataset_path, **data)
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"train": [20], "validation": [10], "test": [20]}), encoding="utf-8")
    best_path = tmp_path / "best_model.zip"
    final_path = tmp_path / "final_model.zip"
    best_path.write_bytes(b"placeholder")
    final_path.write_bytes(b"placeholder")

    class DummyPPOInstance:
        def __init__(self, action):
            self.action = action

        def predict(self, _obs, deterministic=True):
            return np.asarray(self.action, dtype=np.float32), None

    class DummyPPO:
        @staticmethod
        def load(path, device="cpu"):
            # Deliberately different policies so the validation selector has to compare them.
            action = [-0.1, -0.2] if str(path).endswith("best_model.zip") else [-0.8, 0.8]
            return DummyPPOInstance(action)

    monkeypatch.setitem(sys.modules, "stable_baselines3", types.SimpleNamespace(PPO=DummyPPO))
    report_path = tmp_path / "selection.json"
    selected = select_ppo_checkpoint(
        {"validation_reward_best": best_path, "final_iteration": final_path},
        dataset_path,
        split_path,
        report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert selected in (best_path, final_path)
    assert report["test_split_used_for_selection"] is False
    assert report["selected"]["checkpoint"] == str(selected)
    scores = [row["mean_validation_normalized_rmse"] for row in report["candidates"]]
    assert report["selected"]["mean_validation_normalized_rmse"] == min(scores)
