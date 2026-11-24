import json
import sys
import types

import numpy as np
import torch

from presentation.model import DynamicsMLP
from presentation.pinn import PhotonPINN


def test_model_arena_exports_same_ray_predictions_and_metadata(tmp_path, monkeypatch):
    from presentation.evaluate_arena import run_arena

    phi = np.linspace(0.0, 0.25, 8)
    u = 0.2 - 0.03 * phi - 0.02 * phi**2
    v = np.gradient(u, phi)
    data = {
        "case_id": np.array([9], dtype=np.int64),
        "r0": np.array([5.0]),
        "phi": phi,
        "u": u,
        "v": v,
        "offsets": np.array([0, len(phi)], dtype=np.int64),
    }
    dataset = tmp_path / "rays.npz"
    np.savez_compressed(dataset, **data)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"test": [9]}), encoding="utf-8")

    mlp = DynamicsMLP(width=8, depth=2)
    mlp_path = tmp_path / "mlp.pt"
    torch.save({
        "state_dict": mlp.state_dict(),
        "architecture": {"input_dim": 4, "width": 8, "depth": 2, "output_dim": 2},
    }, mlp_path)

    pinn = PhotonPINN(width=8, depth=2)
    pinn_path = tmp_path / "pinn.pt"
    torch.save({"state_dict": pinn.state_dict(), "config": {"width": 8, "depth": 2}}, pinn_path)

    class DummyPPO:
        @staticmethod
        def load(_path, device="cpu"):
            return DummyPPO()

        def predict(self, _obs, deterministic=True):
            return np.asarray([-0.2, -0.3], dtype=np.float32), None

    monkeypatch.setitem(sys.modules, "stable_baselines3", types.SimpleNamespace(PPO=DummyPPO))
    output = tmp_path / "arena"
    summary = run_arena(
        dataset_path=dataset, splits_path=splits, mlp_path=mlp_path,
        ppo_path=tmp_path / "ppo.zip", pinn_path=pinn_path,
        output_dir=output, split="test", example_case=9,
    )
    assert {row["model"] for row in summary} == {"MLP", "PPO", "PINN"}
    for name in ("predictions.csv", "metrics_by_ray.csv", "summary.csv",
                 "summary.json", "run_metadata.json", "model_arena_overview.png"):
        assert (output / name).is_file()
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["live_training"] is False
    assert metadata["fairness"]["same_held_out_complete_rays"] is True
