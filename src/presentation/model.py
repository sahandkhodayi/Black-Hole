"""The selected supervised dynamics network: ReLU, width 128, depth 3."""
from __future__ import annotations

import torch
from torch import nn


class DynamicsMLP(nn.Module):
    """Map [r0, phi, u, v] observations to the next [u, v] action.

    The architecture matches the user's selected run name:
    dynamics_architecture_sweep_relu_w128_d3

    Three hidden Linear(128) layers use ReLU. The output layer is linear,
    as is conventional for a regression model. During rollout, the outputs
    are clipped to the same valid action range as the PPO environment.
    """

    def __init__(self, input_dim: int = 4, width: int = 128, depth: int = 3, output_dim: int = 2):
        super().__init__()
        if input_dim <= 0 or width <= 0 or depth <= 0 or output_dim <= 0:
            raise ValueError("All network dimensions must be positive")
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(in_features, width), nn.ReLU()))
            in_features = width
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_dynamics_mlp(checkpoint_path: str, device: str = "cpu") -> tuple[DynamicsMLP, dict]:
    """Load a checkpoint produced by presentation.train_mlp."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            "Unsupported MLP checkpoint. Expected a presentation checkpoint "
            "containing 'state_dict' and 'architecture'."
        )
    architecture = payload.get("architecture", {})
    model = DynamicsMLP(
        input_dim=int(architecture.get("input_dim", 4)),
        width=int(architecture.get("width", 128)),
        depth=int(architecture.get("depth", 3)),
        output_dim=int(architecture.get("output_dim", 2)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload