"""Physics-informed neural network for Schwarzschild photon trajectories.

The network maps (r0, phi) -> u=1/r. Its objective is the geodesic ODE
u'' + u - 3/2 u^2 = 0 plus closest-approach initial conditions:
u(0)=1/r0 and du/dphi(0)=0. It does not fit reference trajectory labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from presentation.common import PHI_SCALE, R_SCALE, U_SCALE


class PhotonPINN(nn.Module):
    """Tanh MLP representing the solution u(r0, phi), in scaled units."""

    def __init__(self, width: int = 96, depth: int = 4):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        layers: list[nn.Module] = []
        features = 2
        for _ in range(depth):
            layers.extend((nn.Linear(features, width), nn.Tanh()))
            features = width
        layers.append(nn.Linear(features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Output is u / U_SCALE. Inputs are [r0 / 12, phi / 3].
        return self.network(inputs)


def physics_residual(model: PhotonPINN, r0: torch.Tensor, phi: torch.Tensor):
    """Return physical u, du/dphi and the unscaled geodesic residual."""
    phi_scaled = (phi / PHI_SCALE).detach().requires_grad_(True)
    inputs = torch.cat((r0 / R_SCALE, phi_scaled), dim=1)
    u_scaled = model(inputs)
    du_scaled = torch.autograd.grad(
        u_scaled.sum(), phi_scaled, create_graph=True
    )[0]
    d2u_scaled = torch.autograd.grad(
        du_scaled.sum(), phi_scaled, create_graph=True
    )[0]
    u = U_SCALE * u_scaled
    du = (U_SCALE / PHI_SCALE) * du_scaled
    d2u = (U_SCALE / (PHI_SCALE**2)) * d2u_scaled
    residual = d2u + u - 1.5 * u.square()
    return u, du, residual


def predict_pinn(model: PhotonPINN, r0: float, phi: np.ndarray, device: str = "cpu"):
    """Evaluate u and its derivative at arbitrary phi values."""
    model.eval()
    phi_values = np.asarray(phi, dtype=np.float32).reshape(-1, 1)
    r0_values = np.full_like(phi_values, float(r0))
    phi_scaled = torch.as_tensor(phi_values / PHI_SCALE, device=device).requires_grad_(True)
    r0_scaled = torch.as_tensor(r0_values / R_SCALE, device=device)
    inputs = torch.cat((r0_scaled, phi_scaled), dim=1)
    with torch.enable_grad():
        u_scaled = model(inputs)
        du_scaled = torch.autograd.grad(u_scaled.sum(), phi_scaled)[0]
    u = (U_SCALE * u_scaled).detach().cpu().numpy().reshape(-1)
    v = (U_SCALE / PHI_SCALE * du_scaled).detach().cpu().numpy().reshape(-1)
    return u, v


def load_pinn(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("PINN checkpoint must contain state_dict and config")
    config = payload.get("config", {})
    model = PhotonPINN(
        width=int(config.get("width", 96)),
        depth=int(config.get("depth", 4)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
