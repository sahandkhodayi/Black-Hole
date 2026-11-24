import numpy as np
import torch

from presentation.pinn import PhotonPINN, physics_residual, predict_pinn


def test_pinn_physics_residual_is_finite_and_differentiable():
    model = PhotonPINN(width=16, depth=2)
    r0 = torch.full((12, 1), 5.0)
    phi = torch.linspace(0.0, 0.5, 12).reshape(-1, 1)
    u, v, residual = physics_residual(model, r0, phi)
    loss = residual.square().mean() + u.square().mean() + v.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
               for parameter in model.parameters())


def test_pinn_prediction_returns_finite_u_and_derivative():
    model = PhotonPINN(width=16, depth=2)
    phi = np.linspace(0.0, 0.4, 9)
    u, v = predict_pinn(model, 5.0, phi)
    assert u.shape == phi.shape
    assert v.shape == phi.shape
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))
