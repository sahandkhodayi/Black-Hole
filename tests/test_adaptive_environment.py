import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from rl.adaptive_environment import AdaptivePhotonEnv, rk4_step


def tiny_dataset(path):
    phi = np.linspace(0.0, 0.5, 51)
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    u[0] = 0.2
    for i in range(len(phi) - 1):
        u[i + 1], v[i + 1] = rk4_step(u[i], v[i], phi[i + 1] - phi[i])
    np.savez_compressed(path, case_id=np.array([7]), r0=np.array([5.0]),
                        phi=phi, u=u, v=v, offsets=np.array([0, len(phi)]))


def test_rk4_step_returns_finite_state():
    u, v = rk4_step(0.2, 0.0, 0.01)
    assert np.isfinite(u)
    assert np.isfinite(v)
    assert u > 0


def test_adaptive_environment_advances_with_bounded_actions(tmp_path):
    path = tmp_path / "rays.npz"
    tiny_dataset(path)
    env = AdaptivePhotonEnv(path, [7], min_step=0.002, max_step=0.04)
    obs, info = env.reset(options={"case_id": 7})
    assert obs.shape == (5,)
    assert info["case_id"] == 7
    next_obs, reward, terminated, truncated, step_info = env.step(np.array([1.0]))
    assert next_obs.shape == (5,)
    assert np.isfinite(reward)
    assert step_info["step_size"] == pytest.approx(0.04)
    assert step_info["phi"] > 0
    assert not truncated
    env.close()


def test_adaptive_environment_rejects_invalid_action(tmp_path):
    path = tmp_path / "rays.npz"
    tiny_dataset(path)
    env = AdaptivePhotonEnv(path, [7])
    env.reset(options={"case_id": 7})
    with pytest.raises(ValueError):
        env.step(np.array([np.nan]))
