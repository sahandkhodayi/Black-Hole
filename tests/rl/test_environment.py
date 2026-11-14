from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("gymnasium")
from rl.environment import PhotonTrajectoryEnv


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    path = tmp_path / "rays.npz"
    np.savez(path, case_id=np.asarray([10, 20]), r0=np.asarray([5., 8.]),
             phi=np.asarray([0., .1, .2, 0., .1, .2]),
             u=np.asarray([.2, .19, .17, .125, .12, .11]),
             v=np.asarray([0., -.1, -.1, 0., -.05, -.05]),
             offsets=np.asarray([0, 3, 6]))
    return path, [10, 20]


def test_reset_returns_finite_observation(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids, randomize_cases=False)
    obs, info = env.reset(options={"case_id": 20})
    assert obs.shape == (4,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["case_id"] == 20 and info["r0"] == 8.


def test_perfect_action_gets_max_reward(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids)
    env.reset(options={"case_id": 10})
    action = np.asarray([-0.24, -0.5], dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(1.0)
    assert not terminated and not truncated
    assert info["phi"] == pytest.approx(.1)


def test_episode_ends_at_reference_endpoint(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids)
    env.reset(options={"case_id": 10})
    action = np.asarray([-0.24, -0.5], dtype=np.float32)
    env.step(action)
    _, _, terminated, truncated, _ = env.step(action)
    assert terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(action)
