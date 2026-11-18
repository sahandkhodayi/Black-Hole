from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from rl.environment import PhotonTrajectoryEnv


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    path = tmp_path / "rays.npz"
    # The final angular interval is deliberately shorter than the first.
    np.savez(
        path,
        case_id=np.asarray([10, 20]),
        r0=np.asarray([5.0, 8.0]),
        phi=np.asarray([0.0, 0.1, 0.17, 0.0, 0.1, 0.16]),
        u=np.asarray([0.2, 0.19, 0.17, 0.125, 0.12, 0.11]),
        v=np.asarray([0.0, -0.1, -0.1, 0.0, -0.05, -0.05]),
        offsets=np.asarray([0, 3, 6]),
    )
    return path, [10, 20]


def test_reset_returns_finite_observation(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids, randomize_cases=False)
    obs, info = env.reset(options={"case_id": 20})
    assert obs.shape == (5,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["case_id"] == 20 and info["r0"] == 8.0
    assert obs[4] == pytest.approx(1.0)


def test_observation_includes_the_next_angular_step(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids)
    obs, _ = env.reset(options={"case_id": 10})
    assert obs[4] == pytest.approx(1.0)

    next_obs, _, terminated, _, info = env.step(
        np.asarray([-0.24, -0.5], dtype=np.float32)
    )
    assert not terminated
    assert info["delta_phi"] == pytest.approx(0.1)
    assert next_obs[4] == pytest.approx(0.7)


def test_perfect_action_gets_max_reward(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids)
    env.reset(options={"case_id": 10})
    action = np.asarray([-0.24, -0.5], dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == pytest.approx(1.0)
    assert not terminated and not truncated
    assert info["phi"] == pytest.approx(0.1)


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


def test_reward_is_bounded_and_prefers_smaller_prediction_error(tiny_dataset):
    path, ids = tiny_dataset
    env = PhotonTrajectoryEnv(path, ids)
    env.reset(options={"case_id": 10})
    _, perfect_reward, _, _, _ = env.step(
        np.asarray([-0.24, -0.5], dtype=np.float32)
    )

    env.reset(options={"case_id": 10})
    _, poor_reward, _, _, _ = env.step(np.asarray([1.0, 1.0], dtype=np.float32))

    assert 0.0 < poor_reward < perfect_reward == pytest.approx(1.0)
