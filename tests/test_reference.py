"""Tests for dense DOP853 reference trajectories."""

import numpy as np
import pytest

from lab.reference import build_reference


def invariant(u, v):
    """The exact ODE conserves v^2 - u^3 + u^2."""
    return v**2 - u**3 + u**2


def test_dense_reference_returns_initial_and_intermediate_states():
    ray = build_reference(5.0)
    angles = np.asarray([0.0, 0.2, 0.7])
    states = ray.state_at(angles)
    assert states.shape == (2, len(angles))
    assert states[0, 0] == pytest.approx(1.0 / 5.0)
    assert states[1, 0] == pytest.approx(0.0)


def test_dense_reference_invariant_drift_is_small():
    ray = build_reference(5.0)
    angles = np.linspace(0.0, ray.escape_angle, 500)
    u, v = ray.state_at(angles)
    expected = (1.0 / 5.0) ** 2 - (1.0 / 5.0) ** 3
    assert np.max(np.abs(invariant(u, v) - expected)) < 1e-9


def test_reference_escape_angle_and_deflection_are_consistent():
    ray = build_reference(5.0)
    assert ray.escaped
    assert ray.escape_angle is not None
    assert ray.deflection_radian == pytest.approx(
        2.0 * ray.escape_angle - np.pi
    )


def test_short_reference_is_unresolved_not_escaped():
    ray = build_reference(5.0, max_angle=0.01)
    assert not ray.escaped
    assert ray.escape_angle is None
    assert ray.deflection_radian is None


@pytest.mark.parametrize("angles", [[-0.1], [100.0], [np.nan], [np.inf]])
def test_state_at_rejects_angles_outside_or_invalid(angles):
    ray = build_reference(5.0)
    with pytest.raises(ValueError):
        ray.state_at(angles)


@pytest.mark.parametrize("kwargs", [
    {"closest_approach": 1.5},
    {"max_angle": 0.0},
    {"max_step": -1.0},
    {"rtol": np.nan},
    {"atol": 0.0},
])
def test_reference_rejects_invalid_solver_inputs(kwargs):
    with pytest.raises(ValueError):
        if "closest_approach" in kwargs:
            build_reference(kwargs["closest_approach"])
        else:
            build_reference(5.0, **kwargs)
