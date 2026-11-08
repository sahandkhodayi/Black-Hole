"""Fast tests for the physical ODE and the existing trajectory wrapper."""

import numpy as np
import pytest

from lab.geodesics import _rhs, simulate_ray


def test_rhs_uses_the_expected_first_order_system():
    u, v = 0.2, -0.03
    du, dv = _rhs(0.0, np.asarray([u, v], dtype=np.float64))
    assert du == pytest.approx(v)
    assert dv == pytest.approx(1.5 * u**2 - u)


def test_closest_approach_is_the_initial_radius():
    ray = simulate_ray(closest_approach=5.0)
    center = int(np.argmin(np.abs(ray.phi)))
    assert ray.radius[center] == pytest.approx(5.0)


def test_standard_scattering_ray_escapes_and_has_positive_deflection():
    ray = simulate_ray(closest_approach=5.0)
    assert ray.escaped
    assert ray.escape_angle is not None
    assert ray.deflection_radian is not None
    assert ray.deflection_radian > 0.0
    assert ray.deflection_degrees == pytest.approx(
        np.degrees(ray.deflection_radian)
    )


def test_short_integration_is_not_mislabeled_as_escape():
    ray = simulate_ray(closest_approach=5.0, max_angle=0.01)
    assert not ray.escaped
    assert ray.escape_angle is None
    assert ray.deflection_radian is None


@pytest.mark.parametrize("r0", [1.5, 1.0, 0.0, -1.0, np.nan, np.inf])
def test_invalid_closest_approach_is_rejected(r0):
    with pytest.raises(ValueError):
        simulate_ray(closest_approach=r0)


@pytest.mark.parametrize("step", [0.0, -0.1, np.nan, np.inf])
def test_invalid_max_step_is_rejected(step):
    with pytest.raises(ValueError):
        simulate_ray(max_step=step)
