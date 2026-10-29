import numpy as np
import pytest

from lab.geodesics import _rhs, simulate_ray


def test_rhs_derivative_order():
    u, v = 0.2, 0.0
    du, dv = _rhs(0.0, np.array([u, v], dtype=np.float64))
    assert du == pytest.approx(v)
    assert dv == pytest.approx(1.5 * u**2 - u)


def test_radius_at_closest_approach():
    ray = simulate_ray(closest_approach=5.0)
    center = int(np.argmin(np.abs(ray.phi)))
    assert ray.radius[center] == pytest.approx(5.0)


def test_standard_ray_escapes():
    ray = simulate_ray(closest_approach=5.0)
    assert ray.solver_success and ray.escaped
    assert ray.deflection_radians is not None
    assert ray.deflection_radians > 0


def test_short_integration_is_not_escape():
    ray = simulate_ray(closest_approach=5.0, max_angle=0.01)
    assert not ray.escaped
    assert ray.deflection_radians is None


@pytest.mark.parametrize("r_min", [1.5, 1.0, 0.0, float("nan")])
def test_invalid_closest_approach(r_min):
    with pytest.raises(ValueError):
        simulate_ray(closest_approach=r_min)