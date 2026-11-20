import math

import pytest

from bh_analytics.physics import ray_payload


def test_ray_payload_contains_a_real_mirrored_dop853_solution():
    ray = ray_payload(5.0)

    assert ray["escaped"] is True
    assert ray["samples"] > 100
    assert len(ray["phi"]) == len(ray["u"]) == len(ray["v"])
    assert len(ray["x"]) == len(ray["y"]) == ray["samples"]
    assert ray["deflection_degrees"] == pytest.approx(28.66, abs=1.0)

    middle = min(range(len(ray["phi"])), key=lambda index: abs(ray["phi"][index]))
    assert ray["phi"][middle] == pytest.approx(0.0, abs=1e-12)
    assert ray["u"][middle] == pytest.approx(0.2, abs=1e-8)
    assert ray["v"][middle] == pytest.approx(0.0, abs=1e-8)
    assert ray["x"][middle] == pytest.approx(5.0, abs=1e-7)
    assert ray["y"][middle] == pytest.approx(0.0, abs=1e-7)
    assert all(math.isfinite(value) for key in ("phi", "u", "v", "x", "y") for value in ray[key])


@pytest.mark.parametrize("closest_approach", [1.5, 1.49, 0, -1, float("nan"), 12.1])
def test_ray_payload_rejects_invalid_scattering_radii(closest_approach):
    with pytest.raises(ValueError):
        ray_payload(closest_approach)
