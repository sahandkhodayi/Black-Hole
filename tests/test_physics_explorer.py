"""Tests for the browser Physics Explorer simulation payload."""
import numpy as np
import pytest

from bh_analytics.dashboard import _simulate_photon_payload


def test_payload_contains_finite_symmetric_scattering_curve():
    result = _simulate_photon_payload(5.0)
    assert result["escaped"] is True
    assert result["deflection_degrees"] > 0
    assert result["nfev"] > 0
    assert len(result["x"]) > 20
    assert len(result["x"]) == len(result["y"]) == len(result["u"])
    assert np.all(np.isfinite(result["x"]))
    assert np.all(np.isfinite(result["y"]))
    assert min(result["radius"]) == pytest.approx(5.0, abs=1e-8)
    assert max(result["radius"]) <= 24.0


@pytest.mark.parametrize("value", [None, "not-a-number", float("nan"), float("inf"), 1.5, 1.79, 15.01, -2])
def test_payload_rejects_invalid_closest_approach(value):
    with pytest.raises(ValueError):
        _simulate_photon_payload(value)
