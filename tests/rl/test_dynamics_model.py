import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rl.dynamics_model import PhotonDynamicsMLP, make_transition_arrays


def test_transition_arrays_use_current_state_and_next_reference_state(tmp_path):
    path = tmp_path / "rays.npz"
    np.savez(
        path,
        case_id=np.asarray([10, 20]),
        r0=np.asarray([5.0, 8.0]),
        phi=np.asarray([0.0, 0.1, 0.17, 0.0, 0.1, 0.16]),
        u=np.asarray([0.2, 0.19, 0.17, 0.125, 0.12, 0.11]),
        v=np.asarray([0.0, -0.1, -0.1, 0.0, -0.05, -0.05]),
        offsets=np.asarray([0, 3, 6]),
    )
    from rl.dynamics_model import load_packed_dataset
    data = load_packed_dataset(path)
    x, y, step_scale = make_transition_arrays(data, [10])

    assert x.shape == (2, 5)
    assert y.shape == (2, 2)
    assert step_scale == pytest.approx(0.1)
    assert x[0, 0] == pytest.approx(5.0 / 12.0)
    assert x[0, 2] == pytest.approx(0.2 / 0.5)
    assert y[0, 0] == pytest.approx(0.19 / 0.5)
    assert x[1, 4] == pytest.approx(0.7)


def test_mlp_maps_five_features_to_two_state_values():
    model = PhotonDynamicsMLP(width=32)
    output = model(torch.zeros((4, 5)))
    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()
