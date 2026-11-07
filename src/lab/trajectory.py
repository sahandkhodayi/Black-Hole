import json
from pathlib import Path

import numpy as np

from .refrence import build_reference


def sample_reference(reference, delta_phi):
    if not np.isfinite(delta_phi) or delta_phi <= 0:
        raise ValueError("delta_phi must be finite and positive")

    if not reference.escaped:
        raise ValueError("Cannot sample a trajectory without an escape event")

    end = reference.escape_angle
    phi = np.arange(0.0, end, delta_phi, dtype=np.float64)

    if len(phi) == 0:
        phi = np.asarray([end], dtype=np.float64)
    elif np.isclose(phi[-1], end, rtol=0.0, atol=1e-12):
        phi[-1] = end
    else:
        phi = np.append(phi, end)

    states = reference.state_at(phi).T

    return phi, states[:, 0], states[:, 1]


def generate_trajectory_dataset(
    r0_values,
    output_path="data/photon_trajectories.npz",
    manifest_path="data/trajectory_manifest.json",
    *,
    delta_phi=0.01,
    max_angle=20.0,
    rtol=1e-12,
    atol=1e-14,
    max_step=0.01,
):
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    all_phi = []
    all_u = []
    all_v = []
    offsets = [0]

    escaped_r0 = []
    escape_angles = []
    deflections = []
    unresolved = []

    for case_id, r0_value in enumerate(r0_values):
        r0 = float(r0_value)

        reference = build_reference(
            r0,
            max_angle=max_angle,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if not reference.escaped:
            unresolved.append({
                "case_id": case_id,
                "r0": r0,
                "status": "unresolved_at_max_angle",
            })
            continue

        phi, u, v = sample_reference(reference, delta_phi)

        if not (
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
        ):
            unresolved.append({
                "case_id": case_id,
                "r0": r0,
                "status": "non_finite_reference",
            })
            continue

        all_phi.append(phi)
        all_u.append(u)
        all_v.append(v)

        offsets.append(offsets[-1] + len(phi))

        escaped_r0.append(r0)
        escape_angles.append(reference.escape_angle)
        deflections.append(reference.deflection_radian)

    if not escaped_r0:
        raise RuntimeError("No complete escaping trajectories were generated")

    packed_phi = np.concatenate(all_phi)
    packed_u = np.concatenate(all_u)
    packed_v = np.concatenate(all_v)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        r0=np.asarray(escaped_r0, dtype=np.float64),
        phi=packed_phi,
        u=packed_u,
        v=packed_v,
        offsets=np.asarray(offsets, dtype=np.int64),
        escape_angle=np.asarray(escape_angles, dtype=np.float64),
        deflection_radian=np.asarray(deflections, dtype=np.float64),
    )

    manifest = {
        "schema_version": 1,
        "description": "Schwarzschild scattering photon trajectories",
        "equation": "u'' = 1.5*u**2 - u",
        "state": ["u", "v=du/dphi"],
        "length_unit": "Schwarzschild radii",
        "angle_unit": "radians",
        "reference_solver": "SciPy DOP853",
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "max_angle": max_angle,
        "delta_phi": delta_phi,
        "num_escaped": len(escaped_r0),
        "unresolved": unresolved,
        "data_file": str(output_path),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest