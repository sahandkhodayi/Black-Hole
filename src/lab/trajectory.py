"""Generate and store complete reference photon trajectories.

The dataset is packed into flat arrays. Ray i occupies the half-open interval
offsets[i]:offsets[i + 1]. This supports variable-length rays without padding.
"""

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .reference import build_reference


def sample_reference(reference, delta_phi: float):
    """Sample a dense reference on a uniform phi grid and include escape.

    DOP853 chooses its own adaptive steps. We resample its dense interpolant at
    a fixed delta_phi so different rays share the same angular spacing while
    retaining their own (variable) final escape angle.
    """
    try:
        spacing = float(delta_phi)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_phi must be finite and positive") from exc

    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("delta_phi must be finite and positive")
    if not reference.escaped:
        raise ValueError("Cannot sample a trajectory without an escape event")

    end = float(reference.escape_angle)
    phi = np.arange(0.0, end, spacing, dtype=np.float64)

    # Make sure the starting point is present, even for an unusually short ray.
    if phi.size == 0:
        phi = np.asarray([0.0], dtype=np.float64)

    # Include the exact event angle. The final step may be shorter than
    # delta_phi; that is intentional and prevents us losing the escape state.
    if np.isclose(phi[-1], end, rtol=0.0, atol=1e-12):
        phi[-1] = end
    else:
        phi = np.append(phi, end)

    states = reference.state_at(phi).T
    return phi, states[:, 0], states[:, 1]


def generate_trajectory_dataset(
    r0_values: Iterable[float],
    output_path: str | Path = "data/photon_trajectories.npz",
    manifest_path: str | Path = "data/trajectory_manifest.json",
    *,
    delta_phi: float = 0.01,
    max_angle: float = 20.0,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_step: float = 0.01,
    seed: int | None = None,
) -> dict:
    """Generate a packed collection of escaping scattering rays.

    case_id is the original position in r0_values. Unresolved cases are
    reported in the JSON manifest, but are never stored as complete rays.
    """
    try:
        requested_r0 = np.asarray(list(r0_values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("r0_values must be a non-empty iterable of numbers") from exc

    if requested_r0.ndim != 1 or requested_r0.size == 0:
        raise ValueError("r0_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(requested_r0)) or np.any(requested_r0 <= 1.5):
        raise ValueError("Every closest-approach radius must be finite and > 1.5")

    try:
        spacing = float(delta_phi)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_phi must be finite and positive") from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("delta_phi must be finite and positive")

    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    # Every ray may have a different escape angle and therefore a different
    # number of samples. Flat arrays plus offsets preserve all samples.
    all_phi: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    offsets = [0]

    escaped_case_ids: list[int] = []
    escaped_r0: list[float] = []
    escape_angles: list[float] = []
    deflections: list[float] = []
    unresolved: list[dict] = []

    for case_id, r0 in enumerate(requested_r0):
        reference = build_reference(
            float(r0),
            max_angle=max_angle,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if not reference.escaped:
            unresolved.append({
                "case_id": case_id,
                "r0": float(r0),
                "status": "unresolved_at_max_angle",
                "last_phi": float(reference.solution.t[-1]),
            })
            continue

        phi, u, v = sample_reference(reference, spacing)
        if not (
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
        ):
            unresolved.append({
                "case_id": case_id,
                "r0": float(r0),
                "status": "non_finite_reference",
            })
            continue

        escaped_case_ids.append(case_id)
        escaped_r0.append(float(r0))
        all_phi.append(phi)
        all_u.append(u)
        all_v.append(v)
        offsets.append(offsets[-1] + len(phi))
        escape_angles.append(float(reference.escape_angle))
        deflections.append(float(reference.deflection_radian))

    if not escaped_r0:
        raise RuntimeError(
            "No complete escaping trajectories were generated. "
            "Increase max_angle or choose less critical closest approaches."
        )

    packed_phi = np.concatenate(all_phi)
    packed_u = np.concatenate(all_u)
    packed_v = np.concatenate(all_v)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        schema_version=np.asarray(2, dtype=np.int64),
        case_id=np.asarray(escaped_case_ids, dtype=np.int64),
        r0=np.asarray(escaped_r0, dtype=np.float64),
        phi=packed_phi,
        u=packed_u,
        v=packed_v,
        offsets=np.asarray(offsets, dtype=np.int64),
        escape_angle=np.asarray(escape_angles, dtype=np.float64),
        deflection_radian=np.asarray(deflections, dtype=np.float64),
    )

    manifest = {
        "schema_version": 2,
        "description": "Schwarzschild scattering photon trajectories",
        "equation": "u'' = 1.5*u**2 - u",
        "state": ["u=1/r", "v=du/dphi"],
        "length_unit": "Schwarzschild radii (r_s = 1)",
        "angle_unit": "radians",
        "reference_solver": "SciPy solve_ivp(method='DOP853', dense_output=True)",
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step": float(max_step),
        "max_angle": float(max_angle),
        "delta_phi": spacing,
        "seed": seed,
        "num_requested": int(requested_r0.size),
        "num_escaped": len(escaped_r0),
        "escaped_case_ids": escaped_case_ids,
        "unresolved": unresolved,
        "data_file": output_path.as_posix(),
        "arrays": {
            "case_id": "Original index in the requested r0_values sequence",
            "r0": "Closest approach for each successfully escaped ray",
            "phi": "Packed angular coordinates, in radians",
            "u": "Packed inverse radius, u=1/r",
            "v": "Packed derivative, v=du/dphi",
            "offsets": "Ray i occupies [offsets[i], offsets[i+1])",
            "escape_angle": "Escape angle for each stored ray, in radians",
            "deflection_radian": "Total deflection for each stored ray, in radians",
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
