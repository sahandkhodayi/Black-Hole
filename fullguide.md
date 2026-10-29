# Black Hole Lab — Complete Code-First Project Textbook

This guide is the implementation reference across all nine milestones: actual source files, explanations, mathematics, run instructions, tests, experiments, and engineering practice. Follow it in order. Code being present in a guide does not mean it has been tested in your environment; verify each milestone before relying on its results.

Repository: https://github.com/sahandkhodayi/Black-Hole

## Research question

Does adding the Schwarzschild photon equation to a neural network's objective improve physical consistency or generalization compared with a data-only model? The experiment must be allowed to show that it does not.

We use Schwarzschild-radius units, r_s = 1, and inverse radius u = r_s/r. The governing equation is

\[
u''+u=\frac32u^2,\qquad u''=1.5u^2-u.
\]

At closest approach r0 = r_min/r_s, u(0)=1/r0 and u'(0)=0. The photon sphere is r=1.5r_s. The outgoing escape event is u=0. For a symmetric scattering trajectory, deflection alpha = 2 phi_escape - pi radians.

## Repository layout

~~~text
pyproject.toml
src/
  simulate.py
  lab/
    __init__.py
    geodesics.py
    dataset.py
    baseline.py
    models.py
    train_nn.py
    pinn.py
scripts/
  __init__.py
  convergence.py
  validate_weak_field.py
  critical_sweep.py
  generate_dataset.py
tests/
  test_geodesics.py
  test_dataset.py
data/       # generated locally
outputs/    # generated locally
~~~

# 0. Environment and engineering setup

## File: pyproject.toml

~~~toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "black-hole-lab"
version = "0.1.0"
description = "Numerical and ML experiments for Schwarzschild light bending"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "matplotlib>=3.7",
  "pandas>=2.0",
  "scikit-learn>=1.3",
  "torch>=2.1",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra"
~~~

The build-system tells pip how to install the project. Project metadata sets the package name, version, and minimum Python version. Dependencies describe the runtime stack; pytest is development-only. Setuptools discovers the package under src. Pytest adds src to the import path.

From the repository root:

~~~bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Use python -m pip so installation targets the active interpreter. Add these patterns to .gitignore if absent:

~~~text
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
data/*.csv
data/*.npz
outputs/*
!outputs/.gitkeep
*.pt
~~~

Create empty src/lab/__init__.py and scripts/__init__.py.

# MILESTONE 1 — Reliable numerical simulation

## 1.1 File: src/lab/geodesics.py

Replace the existing file with this complete implementation.

~~~python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RayTrajectory:
    """Numerical result for one ray, using r_s = 1 units."""

    closest_approach: float
    phi: FloatArray
    u: FloatArray
    escaped: bool
    escape_angle: float | None
    solver_success: bool
    solver_message: str
    nfev: int
    rtol: float
    atol: float
    max_step: float

    @property
    def radius(self) -> FloatArray:
        """Return r/r_s; infinity represents u = 0 at escape."""
        result = np.full_like(self.u, np.inf, dtype=np.float64)
        np.divide(1.0, self.u, out=result, where=self.u > 0)
        return result

    @property
    def deflection_radians(self) -> float | None:
        if not self.escaped or self.escape_angle is None:
            return None
        return 2.0 * self.escape_angle - np.pi

    @property
    def deflection_degrees(self) -> float | None:
        angle = self.deflection_radians
        return None if angle is None else float(np.degrees(angle))


def _rhs(phi: float, state: FloatArray) -> tuple[float, float]:
    """First-order system equivalent to u'' = 1.5*u**2 - u."""
    del phi
    u, du_dphi = state
    d2u_dphi2 = 1.5 * u**2 - u
    return float(du_dphi), float(d2u_dphi2)


def _escape_event(phi: float, state: FloatArray) -> float:
    """Locate the outgoing zero crossing of inverse radius u."""
    del phi
    u, _du_dphi = state
    return float(u)


_escape_event.terminal = True
_escape_event.direction = -1


def simulate_ray(
    closest_approach: float = 5.0,
    *,
    max_angle: float = 20.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.02,
) -> RayTrajectory:
    """Integrate one scattering candidate from its closest approach."""
    values = (closest_approach, max_angle, rtol, atol, max_step)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("All numerical parameters must be finite.")
    if closest_approach <= 1.5:
        raise ValueError("closest_approach must be greater than 1.5 r_s.")
    if max_angle <= 0 or max_step <= 0:
        raise ValueError("max_angle and max_step must be positive.")
    if not (0 < rtol < 1 and 0 < atol < 1):
        raise ValueError("rtol and atol must be between 0 and 1.")

    u0 = 1.0 / closest_approach
    solution = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(u0, 0.0),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=_escape_event,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    escaped = len(solution.t_events[0]) > 0
    escape_angle = float(solution.t_events[0][0]) if escaped else None
    phi_out = solution.t
    u_out = solution.y[0]

    # Reflect the outgoing branch without duplicating phi = 0.
    phi = np.concatenate((-phi_out[:0:-1], phi_out))
    u = np.concatenate((u_out[:0:-1], u_out))

    return RayTrajectory(
        closest_approach=float(closest_approach),
        phi=phi,
        u=u,
        escaped=escaped,
        escape_angle=escape_angle,
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        nfev=int(solution.nfev),
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
~~~

### Line-by-line concepts

- Imports provide structured results, typed arrays, and SciPy's initial-value solver.
- The dataclass collects the trajectory and numerical metadata. Frozen prevents field reassignment, but does not make array contents immutable.
- The radius property computes 1/u only where u is positive. At escape, u tends to zero and radius tends to infinity; negative u is not converted into a fake finite radius.
- The first-order state is (u,v), where v=u'. Therefore _rhs must return (v, 1.5u²-u). Reversing these changes the differential equation.
- The event function is zero at u=0. terminal=True stops integration at the root; direction=-1 selects the positive-to-negative crossing.
- Validation rejects invalid closest approach and invalid numerical settings before integration.
- DOP853 is a high-order explicit Runge–Kutta method. rtol and atol control estimated local error; max_step limits the angular step.
- solver_success and escaped are separate. A successful integration may simply reach max_angle without finding escape.
- t_events contains event locations. The trajectory is mirrored around phi=0 for visualization, while escape_angle is kept from the outgoing integration.
- Deflection is None if escape was not detected, preventing incomplete simulations from becoming fake training labels.

## 1.2 File: src/simulate.py

~~~python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from lab.geodesics import simulate_ray


def plot_rays(closest_approaches=(3.0, 5.0, 10.0), output="outputs/rays.png"):
    """Run examples and save a trajectory figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(Circle((0, 0), 1.0, color="black", label="Event horizon"))
    ax.add_patch(
        Circle((0, 0), 1.5, fill=False, linestyle="--",
               color="gray", label="Photon sphere")
    )

    for r_min in closest_approaches:
        ray = simulate_ray(closest_approach=float(r_min))
        if not ray.escaped:
            print(f"Unresolved at max_angle: r_min={r_min}")
            continue
        radius = ray.radius
        visible = np.isfinite(radius) & (radius <= 30)
        x = radius[visible] * np.cos(ray.phi[visible])
        y = radius[visible] * np.sin(ray.phi[visible])
        ax.plot(x, y, label=f"r_min = {r_min:g} r_s")
        print(
            f"r_min={r_min:g} | deflection={ray.deflection_degrees:.6f} deg "
            f"| nfev={ray.nfev}"
        )

    ax.set_aspect("equal")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("x / r_s")
    ax.set_ylabel("y / r_s")
    ax.set_title("Schwarzschild light-ray trajectories")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    plot_rays()
~~~

This script owns printing, plotting, and file output. The solver remains reusable and import-safe. The mask removes infinite-radius samples before polar-to-Cartesian conversion. The output directory is created before saving, and the figure is closed afterward.

Run: python -m simulate. Expected: ray diagnostics and outputs/rays.png.

## 1.3 File: tests/test_geodesics.py

~~~python
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
~~~

These test derivative order, inverse-radius conversion, event classification, and input validation. They are software tests, not independent proof of the physics. Run python -m pytest -q.

## 1.4 File: scripts/convergence.py

~~~python
import csv
from pathlib import Path

from lab.geodesics import simulate_ray


def main():
    settings = [
        (1e-7, 1e-9, 0.05),
        (1e-9, 1e-11, 0.02),
        (1e-11, 1e-13, 0.01),
    ]
    rows = []
    for rtol, atol, max_step in settings:
        ray = simulate_ray(
            closest_approach=5.0, rtol=rtol, atol=atol, max_step=max_step
        )
        rows.append({
            "rtol": rtol, "atol": atol, "max_step": max_step,
            "escaped": ray.escaped, "escape_angle_rad": ray.escape_angle,
            "deflection_rad": ray.deflection_radians, "nfev": ray.nfev,
        })

    path = Path("outputs/convergence.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
~~~

Keep the physical initial condition fixed while changing solver controls. Compare successive estimates with Δij=|αi−αj|. Stabilization is evidence of convergence, not proof of correctness. Run python -m scripts.convergence.

**Milestone 1 is complete when** imports are side-effect-free, status is explicit, tests pass, and a convergence table is recorded.

# MILESTONE 2 — Validate the physics

## 2.1 Impact parameter and weak-field approximation

For a Schwarzschild turning point r0=r_min/r_s, the dimensionless impact parameter is

\[
\bar b=\frac{b}{r_s}=\frac{r_0}{\sqrt{1-1/r_0}}.
\]

In the weak field, alpha ≈ 2/b̄ radians. Closest-approach radius and impact parameter are different quantities.

### File: scripts/validate_weak_field.py

~~~python
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab.geodesics import simulate_ray


def impact_parameter(r0):
    return r0 / np.sqrt(1.0 - 1.0 / r0)


def main():
    rows = []
    for r0 in [10.0, 15.0, 20.0, 30.0, 50.0, 80.0]:
        ray = simulate_ray(closest_approach=r0, max_angle=8.0)
        b_bar = impact_parameter(r0)
        if ray.escaped:
            numerical = ray.deflection_radians
            weak = 2.0 / b_bar
            error = abs(numerical - weak) / abs(numerical)
            status = "escaped"
        else:
            numerical, weak, error = "", 2.0 / b_bar, ""
            status = "unresolved"
        rows.append({
            "r_min_rs": r0, "impact_parameter_rs": b_bar,
            "numerical_rad": numerical, "weak_field_rad": weak,
            "relative_error": error, "status": status,
        })

    path = Path("outputs/weak_field.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["status"] == "escaped"]
    plt.loglog([r["impact_parameter_rs"] for r in valid],
               [r["numerical_rad"] for r in valid], "o-", label="Numerical")
    plt.loglog([r["impact_parameter_rs"] for r in valid],
               [r["weak_field_rad"] for r in valid], "--", label="Weak-field")
    plt.xlabel("Impact parameter b / r_s")
    plt.ylabel("Deflection (radians)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/weak_field.png", dpi=180)
    plt.close()
    print("Saved outputs/weak_field.csv and outputs/weak_field.png")


if __name__ == "__main__":
    main()
~~~

The weak-field expression is leading-order, so exact equality is not expected. Its relative error should generally decrease far from the hole. Run python -m scripts.validate_weak_field.

## 2.2 File: scripts/critical_sweep.py

~~~python
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from lab.geodesics import simulate_ray


def main():
    offsets = [10.0**(-power) for power in range(1, 6)]
    rows = []
    for offset in offsets:
        r0 = 1.5 + offset
        try:
            ray = simulate_ray(
                closest_approach=r0, max_angle=200.0,
                rtol=1e-9, atol=1e-11, max_step=0.01,
            )
            rows.append({
                "r_min_rs": r0, "escaped": ray.escaped,
                "escape_angle_rad": ray.escape_angle,
                "deflection_rad": ray.deflection_radians,
                "nfev": ray.nfev,
                "status": "escaped" if ray.escaped else "unresolved_at_limit",
            })
        except (ValueError, RuntimeError) as error:
            rows.append({
                "r_min_rs": r0, "escaped": False,
                "escape_angle_rad": "", "deflection_rad": "",
                "nfev": "", "status": f"error: {error}",
            })

    path = Path("outputs/critical_sweep.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["escaped"]]
    if valid:
        plt.plot([r["r_min_rs"] - 1.5 for r in valid],
                 [r["deflection_rad"] for r in valid], "o-")
        plt.xscale("log")
        plt.xlabel("r_min/r_s - 1.5")
        plt.ylabel("Deflection (radians)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig("outputs/critical_sweep.png", dpi=180)
        plt.close()
    print(f"Saved {path}; unresolved cases remain labelled.")


if __name__ == "__main__":
    main()
~~~

Near the photon sphere, deflection grows rapidly and integration may need a larger angular range. Reaching max_angle is unresolved, not proof of capture. Run python -m scripts.critical_sweep.

# MILESTONE 3 — Reproducible dataset

Each CSV row is one independent ray case. The target is deflection in radians. Unresolved cases are kept with empty targets and status metadata.

## File: src/lab/dataset.py

~~~python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab.geodesics import simulate_ray


def impact_parameter(r0):
    return r0 / np.sqrt(1.0 - 1.0 / r0)


def generate_dataset(
    output_csv="data/deflection.csv",
    metadata_json="data/deflection_metadata.json",
    *,
    r_min_values=None,
    seed=42,
    max_angle=30.0,
):
    if r_min_values is None:
        rng = np.random.default_rng(seed)
        r_min_values = 1.5 + np.geomspace(0.03, 30.0, 180)
        rng.shuffle(r_min_values)

    rows = []
    for case_id, value in enumerate(r_min_values):
        r0 = float(value)
        ray = simulate_ray(closest_approach=r0, max_angle=max_angle)
        rows.append({
            "case_id": case_id,
            "r_min_rs": r0,
            "impact_parameter_rs": impact_parameter(r0),
            "escaped": ray.escaped,
            "escape_angle_rad": ray.escape_angle,
            "deflection_rad": ray.deflection_radians,
            "deflection_deg": ray.deflection_degrees,
            "solver_nfev": ray.nfev,
            "solver_rtol": ray.rtol,
            "solver_atol": ray.atol,
            "solver_max_step": ray.max_step,
            "status": "escaped" if ray.escaped else "unresolved_at_max_angle",
        })

    frame = pd.DataFrame(rows)
    csv_path, json_path = Path(output_csv), Path(metadata_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    metadata = {
        "schema_version": 1,
        "input": "r_min_rs",
        "target": "deflection_rad",
        "length_unit": "Schwarzschild radii",
        "angle_unit": "radians",
        "equation": "u'' + u = 1.5*u**2; u = r_s/r",
        "seed": seed,
        "max_angle": max_angle,
        "solver": "SciPy solve_ivp DOP853",
        "rows_total": int(len(frame)),
        "rows_escaped": int(frame["escaped"].sum()),
        "rows_unresolved": int((~frame["escaped"]).sum()),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame, metadata
~~~

### File: scripts/generate_dataset.py

~~~python
import argparse
import numpy as np

from lab.dataset import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate a ray-deflection dataset.")
    parser.add_argument("--output", default="data/deflection.csv")
    parser.add_argument("--metadata", default="data/deflection_metadata.json")
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-angle", type=float, default=30.0)
    args = parser.parse_args()
    if args.count < 10:
        parser.error("--count must be at least 10")
    offsets = np.geomspace(0.03, 30.0, args.count)
    r_values = 1.5 + offsets
    frame, metadata = generate_dataset(
        args.output, args.metadata, r_min_values=r_values,
        seed=args.seed, max_angle=args.max_angle,
    )
    print(frame["status"].value_counts().to_string())
    print(metadata)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
~~~

The seed makes random operations reproducible but does not make numerical integration exact. Geometric spacing adds resolution near the critical region. Status fields preserve unresolved cases instead of silently deleting them. Metadata records units, equation, solver, and row counts.

## File: tests/test_dataset.py

~~~python
import pandas as pd

from lab.dataset import generate_dataset


def test_dataset_schema(tmp_path):
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "metadata.json"
    frame, metadata = generate_dataset(
        csv_path, json_path, r_min_values=[3.0, 5.0, 10.0], max_angle=10.0
    )
    assert csv_path.exists() and json_path.exists()
    assert len(frame) == 3
    assert {"case_id", "r_min_rs", "escaped", "deflection_rad", "status"} <= set(frame.columns)
    assert metadata["rows_total"] == 3
    assert len(pd.read_csv(csv_path)) == 3
~~~

Run python -m scripts.generate_dataset --count 180 --seed 42, then python -m pytest -q. Since each row is one independent ray, row-level splitting is appropriate. If later data contains many points per trajectory, split by trajectory to prevent leakage.

# MILESTONE 4 — Physics baseline

The weak-field approximation alpha≈2/b̄ is transparent and expected to struggle in the strong-field regime. That limitation is part of the baseline, not a bug to hide.

## File: src/lab/baseline.py

~~~python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def weak_field_prediction(r_min_rs):
    r0 = np.asarray(r_min_rs, dtype=np.float64)
    b_bar = r0 / np.sqrt(1.0 - 1.0 / r0)
    return 2.0 / b_bar


def metric_dict(y_true, y_pred):
    return {
        "mae_rad": float(mean_absolute_error(y_true, y_pred)),
        "rmse_rad": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    frame = pd.read_csv("data/deflection.csv")
    frame = frame.loc[frame["escaped"]].dropna(subset=["deflection_rad"])
    _, test = train_test_split(frame, test_size=0.2, random_state=42)
    pred = weak_field_prediction(test["r_min_rs"].to_numpy())
    result = metric_dict(test["deflection_rad"], pred)
    result["n_test"] = int(len(test))
    print(result)
    test.assign(prediction_rad=pred).to_csv(
        "outputs/baseline_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
~~~

MAE is average absolute error; RMSE is the square root of mean squared error. Both are in radians. This initial split is exploratory; Milestone 7 requires one shared saved split for all methods. Run python -m lab.baseline after generating data.

# MILESTONE 5 — Conventional neural network

## File: src/lab/models.py

~~~python
import torch
from torch import nn


class DeflectionMLP(nn.Module):
    """Regress scalar deflection from scalar closest-approach radius."""

    def __init__(self, input_mean, input_std, width=64):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor([input_mean], dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor([input_std], dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        x_scaled = (x - self.input_mean) / self.input_std
        return self.network(x_scaled)
~~~

Buffers save normalization statistics with the checkpoint without making them trainable. Normalization improves optimization when inputs have awkward scale. Tanh is smooth, but no activation is universally best.

## File: src/lab/train_nn.py

~~~python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab.models import DeflectionMLP


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv("data/deflection.csv")
    frame = frame.loc[frame["escaped"]].dropna(subset=["deflection_rad"])
    train_val, test = train_test_split(frame, test_size=0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=seed)

    def array(part, column):
        return part[column].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train, x_val, x_test = [array(part, "r_min_rs") for part in (train, val, test)]
    y_train, y_val, y_test = [array(part, "deflection_rad") for part in (train, val, test)]
    mean, std = float(x_train.mean()), float(x_train.std())
    if std == 0:
        raise ValueError("Training input has zero variance.")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32, shuffle=True,
    )
    model = DeflectionMLP(mean, std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def validate():
        model.eval()
        with torch.no_grad():
            return float(loss_fn(
                model(torch.from_numpy(x_val).to(device)),
                torch.from_numpy(y_val).to(device),
            ).item())

    history, best_state = [], None
    best_val, stale, patience = float("inf"), 0, 40
    for epoch in range(500):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss = validate()
        history.append({
            "epoch": epoch + 1, "train_mse": float(np.mean(losses)),
            "val_mse": val_loss,
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} train_mse={np.mean(losses):.6g} val_mse={val_loss:.6g}")
        if stale >= patience:
            print(f"Early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).to(device)).cpu().numpy().ravel()
    truth = y_test.ravel()
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))

    Path("outputs").mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "input_mean": mean,
        "input_std": std, "seed": seed, "best_validation_mse": best_val,
    }, "outputs/deflection_mlp.pt")
    pd.DataFrame(history).to_csv("outputs/nn_history.csv", index=False)
    pd.DataFrame({
        "r_min_rs": x_test.ravel(), "target_rad": truth,
        "prediction_rad": pred, "residual_rad": pred - truth,
    }).to_csv("outputs/nn_test_predictions.csv", index=False)
    Path("outputs/nn_metrics.json").write_text(json.dumps({
        "mae_rad": mae, "rmse_rad": rmse, "n_test": len(truth),
    }, indent=2), encoding="utf-8")

    plt.plot([r["epoch"] for r in history], [r["train_mse"] for r in history], label="Train")
    plt.plot([r["epoch"] for r in history], [r["val_mse"] for r in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (rad^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/nn_learning_curve.png", dpi=180)
    plt.close()
    print({"device": str(device), "mae_rad": mae, "rmse_rad": rmse})


if __name__ == "__main__":
    main()
~~~

### Training-loop walkthrough

- Split first, normalize second: only training inputs determine mean and standard deviation, preventing test leakage.
- TensorDataset pairs each input and target; DataLoader batches and shuffles training cases.
- zero_grad clears old gradients, backward calculates gradients, and optimizer.step updates parameters.
- Validation uses eval mode and no_grad so it does not update weights or build a gradient graph.
- Early stopping chooses the best validation checkpoint, not the best test score.
- The checkpoint includes normalization values; without matching preprocessing, loaded predictions would be wrong.
- Test metrics are calculated only after model selection.

Run python -m lab.train_nn. Repeat widths 32, 64, and 128 with multiple seeds, and record all trials rather than only the best-looking result.

# MILESTONE 6 — Physics-Informed Neural Network

## Mathematics

The PINN approximates u_theta(phi,r0). Its residual is

\[
\mathcal R=u_{\phi\phi}+u-\frac32u^2.
\]

The objective is L = L_f + lambda_0 L_0 + lambda_1 L_1. L_f penalizes equation residual; L_0 enforces u(0)=1/r0; L_1 enforces du/dphi(0)=0. The ODE alone does not specify a unique trajectory; initial conditions are essential.

## Automatic differentiation sanity check

~~~python
import torch

x = torch.tensor([[0.2], [0.5], [0.8]], requires_grad=True)
y = x**3
dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
d2y = torch.autograd.grad(dy.sum(), x)[0]
expected = 6 * x.detach()
print("computed:", d2y.detach().flatten().tolist())
print("expected:", expected.flatten().tolist())
assert torch.allclose(d2y, expected, atol=1e-5)
~~~

For f(x)=x³, f''(x)=6x. requires_grad asks PyTorch to track operations on x. create_graph=True keeps the first derivative differentiable so it can be differentiated again.

## File: src/lab/pinn.py

~~~python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


class TrajectoryPINN(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, phi, r_min):
        return self.network(torch.cat((phi, r_min), dim=1))


def derivatives(model, phi, r_min):
    phi = phi.detach().clone().requires_grad_(True)
    u = model(phi, r_min)
    du = torch.autograd.grad(u.sum(), phi, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), phi, create_graph=True)[0]
    return u, du, d2u


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for step in range(5000):
        # First study excludes the difficult near-critical region.
        r_f = 3.0 + 8.0 * torch.rand(2048, 1, device=device)
        phi_f = 1.5 * torch.rand(2048, 1, device=device)
        phi0 = torch.zeros(512, 1, device=device)
        r0 = 3.0 + 8.0 * torch.rand(512, 1, device=device)

        optimizer.zero_grad(set_to_none=True)
        u_f, _, d2u_f = derivatives(model, phi_f, r_f)
        residual = d2u_f + u_f - 1.5 * u_f**2
        physics_loss = residual.square().mean()

        u_initial, slope_initial, _ = derivatives(model, phi0, r0)
        position_loss = (u_initial - 1.0 / r0).square().mean()
        slope_loss = slope_initial.square().mean()
        loss = physics_loss + 10.0 * position_loss + 10.0 * slope_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step % 100 == 0:
            row = {
                "step": step, "total_loss": float(loss.item()),
                "physics_loss": float(physics_loss.item()),
                "position_loss": float(position_loss.item()),
                "slope_loss": float(slope_loss.item()),
            }
            history.append(row)
            print(row)

    Path("outputs").mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "seed": 42},
               "outputs/trajectory_pinn.pt")
    pd.DataFrame(history).to_csv("outputs/pinn_history.csv", index=False)

    # Prediction plot only; accuracy must be checked against the solver.
    model.eval()
    phi = torch.linspace(0.0, 1.5, 400, device=device).reshape(-1, 1)
    r0 = torch.full_like(phi, 5.0)
    with torch.no_grad():
        u = model(phi, r0).cpu().numpy().ravel()
    plt.plot(phi.cpu().numpy().ravel(), u, label="PINN")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("phi (radians)")
    plt.ylabel("u = r_s/r")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pinn_example.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
~~~

### Interpretation

The model takes angular coordinate and closest-approach radius as inputs. Autograd computes first and second angular derivatives. The physics residual measures violation of the ODE; initial losses impose the starting position and zero slope. Loss weights are hyperparameters, not universal constants. Collocation points are resampled each step.

This first PINN has no trajectory-data loss and may learn a poor or trivial compromise. The plot is not validation. Measure residual on unseen collocation points, initial-condition error, trajectory error against the numerical solver, and runtime. A PINN is not automatically better because it uses physics.

Run python -m lab.pinn. If it is too slow, reduce collocation batch sizes and steps, and record the changed configuration.

# MILESTONE 7 — Fair comparison

Use identical held-out parameter cases, units, filtering, and metrics. The baseline and MLP scripts above use independently created exploratory splits; their scores are not a strict head-to-head comparison yet. Before a research claim, add one saved split manifest keyed by case_id and use it in all methods.

Report:
- MAE and RMSE in radians;
- worst absolute error;
- number evaluated and excluded/unresolved;
- runtime and hardware;
- errors by parameter regime.

For the PINN, compare against the numerical solver at common angular samples before escape. Do not treat predictions after u=0 as physical finite radii. Keep test data out of architecture, epoch, and loss-weight selection.

# MILESTONE 8 — One controlled research extension

Recommended question: does adaptive sampling near high-error regions improve deflection prediction under a fixed simulation budget?

1. Train on an initial dataset.
2. Evaluate on a separate candidate pool.
3. Select additional simulation cases using a predeclared error/disagreement rule.
4. Simulate those cases and add them to training data.
5. Compare against random sampling using the same number of additional simulations.
6. Repeat across seeds and report mean and spread.

A random-sampling control is essential; otherwise apparent gains may come simply from more data. Alternatives include inverse parameter estimation, near-critical trajectory learning, or PINN loss-weight studies. Choose one based on observed failure modes.

# MILESTONE 9 — Reproducibility, report, release

Checklist:
- dependencies and Python version documented;
- tests run in a clean environment;
- solver status and unresolved cases retained;
- data schema, units, solver settings, and seeds recorded;
- one shared split used before claiming fair comparison;
- checkpoints include preprocessing metadata;
- figures have units and captions;
- outputs are generated by scripts;
- no secrets, virtual environments, caches, or large outputs committed;
- README commands tested from a clean setup.

Suggested sequence from the repository root:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m simulate
python -m scripts.convergence
python -m scripts.validate_weak_field
python -m scripts.critical_sweep
python -m scripts.generate_dataset --count 180 --seed 42
python -m lab.baseline
python -m lab.train_nn
python -m lab.pinn
~~~

A credible report contains the research question, physical assumptions and units, solver settings and convergence evidence, data-generation and split protocol, baseline/NN/PINN details, metrics and plots, uncertainty across seeds, failure analysis, limitations, and exact reproduction commands.

## Completion checklist

- [ ] M1: reliable solver, explicit status, tests, convergence.
- [ ] M2: weak-field validation and critical sweep.
- [ ] M3: dataset generator, schema, unresolved-case metadata.
- [ ] M4: transparent physics baseline.
- [ ] M5: trained MLP and validation-based checkpointing.
- [ ] M6: checked autograd and PINN diagnostics.
- [ ] M7: shared split and fair comparison.
- [ ] M8: one controlled research extension.
- [ ] M9: clean setup, tests, report, reproducible release.

**Scientific honesty:** the code in this guide is an integrated starting implementation, not a claim that all experiments have been executed end-to-end. Run each milestone, resolve any errors, and validate numerical behavior before treating a plot or metric as evidence.
