# Black Hole Lab — Complete Project Textbook

**A code-first research project on Schwarzschild photon trajectories and reinforcement learning.**

> **Main research question:** Can a reinforcement-learning agent learn to generate photon trajectories near a Schwarzschild black hole, reproduce gravitational light bending, and generalize to initial conditions it has never seen?

This is the one canonical project guide. It combines the useful mathematics, implementation explanations, tests, numerical experiments, machine-learning comparisons, RL design, and research workflow from the previous Markdown guides. Follow it in order. Code included here is a reference implementation or a milestone blueprint; it does not mean that the code has already been added to the repository or tested in your checkout.

---

# How to use this textbook

Each milestone follows this cycle:

1. **Understand** the physics, mathematics, or engineering idea.
2. **Inspect** the current repository implementation.
3. **Attempt** the task yourself before reading the full reference implementation.
4. **Implement** the smallest complete change.
5. **Test** normal cases and failure cases.
6. **Measure** results and save reproducible outputs.
7. **Explain** what the result means before moving to the next milestone.

You already know Python and Git and have completed a full programming project. This guide does not restart from beginner Python. It focuses on the new physics, numerical-analysis, RL, and research-engineering work required by this project.

When a code example differs from the actual source API, the source code wins. Inspect the file and adapt the guide rather than replacing working code blindly.

## The project in one picture

~~~text
Schwarzschild physics
        |
        v
DOP853 reference trajectories
        |
        +------> numerical validation
        |
        +------> multi-ray trajectory dataset
                         |
             +-----------+-----------+
             |                       |
             v                       v
       supervised MLP              RL agent
       / PINN comparison           sequentially generates
             |                     the next photon state
             |                       |
             +-----------+-----------+
                         |
                         v
             held-out trajectory evaluation
                         |
                         v
          curve accuracy, deflection, physics,
              generalization and limitations
~~~

The main agent generates the curve. Numerical solvers provide trusted physics references and baselines; they are not the RL agent's action.

---

# PART I — Project definition and current repository

## 1. What we are building

The original project numerically simulates light bending around a non-rotating Schwarzschild black hole. We are evolving it into a research project where an RL policy generates the photon trajectory sequentially.

The agent will:

- receive the initial photon condition and its current state;
- predict the next point of the trajectory;
- receive feedback about curve accuracy and physical consistency;
- repeat this process until the ray escapes or the episode fails;
- learn to handle different closest-approach radii;
- be evaluated on initial conditions not used during training.

The agent does **not**:
- change the black-hole mass or spacetime;
- alter the governing geodesic equation;
- control a real photon;
- choose the numerical solver step size in the main experiment.

The environment uses a fixed angular increment. The policy's action is the next predicted photon state, not a step-size multiplier.

## 2. Research scope and scientific honesty

The project studies a deterministic differential equation. A numerical method already solves it accurately. Therefore, RL is not automatically a better way to compute the trajectory.

Our question is whether sequential reward-driven learning can generate accurate trajectories and generalize to unseen initial conditions. The work must compare RL with simpler methods, particularly supervised learning and direct numerical integration.

A successful result does not require RL to beat every baseline. A carefully measured result showing that supervised learning or a numerical solver is more suitable is scientifically valid.

We must distinguish three kinds of learning:

| Method | Training signal | Role in this project |
|---|---|---|
| Supervised learning | Explicit reference input/target pairs | Important baseline |
| Physics-Informed Neural Network (PINN) | Differential-equation residual plus initial/boundary conditions | Physics-aware comparison |
| Reinforcement Learning | Rewards from sequential environment interaction | Main research direction |

Our RL agent will receive a dense reference-tracking reward during the first experiment. That makes it **teacher-guided RL**. It is not discovering general relativity from nothing. We will also test physics-only reward variants and compare against supervised models.

## 3. What is already implemented

The inspected repository contains:

| File | Responsibility | Current status |
|---|---|---|
| src/lab/geodesics.py | Schwarzschild photon ODE, escape event, DOP853 simulation, RayTrajectory | Implemented |
| src/lab/integrators.py | Classical RK4 one-step method | Implemented |
| src/lab/compare_plot.py | Early RK4/DOP853 comparison plot | Prototype |
| project.toml | Project metadata | Present under this filename |

The current geodesic module exposes:
- RayTrajectory.phi
- RayTrajectory.u
- RayTrajectory.escaped
- RayTrajectory.escape_angle
- RayTrajectory.nfev
- RayTrajectory.radius
- RayTrajectory.deflection_radian
- RayTrajectory.deflection_angle
- simulate_ray()
- simulate_ray_raw()
- _rhs()
- _escape_event()

The current integrator exposes:

~~~python
rk4_step(rhs, phi, state, step_size)
~~~

The current state order is **[u, v]**, where u is inverse radius and v is its derivative with respect to phi.

### Known numerical observations

Earlier runs produced these escape-angle values:

| Closest approach r0 | Escape angle |
|---:|---:|
| 3.0 | about 2.078234 rad |
| 5.0 | about 1.820914 rad |
| 10.0 | about 1.681734 rad |

These are regression references from previous runs, not exact analytical constants. Slight changes may occur with different solver tolerances or library versions.

An earlier RK4 comparison at r0=5 and h=0.01 reported 182 RK4 steps, 728 RK4 RHS evaluations, 1109 DOP853 RHS evaluations, and a sampled maximum u difference around 6.70e-6. This one run does not prove RK4 is generally more efficient; it was not a full matched-accuracy benchmark.

## 4. Target project structure

This is a target structure, not a claim that every future file already exists. Create modules when their milestone begins.

~~~text
pyproject.toml                 # packaging metadata after resolving current filename
src/
  lab/
    __init__.py
    geodesics.py                # physical model and DOP853
    integrators.py              # RK4 one-step method
    reference.py                # dense reference trajectories
    validation.py               # weak-field and critical sweeps
    trajectory_data.py          # full trajectory dataset
    trajectory_metrics.py       # curve and physics metrics
    models.py                   # supervised neural baselines
    pinn.py                     # physics-informed model
    trajectory_env.py           # RL environment
    policies.py                 # baseline policies
    train_rl.py                 # optional package-level training entry
scripts/
  generate_trajectories.py
  inspect_trajectories.py
  validate_weak_field.py
  critical_sweep.py
  train_supervised.py
  train_rl.py
  evaluate_models.py
  render_trajectories.py
tests/
  test_geodesics.py
  test_integrators.py
  test_reference.py
  test_validation.py
  test_trajectory_data.py
  test_trajectory_metrics.py
  test_models.py
  test_pinn.py
  test_trajectory_env.py
data/                            # generated locally; normally gitignored
outputs/                         # plots, CSVs, metrics; selected artifacts only
~~~

Do not create all these files immediately. The next milestone queue at the end tells you what to create now.

---

# PART II — Schwarzschild photon physics

## 5. Units, variables, and equation

Use normalized Schwarzschild-radius units:

\[
r_s=1.
\]

Here r is measured in units of the Schwarzschild radius. Define inverse radius:

\[
u(\phi)=\frac{1}{r(\phi)}.
\]

The independent variable is the angular coordinate phi. The reduced photon orbit equation implemented in this project is:

\[
\frac{d^2u}{d\phi^2}+u=\frac{3}{2}u^2,
\]

or equivalently:

\[
\boxed{u''=\frac{3}{2}u^2-u}.
\]

This equation describes equatorial null geodesics in Schwarzschild spacetime under the chosen normalization.

## 6. Convert the second-order equation to a first-order system

Numerical ODE interfaces usually operate on a first-order system. Define:

\[
v=\frac{du}{d\phi}.
\]

Then:

\[
\frac{du}{d\phi}=v,
\]

\[
\frac{dv}{d\phi}=\frac{3}{2}u^2-u.
\]

The state is:

\[
\mathbf y=
\begin{bmatrix}
u\\v
\end{bmatrix},
\qquad
\mathbf y'=f(\phi,\mathbf y).
\]

Therefore the right-hand side must return:

~~~python
return du_dphi, d2u_dphi2
~~~

Do not reverse the state order. The first component is u; the second is v.

## 7. Initial conditions and the photon sphere

The existing simulation starts at the point of closest approach r0. By symmetry:

\[
u(0)=\frac{1}{r_0},\qquad v(0)=0.
\]

The photon sphere is at:

\[
r_{\rm ph}=\frac{3}{2}r_s.
\]

For the current scattering setup, use:

\[
r_0>1.5.
\]

As r0 approaches 1.5 from above, the photon can execute increasingly strong bending and orbit near the photon sphere before escaping. The required angular integration interval can become very large.

At exactly r0=1.5, the current initial state lies on the circular photon orbit. It should not be treated as an ordinary escaping ray.

For r0 below 1.5, the existing closest-approach initialization is not an appropriate way to represent a captured ray. Captured-ray experiments need a physically correct incoming initial condition, for example one parameterized by impact parameter at large radius. Add capture only as a later extension.

## 8. Escape and deflection

The outgoing scattering ray escapes when u crosses zero:

\[
u=0.
\]

This corresponds to radius tending to infinity. SciPy's event function returns u. It is terminal and has direction -1 so that it detects the positive-to-negative crossing.

The positive-side event coordinate is the escape angle:

\[
\phi_{\rm escape}.
\]

For the symmetric scattering setup, the deflection is:

\[
\boxed{\alpha=2\phi_{\rm escape}-\pi}.
\]

The result property in the existing code is named **deflection_radian**, singular. Do not copy old snippets that refer to deflection_radians or deflection_degrees unless you add those properties deliberately.

The current plot reflects the positive-side trajectory across the x-axis. This uses the symmetry of the scattering solution; it does not mean the incoming half was separately integrated.

## 9. Cartesian visualization

Since u=1/r:

\[
r=\frac{1}{u}.
\]

Convert to Cartesian coordinates:

\[
x=r\cos\phi,\qquad y=r\sin\phi.
\]

The event horizon is a circle of radius one in the normalized plot. The photon sphere is a circle of radius 1.5 and can be added as a visual reference, but it is not the event horizon.

At escape u tends to zero, so r tends to infinity. Plotting code must mask u<=0 and optionally mask radii beyond a chosen visible radius. Do not allow infinity to determine the axes.

For a symmetric scattering plot, mirror the positive-side ray:

- original path: (x(phi), y(phi));
- reflected path: (x(phi), -y(phi)).

Use equal aspect ratio. Keep a reference curve and a predicted curve on the same axes when evaluating a model.

## 10. Conserved quantity

The first-order system has a useful first integral.

Starting with:

\[
v'=\frac{3}{2}u^2-u,\qquad u'=v,
\]

multiply the first equation by 2v:

\[
2vv'=3u^2v-2uv.
\]

Since u'=v:

\[
\frac{d(v^2)}{d\phi}
=
\frac{d(u^3-u^2)}{d\phi}.
\]

Thus:

\[
v^2-u^3+u^2=C.
\]

Define:

\[
\boxed{I(u,v)=v^2-u^3+u^2}.
\]

It should be constant along an exact trajectory. At closest approach v0=0 and u0=1/r0:

\[
I_0=u_0^2-u_0^3.
\]

This invariant is a useful independent diagnostic and possible reward term. It does not replace the ODE residual or direct trajectory comparison.

---

# PART III — Make the numerical physics trustworthy

## 11. Understand the output of solve_ivp

The existing simulator calls SciPy's solve_ivp. It returns a structured result containing, among other fields:

| Field | Meaning |
|---|---|
| t | independent-variable sample points, here phi |
| y | state values; row 0 is u and row 1 is v |
| success | whether the solver completed its requested integration successfully |
| message | explanation of solver completion or failure |
| t_events | event coordinates for each event function |
| y_events | state at each event |
| nfev | number of right-hand-side evaluations |
| sol | continuous interpolant when dense_output=True |

Important distinctions:
- solver success is not the same thing as physical escape;
- a solver may reach max_angle successfully without an escape event;
- escaped must be based on the event list;
- escape_angle must come from the event time;
- nfev measures RHS work, not wall-clock runtime.

For the existing result object, escaped is derived from the event list. Preserve that behavior.

## 12. DOP853 as the reference solver

The project uses DOP853 with tight tolerances and a maximum angular step. It is a high-order explicit Runge–Kutta method suitable for smooth non-stiff trajectories.

Typical current settings:

~~~python
solution = solve_ivp(
    _rhs,
    (0.0, max_angle),
    y0=(u0, 0.0),
    method="DOP853",
    rtol=1e-9,
    atol=1e-11,
    events=_escape_event,
    max_step=0.02,
)
~~~

For a dense reference dataset, use dense_output=True and tighter tolerances after checking computational cost:

~~~python
solution = solve_ivp(
    _rhs,
    (0.0, max_angle),
    y0=(u0, 0.0),
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
    events=_escape_event,
    max_step=0.01,
    dense_output=True,
)
~~~

DOP853 is not an analytical solution. It is a numerical reference. Repeat selected cases with tighter tolerances and a smaller max_step to check that the reference itself is sufficiently stable.

## 13. Protect the existing geodesic implementation with tests

Create tests/test_geodesics.py:

~~~python
import numpy as np
import pytest

from lab.geodesics import simulate_ray


def test_normal_scattering_ray_escapes():
    ray = simulate_ray(closest_approach=5.0)
    assert ray.escaped
    assert ray.escape_angle is not None
    assert ray.escape_angle > 0.0
    assert ray.nfev > 0


def test_escape_angle_and_deflection_are_consistent():
    ray = simulate_ray(closest_approach=5.0)
    assert ray.deflection_radian is not None
    assert np.isclose(
        ray.deflection_radian,
        2.0 * ray.escape_angle - np.pi,
    )


@pytest.mark.parametrize("r0", [1.0, 1.5, 0.0, -2.0, np.inf, np.nan])
def test_invalid_closest_approach_is_rejected(r0):
    with pytest.raises(ValueError):
        simulate_ray(closest_approach=r0)


def test_farther_ray_has_weaker_bending_for_selected_cases():
    near = simulate_ray(closest_approach=3.0)
    far = simulate_ray(closest_approach=10.0)
    assert near.deflection_radian > far.deflection_radian
~~~

The last assertion protects an expected trend for selected test cases. It is not a proof for every possible parameter.

Also test a deliberately short max_angle. It should complete without escape, with escaped=False and no deflection value. Never treat the final integration sample as an escape angle when no event was detected.

## 14. Stop simulations from running at import time

A module should define functions, classes, and constants when imported. It should not unexpectedly run a simulation or create a plot.

If geodesics.py contains a demonstration loop at module scope, move it into main():

~~~python
def main():
    for distance in (3.0, 5.0, 10.0):
        ray = simulate_ray(closest_approach=distance)
        print(
            distance,
            ray.escaped,
            ray.escape_angle,
            ray.deflection_radian,
        )


if __name__ == "__main__":
    main()
~~~

The import guard ensures that the demo runs only when the module is executed directly.

## 15. Validate RK4 independently of black-hole physics

The existing generic integrator is classical RK4. Validate it on a known solution before trusting it on geodesics.

Use:

\[
y'=y,\qquad y(0)=1,\qquad y(t)=e^t.
\]

Create tests/test_integrators.py:

~~~python
import numpy as np
import pytest

from lab.integrators import rk4_step


def exponential_rhs(t, state):
    return state


def test_one_step_matches_exponential():
    h = 0.1
    actual = rk4_step(exponential_rhs, 0.0, np.array([1.0]), h)
    expected = np.array([np.exp(h)])
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9)


def test_rk4_does_not_mutate_input():
    state = np.array([1.0, 2.0])
    before = state.copy()
    rk4_step(exponential_rhs, 0.0, state, 0.01)
    assert np.array_equal(state, before)


def test_one_step_uses_four_rhs_calls():
    calls = []

    def counted_rhs(t, state):
        calls.append(t)
        return state

    rk4_step(counted_rhs, 0.0, np.array([1.0]), 0.1)
    assert len(calls) == 4


@pytest.mark.parametrize("h", [0.0, -0.1, np.inf, np.nan])
def test_invalid_step_is_rejected(h):
    with pytest.raises(ValueError):
        rk4_step(exponential_rhs, 0.0, np.array([1.0]), h)


def test_state_must_be_one_dimensional():
    with pytest.raises(ValueError):
        rk4_step(exponential_rhs, 0.0, np.ones((2, 2)), 0.1)
~~~

Classical RK4 uses:

\[
k_1=f(t,y),
\]

\[
k_2=f(t+h/2,y+hk_1/2),
\]

\[
k_3=f(t+h/2,y+hk_2/2),
\]

\[
k_4=f(t+h,y+hk_3),
\]

\[
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\]

The integrator must remain generic. Do not hardcode the Schwarzschild equation into rk4_step.

### Test fourth-order convergence

Integrate the exponential equation to a fixed final time T with h, h/2, h/4. The global RK4 error should approach O(h^4).

\[
p_{\rm observed}=
\log_2\left(\frac{E(h)}{E(h/2)}\right).
\]

The observed order should approach four in the asymptotic regime. Coarse steps and very small steps can deviate.

## 16. Weak-field validation

Closest-approach radius and impact parameter are not the same quantity.

For a Schwarzschild turning point, the dimensionless impact parameter is:

\[
\bar b=\frac{b}{r_s}
=\frac{r_0}{\sqrt{1-1/r_0}}.
\]

In the weak-field regime, the leading-order deflection is:

\[
\alpha\approx\frac{2}{\bar b}.
\]

This is an approximation, not an exact formula. Its relative error should generally shrink when the ray passes farther from the black hole.

Create scripts/validate_weak_field.py:

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
        weak = 2.0 / b_bar

        if ray.escaped and ray.deflection_radian is not None:
            numerical = ray.deflection_radian
            relative_error = abs(numerical - weak) / abs(numerical)
            status = "escaped"
        else:
            numerical = ""
            relative_error = ""
            status = "unresolved"

        rows.append({
            "r_min_rs": r0,
            "impact_parameter_rs": b_bar,
            "numerical_rad": numerical,
            "weak_field_rad": weak,
            "relative_error": relative_error,
            "status": status,
        })

    output = Path("outputs/weak_field.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["status"] == "escaped"]
    plt.loglog(
        [row["impact_parameter_rs"] for row in valid],
        [row["numerical_rad"] for row in valid],
        "o-",
        label="DOP853",
    )
    plt.loglog(
        [row["impact_parameter_rs"] for row in valid],
        [row["weak_field_rad"] for row in valid],
        "--",
        label="Leading weak-field approximation",
    )
    plt.xlabel("Impact parameter b / r_s")
    plt.ylabel("Deflection (radians)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/weak_field.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
~~~

Use the actual property name deflection_radian from the current API. Older notes may contain deflection_radians; that name does not exist in the inspected current class.

## 17. Critical photon-sphere sweep

The deflection grows rapidly as r0 approaches 1.5 from above. This is the strong-bending regime and is especially important for later generalization tests.

Use offsets:

~~~python
offsets = [10.0 ** (-power) for power in range(1, 6)]
r0_values = [1.5 + offset for offset in offsets]
~~~

Integrate with a larger max_angle, such as 200, and a sufficiently small maximum step. Record:
- r0;
- offset from 1.5;
- escaped status;
- escape angle;
- deflection;
- nfev;
- solver status;
- maximum angle reached.

If the solver reaches max_angle without the escape event, label the case **unresolved_at_limit**. Do not call it captured. A long-lived near-critical scattering ray may simply need a larger angular interval.

Keep unresolved cases in the CSV with empty target values and explicit status metadata. Do not silently drop hard cases.

---

# PART IV — Build a reliable reference trajectory dataset

## 18. Why we need full curves, not only deflection labels

A scalar dataset of (r0, deflection_angle) can train a model to predict only the final angle. That is not our main task.

The RL agent must generate the full path. Each reference example therefore needs:
- closest approach r0;
- initial u0;
- angular coordinates phi;
- inverse-radius values u;
- derivatives v;
- escape angle;
- deflection angle;
- solver tolerances and status.

The existing RayTrajectory contains phi and u but not the full two-component state. For the dataset, use the raw solve_ivp result or a dense reference object that preserves both u and v. Do not estimate v by finite-differencing a sparse plotted curve when the solver already computes it.

## 19. Dense reference object

Create src/lab/reference.py:

~~~python
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from lab.geodesics import _rhs, _escape_event


@dataclass(frozen=True)
class ReferenceRay:
    solution: object
    closest_approach: float

    @property
    def escaped(self):
        return len(self.solution.t_events[0]) > 0

    @property
    def escape_angle(self):
        events = self.solution.t_events[0]
        return float(events[0]) if len(events) else None

    @property
    def rhs_evaluations(self):
        return int(self.solution.nfev)

    def state_at(self, phi):
        if self.solution.sol is None:
            raise RuntimeError("Reference solver was not run with dense_output=True")
        if not self.solution.t[0] <= phi <= self.solution.t[-1]:
            raise ValueError("phi is outside the integrated interval")
        return np.asarray(self.solution.sol(phi), dtype=np.float64)


def build_reference(
    closest_approach,
    *,
    max_angle=30.0,
    rtol=1e-12,
    atol=1e-14,
    max_step=0.01,
):
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")
    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive")

    result = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(1.0 / closest_approach, 0.0),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=_escape_event,
        dense_output=True,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return ReferenceRay(result, float(closest_approach))
~~~

This uses the current private physics functions to ensure the reference and existing simulator solve the same equation. A later refactor can expose a stable public physics API. Never query dense output after the event or beyond the actual integration interval.

## 20. Dataset sampling strategy

Start with a modest range of scattering rays. A useful initial domain is r0 from 3 to 10, then expand:
- farther rays provide weak bending;
- intermediate rays provide moderate bending;
- near-critical rays provide strong bending and stress-test generalization.

Use a mix of broad coverage and denser sampling near the photon sphere. Geometric offsets are useful:

~~~python
offsets = np.geomspace(0.03, 30.0, 180)
r0_values = 1.5 + offsets
~~~

This covers a wide range of distances from the critical value. However, some near-critical cases may not escape before max_angle. Keep their status rather than fabricating labels.

## 21. Store variable-length trajectories safely

Near-critical rays can require much longer angular ranges than weakly bent rays. Avoid forcing every trajectory into an object array that requires unsafe pickle loading.

A robust packed format stores:
- one flat phi array;
- one flat u array;
- one flat v array;
- an offsets array marking where each trajectory begins and ends;
- a case table containing r0, escape angle, deflection, status, and solver settings.

For example, trajectory i occupies:

\[
[\mathrm{offsets}[i],\ \mathrm{offsets}[i+1]).
\]

Save numeric arrays with NumPy NPZ and metadata in JSON. Alternatively, store one NPZ per trajectory plus a manifest. Either way, the loader must validate shapes, finite values, and offsets.

## 22. Split by trajectory, never by adjacent points

The train/validation/test split must be grouped by initial condition r0.

Do not randomly split all points from all curves. Adjacent points from the same trajectory are highly correlated; putting neighboring points in train and test creates severe leakage.

Create disjoint sets:
- training r0 values;
- validation r0 values;
- held-out test r0 values.

Only training trajectories determine normalization means, standard deviations, action bounds, and reward scales. Validation tunes the model and reward. Test is reserved for final evaluation.

Save all case IDs, r0 values, seeds, and solver settings in a dataset manifest.

## 23. Dataset quality gates

Before using data for training:
- every escaped trajectory has a valid escape event;
- no unresolved trajectory receives a fabricated deflection label;
- u and v have the same sample count;
- phi is strictly increasing;
- all values before the event are finite;
- no sample is taken beyond the reference interval;
- train/validation/test case IDs are disjoint;
- units and angle conventions are documented;
- selected curves agree with the existing simulator.

Create:
- src/lab/trajectory_data.py
- scripts/generate_trajectories.py
- scripts/inspect_trajectories.py
- tests/test_trajectory_data.py

---

# PART V — Baselines before RL

## 24. Numerical baseline

DOP853 is our high-accuracy reference. RK4 remains a numerical baseline and is useful for verifying convergence and discretization effects.

If a fixed-step RK4 convergence study is run, compare at common angular coordinates. Never compare arrays by index when they have different sampling grids.

For each h, record:
- number of steps;
- RHS evaluations;
- maximum u error;
- maximum v error;
- escape-angle error;
- deflection error;
- wall-clock runtime over repeated trials.

Classical RK4 has global error O(h^4) under the usual smooth, stable assumptions. Halving h should approach a factor of 16 error reduction in the asymptotic regime.

This validation is important for the quality of our references and plotting, but it is not the RL research question.

## 25. Weak-field baseline

The leading weak-field estimate:

\[
\alpha_{\rm weak}\approx\frac{2}{\bar b}
\]

is a transparent physical baseline. It should perform well far from the photon sphere and degrade in the strong-field regime. That limitation is expected, not a bug to hide.

## 26. Scalar deflection predictor (optional supporting baseline)

A simple model can predict the final deflection angle from r0 or impact parameter. This is not the target RL task because it does not generate a curve. It is useful for checking whether the dataset contains a learnable scalar relationship and for visualizing how much harder full trajectory generation is.

For scalar regression:
- input: r0 or dimensionless impact parameter;
- target: deflection in radians;
- metrics: MAE, RMSE, residual plots;
- train/validation/test split: grouped by ray case;
- normalization: fit on training only.

Do not present scalar deflection prediction as the main deliverable.

## 27. Supervised trajectory model

A more relevant baseline is a model that maps:

\[
(\phi,r_0)\longrightarrow(u,v).
\]

It predicts the trajectory directly instead of generating a sequential rollout. This is a useful supervised baseline because the same DOP853 trajectory data supplies labels.

A simple PyTorch model:

~~~python
import torch
from torch import nn


class TrajectoryMLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2),
        )

    def forward(self, phi_and_r0):
        return self.network(phi_and_r0)
~~~

Inputs should be normalized using training-set statistics. Targets u and v should also be scaled. Train with an MSE or weighted state loss, use validation early stopping, and evaluate only after checkpoint selection.

The held-out test set must contain r0 values absent from training, not just randomly held-out adjacent points.

This model is a fair and important comparison:
- it has direct access to labelled reference pairs during training;
- it predicts the state at an arbitrary phi;
- it does not accumulate sequential rollout errors in the same way as the RL policy.

The comparison will reveal whether the sequential reward-driven task offers any advantage beyond supervised interpolation.

## 28. Physics-Informed Neural Network

A PINN approximates the trajectory as a continuous function:

\[
u_\theta(\phi,r_0).
\]

Autograd computes derivatives with respect to phi. The physics residual is:

\[
\mathcal R_\theta=
\frac{\partial^2u_\theta}{\partial\phi^2}
+u_\theta-\frac32u_\theta^2.
\]

The ODE residual loss is:

\[
L_f=\mathbb E[\mathcal R_\theta^2].
\]

The initial-condition losses are:

\[
L_0=\mathbb E\left[
\left(u_\theta(0,r_0)-\frac{1}{r_0}\right)^2
\right],
\]

\[
L_1=\mathbb E\left[
\left(\frac{\partial u_\theta}{\partial\phi}(0,r_0)\right)^2
\right].
\]

The total loss:

\[
L=L_f+\lambda_0L_0+\lambda_1L_1.
\]

The ODE alone does not determine a unique trajectory. Initial conditions are essential.

### Autograd sanity check

Before implementing the PINN, verify second derivatives:

~~~python
import torch

x = torch.tensor([[0.2], [0.5], [0.8]], requires_grad=True)
y = x**3
dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
d2y = torch.autograd.grad(dy.sum(), x)[0]

expected = 6.0 * x.detach()
assert torch.allclose(d2y, expected, atol=1e-5)
~~~

For f(x)=x³, f''(x)=6x. create_graph=True preserves the derivative graph so that the first derivative can itself be differentiated.

### PINN model skeleton

~~~python
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

    def forward(self, phi, r0):
        inputs = torch.cat((phi, r0), dim=1)
        return self.network(inputs)


def derivatives(model, phi, r0):
    phi = phi.detach().clone().requires_grad_(True)
    u = model(phi, r0)
    du = torch.autograd.grad(
        u.sum(), phi, create_graph=True
    )[0]
    d2u = torch.autograd.grad(
        du.sum(), phi, create_graph=True
    )[0]
    return u, du, d2u
~~~

A complete PINN training loop must sample collocation points in the valid domain, sample r0 values, enforce initial conditions, calculate the residual, and validate against held-out DOP853 trajectories. Do not claim success based only on a low training residual; compare predicted curves and deflection angles.

The PINN is a later comparison branch, not a second main project. Do not build it before the RL environment and supervised baseline are functioning.

---

# PART VI — Reinforcement learning: the main project

## 29. Define the MDP precisely

A Markov Decision Process consists of:
- state/observation;
- action;
- transition;
- reward;
- episode termination;
- policy.

For this project, one episode is one scattering ray.

**Environment state:** current u, v, phi, initial u0 or r0, previous generated-state information, and the selected reference case.

**Policy observation:** a normalized subset of the environment state. It must not include future reference states.

**Action:** proposed next u and v, in normalized continuous coordinates.

**Transition:** the environment advances phi by the fixed delta_phi and applies the proposed state if valid.

**Reward:** a combination of reference tracking, ODE residual, conserved-quantity drift, and terminal behavior.

**Episode end:** valid outward escape, invalid state, or a configured horizon.

This is sequential because every chosen state becomes the current state for the next action. An incorrect action can move the rollout away from the physical curve, making later decisions harder.

## 30. Observation design

A first observation vector is:

\[
o_t=[
u_0,\ u_t,\ v_t,\ \phi_t/\phi_{\max},\
u_t/u_0,\ e_{t-1}
].
\]

Meaning:
- u0 identifies the ray's initial closest approach;
- ut, vt describe the current state;
- normalized phi describes progress;
- u/u0 measures progress from closest approach;
- e_(t-1) stores the previous transition error.

An optional later feature is previous action or recent tracking-error trend. Add features only through ablations.

Normalization:
1. calculate means and scales on training trajectories;
2. handle zero-variance columns explicitly;
3. save the normalization configuration;
4. apply it identically in validation and test;
5. never use held-out data to fit normalization.

## 31. Action design

Use a two-dimensional continuous action:

\[
a_t=(a_u,a_v),\qquad a_u,a_v\in[-1,1].
\]

Map to normalized next-state ranges:

\[
u_{t+1}=u_{\min}+
\frac{a_u+1}{2}(u_{\max}-u_{\min}),
\]

\[
v_{t+1}=v_{\min}+
\frac{a_v+1}{2}(v_{\max}-v_{\min}).
\]

Estimate ranges from training data with a safety margin. Freeze them before validation. An alternative experiment can predict state increments rather than absolute values, but do not mix action parameterizations in one comparison.

The environment must reject non-finite values and define what happens for physically invalid proposals. Do not silently clip arbitrary states into valid ones, because that can hide a poor policy.

## 32. Reward design

Use multiple independently logged components.

### 32.1 Reference tracking

For training only, obtain the DOP853 state at the next angular coordinate:

\[
s_{\rm ref}(\phi_{t+1})=
[u_{\rm ref}(\phi_{t+1}),v_{\rm ref}(\phi_{t+1})].
\]

Normalized tracking error:

\[
E_{\rm ref}=
\left\|
\frac{
s_{t+1}-s_{\rm ref}(\phi_{t+1})
}{
\sigma_s
}
\right\|_2^2.
\]

This is a teacher signal. Do not put future reference values in the observation.

### 32.2 ODE consistency

For h=delta_phi, define a midpoint residual:

\[
R_u=(u_{t+1}-u_t)
-h\frac{v_t+v_{t+1}}{2},
\]

\[
u_m=\frac{u_t+u_{t+1}}{2},
\]

\[
R_v=(v_{t+1}-v_t)
-h\left(\frac32u_m^2-u_m\right).
\]

Then:

\[
E_{\rm ODE}=
(R_u/\sigma_u)^2+(R_v/\sigma_v)^2.
\]

This is a discrete consistency residual. It is not exactly zero at finite h, even on a correct numerical trajectory.

### 32.3 Invariant drift

\[
I_t=v_t^2-u_t^3+u_t^2,
\qquad I_0=u_0^2-u_0^3.
\]

\[
E_I=(I_t-I_0)^2/\sigma_I^2.
\]

This measures violation of the conserved quantity.

### 32.4 Combined reward

A starting formula:

\[
r_t=
-\lambda_{\rm ref}E_{\rm ref}
-\lambda_{\rm ODE}E_{\rm ODE}
-\lambda_IE_I.
\]

Add a terminal success bonus for a valid escape. Add penalties for non-finite states, invalid outward/inward direction, or a horizon ending without a valid outcome.

Do not choose weights by intuition alone. First plot the distributions of each reward component on valid reference transitions and deliberately perturbed transitions. Normalize them, then tune weights on validation cases.

### 32.5 Reward ablations

Run separate experiments:
1. reference tracking + ODE residual + invariant;
2. ODE residual + invariant only;
3. reference tracking only;
4. ODE residual only.

This helps answer whether the policy follows the reference because it imitates the teacher or because physics constraints guide it.

### 32.6 Reward hacking checklist

Ask:
- Can the policy set u<=0 immediately to collect an escape bonus?
- Can it remain at the same state forever and collect small per-step penalties?
- Can it jump over the reference trajectory and receive a high reward?
- Can NaN values bypass comparisons?
- Can the agent terminate before generating a meaningful curve?
- Does every failed attempt have a finite, well-defined penalty?
- Is the success bonus large enough to distinguish valid completion but not so large that it hides a terrible path?

Use explicit terminal conditions. A valid escape must have a physically sensible outward crossing, accumulated path quality, and an escape angle consistent with the selected trajectory.

---

# PART VII — Gymnasium environment

## 33. What Gymnasium provides

Gymnasium standardizes the interaction between the policy and environment.

The essential methods:
- __init__: configure action and observation spaces;
- reset(seed=None, options=None): initialize an episode and return observation plus info;
- step(action): return observation, reward, terminated, truncated, info.

terminated=True means the task reached a terminal state such as valid escape or numerical failure.

truncated=True means an external limit ended the episode, such as maximum angular range or decision count.

Do not confuse solver success with escape. The environment must retain explicit status.

Install dependencies only when this milestone begins:

~~~bash
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

## 34. Environment file layout

Create src/lab/trajectory_env.py. It should depend on:
- a training-trajectory dataset loader;
- normalization and action conversion utilities;
- reward component functions;
- terminal/escape checks.

Keep the current physical ODE in geodesics.py. Do not duplicate the formula in several slightly different modules.

## 35. Environment blueprint

The following is a structural outline, not a complete implementation. It deliberately leaves the dataset adapter and reward calculation for the milestone where those pieces have been tested.

~~~python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PhotonTrajectoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        training_cases,
        delta_phi,
        observation_low,
        observation_high,
        action_u_range,
        action_v_range,
        max_decisions=20000,
    ):
        super().__init__()

        self.training_cases = training_cases
        self.delta_phi = float(delta_phi)
        self.action_u_range = tuple(action_u_range)
        self.action_v_range = tuple(action_v_range)
        self.max_decisions = int(max_decisions)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray(observation_low, dtype=np.float32),
            high=np.asarray(observation_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.phi = 0.0
        self.r0 = None
        self.case = None
        self.previous_error = 0.0
        self.decision_count = 0
        self.generated_states = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Select only from training cases, using the seeded RNG.
        index = int(self.np_random.integers(len(self.training_cases)))
        self.case = self.training_cases[index]
        self.r0 = float(self.case["r0"])

        self.state = np.array([1.0 / self.r0, 0.0], dtype=np.float64)
        self.phi = 0.0
        self.previous_error = 0.0
        self.decision_count = 0
        self.generated_states = [(self.phi, self.state.copy())]

        obs = self._observation()
        if not self.observation_space.contains(obs):
            raise RuntimeError("Reset observation is outside observation_space")

        return obs, {"r0": self.r0}

    def _observation(self):
        u, v = self.state
        # Apply the frozen training-set normalizer here.
        return self.normalizer.transform(
            [self.case["u0"], u, v, self.phi, self.previous_error]
        ).astype(np.float32)

    def step(self, action):
        raise NotImplementedError(
            "Implement action mapping, reward, and terminal checks after tests."
        )
~~~

Before using this outline:
- implement and test self.normalizer;
- validate training_cases is non-empty;
- check all actions are finite;
- verify observation bounds against actual rollouts;
- implement next-state action conversion;
- implement reward terms separately;
- implement valid escape and horizon conditions.

The reference state may be queried inside reward calculation, but it must not be copied into the policy observation.

## 36. Transition logic

At each step:
1. validate action shape and finiteness;
2. clip only to the declared action bounds;
3. map action coordinates to proposed u_next and v_next;
4. set phi_next=phi+delta_phi;
5. query the training reference at phi_next if inside its valid interval;
6. calculate E_ref, E_ODE, and E_I;
7. check numerical and physical validity;
8. calculate finite reward;
9. store the point in the generated path;
10. update state only according to the documented transition;
11. detect escape, failure, or truncation;
12. return the five Gymnasium outputs.

The environment's info dictionary should expose:
- r0;
- phi;
- proposed u and v;
- reference u and v used for reward;
- each reward component;
- accumulated reward;
- valid/invalid status;
- termination reason;
- escape angle, where defined.

Rejected or invalid proposals still consume a decision and must receive a penalty. Do not let the agent avoid cost by retrying for free.

## 37. Environment tests and checker

Test:
- reset reproducibility under a fixed seed;
- initial state equals [1/r0, 0];
- observation belongs to observation_space;
- action shape and range are handled correctly;
- non-finite actions/states never receive success reward;
- phi advances by exactly delta_phi;
- generated points are stored in order;
- a better reference-tracking proposal receives a better tracking term;
- ODE residual is smaller for a physically consistent transition;
- invariant error uses the correct I0;
- escape requires a valid outward crossing;
- horizon exhaustion is truncation, not success;
- reset works after a terminal episode.

Use the Gymnasium environment checker before training. Do not start PPO against an incomplete environment.

---

# PART VIII — Reinforcement learning concepts and PPO

## 38. From supervised learning to RL

In supervised learning, a model sees labelled pairs and reduces prediction loss.

In RL, the agent interacts with an environment:
- it observes a state;
- selects an action;
- receives a reward;
- transitions to a new state;
- updates its policy to improve expected cumulative reward.

Our policy outputs two continuous action values representing the next normalized u and v.

## 39. Policy, actor, and critic

A policy is written:

\[
\pi_\theta(a\mid s).
\]

It describes the action distribution for observation s.

PPO commonly uses:
- **Actor:** produces the action distribution.
- **Critic:** estimates expected future return V(s).

For continuous actions, the actor often outputs the parameters of a Gaussian distribution. During training actions are sampled; evaluation often uses the deterministic mean action.

## 40. Return and discounting

The discounted return is:

\[
G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}.
\]

Gamma controls how much future rewards matter. A trajectory needs long-term consistency: a locally good point can lead to a poor later curve. This is why cumulative return matters.

## 41. Advantage and GAE

Advantage estimates whether an action did better or worse than expected:

\[
A_t=Q(s_t,a_t)-V(s_t).
\]

Generalized Advantage Estimation combines multi-step temporal-difference residuals:

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\]

\[
\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
\]

Lambda controls the bias/variance trade-off.

## 42. PPO clipped objective

Define the probability ratio:

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\rm old}}(a_t\mid s_t)}.
\]

The clipped objective is:

\[
L^{\rm CLIP}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

Clipping discourages excessively large policy updates. You do not need to implement PPO from scratch initially, but you should understand what the library optimizes.

## 43. First training script

After the environment passes tests, create scripts/train_rl.py.

A first PPO smoke test using Stable-Baselines3:

~~~python
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from lab.trajectory_env import make_training_env


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "rl_logs"
MODEL_DIR = ROOT / "outputs" / "models"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = make_training_env(seed=7)
    check_env(env)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        seed=7,
    )

    # Smoke test only. It is not a scientific training budget.
    model.learn(total_timesteps=5000)
    model.save(MODEL_DIR / "ppo_photon_trajectory")


if __name__ == "__main__":
    main()
~~~

The factory make_training_env is a project function to implement; this script is not runnable until the dataset loader and environment are complete.

Run TensorBoard:

~~~bash
tensorboard --logdir outputs/rl_logs
~~~

A 5,000-step run verifies basic execution only. Inspect:
- episode reward;
- each reward component separately;
- episode length;
- valid escape rate;
- invalid-state rate;
- reference tracking error;
- ODE residual;
- invariant drift;
- rendered generated curves.

If reward rises while actual curve error becomes worse, the reward is misweighted or exploitable. Stop and diagnose; do not merely train longer.

## 44. PPO is an algorithm, not the project goal

The research objective is RL trajectory generation. PPO is the first algorithm because it supports continuous bounded actions and has a mature implementation. Later, compare an off-policy method such as SAC if the environment is stable and the comparison is justified.

Do not train several RL algorithms before the environment is validated. A different algorithm cannot repair incorrect reward or transition logic.

---

# PART IX — Evaluation and controlled experiments

## 45. Held-out initial conditions

Split by complete ray cases, not by individual trajectory points.

- **Train:** used for policy updates and training-only normalization.
- **Validation:** used for reward weights, model selection, and hyperparameters.
- **Test:** used only for final evaluation.

Include weak, moderate, and strong deflection cases. Test unseen r0 values inside the training range, near its boundaries, and eventually closer to the photon sphere after the training domain has been expanded safely.

Near-critical values can require very long trajectories. Unresolved integration at max_angle is not proof of capture.

## 46. Metrics

For every held-out trajectory, record:

| Metric | Definition |
|---|---|
| u error | Difference in inverse radius on a common angular grid |
| v error | Difference in derivative state |
| Cartesian curve error | Difference in x/y after consistent masking |
| escape-angle error | Absolute difference in escape angle |
| deflection error | Absolute difference in bending angle |
| ODE residual | Local differential-equation consistency |
| invariant drift | Change in the conserved quantity |
| escape success | Whether a valid outward escape was generated |
| rollout length | Number of agent decisions |
| runtime | End-to-end evaluation time |

Use the same angular grid or interpolate over the common valid interval. Never compare arrays of different lengths by their indices.

Report mean, median, standard deviation, and worst-case errors where appropriate. Keep raw per-ray outputs. Do not hide failed rays by reporting only successful cases.

## 47. Visual evaluation

For each test case:
- draw the DOP853 reference;
- draw the generated RL path;
- use identical Cartesian axes and radius masks;
- draw the event horizon;
- include r0 and deflection in the title/legend;
- show an error plot below or beside the curve.

A visually convincing curve can still have a poor escape angle. Visual and numerical evaluation are both necessary.

## 48. Generalization and multiple seeds

Run multiple random seeds. Do not publish only the best run.

Generalization should test:
- unseen r0 values within the training range;
- cases near the boundaries of that range;
- near-critical scattering rays after extending the training data;
- a broader domain only after the model works on the initial domain.

Do not treat captured trajectories as part of the first experiment. They need a distinct incoming-ray initialization and terminal condition.

## 49. Ablation experiments

Change one design choice at a time:
- teacher-tracking reward versus physics-only reward;
- with versus without invariant penalty;
- with versus without initial u0 in the observation;
- absolute-state action versus increment action;
- different fixed angular increments;
- PPO versus SAC;
- RL versus supervised MLP;
- PINN versus data-only MLP.

Ablations explain what the policy learned. Do not change multiple ingredients at once and attribute the final difference to one of them.

## 50. Fair comparison

Compare:
1. DOP853 reference;
2. fixed-step RK4 at documented h;
3. supervised trajectory MLP;
4. RL sequential trajectory generator;
5. PINN, once implemented.

Use identical held-out r0 cases and the same evaluation metrics. DOP853 is the reference for measuring error; it can also be reported as a numerical method, but be clear about which role it plays.

The supervised MLP is especially important. If it predicts full curves more accurately and efficiently than RL, the result tells us that the deterministic mapping is better suited to supervised learning. That is not a failure of the project.

---

# PART X — Optional research extensions

## 51. Captured photon trajectories

After scattering-ray generation works, extend the simulator to incoming rays initialized at a large radius using impact parameter. Define and test the incoming boundary conditions carefully. Add separate capture and escape termination conditions.

Do not classify a trajectory as captured just because it did not escape before a time limit.

## 52. PINN comparison

The PINN section above can be expanded into a full implementation after the RL and supervised baselines exist. Evaluate its ODE residual and, more importantly, its held-out trajectory and deflection errors.

## 53. Adaptive sampling

A later controlled experiment can compare:
- adding training rays near large model errors;
- adding the same number of randomly selected rays.

Keep a final test set untouched. This is an active-learning/data-acquisition experiment, not part of the first RL milestone.

## 54. Adaptive numerical integration

The earlier step-size-control research can remain a numerical-analysis experiment:
- RK4 step-doubling;
- normalized local error;
- accepted/rejected counts;
- conventional h controller.

It is no longer the main RL objective. Do it only if it improves reference quality or provides a useful numerical baseline.

---

# PART XI — Engineering, environment, and reproducibility

## 55. Packaging detail in the current repository

The inspected repository currently contains a file named **project.toml**. Python packaging tools expect **pyproject.toml**. The earlier guides sometimes assumed the latter without confirming the rename. Before relying on editable installation, check the actual checkout and, if it is still project.toml, rename it:

~~~bash
git mv project.toml pyproject.toml
~~~

Do not keep two competing metadata files.

The existing metadata specifies Python 3.11+, NumPy, SciPy, Matplotlib, and pytest as a development dependency. The RL environment will later add Gymnasium, Stable-Baselines3, and TensorBoard. PyTorch, scikit-learn, and any data-handling library should be added only when their milestones begin.

## 56. Setup on Linux / WSL

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Windows PowerShell activation:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Run tests:

~~~bash
python -m pytest -q
~~~

Run scripts from repository root, using package imports. Avoid imports that work only because the current directory happens to be src/lab.

## 57. Test each layer independently

1. Physics RHS and initial conditions.
2. Escape event and status semantics.
3. DOP853 reference, dense output, and v values.
4. RK4 on a known analytic ODE.
5. Dataset shape, metadata, and split leakage.
6. Trajectory metrics and Cartesian plotting.
7. Neural-network forward passes and finite losses.
8. PINN derivative sanity checks.
9. RL environment spaces, seeding, transition logic, and reward.
10. End-to-end training and evaluation scripts.

A green unit-test suite does not prove the physics is correct. A low training loss does not prove generalization. A high episodic reward does not prove the curve is accurate.

## 58. Keep imports safe

Importing a module should not:
- run a simulation;
- train a model;
- write output files;
- open a plot window;
- start a training loop.

Use a main guard in scripts:

~~~python
def main():
    ...


if __name__ == "__main__":
    main()
~~~

## 59. Data and artifacts

Keep generated datasets, checkpoints, logs, and figures under data/ and outputs/. Normally gitignore:
- .venv/
- __pycache__/
- *.pyc
- generated raw datasets if large;
- transient training logs;
- temporary checkpoints.

Publish selected final figures, compact metrics, and model artifacts intentionally. Do not accidentally commit large caches or local environments.

## 60. Reproducibility manifest

Every dataset should save:
- schema version;
- equation and units;
- DOP853 method and tolerances;
- maximum angle and maximum step;
- exact r0 values;
- escaped/unresolved status;
- train/validation/test case IDs;
- generation seed.

Every training run should additionally save:
- Git commit;
- dataset manifest identifier;
- observation and action definitions;
- normalization scales;
- reward components and weights;
- delta_phi;
- algorithm and hyperparameters;
- random seed and training budget;
- checkpoint path;
- evaluation metrics.

Suggested result table:

~~~text
method,seed,case_id,r0,escaped,escape_angle,deflection_rad,max_u_error,max_v_error,curve_error,ode_residual,invariant_drift,rollout_steps,runtime_seconds
~~~

---

# PART XII — Final report and release

## 61. Suggested research report structure

1. **Abstract:** research question, method, and measured result.
2. **Introduction:** gravitational light bending and why trajectory generation is interesting.
3. **Physical model:** Schwarzschild assumptions, units, ODE, initial conditions, photon sphere.
4. **Numerical reference:** DOP853, event handling, tolerances, convergence and weak-field validation.
5. **Dataset:** sampling, split protocol, unresolved cases, normalization.
6. **RL formulation:** MDP, observation, action, reward, termination.
7. **Algorithms:** PPO and any comparison methods.
8. **Baselines:** RK4, supervised MLP, weak-field estimate, optional PINN.
9. **Evaluation:** curve error, state error, escape/deflection error, invariant drift, generalization.
10. **Ablations:** reward and observation decisions.
11. **Limitations:** teacher-guided reward, fixed angular grid, scattering-only domain, numerical reference uncertainty.
12. **Conclusion:** what worked, what did not, and what the evidence supports.

## 62. Final deliverables

- Tested Schwarzschild simulator.
- Reproducible dense DOP853 reference.
- Physics-validation CSVs and plots.
- Full-trajectory dataset with a frozen split manifest.
- Reference gallery of weak, moderate, and strong bending.
- Tested Gymnasium trajectory environment.
- Trained RL model and configuration.
- Supervised baseline and optional PINN comparison.
- Held-out generated curves and error plots.
- Final report with reproducible commands and limitations.

---

# PART XIII — Ordered roadmap and next actions

| Order | Milestone | Main deliverable | Completion gate |
|---:|---|---|---|
| 0 | Repository and solver inspection | Current code/API map | No stale assumptions about field names |
| 1 | Physics regression tests | tests/test_geodesics.py | Escape and deflection semantics protected |
| 2 | Dense DOP853 reference | reference.py | Arbitrary-angle u and v available |
| 3 | Physics validation | weak-field and critical sweeps | Weak/strong-field behavior measured |
| 4 | Trajectory dataset | trajectory_data.py and manifest | Disjoint ray-level splits |
| 5 | Reference visualization | multi-ray gallery | Weak, moderate, strong bending visible |
| 6 | Metrics | trajectory_metrics.py | State, curve, event, ODE, invariant scores |
| 7 | Supervised MLP baseline | model and evaluator | Held-out trajectory errors recorded |
| 8 | RL environment | trajectory_env.py | Fixed-angle sequential state generation passes tests |
| 9 | PPO smoke test | train_rl.py and logs | Finite rewards and meaningful rollouts |
| 10 | RL experiments | reward/observation ablations | Behavior and failure modes understood |
| 11 | Final evaluation | multi-seed held-out report | Fair comparison with baselines |
| 12 | PINN and extensions | optional branch | Added after core result is reproducible |

## Immediate next session — do this, not PPO

1. Inspect the current geodesics.py and run the existing simulator for r0 = 3, 5, 10.
2. Add tests for normal escape, no-escape time limit, invalid r0, and alpha = 2 phi_escape - pi.
3. Create a dense DOP853 reference exposing both u and v.
4. Generate a small reference gallery for several r0 values.
5. Numerically evaluate the conserved quantity I(u,v) along each ray.
6. Run weak-field validation and a small near-critical sweep.
7. Only then design and generate the multi-trajectory dataset.

The next major coding task is **reference trajectory data generation**. The RL environment comes later, after we have enough accurate curves to train and evaluate it.

---

# Learning resources

Use these resources to answer the next implementation question, not as a substitute for building and testing.

## Numerical methods and ODEs
- SciPy solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- SciPy integration tutorial: https://docs.scipy.org/doc/scipy/tutorial/integrate.html
- Hairer, Nørsett, Wanner, Solving Ordinary Differential Equations I.

## Reinforcement learning
- Sutton and Barto, Reinforcement Learning: An Introduction: http://incompleteideas.net/book/the-book-2nd.html
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

## Physics
Use a reliable general-relativity reference for Schwarzschild geometry, null geodesics, photon sphere, impact parameter, and weak-field deflection. Keep assumptions and units explicit in the report.

---

# Guide maintenance rules

- Update the project status when a milestone is actually complete.
- Never mark a milestone complete because code was merely written; tests and evidence are required.
- If the repository API differs from an example, update the example to match the source.
- Keep one primary research question: RL generation of photon trajectories.
- Keep scalar regression, PINN, adaptive integration, capture, and active sampling as comparisons or extensions—not competing main projects.
- Record failed experiments and unexpected results; do not erase them to make the project look successful.
- Every result in the final report must be reproducible from a script and saved configuration.

**Guiding principle:** the agent learns to generate the light path. Physics, numerical analysis, and machine learning all support that one objective.
