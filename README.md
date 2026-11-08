# Black Hole Lab — The Complete RL Photon-Trajectory Project

**One project. One main objective. One implementation path.**

> **Research question:** Can a reinforcement-learning agent learn to generate photon trajectories around a Schwarzschild black hole, reproduce gravitational light bending, and generalize to initial conditions it has never encountered?

The agent's job is to generate the light curve. It observes the current photon state and predicts the next point. It does **not** choose the numerical integrator's step size. The angular increment is fixed by the environment.

This textbook is organized around building the RL system first:

1. Generate a dataset of complete reference photon curves.
2. Build the sequential trajectory-generation environment.
3. Design the action, observation, and reward.
4. Implement and train the RL agent.
5. Render and inspect the curves it generates.
6. Add the complete test suite and scientific evaluation.
7. Only after the main RL system works, build supervised MLP, PINN, weak-field, and other comparison experiments.

The physics simulator already exists, so we will reuse it. We are not going to spend most of the project writing tests for infrastructure before the agent exists. We will still use small smoke checks while developing, but the systematic test suite is a later milestone.

---

# How to use this textbook

Each implementation chapter contains:

- **What we are building** — the concrete deliverable.
- **Why it exists** — the research or engineering reason.
- **How the data flows** — what enters a function and what comes out.
- **Implementation** — code that belongs to this milestone.
- **Code walkthrough** — why each important block exists.
- **How to run it** — commands and expected artifacts.
- **What to inspect** — what a successful result should look like.
- **Common mistakes** — what can make the result misleading.

Do not blindly copy an entire file before understanding its responsibility. Work through one file at a time, run it, inspect its output, then continue. The code examples are implementation references; they are not claims that future files have already been committed or executed.

You already know Python, Git, and basic programming. This is not a beginner Python course. The new learning is in numerical physics, trajectory representation, RL environment design, reward shaping, sequential policies, and scientific evaluation.

## Important scope decision

This project is **RL-based photon trajectory generation**.

- The numerical solver gives us accurate reference curves.
- The RL policy generates the next photon state.
- The environment evaluates the generated point.
- PPO is our first algorithm, not the research goal itself.
- MLP and PINN are later comparison methods, not parallel main projects.
- Adaptive step-size control is no longer the main RL task.

The first experiment focuses on **escaping scattering rays** with closest approach above the photon sphere. Captured rays require a different incoming-ray initialization and are a later extension.

---

# PART I — Understand the physics we already have

## 1. Current repository state

The inspected code contains:

| File | Existing responsibility |
|---|---|
| src/lab/geodesics.py | Schwarzschild photon ODE, escape event, DOP853 simulation, RayTrajectory |
| src/lab/integrators.py | Generic classical RK4 one-step integrator |
| src/lab/compare_plot.py | Early RK4/DOP853 comparison visualization |
| project.toml | Project metadata under its current filename |

The current physics module exposes the following names:

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

The RK4 function is:

~~~python
rk4_step(rhs, phi, state, step_size)
~~~

The state order is always:

~~~python
state = np.array([u, v])
~~~

where u is inverse radius and v is du/dphi. Do not reverse these components.

The current comparison script is a prototype. It is useful for visual intuition, but it does not yet localize the escape event accurately, compare every state component, or provide a complete convergence analysis. We will not make the RL project wait for a perfect numerical-analysis package. DOP853 already provides the reference solver needed to build our first dataset.

## 2. Schwarzschild orbit equation

We use normalized units where the Schwarzschild radius is:

\[
r_s=1.
\]

Define inverse radius:

\[
u(\phi)=\frac{1}{r(\phi)}.
\]

For the equatorial photon orbit in the current model:

\[
\frac{d^2u}{d\phi^2}+u=\frac{3}{2}u^2.
\]

Equivalently:

\[
u''=\frac{3}{2}u^2-u.
\]

Introduce:

\[
v=\frac{du}{d\phi}.
\]

Then the second-order ODE becomes the first-order system:

\[
\boxed{u'=v,\qquad v'=\frac{3}{2}u^2-u}.
\]

The state vector is:

\[
s(\phi)=
\begin{bmatrix}
u(\phi)\\
v(\phi)
\end{bmatrix}.
\]

At the closest-approach point r0, symmetry gives:

\[
u(0)=u_0=\frac{1}{r_0},\qquad v(0)=0.
\]

The photon sphere is at:

\[
r_{\rm ph}=1.5r_s.
\]

For the current scattering initialization, use r0 greater than 1.5. As r0 approaches 1.5 from above, the ray experiences increasingly strong bending and can spend a long angular interval near the photon sphere.

Do not treat r0 below 1.5 as a normal escaping ray using the current closest-approach initialization. Captured-ray experiments will require incoming initial conditions at a large radius and a separately defined capture event.

## 3. Escape and deflection

The outgoing scattering ray escapes when:

\[
u=0.
\]

That corresponds to radius tending to infinity. The existing event function returns u and is configured as terminal with direction -1, so it detects a positive-to-negative crossing.

The positive-side escape angle is:

\[
\phi_{\rm escape}.
\]

For the symmetric scattering trajectory, the total deflection is:

\[
\boxed{\alpha=2\phi_{\rm escape}-\pi}.
\]

The existing result property is named **deflection_radian**. Older guide snippets sometimes used different plural property names. Follow the actual code API.

## 4. Convert a trajectory into a picture

For every valid positive-u sample:

\[
r=\frac{1}{u},
\]

\[
x=r\cos\phi,\qquad y=r\sin\phi.
\]

The event horizon is a circle of radius one. The photon sphere is a circle of radius 1.5; they are not the same thing.

To draw the full symmetric scattering path, mirror the positive-side trajectory across the x-axis. This uses the symmetry of the current setup; the incoming half was not independently integrated.

At escape, u approaches zero and r approaches infinity. Plotting code must mask non-positive u and optionally hide points beyond a chosen plotting radius. Do not let infinity determine the graph's axes.

## 5. A useful conserved quantity

The system has a first integral. Since:

\[
v'=\frac{3}{2}u^2-u,\qquad u'=v,
\]

we can derive:

\[
2vv'=(3u^2-2u)u'.
\]

Integrating gives:

\[
v^2=u^3-u^2+C.
\]

Therefore:

\[
\boxed{I(u,v)=v^2-u^3+u^2}
\]

is constant along an exact trajectory.

At closest approach, v0=0 and u0=1/r0, so:

\[
I_0=u_0^2-u_0^3.
\]

This invariant becomes one of our physical diagnostics and one possible reward term. It helps us detect generated curves that drift away from the physical solution.

---

# PART II — Milestone 1: Generate the photon-trajectory dataset

<badge color="success">First implementation milestone</badge>

This is where we begin coding. Before an RL agent can learn a curve, we need many examples of what valid curves look like.

## 6. What the dataset must contain

A dataset containing only r0 and deflection angle is not enough. That would let us train a model that predicts one final number, but our agent must generate the whole light path.

Each trajectory case needs:

| Field | Meaning |
|---|---|
| case_id | Stable identifier for this ray |
| r0 | Closest-approach radius in Schwarzschild-radius units |
| u0 | Initial inverse radius, equal to 1/r0 |
| phi | Angular coordinates along the outgoing trajectory |
| u | Inverse radius at each angular coordinate |
| v | du/dphi at each angular coordinate |
| escape_angle | Positive-side angle at the escape event |
| deflection_radian | 2 times escape angle minus pi |
| status | Escaped or unresolved |
| solver settings | Method, tolerances, max step, max angle |

Why store both u and v? Because the RL policy's output is the next state [u,v]. If we only store u, we would have to estimate v numerically from the sampled curve. That would introduce an avoidable derivative error. SciPy already calculates v as part of the ODE state, so we should preserve it.

## 7. Choose the first training domain

Start with ordinary escaping rays, not the extremely difficult near-critical edge.

A practical first range is:

~~~python
r0_values = np.linspace(2.2, 12.0, 120)
~~~

This gives us weak, moderate, and relatively strong bending while staying away from the most difficult critical cases in the first dataset.

Later, extend toward:

~~~python
r0 = 1.5 + offset
~~~

with positive offsets distributed geometrically. Near-critical rays may require a much larger max_angle. A solver reaching max_angle without detecting escape is **unresolved at the limit**, not automatically captured.

Do not mix unresolved curves into the training set as though they were complete escaping trajectories. Keep their metadata so we can inspect them, but initially train the scattering policy on successfully escaped cases.

## 8. Why we need dense reference output

The existing simulate_ray function returns a RayTrajectory intended for visualization. For learning, we need the full state [u,v] at chosen angular coordinates.

SciPy's solve_ivp supports dense_output=True. This creates a continuous interpolating solution over the interval actually integrated.

We will wrap it in a small reference object. This object has one job: given an angle phi, return the reference u and v.

### File: src/lab/reference.py

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
    def deflection_radian(self):
        angle = self.escape_angle
        if angle is None:
            return None
        return 2.0 * angle - np.pi

    @property
    def rhs_evaluations(self):
        return int(self.solution.nfev)

    def state_at(self, phi):
        if self.solution.sol is None:
            raise RuntimeError("Reference was not created with dense_output=True")

        values = np.asarray(phi, dtype=float)

        if np.any(values < self.solution.t[0]) or np.any(values > self.solution.t[-1]):
            raise ValueError("Requested angle is outside the integrated interval")

        return np.asarray(self.solution.sol(values), dtype=np.float64)


def build_reference(
    closest_approach,
    *,
    max_angle=20.0,
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

    return ReferenceRay(
        solution=result,
        closest_approach=float(closest_approach),
    )
~~~

### Code walkthrough

**Imports**

- dataclass gives us a compact immutable container for a reference trajectory.
- NumPy handles scalar and array inputs consistently.
- solve_ivp performs the numerical integration.
- _rhs and _escape_event are imported from the existing physics module. This is important: we do not want a second copy of the Schwarzschild equation that could accidentally differ from the simulator.

**ReferenceRay**

The class stores the raw SciPy solution and the closest-approach parameter.

- escaped checks whether the first event list contains an event. Solver success alone is not the same as physical escape.
- escape_angle reads the actual event time, rather than guessing from the final sample.
- deflection_radian applies the symmetry formula.
- rhs_evaluations exposes the solver's work count.
- state_at calls the dense interpolant. It checks the requested angles first because dense output must not be used outside the interval the solver actually integrated.

**build_reference**

The function validates r0 and max_angle, constructs the initial state [1/r0, 0], and calls DOP853.

- rtol and atol control the solver's error tolerances.
- max_step limits the solver's angular step.
- events terminates integration at escape.
- dense_output=True makes arbitrary-angle state queries possible.

The return object is not an analytical solution. It is our high-accuracy numerical teacher. Later, we will check selected cases again with tighter tolerances to ensure reference error is negligible.

## 9. Resample each reference curve onto a shared angular grid

The solver's internal sample points are adaptive, so different curves contain different phi coordinates. A learning environment is easier to build if every ray uses a fixed angular increment.

Choose:

~~~python
delta_phi = 0.01
~~~

For each ray:
1. create angles 0, delta_phi, 2*delta_phi, and so on;
2. stop before passing the escape event;
3. append the exact escape angle as the final coordinate;
4. query dense reference output at all these angles.

The final interval may be shorter than delta_phi. That is intentional. It ensures the final reference point is exactly at the escape event.

Why not use a common fixed number of points for every ray? Because escape angles differ. We want the grid spacing to be comparable while allowing different trajectory lengths.

### File: src/lab/trajectory_data.py

~~~python
import json
from pathlib import Path

import numpy as np

from lab.reference import build_reference


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
~~~

### Code walkthrough

**sample_reference**

This function converts one adaptive DOP853 solution into a fixed-grid trajectory.

- It refuses invalid delta_phi.
- It refuses unresolved rays, because there is no valid escape endpoint to sample.
- np.arange creates the regular angular grid.
- The exact escape angle is appended as the last point.
- state_at returns a matrix with shape (2, number_of_angles). Transposing it gives one row per angle.
- Column zero is u; column one is v.

At the final point, u should be approximately zero and v should be negative for an outgoing scattering ray.

**generate_trajectory_dataset**

This function loops over the requested closest approaches.

- Each r0 receives a stable case_id during generation.
- build_reference solves the physical ray.
- Unresolved rays are recorded in the manifest instead of being mislabeled as valid.
- Complete curves are packed into flat arrays.
- offsets records where each trajectory begins and ends in those arrays.
- np.savez_compressed stores numeric arrays without Python object pickling.
- The JSON manifest records the physics, units, solver settings, grid increment, and unresolved cases.

**Why packed arrays?**

Trajectories have different lengths because their escape angles differ. Instead of forcing them into a rectangular array with fake padding, we concatenate them. The offsets array tells us which slice belongs to each ray.

For trajectory i:

~~~python
start = offsets[i]
end = offsets[i + 1]

phi_i = phi[start:end]
u_i = u[start:end]
v_i = v[start:end]
~~~

This is memory-efficient, simple to load, and safe with allow_pickle=False.

## 10. Dataset generation script

### File: scripts/generate_trajectories.py

~~~python
import argparse

import numpy as np

from lab.trajectory_data import generate_trajectory_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference Schwarzschild photon trajectories."
    )
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--r-min", type=float, default=2.2)
    parser.add_argument("--r-max", type=float, default=12.0)
    parser.add_argument("--delta-phi", type=float, default=0.01)
    parser.add_argument("--max-angle", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.count < 2:
        parser.error("--count must be at least 2")
    if args.r_min <= 1.5:
        parser.error("--r-min must be greater than 1.5")
    if args.r_max <= args.r_min:
        parser.error("--r-max must be greater than --r-min")

    # A fixed seed makes the parameter ordering reproducible.
    rng = np.random.default_rng(args.seed)
    r0_values = np.linspace(args.r_min, args.r_max, args.count)
    rng.shuffle(r0_values)

    manifest = generate_trajectory_dataset(
        r0_values,
        delta_phi=args.delta_phi,
        max_angle=args.max_angle,
    )

    print("Generated trajectories:", manifest["num_escaped"])
    print("Unresolved cases:", len(manifest["unresolved"]))
    print("Dataset:", manifest["data_file"])


if __name__ == "__main__":
    main()
~~~

### Explanation

- argparse exposes the important dataset parameters at the command line.
- count controls how many independent rays are generated.
- r-min and r-max define the first scattering domain.
- delta-phi is the angular spacing used when resampling reference curves.
- max-angle bounds how long the solver searches for an escape event.
- The seeded generator shuffles the r0 values so the packed file is not ordered only from weak to strong bending. The values themselves remain the same.
- generate_trajectory_dataset does the actual simulation and storage.
- The main guard prevents dataset generation when the module is imported.

Run from the repository root:

~~~bash
python -m scripts.generate_trajectories --count 120 --r-min 2.2 --r-max 12 --delta-phi 0.01
~~~

Expected files:

~~~text
data/
    photon_trajectories.npz
    trajectory_manifest.json
~~~

Do not move to RL before you have inspected at least several generated curves.

## 11. Inspect the dataset visually

The point of this script is not just to print array shapes. We want to see if the dataset actually covers different bending regimes.

### File: scripts/inspect_trajectories.py

~~~python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.load(
        "data/photon_trajectories.npz",
        allow_pickle=False,
    )

    r0_values = data["r0"]
    phi = data["phi"]
    u = data["u"]
    offsets = data["offsets"]
    escape_angles = data["escape_angle"]

    print("Number of trajectories:", len(r0_values))
    print("Packed samples:", len(phi))
    print("Average points per ray:", np.diff(offsets).mean())
    print("Closest approaches:", r0_values.min(), r0_values.max())

    figure, axis = plt.subplots(figsize=(9, 9))

    chosen = np.linspace(
        0,
        len(r0_values) - 1,
        min(8, len(r0_values)),
        dtype=int,
    )

    max_plot_radius = 30.0

    for index in chosen:
        start, end = offsets[index:index + 2]
        local_phi = phi[start:end]
        local_u = u[start:end]

        valid = local_u > 0
        radius = np.full_like(local_u, np.inf)
        np.divide(1.0, local_u, out=radius, where=valid)

        x = radius * np.cos(local_phi)
        y = radius * np.sin(local_phi)

        visible = valid & (radius <= max_plot_radius)

        axis.plot(
            x[visible],
            y[visible],
            label=f"r0={r0_values[index]:.2f}",
        )
        axis.plot(x[visible], -y[visible], alpha=0.7)

    axis.add_patch(plt.Circle((0, 0), 1.0, color="black", label="Event horizon"))
    axis.add_patch(
        plt.Circle(
            (0, 0),
            1.5,
            fill=False,
            linestyle=":",
            label="Photon sphere",
        )
    )

    axis.set_aspect("equal")
    axis.set_xlim(-max_plot_radius, max_plot_radius)
    axis.set_ylim(-max_plot_radius, max_plot_radius)
    axis.set_xlabel("x / Schwarzschild radius")
    axis.set_ylabel("y / Schwarzschild radius")
    axis.set_title("Reference photon trajectories")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    Path("outputs").mkdir(exist_ok=True)
    figure.savefig("outputs/reference_trajectory_gallery.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
~~~

### Code walkthrough

The data file is opened with allow_pickle=False because we deliberately store only numeric arrays.

The offsets array is the key to recovering each variable-length curve. For each selected ray, we slice the packed arrays with its start and end offsets.

We only compute radius where u is positive. np.full_like initializes radius to infinity; np.divide fills the valid positions. The visible mask removes invalid and excessively distant points.

The plot draws:
- the positive-side path;
- its mirrored path;
- the event horizon at radius one;
- the photon sphere at radius 1.5.

Run:

~~~bash
python -m scripts.inspect_trajectories
~~~

Expected artifact:

~~~text
outputs/reference_trajectory_gallery.png
~~~

### Dataset gate

Before implementing the agent, inspect:
- Are weak, moderate, and strong bending cases represented?
- Are all stored u and v values finite?
- Are phi values increasing in every curve?
- Does the final sample correspond to the escape event?
- Are there unresolved cases, and are they marked?
- Do the curves look like the trajectories produced by the existing simulator?

This is a practical smoke check, not yet the final test suite.

---

# PART III — Milestone 2: Design the sequential RL problem

## 12. What does the agent actually learn?

At angular coordinate phi_t, the agent receives the current photon state and predicts the next state:

\[
(u_{t+1},v_{t+1}).
\]

The environment advances the angle by a fixed increment:

\[
\phi_{t+1}=\min(\phi_t+\Delta\phi,\phi_{\rm escape}).
\]

The trajectory is generated one decision at a time:

\[
s_0\rightarrow s_1\rightarrow s_2\rightarrow\cdots\rightarrow s_T.
\]

The next state chosen by the policy becomes the current state for the following decision. If the agent makes a poor prediction, its future observations are now based on that imperfect prediction. This sequential dependency is what makes the trajectory rollout different from a simple one-shot curve plot.

## 13. Define the MDP

| RL concept | Meaning in Black Hole Lab |
|---|---|
| Environment | Photon trajectory-generation simulator |
| Episode | One scattering ray for a selected r0 |
| Observation | Initial ray parameter and current generated photon state |
| Action | Proposed next u and v |
| Transition | Advance phi by fixed delta_phi and accept the proposed state |
| Reward | Curve accuracy and physical consistency |
| Terminal success | Valid completion at the reference escape angle |
| Failure | Invalid state or physically implausible trajectory |
| Truncation | Maximum number of decisions reached |

The policy must never see the future reference values in its observation. The environment may use the reference state to calculate teacher-tracking reward during training.

## 14. Observation design

Start with a small observation vector:

\[
o_t=[
u_0,\ u_t,\ v_t,\ \phi_t/\phi_{\rm escape},\ e_{t-1}
].
\]

where:
- u0 identifies the ray's initial closest approach;
- ut and vt describe the current generated state;
- normalized phi describes progress through the episode;
- e_(t-1) is the previous normalized transition error.

The reference future state is not included.

Why include u0? The same local state may appear in different ray cases. The initial parameter tells the policy which trajectory family it is generating.

Why include progress? The agent needs to distinguish early, middle, and terminal parts of the ray. The same current state can require different behavior depending on where it is in the episode.

Why include previous error? It gives the policy a short summary of how well its last action matched the physical trajectory.

Normalize all observation features with scales calculated from training data only. Save those scales and reuse them unchanged for validation and test.

## 15. Action design

Use a continuous two-dimensional action:

\[
a_t=(a_u,a_v),\qquad a_u,a_v\in[-1,1].
\]

Convert normalized actions into physical state proposals:

\[
u_{t+1}=u_{\min}+\frac{a_u+1}{2}(u_{\max}-u_{\min}),
\]

\[
v_{t+1}=v_{\min}+\frac{a_v+1}{2}(v_{\max}-v_{\min}).
\]

The action bounds are derived from the training trajectories with a safety margin, then frozen. They must not be calculated from test trajectories.

This first version predicts the absolute next state. That is easier to inspect than predicting a delta. Later we can compare a delta-action formulation as an ablation.

The environment must explicitly handle:
- non-finite action values;
- actions outside the declared bounds;
- negative u before the expected escape;
- implausible v values;
- failure to reach a valid escape by the episode horizon.

Do not silently clip a wildly invalid physical proposal into a valid state. That hides the policy's mistakes.

---

# PART IV — Milestone 3: Build the reward

The reward must teach the agent to follow the curve while respecting the physics. We use three terms and log them separately.

## 16. Reference-trajectory tracking

For a training ray, the environment can query the DOP853 teacher at the next angular coordinate:

\[
s_{\rm ref}(\phi_{t+1})=
[u_{\rm ref}(\phi_{t+1}),v_{\rm ref}(\phi_{t+1})].
\]

Define:

\[
E_{\rm ref}=
\left(\frac{u_{t+1}-u_{\rm ref}}{\sigma_u}\right)^2+
\left(\frac{v_{t+1}-v_{\rm ref}}{\sigma_v}\right)^2.
\]

The scales sigma_u and sigma_v come from the training dataset.

This is a teacher-guided reward: reference values teach the environment how good the next point is. They are not part of the policy observation. We must state this clearly in the final research report.

## 17. Differential-equation consistency

A proposed transition should approximately satisfy the ODE over h=delta_phi.

Using a midpoint approximation:

\[
R_u=(u_{t+1}-u_t)-h\frac{v_t+v_{t+1}}{2},
\]

\[
u_m=\frac{u_t+u_{t+1}}{2},
\]

\[
R_v=(v_{t+1}-v_t)-h\left(\frac32u_m^2-u_m\right).
\]

Normalize the residuals:

\[
E_{\rm ODE}=
\left(\frac{R_u}{\sigma_{R_u}}\right)^2+
\left(\frac{R_v}{\sigma_{R_v}}\right)^2.
\]

These are discrete consistency residuals. They are not expected to be exactly zero at finite angular spacing.

## 18. Conserved-quantity consistency

At the generated point:

\[
I_{t+1}=v_{t+1}^2-u_{t+1}^3+u_{t+1}^2.
\]

At closest approach:

\[
I_0=u_0^2-u_0^3.
\]

Define a relative invariant error:

\[
E_I=\left(
\frac{I_{t+1}-I_0}{|I_0|+\epsilon}
\right)^2.
\]

The small epsilon avoids division by zero for parameter values where the invariant magnitude is very small.

## 19. Combined reward

Begin with:

\[
r_t=
-\lambda_{\rm ref}E_{\rm ref}
-\lambda_{\rm ODE}E_{\rm ODE}
-\lambda_I E_I.
\]

Add a terminal success bonus only when:
- the episode reached the reference escape angle;
- the generated state is close enough to the reference terminal state;
- the ray is moving outward, meaning v is negative in this coordinate convention;
- the trajectory did not previously enter an invalid state.

Give a finite penalty for invalid actions, non-finite states, early escape, or an episode horizon failure.

Do not choose reward weights randomly. First log the three raw error terms on reference-like transitions and deliberately perturbed transitions. Normalize them so their magnitudes are comparable, then tune weights on validation rays.

### Reward hacking questions

- Can the policy set u to zero at the first step and earn a terminal bonus?
- Can it stay at one state forever?
- Can it generate a visually bad curve but still get high reward?
- Can NaN bypass a comparison?
- Can the agent stop before the reference escape angle?
- Does the terminal reward hide poor tracking over the earlier trajectory?

The answer to each must be no.

---

# PART V — Milestone 4: Implement the Gymnasium environment

## 20. Why Gymnasium?

Gymnasium defines a standard interface between an RL algorithm and an environment.

The essential methods are:
- __init__: configure spaces and immutable experiment parameters;
- reset: select a training ray and initialize the generated trajectory;
- step: apply the policy action, calculate reward, and return the next observation.

The step method returns:

~~~python
observation, reward, terminated, truncated, info
~~~

terminated means a task terminal event occurred, such as valid escape or invalid physical state.

truncated means an external limit ended the episode, such as maximum decisions.

Install RL packages when the dataset is ready:

~~~bash
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

## 21. Load the packed dataset

The environment needs fast access to one trajectory at a time. This helper loads the packed arrays and returns the selected case slice.

### File: src/lab/trajectory_env.py — dataset loader

~~~python
import numpy as np


class TrajectoryStore:
    def __init__(self, path):
        archive = np.load(path, allow_pickle=False)

        self.r0 = archive["r0"]
        self.phi = archive["phi"]
        self.u = archive["u"]
        self.v = archive["v"]
        self.offsets = archive["offsets"]
        self.escape_angle = archive["escape_angle"]
        self.deflection_radian = archive["deflection_radian"]

        count = len(self.r0)

        if len(self.offsets) != count + 1:
            raise ValueError("offsets must have one more entry than trajectories")

        if not (
            len(self.phi) == len(self.u) == len(self.v)
        ):
            raise ValueError("Packed phi, u, and v arrays must have equal lengths")

    def get(self, index):
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])

        return {
            "r0": float(self.r0[index]),
            "phi": self.phi[start:end],
            "u": self.u[start:end],
            "v": self.v[start:end],
            "escape_angle": float(self.escape_angle[index]),
            "deflection_radian": float(self.deflection_radian[index]),
        }
~~~

### Explanation

The constructor loads numeric arrays from NPZ. allow_pickle=False prevents arbitrary Python object deserialization and matches our packed numeric format.

The offsets array has one more entry than the number of trajectories. For trajectory i, offsets[i] is the start and offsets[i+1] is the end.

The loader checks the basic shape contract. Later, the comprehensive test milestone will add monotonic offsets, finite values, and valid per-trajectory lengths.

get(index) returns a dictionary for one ray. This keeps the environment from manually slicing the packed arrays every time it resets.

## 22. Environment state and transition

The environment maintains two distinct states:

1. **Reference trajectory:** the immutable DOP853 curve for the current training ray.
2. **Generated trajectory:** the points proposed by the agent so far.

The agent receives only the generated current state, initial ray parameter, progress, and previous error. It does not receive the next reference point.

At reset:
- select a training trajectory using Gymnasium's seeded random generator;
- set the generated state to [1/r0, 0];
- set phi to zero;
- clear the previous error;
- store the initial generated point.

At step:
- map the action into proposed u and v;
- advance phi by the fixed delta_phi, stopping exactly at the reference escape angle on the last transition;
- interpolate the teacher state at the new angle for reward only;
- calculate tracking, ODE, and invariant error;
- update the generated state;
- decide whether the ray escaped, failed, or reached the decision limit.

## 23. Complete environment implementation

The following is a first complete implementation blueprint. The reward weights and action bounds are experiment configuration, not universal constants. The bounds here are intentionally calculated from the training dataset and supplied to the environment.

### File: src/lab/trajectory_env.py — environment

~~~python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PhotonTrajectoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        store,
        case_indices,
        delta_phi=0.01,
        action_u_bounds=None,
        action_v_bounds=None,
        scaling_config=None,
        max_decisions=5000,
        reward_weights=(1.0, 0.05, 0.05),
        terminal_bonus=2.0,
        failure_penalty=2.0,
    ):
        super().__init__()

        self.store = store
        self.case_indices = np.asarray(case_indices, dtype=int)

        if len(self.case_indices) == 0:
            raise ValueError("At least one trajectory index is required")

        if np.any(self.case_indices < 0) or np.any(self.case_indices >= len(store.r0)):
            raise ValueError("case_indices contains an index outside the dataset")

        self.delta_phi = float(delta_phi)
        self.max_decisions = int(max_decisions)

        # Fit scales and action limits on training data once. During
        # evaluation, pass the saved scaling_config from the training run.
        if scaling_config is None:
            training_curves = [store.get(int(i)) for i in self.case_indices]
            training_u = np.concatenate([curve["u"] for curve in training_curves])
            training_v = np.concatenate([curve["v"] for curve in training_curves])

            def padded_bounds(values, lower_limit=None):
                lo = float(np.min(values))
                hi = float(np.max(values))
                margin = max(0.05 * (hi - lo), 1e-5)
                lower = lo - margin
                upper = hi + margin
                if lower_limit is not None:
                    lower = max(lower_limit, lower)
                if upper <= lower:
                    upper = lower + 1e-4
                return (lower, upper)

            self.u_scale = max(float(np.std(training_u)), 1e-6)
            self.v_scale = max(float(np.std(training_v)), 1e-6)
            self.rhs_scale = max(
                float(np.std(1.5 * training_u**2 - training_u)),
                1e-6,
            )

            self.u_bounds = (
                tuple(action_u_bounds)
                if action_u_bounds is not None
                else padded_bounds(training_u, lower_limit=0.0)
            )
            self.v_bounds = (
                tuple(action_v_bounds)
                if action_v_bounds is not None
                else padded_bounds(training_v)
            )
        else:
            self.u_scale = float(scaling_config["u_scale"])
            self.v_scale = float(scaling_config["v_scale"])
            self.rhs_scale = float(scaling_config["rhs_scale"])
            self.u_bounds = tuple(scaling_config["u_bounds"])
            self.v_bounds = tuple(scaling_config["v_bounds"])

        self.scaling_config = {
            "u_scale": self.u_scale,
            "v_scale": self.v_scale,
            "rhs_scale": self.rhs_scale,
            "u_bounds": list(self.u_bounds),
            "v_bounds": list(self.v_bounds),
        }

        self.w_ref, self.w_ode, self.w_inv = reward_weights
        self.terminal_bonus = float(terminal_bonus)
        self.failure_penalty = float(failure_penalty)

        if self.delta_phi <= 0:
            raise ValueError("delta_phi must be positive")

        if not self.u_bounds[0] < self.u_bounds[1]:
            raise ValueError("Invalid u action bounds")

        if not self.v_bounds[0] < self.v_bounds[1]:
            raise ValueError("Invalid v action bounds")

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # We normalize physical state variables before returning observations.
        # Infinite Box limits avoid falsely declaring a narrow physical range.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self.case = None
        self.r0 = None
        self.u0 = None
        self.phi = 0.0
        self.state = None
        self.previous_error = 0.0
        self.decision_count = 0
        self.generated_phi = []
        self.generated_states = []
        self.done = False


    def _observation(self):
        u, v = self.state

        progress = self.phi / max(self.case["escape_angle"], 1e-8)
        progress = float(np.clip(progress, 0.0, 1.0))

        observation = np.array(
            [
                self.u0 / self.u_scale,
                u / self.u_scale,
                v / self.v_scale,
                progress,
                np.log10(self.previous_error + 1e-8),
            ],
            dtype=np.float32,
        )

        return observation

    def _action_to_state(self, action):
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain two finite values")

        action = np.clip(action, -1.0, 1.0)

        u = self.u_bounds[0] + 0.5 * (action[0] + 1.0) * (
            self.u_bounds[1] - self.u_bounds[0]
        )

        v = self.v_bounds[0] + 0.5 * (action[1] + 1.0) * (
            self.v_bounds[1] - self.v_bounds[0]
        )

        return np.array([u, v], dtype=np.float64)

    @staticmethod
    def _invariant(state):
        u, v = state
        return v**2 - u**3 + u**2

    def _reward_components(self, proposed_state, next_phi, step):
        target_u = np.interp(
            next_phi,
            self.case["phi"],
            self.case["u"],
        )
        target_v = np.interp(
            next_phi,
            self.case["phi"],
            self.case["v"],
        )
        target = np.array([target_u, target_v])

        tracking_error = np.array([
            (proposed_state[0] - target[0]) / self.u_scale,
            (proposed_state[1] - target[1]) / self.v_scale,
        ])
        e_ref = float(np.mean(tracking_error**2))

        old_u, old_v = self.state
        new_u, new_v = proposed_state

        midpoint_u = 0.5 * (old_u + new_u)

        residual_u = (new_u - old_u) - step * 0.5 * (old_v + new_v)
        residual_v = (
            (new_v - old_v)
            - step * (1.5 * midpoint_u**2 - midpoint_u)
        )

        scale_u = max(step * self.v_scale, 1e-7)
        scale_v = max(step * self.rhs_scale, 1e-7)

        e_ode = float(
            0.5 * (
                (residual_u / scale_u) ** 2
                + (residual_v / scale_v) ** 2
            )
        )

        initial_invariant = self.u0**2 - self.u0**3
        new_invariant = self._invariant(proposed_state)
        relative_drift = (
            (new_invariant - initial_invariant)
            / (abs(initial_invariant) + 1e-8)
        )
        e_invariant = float(relative_drift**2)

        return {
            "reference": e_ref,
            "ode": e_ode,
            "invariant": e_invariant,
            "target_state": target,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "case_index" in options:
            selected = int(options["case_index"])
            if selected not in self.case_indices:
                raise ValueError("Requested case is not enabled in this environment")
        else:
            selected = int(self.np_random.choice(self.case_indices))

        self.case = self.store.get(selected)

        self.r0 = self.case["r0"]
        self.u0 = 1.0 / self.r0

        self.phi = 0.0
        self.state = np.array([self.u0, 0.0], dtype=np.float64)
        self.previous_error = 0.0
        self.decision_count = 0

        self.generated_phi = [self.phi]
        self.generated_states = [self.state.copy()]
        self.done = False

        return self._observation(), {
            "case_index": selected,
            "r0": self.r0,
            "escape_angle": self.case["escape_angle"],
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        try:
            proposed_state = self._action_to_state(action)
        except ValueError:
            self.done = True
            return (
                self._observation(),
                -self.failure_penalty,
                True,
                False,
                {"termination_reason": "invalid_action"},
            )

        escape_angle = self.case["escape_angle"]
        next_phi = min(self.phi + self.delta_phi, escape_angle)
        actual_step = next_phi - self.phi

        components = self._reward_components(
            proposed_state,
            next_phi,
            actual_step,
        )

        reward = (
            -self.w_ref * components["reference"]
            -self.w_ode * components["ode"]
            -self.w_inv * components["invariant"]
        )

        finite = np.all(np.isfinite(proposed_state))
        nonnegative_radius_coordinate = proposed_state[0] >= 0.0
        not_early_escape = not (
            proposed_state[0] <= 0.0
            and next_phi < escape_angle
        )

        valid = finite and nonnegative_radius_coordinate and not_early_escape

        self.decision_count += 1

        if not valid:
            reward -= self.failure_penalty
            self.done = True

            info = {
                "termination_reason": "invalid_trajectory_state",
                "reward_reference": components["reference"],
                "reward_ode": components["ode"],
                "reward_invariant": components["invariant"],
                "decision_count": self.decision_count,
            }

            return self._observation(), float(reward), True, False, info

        self.phi = next_phi
        self.state = proposed_state
        self.previous_error = components["reference"]

        self.generated_phi.append(self.phi)
        self.generated_states.append(self.state.copy())

        reached_reference_end = self.phi >= escape_angle - 1e-12

        if reached_reference_end:
            final_error = components["reference"]
            outward = self.state[1] < 0.0
            close_to_escape = self.state[0] <= max(1e-5, self.u_scale * 0.05)

            success = (
                final_error < 0.05
                and outward
                and close_to_escape
            )

            if success:
                reward += self.terminal_bonus
                reason = "successful_escape"
            else:
                reward -= self.failure_penalty
                reason = "poor_terminal_prediction"

            self.done = True
            terminated = True
            truncated = False

        elif self.decision_count >= self.max_decisions:
            reward -= self.failure_penalty
            self.done = True
            reason = "decision_limit"
            terminated = False
            truncated = True
        else:
            reason = "continuing"
            terminated = False
            truncated = False

        info = {
            "r0": self.r0,
            "phi": self.phi,
            "proposed_u": float(self.state[0]),
            "proposed_v": float(self.state[1]),
            "reward_reference": components["reference"],
            "reward_ode": components["ode"],
            "reward_invariant": components["invariant"],
            "decision_count": self.decision_count,
            "termination_reason": reason,
            "escape_angle": escape_angle if terminated else None,
        }

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
~~~

### Important implementation notes

This blueprint shows the whole environment data flow, but it must be reviewed against real dataset ranges before serious training.

**TrajectoryStore**

It loads the packed arrays from the dataset. This object is not a Gymnasium environment; it only retrieves the teacher trajectory for a chosen case.

**Action space**

The policy outputs two normalized numbers in [-1,1]. _action_to_state converts those numbers into u and v using frozen physical bounds. This means the neural policy never has to output values with completely different numerical scales directly.

**Observation**

The environment returns normalized initial u0, current u, current v, normalized angular progress, and log previous error. The teacher's next u and v are deliberately not exposed to the policy.

**Reward calculation**

_reward_components is separate so each term can be inspected:
- reference: closeness to the teacher's next state;
- ode: whether the transition approximately follows the differential equation;
- invariant: whether the conserved quantity remains stable.

The reward uses negative weighted errors. Better trajectory predictions should receive less negative reward.

**Fixed angular increment**

The environment advances phi by delta_phi, except the final interval can be shorter so that the episode ends exactly at the reference escape angle. The agent does not control this increment.

**Terminal logic**

Reaching the reference angular endpoint alone is not success. The generated state must also be close to the terminal reference state and have the correct outward direction.

**Important first-run limitation**

The example's initial action bounds and reward scales are starting values. We must inspect the actual training data, verify the observation range, and tune reward weights with validation trajectories. The first implementation should log every reward component. Do not judge the agent only by total reward.

---

# PART VI — Milestone 5: Train the RL policy

## 24. Why PPO first?

The main research goal is RL. PPO is simply the first algorithm we will try because it supports continuous bounded actions and has a practical implementation in Stable-Baselines3.

PPO has:
- an actor policy that chooses an action distribution;
- a critic that estimates expected future return;
- an objective that discourages excessively large policy updates.

For our environment, the actor outputs a distribution over two continuous values: the proposed next u and v.

PPO is not guaranteed to be best. Once the first RL pipeline works, we can compare SAC. Do not spend the first iteration implementing multiple algorithms before one complete agent can generate a curve.

## 25. Understand the PPO terms we will monitor

The policy is:

\[
\pi_\theta(a\mid s).
\]

It maps an observation to an action distribution.

The critic estimates:

\[
V_\phi(s),
\]

the expected future return from state s.

The discounted return is:

\[
G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}.
\]

The advantage estimates whether an action performed better or worse than the critic expected:

\[
A_t=Q(s_t,a_t)-V(s_t).
\]

PPO uses the probability ratio:

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\rm old}}(a_t\mid s_t)}.
\]

Its clipped policy objective is:

\[
L^{\rm CLIP}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

The clipping discourages excessively large policy updates. We will use the library implementation, but we should understand that PPO optimizes expected reward, not the physical curve error directly. That is why evaluation must measure the generated trajectories.

## 26. Split dataset by ray before training

Never split individual points from a trajectory between training and test. Neighboring points from one ray are strongly correlated. If one point is in training and the next point is in test, the evaluation is contaminated.

Split by complete ray index. Each entry of the stored r0 array identifies one independent trajectory.

Create scripts/make_splits.py:

~~~python
import json
from pathlib import Path

import numpy as np


def make_splits(
    dataset_path="data/photon_trajectories.npz",
    output_path="data/split_manifest.json",
    seed=42,
    train_fraction=0.70,
    validation_fraction=0.15,
):
    data = np.load(dataset_path, allow_pickle=False)
    count = len(data["r0"])

    if count < 10:
        raise ValueError("Generate more trajectories before splitting")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test fraction is required")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)

    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)

    train = indices[:train_end]
    validation = indices[train_end:validation_end]
    test = indices[validation_end:]

    manifest = {
        "seed": seed,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "train_r0": data["r0"][train].tolist(),
        "validation_r0": data["r0"][validation].tolist(),
        "test_r0": data["r0"][test].tolist(),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


if __name__ == "__main__":
    result = make_splits()
    print(
        len(result["train_indices"]),
        len(result["validation_indices"]),
        len(result["test_indices"]),
    )
~~~

### Walkthrough

- The script reads only the r0 case list. It does not split the packed trajectory samples.
- rng.permutation produces one reproducible shuffled ordering.
- The first 70 percent of case IDs become training rays.
- The next 15 percent become validation rays.
- The remaining rays become held-out test rays.
- The manifest stores both indices and the corresponding physical r0 values. This makes it possible to inspect exactly which rays were used.

Run it once:

~~~bash
python -m scripts.make_splits
~~~

Do not regenerate the split every time you train. A new split changes the experiment. Keep this manifest fixed when comparing PPO runs and later comparison models.

The same split must be used by RL, the supervised MLP, and the PINN. Normalization statistics and action bounds must be estimated from training rays only. Validation is used for tuning. Test rays must remain untouched until final evaluation.

## 27. Environment factory

Create a small factory so training and evaluation construct the same environment configuration.

### File: src/lab/trajectory_env.py — factory

~~~python
# Add these imports at the top of src/lab/trajectory_env.py.
import json
from pathlib import Path


def load_split(path="data/split_manifest.json"):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    return (
        np.asarray(payload["train_indices"], dtype=int),
        np.asarray(payload["validation_indices"], dtype=int),
        np.asarray(payload["test_indices"], dtype=int),
    )


def make_training_env(
    dataset_path="data/photon_trajectories.npz",
    split_path="data/split_manifest.json",
    seed=7,
):
    store = TrajectoryStore(dataset_path)
    train_indices, _, _ = load_split(split_path)

    env = PhotonTrajectoryEnv(
        store=store,
        case_indices=train_indices,
        delta_phi=0.01,
    )
    env.reset(seed=seed)
    return env
~~~

This example shows the intended factory responsibilities. In the final source tree, keep the factory inside the same trajectory_env.py module rather than importing the module from itself. The snippet is separated here to explain the boundary: data loading and split selection happen once; episode sampling happens inside the environment.

## 28. Training script

### File: scripts/train_rl.py

~~~python
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from lab.trajectory_env import make_training_env


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOG_DIR = OUTPUTS / "rl_logs"
MODEL_DIR = OUTPUTS / "models"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = make_training_env(seed=7)
    env = Monitor(env)

    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_photon_trajectory",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        seed=7,
    )

    configuration = {
        "algorithm": "PPO",
        "seed": 7,
        "total_timesteps": 100_000,
        "observation": [
            "normalized_u0",
            "normalized_current_u",
            "normalized_current_v",
            "angular_progress",
            "log_previous_tracking_error",
        ],
        "action": ["normalized_next_u", "normalized_next_v"],
        "delta_phi": 0.01,
        # Critical: save training-only normalization and action limits.
        "environment_scaling": env.unwrapped.scaling_config,
    }

    (OUTPUTS / "ppo_config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    model.learn(
        total_timesteps=configuration["total_timesteps"],
        callback=checkpoint,
    )

    model.save(MODEL_DIR / "ppo_photon_trajectory")


if __name__ == "__main__":
    main()
~~~

### Code walkthrough

**Path configuration**

Path(__file__).resolve().parents[1] finds the repository root relative to this script. This avoids depending on whichever terminal directory happens to be active.

**Environment construction**

make_training_env loads the trajectory dataset and selects only training case IDs. Monitor wraps the environment so Stable-Baselines3 can record episode returns and lengths.

**CheckpointCallback**

Training can take a long time. The callback periodically saves intermediate models. If a later run fails, we do not lose every earlier checkpoint.

**PPO configuration**

- MlpPolicy uses a multilayer perceptron for the actor and critic.
- learning_rate controls optimizer step size.
- n_steps controls rollout collection before an update.
- batch_size controls minibatch size.
- gamma discounts future rewards.
- gae_lambda controls generalized advantage estimation.
- clip_range controls PPO's clipped update objective.
- seed makes the run more reproducible.

These are starting hyperparameters, not magic values.

**Configuration file**

The JSON stores the action and observation meanings alongside the algorithm settings. A saved neural-network checkpoint without its environment definition is not a reproducible experiment.

### First run

Start with a short smoke run by temporarily setting total_timesteps to 5,000. This checks that rollouts, rewards, saving, and logging work. It is not the final training budget.

Run:

~~~bash
python -m scripts.train_rl
~~~

TensorBoard:

~~~bash
tensorboard --logdir outputs/rl_logs
~~~

Inspect:
- episode reward;
- episode length;
- reward components;
- valid escape rate;
- failure reasons;
- action distribution;
- whether the generated paths become closer to the reference.

A rising PPO reward is not proof that the light curve is correct.

## 29. Evaluate one fixed case and save the generated path

The environment reset method accepts an optional case_index. This lets us create a separate environment containing only validation or test indices, then evaluate a specific ray reproducibly.

Create scripts/evaluate_rl.py:

~~~python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from lab.trajectory_env import (
    PhotonTrajectoryEnv,
    TrajectoryStore,
    load_split,
)


def rollout(model, env, case_index):
    observation, info = env.reset(
        seed=123,
        options={"case_index": int(case_index)},
    )

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

    generated_phi = np.asarray(env.generated_phi, dtype=np.float64)
    generated_states = np.asarray(env.generated_states, dtype=np.float64)

    reference_phi = np.asarray(env.case["phi"], dtype=np.float64)
    reference_u = np.asarray(env.case["u"], dtype=np.float64)
    reference_v = np.asarray(env.case["v"], dtype=np.float64)

    common_end = min(generated_phi[-1], reference_phi[-1])
    comparison_phi = np.linspace(0.0, common_end, 500)

    generated_u = np.interp(
        comparison_phi, generated_phi, generated_states[:, 0]
    )
    generated_v = np.interp(
        comparison_phi, generated_phi, generated_states[:, 1]
    )
    target_u = np.interp(comparison_phi, reference_phi, reference_u)
    target_v = np.interp(comparison_phi, reference_phi, reference_v)

    max_u_error = float(np.max(np.abs(generated_u - target_u)))
    max_v_error = float(np.max(np.abs(generated_v - target_v)))

    valid_cartesian = (
        (generated_u > 0.0)
        & (target_u > 0.0)
        & (generated_u >= 1.0 / 30.0)
        & (target_u >= 1.0 / 30.0)
    )

    if np.any(valid_cartesian):
        generated_r = 1.0 / generated_u[valid_cartesian]
        target_r = 1.0 / target_u[valid_cartesian]

        generated_x = generated_r * np.cos(comparison_phi[valid_cartesian])
        generated_y = generated_r * np.sin(comparison_phi[valid_cartesian])
        target_x = target_r * np.cos(comparison_phi[valid_cartesian])
        target_y = target_r * np.sin(comparison_phi[valid_cartesian])

        cartesian_error = np.sqrt(
            (generated_x - target_x) ** 2
            + (generated_y - target_y) ** 2
        )
        mean_cartesian_error = float(np.mean(cartesian_error))
    else:
        mean_cartesian_error = None

    successful = step_info["termination_reason"] == "successful_escape"
    agent_escape_angle = float(generated_phi[-1]) if successful else None
    agent_deflection = (
        2.0 * agent_escape_angle - np.pi
        if agent_escape_angle is not None
        else None
    )

    return {
        "case_index": int(case_index),
        "r0": float(info["r0"]),
        "phi": generated_phi,
        "states": generated_states,
        "reference_phi": reference_phi,
        "reference_u": reference_u,
        "reference_v": reference_v,
        "escape_angle": float(env.case["escape_angle"]),
        "deflection_radian": float(env.case["deflection_radian"]),
        "agent_escape_angle": agent_escape_angle,
        "agent_deflection_radian": agent_deflection,
        "max_u_error": max_u_error,
        "max_v_error": max_v_error,
        "mean_cartesian_curve_error": mean_cartesian_error,
        "return": total_reward,
        "termination_reason": step_info["termination_reason"],
    }


def plot_rollout(result, output_path):
    generated = result["states"]
    generated_phi = result["phi"]

    reference_phi = result["reference_phi"]
    reference_u = result["reference_u"]

    valid_generated = generated[:, 0] > 0
    valid_reference = reference_u > 0

    generated_r = 1.0 / generated[valid_generated, 0]
    reference_r = 1.0 / reference_u[valid_reference]

    gx = generated_r * np.cos(generated_phi[valid_generated])
    gy = generated_r * np.sin(generated_phi[valid_generated])

    rx = reference_r * np.cos(reference_phi[valid_reference])
    ry = reference_r * np.sin(reference_phi[valid_reference])

    limit = 30.0
    gmask = np.hypot(gx, gy) <= limit
    rmask = np.hypot(rx, ry) <= limit

    figure, axis = plt.subplots(figsize=(9, 9))

    axis.plot(rx[rmask], ry[rmask], label="DOP853 reference", linewidth=2.5)
    axis.plot(gx[gmask], gy[gmask], "--", label="RL generated", linewidth=2.0)

    axis.plot(rx[rmask], -ry[rmask], alpha=0.5)
    axis.plot(gx[gmask], -gy[gmask], "--", alpha=0.5)

    axis.add_patch(plt.Circle((0, 0), 1.0, color="black", label="Event horizon"))
    axis.add_patch(
        plt.Circle((0, 0), 1.5, fill=False, linestyle=":", label="Photon sphere")
    )

    axis.set_aspect("equal")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("x / Schwarzschild radius")
    axis.set_ylabel("y / Schwarzschild radius")
    axis.set_title(f"RL trajectory, r0={result['r0']:.3f}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    dataset_path = "data/photon_trajectories.npz"
    split_path = "data/split_manifest.json"
    model_path = "outputs/models/ppo_photon_trajectory.zip"

    store = TrajectoryStore(dataset_path)
    _, validation_indices, _ = load_split(split_path)

    saved_config = json.loads(
        Path("outputs/ppo_config.json").read_text(encoding="utf-8")
    )

    env = PhotonTrajectoryEnv(
        store=store,
        case_indices=validation_indices,
        delta_phi=saved_config["delta_phi"],
        scaling_config=saved_config["environment_scaling"],
    )

    model = PPO.load(model_path)

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for case_index in validation_indices[:5]:
        result = rollout(model, env, int(case_index))

        filename = output_dir / f"case_{case_index:04d}.png"
        plot_rollout(result, filename)

        summaries.append({
            "case_index": result["case_index"],
            "r0": result["r0"],
            "return": result["return"],
            "termination_reason": result["termination_reason"],
            "escape_angle_reference": result["escape_angle"],
            "escape_angle_agent": result["agent_escape_angle"],
            "deflection_reference": result["deflection_radian"],
            "deflection_agent": result["agent_deflection_radian"],
            "max_u_error": result["max_u_error"],
            "max_v_error": result["max_v_error"],
            "mean_cartesian_curve_error": result["mean_cartesian_curve_error"],
        })

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
~~~

### Walkthrough

- The evaluation environment receives validation indices only. It cannot sample training or test rays by accident.
- It also receives the exact normalization scales and action bounds saved from training. Re-fitting those values on validation/test data would leak information and change the meaning of the policy's observations/actions.
- reset(options={"case_index": ...}) selects an exact ray, making repeated visualization reproducible.
- model.predict uses deterministic=True so the plotted policy output is not randomly sampled.
- rollout stores the generated state sequence from the environment and the reference sequence from the selected data case.
- plot_rollout converts u to radius only where u is positive, then applies the same plotting limit to both curves.
- The summary JSON records the case, return, termination reason, and reference terminal values. We will add actual curve-error metrics in the evaluation milestone.

Run after training:

~~~bash
python -m scripts.evaluate_rl
~~~

At this stage we want to inspect a handful of validation curves manually before running a large evaluation.



Before increasing training time, load the saved model and let it generate trajectories for several training and validation rays.

The evaluation script should:
1. reset the environment for a selected case;
2. repeatedly call model.predict(observation, deterministic=True);
3. collect generated phi, u, and v from the environment;
4. stop when terminated or truncated;
5. convert u to Cartesian coordinates;
6. plot generated and reference paths on the same axes.

The environment should expose a method or info field for the generated path. Do not reconstruct it from printed logs.

What we want to see:
- the generated path begins at the correct closest-approach point;
- it bends in the correct direction;
- it does not jump to negative u early;
- it approaches the escape point at the right angular coordinate;
- its mirrored curve looks physically plausible.

If it does not, inspect the per-step reward terms and action outputs. Do not simply add more training steps.

---

# PART VII — Milestone 6: Evaluate the RL-generated curves

## 30. What counts as a good generated curve?

A good curve must be correct numerically and physically, not merely attractive in a plot.

For every ray, compare the generated state with DOP853 on a common angular grid.

### State errors

\[
E_u=\max_\phi|u_{\rm agent}(\phi)-u_{\rm ref}(\phi)|,
\]

\[
E_v=\max_\phi|v_{\rm agent}(\phi)-v_{\rm ref}(\phi)|.
\]

### Terminal errors

\[
E_\phi=
|\phi_{\rm escape,agent}-\phi_{\rm escape,ref}|,
\]

\[
E_\alpha=
|\alpha_{\rm agent}-\alpha_{\rm ref}|.
\]

Since alpha = 2 phi_escape - pi:

\[
E_\alpha=2E_\phi
\]

when both are computed consistently.

### Physics diagnostics

Record the maximum ODE residual and the maximum invariant drift along the generated path.

### Operational diagnostics

Record:
- valid escape rate;
- episode length;
- invalid-state count;
- average reward components;
- runtime per ray.

## 31. Visual comparison

For each test ray:
- draw the DOP853 curve;
- draw the agent-generated curve;
- use identical axes and plotting masks;
- mark the event horizon and photon sphere;
- put r0 and deflection angle in the legend;
- add a second plot showing u and v error over phi.

Do not compare trajectories by array index if their lengths differ. Interpolate on the common valid angular interval.

## 32. Held-out generalization

The final test must use r0 values never used for training or validation.

Test:
- unseen r0 values inside the training interval;
- values near the training boundaries;
- a separate near-critical set after expanding the data range.

Do not tune reward weights or normalization based on test performance. The test set is for the final report.

## 33. Multiple seeds

Run at least several independent training seeds after the first complete pipeline works. Report mean and variation across seeds, not only the best model.

Keep:
- dataset manifest;
- train/validation/test split;
- model configuration;
- training seed;
- checkpoint;
- per-ray metrics;
- rendered output curves.

---

# PART VIII — Milestone 7: Comprehensive tests (now, after the RL pipeline exists)

The comprehensive tests are intentionally placed after the first dataset, environment, and training pipeline. During earlier implementation, use small smoke checks to catch obvious errors. At this milestone, formalize the behavior and failure cases across the whole system.

## 34. Physics and reference tests

Test:
- normal scattering rays escape;
- short max_angle runs are not mislabeled as escaped;
- invalid r0 is rejected;
- deflection equals 2*escape_angle-pi;
- dense reference returns both u and v;
- dense reference rejects queries outside its interval;
- the conserved quantity remains approximately constant on reference trajectories.

## 35. Dataset tests

Test:
- same configuration and seed generate the same r0 cases;
- offsets has length number_of_trajectories plus one;
- offsets is monotonically increasing;
- phi, u, and v packed arrays have equal length;
- each trajectory begins at [1/r0, 0];
- phi is strictly increasing within each trajectory;
- the last phi equals the escape angle;
- final u is close to zero;
- all stored states are finite;
- unresolved cases are not present as complete training trajectories;
- the JSON manifest matches the NPZ contents.

## 36. Environment tests

Test:
- reset with the same seed chooses the same training case;
- initial observation has the declared shape and dtype;
- observation contains no NaN or infinity;
- action shape and bounds are handled;
- policy observation does not include future reference state;
- phi advances by the fixed increment;
- final increment is clipped to the escape angle;
- invalid proposals cannot earn terminal success;
- terminal success requires good final tracking and outward direction;
- horizon exhaustion truncates;
- step after episode completion is rejected;
- reward and info fields are finite and present.

## 37. RL pipeline smoke tests

Test:
- environment factory selects only training indices;
- one short PPO rollout can be collected;
- model checkpoint is written;
- model can be reloaded;
- deterministic prediction has correct action shape;
- evaluation stops on termination/truncation;
- generated path arrays have matching lengths;
- plotting does not crash on the escape endpoint.

## 38. Run the tests

~~~bash
python -m pytest -q
~~~

Use separate tests for physics, data, environment, and training integration. Do not make the entire RL training budget part of the normal fast test suite. Use a short smoke test or marked integration test for the training pipeline.

---

# PART IX — Only after RL works: comparison models and physics extensions

These are supporting experiments. Do not start all of them before the RL agent can produce and evaluate curves.

## 39. Supervised trajectory MLP

The supervised baseline predicts the state directly:

\[
(\phi,r_0)\longrightarrow(u,v).
\]

It sees labelled reference pairs during training and does not need to roll out one state at a time.

A simple network:

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

    def forward(self, inputs):
        return self.network(inputs)
~~~

### Walkthrough

The input has two features: angular coordinate phi and closest approach r0. The output has two features: u and v.

Tanh is smooth and bounded, which is often useful for smooth function approximation. It is not guaranteed to be the best activation.

Normalize inputs and targets using training-set statistics only. Train with a weighted MSE, select checkpoints using validation loss, then calculate test metrics once.

The comparison is important: supervised learning may be more efficient than RL for this deterministic mapping. If so, report it honestly.

## 40. Physics-Informed Neural Network

A PINN approximates a continuous function:

\[
u_\theta(\phi,r_0).
\]

Its differential-equation residual is:

\[
\mathcal R_\theta=
\frac{\partial^2u_\theta}{\partial\phi^2}
+u_\theta-\frac32u_\theta^2.
\]

The physics loss:

\[
L_f=\mathbb E[\mathcal R_\theta^2].
\]

Initial-condition losses:

\[
L_0=\mathbb E[
(u_\theta(0,r_0)-1/r_0)^2
],
\]

\[
L_1=\mathbb E[
(\partial_\phi u_\theta(0,r_0))^2
].
\]

The total loss:

\[
L=L_f+\lambda_0L_0+\lambda_1L_1.
\]

The ODE alone does not select a unique trajectory. The initial conditions must be enforced.

### Autograd second-derivative check

~~~python
import torch

x = torch.tensor([[0.2], [0.5], [0.8]], requires_grad=True)
y = x**3

dy = torch.autograd.grad(
    y.sum(),
    x,
    create_graph=True,
)[0]

d2y = torch.autograd.grad(
    dy.sum(),
    x,
)[0]

expected = 6.0 * x.detach()
assert torch.allclose(d2y, expected, atol=1e-5)
~~~

The example uses f(x)=x³, whose second derivative is 6x. create_graph=True ensures that the first derivative remains differentiable so we can calculate the second derivative.

The PINN branch should use the same ray-level split and held-out metrics as the RL and MLP models. A small physics residual does not automatically mean a correct trajectory; check curve and deflection errors.

## 41. Weak-field comparison

For a Schwarzschild turning point r0, the dimensionless impact parameter is:

\[
\bar b=\frac{b}{r_s}
=\frac{r_0}{\sqrt{1-1/r_0}}.
\]

In the weak field:

\[
\alpha\approx\frac{2}{\bar b}.
\]

This leading-order approximation should work better far from the black hole and degrade in the strong-field regime. It is a physical baseline, not a replacement for the numerical reference.

Keep this experiment after the RL pipeline is working, as requested. It helps contextualize how difficult the strong-bending cases are.

## 42. Near-critical photon-sphere sweep

For offsets approaching zero:

\[
r_0=1.5+\delta.
\]

Use geometrically decreasing positive delta values. Record escape angle, deflection, nfev, status, and max_angle.

The deflection grows rapidly near the photon sphere. If the run reaches max_angle with no event, label it unresolved. Do not label it captured without an incoming initial condition and a capture event.

This sweep is also useful for deciding which near-critical rays should be added to a later training dataset.

## 43. Adaptive numerical integration

The former step-size-control work remains an optional numerical-method branch. It can be useful to improve reference efficiency or provide another baseline.

For RK4 step-doubling:
- calculate one full h step;
- calculate two half steps;
- estimate local error with the fourth-order factor 15;
- accept/reject using a normalized error;
- count every RHS call.

Do not make this the RL objective again. The agent's main action remains the proposed next photon state.

---

# PART X — Reproducibility, engineering, and final report

## 44. Project metadata

The repository was inspected with a file named project.toml. Python packaging frontends expect pyproject.toml. Check the working tree. If it is still named project.toml, rename it:

~~~bash
git mv project.toml pyproject.toml
~~~

Do not keep two competing project metadata files.

The existing dependencies include NumPy, SciPy, and Matplotlib, with pytest as a development dependency. Add Gymnasium, Stable-Baselines3, and TensorBoard when the RL environment is ready. Add PyTorch and any dataset tooling when the comparison-model milestones begin.

## 45. Development setup

Linux / WSL:

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

Run scripts from the repository root and use package imports. Avoid relying on the current working directory accidentally placing src/lab on the Python path.

## 46. Repository hygiene

- Keep source code under src/.
- Keep scripts under scripts/.
- Keep tests under tests/.
- Keep generated datasets under data/.
- Keep figures, logs, and checkpoints under outputs/.
- Ignore virtual environments, Python bytecode, transient logs, and large generated files unless deliberately publishing selected artifacts.
- Do not let imports start simulations or training jobs.
- Record the Git commit associated with each reported experiment.

Every executable script should use a main guard:

~~~python
def main():
    ...


if __name__ == "__main__":
    main()
~~~

## 47. Experiment manifest

For each dataset, save:
- schema version;
- governing equation and state convention;
- units;
- DOP853 method and tolerances;
- max_angle and max_step;
- delta_phi;
- all r0 case IDs;
- escaped and unresolved status;
- training, validation, and test case IDs;
- random seed.

For each training run, save:
- Git commit;
- dataset manifest path/hash;
- observation and action definitions;
- normalization values;
- reward formula and weights;
- delta_phi;
- RL algorithm and hyperparameters;
- random seed and total training steps;
- checkpoint path;
- validation metrics.

A model checkpoint by itself is not a reproducible experiment.

## 48. Suggested per-ray evaluation CSV

~~~text
method,seed,case_id,r0,escaped,escape_angle,deflection_radian,max_u_error,max_v_error,cartesian_curve_error,ode_residual,invariant_drift,rollout_steps,runtime_seconds
~~~

Keep one row per ray. Do not average away failures before saving the raw results.

## 49. Final report structure

1. Abstract — research question, methods, and measured result.
2. Introduction — gravitational light bending and sequential trajectory generation.
3. Physics — Schwarzschild assumptions, variables, ODE, initial conditions, photon sphere.
4. Reference solver — DOP853, escape event, dense interpolation, tolerances.
5. Dataset — ray sampling, resampling, packed format, splits, unresolved cases.
6. RL formulation — observation, action, transition, reward, terminal conditions.
7. Algorithm — PPO and any later SAC comparison.
8. Baselines — numerical solver, supervised MLP, optional PINN and weak-field approximation.
9. Evaluation — state/curve/deflection error, invariant drift, generalization, failure cases.
10. Ablations — reward and observation choices.
11. Limitations — teacher-guided reward, fixed angular grid, scattering-only first version.
12. Conclusion — what the evidence supports and what remains open.

## 50. Final deliverables

- Existing Schwarzschild simulator preserved.
- Dense DOP853 reference for u and v.
- Reproducible full-trajectory dataset.
- Dataset manifest and ray-level splits.
- Reference trajectory gallery.
- RL trajectory-generation environment.
- Trained RL model and full configuration.
- Generated-vs-reference plots for held-out rays.
- Per-ray numerical and physics metrics.
- Comprehensive tests.
- Later supervised MLP and PINN comparisons.
- Reproducible final report.

---

# PART XI — The actual order we will follow

This is the project order. Do not reorder it into a test-first or baseline-first project.

| Step | Work | Main output |
|---:|---|---|
| 1 | Build the dense DOP853 reference | reference.py |
| 2 | Generate complete photon curves | trajectory_data.py and NPZ |
| 3 | Inspect and render the dataset | reference_trajectory_gallery.png |
| 4 | Freeze train/validation/test ray splits | split_manifest.json |
| 5 | Implement the trajectory-generation environment | trajectory_env.py |
| 6 | Implement and inspect each reward component | reward diagnostics |
| 7 | Train a short PPO smoke run | first checkpoint and logs |
| 8 | Render agent-generated trajectories | generated-vs-reference figures |
| 9 | Improve reward, observations, and training | stable RL policy |
| 10 | Complete the systematic test suite | tests/ |
| 11 | Run held-out and multi-seed evaluation | evaluation CSV and report |
| 12 | Compare supervised MLP and PINN | model comparison |
| 13 | Add weak-field, near-critical, capture, or adaptive extensions | optional research results |

Small smoke checks are allowed during development. The point is that we do not spend the first half of the project writing a large test suite for a system that does not exist yet. Formalize tests after the data, environment, and first learning pipeline are in place.

## Immediate next coding session

We start with the dataset, not tests and not PPO.

1. Add src/lab/reference.py using the existing _rhs and _escape_event.
2. Run build_reference for r0 = 3, 5, and 10.
3. Confirm that each case escapes and returns both u and v.
4. Add src/lab/trajectory_data.py.
5. Generate a small dataset of 20 to 30 rays first.
6. Inspect the NPZ arrays and render the reference curve gallery.
7. Only after the data looks correct, implement the Gymnasium environment.

The first concrete deliverable is **a dataset of complete photon trajectories**, not a test suite, not a scalar deflection predictor, and not an RL step-size controller.

---

# Learning resources

Use resources when they help with the next implementation step. Do not delay project work to finish an entire textbook first.

## Numerical ODEs
- SciPy solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- SciPy integration tutorial: https://docs.scipy.org/doc/scipy/tutorial/integrate.html
- Hairer, Nørsett, and Wanner, *Solving Ordinary Differential Equations I*.

## Reinforcement learning
- Sutton and Barto, *Reinforcement Learning: An Introduction*: http://incompleteideas.net/book/the-book-2nd.html
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

## Physics
Use a reliable general-relativity source for Schwarzschild geometry, null geodesics, the photon sphere, impact parameter, and weak-field bending. Keep all assumptions and units explicit.

---

# Rules for keeping this guide useful

- The project has one main objective: RL-based photon-trajectory generation.
- Update this document when the implementation changes; do not create another competing master guide.
- Code examples must match the actual repository API.
- Explain new code in the chapter where it is introduced.
- Do not mark a milestone complete because code was pasted; inspect outputs and behavior.
- Keep tests systematic but do not let them displace the main build sequence.
- Keep MLP, PINN, weak-field checks, critical sweeps, captured rays, and adaptive integration as later comparisons or extensions.
- Record failed runs and unexpected behavior; do not hide them.
- Every final claim must be supported by reproducible metrics.

**The agent's job is to learn the shape of the light path. We build the dataset, give the agent the ability to generate trajectories, train it, inspect what it learned, and only then run the full scientific test and comparison program.**

---

# Current repository setup and pre-RL workflow

This section records the implementation names and commands that exist in the repository. If an older example above uses a different filename or command, follow this section.

## Environment setup

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux or WSL:

```bash
source .venv/bin/activate
```

Install the project in editable mode with development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The pyproject.toml file is the package/build configuration. The src directory is the Python source root, and modules inside lab use package-relative imports.

## Generate the reference dataset

The reference implementation is src/lab/reference.py. It uses SciPy DOP853 with dense output. The dataset generator is src/lab/trajectory.py, and the command-line entry point is src/generate_data.py.

Start with a small dataset:

```bash
python src/generate_data.py --count 30 --r-min 2.2 --r-max 12
```

The default outputs are:

- data/photon_trajectories.npz — packed numerical arrays.
- data/trajectory_manifest.json — solver settings, units, case IDs, and unresolved cases.

Each ray has a different escape angle and may have a different number of samples. Therefore, the NPZ uses flat arrays and an offsets array: ray i occupies [offsets[i], offsets[i + 1]). The archive stores case_id, r0, phi, u, v, offsets, escape_angle, and deflection_radian. Its schema version is 2.

DOP853 chooses adaptive internal steps. Dense interpolation resamples each reference at the requested fixed angular spacing delta_phi; the exact escape-event angle is always appended as the final sample. A ray that does not escape before max_angle is recorded as unresolved in the manifest and is not stored as a complete ray.

## Inspect the dataset before doing anything with learning

Run:

```bash
python src/inspect_data.py
```

This validates the packed-array structure and saves outputs/reference_trajectory_gallery.png. To also open the figure:

```bash
python src/inspect_data.py --show
```

Inspect whether the curves bend in the expected direction, start at the requested closest approach, and finish at their escape angles. Check the printed radius and deflection ranges and inspect unresolved cases in the JSON manifest.

## Run the current checks

```bash
python -m pytest -q
```

These tests cover the geodesic right-hand side, escape behavior, dense reference interpolation, the conserved quantity, and the packed dataset format. They do not train or evaluate an RL or ML model.

## Scope boundary

This repository preparation includes the physical reference solver, dataset generation and inspection, packaging, import cleanup, and tests for those components. It deliberately does not implement an RL environment, reward function, policy, training loop, neural network, PINN, or any other machine-learning component. Those remain your implementation work.


## Whole-trajectory data splits (pre-learning preparation)

After generating and inspecting the reference curves, freeze a split manifest:

~~~bash
python src/make_splits.py --seed 42
~~~

This writes data/split_manifest.json with 70% training, 15% validation, and 15% test trajectories by default. The split unit is the complete ray case_id, never an individual phi sample. All samples from one photon trajectory therefore remain in exactly one split. Reuse this same manifest for later experiments so different methods are evaluated on identical held-out rays. The splitter also reads the older dataset format that did not store case_id.

No normalization, reward, environment, policy, or model is created by this utility.
