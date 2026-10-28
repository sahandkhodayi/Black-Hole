# Black Hole Lab — Complete Project Guide

> **Purpose**
>
> This is the code-first textbook for the current state of Black Hole Lab. Each chapter explains the goal, mathematics, relevant code, why the code exists, how to verify it, and what to build next.
>
> We will not add a PINN merely to say the project uses deep learning. The goal is a reliable scientific-computing project first, followed by a dataset, a conventional ML baseline, a Physics-Informed Neural Network (PINN), controlled experiments, and a research-style report.
>
> **Important:** the repository is an early prototype. Sections marked **Planned** describe future work, not features that already exist.

---

# 0. Where We Are Right Now

Repository: [sahandkhodayi/Black-Hole](https://github.com/sahandkhodayi/Black-Hole)

## What currently works

- A numerical solver integrates a reduced Schwarzschild photon-trajectory equation.
- The ray is initialized at a chosen closest-approach radius.
- An event stops integration when the inverse-radius variable reaches zero while decreasing.
- The calculated half-trajectory is mirrored around the closest-approach point.
- A Matplotlib visualization shows several trajectories, the event horizon, and the photon sphere.
- Early experiments have been run for closest approaches of 3, 5, and 10 Schwarzschild radii.

## Current repository map

~~~text
Black-Hole/
├── DOCS/
│   └── Black-hole-png/
│       └── image.png
├── outputs/
│   └── first_ray.png
├── src/
│   ├── lab/
│   │   ├── __init__.py
│   │   ├── geodesics.py
│   │   └── plot.py
│   └── simulate.py
├── project.toml
├── README.md
└── ...
~~~

The root-level test.py is currently only a placeholder, not a working test suite.

## Technical debt to address

These are normal next tasks for an early prototype:

1. geodesics.py runs an experiment loop at module import time. Importing a function should not unexpectedly run simulations and print output.
2. The solver prints a calculated deflection value while labelling it “Escape angle”. The formula includes the factor of two and subtracts pi, so the label is misleading.
3. simulate.py says the plot is saved to outputs/first_ray.png, but the current plot function calls plt.show() and does not itself save the figure.
4. project.toml is not the conventional Python build configuration filename. Standard Python packaging tools expect pyproject.toml. We will correct this rather than assuming editable installation already works.
5. There is no proper automated test suite yet.
6. We must validate the physical interpretation and numerical behavior before treating every trajectory as ground truth.

## Our standard

~~~text
Question
  ↓
Physics / design
  ↓
Implementation
  ↓
Run
  ↓
Automated test
  ↓
Numerical / physical validation
  ↓
Experiment and measurement
  ↓
Document
  ↓
Commit
~~~

A feature is not complete merely because it produces a plot.

**Definition of done:** implemented, explained, tested where practical, failure behavior understood, and documented.

---

# 1. Project Goal and Research Direction

The project studies light trajectories around a **nonrotating Schwarzschild black hole**. The first stage is a numerical simulator. Later, we will use simulated data to investigate machine learning and physics-informed learning.

Our eventual research question is:

> Can a Physics-Informed Neural Network learn light-ray trajectories or deflection angles accurately, especially in difficult regimes, and how does it compare with a conventional neural network and a trusted numerical solver?

This question is not answered in advance. A PINN may help in some settings and perform poorly in others. We must measure the result.

## Roadmap

| Stage | Status | Deliverable |
|---|---|---|
| Understand the physics and variables | Foundational / ongoing | Explain the model and units |
| Numerical geodesic solver | Prototype implemented | Reliable trajectories |
| Visualize and compare rays | Prototype implemented | Reproducible plots |
| Deflection function and automated tests | Next | Testable numerical API |
| Validate solver and physical assumptions | Planned | Validation experiments |
| Generate a scientific dataset | Planned | Versioned data and metadata |
| Train a conventional ML baseline | Planned | Baseline model and metrics |
| Implement a PINN | Planned | Physics-residual loss |
| Compare models fairly | Planned | Experiments and results |
| Investigate inverse problems / harder regimes | Future | Research extension |
| Report, reproducibility, and release | Future | Technical report |

We will implement one stage at a time.

---

# 2. The Physical Model

## 2.1 What are we simulating?

A black hole changes the geometry of spacetime. Light follows null geodesics in that geometry. In this project, we use a reduced equation for a photon trajectory in the equatorial plane of a Schwarzschild spacetime.

We are not yet simulating a glowing accretion disk, plasma, a camera image, or a complete relativistic ray-tracing renderer. The first model calculates the path of a light ray.

## 2.2 Schwarzschild radius

The Schwarzschild radius is

\[
r_s = \frac{2GM}{c^2},
\]

where G is the gravitational constant, M is the black-hole mass, and c is the speed of light.

The event horizon of a Schwarzschild black hole lies at r = r_s.

For this project, we use dimensionless units by measuring radius in units of r_s. We can therefore set r_s = 1 in the simulation without specifying a particular black-hole mass.

A normalized radius r = 5 means 5 r_s, not five metres.

## 2.3 Inverse-radius variable

We define

\[
u = \frac{r_s}{r}.
\]

With normalized radius rho = r/r_s, this is u = 1/rho.

| Normalized radius r/r_s | u |
|---:|---:|
| 2 | 0.5 |
| 5 | 0.2 |
| 10 | 0.1 |
| 100 | 0.01 |
| infinity | approaches 0 |

When a ray moves closer, r decreases and u increases. When it moves farther away, r increases and u decreases. When u approaches zero from above, the ray approaches infinite radius in this coordinate description.

The code treats nonpositive u as infinite radius for plotting purposes. This is a numerical representation choice, not a claim that negative inverse radius is a physical part of the trajectory.

## 2.4 The trajectory equation

The model currently uses

\[
\frac{d^2u}{d\phi^2} + u = \frac{3}{2}u^2,
\]

or equivalently,

\[
\boxed{\frac{d^2u}{d\phi^2} = \frac{3}{2}u^2-u}.
\]

Here phi is the angular coordinate around the black hole, measured in radians.

The second derivative describes how the slope du/dphi changes as the angular coordinate changes. The right-hand side is determined by the current value of u.

This is a differential equation: it defines a relationship between a function and its derivatives. It does not directly hand us the whole path. We integrate it numerically.

**Scope note:** this is the reduced photon-orbit equation used by this project under its Schwarzschild assumptions. It is not a universal equation for every black-hole spacetime or every coordinate choice.

---

# 3. From a Second-Order Equation to Solver State

SciPy’s solve_ivp interface solves first-order systems. We therefore convert the second-order equation into two first-order equations.

Define

\[
v = \frac{du}{d\phi}.
\]

Then

\[
\frac{du}{d\phi}=v,
\qquad
\frac{dv}{d\phi}=\frac{3}{2}u^2-u.
\]

The state is

\[
\mathbf{y} =
\begin{bmatrix}
u\\
v
\end{bmatrix}.
\]

At each integration step, the solver needs the current state and the derivative of the state with respect to phi.

This resembles the position/velocity/acceleration pattern from mechanics, but do not confuse v = du/dphi with the physical speed of light. It is the angular derivative of the inverse-radius variable.

## A small mathematical check

If u = 0.2, then

\[
\frac{d^2u}{d\phi^2}
=1.5(0.2)^2-0.2
=0.06-0.2=-0.14.
\]

The second derivative is negative. If the first derivative starts at zero, this tells the solver how the slope initially begins to change.

---

# 4. Code Chapter: RayTrajectory

**File:** src/lab/geodesics.py

The current implementation defines a small data structure. Its cleaned-up form is:

~~~python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RayTrajectory:
    """Numerical samples describing a light-ray trajectory."""

    phi: FloatArray
    u: FloatArray

    @property
    def radius(self) -> FloatArray:
        """Return normalized radius r / r_s."""
        radius = np.full_like(self.u, np.inf)
        np.divide(1.0, self.u, out=radius, where=self.u > 0)
        return radius
~~~

This snippet presents the intended clean form; it is not a claim that the repository already matches every line exactly.

## Why this structure exists

The solver produces arrays. RayTrajectory groups the arrays that belong together:

- phi[i] is an angular sample.
- u[i] is the inverse radius at the same sample.
- radius[i] converts that inverse radius into normalized radial distance.

## Line-by-line explanation

~~~python
@dataclass(frozen=True)
class RayTrajectory:
~~~

A dataclass supplies common data-container behavior. frozen=True prevents rebinding fields after creation. It does not make NumPy arrays deeply immutable.

~~~python
phi: FloatArray
u: FloatArray
~~~

These are type annotations. They communicate expected array types to readers and static-analysis tools; they do not automatically validate all values at runtime.

~~~python
@property
def radius(self) -> FloatArray:
~~~

radius behaves like an attribute: use trajectory.radius, not trajectory.radius().

~~~python
radius = np.full_like(self.u, np.inf)
~~~

Create an output array with the same shape and dtype as u, initially filled with infinity.

~~~python
np.divide(1.0, self.u, out=radius, where=self.u > 0)
~~~

Compute 1/u only where u > 0, writing results into radius. Values where the condition is false remain infinity. This avoids dividing by zero at the escape boundary.

~~~python
return radius
~~~

Return normalized radius r/r_s = 1/u.

## Design limitation

This representation is suitable for plotting an escaping trajectory. It does not itself distinguish every physical event, such as a captured ray, numerical failure, or integration that simply reached its maximum angle. Later, the simulation result should carry explicit status and event metadata.

## Exercise

Explain why radius uses where=self.u > 0 instead of calculating 1.0/self.u directly.

---

# 5. Code Chapter: The Differential-Equation Function

**File:** src/lab/geodesics.py

The core right-hand-side function follows this pattern:

~~~python
def _rhs(phi: float, state: FloatArray) -> tuple[float, float]:
    del phi
    u, du_dphi = state
    d2u_dphi2 = 1.5 * u**2 - u
    return du_dphi, d2u_dphi2
~~~

## Why does _rhs exist?

The integrator repeatedly asks: at this angle and state, what are the derivatives of the state?

Our state is (u, v), where v = du/dphi. Therefore,

\[
\frac{d}{d\phi}
\begin{bmatrix}
u\\
v
\end{bmatrix}
=
\begin{bmatrix}
v\\
1.5u^2-u
\end{bmatrix}.
\]

## Explain each line

- The function receives the independent variable phi and current state.
- del phi makes the unused argument intentional. The equation is autonomous: the right-hand side depends on u, not explicitly on phi.
- u, du_dphi = state unpacks the state vector.
- d2u_dphi2 evaluates the second-order equation at the current state.
- The return order matters: return du_dphi first and d2u_dphi2 second. It must match the ordering in state.

## Common bug: wrong order

Returning (d2u_dphi2, du_dphi) would tell the solver that the derivative of u is the second derivative and that the derivative of v is merely v. The program might run, but it would integrate the wrong system.

## Exercise

For u = 0.1 and v = -0.02, calculate both values _rhs should return. Use the two first-order equations.

---

# 6. Initial Conditions and Closest Approach

The current simulation starts at a chosen closest-approach radius:

~~~python
u0 = 1.0 / closest_approach

solution = solve_ivp(
    _rhs,
    (0.0, max_angle),
    y0=(u0, 0.0),
    method="DOP853",
    rtol=rtol,
    atol=atol,
    events=escape_event,
    max_step=0.02,
)
~~~

## What does closest approach mean?

The closest-approach radius is the minimum radial distance reached by a scattering ray before it heads outward again. For this model, we initialize integration at that turning point and use symmetry to reconstruct the other half of the path.

If closest_approach = 5.0, then

\[
u_0=\frac{1}{5}=0.2,\qquad
\left.\frac{du}{d\phi}\right|_{\phi=0}=0.
\]

The initial state is (0.2, 0.0).

The zero derivative is not the physical speed of light being zero. It means radial distance is momentarily at a turning point with respect to the angular coordinate.

## Why reject values at or below 1.5?

The Schwarzschild photon sphere is at

\[
r_{\text{ph}}=\frac{3GM}{c^2}=\frac{3}{2}r_s=1.5r_s.
\]

Circular photon orbits exist at this radius. Near the critical scattering trajectory, the ray can bend by a very large amount. A function that assumes every ray escapes needs careful physical constraints.

The current validation rule is a conservative restriction for the prototype, not a complete classifier for all possible photon trajectories. Future work should distinguish invalid inputs from valid-but-captured rays and near-critical scattering rays.

## Input validation

The current function checks that closest_approach is finite and greater than 1.5, max_angle is finite and positive, and rtol and atol lie between zero and one. Tests should confirm each behavior.

---

# 7. solve_ivp and Numerical Integration

SciPy’s solve_ivp integrates an initial-value problem. It does not know anything about black holes; it only knows the system of differential equations we provide.

| Argument | Meaning |
|---|---|
| _rhs | Defines the derivative of the state |
| (0.0, max_angle) | Angular interval for integration |
| y0=(u0, 0.0) | Initial state at closest approach |
| method="DOP853" | Explicit high-order Runge–Kutta method |
| rtol | Relative tolerance for local error control |
| atol | Absolute tolerance for local error control |
| events=escape_event | Stops at the defined escape crossing |
| max_step=0.02 | Caps the angular size of an integration step |

## What is a numerical solver doing?

The differential equation gives a local rule. A numerical integrator uses that rule to estimate the solution over a finite interval. It chooses step sizes and evaluates the derivative multiple times to control error.

The returned values are approximate numerical samples, not an exact symbolic solution.

## Tolerances are not global error guarantees

rtol and atol guide local error control. Tighter tolerances often increase computation and may improve the result, but they do not automatically prove that every reported physical quantity is accurate to that many digits.

We must test convergence of quantities we care about, such as deflection angle, by comparing results at different tolerances and maximum step sizes.

DOP853 is a high-order explicit Runge–Kutta method suitable for many high-accuracy non-stiff problems. It is a practical starting point, not a guarantee of optimality for every future regime.

---

# 8. Escape Events

The prototype uses an event function like this:

~~~python
def escape_event(phi, state):
    u, du_dphi = state
    return u

escape_event.terminal = True
escape_event.direction = -1
~~~

An event is a condition monitored by the solver. Here, the event function returns u, so an event occurs when u crosses zero.

- terminal = True asks the solver to stop after detecting the event.
- direction = -1 asks it to detect crossings in the decreasing direction.

This matches an escaping ray whose positive inverse radius approaches zero. Continuing into negative u would not represent the intended escaping branch of this coordinate model.

## Floating-point detail

The event state may be a tiny negative number such as -1e-17 rather than exactly zero. This is normal floating-point behavior, not evidence of a physically meaningful negative-radius region.

## What this event does not tell us

It does not classify capture. It only detects the chosen zero crossing. A robust simulator should report whether the ray escaped, was captured, reached the integration limit, or failed numerically.

That classification is a planned improvement.

---

# 9. Reconstructing the Full Trajectory by Symmetry

The integration starts at the closest-approach point with phi = 0 and initial derivative zero. The prototype mirrors the computed half-path:

~~~python
phi_out = solution.t
u_out = solution.y[0]

phi = np.concatenate((-phi_out[:0:-1], phi_out))
u = np.concatenate((u_out[:0:-1], u_out))
~~~

## Why does this work?

For this symmetric scattering setup, the trajectory is mirrored around the radial line through closest approach. If the solver gives the branch for nonnegative phi, we can reflect its samples to negative angles.

The slice [:0:-1] reverses the samples while excluding index zero. This avoids duplicating the closest-approach sample when concatenating both halves. The same operation is applied to u_out so each angle remains paired with the correct inverse radius.

Mirroring is appropriate only if the physical setup and boundary conditions are symmetric. Do not automatically reuse it for asymmetric setups or external perturbations.

## Exercise

Why use [:0:-1] rather than [::-1]? What would happen at the join if the closest-approach sample appeared twice?

---

# 10. From Polar Coordinates to a Plot

The path is represented by (r, phi). Matplotlib plots Cartesian coordinates, so we convert:

\[
x=r\cos\phi,\qquad y=r\sin\phi.
\]

~~~python
r = trajectory.radius
phi = trajectory.phi

visible = np.isfinite(r) & (r <= 30)

x = r[visible] * np.cos(phi[visible])
y = r[visible] * np.sin(phi[visible])

ax.plot(x, y)
~~~

- trajectory.radius converts inverse radius into normalized radius.
- np.isfinite(r) excludes infinite radii at the escape boundary.
- r <= 30 keeps the graph within a useful plotting range.
- visible selects matching values from both arrays.
- cosine and sine convert polar coordinates into Cartesian coordinates.
- ax.plot draws the visible samples as a connected line.

The cutoff at 30 is a visualization choice, not a physical cutoff in the trajectory.

## Event horizon and photon sphere

In normalized units:

- Event horizon: r/r_s = 1.
- Photon sphere: r/r_s = 1.5.

The plot draws circles at these radii. The event horizon is a boundary in the Schwarzschild geometry; the photon sphere is a radius where circular photon orbits are possible. Neither circle is a physical wall that the simulated ray bounces off.

## Current plotting caveat

The current plot function calls plt.show(). If the program prints that a file was saved, verify that fig.savefig(...) is actually implemented before trusting that message. We will make saving explicit and tested during plotting cleanup.

---

# 11. Deflection Angle: What We Measure

The angle at which the ray escapes from the numerical half-path is not itself the total deflection angle.

For the symmetric setup, if phi_escape is the angular distance from the closest-approach point to the asymptotic outgoing direction, then

\[
\boxed{\alpha=2\phi_{\text{escape}}-\pi}.
\]

This angle is in radians. Convert it to degrees with np.degrees(alpha).

## Important terminology

- **Escape angle:** the angular coordinate at which the event function detects u = 0, as approximated by the numerical solver.
- **Deflection angle:** the change in asymptotic direction relative to a straight path.
- **Closest-approach radius:** minimum radial distance of the selected scattering trajectory.
- **Impact parameter:** a different quantity defined from the ray’s asymptotic incoming geometry and conserved quantities. Do not equate it with closest-approach radius.

## A cleaner function to implement next

The current repository prints a calculated deflection value from inside simulate_ray, mixing simulation and presentation. A better design is to keep the simulator responsible for producing a trajectory and put derived quantities in a separate function.

One possible future API is:

~~~python
def deflection_angle(escape_angle: float) -> float:
    """Convert a symmetric half-trajectory escape angle to radians."""
    return 2.0 * escape_angle - np.pi
~~~

This is a design proposal, not yet a verified implementation in the repository. Before using it, the project should store escape metadata explicitly rather than reconstructing it from mirrored trajectory arrays.

## Results from the early experiments

The early runs reported approximate half-path escape angles:

| Closest approach | Escape angle |
|---:|---:|
| 3 r_s | 2.0782 rad |
| 5 r_s | 1.8209 rad |
| 10 r_s | 1.6817 rad |

Using alpha = 2 phi_escape - pi, the corresponding deflections are approximately 58.2°, 28.6°, and 12.7°. These are prototype results, not a complete validation study.

For these escaping examples, the ray that passes closer bends more strongly. Do not extrapolate this trend into every regime without checking the model and assumptions.

---

# 12. Controlled Experiments

A plot is useful, but a scientific project also needs measurements.

## Experiment A — Vary closest approach

Current experiment values:

~~~python
closest_approaches = [3.0, 5.0, 10.0]
~~~

Keep solver settings fixed and change only the closest-approach distance. Record input distance, whether the escape event was detected, escape angle, deflection angle, number of solver samples, solver status, tolerances, and maximum step.

Expected qualitative result for these escaping examples: larger closest-approach distances produce weaker deflection.

## Experiment B — Numerical convergence

Run the same initial condition with different tolerances, for example:

- rtol=1e-7, atol=1e-9
- rtol=1e-9, atol=1e-11
- rtol=1e-11, atol=1e-13

Also investigate changing max_step. Compare the final deflection estimate. Do not report many digits unless the convergence study supports them.

## Experiment C — Input validation

Test invalid values such as closest_approach = 1.5, closest_approach = 0, closest_approach = NaN, max_angle = 0, and negative tolerances.

Expected behavior is a clear ValueError, not an obscure solver exception or misleading plot.

## Experiment D — Event handling

Check that a normal escaping case detects the escape event and that the event direction is decreasing. Check behavior when the solver reaches max_angle without detecting the event. These outcomes must not be silently treated as equivalent.

---

# 13. Testing Strategy — Planned

The repository does not yet have a proper automated test suite. The root-level test.py is not a usable test suite. The intended test layout is:

~~~text
tests/
├── test_geodesics.py
├── test_deflection.py
└── test_plot.py
~~~

We will add tests gradually; do not create every file empty just to match the tree.

## Unit tests

Test small behaviors independently:

- the right-hand-side function returns the expected derivatives,
- radius maps positive u to 1/u,
- nonpositive u maps to infinity for plotting,
- invalid inputs raise the expected exception,
- the deflection formula has the correct factor of two.

## Integration tests

Test the complete simulation:

- the solver succeeds for a known escaping case,
- the escape event is detected,
- arrays have matching lengths,
- values are finite where expected,
- the minimum normalized radius agrees with the requested closest approach within numerical tolerance.

## Scientific validation

A passing test is not proof that the physics is correct. Scientific validation also needs comparison with analytical limits, independent calculations, or trusted references where available.

For example, in the weak-field regime, the leading-order deflection of a Schwarzschild light ray is

\[
\alpha \approx \frac{2r_s}{b},
\]

where b is the impact parameter and the ray is far from the strong-field critical region. Because b is not exactly the closest-approach radius, compare like with like and account for the relationship between them before treating this as a direct unit test.

After adding real tests and ensuring pytest is installed, run:

~~~bash
python -m pytest -q
~~~

Do not mark testing complete until the command runs against real assertions.

---

# 14. Engineering Cleanup Roadmap

Before dataset generation, make the simulator a stable API.

## Task 1 — Remove import-time experiments

The loop at the bottom of geodesics.py should move into an experiment script or a main() function guarded by:

~~~python
if __name__ == "__main__":
    main()
~~~

A module import should define functions and classes, not unexpectedly run simulations.

## Task 2 — Separate computation from output

simulate_ray() should return data. It should not print user-facing messages on every call. An experiment or plotting script should decide what to print.

## Task 3 — Record termination metadata

Return the event angle and status from the solver. Distinguish escape event detected, maximum integration angle reached, solver failure, and a future captured-ray condition.

## Task 4 — Correct package configuration

The repository currently uses project.toml. Standard Python build tooling expects pyproject.toml. Migrate the configuration carefully and test installation in a clean virtual environment.

## Task 5 — Save figures explicitly

The plot function should accept an output path or return the figure/axes. If saving is requested, use fig.savefig(...). Test that the file is actually created.

## Task 6 — Add automated tests

Create the test suite described above. Make tests deterministic and avoid relying on visual judgment alone.

---

# 15. Dataset Design — Planned

Once the simulator is reliable, we will generate a dataset.

## What is one sample?

A first dataset could use one row per simulated ray, with fields such as closest-approach radius in units of r_s, solver tolerances, escape/capture/termination status, escape angle, deflection angle, and numerical diagnostics.

A trajectory dataset could instead store many (phi, u) samples per ray. These two dataset types answer different learning questions, so choose deliberately rather than flattening everything into one arbitrary table.

## Avoid data leakage

If many points come from the same trajectory, do not randomly scatter those points across train and test sets if the evaluation question is whether the model generalizes to unseen trajectories. Split by trajectory or parameter regime as appropriate.

## Save metadata

Each dataset should record generation-code version, solver method, tolerances, sampling policy, parameter ranges, units and normalization, random seed if used, and event/termination rules.

Without metadata, it becomes difficult to reproduce or interpret results.

## Difficult regimes

Near the critical trajectory, the ray may spend a long angular interval near the photon sphere and bend strongly. Such cases can be numerically demanding and scientifically interesting. Sampling them requires careful event handling and validation; they should not be generated blindly.

---

# 16. Machine-Learning Baseline — Planned

Before a PINN, build a conventional neural-network baseline. This answers whether a data-driven model can learn the chosen mapping at all.

Potential tasks include:

1. **Scalar regression:** a well-defined ray parameter → deflection angle.
2. **Trajectory prediction:** initial conditions and angular coordinate → u(phi), with a clearly specified representation.
3. **Classification:** predict a defined termination class, once the simulator correctly models more than escaping rays.

Start with scalar regression because it is simpler to evaluate. It is not necessarily the most scientifically interesting final task.

## Baseline workflow

1. Define inputs and targets mathematically.
2. Create train, validation, and test splits.
3. Normalize inputs where appropriate.
4. Train a simple baseline.
5. Compare predictions against numerical-solver outputs.
6. Report error metrics on held-out cases.
7. Inspect errors by parameter regime, not only one aggregate score.

Useful metrics for angle regression include MAE and RMSE. Metrics do not replace residual plots or checks for systematic failure.

A model that performs well on random test points sampled from a narrow interval may still fail near the critical region or outside the training distribution.

---

# 17. Physics-Informed Neural Network (PINN) — Planned

A PINN incorporates the differential equation into its training objective. It does not automatically know the correct physics, and adding a physics loss does not guarantee better predictions.

Suppose a neural network approximates u_theta(phi). The differential-equation residual is

\[
\mathcal{R}_\theta(\phi)
=
\frac{d^2u_\theta}{d\phi^2}
+u_\theta
-\frac{3}{2}u_\theta^2.
\]

For an exact solution of the chosen equation, this residual is zero wherever the equation applies.

A simple residual loss is

\[
\mathcal{L}_{\text{physics}}
=
\frac{1}{N_f}\sum_{i=1}^{N_f}
\left|\mathcal{R}_\theta(\phi_i)\right|^2.
\]

If training data are available, a data loss might be

\[
\mathcal{L}_{\text{data}}
=
\frac{1}{N_d}\sum_{i=1}^{N_d}
\left|u_\theta(\phi_i)-u_i\right|^2.
\]

A combined objective could be

\[
\mathcal{L}
=
\lambda_d\mathcal{L}_{\text{data}}
+\lambda_f\mathcal{L}_{\text{physics}}
+\lambda_b\mathcal{L}_{\text{boundary}},
\]

where the boundary/initial-condition loss enforces the required initial conditions. The weights lambda control the contribution of each term.

## Critical implementation detail

A PINN needs derivatives of network output with respect to its input, including a second derivative here. In PyTorch, this generally requires automatic differentiation with graph construction configured correctly. We will study that when we reach this chapter.

## Why compare against a conventional network?

The conventional model tells us what data-driven learning can achieve without the physics residual. The PINN tests whether adding the equation improves accuracy, data efficiency, generalization, or stability.

A fair comparison uses the same held-out cases and reports training setup and computational cost. If the PINN performs worse, that is still a useful experimental result.

---

# 18. Experiments and Research Reporting — Planned

Every serious experiment should begin with a question and a prediction.

**Question:** Does adding the differential-equation residual improve predictions for trajectories outside the training interval?

**Hypothesis:** The PINN may generalize better because its objective penalizes violations of the governing equation.

**Test:** Train a conventional network and a PINN on the same training set. Evaluate both on held-out parameter ranges. Compare MAE/RMSE, residuals, and runtime.

**Possible outcome:** The PINN may improve physical consistency but be harder to optimize or less accurate on a particular target. Report what the experiments show rather than forcing the expected result.

A report should include model and equation, assumptions and limitations, solver settings, dataset construction, train/validation/test protocol, model architectures, losses and optimizer, metrics, plots and failure cases, hardware and runtime, and reproducibility instructions.

---

# 19. Glossary

| Term | Meaning in this project |
|---|---|
| Schwarzschild black hole | Idealized, nonrotating, uncharged black hole |
| Schwarzschild radius r_s | Event-horizon radius in the Schwarzschild solution |
| Event horizon | Boundary beyond which outward escape is impossible in the classical model |
| Photon sphere | Radius r = 1.5 r_s for circular photon orbits in Schwarzschild spacetime |
| Geodesic | A path followed freely through spacetime; light follows null geodesics |
| Closest approach | Minimum radial distance of the selected scattering trajectory |
| Impact parameter | Asymptotic geometric/conserved-quantity parameter; not identical to closest approach |
| Inverse-radius variable u | u = r_s/r |
| Initial condition | State that selects one solution of the differential equation |
| Numerical integration | Approximate solution of a differential equation by a numerical method |
| Event function | Condition monitored by the integrator to locate a particular crossing |
| Deflection angle | Change in asymptotic direction relative to an undeflected path |
| Baseline model | Simple reference model used for comparison |
| PINN | Neural network trained with a loss containing physics-based constraints |
| Residual | Difference between the equation’s two sides when evaluated on a predicted solution |
| Generalization | Performance on cases not used for training |

---

# 20. Working Rules and Learning Cycle

For every new chapter:

1. Read the goal and physics.
2. Inspect the existing file before editing.
3. Make one focused change.
4. Run the smallest relevant experiment.
5. Explain the changed lines in your own words.
6. Add or update a test.
7. Inspect failure cases.
8. Record the result and commit.

Do not copy a full implementation without understanding it. The goal is to understand the code’s purpose, assumptions, data flow, mathematics, and failure modes.

## Current next task

The immediate next milestone is **make the simulator testable**:

- remove the experiment loop from module import time,
- separate printed output from simulation,
- represent escape metadata explicitly,
- implement a standalone deflection-angle calculation,
- add pytest tests for the right-hand side, radius conversion, input validation, event detection, and deflection formula,
- verify numerical convergence.

Only after those checks pass should we start generating the ML dataset.

---

# 21. Definition of Done for the First Milestone

The first milestone is complete when:

- [ ] The solver is a reusable function with no import-time experiments.
- [ ] The result records enough metadata to determine why integration stopped.
- [ ] The deflection angle is calculated in a separate, documented function.
- [ ] Automated tests cover basic mathematical and API behavior.
- [ ] Numerical convergence has been measured.
- [ ] The plotting function labels and saves outputs accurately.
- [ ] Setup instructions work in a clean environment.
- [ ] The repository documents model assumptions and known limitations.

**This is the beginning of the project guide, not the end of the project.** Add new chapters as each milestone is implemented, keeping the documentation aligned with real code rather than pretending future features already exist.
