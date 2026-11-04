# Black Hole Lab — Integrated Guide: RK4 to PPO

**Purpose:** a single implementation path from the repository's current custom RK4 solver to a measured reinforcement-learning experiment.

**Research question**

> Can a reinforcement-learning policy learn to choose numerical integration step sizes for Schwarzschild photon geodesics, balancing computational cost against numerical accuracy?

The agent does not steer the photon or change the physics. It controls the numerical solver's next step size. The geodesic equation determines the physical trajectory; the policy determines how the numerical approximation advances.

This guide is written for the current repository state: the Schwarzschild ODE and DOP853 simulation exist, a classical RK4 one-step function exists, and a comparison script exists. The next task is numerical validation. PPO is deliberately near the end, not the beginning.

---

# 1. Project state, boundaries, and workflow

## 1.1 What already exists

The inspected repository contains:

- **src/lab/geodesics.py** — the physical ODE, escape event, DOP853 simulation, and RayTrajectory.
- **src/lab/integrators.py** — the classical RK4 one-step function.
- **src/lab/compare_plot.py** — an early RK4 versus DOP853 comparison and plot.
- **project.toml** — project metadata as currently named in the repository.

The geodesic module exposes the following relevant API:

- **RayTrajectory.phi**
- **RayTrajectory.u**
- **RayTrajectory.escaped**
- **RayTrajectory.escape_angle**
- **RayTrajectory.nfev**
- **RayTrajectory.deflection_radian**
- **RayTrajectory.deflection_angle**
- **simulate_ray**
- **simulate_ray_raw**
- **_rhs**
- **_escape_event**

The existing RK4 function is called **rk4_step(rhs, phi, state, step_size)**.

The current comparison script is a prototype, not a final benchmark. It:
- imports geodesics and integrators as top-level modules, which is inconsistent with the package layout;
- stops RK4 after it reaches u <= 0, without accurately localizing the escape event;
- compares interpolated u values on a shared interval;
- does not compare v, escape-angle error, or deflection-angle error;
- does not yet provide a complete convergence study.

Do not replace functioning code just because a longer example appears in this guide. Add tests first, inspect the current implementation, then make the smallest useful change.

## 1.2 Research architecture

As the project grows, aim for this structure:

~~~text
src/
  lab/
    __init__.py
    geodesics.py
    integrators.py
    reference.py
    fixed_step.py
    convergence.py
    adaptive.py
    rl_env.py
    policies.py

scripts/
  run_convergence.py
  evaluate_baselines.py
  train_ppo.py
  evaluate_ppo.py

tests/
  test_integrators.py
  test_geodesics.py
  test_reference.py
  test_fixed_step.py
  test_adaptive.py
  test_rl_env.py

results/
  convergence.csv
  convergence.png
  baseline_comparison.csv
  ppo_evaluation.csv
~~~

Create each module only when its milestone begins. Do not create empty files for every future milestone now.

## 1.3 Correct project metadata before installing the package

The repository currently has a file named **project.toml**. Python's standard build frontend looks for **pyproject.toml**. Check the actual checkout; if it is still named project.toml, rename it:

~~~bash
git mv project.toml pyproject.toml
~~~

Then create and activate a virtual environment:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Run tests from the repository root:

~~~bash
python -m pytest -q
~~~

The RL dependencies are not needed yet. Install them only when the environment milestone begins:

~~~bash
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

---

# 2. The physical model — what the solver is integrating

The project uses normalized Schwarzschild radius, so the event horizon is at

\[
r_s=1.
\]

Define inverse radius

\[
u(\phi)=\frac{1}{r(\phi)}.
\]

For the photon orbit in the model used by this repository,

\[
\frac{d^2u}{d\phi^2}=\frac{3}{2}u^2-u.
\]

Introduce

\[
v=\frac{du}{d\phi}.
\]

The second-order equation becomes the first-order system

\[
u'=v,\qquad v'=\frac{3}{2}u^2-u.
\]

With the state vector

\[
\mathbf y =
\begin{bmatrix}u\\v\end{bmatrix},
\]

we integrate

\[
\mathbf y'=f(\phi,\mathbf y).
\]

At the point of closest approach, symmetry gives

\[
u(0)=\frac{1}{r_0},\qquad v(0)=0.
\]

A scattering ray that comes from infinity and escapes requires

\[
r_0>1.5.
\]

The critical photon orbit is at 1.5 Schwarzschild radii. It is not an ordinary escaping scattering trajectory.

The escape event is the outgoing crossing of

\[
u=0,
\]

which corresponds to radius tending to infinity. The existing event function returns u and is configured to terminate on a crossing in the negative direction.

If the positive-side escape angle is \(\phi_{\mathrm{escape}}\), the deflection is

\[
\alpha=2\phi_{\mathrm{escape}}-\pi.
\]

This project must evaluate error in the escape angle and deflection, not merely whether two plotted curves look similar.

## 2.1 Existing physics implementation

The repository's current physical right-hand side is equivalent to:

~~~python
def _rhs(phi, state):
    del phi
    u, du_dphi = state
    d2u_dphi2 = 1.5 * u**2 - u
    return du_dphi, d2u_dphi2
~~~

The escape event is equivalent to:

~~~python
def _escape_event(phi, state):
    u, du_dphi = state
    return u


_escape_event.terminal = True
_escape_event.direction = -1
~~~

These are descriptions of the existing implementation, not instructions to duplicate the functions in another module. The numerical solvers should share one source of truth for the physics.

## 2.2 Existing reference solver

The current **simulate_ray** uses SciPy's DOP853 solver with tight tolerances and the escape event. Treat it as a high-accuracy numerical reference, not an exact analytical solution.

Earlier runs gave the following escape-angle values:

| Closest approach r0 | Observed escape angle |
|---:|---:|
| 3.0 | 2.078234 |
| 5.0 | 1.820914 |
| 10.0 | 1.681734 |

These are useful regression values from the current implementation. They are not exact constants; tolerances and solver versions can change the last digits.

---

# 3. Milestone 1 — validate RK4 independently

Do not begin with the black-hole equation. First test the integrator on an ODE whose exact solution is known.

Use

\[
y'=y,\qquad y(0)=1,
\]

whose exact solution is

\[
y(t)=e^t.
\]

This separates bugs in RK4 from bugs in the physical model.

## 3.1 Understand the RK4 step

For the system

\[
\mathbf y'=f(t,\mathbf y),
\]

classical RK4 computes

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

then

\[
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\]

The current **rk4_step** implementation already follows this structure, converts the state to a NumPy array, validates the step size and state shape, and returns a new state.

## 3.2 Add unit tests

Create **tests/test_integrators.py**:

~~~python
import numpy as np
import pytest

from lab.integrators import rk4_step


def exponential_rhs(t, state):
    return state


def test_one_step_matches_exponential():
    initial = np.array([1.0])
    h = 0.1

    actual = rk4_step(exponential_rhs, 0.0, initial, h)
    expected = np.array([np.exp(h)])

    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9)


def test_input_state_is_not_mutated():
    initial = np.array([1.0, 2.0])
    before = initial.copy()

    rk4_step(exponential_rhs, 0.0, initial, 0.01)

    assert np.array_equal(initial, before)


def test_rhs_is_called_four_times():
    calls = []

    def rhs(t, state):
        calls.append(t)
        return state

    rk4_step(rhs, 0.0, np.array([1.0]), 0.1)

    assert len(calls) == 4


@pytest.mark.parametrize("step", [0.0, -0.1, np.inf, np.nan])
def test_invalid_step_is_rejected(step):
    with pytest.raises(ValueError):
        rk4_step(exponential_rhs, 0.0, np.array([1.0]), step)


def test_non_vector_state_is_rejected():
    with pytest.raises(ValueError):
        rk4_step(exponential_rhs, 0.0, np.ones((2, 2)), 0.1)
~~~

Run:

~~~bash
python -m pytest tests/test_integrators.py -q
~~~

If the project is not installed yet, fix the metadata/package installation before changing imports to work around it.

## 3.3 Test fourth-order convergence

A single step test is not enough. Integrate the exponential ODE to the same final time T using several step sizes. Compare the numerical result to exp(T).

For a fourth-order method, the global error should asymptotically shrink by about 16 when h is halved:

\[
E(h)\approx Ch^4.
\]

The observed order is

\[
p=\log_2\left(\frac{E(h)}{E(h/2)}\right).
\]

A measured order near four is the expected result once the step sizes are sufficiently small. At coarse steps, the asymptotic relationship may not hold; at extremely small steps, floating-point error can dominate.

**Milestone 1 is complete when:** all tests pass, the state is not mutated, one step uses exactly four RHS calls, and the measured order approaches four.

---

# 4. Milestone 2 — protect the geodesic model with tests

Create **tests/test_geodesics.py**:

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


def test_deflection_exists_after_escape():
    ray = simulate_ray(closest_approach=5.0)

    assert ray.deflection_radian is not None
    assert np.isfinite(ray.deflection_radian)


@pytest.mark.parametrize("r0", [1.0, 1.5, 0.0, -2.0, np.inf, np.nan])
def test_invalid_closest_approach_is_rejected(r0):
    with pytest.raises(ValueError):
        simulate_ray(closest_approach=r0)


def test_closer_scattering_ray_has_larger_deflection():
    near = simulate_ray(closest_approach=3.0)
    far = simulate_ray(closest_approach=10.0)

    assert near.deflection_radian > far.deflection_radian
~~~

Run:

~~~bash
python -m pytest tests/test_geodesics.py -q
~~~

The final test protects a trend observed for these selected rays. It does not establish a theorem for every possible parameter value.

**Milestone 2 is complete when:** the existing physics behavior is protected by regression tests and invalid parameters fail clearly.

---

# 5. Milestone 3 — create a reusable DOP853 reference

Convergence tests need the reference solution at arbitrary angles. The current raw SciPy result does not necessarily store points at the angles requested by the RK4 solver.

SciPy's dense-output option provides an interpolant over the interval actually integrated.

Create **src/lab/reference.py**:

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
        if not self.solution.success:
            raise RuntimeError(self.solution.message)

        if not self.solution.t[0] <= phi <= self.solution.t[-1]:
            raise ValueError("Requested angle is outside the reference interval")

        return np.asarray(self.solution.sol(phi), dtype=float)


def build_reference(closest_approach, max_angle=8.0):
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")

    result = solve_ivp(
        _rhs,
        (0.0, max_angle),
        y0=(1.0 / closest_approach, 0.0),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        max_step=0.01,
        events=_escape_event,
        dense_output=True,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return ReferenceRay(result, closest_approach)
~~~

The code intentionally reuses the existing physical RHS and event instead of defining a second version. They are private names in the current module; if the physics module is later cleaned up, consider exposing a stable public interface.

The dense interpolant must never be queried beyond the event or the final integration interval.

Add tests verifying:
- the reference escapes for a representative scattering ray;
- state_at(0) returns [1/r0, 0];
- state_at rejects angles outside the integrated interval;
- reference output is finite before escape.

**Why DOP853 is still not exact:** it is another numerical method. Tight tolerances make it a suitable reference for this experiment, but its own error should be checked to ensure it is much smaller than the errors being attributed to RK4.

---

# 6. Milestone 4 — implement fixed-step RK4 trajectories

The current RK4 function advances one state by one step. The trajectory runner should own the integration loop and diagnostics.

Create **src/lab/fixed_step.py**:

~~~python
from dataclasses import dataclass

import numpy as np

from lab.geodesics import _rhs
from lab.integrators import rk4_step


@dataclass(frozen=True)
class FixedStepRay:
    phi: np.ndarray
    states: np.ndarray
    steps: int
    rhs_evaluations: int
    escaped: bool
    escape_angle: float | None

    @property
    def u(self):
        return self.states[:, 0]

    @property
    def v(self):
        return self.states[:, 1]


def integrate_fixed(closest_approach, step_size, max_angle=8.0):
    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive")

    phi = 0.0
    state = np.array([1.0 / closest_approach, 0.0], dtype=float)

    angles = [phi]
    states = [state.copy()]
    steps = 0
    escaped = False
    escape_angle = None

    while phi < max_angle:
        h = min(step_size, max_angle - phi)
        next_state = rk4_step(_rhs, phi, state, h)
        next_phi = phi + h
        steps += 1

        if state[0] > 0.0 and next_state[0] <= 0.0:
            fraction = state[0] / (state[0] - next_state[0])
            fraction = float(np.clip(fraction, 0.0, 1.0))
            escape_angle = phi + fraction * h
            escaped = True
            # This is a plotting/bookkeeping boundary point, not a true
            # finite-radius physical state at infinity.
            angles.append(escape_angle)
            states.append(np.array([0.0, next_state[1]], dtype=float))
            break

        phi = next_phi
        state = next_state
        angles.append(phi)
        states.append(state.copy())

    return FixedStepRay(
        phi=np.asarray(angles),
        states=np.asarray(states),
        steps=steps,
        rhs_evaluations=4 * steps,
        escaped=escaped,
        escape_angle=escape_angle,
    )
~~~

The linear interpolation in this first version is provisional. It gives a basic estimate of the crossing inside the final step, but it is not a high-order event locator. Improve it in the next section.

Test:
- initial state;
- fixed-step count;
- RHS count equals four times the RK4 step count;
- clipping of the final step at max_angle;
- escape on representative cases;
- invalid closest approach and step sizes.

---

# 7. Milestone 5 — accurate event localization

A fixed RK4 step may go from positive u to negative u. Stopping at the first negative state makes the escape angle depend on step size.

Given the final step's starting state, bracket the event between 0 and h. Use bisection to find the fraction of the step whose integrated u is closest to zero.

~~~python
def locate_escape(phi, state, h, rhs, rk4_step, iterations=30):
    low = 0.0
    high = h

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        candidate = rk4_step(rhs, phi, state, mid)

        if candidate[0] > 0.0:
            low = mid
        else:
            high = mid

    return phi + 0.5 * (low + high), iterations * 4
~~~

Preconditions:
- u at the beginning of the step is positive;
- u at the end of the full step is non-positive;
- the crossing is bracketed by this step.

The returned work count is the bisection integrations' RHS evaluations. Add it to the normal trajectory work. If you later use a more sophisticated root finder, count its solver calls too.

This helper re-integrates from the same initial state for every midpoint. Do not advance from one midpoint candidate to the next; bisection is locating the crossing within the original step.

**Acceptance:** the escape-angle estimate converges as h is reduced, and event-localization work is included in every reported evaluation count.

---

# 8. Milestone 6 — run a convergence experiment

This is the first quantitative numerical-analysis result.

Start with:

~~~python
step_sizes = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
closest_approaches = [3.0, 5.0, 10.0]
~~~

For each pair, integrate with fixed RK4 and build a dense DOP853 reference.

Compare both components of the state:

\[
E_u=\max_\phi |u_{\mathrm{RK4}}(\phi)-u_{\mathrm{ref}}(\phi)|,
\]

\[
E_v=\max_\phi |v_{\mathrm{RK4}}(\phi)-v_{\mathrm{ref}}(\phi)|.
\]

Use a shared angular interval ending before either solution's escape event. Evaluate DOP853 with its dense interpolant at the RK4 sample angles and on a dense common grid. Interpolation should not conceal an event mismatch.

Compare the two terminal quantities:

\[
E_\phi=|\phi_{\mathrm{escape,RK4}}-
\phi_{\mathrm{escape,ref}}|,
\]

\[
E_\alpha=|\alpha_{\mathrm{RK4}}-\alpha_{\mathrm{ref}}|.
\]

Because alpha equals twice the escape angle minus pi, consistent measurements should satisfy

\[
E_\alpha=2E_\phi.
\]

This identity is a useful implementation check.

## 8.1 Metrics to save

| Metric | Meaning |
|---|---|
| closest_approach | initial physical case |
| step_size | fixed h |
| steps | number of full RK4 steps |
| rhs_evaluations | all RHS calls, including event localization |
| max_u_error | maximum absolute u error |
| max_v_error | maximum absolute v error |
| escape_angle_error | absolute escape-angle error |
| deflection_error | absolute deflection error |
| runtime_seconds | wall-clock runtime |

Use **time.perf_counter()** and repeat each configuration several times. Report the median runtime. Runtime is machine- and load-dependent; RHS evaluation count is a more reproducible computational-work metric.

## 8.2 Plot and interpret convergence

Create a log-log plot of error against h. In the asymptotic region, RK4 global state error should have slope close to four.

Do not blindly fit all data points:
- coarse steps may not be in the asymptotic regime;
- tiny steps may be limited by round-off or reference error;
- event localization can have a different error behavior from state integration.

**Milestone complete when:** errors decrease as h decreases, the observed state-error order approaches four in an appropriate region, escape/deflection errors decrease, and CSV plus plots can be regenerated by one command.

## 8.3 Fair solver comparison

An earlier run for r0=5 and h=0.01 reported:
- 182 RK4 steps;
- 728 RK4 RHS evaluations;
- 1109 DOP853 RHS evaluations;
- maximum sampled u difference around 6.70e-6.

This does not establish that RK4 is generally more efficient. It compares one configuration, and the accuracy targets were not matched.

A fair comparison asks how much work each solver needs to reach the same target error. For example, compare minimum RHS evaluations needed to get deflection error below 1e-5, then repeat with stricter targets.

---

# 9. Milestone 7 — implement a conventional adaptive RK4 baseline

Do not jump from fixed RK4 directly to RL. A conventional adaptive controller is the baseline that a learned controller must try to improve.

## 9.1 Step doubling

For a trial step h:
1. Calculate one full RK4 step of size h.
2. Calculate two RK4 half-steps of size h/2.
3. Compare the results.
4. Accept the two-half-step result if the estimated error is acceptable.
5. Otherwise reject the trial and try a smaller step.

For fourth-order RK4, the step-doubling difference is approximately 15 times the local error of the more accurate two-half-step result:

\[
e\approx\frac{y_{\mathrm{two\ half}}-
y_{\mathrm{one\ full}}}{15}.
\]

The attempt uses three RK4 steps, hence 12 RHS evaluations.

Create **src/lab/adaptive.py**:

~~~python
import numpy as np

from lab.geodesics import _rhs
from lab.integrators import rk4_step


def rk4_step_doubling(phi, state, h, atol=1e-9, rtol=1e-7):
    full = rk4_step(_rhs, phi, state, h)

    half = rk4_step(_rhs, phi, state, h / 2)
    two_half = rk4_step(_rhs, phi + h / 2, half, h / 2)

    error_vector = (two_half - full) / 15.0
    scale = atol + rtol * np.maximum(np.abs(state), np.abs(two_half))
    normalized = error_vector / scale
    error_norm = float(np.sqrt(np.mean(normalized**2)))

    return two_half, error_norm, 12
~~~

This function is only one trial attempt. It does not yet manage acceptance, rejection, event localization, or the next step size.

## 9.2 Error controller

Use a normalized RMS error:

\[
\mathrm{err}=
\sqrt{\frac{1}{2}\sum_{i=1}^{2}
\left(\frac{e_i}{a_i+r_i|y_i|}\right)^2}.
\]

Accept when err <= 1.

A simple step-size controller is

\[
h_{\mathrm{new}}=h\cdot
\operatorname{clip}\left(
s\,\mathrm{err}^{-1/5},f_{\min},f_{\max}
\right).
\]

The exponent 1/5 follows from the local error scaling of RK4. Use a safety factor s below one and clamp the growth/shrinkage factor. Handle zero error explicitly, for example by allowing maximum bounded growth.

The controller loop must:
- bound every trial by h_min and h_max;
- clip the final step to the remaining angular interval;
- count all 12 RHS evaluations per attempted step;
- count accepted and rejected attempts separately;
- never advance the physical state after a rejected attempt;
- pay computational cost for rejected attempts;
- have a maximum attempt count;
- stop cleanly on successful escape or numerical failure;
- localize escape and count that extra work.

Never allow a retry loop to run forever. If h reaches h_min and the error still fails, terminate with a documented failure reason.

**Milestone complete when:** tightening tolerances generally increases work and decreases measured error, all work is counted, and the solver handles escape and failure robustly.

---

# 10. Milestone 8 — formulate the reinforcement-learning task

Only begin this after fixed-step and adaptive baselines are tested.

## 10.1 Markov decision process

- **Observation:** current geodesic state and recent numerical-controller information.
- **Action:** proposed next integration step size.
- **Transition:** numerical solver advances or rejects the proposed step.
- **Reward:** trade-off between local estimated error and computational work.
- **Episode:** one scattering ray, from closest approach to escape or failure.

The agent controls the numerical method, not the physical photon.

## 10.2 Observation

A first observation design is

\[
o_t=
\left[
u_t,\,
v_t,\,
h_{t-1}/h_{\max},\,
\phi_t/\phi_{\max},\,
\log_{10}(\epsilon_{t-1}+\delta)
\right].
\]

The features mean:
- u: inverse radius;
- v: radial trend in inverse-radius coordinates;
- previous normalized step size;
- normalized angular progress;
- previous normalized local error in log scale.

Use a documented initial value when there is no previous error. Keep feature scaling consistent between training and evaluation.

Do not add many features immediately. Start with a small observation and add features only through controlled ablations.

## 10.3 Continuous action mapping

Let the policy output a in [-1, 1]. Map it logarithmically:

\[
q=\frac{a+1}{2},
\]

\[
h=10^{\log_{10}h_{\min}
+q(\log_{10}h_{\max}-\log_{10}h_{\min})}.
\]

This makes equal action increments correspond to multiplicative changes in step size. Clamp the action and enforce hard numerical bounds inside the environment.

## 10.4 Reward

A starting reward penalizes normalized local error and computational cost:

\[
r_t=-\lambda_e g(\mathrm{err}_t)
-\lambda_c\frac{N_{\mathrm{RHS},t}}{N_{\mathrm{scale}}}.
\]

Here g is a bounded or smoothly growing error penalty. The constants must be selected after inspecting error/work distributions, not guessed blindly.

Add a terminal success reward for valid escape and a penalty for numerical failure or attempt-budget exhaustion.

**Do not leak the reference solution into the online reward.** The environment must not query the DOP853 future trajectory at every step to tell the policy whether its action was good. Use the local step-doubling estimate for online decisions and reward. Use DOP853 after the episode for offline global-error and deflection evaluation.

## 10.5 Reward-hacking tests

Before training, prove the following:
- terminating immediately cannot earn a high reward;
- taking tiny steps forever incurs work and eventually hits a limit;
- a huge step cannot silently jump over escape and be rewarded;
- NaN or infinite states cannot earn success;
- rejected attempts incur computational cost;
- only valid escape earns the success bonus;
- the episode horizon is sufficient for valid rays.

The reward formula is part of the experiment. Save it with every run.

---

# 11. Milestone 9 — build and test a Gymnasium environment

Install:

~~~bash
python -m pip install gymnasium
~~~

Create **src/lab/rl_env.py**. The environment should implement the standard Gymnasium interface:
- **__init__** — action and observation spaces;
- **reset(seed=None, options=None)** — returns observation and info;
- **step(action)** — returns observation, reward, terminated, truncated, info.

The key distinction:
- **terminated** means the task reached a terminal state, such as valid escape or numerical failure;
- **truncated** means an external limit ended the episode, such as maximum attempts.

A safe environment architecture is:

~~~python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GeodesicStepControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        closest_approach_range=(3.0, 10.0),
        h_min=1e-4,
        h_max=0.05,
        max_angle=8.0,
        max_attempts=20000,
    ):
        super().__init__()

        low, high = closest_approach_range
        if not (np.isfinite(low) and np.isfinite(high) and 1.5 < low < high):
            raise ValueError("Invalid closest-approach range")
        if not (0 < h_min < h_max):
            raise ValueError("Require 0 < h_min < h_max")
        if not (np.isfinite(max_angle) and max_angle > 0):
            raise ValueError("max_angle must be finite and positive")

        self.closest_approach_range = (float(low), float(high))
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.max_angle = float(max_angle)
        self.max_attempts = int(max_attempts)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # These are initial engineering bounds, not universal physical limits.
        # Verify every reachable observation lies inside them.
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, 0.0, 0.0, -12.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.phi = 0.0
        self.previous_h = self.h_min
        self.previous_error = 1.0
        self.attempts = 0
        self.rhs_evaluations = 0
        self.closest_approach = None

    def _observation(self):
        u, v = self.state
        return np.asarray(
            [
                u,
                v,
                self.previous_h / self.h_max,
                self.phi / self.max_angle,
                np.log10(max(self.previous_error, 1e-12)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        low, high = self.closest_approach_range
        self.closest_approach = float(self.np_random.uniform(low, high))
        self.state = np.array([1.0 / self.closest_approach, 0.0])
        self.phi = 0.0
        self.previous_h = self.h_min
        self.previous_error = 1.0
        self.attempts = 0
        self.rhs_evaluations = 0

        obs = self._observation()
        if not self.observation_space.contains(obs):
            raise RuntimeError("Initial observation is outside observation_space")

        return obs, {"closest_approach": self.closest_approach}

    def step(self, action):
        raise NotImplementedError(
            "Implement this after connecting the tested adaptive step attempt."
        )
~~~

This is an interface skeleton, not a functioning environment. The **step** method must be completed before training. Its responsibilities are:
1. Validate and clamp the incoming action.
2. Map action to proposed h.
3. Run the step-doubling attempt.
4. Accept or reject according to the safety rule.
5. Update state only on acceptance.
6. Count work even on rejection.
7. Update the previous-error and previous-h observation values.
8. Detect and localize escape or failure.
9. Compute a finite reward.
10. Return the five Gymnasium values.

The info dictionary should contain closest approach, phi, proposed/used h, error estimate, accepted/rejected flag, RHS work for the step and episode, termination reason, and escape angle if available.

If rejected, the physical state does not advance, but computational work was still spent. The agent must pay for it.

## 11.1 Environment tests

Test:
- reset observation belongs to observation_space;
- same seed reproduces initial sampled parameters;
- action and observation shape/dtype;
- no NaN or infinite observations;
- rejected attempts do not advance the physical state;
- rejected attempts increase work;
- successful escape terminates;
- attempt limit truncates;
- invalid states never receive success reward.

Run Gymnasium's checker:

~~~python
from gymnasium.utils.env_checker import check_env
from lab.rl_env import GeodesicStepControlEnv

check_env(GeodesicStepControlEnv())
~~~

Do not proceed until the checker and tests pass.

---

# 12. Milestone 10 — establish non-learning policies

PPO must be compared with meaningful baselines on identical rays.

Implement these policies:
1. **Minimum step:** always choose h_min. Usually accurate but expensive.
2. **Maximum step:** always choose h_max. Cheap but can be inaccurate or fail.
3. **Geometric midpoint:** always choose sqrt(h_min * h_max).
4. **Tuned fixed step:** best constant step found on a validation set.
5. **Adaptive step-doubling:** conventional controller from Part III.

The tuned fixed-step baseline is essential. Otherwise PPO may beat a badly chosen fixed step while losing to a properly tuned one.

Run all policies on the same held-out closest-approach cases and use the same reference solver configuration.

---

# 13. Milestone 11 — train PPO

Install the remaining dependencies:

~~~bash
python -m pip install stable-baselines3 tensorboard
~~~

PPO is a practical first algorithm for bounded continuous actions and episodic environments. It does not guarantee that learning will discover a better controller.

Create **scripts/train_ppo.py** after the environment is complete:

~~~python
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from lab.rl_env import GeodesicStepControlEnv


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "ppo_logs"
MODEL_DIR = ROOT / "results" / "models"


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_env = GeodesicStepControlEnv()
    check_env(raw_env)
    env = Monitor(raw_env)

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

    # This is a smoke test, not the final research training budget.
    model.learn(total_timesteps=5000)
    model.save(MODEL_DIR / "ppo_geodesic_step_control")


if __name__ == "__main__":
    main()
~~~

Run:

~~~bash
python scripts/train_ppo.py
tensorboard --logdir results/ppo_logs
~~~

Inspect:
- episode reward and length;
- success and failure rate;
- RHS evaluations per episode;
- accepted and rejected steps;
- action distribution;
- whether the policy collapses to h_min or h_max.

A 5,000-step run only verifies that training can execute. Do not call it a scientific result. Increase training only after the environment and reward have been inspected.

For every run, save:
- Git commit;
- environment configuration;
- observation/action definitions;
- reward formula and weights;
- h bounds and error tolerances;
- PPO hyperparameters;
- random seed;
- training timestep count;
- model file and evaluation metrics.

---

# 14. Milestone 12 — scientific evaluation

## 14.1 Train, validation, and held-out test cases

Do not evaluate only on the same closest approaches used during training.

Separate:
- training rays used to update the policy;
- validation rays used to tune reward and PPO settings;
- held-out test rays used only after the method is fixed.

Use more than one random seed. Report variation across seeds instead of publishing only the best run.

## 14.2 Metrics

For each test ray, record:
- escape success;
- escape-angle error;
- deflection-angle error;
- maximum u error;
- maximum v error;
- total RHS evaluations;
- accepted and rejected attempts;
- runtime;
- step-size distribution.

Report median, mean, standard deviation, and worst-case error where useful. Save raw per-ray rows before calculating aggregate statistics.

## 14.3 Matched-accuracy comparison

The core question is:

> At a specified accuracy target, how much computational work does each controller require?

Plot:
- RHS evaluations versus deflection error;
- runtime versus deflection error;
- success rate versus work;
- error distributions over held-out rays.

Compare PPO against tuned fixed-step RK4, adaptive step-doubling RK4, and DOP853. Be explicit that DOP853 is the reference for error measurement in one part of the experiment and a benchmark method in another.

## 14.4 Ablations

Change one design choice at a time:
- remove previous error from the observation;
- remove previous h;
- switch logarithmic action mapping to linear;
- vary the work penalty;
- vary the error penalty;
- train on narrow versus broad closest-approach ranges;
- compare deterministic and stochastic policy evaluation.

Ablations help explain why the policy works. Without them, the conclusion is limited to one particular configuration.

## 14.5 What counts as success?

A credible positive result requires:
- reliable episode completion;
- generalization to unseen closest approaches;
- a defined accuracy target;
- lower work than a meaningful baseline at matched accuracy, or a clear measured trade-off;
- repeatability across seeds;
- an independent high-accuracy reference;
- no future-reference leakage into online reward;
- transparent reporting of failures.

PPO may not outperform adaptive step-doubling. That is a valid scientific result. A carefully measured negative result is more informative than a claim based on a weak baseline.

---

# 15. Engineering cleanup and reproducibility

Before the repository becomes larger:

1. Standardize imports. Use package imports such as **from lab.integrators import rk4_step**.
2. Avoid maintaining duplicate solver settings in **simulate_ray** and **simulate_ray_raw**. Share an internal implementation or clearly document the distinction.
3. Put tests in the configured **tests/** directory. The current pytest configuration points there.
4. Remove tracked Python bytecode and ignore **__pycache__/** and **.pyc** files.
5. Keep results under **results/**. Ignore transient logs and large checkpoints unless intentionally publishing selected artifacts.
6. Make scripts runnable from the repository root.
7. Record Git commit hashes with experiment outputs.
8. Keep all solver settings in one explicit configuration rather than scattering constants through scripts.

Suggested CSV columns:

~~~text
method,seed,closest_approach,rtol,atol,h_min,h_max,steps,accepted,rejected,rhs_evaluations,runtime_seconds,escaped,escape_angle,deflection_radian,escape_error,deflection_error,max_u_error,max_v_error
~~~

Reproducibility checklist:

- [ ] Dependencies and versions are recorded.
- [ ] Git commit is recorded.
- [ ] Every experiment has a saved configuration.
- [ ] Random seeds are saved.
- [ ] Training, validation, and test cases are separate.
- [ ] Reference tolerances are documented.
- [ ] Rejected attempts and event localization are included in work counts.
- [ ] Runtime uses repeated measurements and a stated statistic.
- [ ] Raw per-ray measurements are retained.
- [ ] Plots can be regenerated from CSV.

---

# 16. The exact order to work in

| Order | Milestone | Deliverable | Do not proceed until |
|---:|---|---|---|
| 1 | RK4 tests | tests/test_integrators.py | Exact ODE tests and order check pass |
| 2 | Physics tests | tests/test_geodesics.py | Physical behavior is protected |
| 3 | Dense reference | reference.py | Arbitrary-angle reference queries work |
| 4 | Fixed-step runner | fixed_step.py | Trajectory and work counters are correct |
| 5 | Event localization | Bisection helper | Escape angle converges with h |
| 6 | Convergence study | CSV and log-log plot | State and terminal errors are measured |
| 7 | Adaptive RK4 | adaptive.py | Accuracy/work baseline is trustworthy |
| 8 | Gymnasium environment | rl_env.py and tests | Checker passes and no reward exploit exists |
| 9 | Baseline policies | common evaluation | All baselines use identical cases |
| 10 | PPO smoke test | training script and logs | Training runs with finite rewards |
| 11 | Held-out evaluation | evaluation CSV/plots | Multi-seed matched-accuracy results exist |

## Immediate next session

Do only this:

1. Verify the project metadata filename and rename project.toml to pyproject.toml if needed.
2. Create tests/test_integrators.py with the tests in Milestone 1.
3. Run pytest and fix only issues those tests reveal.
4. Add the multi-step exponential convergence-order experiment.
5. Share the test output and current tree before implementing the next module.

Do not start PPO now. The agent's reward depends on numerical error estimates. Until those estimates are trustworthy, training would optimize a measurement system that has not been validated.

---

# References

- SciPy solve_ivp documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Gymnasium documentation: https://gymnasium.farama.org/
- Stable-Baselines3 documentation: https://stable-baselines3.readthedocs.io/
- Sutton and Barto, Reinforcement Learning: An Introduction: http://incompleteideas.net/book/the-book-2nd.html
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/

**Guiding principle:** every new concept must become a tested, measured part of the project. Learn what is needed for the next decision, implement it, inspect the result, and only then move forward.
