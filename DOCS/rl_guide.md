# Black Hole Lab — The Complete Reinforcement Learning Project Guide

> **How to use this book:** Read the chapter, understand the concept, attempt the implementation yourself, use the hint only if needed, then reveal the complete reference implementation and study it line by line. Run the tests before moving on. Do not treat the code as something to paste without understanding.

Repository: https://github.com/sahandkhodayi/Black-Hole

This is a living textbook for evolving the existing Schwarzschild ray-tracing simulator into a research project involving reinforcement learning. It starts from the code that already exists in this repository. It does not assume that future files or milestones have already been implemented.

---

# Part I — Project definition

## 1. What are we building?

The original project numerically simulates light bending around a non-rotating Schwarzschild black hole. The next stage is an RL-controlled numerical integrator.

**Research question:**

> Can a reinforcement-learning policy learn to choose numerical integration step sizes for Schwarzschild photon geodesics, producing a useful trade-off between trajectory accuracy and computational cost?

The agent will not steer a physical photon. A freely travelling photon follows a geodesic; it does not choose its direction or apply thrust. Instead, the agent controls the numerical solver's step size. This creates a legitimate sequential decision problem without changing the physical equation.

The project is successful if we perform a rigorous experiment. It is not necessary for RL to beat a conventional adaptive solver. A well-measured negative result is still a valid result.

## 2. Is changing from an MLP/PINN plan to RL too late?

No. The numerical physics work is still the foundation. The neural-network knowledge transfers too:

- A policy network and value network are neural networks.
- Optimization, tensors, activations, loss functions, and training loops remain relevant.
- Your previous supervised MLP weights are not automatically reusable as a PPO policy. The task and training objective are different.
- The old supervised MLP/PINN direction can remain a comparison or future extension. We do not need to delete it.

## 3. What the existing physics means

We use units where the Schwarzschild radius is (r_s=1), and define inverse radius (u=1/r). The orbit equation for the photon is

\[
\frac{d^2u}{d\phi^2}=\frac{3}{2}u^2-u.
\]

Define (v=du/d\phi). The equivalent first-order system is

\[
\frac{du}{d\phi}=v,\qquad
\frac{dv}{d\phi}=\frac{3}{2}u^2-u.
\]

At closest approach (r_0), the initial conditions are

\[
u(0)=\frac{1}{r_0},\qquad v(0)=0.
\]

The photon sphere is at (r=1.5r_s). For a scattering ray, escape corresponds to the outgoing crossing (u=0). The deflection angle in this symmetric setup is

\[
\alpha=2\phi_{\mathrm{escape}}-\pi.
\]

The existing DOP853 solver will be our high-accuracy numerical reference. It is not an exact analytical solution.

## 4. Why numerical integration is the RL control problem

A numerical integrator advances the independent variable (phi) by a step (h). Smaller steps generally cost more function evaluations but can improve accuracy. Larger steps are cheaper but can miss important changes or accumulate error.

| RL term | Project meaning |
|---|---|
| Environment | Our numerical integration process |
| Observation | Current (phi,u,v), previous step, and error information |
| Action | Choose a bounded multiplier for the next step size |
| Transition | Advance the ODE by the selected step |
| Reward | Balance trajectory accuracy and computational cost |
| Episode | Integrate one ray |
| Termination | Escape, invalid state, or a defined integration limit |

We will prevent the agent from seeing reference-only information that would leak the answer. The reference trajectory is for training diagnostics and evaluation design; whether it can be used during training reward calculation must be stated explicitly and tested for leakage.

---

# Part II — The learning and implementation workflow

Every chapter follows the same format:

1. **Concept:** what the idea means and why we need it.
2. **Connection to this project:** where it fits into Black Hole Lab.
3. **What to do:** a precise implementation task.
4. **Your attempt:** a small goal with constraints.
5. **Hint:** the next level of help, without giving everything away.
6. **Complete implementation:** a working reference answer if you are stuck.
7. **Line-by-line explanation:** what each important line does and why.
8. **Verification:** tests, expected behavior, and common mistakes.
9. **Completion check:** what you must understand before moving on.

Try the implementation before reading the complete implementation. If you do read it, close it and recreate the solution from understanding.

---

# MILESTONE 0 — Make the existing physics solver trustworthy

## Chapter 0.1 — Understand the output of solve_ivp

### Concept: numerical ODE solvers return more than a trajectory

Your current code calls SciPy's `solve_ivp`. It numerically solves an initial-value problem:

\[
y'(\phi)=f(\phi,y),\qquad y(0)=y_0.
\]

For our black hole, (y=[u,v]).

The returned object contains several important fields:

- `solution.t`: independent-variable samples at the solver's accepted points.
- `solution.y`: state values at those points. Row 0 is (u); row 1 is (v).
- `solution.t_events`: event times at which a configured event function crossed zero.
- `solution.y_events`: state values at those event times.
- `solution.nfev`: number of calls to the ODE right-hand-side function.
- `solution.success`: whether the solver completed successfully.
- `solution.message`: human-readable solver status.

**Important distinction:** `solution.t[-1]` is the last stored integration point. It might be the event time, but it might instead be the requested end of integration. The event time must be read from `solution.t_events`.

### What to do

Open `src/lab/geodesics.py` and trace the result from `solve_ivp` to the `RayTrajectory` return value.

Answer in your notes:

1. What is the shape of `solution.y` for this two-variable ODE?
2. Why is `solution.y[0]` the inverse radius?
3. What does an empty `solution.t_events[0]` mean?
4. Why does `nfev` matter when comparing integrators?

### Hint

The event list is a list for each event function. We only defined one event, so inspect the first list. It can contain zero or more event times.

### Complete implementation — only if you need it

Replace the current `simulate_ray` function with the version below. Keep your existing imports, dataclass, `_rhs`, and `_escape_event`.

\`\`\`python
def simulate_ray(
    closest_approach: float = 10.0,
    *,
    max_angle: float = 8.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> RayTrajectory:
    """Integrate a scattering ray from closest approach."""

    if not np.isfinite(closest_approach) or closest_approach <= 1.5:
        raise ValueError("closest_approach must be finite and > 1.5.")

    if not np.isfinite(max_angle) or max_angle <= 0:
        raise ValueError("max_angle must be finite and positive.")

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
        events=_escape_event,
        max_step=0.02,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    event_times = solution.t_events[0]
    escaped = len(event_times) > 0
    escape_angle = float(event_times[0]) if escaped else None

    phi_out = solution.t
    u_out = solution.y[0]

    # Reflect the outgoing half across the point of closest approach.
    phi = np.concatenate((-phi_out[:0:-1], phi_out))
    u = np.concatenate((u_out[:0:-1], u_out))

    return RayTrajectory(
        phi=phi,
        u=u,
        escaped=escaped,
        escape_angle=escape_angle,
        nfev=solution.nfev,
    )
\`\`\`

### Line-by-line explanation

- The function signature preserves the public parameters you already use.
- The first three checks reject invalid physical input and invalid solver tolerances before numerical work begins.
- `u0 = 1.0 / closest_approach` converts the closest-approach radius into inverse-radius coordinates.
- `solve_ivp` integrates the first-order system from angle zero to `max_angle`.
- `events=_escape_event` tells SciPy to monitor the escape condition while integrating.
- `solution.success` is a numerical-solver status, not an escape status. We raise if the integration itself failed.
- `event_times = solution.t_events[0]` obtains the roots detected for our one event function.
- `escaped` is true only if at least one event time exists.
- `escape_angle` is the first detected event time; if there was no event, it is `None`.
- `phi_out` and `u_out` are the computed half-trajectory from closest approach outward.
- The concatenations reflect that half-trajectory around (phi=0) to draw the incoming and outgoing sides.
- The returned dataclass now contains truthful event metadata and the solver's function-evaluation count.

### Important note about the symmetry

The reflected plot is a geometric construction using the symmetry of this scattering setup. It does not mean the solver separately integrated the incoming half.

## Chapter 0.2 — Stop code from running when imported

### Concept: import-time side effects

A Python module should normally define functions, classes, and constants when imported. It should not unexpectedly run a simulation, print output, or create plots.

Your current `geodesics.py` has a loop at module scope:

\`\`\`python
closest_approaches = [3.0, 5.0, 10.0]
for distance in closest_approaches:
    trajectory = simulate_ray(closest_approach=distance)
\`\`\`

That loop runs whenever another module imports `lab.geodesics`. Your `plot.py` imports `simulate_ray`, so importing the plotting module can trigger those simulations too.

### What to do

1. Remove the module-level loop from `geodesics.py`.
2. If you still want a quick demonstration, move it into a `main()` function.
3. Call `main()` only inside the `if __name__ == "__main__":` guard.

### Hint

Python sets `__name__` to the module's import name when imported, and to `"__main__"` when that file is run directly.

### Complete implementation

\`\`\`python
def main() -> None:
    for distance in (3.0, 5.0, 10.0):
        ray = simulate_ray(closest_approach=distance)
        print(
            f"r_min={distance:g} | escaped={ray.escaped} "
            f"| escape_angle={ray.escape_angle} | nfev={ray.nfev}"
        )


if __name__ == "__main__":
    main()
\`\`\`

### Line-by-line explanation

- `def main() -> None` groups the demonstration into an explicit entry point.
- The tuple supplies the three example closest-approach values.
- `simulate_ray` is called only when the demonstration runs.
- The formatted output reports metadata instead of calculating a deflection from an angle that may not be an escape event.
- The guard prevents the demonstration from running when another module imports this file.

## Chapter 0.3 — Verify the corrected solver

### What to do

Run one normal ray and one deliberately short integration. You can temporarily run these calls from a small scratch script or Python shell:

\`\`\`python
normal = simulate_ray(closest_approach=5.0)
short = simulate_ray(closest_approach=5.0, max_angle=0.01)

print(normal.escaped, normal.escape_angle, normal.nfev)
print(short.escaped, short.escape_angle, short.nfev)
print(normal.deflection_angle)
print(short.deflection_angle)
\`\`\`

### Expected logic

- The ordinary ray should detect the outgoing (u=0) event if the configured interval is long enough.
- The short run should complete without an escape event.
- The short run must have `escaped=False`, `escape_angle=None`, and `deflection_angle=None`.
- `nfev` should be a positive integer.

### Common mistakes

- Treating `solution.success` as equivalent to escape.
- Using the final sample as the escape angle without checking the event.
- Setting `escaped=True` by default in the return statement.
- Leaving the demonstration loop at module scope.
- Forgetting that `deflection_angle` is in degrees in your current dataclass property, despite the property name `deflection_radian` being in radians.

### Completion checklist

- [ ] Importing `lab.geodesics` does not run a simulation.
- [ ] Escape status is based on the event list.
- [ ] Escape angle comes from `t_events`.
- [ ] `nfev` is populated.
- [ ] A no-event run has no deflection angle.
- [ ] You can explain the difference between solver success and physical escape.

---

# MILESTONE 1 — Implement a numerical integrator we can control

## Chapter 1.1 — Why we cannot hand the whole trajectory to solve_ivp

### Concept

Your current DOP853 call integrates an entire interval. The solver internally chooses its own step sizes. That is excellent for obtaining a reference solution, but it does not give an RL policy direct control over each individual step.

We need a one-step method:

\[
y_{n+1}=\Psi(\phi_n,y_n,h_n),
\]

where the agent will eventually choose (h_n).

We will first implement classical fourth-order Runge–Kutta (RK4). RK4 is transparent and gives us direct control over (h). It is not expected to outperform DOP853.

### Concept: the four RK4 stages

For (y'=f(\phi,y)):

\[
\begin{aligned}
k_1 &= f(\phi,y),\\
k_2 &= f(\phi+h/2,y+hk_1/2),\\
k_3 &= f(\phi+h/2,y+hk_2/2),\\
k_4 &= f(\phi+h,y+hk_3),\\
y_{n+1} &= y+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

Each (k) is a slope estimate. RK4 samples the slope at the beginning, two midpoint estimates, and the end. The weighted average produces fourth-order global accuracy for sufficiently smooth problems and sufficiently small steps.

### What to do

Create `src/lab/integrators.py`.

Implement this function:

\`\`\`python
def rk4_step(rhs, phi, state, step_size):
    ...
\`\`\`

Constraints:

- `rhs(phi, state)` returns the derivative vector.
- `state` is a NumPy array, here containing `[u, v]`.
- Return a new array. Do not mutate the input state.
- Do not hardcode the Schwarzschild equation in the integrator.
- Reject invalid or non-positive step sizes.

### Your attempt

Write the four stages from the equations. Use a separate variable for each stage. Then calculate the weighted sum.

### Hint

The second stage evaluates the right-hand side at `phi + h/2` and a state shifted by `h*k1/2`. The third stage uses `k2`, not `k1`. The fourth uses a full-step state.

### Complete implementation — reveal only if needed

\`\`\`python
import numpy as np


def rk4_step(rhs, phi, state, step_size):
    """Advance y' = rhs(phi, y) by one classical RK4 step."""

    state = np.asarray(state, dtype=float)

    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive.")

    k1 = np.asarray(rhs(phi, state), dtype=float)
    k2 = np.asarray(
        rhs(phi + step_size / 2, state + step_size * k1 / 2),
        dtype=float,
    )
    k3 = np.asarray(
        rhs(phi + step_size / 2, state + step_size * k2 / 2),
        dtype=float,
    )
    k4 = np.asarray(
        rhs(phi + step_size, state + step_size * k3),
        dtype=float,
    )

    return state + (step_size / 6) * (k1 + 2*k2 + 2*k3 + k4)
\`\`\`

### Line-by-line explanation

- `import numpy as np` gives us vector arithmetic for the two state variables.
- `np.asarray(state, dtype=float)` ensures the state supports floating-point vector operations. It does not require modifying the caller's original array.
- The validation rejects zero, negative, infinite, and NaN step sizes.
- `k1` samples the derivative at the beginning of the interval.
- `k2` samples a predicted midpoint using the first slope.
- `k3` improves the midpoint estimate using the second slope.
- `k4` samples the derivative at the predicted end of the interval.
- The final weighted sum combines the four slopes. The midpoint estimates receive twice the weight of the endpoint estimates.

## Chapter 1.2 — Test a numerical method on a problem with a known answer

### Concept

Before testing RK4 on black-hole physics, test it on an equation whose exact solution is known. This separates bugs in our numerical method from bugs in the physics model.

Use

\[
y'=y,\qquad y(0)=1.
\]

The exact solution is (y(\phi)=e^\phi). After one step of size (h), compare the RK4 result with (e^h).

### What to do

Write a test using (h=0.1). Then repeat with (h=0.05) and (h=0.025). Record absolute error:

\[
E=|y_{\mathrm{RK4}}-e^h|.
\]

### Hint

For this test, the right-hand side returns the state itself. The initial state can be a one-element NumPy array.

### Complete implementation

\`\`\`python
import numpy as np

from lab.integrators import rk4_step


def exponential_rhs(phi, state):
    return state


initial_state = np.array([1.0])

for step_size in (0.1, 0.05, 0.025):
    result = rk4_step(
        exponential_rhs,
        phi=0.0,
        state=initial_state,
        step_size=step_size,
    )
    exact = np.exp(step_size)
    error = abs(result[0] - exact)
    print(step_size, result[0], exact, error)

assert initial_state[0] == 1.0
\`\`\`

### Line-by-line explanation

- The test imports the integrator independently of the black-hole model.
- `exponential_rhs` represents (y'=y).
- `initial_state` is one-dimensional, showing that the integrator is generic rather than tied to two variables.
- The loop repeats the same mathematical experiment at smaller step sizes.
- `exact` computes the known analytical answer.
- `error` measures the discrepancy.
- The assertion checks that the original state was not modified.

### Verification

For a single RK4 step on this smooth equation, reducing (h) should reduce the error. Do not expect a perfect exact factor from a tiny number of measurements; floating-point rounding and the tested range matter.

## Chapter 1.3 — Apply RK4 to the Schwarzschild state

### What to do

Import or define the same `_rhs` function used by `geodesics.py`. Start from:

\[
y_0=[1/r_0,0].
\]

For (r_0=5), take one small RK4 step and inspect the new (u) and (v).

### Hint

The state order must match the physics solver: index 0 is (u), index 1 is (v). Do not swap them.

### Complete implementation

\`\`\`python
import numpy as np

from lab.geodesics import _rhs
from lab.integrators import rk4_step

closest_approach = 5.0
state = np.array([1.0 / closest_approach, 0.0])

next_state = rk4_step(
    _rhs,
    phi=0.0,
    state=state,
    step_size=0.01,
)

print("before:", state)
print("after: ", next_state)
\`\`\`

### Explanation

- The test reuses the exact same physical right-hand side as DOP853.
- The initial inverse radius is (1/5=0.2).
- The initial radial derivative with respect to angle is zero at closest approach.
- The small step advances the solution slightly away from the turning point.
- Reusing the same right-hand side ensures the two numerical methods solve the same equation.

### Completion checklist

- [ ] Generic RK4 function exists in `integrators.py`.
- [ ] Invalid step sizes are rejected.
- [ ] The input state is not mutated.
- [ ] The exponential test behaves as expected.
- [ ] A Schwarzschild state advances without shape or type errors.
- [ ] You can explain all four RK stages.

---

# MILESTONE 2 — Estimate numerical error and control step size

## Chapter 2.1 — Local error versus global error

### Concept

A numerical step introduces truncation error. Local error concerns the error introduced over one step, while global error is the accumulated difference between the computed trajectory and the true solution over many steps.

We do not know the exact solution for every black-hole trajectory. We can estimate local error by step doubling:

1. Take one RK4 step of size (h).
2. Take two RK4 steps of size (h/2).
3. Compare the results at the same final angle.

RK4 has a leading local truncation error proportional to (h^5). The difference between one full step and two half steps therefore contains information about the local error. The two-half-step result is generally more accurate.

A calibrated estimate uses the method's order. We will derive and test the scaling rather than treating the raw difference as exact error.

### What to do

Implement `rk4_step_doubling(rhs, phi, state, step_size)`. Return:

- The accepted state (two half steps).
- An error estimate.
- Optionally, the number of RHS evaluations used.

### Hint

Compute `full = rk4_step(..., h)`; compute `half = rk4_step(..., h/2)`; then take another half step starting at `phi+h/2` from the first half-step result. The two results end at the same angle.

### Complete implementation

\`\`\`python
def rk4_step_doubling(rhs, phi, state, step_size):
    full = rk4_step(rhs, phi, state, step_size)

    half = rk4_step(rhs, phi, state, step_size / 2)
    two_half = rk4_step(
        rhs,
        phi + step_size / 2,
        half,
        step_size / 2,
    )

    # For a fourth-order method, Richardson scaling uses 2**4 - 1.
    error_estimate = (two_half - full) / 15.0

    # The two-half-step result is the more accurate approximation.
    accepted_state = two_half + error_estimate

    return accepted_state, np.abs(error_estimate)
\`\`\`

### Line-by-line explanation

- `full` advances once by (h).
- `half` advances from the initial state by (h/2).
- `two_half` advances from the midpoint to the same final angle.
- `two_half - full` measures disagreement between the two approximations.
- `15` is (2^4-1), arising from fourth-order Richardson extrapolation under the asymptotic error model.
- `accepted_state` applies the Richardson correction to the two-half-step estimate.
- The absolute error estimate is returned component-wise.

**Important:** This is an estimator based on the asymptotic behavior of RK4, not a proof of the true error. We must validate it against DOP853 over the regimes we intend to study. Also count RHS calls consistently: this implementation performs 12 evaluations if each RK4 step uses four calls.

## Chapter 2.2 — Scale the error

### Concept

The state components (u) and (v) can have different magnitudes. A raw absolute error may overemphasize one component. Numerical solvers commonly scale errors using absolute and relative tolerances.

For each component, define a scale such as

\[
s_i=\mathrm{atol}+\mathrm{rtol}\max(|y_i|,|y_{i,\mathrm{new}}|).
\]

Then calculate a normalized error measure from (|e_i|/s_i), for example the maximum component ratio or a weighted RMS norm.

### What to do

Implement and test a function that receives the estimated error, old state, new state, `atol`, and `rtol`, and returns one scalar normalized error.

### Hint

Use NumPy arrays and elementwise division. Reject non-positive tolerances.

### Complete implementation

\`\`\`python
def scaled_error_norm(error, old_state, new_state, atol, rtol):
    if atol <= 0 or rtol <= 0:
        raise ValueError("atol and rtol must be positive.")

    scale = atol + rtol * np.maximum(
        np.abs(old_state),
        np.abs(new_state),
    )
    ratios = np.asarray(error) / scale
    return float(np.sqrt(np.mean(ratios**2)))
\`\`\`

### Explanation

- The tolerances define the error scale expected for each component.
- `np.maximum` chooses the larger magnitude between old and new values.
- Dividing by the scale makes the error dimensionless.
- The RMS combines component errors into one scalar.
- A normalized error around 1 means the estimate is around the requested tolerance scale. It is not a guarantee of global trajectory accuracy.

## Chapter 2.3 — Build a conventional controller before RL

### Concept

Before asking an RL agent to choose step sizes, we need a non-learning baseline. A conventional controller can increase the step when error is comfortably small and decrease it when error is too large.

This baseline matters scientifically: if a simple deterministic controller already performs well, PPO has to demonstrate a measurable advantage.

### What to do

Implement a bounded step-size update:

\[
h_{\mathrm{new}}=h\cdot
\operatorname{clip}(s\,e^{-1/5}, f_{\min},f_{\max}),
\]

where (e) is the normalized error and (s) is a safety factor. This exponent is associated with a fifth-order local error scaling for RK4 step error; the controller must still be validated empirically.

### Hint

Handle (e=0) separately to avoid division or numerical issues. Clamp the resulting step to `min_step` and `max_step`.

### Complete implementation

\`\`\`python
def update_step_size(
    step_size,
    error_norm,
    *,
    min_step=1e-5,
    max_step=0.05,
    safety=0.9,
    min_factor=0.2,
    max_factor=5.0,
):
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive.")

    if not np.isfinite(error_norm) or error_norm < 0:
        raise ValueError("error_norm must be finite and non-negative.")

    if error_norm == 0:
        factor = max_factor
    else:
        factor = safety * error_norm ** (-1.0 / 5.0)
        factor = np.clip(factor, min_factor, max_factor)

    return float(np.clip(step_size * factor, min_step, max_step))
\`\`\`

### Explanation

- Input validation prevents invalid controller state.
- Zero estimated error permits the maximum growth factor.
- Otherwise, the error-based formula proposes a growth or shrink factor.
- `np.clip` limits abrupt changes.
- The final clamp enforces global minimum and maximum step sizes.

---

# MILESTONE 3 — Design the reinforcement-learning problem

## Chapter 3.1 — What is an MDP?

### Concept

A Markov Decision Process (MDP) describes sequential decisions using:

- State (s_t): information describing the current situation.
- Action (a_t): the decision made by the agent.
- Transition (P(s_{t+1}|s_t,a_t)): how the environment changes.
- Reward (r_t): immediate feedback.
- Discount (gamma): how future rewards are weighted.

The policy (pi(a|s)) describes how the agent chooses actions from states.

### Apply it to Black Hole Lab

The agent chooses a step-size multiplier. It does not choose the black-hole physics or the initial ray's physical laws.

Initial proposed design:

- Observation: normalized (phi,u,v), previous step size, previous normalized error, and remaining angular interval.
- Action: one continuous scalar mapped to a bounded step multiplier.
- Transition: run the RK4 step-doubling integrator for the selected step.
- Reward: encourage progress while penalizing computational cost and inaccurate steps.
- Episode: one ray from closest approach until escape or a defined limit.

### What to do

Write a short design document before coding. Specify exact observation bounds, normalization, action mapping, reward terms, and termination rules.

### Hint

Keep two categories separate:
1. Information the agent can observe when choosing an action.
2. Information the evaluator can use to score the final trajectory.

Do not expose future reference trajectory values to the agent.

## Chapter 3.2 — Reward design and reward hacking

### Concept

A reward is an optimization target, not a natural-language intention. If we reward only escape, the agent may take enormous steps, cross the event, and produce an inaccurate trajectory.

A candidate reward can combine:

\[
r_t = w_p\,\Delta\phi
      - w_c\,C_t
      - w_e\,E_t
      - w_f\,F_t,
\]

where progress, computation, normalized error, and failure penalties are scaled terms. The weights are experimental parameters, not universal constants.

We must decide whether the reward uses a local error estimate, a reference-derived global error, or both. If reference-derived information is used during training, state that explicitly: it changes what the agent is learning and may create information leakage if the same reference is also used as an evaluation target.

### What to do

Propose a reward equation and justify every term. Test it on hand-designed behaviors before training PPO.

### Hint

Ask what happens if the agent always chooses the minimum step, always chooses the maximum step, or deliberately ends every episode early. If any of those strategies earns a high reward, the reward design is flawed.

---

# MILESTONE 4 — Implement the Gymnasium environment

## Chapter 4.1 — What Gymnasium provides

### Concept

Gymnasium standardizes the interface between an RL algorithm and an environment.

The essential methods are:

- `__init__`: configure spaces and parameters.
- `reset(seed=None, options=None)`: initialize an episode and return `(observation, info)`.
- `step(action)`: apply an action and return `(observation, reward, terminated, truncated, info)`.

The distinction between `terminated` and `truncated` matters:
- `terminated=True`: a terminal state defined by the task occurred, such as escape or invalid physics state.
- `truncated=True`: an external limit ended the episode, such as a maximum number of steps or integration angle.

### What to do

Create `src/lab/environment.py`. Implement the environment only after the MDP design is written down.

Start with a narrow range of closest-approach radii safely above the photon sphere. Widen the range after tests pass.

### Hint

Use `gymnasium.Env`, `spaces.Box`, and a seeded NumPy random generator. Keep the physical state and the observation vector as separate variables.

### Implementation blueprint

The exact implementation depends on the observation and reward choices you document in Chapter 3. Do not blindly copy an environment with arbitrary reward weights. Your first working version must include:

1. A bounded continuous action space.
2. A finite observation space with documented scaling.
3. Deterministic reset when the same seed and options are supplied.
4. Safe clipping of action-derived step sizes.
5. Correct escape-event detection.
6. Distinct escape, numerical-failure, and time-limit statuses.
7. Finite rewards and observations for every valid transition.

### Line-by-line study checklist

When you implement the class, be able to explain:
- Why the class inherits from `gym.Env`.
- What `observation_space` and `action_space` guarantee.
- Why `reset` calls the base reset with a seed.
- Why observations must match the declared space.
- Why `step` returns five values.
- Why the environment must not silently turn a numerical failure into an escape.

## Chapter 4.2 — Test the environment before training

### What to do

Test:
- Observation shape and finite values.
- Action clipping.
- Minimum and maximum step size.
- Deterministic reset.
- Escape detection.
- Maximum-angle truncation.
- Invalid numerical state handling.
- Finite reward.
- Repeated reset after episode termination.

Use Gymnasium's environment checker when the environment is ready.

### Completion gate

Do not start PPO until the environment passes its tests. A learning algorithm cannot compensate for a broken transition or reward definition.

---

# MILESTONE 5 — Understand and train PPO

## Chapter 5.1 — From supervised learning to reinforcement learning

### Concept

In supervised learning, a model learns from labelled examples such as ((x,y)). The training process compares predictions with known targets.

In policy-based RL, the agent interacts with an environment. It collects observations, actions, rewards, and next observations. It updates its policy to increase expected return.

The network is still a neural network, but the training signal is different.

## Chapter 5.2 — Actor and critic

PPO commonly uses two learned functions:

- **Actor:** policy that produces an action distribution given an observation.
- **Critic:** estimates the expected future return from an observation.

For a continuous action, the policy often models a distribution such as a Gaussian. The agent samples an action from it during training.

## Chapter 5.3 — PPO's clipped objective

PPO compares the new policy with the old policy using a probability ratio:

\[
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
     {\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}.
\]

A common clipped objective is

\[
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

Here (A_t) is an advantage estimate: how much better or worse an action performed than the baseline expectation. Clipping discourages excessively large policy updates.

You do not need to implement PPO from scratch in the first version. You do need to understand what the library is optimizing.

## Chapter 5.4 — Train the first policy

### What to do

Create `scripts/train_ppo.py`. Use a maintained PPO implementation compatible with Gymnasium, such as Stable-Baselines3 after checking the installed version and its API.

The script should:
1. Construct the environment.
2. Run environment checks.
3. Create PPO with a recorded random seed.
4. Train for a small test budget.
5. Save the model and configuration.
6. Record episode rewards, lengths, and termination reasons.

### Hint

Keep training configuration explicit: seed, total timesteps, learning rate, rollout length, batch size, discount, and network architecture. Do not hide important experimental settings in library defaults.

### Implementation policy

The full training script will be added after the environment API and reward design are fixed. This prevents giving you code that assumes a different observation shape or reward definition than the one you actually implemented.

### Line-by-line study checklist

For every line in the training script, be able to explain:
- What environment is being constructed.
- What the seed controls.
- What PPO's constructor parameters mean.
- What one timestep means in this environment.
- What is saved and how to reload it.
- How training metrics are recorded.

---

# MILESTONE 6 — Evaluate the policy scientifically

## Chapter 6.1 — A trained model is not proof

A rising reward curve only shows that the agent is optimizing the reward we designed. It does not prove that it is solving the numerical problem accurately.

We need baselines:

1. Fixed-step RK4.
2. Conventional RK4 step-doubling controller.
3. PPO-controlled RK4.
4. SciPy DOP853 reference.

Use the same initial conditions and evaluation cases.

## Chapter 6.2 — Metrics

Report:

- Global trajectory error at common angular samples.
- Escape-angle error.
- Deflection-angle error.
- RHS evaluation count.
- Wall-clock runtime.
- Numerical failure rate.
- Unresolved episode rate.
- Accuracy versus computational cost.

Do not report only reward. Do not compare methods using different accuracy requirements without explaining the difference.

## Chapter 6.3 — Build the evaluation script

Create `src/lab/evaluation.py` and `scripts/evaluate_policy.py`.

The evaluator should:
1. Generate a fixed test set of closest-approach radii.
2. Compute reference trajectories with strict DOP853 tolerances.
3. Run every baseline and the trained policy on the same cases.
4. Interpolate trajectories onto common angular sample points where appropriate.
5. Calculate error and cost metrics.
6. Save machine-readable results and plots.

### Hint

Separate evaluation from training. The evaluation script should load a saved policy and should not update its weights.

### Completion gate

A result is only meaningful if another person can reproduce it from the saved configuration and evaluation cases.

---

# MILESTONE 7 — Generalization and research experiments

## Chapter 7.1 — Test conditions the agent did not train on

Test on:
- Closest-approach radii outside the training sample.
- Weak-field rays.
- More challenging rays closer to the photon sphere.
- Different maximum integration angles.
- Different tolerance targets.

Keep a held-out test set. Do not repeatedly tune against it and then call it unseen test data.

## Chapter 7.2 — Multiple random seeds

Neural-network training is stochastic. Run multiple seeds and report variation, not only the best run.

## Chapter 7.3 — Choose one extension

Only after the baseline study works, choose one:
- Compare PPO and SAC.
- Train over a wider range of physical parameters.
- Study transfer to a new parameter range.
- Explore a more demanding reward/error budget.
- Later, investigate rotating Kerr black-hole geometry if the physics and numerical method are extended carefully.

Do not add all extensions at once.

---

# MILESTONE 8 — Reproducibility and final research report

The final report must contain:

1. Research question and motivation.
2. Schwarzschild assumptions, units, equations, and event definitions.
3. Numerical methods and error-estimation derivation.
4. MDP definition: observation, action, transition, reward, and termination.
5. PPO architecture and hyperparameters.
6. Baselines and reference solver settings.
7. Accuracy-versus-computation results.
8. Multiple seeds and generalization results.
9. Failure cases and limitations.
10. Reproduction instructions and software versions.

The project should be understandable to another student who has never seen the repository.

---

# Final project checklist

- [ ] M0: reference solver has correct event metadata and no import-time simulation.
- [ ] M1: RK4 implemented and tested independently.
- [ ] M2: step-doubling estimate and conventional controller validated.
- [ ] M3: MDP and reward design written down and reviewed.
- [ ] M4: Gymnasium environment passes tests.
- [ ] M5: PPO training is reproducible and saved.
- [ ] M6: fair comparison against all numerical baselines.
- [ ] M7: held-out generalization and multiple seeds.
- [ ] M8: reproducible report and documentation.

## How we will work together

For each chapter, you should first attempt the task. Send me your code, test output, or the exact point where you are stuck. I will review it, explain errors, and ask you to reason about the behavior. If you cannot solve it from the hint, use the reference implementation and study the line-by-line explanation.

**Start at Chapter 0.1. Do not jump to PPO yet.**
