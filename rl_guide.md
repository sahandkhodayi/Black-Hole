# Black Hole Lab — Reinforcement Learning Continuation

This guide continues the existing Black-Hole project. It is deliberately organized as an implementation course: understand the goal, implement one small part yourself, run it, inspect the result, and only then move on. Code snippets will be introduced milestone by milestone rather than asking you to paste a complete system.

Repository: https://github.com/sahandkhodayi/Black-Hole

## The direction change

The project began as a numerical Schwarzschild light-bending simulation, with a planned supervised MLP and PINN extension. We are now adding a reinforcement-learning research track.

**New research question:**

> Can a reinforcement-learning policy learn to adapt numerical integration step sizes for Schwarzschild photon geodesics, achieving a useful accuracy-versus-computation trade-off compared with fixed-step integration and a conventional adaptive ODE solver?

This is a more defensible RL problem than asking an agent to steer a freely travelling photon. A photon following a geodesic does not choose its direction or apply thrust. Giving it arbitrary mid-flight steering would change the physical problem.

In this project, the agent does not change the physics. It controls **how the numerical solver advances through the same physical differential equation**.

The result is an RL-guided numerical integrator for black-hole ray tracing.

## Is it too late to switch?

No. The existing numerical solver is the foundation of the new environment. Your previous neural-network work is also useful:

- You already understand tensors, neural-network layers, optimization, and training loops.
- PPO policies and value functions commonly use neural networks.
- The old scalar supervised MLP is not an RL agent and its trained weights are not directly reusable, but the knowledge used to build it transfers.
- The old MLP/PINN milestones become optional comparison experiments rather than the main project direction.

Do not delete the old plan or code. Keep it as a historical branch of the project and compare methods where that comparison is meaningful.

## Scientific setup

We keep Schwarzschild-radius units, (r_s=1), and inverse radius (u=1/r). The photon equation is

\[
\frac{d^2u}{d\phi^2}=1.5u^2-u.
\]

Write it as a first-order system with (v=du/d\phi):

\[
\frac{du}{d\phi}=v,\qquad
\frac{dv}{d\phi}=1.5u^2-u.
\]

The physical initial conditions at closest approach (r_0) are

\[
u(0)=1/r_0,\qquad v(0)=0.
\]

The photon sphere is at (r=1.5r_s). The outgoing escape event is (u=0). The high-accuracy SciPy DOP853 solution will initially act as a reference trajectory; it is a numerical reference, not an exact analytical solution.

## Why this is an RL problem

An adaptive integrator repeatedly decides how far to advance its independent variable. A small step usually costs more computation but can reduce numerical error. A large step is cheaper but may miss important trajectory behavior or accumulate error.

We can formulate this as a sequential decision problem:

| RL concept | In this project |
|---|---|
| Environment | A numerical integrator evolving the Schwarzschild ODE |
| Observation/state | Current (\phi,u,v), previous step size, and numerical-error information |
| Action | Increase, decrease, or maintain the next angular step size |
| Transition | Advance the ODE using the chosen step |
| Reward | Favor accurate progress while penalizing excessive computation |
| Episode | Integrate one ray from closest approach to escape or a defined limit |
| Terminal condition | Escape event, invalid/horizon-side trajectory, or maximum integration angle |

The agent is not rewarded merely for reaching the escape event. It must reach it with a trajectory sufficiently close to the reference solution while using fewer function evaluations or less computation.

### Important scientific limitation

A conventional adaptive solver such as DOP853 already adjusts its step size using error estimates. RL is not automatically better. The project is successful even if PPO loses to DOP853: a fair negative result is useful. We must compare accuracy, function evaluations, runtime, stability, and failure rate.

## Final project architecture

\`\`\`text
src/
  lab/
    geodesics.py          # existing physical ODE and reference solver
    integrators.py        # our fixed/adaptive one-step numerical methods
    environment.py        # custom Gymnasium environment
    policies.py           # policy configuration and wrappers
    evaluation.py         # reference comparison and metrics
  simulate.py

scripts/
  inspect_reference.py
  benchmark_integrators.py
  train_ppo.py
  evaluate_policy.py
  plot_policy_behavior.py

tests/
  test_geodesics.py
  test_integrators.py
  test_environment.py

outputs/
  reference/
  benchmarks/
  rl/
\`\`\`

This is the target layout, not a claim that these files already exist. We will add them as each milestone is implemented.

---

# MILESTONE 0 — Make the existing simulator trustworthy

**Do this before installing Gymnasium or training an agent.** RL will amplify bugs in the environment. If the reference solver gives incorrect labels or status, the agent will optimize the bug.

## What I found in the current repository

Your current `src/lab/geodesics.py` has several issues that should be corrected before building on it:

1. `RayTrajectory` has `escape_angle` and `nfev` fields, but the return statement does not populate them.
2. The return statement sets `escaped=True` unconditionally, even if the escape event was never detected.
3. The printed deflection uses `solution.t[-1]` as if it were always an escape angle. Reaching `max_angle` is not the same as detecting escape.
4. A loop at module scope runs simulations whenever the module is imported. Imports should define functions and classes, not unexpectedly launch experiments.
5. The event's actual root is available in `solution.t_events`; use that event data to classify escape.

These are correctness and software-design issues, not cosmetic style issues.

## Your task

Open `src/lab/geodesics.py`. Do not replace the entire file blindly.

### Task 0.1 — Understand the solver result

Inspect the result returned by `solve_ivp`. Identify what each of these contains:

- `solution.t`
- `solution.y`
- `solution.t_events`
- `solution.nfev`
- `solution.success`

Write a short comment or note in your own words explaining why `solution.t[-1]` and the event time are not interchangeable.

### Task 0.2 — Fix the trajectory metadata

After integration, derive these values from the solver result:

- Whether an escape event actually occurred.
- The escape angle, or `None` if no event occurred.
- The number of right-hand-side evaluations.
- The solver success status.

Then pass those values into `RayTrajectory`. Do not hardcode a successful escape.

### Task 0.3 — Remove import-time execution

Move the example loop into a function such as `main()`, and protect its invocation with:

\`\`\`python
if __name__ == "__main__":
    main()
\`\`\`

The exact function structure is yours to implement.

### Task 0.4 — Test two different outcomes

Run one ordinary ray, for example (r_0=5), and one deliberately short integration, for example `max_angle=0.01`.

Expected logic:

- The ordinary case should detect escape if the integration reaches the (u=0) event.
- The short case should report that no escape was detected.
- A successful ODE integration that reaches its end without an event must not be labelled as escaped.
- Deflection must be `None` when no escape angle exists.

### Completion criteria

- [ ] Importing `lab.geodesics` does not run simulations or print output.
- [ ] Escape status is based on event detection.
- [ ] Escape angle comes from event data.
- [ ] `nfev` is stored in the result.
- [ ] A no-event integration is represented honestly.
- [ ] You can explain each of these decisions.

**Do not start the RL environment until this milestone is correct.**

---

# MILESTONE 1 — Understand and implement a one-step integrator

The RL agent needs control over individual numerical steps. A call to `solve_ivp` integrates an entire interval and makes its internal step-size decisions itself. We therefore need a small one-step integrator that we can call repeatedly.

We will first implement classical fourth-order Runge–Kutta (RK4). This is not intended to beat DOP853. It gives us a transparent numerical method whose step size we can control.

For a system

\[
\frac{dy}{d\phi}=f(\phi,y),
\]

one RK4 step of size (h) is

\[
\begin{aligned}
k_1 &= f(\phi,y),\\
k_2 &= f(\phi+h/2,y+hk_1/2),\\
k_3 &= f(\phi+h/2,y+hk_2/2),\\
k_4 &= f(\phi+h,y+hk_3),\\
y_{n+1} &= y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

Here (y=(u,v)). The same physical right-hand side used by the reference solver must be used by RK4. Otherwise we would be comparing different equations.

## Your implementation task

Create `src/lab/integrators.py`.

Start with the function signature:

\`\`\`python
def rk4_step(rhs, phi, state, step_size):
    ...
\`\`\`

Implement it yourself using the four RK stages above.

### Constraints

- `state` is a NumPy array containing `[u, v]`.
- The function returns the new state; it should not mutate the caller's input array.
- Keep the function generic: it should accept the right-hand-side function instead of hardcoding the black-hole equation.
- Do not add an RL dependency yet.

### Questions you must be able to answer

1. Why are there four calls to `rhs`?
2. Why are the second and third stages evaluated at the half-step?
3. Why does the fourth stage use the full step?
4. What happens to accuracy and computational cost when (h) is reduced?
5. Why should the one-step method be tested against DOP853?

### Tests to write

- A single RK4 step for (y'=y), where the exact solution is known.
- A single step for the two-dimensional Schwarzschild state.
- Confirm that the input state is unchanged.
- Compare errors at (h), (h/2), and (h/4). Do not assume the measured convergence is correct until you inspect it.

### Completion criteria

- [ ] RK4 step implemented independently of the black-hole equation.
- [ ] Tests pass.
- [ ] Error decreases as the step is refined in the expected regime.
- [ ] You can explain the four stages without memorizing the code.

---

# MILESTONE 2 — Build an error estimate and a controlled integrator

A policy needs feedback. If the agent only sees position and velocity, it may not know whether its last step was numerically safe.

We will estimate local error by step doubling:

1. Compute one RK4 step of size (h).
2. Compute two RK4 steps of size (h/2).
3. Compare the two results.
4. Use the two-half-step result as the accepted approximation.
5. Use their difference as an error estimate.

The difference is an estimate, not the exact global error. We will calibrate it against DOP853.

For a first implementation, expose a function conceptually like:

\`\`\`python
def rk4_step_doubling(rhs, phi, state, step_size):
    # Return accepted_state and error_estimate.
    ...
\`\`\`

Do not invent an error formula from memory: derive what the difference between one full RK4 step and two half steps tells you about the leading truncation error. Record whether you use a raw norm, a scaled norm, or a component-wise tolerance.

The controller must also enforce a minimum and maximum step size. An agent must never be allowed to request a zero, negative, NaN, or unbounded step.

---

# MILESTONE 3 — Define the RL environment before writing the agent

We will use Gymnasium's environment interface. The environment is the most important RL component: it defines what the agent observes, what it can do, how the world changes, and what reward means.

## Initial environment design

**Observation:** a normalized vector containing:

- Current angle (\phi).
- Current inverse radius (u).
- Current derivative (v=du/d\phi).
- Previous step size.
- Previous scaled error estimate.
- Remaining angular interval or normalized progress.

**Action:** a continuous scalar controlling the next step-size multiplier. For example, the policy may request a multiplier in a bounded interval such as ([0.5,2.0]). The environment clips or safely maps the action to valid step sizes.

**Transition:** perform the numerical step using the selected step size, update the state, calculate error information, and check terminal events.

**Reward:** balance numerical accuracy and computation. A first reward design should penalize excessive scaled local error and function evaluations, with a severe penalty if the trajectory exceeds a declared accuracy tolerance. Reward terms must be dimensionless or explicitly normalized.

**Episode termination:**

- Escape event detected.
- A predeclared integration limit is reached.
- Numerical state becomes invalid.
- A safety limit is exceeded.

A maximum-angle termination is not an escape event. Keep those statuses distinct.

## Why not reward only escape?

A policy could take huge steps and still eventually cross (u=0), while producing a useless trajectory. The reward must measure trajectory quality against the reference solution, not just the final event label.

## Your design exercise

Before implementing the environment, write down:

1. The exact observation vector and its units/scaling.
2. The exact action range and how it maps to step size.
3. The error metric and tolerance.
4. The reward equation.
5. Every terminal condition and its status.
6. Which quantities are available to the agent and which are only used by the evaluator.

This is a required design decision, not paperwork. It prevents accidental leakage of the reference answer into the policy observation.

---

# MILESTONE 4 — Implement and test the Gymnasium environment

Create `src/lab/environment.py`.

Implement the Gymnasium methods:

- `__init__`: define action and observation spaces, numerical limits, and configuration.
- `reset(seed=None, options=None)`: initialize a new ray and return the initial observation plus metadata.
- `step(action)`: apply the action, advance the integrator, calculate reward, determine `terminated` and `truncated`, and return the Gymnasium five-value result.

Use a range of closest-approach values, not just (r_0=5). Start away from the numerically difficult photon-sphere region, then widen the range only after the environment is stable.

Tests must check:

- Observation shape and finite values.
- Action clipping and step-size bounds.
- Deterministic reset with a fixed seed.
- Correct escape detection.
- No-event maximum-angle truncation.
- Invalid numerical state handling.
- Reward is finite for every valid transition.

Do not train PPO until these tests pass.

---

# MILESTONE 5 — Train the first policy with PPO

Only after the environment is tested should we add the RL library.

The first algorithm will be Proximal Policy Optimization (PPO). PPO is a policy-gradient method that learns a stochastic policy while constraining how much the policy changes during an update. In common implementations, an actor network proposes actions and a critic network estimates expected return.

The agent's neural network is not trained from labelled ((r_0,\alpha)) pairs. It is trained from rollouts containing observations, actions, rewards, and episode outcomes.

## Training experiment

Create `scripts/train_ppo.py`.

The script will:

1. Construct the Gymnasium environment.
2. Run environment checks.
3. Create a PPO model with a recorded seed and hyperparameters.
4. Train for a small initial number of environment steps.
5. Save the policy and training configuration.
6. Record episode reward and termination statistics.

Start with a small run to validate the pipeline. Increase training only after inspecting reward curves and policy behavior.

Do not copy a large training configuration without understanding each parameter. We will introduce the code in small blocks and explain every line when implementing this milestone.

---

# MILESTONE 6 — Evaluate against numerical baselines

A trained policy is not evidence of improvement by itself.

Compare at least:

1. Fixed-step RK4.
2. A conventional non-learning step-size controller using the same RK4 step-doubling error estimate.
3. The PPO-controlled RK4 integrator.
4. SciPy DOP853 as the high-accuracy reference.

Use identical initial rays and the same declared error criteria.

Report:

- Global trajectory error at common angular samples.
- Final escape-angle error when escape occurs.
- Deflection-angle error.
- Number of right-hand-side evaluations.
- Wall-clock runtime.
- Failure and unresolved-episode rates.
- Performance across weak-field and near-critical regimes.

A policy that uses fewer steps but produces larger physical error is not necessarily better. Plot accuracy against computation rather than reporting only reward.

---

# MILESTONE 7 — Generalization and controlled experiments

Test initial conditions not used during training. In particular, evaluate whether the policy generalizes to:

- Different closest-approach radii.
- Weak-field rays.
- Rays closer to the photon sphere.
- Different maximum integration angles.
- Different requested accuracy tolerances.

Run multiple training seeds. Compare the learned policy with the conventional controller using the same simulation budget and test cases.

Possible research extensions, selected only after seeing the results:

- Train across a wider range of black-hole parameters.
- Compare PPO with SAC for continuous step-size actions.
- Study whether a policy trained on one parameter range transfers to another.
- Investigate reward designs that target a fixed global-error budget.

Do not add all extensions at once. Choose one clear hypothesis and one controlled comparison.

---

# MILESTONE 8 — Reproducibility and final report

The final report must include:

- Research question and why RL is appropriate for this numerical-control problem.
- Schwarzschild assumptions, equations, units, and event definitions.
- RK4 and step-doubling derivation.
- Environment observation/action/reward/termination specification.
- PPO architecture and training configuration.
- Baselines and reference-solver tolerances.
- Accuracy-versus-computation results.
- Multiple seeds and generalization results.
- Failure cases and limitations.
- Exact commands to reproduce every table and figure.

## Final completion checklist

- [ ] M0: existing reference solver has correct event metadata and no import-time side effects.
- [ ] M1: independently tested RK4 one-step method.
- [ ] M2: step-doubling error estimate and bounded controller.
- [ ] M3: written and reviewed MDP specification.
- [ ] M4: tested Gymnasium environment.
- [ ] M5: PPO training and saved policy.
- [ ] M6: fair comparison with fixed-step RK4, conventional adaptive RK4, and DOP853.
- [ ] M7: generalization tests and controlled experiment.
- [ ] M8: reproducible report and clean project instructions.

## How we will work through this guide

For each milestone:

1. I explain the mathematical and software concepts needed for that milestone.
2. I give you the file, function signature, constraints, and a focused implementation task.
3. You write the code yourself.
4. You share your implementation and output.
5. We debug it and inspect it line by line.
6. Only after it works do we move to the next milestone.

The guide is a roadmap, not a claim that future code already exists or has been tested. **Start with Milestone 0. Do not install PPO or write the agent yet.**
