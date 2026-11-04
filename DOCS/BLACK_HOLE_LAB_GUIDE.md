> **Current guide:** This document is retained as an earlier research/implementation reference. The active implementation sequence, with code placed directly beside the milestone that uses it, is now in [the integrated RK4-to-PPO guide](../GUIDE_FROM_RK4_TO_PPO.md). Start there to avoid following overlapping older instructions.
> 
> Repository documentation index: [README.md](../README.md).
> 
---
# Black Hole Lab — Research & Implementation Guide

> **Current project state:** Schwarzschild photon trajectories are working with SciPy's DOP853 reference solver. A custom fixed-step RK4 integrator has been implemented, and `src/lab/compare_plot.py` currently compares its trajectory with DOP853. The next task is not to start PPO. It is to establish a trustworthy numerical baseline and quantify how RK4 behaves as its step size changes.

This is a living project guide. Completed work is recorded as completed; future chapters are implementation milestones. Each chapter explains the mathematical idea, why the project needs it, which files to change, what to implement, and how to verify the result. Do not skip the verification criteria.

---

## 0. Research question and project boundaries

### The research question

**Can a reinforcement-learning policy learn to choose numerical integration step sizes for Schwarzschild photon geodesics, balancing computational cost against numerical accuracy?**

The agent does **not** steer the photon, change gravity, or choose its physical trajectory. The physical problem is fixed. The agent controls the numerical solver's next step size.

The project therefore combines:

1. General-relativistic photon motion in a Schwarzschild spacetime.
2. Numerical solution of an ordinary differential equation (ODE).
3. Adaptive numerical integration and error estimation.
4. Reinforcement learning (RL), eventually using PPO.
5. Experimental evaluation against conventional numerical methods.

A learned controller is not automatically better than a conventional adaptive solver. A negative result is scientifically useful if the comparison is fair, reproducible, and accurate.

### What counts as success?

The final result must answer, with evidence:

- Can the policy complete photon trajectories reliably?
- How accurate are its trajectories and deflection angles?
- How many right-hand-side (RHS) evaluations does it require?
- How much wall-clock time does it require?
- Does it generalize to closest-approach radii not encountered during training?
- Does it outperform, match, or underperform a conventional adaptive controller?

Do not use a visually convincing plot as the only accuracy test.

---

## 1. Completed physics model

### 1.1 Normalization

The code uses normalized units in which the Schwarzschild radius is

\[
r_s = 1.
\]

The inverse-radius variable is

\[
u = \frac{1}{r}.
\]

The independent variable is the angular coordinate \(\phi\). For an equatorial photon trajectory, the reduced Schwarzschild orbit equation used in this project is

\[
\frac{d^2u}{d\phi^2} = \frac{3}{2}u^2-u.
\]

This is the physical ODE being solved. RK4, DOP853, and the future RL controller are numerical machinery around this same equation; they must not silently change it.

### 1.2 Convert the second-order equation to a first-order system

Define

\[
v = \frac{du}{d\phi}.
\]

Then the state is

\[
y =
\begin{bmatrix}
u\\v
\end{bmatrix},
\qquad
\frac{dy}{d\phi} =
\begin{bmatrix}
v\\
\frac{3}{2}u^2-u
\end{bmatrix}.
\]

This is exactly the role of `_rhs()` in `src/lab/geodesics.py`:

- `state[0]` is \(u\), inverse radius.
- `state[1]` is \(v=du/d\phi\).
- The returned pair contains the derivatives of those state components.

The state is **not** radius and radial velocity, and it is not the first and second derivatives. It is \([u,du/d\phi]\).

### 1.3 Initial conditions

At the point of closest approach \(r_0\), the inverse radius is largest and its angular derivative is zero:

\[
u(0)=\frac{1}{r_0},
\qquad
v(0)=0.
\]

The code accepts \(r_0\) as `closest_approach`. Scattering trajectories in this setup require \(r_0>1.5r_s\), outside the photon sphere.

### 1.4 Escape and deflection

The integration terminates when the outgoing ray crosses \(u=0\). The event function in `geodesics.py` returns \(u\), is terminal, and has direction \(-1\). The event time is the escape angle \(\phi_{\rm escape}\).

For the symmetric trajectory,

\[
\alpha = 2\phi_{\rm escape}-\pi.
\]

The code stores the escape and deflection information in `RayTrajectory`. A ray that did not reach the escape event must not be reported as an escaped ray and must not be assigned a deflection angle.

### 1.5 Existing implementation

| File | Current responsibility |
|---|---|
| `src/lab/geodesics.py` | ODE, escape event, DOP853 simulation, `RayTrajectory` |
| `src/lab/integrators.py` | Generic one-step fixed-step RK4 implementation |
| `src/lab/plot.py` | Plot several photon trajectories |
| `src/lab/compare_plot.py` | Compare fixed-step RK4 with DOP853 |
| `src/lab/lab_graph.py` | Inspect DOP853 state, phase-space path, and solver step sizes |
| `src/simulate.py` | Run the current simulation and plotting workflow |

**Status: physics simulation and visualization implemented.** Do not redo the basic geodesic derivation or the first ray plot as if they were new milestones.

---

## 2. Completed numerical method: classical RK4

### 2.1 What RK4 does

For a first-order system

\[
y'=f(\phi,y),
\]

one fixed RK4 step of size \(h\) is

\[
\begin{aligned}
k_1 &= f(\phi_n,y_n),\\
k_2 &= f\left(\phi_n+\frac h2,y_n+\frac h2k_1\right),\\
k_3 &= f\left(\phi_n+\frac h2,y_n+\frac h2k_2\right),\\
k_4 &= f(\phi_n+h,y_n+hk_3),\\
y_{n+1} &= y_n+\frac h6(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

Each \(k_i\) is a derivative estimate. The weighted sum advances the state. For this project, \(h\) is a step in \(\phi\), not a step in physical time or radius.

### 2.2 Current code

The implementation is in `src/lab/integrators.py` as `rk4_step(rhs, phi, state, step_size)`.

It:
1. converts the state to a NumPy floating-point array;
2. checks that the state is one-dimensional;
3. rejects non-finite or non-positive step sizes;
4. evaluates the four RK4 stages;
5. checks that each derivative has the same shape as the state;
6. returns the new state without mutating the caller's state array.

The function is deliberately generic: it knows how to integrate an ODE, but it does not know anything about black holes.

### 2.3 Existing comparison

`src/lab/compare_plot.py` integrates the ray with fixed-step RK4 and obtains a DOP853 solution through `simulate_ray_raw()`. It interpolates both solutions onto common angular samples, reports a maximum difference in \(u\), and plots their paths.

A previous run at \(r_0=5r_s\), \(h=0.01\), reported approximately:

- RK4 steps: 182
- RK4 RHS evaluations: 728 (four per complete step)
- DOP853 RHS evaluations: 1109
- maximum sampled difference in \(u\): \(6.70\times10^{-6}\)

These are one experiment's results, not a general proof that RK4 is more efficient. The current comparison also stops RK4 when a step crosses \(u=0\), so its last stored point is not necessarily the escape event. The next chapter improves the experiment rather than treating this first number as a final benchmark.

---

# Part II — Establish a reliable numerical baseline

## 3. Next milestone: convergence and accuracy study

<small>**Status: not yet completed. This is the immediate next task.**</small>

### 3.1 Why this comes before adaptive control

Before allowing an agent to choose \(h\), we need to know what happens when a human chooses \(h\). We need evidence that:

- reducing the step size improves the solution in the expected regime;
- the error metric is measured consistently;
- escape-angle and deflection-angle errors are tracked;
- computational cost is measured separately from accuracy.

Without this baseline, an RL reward has no defensible accuracy target and the final comparison has no trustworthy reference.

### 3.2 The experiment

Fix a closest approach first, for example

\[
r_0=5r_s.
\]

Run RK4 for a sequence of step sizes:

\[
h\in\{0.1,0.05,0.02,0.01,0.005,0.0025\}.
\]

Keep the physical initial condition identical across runs. Use DOP853 with tight tolerances as the numerical reference. Record the actual parameters and solver status with every result.

Do not compare arrays by index: different solvers have different sample locations. Compare them at the same \(\phi\) values.

### 3.3 Improve the comparison implementation

Work in `src/lab/compare_plot.py` or move reusable experiment logic into a new `src/lab/experiments.py`.

Implement these pieces in order:

**A. A result object**

Create a small dataclass such as `IntegrationResult` containing:

- sampled `phi`;
- sampled state values (at least `u` and `v`);
- number of accepted steps;
- RHS evaluation count;
- whether escape was detected;
- escape angle, if detected;
- elapsed wall-clock time.

Keep physical trajectory data separate from benchmark metadata if that makes the API clearer.

**B. Consistent endpoint handling**

The current RK4 loop breaks after a step produces \(u\leq0\), without recording the exact root. Improve it so the escape angle is estimated consistently. A simple linear interpolation between the last positive \(u_n\) and the first non-positive \(u_{n+1}\) is an acceptable first baseline:

\[
\phi_{\rm esc}\approx \phi_n+
h\frac{u_n}{u_n-u_{n+1}}.
\]

This is an approximation, not a high-order event locator. Record that limitation. Later, you can implement a more accurate event-location method.

Do not pass \(u=0\) into a Cartesian conversion that computes \(r=1/u\).

**C. Reference interpolation**

Use DOP853 dense output if available by requesting `dense_output=True` in `solve_ivp`. Evaluate the reference solution at a common set of \(\phi\) values inside the shared valid interval. Avoid extrapolation beyond either solution's domain.

**D. Error metrics**

At minimum calculate:

\[
E_u=\max_i|u_{\rm RK4}(\phi_i)-u_{\rm ref}(\phi_i)|
\]

and

\[
E_v=\max_i|v_{\rm RK4}(\phi_i)-v_{\rm ref}(\phi_i)|.
\]

Also calculate escape-angle error and deflection-angle error when both methods detect escape:

\[
E_\phi=|\phi_{\rm esc}^{\rm RK4}-\phi_{\rm esc}^{\rm ref}|,
\qquad
E_\alpha=|\alpha_{\rm RK4}-\alpha_{\rm ref}|.
\]

The trajectory errors and event errors answer different questions. Report both.

**E. Cost metrics**

For fixed-step RK4, each full step uses four RHS evaluations. Count the actual number of steps and account for any partial final step. Measure wall-clock time with `time.perf_counter()`; repeat runs and report a median because single timings are noisy.

For DOP853, use `solution.nfev`. Do not compare only the number of stored points: stored points and RHS evaluations are different quantities.

### 3.4 Expected convergence behavior

Classical RK4 has global error of order \(O(h^4)\) in the usual smooth, stable regime. If the step size is halved, the global error should approach a factor of 16 reduction once the asymptotic regime is reached.

Estimate the observed order from consecutive errors:

\[
p_{\rm observed}=
\frac{\log(E(h)/E(h/2))}{\log 2}.
\]

Do not expect exactly 4 for every coarse step size, or once reference error, event-location error, and floating-point effects dominate.

### 3.5 Deliverables

Create:

- a reproducible convergence table (CSV is suitable);
- a log-log plot of error against step size;
- a plot of RHS evaluations against error;
- a short experiment note stating machine, Python/package versions, parameters, and reference tolerances.

### 3.6 Verification checklist

- [ ] All runs use the same \(r_0\) and initial state.
- [ ] Both solvers are compared at common \(\phi\) values.
- [ ] No interpolation extrapolation is used.
- [ ] Escape is recorded as an event, not silently assumed.
- [ ] \(E_u\), \(E_v\), escape-angle error, and deflection-angle error are reported.
- [ ] RHS evaluations and wall time are measured separately.
- [ ] Error generally decreases as \(h\) decreases.
- [ ] The observed order is discussed, including deviations from 4.

**Do not move on until the table and plots are reproducible.**

---

## 4. Build an adaptive RK4 controller without RL

<small>**Status: future milestone.**</small>

The RL agent should eventually learn a job that can already be performed by a conventional controller. We need that conventional baseline first.

### 4.1 Step-doubling error estimate

A practical approach is to compare:

- one RK4 step of size \(h\);
- two consecutive RK4 half-steps of size \(h/2\).

Let the results be \(y_h\) and \(y_{h/2,h/2}\). For a fourth-order method, the difference estimates the local error scale. A common step-doubling estimate is

\[
e\approx\frac{y_{h/2,h/2}-y_h}{2^4-1}
=\frac{y_{h/2,h/2}-y_h}{15}.
\]

Use a scaled error norm rather than comparing raw components with one absolute threshold:

\[
\mathrm{scale}_i=\mathrm{atol}
+\mathrm{rtol}\max(|y_{n,i}|,|y_{n+1,i}|),
\]

\[
\mathrm{err}_{\rm norm}
=\sqrt{\frac{1}{m}\sum_{i=1}^{m}
\left(\frac{e_i}{\mathrm{scale}_i}\right)^2}.
\]

Here \(m\) is the number of state components (two in this project).

Accept a step when the normalized error is within the chosen tolerance. Otherwise reject it and retry with a smaller step. A rejected step must not advance \(\phi\) or change the accepted physical state.

### 4.2 Step-size update

A proportional controller can begin with

\[
h_{\rm new}=h\cdot
\mathrm{safety}\cdot
\mathrm{err}_{\rm norm}^{-1/5}.
\]

The exponent \(1/5\) follows from the local error scaling of a fourth-order method. Clamp the multiplier and the step size to safe minimum and maximum values. Handle zero error explicitly; do not divide by zero.

This is a starting controller, not a universal optimal formula. Later experiments can compare proportional and PI-style controllers.

### 4.3 Suggested code organization

Add a separate module, for example `src/lab/adaptive.py`, rather than making `rk4_step()` itself responsible for acceptance policy.

Suggested responsibilities:

- `rk4_step`: perform exactly one RK4 step;
- `estimate_step_error`: perform full-step/two-half-step comparison;
- `AdaptiveRK4`: own tolerances, safety factor, bounds, and accept/reject logic;
- trajectory runner: handle the ODE, escape event, maximum angle, and result collection.

The integrator should not import the black-hole geodesic module. Keep the numerical method generic.

### 4.4 Tests

- A small error should allow a step to be accepted.
- A large error should cause rejection and a smaller retry.
- Rejected attempts must not mutate the accepted state.
- The step size must remain within configured bounds.
- Invalid or non-finite states must fail clearly.
- Adaptive RK4 should be compared with the same DOP853 reference used in Chapter 3.

### 4.5 Deliverables

A working adaptive RK4 baseline, its error/cost table, and plots against fixed-step RK4 and DOP853.

---

# Part III — Formulate the reinforcement-learning problem

## 5. Understand the MDP before writing an environment

<small>**Status: future study and design milestone.**</small>

Read the relevant foundations in Sutton and Barto before implementing the agent. Focus on the Markov decision process, returns, value functions, and policy-based methods. Do not attempt to read the entire book before continuing the project.

### 5.1 What is one episode?

One episode is one photon-ray integration from its initial closest-approach state until escape, invalid numerical behavior, or a configured integration limit.

The environment contains the numerical solver. It does not simulate a different physical system.

### 5.2 State / observation

The observation should expose information useful for selecting the next numerical step. A first candidate is:

\[
o_t = [
\tilde\phi_t,
\tilde u_t,
\tilde v_t,
\widetilde h_{t-1},
\widetilde e_{t-1},
\widetilde\Delta\phi_t
].
\]

Possible meanings:

- normalized current angle;
- current inverse radius;
- current derivative;
- previous accepted step size;
- previous normalized error estimate;
- remaining angular interval or normalized distance to a configured limit.

Normalize values to reasonable ranges. Define what happens near escape, where \(u\to0\). Do not include hidden future information that a real controller would not have at decision time.

The observation design is an experiment. Start with a compact state and add features only when you can explain why they may help.

### 5.3 Action

The policy chooses a bounded continuous action controlling the next step size.

A useful parameterization is a log-scale multiplier:

\[
h_t=\mathrm{clip}(h_{\rm base}\exp(a_t),
h_{\min},h_{\max}).
\]

Alternatively, use a multiplier relative to the previous step:

\[
h_t=\mathrm{clip}(h_{t-1}\exp(a_t),
h_{\min},h_{\max}).
\]

Choose one definition and keep it consistent in the environment, documentation, and experiments. A bounded action prevents the policy from requesting negative, zero, or arbitrarily large steps.

### 5.4 Reward

The reward should express the research trade-off:

- numerical accuracy matters;
- computational work has a cost;
- successful completion is desirable;
- invalid states and failed trajectories are undesirable.

A conceptual per-step reward is

\[
r_t=-\lambda_e\,g(e_t)-\lambda_c\,c_t,
\]

with a terminal success/failure term. Here \(g(e_t)\) penalizes error relative to a tolerance, and \(c_t\) measures computation (for example, RHS evaluations).

Do not copy this formula blindly. Define the scale and units of every term. If the error penalty is always tiny compared with the compute penalty, the agent may learn to take reckless steps. If the success bonus dominates everything, it may learn to waste computation.

**Avoid reward hacking.** The agent must not get a high reward by terminating early, skipping difficult parts of the trajectory, or exploiting a weak error metric.

### 5.5 Termination versus truncation

- **Termination:** the physical ray successfully escapes, or the numerical state becomes irrecoverably invalid according to the task definition.
- **Truncation:** a configured maximum number of steps or integration limit is reached without completing the task.

Keep these concepts separate in the environment API. They have different meanings for RL algorithms.

### 5.6 Design document before code

Write down:

- observation fields and normalization;
- action definition and bounds;
- initial-condition distribution;
- integrator transition;
- error estimator;
- reward terms and coefficients;
- success, failure, termination, and truncation conditions;
- episode logging fields.

Do not begin PPO until this design can be explained and tested without a neural network.

---

## 6. Implement a Gymnasium environment

<small>**Status: future implementation milestone.**</small>

Create a dedicated module such as `src/lab/rl_env.py`.

The environment should follow Gymnasium's API:

- `reset(seed=None, options=None)` returns `(observation, info)`;
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.

### 6.1 Reset

At reset:

1. initialize the random generator through Gymnasium's seeding mechanism;
2. sample or receive a valid closest-approach radius;
3. initialize \(\phi=0\), \(u=1/r_0\), and \(v=0\);
4. initialize previous-step and previous-error features;
5. clear per-episode counters;
6. return a finite observation and diagnostic info.

Use a fixed evaluation set separate from randomized training episodes.

### 6.2 Step

At each step:

1. validate and transform the action into a bounded step size;
2. run the chosen numerical transition;
3. estimate numerical error;
4. detect escape and invalid states;
5. update counters and previous-step information;
6. calculate reward;
7. determine termination or truncation;
8. return observation, reward, flags, and info.

The environment's `info` should expose diagnostics such as current \(\phi\), \(u\), \(v\), chosen step, estimated error, accepted/rejected status, and cumulative RHS evaluations. These diagnostics are for debugging and evaluation; do not automatically put all of them in the policy observation.

### 6.3 Validate the environment before training

Use Gymnasium's environment checker. Also write tests for:

- reset reproducibility with the same seed;
- valid observation shape and finite values;
- action bounds;
- step counter behavior;
- successful escape;
- invalid-state failure;
- time-limit truncation;
- no calls to `step()` after the episode is done unless the API explicitly handles it.

First run episodes with random actions and inspect their logs. If random actions reveal broken transitions, fix the environment before training a policy.

---

# Part IV — Train the policy

## 7. RL theory needed for this project

<small>**Status: future study milestone.**</small>

Study these concepts in this order:

1. **MDP and return:** state, action, transition, reward, discount factor, episode return.
2. **Value functions:** state value \(V^\pi(s)\), action value \(Q^\pi(s,a)\), and advantage.
3. **Policy gradient:** directly optimize a parameterized policy.
4. **Actor-critic:** actor selects actions; critic estimates value.
5. **Advantage estimation:** reduce variance in policy-gradient updates.
6. **PPO:** constrain policy updates using the clipped surrogate objective.

The agent is a controller for numerical computation. Keep connecting every RL concept back to the solver: observation = current numerical situation, action = next step-size decision, reward = accuracy/computation trade-off.

## 8. PPO implementation and training

<small>**Status: future implementation milestone.**</small>

Use a proven PPO implementation such as Stable-Baselines3 for the first research baseline. The purpose is to investigate the numerical-control problem, not to reimplement every RL algorithm at once. A custom PPO implementation can be a later extension if it serves a clear research question.

### 8.1 Training setup

- Create a vectorized training environment only after a single environment is correct.
- Sample training closest-approach radii from a documented interval.
- Keep a deterministic validation set and a separate held-out test set.
- Fix and record random seeds.
- Normalize observations if needed; keep normalization statistics with the saved model.
- Save checkpoints and training logs.
- Evaluate periodically without exploration noise where appropriate.

### 8.2 Baselines that must exist

The final comparison must include:

1. fixed-step RK4 at several step sizes;
2. adaptive RK4 with the conventional step-doubling controller from Chapter 4;
3. PPO-controlled RK4;
4. SciPy DOP853 as a high-quality numerical reference.

DOP853 is a reference and practical solver baseline, not exact ground truth. Tighten tolerances and verify reference stability.

### 8.3 Training diagnostics

Track:

- episode return;
- successful escape rate;
- invalid/failure rate;
- episode length;
- mean and distribution of chosen step sizes;
- estimated error distribution;
- actual error against the reference on evaluation episodes;
- RHS evaluations per completed ray.

A rising reward alone does not prove that numerical performance is improving.

---

# Part V — Research evaluation

## 9. Fair evaluation protocol

<small>**Status: future research milestone.**</small>

### 9.1 Separate data by purpose

- **Training:** closest-approach radii used to learn the policy.
- **Validation:** used to choose reward coefficients, policy settings, and stopping criteria.
- **Test:** held out until the design is frozen.

Include interpolation points between training radii and radii outside the training interval to test generalization. Be especially careful near the photon-sphere boundary, where trajectories become more sensitive.

### 9.2 Metrics

For every method, report:

| Category | Metric |
|---|---|
| Accuracy | maximum sampled \(|u-u_{\rm ref}|\) |
| Accuracy | maximum sampled \(|v-v_{\rm ref}|\) |
| Event | absolute escape-angle error |
| Physical result | absolute deflection-angle error |
| Cost | RHS evaluations |
| Cost | accepted steps and rejected attempts |
| Runtime | median wall-clock time |
| Reliability | successful completion rate |
| Robustness | invalid and truncated episode rates |

Report distributions across multiple rays, not only a single favorable example. Include mean and standard deviation or median and interquartile range where appropriate.

### 9.3 Ablation experiments

Ablations help establish which design choices matter. Possible experiments:

- observation with and without previous error;
- action as absolute step size versus step-size multiplier;
- reward with and without an explicit error penalty;
- different accuracy/cost trade-off coefficients;
- training on a narrow versus broad closest-approach distribution.

Change one factor at a time and preserve the same evaluation set.

### 9.4 Interpret results honestly

Possible outcomes:

- PPO is more efficient at a similar accuracy;
- PPO achieves better accuracy at similar cost;
- PPO matches the conventional controller;
- PPO is less reliable or less efficient.

All are legitimate outcomes. If a conventional controller is better, explain why and identify what the RL policy learned or failed to learn. Do not hide unsuccessful runs or tune on the final test set.

---

# Part VI — Engineering and final deliverables

## 10. Project structure target

The repository can evolve toward a structure like this; do not create empty modules just to match the diagram.

```text
Black-Hole/
├── DOCS/
│   └── BLACK_HOLE_LAB_GUIDE.md
├── outputs/
│   ├── figures/
│   └── experiments/
├── src/
│   └── lab/
│       ├── geodesics.py
│       ├── integrators.py
│       ├── adaptive.py
│       ├── experiments.py
│       ├── rl_env.py
│       ├── train.py
│       ├── evaluate.py
│       └── plot.py
├── tests/
├── project.toml
└── .gitignore
```

Keep the existing modules until their responsibilities have been deliberately migrated. Avoid unrelated refactors during numerical experiments.

## 11. Reproducibility

Record:

- Python version;
- NumPy, SciPy, Matplotlib, Gymnasium, and RL-library versions;
- operating system and hardware;
- all solver tolerances and step bounds;
- training and evaluation seeds;
- closest-approach distributions;
- reward coefficients;
- PPO hyperparameters;
- checkpoint identifier;
- exact command used to run each experiment.

A result that cannot be reproduced is not a strong research result.

## 12. Final outputs

The finished project should contain:

- a correct and documented Schwarzschild photon model;
- fixed-step RK4 and convergence evidence;
- a conventional adaptive RK4 implementation;
- a tested Gymnasium environment;
- a trained PPO controller and saved configuration;
- evaluation against all baselines;
- accuracy-versus-cost plots;
- generalization and ablation results;
- a research report explaining limitations and conclusions;
- clear commands to reproduce the experiments.

---

# Immediate work queue

Do these tasks in order. The next working session should start with item 1, not PPO.

1. **Finish the RK4 convergence experiment** for \(r_0=5r_s\) over the step-size list in Chapter 3.
2. Fix event handling so RK4 and DOP853 escape angles can be compared consistently.
3. Add dense-output reference interpolation and calculate both \(u\) and \(v\) errors.
4. Produce the convergence table and log-log plot; explain whether the observed order approaches four.
5. Repeat for at least two additional closest-approach radii, including a more strongly bent ray while remaining outside \(1.5r_s\).
6. Only then implement the step-doubling adaptive RK4 controller.

## Current code-review notes to address as you progress

These are observations from the repository's current tracked source, not hypothetical requirements:

- `src/lab/compare_plot.py` imports `geodesics` and `integrators` as top-level modules. The other project entry points use the `lab` package. Standardize imports and run commands so modules work consistently from the repository root.
- `compare_plot.py` currently stores only positive-`u` RK4 samples and breaks on a crossing; it does not locate the RK4 escape event. Chapter 3 addresses this.
- `src/lab/geodesics.py` contains both `simulate_ray()` and `simulate_ray_raw()`, with duplicated solver settings. Later, consider a shared internal solver function so tolerances and event configuration cannot drift.
- `src/test.py` appears stale relative to the current `RayTrajectory` API: it refers to `solver_success` and `deflection_radians`, while the current dataclass exposes `escaped`, `deflection_radian`, and `deflection_angle`. It also sits outside the configured `tests/` directory. Reconcile tests with the actual API before relying on the test suite.
- `src/lab/test_rk4.py` is currently empty. Add focused unit tests for RK4, including a known ODE with a known solution, shape validation, and invalid step sizes.
- `project.toml` declares pytest and sets `testpaths = ["tests"]`, but the tracked tree currently shows no `tests/` directory. Establish one canonical test location.
- The tracked tree includes `__pycache__` bytecode files. Remove generated bytecode from version control and keep it ignored; it should not be part of the source repository.

These cleanup items matter, but do not let them distract from the immediate convergence experiment. Fix the test/import structure as part of the relevant chapter, not through a large unrelated rewrite.

---

## Learning resources

Use references as tools for the current milestone, not as a reason to pause implementation indefinitely.

- **Numerical ODEs:** study local versus global truncation error, stability, convergence order, adaptive step-size control, and event location. Use SciPy's `solve_ivp` documentation to understand DOP853, dense output, events, and `nfev`.
- **RL foundations:** Sutton & Barto, especially the early MDP/return/value-function material, followed by policy-gradient and actor-critic chapters when the project reaches them.
- **PPO implementation:** OpenAI Spinning Up for conceptual explanations; Stable-Baselines3 documentation for a practical implementation.
- **Environment API:** Gymnasium documentation for `reset`, `step`, seeding, spaces, and environment checking.

Useful links:

- SciPy `solve_ivp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Sutton & Barto: http://incompleteideas.net/book/the-book-2nd.html
- Spinning Up: https://spinningup.openai.com/en/latest/
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/

**Guiding rule:** every new concept must become a measured, tested part of this project. Learn enough theory to make the next implementation decision, implement it, inspect the result, and only then move to the next milestone.
