# Black Hole Lab — The Unified Research & Implementation Textbook

**Repository:** https://github.com/sahandkhodayi/Black-Hole  
**Primary research direction:** Reinforcement learning for adaptive numerical integration  
**Status:** Living guide — completed work is distinguished from planned work.

---

# How to use this guide

This document combines the three earlier guides into one project roadmap. It is not three books pasted together: overlapping chapters have been consolidated, conflicting research directions have been resolved, and the implementation sequence is organized around the project's current state.

Each milestone follows this learning cycle:

1. **Understand** the mathematical or engineering idea.
2. **Inspect** the existing implementation before changing it.
3. **Attempt** the task yourself.
4. **Implement** the smallest useful change.
5. **Test** behavior, including failure cases.
6. **Measure** results and save reproducible outputs.
7. **Explain** what the result means before moving on.

Code examples are reference patterns, not claims that files already exist or have been tested in your checkout. Adapt names and APIs to the actual repository. Never replace a working file wholesale just because a guide contains a longer version.

## Project status at the time this guide was assembled

### Completed

- Schwarzschild photon-ray equation and first-order state formulation.
- DOP853-based ray integration and escape-event handling.
- Symmetric ray visualization for different closest-approach radii.
- Generic classical fixed-step RK4 one-step integrator.
- Initial RK4-versus-DOP853 comparison.

### Immediate next work

**Do not start PPO yet.** First establish a reliable fixed-step convergence and accuracy benchmark. Then build a conventional adaptive RK4 controller. Only after that should the RL environment be designed.

### Research direction decision

The primary project is now **RL-controlled numerical integration**. The older supervised MLP/PINN guide is retained as an optional comparison or future extension. It is not a second mandatory roadmap running in parallel with RL. This avoids building several unrelated models before the core research question has been answered.

---

# PART I — Research question and scientific boundaries

## 1. The research question

> **Can a reinforcement-learning policy learn to control numerical integration step sizes for Schwarzschild photon geodesics, balancing computational cost against numerical accuracy?**

The agent does not steer the photon, alter gravity, or choose the physical trajectory. The physical ODE is fixed. The agent chooses the numerical solver's next angular step size.

This is a legitimate sequential decision problem:

- The numerical state changes after every integration step.
- The controller observes information about the current numerical state and recent error.
- It chooses a step size.
- That choice changes the next numerical transition, computational cost, and accumulated error.
- The episode ends when the ray escapes or a defined failure/limit condition occurs.

The project combines:

1. Schwarzschild spacetime and photon geodesics.
2. Initial-value problems and ordinary differential equations.
3. Numerical integration and error estimation.
4. Adaptive step-size control.
5. Markov decision processes and reinforcement learning.
6. Neural-network policies and value functions.
7. Controlled numerical experiments and scientific reporting.

## 2. What counts as success?

The final project must answer with evidence:

- Does the learned controller complete trajectories reliably?
- How accurate are its state trajectories?
- How accurate are its escape and deflection angles?
- How many RHS evaluations does it use?
- How much wall-clock time does it require?
- How does it compare with fixed-step RK4 and a conventional adaptive solver?
- Does it generalize to closest-approach radii not seen during training?
- Does it remain reliable across multiple random seeds?

RL does **not** have to beat DOP853 for the project to be successful. A carefully measured negative result is scientifically useful. A visually attractive plot or a high training reward is not sufficient evidence.

## 3. What the project is not

- It is not an RL agent controlling a real photon.
- It is not a claim that a neural network discovers general relativity.
- It is not a benchmark where one method is allowed a looser accuracy target than another.
- It is not successful merely because PPO's reward rises.
- It is not necessary to add a PINN, supervised predictor, or multiple algorithms before the main RL experiment works.

---

# PART II — Physics foundation (implemented)

## 4. Units and variables

Use normalized Schwarzschild-radius units:

\[
r_s=1,\qquad u=\frac{1}{r}.
\]

Here \(r\) is measured in units of the Schwarzschild radius, and \(u\) is inverse radius. The independent variable is the angular coordinate \(\phi\).

For an equatorial photon trajectory in Schwarzschild spacetime, the reduced orbit equation used in this project is

\[
\frac{d^2u}{d\phi^2}=\frac{3}{2}u^2-u.
\]

Equivalently,

\[
u''+u=\frac32u^2.
\]

This equation defines the physical problem. Every numerical method must solve the same equation.

## 5. Convert the second-order ODE to a first-order system

Define

\[
v=\frac{du}{d\phi}.
\]

Then

\[
y=
\begin{bmatrix}u\\v\end{bmatrix},
\qquad
\frac{dy}{d\phi}=
\begin{bmatrix}
v\\
\frac32u^2-u
\end{bmatrix}.
\]

In code:

```python
def rhs(phi, state):
    u, v = state
    return np.array([v, 1.5 * u**2 - u], dtype=float)
```

The state order is important:

- `state[0]` is \(u\), inverse radius.
- `state[1]` is \(v=du/d\phi\).
- The returned vector contains \((du/d\phi,dv/d\phi)\).

It is not `[radius, radial_velocity]`.

## 6. Initial conditions and the photon sphere

At closest approach \(r_0\), the inverse radius is at its maximum and its angular derivative is zero:

\[
u(0)=\frac{1}{r_0},\qquad v(0)=0.
\]

The photon sphere is at

\[
r=1.5r_s.
\]

The current scattering setup uses \(r_0>1.5r_s\). Behavior close to the critical orbit becomes numerically demanding; do not assume that a solver configuration suitable for \(r_0=5\) is automatically suitable near \(1.5\).

## 7. Escape and deflection

The outgoing ray escapes when it crosses \(u=0\). In SciPy, the event function returns `u`, is terminal, and has direction `-1`.

The event time is the escape angle \(\phi_{\rm escape}\). For the symmetric trajectory,

\[
\alpha=2\phi_{\rm escape}-\pi.
\]

The code must distinguish:

- **solver success:** the numerical integration completed without solver failure;
- **physical escape:** the configured escape event was detected.

A run can finish successfully at its maximum angle without escaping. In that case, it must not be assigned a deflection angle.

## 8. Existing physics implementation

Current repository components include:

| File | Responsibility |
|---|---|
| `src/lab/geodesics.py` | ODE, escape event, DOP853 simulation, trajectory result |
| `src/lab/integrators.py` | Generic one-step RK4 |
| `src/lab/plot.py` | Photon trajectory visualization |
| `src/lab/compare_plot.py` | RK4 and DOP853 comparison |
| `src/lab/lab_graph.py` | Solver/state inspection plots |
| `src/simulate.py` | Simulation entry point |

The exact current API should be checked in the repository before editing. Earlier versions of the guide used different dataclass field names; the live source code is authoritative.

---

# PART III — Numerical integration foundation

## 9. What an ODE solver returns

For an initial-value problem

\[
y'(\phi)=f(\phi,y),\qquad y(0)=y_0,
\]

SciPy's `solve_ivp` returns a result object. Important fields:

| Field | Meaning |
|---|---|
| `solution.t` | Independent-variable samples |
| `solution.y` | State values; each row corresponds to a state component |
| `solution.t_events` | Detected event coordinates |
| `solution.y_events` | State values at detected events |
| `solution.nfev` | Number of RHS evaluations |
| `solution.success` | Whether the integration completed successfully |
| `solution.message` | Human-readable solver status |
| `solution.sol` | Dense interpolant, if `dense_output=True` |

Do not assume `solution.t[-1]` is an escape event. Read `solution.t_events`. `nfev` is useful for measuring computational work; the number of saved points is not a substitute for it.

## 10. DOP853 as a reference solver

The project currently uses SciPy's DOP853, a high-order explicit Runge–Kutta method, as a high-accuracy numerical reference.

Typical settings already used in the project:

```python
solve_ivp(
    rhs,
    (0.0, max_angle),
    y0=(u0, 0.0),
    method="DOP853",
    rtol=1e-9,
    atol=1e-11,
    max_step=0.02,
    events=escape_event,
)
```

A numerical reference is not an exact analytical solution. To make reference uncertainty negligible compared with the method under test:

1. tighten tolerances;
2. reduce `max_step`;
3. compare successive reference configurations;
4. record the exact settings used;
5. avoid calling a reference “ground truth” without qualification.

## 11. Classical fixed-step RK4 (implemented)

For

\[
y'=f(\phi,y),
\]

one RK4 step of size \(h\) is

\[
\begin{aligned}
k_1&=f(\phi_n,y_n),\\
k_2&=f(\phi_n+h/2,y_n+hk_1/2),\\
k_3&=f(\phi_n+h/2,y_n+hk_2/2),\\
k_4&=f(\phi_n+h,y_n+hk_3),\\
y_{n+1}&=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

For this project, \(h\) is a step in \(\phi\), not physical time or radius.

The existing function is `rk4_step(rhs, phi, state, step_size)` in `src/lab/integrators.py`. It should remain generic: it knows how to advance an ODE, not what a black hole is.

Expected implementation properties:

- accepts a derivative function, current \(\phi\), state vector, and positive finite \(h\);
- evaluates four stages;
- validates derivative shape;
- returns a new state;
- does not mutate the caller's state;
- does not hardcode the Schwarzschild RHS.

## 12. Test RK4 independently before trusting it

Use an ODE with a known exact solution:

\[
y'=y,\qquad y(0)=1,\qquad y(\phi)=e^\phi.
\]

For one RK4 step, compare the numerical result with \(e^h\). Repeat with decreasing step sizes. This tests the integrator independently of the physics implementation.

Also test that the original input state remains unchanged and that invalid step sizes are rejected.

## 13. Existing RK4/DOP853 comparison

A previous run at \(r_0=5r_s\), \(h=0.01\), reported approximately:

- RK4 steps: 182
- RK4 RHS evaluations: 728
- DOP853 RHS evaluations: 1109
- maximum sampled difference in \(u\): \(6.70\times10^{-6}\)

This is one run, not proof that RK4 is generally more efficient. The existing comparison also needs improved endpoint handling: stopping after a step crosses \(u\le0\) does not locate the escape root exactly.

---

# PART IV — Establish the numerical baseline (next priority)

## 14. Milestone 1: fixed-step RK4 convergence study

**Status: immediate next milestone. Do this before adaptive integration or RL.**

### Why this matters

Before an agent chooses step sizes, we need to know what happens when we choose them ourselves. This study establishes:

- whether reducing \(h\) improves accuracy;
- whether the error metric is consistent;
- how trajectory errors relate to escape-angle and deflection-angle errors;
- the computational cost associated with each accuracy level.

Without this baseline, an RL reward has no defensible target.

### Experiment setup

Start with one fixed closest approach:

\[
r_0=5r_s.
\]

Use:

\[
h\in\{0.1,0.05,0.02,0.01,0.005,0.0025\}.
\]

Keep the physical initial condition identical for every run. Use a tightly configured DOP853 solution as the numerical reference.

Later, repeat for several \(r_0\) values, including a more difficult case nearer to the photon sphere and a weak-field case farther away.

### Implementation tasks

#### A. Create a structured integration result

Move reusable comparison logic out of plotting code if necessary, for example into `src/lab/experiments.py`.

A result structure should contain:

- sampled `phi`;
- sampled state values `u` and `v`;
- accepted step count;
- RHS evaluation count;
- escape detected flag;
- escape angle, if detected;
- elapsed wall-clock time;
- configuration and status metadata.

Keep physical trajectory data distinct from benchmark metadata where that improves clarity.

#### B. Locate the RK4 escape event

The current fixed-step loop may cross from positive \(u_n\) to non-positive \(u_{n+1}\). A first approximation is linear interpolation:

\[
\phi_{\rm esc}\approx\phi_n+
h\frac{u_n}{u_n-u_{n+1}}.
\]

This is only a first-order event-location approximation. Record the limitation. Do not convert the exact \(u=0\) event sample to \(r=1/u\) in a Cartesian plot.

Later, improve event location using a suitable continuous interpolant or a higher-order root estimate.

#### C. Compare at common angular coordinates

Different solvers produce samples at different \(\phi\) values. Never compare arrays by index.

Request `dense_output=True` for DOP853 and evaluate both trajectories at common \(\phi\) samples inside their shared valid interval. Do not extrapolate beyond either trajectory's domain.

#### D. Calculate error metrics

At common samples:

\[
E_u=\max_i|u_{\rm RK4}(\phi_i)-u_{\rm ref}(\phi_i)|,
\]

\[
E_v=\max_i|v_{\rm RK4}(\phi_i)-v_{\rm ref}(\phi_i)|.
\]

If both methods detect escape:

\[
E_\phi=|\phi_{\rm esc}^{\rm RK4}-\phi_{\rm esc}^{\rm ref}|,
\qquad
E_\alpha=|\alpha_{\rm RK4}-\alpha_{\rm ref}|.
\]

These metrics answer different questions. Report all of them.

#### E. Measure cost honestly

- RK4 uses four RHS evaluations per complete step.
- Count actual evaluations, including any partial final step.
- DOP853 exposes `solution.nfev`.
- Measure wall-clock time with `time.perf_counter()`.
- Repeat timing runs and report the median; single timings are noisy.
- Record accepted and rejected steps once adaptive methods exist.

### Expected convergence behavior

Classical RK4 has global error \(O(h^4)\) in the usual smooth, stable regime. Halving \(h\) should approach a factor of 16 reduction once the asymptotic regime is reached.

Estimate observed order:

\[
p_{\rm observed}=
\frac{\log(E(h)/E(h/2))}{\log 2}.
\]

Do not expect exactly four for every coarse step size. Reference error, event-location error, and floating-point effects can dominate.

### Deliverables

- `outputs/convergence.csv`
- log-log plot of error versus step size;
- plot of RHS evaluations versus error;
- short experiment note with machine, Python/package versions, \(r_0\), step sizes, and DOP853 settings.

### Completion gate

- [ ] Same physical initial condition for all runs.
- [ ] Common-\(\phi\) comparison.
- [ ] No extrapolation.
- [ ] Escape handled as an event.
- [ ] \(E_u,E_v,E_\phi,E_\alpha\) reported.
- [ ] RHS evaluations and wall time reported separately.
- [ ] Error generally decreases as \(h\) decreases.
- [ ] Observed order and deviations explained.

Do not move on until this experiment is reproducible.

---

## 15. Milestone 2: build a conventional adaptive RK4 controller

**Status: planned. This is the baseline RL must be compared against.**

RL should not be the first method that can adapt its step size. Build a conventional controller first.

### Step-doubling error estimate

Compare:

- one RK4 step of size \(h\), producing \(y_h\);
- two consecutive RK4 half-steps, producing \(y_{h/2,h/2}\).

For a fourth-order method, the step-doubling estimate is

\[
e\approx\frac{y_{h/2,h/2}-y_h}{2^4-1}
=\frac{y_{h/2,h/2}-y_h}{15}.
\]

Use a scaled error norm:

\[
s_i=\mathrm{atol}+
\mathrm{rtol}\max(|y_{n,i}|,|y_{n+1,i}|),
\]

\[
\mathrm{err}_{\rm norm}=
\sqrt{\frac1m\sum_{i=1}^m
\left(\frac{e_i}{s_i}\right)^2}.
\]

Here \(m=2\) for the state \([u,v]\).

Accept a step when the normalized error is within tolerance. Otherwise reject it and retry with a smaller step. A rejected attempt must not advance \(\phi\) or mutate the accepted state.

### Step-size controller

A starting proportional update is

\[
h_{\rm new}=h\cdot
\mathrm{safety}\cdot
\mathrm{err}_{\rm norm}^{-1/5}.
\]

The exponent \(1/5\) follows from the local error scaling of a fourth-order method. Clamp the multiplier and \(h\) to configured bounds. Handle zero error explicitly.

This is a starting controller, not a universal optimum. A PI controller can be evaluated later.

### Suggested module design

Create `src/lab/adaptive.py`.

Responsibilities:

- `rk4_step`: one numerical step only;
- `estimate_step_error`: full-step/two-half-step estimate;
- `AdaptiveRK4`: tolerances, safety factor, bounds, accept/reject;
- trajectory runner: ODE, escape event, maximum angle, result collection.

Keep the numerical integrator independent of the black-hole physics module.

### Tests

- Small error is accepted.
- Large error is rejected and retried smaller.
- Rejected attempts do not mutate accepted state.
- Step size stays within bounds.
- Invalid and non-finite states fail clearly.
- Adaptive RK4 is compared with the same DOP853 reference.

### Deliverables

A working adaptive RK4 baseline, accuracy/cost table, and plots against fixed-step RK4 and DOP853.

---

# PART V — Design the RL problem

## 16. Milestone 3: formulate the MDP before coding

**Status: planned design milestone.**

Read the relevant sections of Sutton and Barto on MDPs, returns, value functions, and policy-based methods. Study only what is needed for this implementation; do not block the project waiting to finish an entire textbook.

### One episode

One episode is one photon-ray integration, starting at closest approach and ending when:

- the outgoing ray escapes;
- the numerical state becomes irrecoverably invalid under the task definition;
- a configured step or integration limit is reached.

The environment contains the numerical solver. It does not create a different physical system.

### Observation / state

A candidate observation is

\[
o_t=[
\tilde\phi_t,\tilde u_t,\tilde v_t,
\widetilde h_{t-1},\widetilde e_{t-1},
\widetilde{\Delta\phi}_t].
\]

Potential features:

- normalized current angle;
- current inverse radius;
- current derivative;
- previous accepted step size;
- previous normalized error estimate;
- remaining angular interval or distance to a configured limit.

Normalize values to reasonable ranges. Define behavior near escape where \(u\to0\). Do not expose future information unavailable to a real controller.

Start with a compact observation. Add features only when you can explain and test why they may help.

### Action

Use a bounded continuous action that controls the next step size. One useful parameterization is

\[
h_t=\operatorname{clip}
(h_{t-1}\exp(a_t),h_{\min},h_{\max}).
\]

A bounded log multiplier makes multiplicative increases/decreases natural and prevents zero, negative, or unbounded steps.

Choose whether the multiplier is relative to the previous step or a fixed base step. Document it and keep the definition consistent across the environment, training, and evaluation.

### Reward

The reward expresses the research trade-off:

- accuracy matters;
- computational work costs something;
- successful completion is desirable;
- invalid states and failed trajectories are undesirable.

A conceptual reward is

\[
r_t=-\lambda_e g(e_t)-\lambda_c c_t,
\]

plus a terminal success/failure term. Here \(g(e_t)\) penalizes error relative to a tolerance, and \(c_t\) measures computational work such as RHS evaluations.

Every term needs a defined scale. If compute cost dominates, the policy may take reckless steps. If success bonus dominates, it may waste computation.

**Reward hacking risks:**

- terminating early to avoid difficult integration;
- taking huge steps and skipping the escape event;
- exploiting an error metric that only checks a few points;
- learning the training distribution rather than a general controller.

A successful episode must require an accurate, valid completed trajectory.

### Critical design question: where does the error signal come from?

An error estimator can be computed from step-doubling inside the environment. A high-accuracy reference solution can be used for offline evaluation. If reference-only information is used to calculate training rewards, explicitly state that design and ensure it does not leak future trajectory values into the observation or make evaluation circular.

A strong first design uses a solver-local error estimate for the controller and reserves the DOP853 reference for independent evaluation.

### Termination versus truncation

- **Terminated:** successful escape or a terminal invalid-state condition.
- **Truncated:** a maximum number of decisions or configured integration horizon is reached without completion.

Keep these distinct; Gymnasium and RL algorithms use them differently.

### Write the design document before coding

Specify:

- observation fields and normalization;
- action definition and bounds;
- initial-condition distribution;
- integrator transition;
- error estimator;
- reward terms and coefficients;
- success/failure criteria;
- termination and truncation;
- episode logging fields;
- random seed behavior.

Do not start PPO until this design can be explained without a neural network.

---

## 17. Milestone 4: implement a Gymnasium environment

**Status: planned.**

Create `src/lab/rl_env.py` or a clearly named environment module.

Gymnasium API:

```python
reset(seed=None, options=None) -> observation, info

step(action) -> observation, reward, terminated, truncated, info
```

### Reset

At reset:

1. initialize the random generator using Gymnasium's seeding mechanism;
2. sample or receive a valid closest-approach radius;
3. initialize \(\phi=0\), \(u=1/r_0\), \(v=0\);
4. initialize the step-size controller state;
5. clear counters and episode diagnostics;
6. return the initial observation and metadata.

### Step

At each action:

1. validate and clip the action to the declared action space;
2. convert it to a positive bounded step size;
3. advance the ODE using the chosen numerical method;
4. estimate local error;
5. detect invalid numerical states and escape;
6. calculate reward;
7. update counters and logs;
8. return observation, reward, terminated, truncated, info.

The `info` dictionary should expose diagnostics such as current step size, error estimate, RHS evaluations, accepted/rejected counts, current \(\phi\), and terminal reason. Avoid putting evaluation-only reference values into the policy observation.

### Environment validation before training

- `observation_space` contains every returned observation.
- `action_space` matches the policy action.
- all observations and rewards are finite;
- reset is reproducible with a fixed seed;
- invalid actions are handled predictably;
- success and failure conditions are tested;
- episode limits work;
- Gymnasium's environment checker passes.

Run a random policy and a constant-action policy. If these behave incorrectly, PPO will only make debugging harder.

---

# PART VI — Reinforcement learning and PPO

## 18. Milestone 5: learn the RL concepts needed

The policy is a neural network, but its training objective differs from supervised learning.

### Policy and value function

The **actor** represents a policy \(\pi_\theta(a\mid s)\), which chooses a distribution over actions given an observation.

The **critic** estimates the value \(V_\phi(s)\), the expected future return from a state.

For continuous actions, a common policy outputs a distribution (often Gaussian) whose parameters define action sampling. The action is then bounded/transformed to the environment's valid range.

### Return and discounting

The return is a discounted sum of rewards:

\[
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
\]

The discount factor \(\gamma\) determines how much future rewards contribute. Because the episode is a finite integration trajectory, choose and justify it rather than copying a default blindly.

### Advantage

An advantage estimate describes whether an action did better or worse than expected under the current value estimate. PPO commonly uses generalized advantage estimation (GAE), controlled by \(\lambda\).

### PPO clipped objective

PPO compares the probability ratio

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\rm old}}(a_t\mid s_t)}
\]

and optimizes a clipped surrogate objective:

\[
L^{\rm CLIP}(\theta)=
\mathbb E_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
\]

Clipping discourages excessively large policy updates. PPO is not guaranteed to find the best controller; training must be evaluated across seeds and against baselines.

### Study checklist

Understand:

- policy versus value function;
- stochastic continuous policies;
- return and discount factor;
- advantage and GAE;
- policy gradient intuition;
- PPO probability ratio and clipping;
- entropy bonus;
- value loss;
- rollout collection and minibatch updates;
- reward normalization and observation normalization.

---

## 19. Milestone 6: train the first PPO policy

**Status: planned. Train only after the environment and baselines pass.**

Use Stable-Baselines3 PPO initially rather than writing PPO from scratch. The goal is to study the control problem, not spend the first iteration debugging an RL library implementation.

A training script should:

1. create the environment;
2. wrap it for vectorized rollout if needed;
3. set seeds;
4. configure policy network and PPO hyperparameters;
5. train for a declared number of environment steps;
6. save checkpoints and logs;
7. evaluate periodically on fixed validation scenarios;
8. save the final policy and full configuration.

Record at minimum:

- seed;
- total environment steps;
- learning rate;
- rollout length;
- batch size;
- number of epochs;
- discount factor;
- GAE lambda;
- clipping coefficient;
- entropy and value coefficients;
- network architecture;
- observation normalization settings;
- reward definition and coefficients.

Do not tune every parameter simultaneously. Begin with a modest baseline configuration, establish that learning is stable, then change one factor at a time.

### Training diagnostics

Monitor:

- episodic return;
- episode length;
- successful escape rate;
- invalid-state rate;
- mean and distribution of selected step sizes;
- estimated error;
- RHS evaluations;
- value loss;
- policy loss;
- entropy;
- approximate KL divergence;
- explained variance.

A rising reward is not enough. The agent may be exploiting the reward. Always inspect actual numerical accuracy and completion.

---

# PART VII — Scientific evaluation

## 20. Milestone 7: compare methods fairly

Compare at least:

1. fixed-step RK4 at several \(h\) values;
2. conventional adaptive RK4;
3. DOP853;
4. RL-controlled RK4.

Use the same physical initial conditions and comparable accuracy requirements. Do not compare an accurate slow method against an inaccurate fast method and call the latter better.

### Metrics

| Category | Metrics |
|---|---|
| Accuracy | max \(u\) error, max \(v\) error, escape-angle error, deflection-angle error |
| Work | RHS evaluations, accepted steps, rejected attempts |
| Runtime | median wall-clock time and spread |
| Reliability | escape success rate, invalid-state rate, truncation rate |
| Policy behavior | step-size distribution, step-size changes, error estimates |
| Generalization | performance on held-out closest-approach radii |

Plot cost versus error. This is more informative than one number or one trajectory plot.

### Evaluation protocol

Separate scenarios by purpose:

- **training:** used for policy updates;
- **validation:** used for design choices and checkpoint selection;
- **test:** held back until the final evaluation.

Do not repeatedly tune against the test set.

Use several random seeds for training. Report mean and standard deviation or confidence intervals where appropriate. Save per-episode data, not only aggregate metrics.

### Generalization

Train on a range of \(r_0\) values and test on:

- unseen values inside the training interval;
- values near (but outside) the training interval;
- more difficult trajectories nearer the photon sphere;
- weak-field trajectories farther away.

Do not claim generalization from a random split if near-identical trajectories appear in both training and test sets.

### Ablations

Once the main comparison works, perform controlled ablations such as:

- remove previous error from the observation;
- remove previous step size;
- change reward weights;
- narrow or widen action bounds;
- train on a narrower or wider \(r_0\) distribution.

Change one factor at a time.

### Interpret results honestly

A policy may:

- learn a useful heuristic;
- match a conventional controller;
- reduce RHS evaluations while slightly worsening accuracy;
- fail to generalize;
- be slower than DOP853 because of policy inference overhead.

All are valid findings if measured correctly. Include inference overhead in wall-clock comparisons.

---

# PART VIII — Optional supervised MLP and PINN branch

This section preserves the useful material from the earlier MLP/PINN guide without confusing it with the primary RL research question.

**Do not start this branch until the RL controller has a defensible baseline and evaluation.** If the research question changes back to neural prediction, revise the project definition explicitly.

## 21. Optional scalar prediction task

A supervised model could predict a scalar physical quantity, such as deflection angle, from a parameter such as closest-approach radius or impact parameter.

This is a different task from RL:

- supervised model: maps input parameters to a predicted output;
- RL policy: selects numerical actions sequentially during integration.

A scalar MLP does not control the solver.

### Dataset generation

Generate samples using the trusted numerical solver. For every sample, store:

- input parameter(s);
- target deflection angle;
- escape status;
- solver tolerances and configuration;
- any metadata needed to reproduce the sample.

Do not silently include failed or unresolved trajectories as valid labels.

Use a deterministic dataset-generation script and save the random seed and solver configuration.

### Baseline model

Start with a simple model and a fixed train/validation/test split. Scale inputs when appropriate. Compare against a simple non-neural baseline, not only against the training loss.

Evaluate MAE, RMSE, residual plots, and behavior outside the training interval. Never use test data to choose architecture or hyperparameters.

## 22. Optional PINN task

A PINN uses the governing differential equation as part of the training objective. For a predicted function \(u_\theta(\phi)\), the residual is

\[
\mathcal R_\theta(\phi)=
\frac{d^2u_\theta}{d\phi^2}
+u_\theta-\frac32u_\theta^2.
\]

A conceptual loss is

\[
\mathcal L=
\lambda_{\rm data}\mathcal L_{\rm data}
+\lambda_{\rm physics}\mathcal L_{\rm physics}
+\lambda_{\rm IC}\mathcal L_{\rm IC}.
\]

Automatic differentiation computes derivatives of the neural network output with respect to \(\phi\). The initial conditions must be enforced either through the loss or an architecture that satisfies them by construction.

A PINN is not automatically more accurate than a numerical solver. Compare it against the DOP853 reference and report residuals, boundary/initial-condition errors, and generalization.

## 23. How the optional branch can support the main project

Possible future research questions:

- Does a learned scalar surrogate approximate deflection faster than numerical integration?
- Does a PINN satisfy the ODE better than a data-only MLP?
- Can a surrogate help initialize or diagnose the RL controller?

These are separate experiments. Do not mix their metrics with the RL step-control results.

---

# PART IX — Software engineering and repository structure

## 24. Target repository structure

The structure below is a target, not a claim that every file already exists. Keep the current repository working and introduce files as milestones require them.

```text
Black-Hole/
├── pyproject.toml
├── README.md
├── DOCS/
│   └── BLACK_HOLE_LAB_GUIDE.md
├── src/
│   ├── simulate.py
│   └── lab/
│       ├── __init__.py
│       ├── geodesics.py
│       ├── integrators.py
│       ├── adaptive.py
│       ├── experiments.py
│       ├── rl_env.py
│       ├── train_rl.py
│       ├── evaluate_rl.py
│       └── plotting.py
├── scripts/
│   ├── convergence.py
│   ├── validate_weak_field.py
│   ├── critical_sweep.py
│   └── evaluate_methods.py
├── tests/
│   ├── test_geodesics.py
│   ├── test_integrators.py
│   ├── test_adaptive.py
│   └── test_rl_env.py
├── data/          # generated locally; not necessarily committed
├── outputs/       # plots, tables, logs
└── configs/       # experiment configurations
```

Optional MLP/PINN modules can be added later if that branch is approved.

## 25. Packaging and environment

The earlier code-first guide proposed a `pyproject.toml` using setuptools, a `src` layout, and pytest. The current repository has previously used `project.toml`; inspect the actual tracked filename before deciding whether to rename or consolidate it. Do not keep two competing project configuration files.

A typical environment setup is:

```bash
python -m venv .venv

# Linux / WSL
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use `python -m pip` so installation targets the active interpreter.

Recommended ignore patterns:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
outputs/*
!outputs/.gitkeep
data/*.csv
data/*.npz
*.pt
```

Keep the maintained project guide explicitly tracked if a broad `*.md` ignore rule exists. Generated data and model checkpoints should not accidentally enter version control.

## 26. Tests and code quality

Test each layer independently:

- physics RHS and initial conditions;
- escape-event classification;
- radius conversion;
- RK4 on a known analytic ODE;
- RK4 input immutability and invalid parameters;
- adaptive accept/reject behavior;
- environment API and deterministic reset;
- reward and termination edge cases.

Run tests from the repository root:

```bash
python -m pytest -q
```

Keep imports package-consistent. Avoid mixing bare sibling imports with package imports. Avoid simulation or plotting side effects when a module is imported. Use `if __name__ == "__main__":` for executable demonstrations.

## 27. Reproducibility

Every experiment should save:

- source revision / Git commit;
- Python and dependency versions;
- operating system and hardware;
- random seed;
- physical parameters and initial-condition distribution;
- solver method and tolerances;
- RL configuration;
- data split identifiers;
- output metrics.

Use configuration files rather than burying all experiment parameters in code. Save raw episode-level results as CSV or another transparent format. Figures should be generated from saved results, not manually edited screenshots.

---

# PART X — Final research report and release

## 28. Final report structure

1. **Abstract:** question, method, main result.
2. **Introduction:** why numerical step-size control matters.
3. **Physical model:** Schwarzschild orbit equation and assumptions.
4. **Numerical methods:** DOP853, RK4, error estimation, adaptive RK4.
5. **RL formulation:** observation, action, reward, termination.
6. **Training:** PPO configuration, seeds, and diagnostics.
7. **Experimental protocol:** data/scenario splits and baselines.
8. **Results:** accuracy, cost, reliability, runtime, generalization.
9. **Ablations:** what design choices mattered.
10. **Limitations:** reference-solver uncertainty, event handling, reward design, compute, generalization.
11. **Conclusion:** answer the research question without overstating it.

## 29. Final deliverables

- working simulator;
- tested fixed-step RK4;
- convergence study;
- conventional adaptive RK4;
- tested Gymnasium environment;
- trained PPO policy and configuration;
- evaluation scripts;
- baseline comparison tables and plots;
- saved episode-level results;
- reproducibility instructions;
- final research report;
- clean README and tagged release.

---

# PART XI — Ordered roadmap and completion gates

| Order | Milestone | Status / gate |
|---:|---|---|
| 0 | Schwarzschild ray simulation and visualization | Implemented; preserve and test |
| 1 | Generic RK4 one-step integrator | Implemented; strengthen tests |
| 2 | Fixed-step convergence and event-accuracy study | **Next task**; reproducible table and plots required |
| 3 | Conventional adaptive RK4 | Planned; accept/reject tests and DOP853 comparison |
| 4 | MDP and reward design | Planned; written specification before coding |
| 5 | Gymnasium environment | Planned; checker and deterministic tests pass |
| 6 | PPO baseline training | Planned; stable diagnostics across seeds |
| 7 | Scientific evaluation and generalization | Planned; fair baselines and held-out cases |
| 8 | Ablations and interpretation | Planned; controlled experiments |
| 9 | Report, reproducibility, release | Planned; another person can reproduce results |
| Optional | Supervised MLP / PINN comparison | Deferred; separate research question |

## Immediate work queue

Do these in order:

1. Inspect the current `compare_plot.py`, `integrators.py`, and `geodesics.py` before editing.
2. Refactor only the reusable comparison logic; do not rewrite working physics code unnecessarily.
3. Add exact/common-\(\phi\) reference sampling using DOP853 dense output.
4. Add RK4 escape-angle estimation and clearly label its accuracy.
5. Calculate \(E_u\), \(E_v\), escape-angle error, and deflection-angle error.
6. Count RHS evaluations and measure repeated median runtime.
7. Run the specified step-size sweep.
8. Save CSV and plots under `outputs/`.
9. Explain whether observed convergence matches RK4's expected order.
10. Only then begin adaptive RK4.

---

# PART XII — Learning resources

Use resources to answer the next implementation question, not as a substitute for building and testing.

## Numerical methods and ODEs

- SciPy `solve_ivp` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- SciPy integration tutorial: https://docs.scipy.org/doc/scipy/tutorial/integrate.html

Study: initial-value problems, local/global error, Runge–Kutta methods, dense output, event detection, adaptive step-size control.

## Reinforcement learning

- Sutton & Barto, *Reinforcement Learning: An Introduction*: http://incompleteideas.net/book/the-book-2nd.html
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

Study in sequence: MDPs → returns/value functions → policy gradients → actor-critic → PPO → Gymnasium API → evaluation.

## Physics

Use a reliable general-relativity reference for Schwarzschild geometry, null geodesics, photon sphere, impact parameter, and weak-field deflection. Keep assumptions and units explicit in the report.

---

# Appendix A — Important equations at a glance

**Schwarzschild photon orbit**

\[
u''=\frac32u^2-u
\]

**First-order state**

\[
u'=v,\qquad v'=\frac32u^2-u
\]

**Initial conditions**

\[
u(0)=1/r_0,\qquad v(0)=0
\]

**Escape and deflection**

\[
u(\phi_{\rm escape})=0,\qquad
\alpha=2\phi_{\rm escape}-\pi
\]

**RK4 global order**

\[
E(h)=O(h^4)
\]

**Observed convergence order**

\[
p=\frac{\log(E(h)/E(h/2))}{\log 2}
\]

**Step-doubling estimate**

\[
e\approx\frac{y_{h/2,h/2}-y_h}{15}
\]

**Adaptive step proposal**

\[
h_{\rm new}=h\cdot\mathrm{safety}\cdot
\mathrm{err}_{\rm norm}^{-1/5}
\]

**RL action mapping example**

\[
h_t=\operatorname{clip}
(h_{t-1}\exp(a_t),h_{\min},h_{\max})
\]

**Conceptual reward**

\[
r_t=-\lambda_e g(e_t)-\lambda_c c_t
\]

---

# Appendix B — Rules for maintaining this guide

- Update the status at the top whenever a milestone is completed.
- Never mark a milestone complete because code was merely written; tests and evidence are required.
- If the repository API differs from an example, update the example to match the source rather than forcing the source to match stale documentation.
- Keep one primary research question.
- Keep optional experiments clearly optional.
- Record failed experiments and unexpected results; do not erase them to make the project look successful.
- Every result in the report must be reproducible from a script and saved configuration.
