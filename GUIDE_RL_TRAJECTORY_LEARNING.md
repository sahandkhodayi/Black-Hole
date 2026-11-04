# Black Hole Lab — Reinforcement Learning for Photon-Trajectory Generation

**Active research guide.** This replaces the previous idea of using RL only to choose numerical integration step sizes.

## Research objective

> **Can a reinforcement-learning agent learn to generate photon trajectories near a Schwarzschild black hole, generalize to unseen initial conditions, and reproduce gravitational light bending?**

The agent generates the light path sequentially. At each angular position, it observes the current photon state and proposes the next state. It does not choose the numerical solver's step size.

The existing geodesic equation and DOP853 implementation remain essential: they provide the physical model and reference trajectories. They are no longer the agent's action. Step-size control can remain a numerical baseline, but it is not the main research objective.

This is a research experiment, not a claim that an agent discovers general relativity from nothing. We supply the governing physics through the environment and reward, then test whether an RL policy can generate accurate, physically consistent curves.

---

# 1. Current code and target architecture

## 1.1 Existing implementation

The inspected repository contains:

- **src/lab/geodesics.py** — Schwarzschild photon ODE, escape event, DOP853 simulation, and RayTrajectory.
- **src/lab/integrators.py** — classical RK4 one-step method, rk4_step(rhs, phi, state, step_size).
- **src/lab/compare_plot.py** — early RK4/DOP853 comparison.
- **project.toml** — project metadata under this current filename.

The current RayTrajectory exposes phi, u, escaped, escape_angle, nfev, deflection_radian, and deflection_angle.

The existing comparison script is a prototype: it uses top-level imports, stops RK4 after u becomes non-positive without accurate event localization, and compares only u. We will preserve working code and improve it with tests rather than replacing everything.

## 1.2 Target repository structure

~~~text
src/lab/
    geodesics.py
    integrators.py
    reference.py
    trajectory_data.py
    trajectory_env.py
    trajectory_metrics.py
    policies.py

scripts/
    generate_dataset.py
    inspect_dataset.py
    train_rl.py
    evaluate_rl.py
    render_trajectories.py

tests/
    test_geodesics.py
    test_reference.py
    test_trajectory_data.py
    test_trajectory_env.py
    test_trajectory_metrics.py

results/
    dataset_manifest.json
    training_logs/
    models/
    evaluations/
    figures/
~~~

Create each file only when its milestone begins.

## 1.3 Environment setup

The repository metadata was inspected under the name project.toml. Python packaging expects pyproject.toml. Check the checkout; if it is still named project.toml, rename it:

~~~bash
git mv project.toml pyproject.toml
~~~

Then:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
~~~

Install RL dependencies only when the environment milestone begins:

~~~bash
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

---

# 2. Physics foundation

Use normalized Schwarzschild radius, so the event horizon is at r_s = 1. Define inverse radius:

\[
u(\phi)=\frac{1}{r(\phi)}.
\]

The orbit equation already implemented in this repository is:

\[
\frac{d^2u}{d\phi^2}=\frac{3}{2}u^2-u.
\]

Set v = du/dphi. The first-order system is:

\[
u'=v,\qquad v'=\frac{3}{2}u^2-u.
\]

The state is s = [u, v]. The current scattering setup starts at closest approach r0:

\[
u(0)=u_0=\frac{1}{r_0},\qquad v(0)=0.
\]

For this scattering initialization, use r0 > 1.5. The critical photon orbit is at r = 1.5 in the normalized units. Captured rays need a separate, physically correct incoming-ray initialization; do not pretend that the current closest-approach initialization already represents capture.

Escape occurs when u crosses zero in the outward direction. If the positive-side escape angle is phi_escape, the deflection is:

\[
\alpha=2\phi_{\mathrm{escape}}-\pi.
\]

To render the curve:

\[
r=1/u,\qquad x=r\cos\phi,\qquad y=r\sin\phi.
\]

At u = 0, radius is infinite. Mask or clip points beyond a plotting radius rather than passing infinity into the plotting code. Mirror the positive-side path across the x-axis for the complete symmetric scattering curve.

## 2.1 A conserved quantity

The ODE has a first integral. Starting from v' = 1.5u² - u and u' = v:

\[
2v\frac{dv}{d\phi}=(3u^2-2u)\frac{du}{d\phi}.
\]

Integrating:

\[
v^2=u^3-u^2+C.
\]

Therefore:

\[
I(u,v)=v^2-u^3+u^2
\]

is constant along a trajectory. At closest approach, v0 = 0, so:

\[
I_0=u_0^2-u_0^3.
\]

This invariant is a second physics diagnostic alongside the differential equation residual. It is useful for reward shaping, but does not replace comparison with the reference curve.

---

# 3. The RL task: generate the curve, not the step size

At angular position phi_t, the agent proposes the next state:

\[
(u_{t+1},v_{t+1}).
\]

The environment advances by a fixed angular increment:

\[
\phi_{t+1}=\phi_t+\Delta\phi.
\]

The sequence is:

\[
s_0\rightarrow s_1\rightarrow s_2\rightarrow\cdots\rightarrow s_T.
\]

**The angular increment is fixed by the environment.** The policy does not choose h. This makes the action explicitly about predicting the next point of the light path.

## 3.1 Observation

A first observation design is:

\[
o_t=[
u_0,\ u_t,\ v_t,\ \phi_t/\phi_{\max},\
u_t/u_0,\ e_{t-1}
].
\]

Where:
- u0 identifies the initial closest approach;
- ut and vt describe the current state;
- normalized phi describes trajectory progress;
- u/u0 describes progress toward escape;
- e_(t-1) is the previous transition error.

Normalize features using scales calculated from the training trajectories only. Save those scales and reuse them unchanged for validation and test. Do not give the policy the future reference state it is expected to predict.

Start with a small observation. Add history or other features only through ablations.

## 3.2 Action

Use a continuous two-dimensional action:

\[
a_t=(a_u,a_v),\qquad a_u,a_v\in[-1,1].
\]

Map it to a proposed next state:

\[
u_{t+1}=u_{\min}+
\frac{a_u+1}{2}(u_{\max}-u_{\min}),
\]

\[
v_{t+1}=v_{\min}+
\frac{a_v+1}{2}(v_{\max}-v_{\min}).
\]

Choose action limits from training-trajectory statistics with a safety margin, then freeze them. Never calculate scaling from the held-out test set. Reject non-finite proposals explicitly.

This first version uses absolute normalized next-state predictions because their meaning is easy to inspect. A later experiment can predict state increments instead. Record the action parameterization in each experiment configuration.

## 3.3 Transition

For each action, the environment:
1. converts the normalized action into proposed u and v;
2. advances phi by the fixed delta_phi;
3. measures reference-curve deviation;
4. measures differential-equation consistency;
5. measures invariant drift;
6. returns reward and the next observation;
7. terminates at a valid escape, invalid state, or episode limit.

This is a sequential trajectory-generation environment. It is not a numerical step-size optimizer.

---

# 4. Reward: trajectory accuracy and physical consistency

The first reward should combine three independently logged terms.

## 4.1 Reference tracking reward

For training trajectories, query the dense DOP853 reference at the next angular coordinate:

\[
s_{\rm ref}(\phi_{t+1})=
[u_{\rm ref}(\phi_{t+1}),v_{\rm ref}(\phi_{t+1})].
\]

Define normalized tracking error:

\[
E_{\rm ref}=
\left\|
\frac{s_{t+1}-s_{\rm ref}(\phi_{t+1})}{\sigma_s}
\right\|_2^2.
\]

The reference trajectory is a teacher during training. The policy still generates each next state through its action, but receives feedback about how close that state is to the known physical trajectory.

Be precise about this in the project report: this is **teacher-guided RL with trajectory-tracking reward**. It is not RL discovering the equations without data.

## 4.2 Differential-equation residual

For one transition with h = delta_phi, use midpoint residuals:

\[
R_u=(u_{t+1}-u_t)
-h\frac{v_t+v_{t+1}}{2},
\]

\[
u_m=\frac{u_t+u_{t+1}}{2},
\]

\[
R_v=(v_{t+1}-v_t)
-h\left(\frac{3}{2}u_m^2-u_m\right).
\]

Normalize each residual using fixed scales obtained from training data:

\[
E_{\rm ODE}=
(R_u/\sigma_u)^2+(R_v/\sigma_v)^2.
\]

These are discrete consistency residuals, not exact identities at finite h. Their magnitude depends on delta_phi.

## 4.3 Invariant reward

At each generated point compute:

\[
I_t=v_t^2-u_t^3+u_t^2,\qquad
I_0=u_0^2-u_0^3.
\]

Then:

\[
E_I=(I_t-I_0)^2/\sigma_I^2.
\]

This penalizes drift in a conserved quantity. Normalize it so it does not dominate the other reward terms merely because of its numeric scale.

## 4.4 Combined reward

Start from:

\[
r_t=-\lambda_{\rm ref}E_{\rm ref}
-\lambda_{\rm ODE}E_{\rm ODE}
-\lambda_I E_I.
\]

Add a terminal reward for a correct escape, and a penalty for invalid states or reaching the horizon without a valid outcome.

Do not choose weights blindly. First inspect the magnitudes of all terms on exact reference trajectories and deliberately perturbed trajectories. Normalize them, then tune weights on validation rays.

## 4.5 Reward ablations

Compare at least:
1. reference tracking + ODE residual + invariant;
2. ODE residual + invariant only;
3. reference tracking only;
4. ODE residual only.

This reveals whether the policy follows the curve because it imitates the reference, respects the physics, or both.

**Important research limitation:** the reference-tracking reward uses DOP853 teacher feedback during training. This is legitimate, but the claim must be that RL learns to reproduce reference trajectories from sequential feedback—not that RL independently discovers general relativity.

---

# 5. Generate a multi-trajectory reference dataset

One ray teaches only one initial condition. The goal is a policy that produces curves for unseen r0 values.

Generate reference trajectories across a scattering range with r0 > 1.5. Begin with a safe range such as 3 to 10 Schwarzschild radii. Later, add cases closer to the critical orbit only after checking numerical stability and reference accuracy.

Use more samples where bending changes rapidly. A uniform r0 grid is a reasonable first dataset, but it may not be sufficient near the critical region.

## 5.1 Data fields

Each trajectory should retain:
- r0 and u0;
- angular grid;
- reference u and v arrays;
- escape angle;
- deflection angle;
- solver name and tolerances;
- success and escape metadata.

The existing RayTrajectory stores u and phi but not v. For learning, v is essential. Obtain it from the raw solve_ivp result or extend the trajectory representation with a tested state array. Do not estimate v by finite differences of a sparse plotted curve when the solver already has the derivative state.

Prefer dense DOP853 output so all curves can be sampled on the same angular grid.

## 5.2 Split by initial condition

Do not randomly split points from trajectories. Adjacent points from a single ray are highly correlated; if they appear in both train and test, the evaluation leaks information.

Split by r0:
- training r0 values;
- validation r0 values;
- held-out test r0 values.

Keep the test values out of reward tuning, normalization, and checkpoint selection. Save exact split values and random seeds in a dataset manifest.

Create, at this milestone:
- src/lab/trajectory_data.py
- scripts/generate_dataset.py
- scripts/inspect_dataset.py

Save numeric arrays in NPZ or another structured format and metadata in JSON. Keep raw arrays separate from rendered images.

Acceptance criteria:
- same configuration and seed reproduce the same set of ray cases;
- all states are finite over the valid interval;
- every scattering trajectory has a valid escape and deflection;
- v is stored correctly;
- train, validation, and test r0 values are disjoint;
- plots of random dataset examples agree with the existing simulator.

---

# 6. Build the reference visualization gallery

Before training, visualize the curves the agent must learn.

For selected r0 values:
1. obtain the positive-side DOP853 trajectory;
2. convert u to r only while u is positive;
3. calculate x and y from r and phi;
4. mirror the trajectory across the x-axis;
5. draw the event horizon at radius one;
6. mask points outside a chosen plotting radius;
7. label the curve with r0 and deflection angle.

Include weak, moderate, and strong bending, with strong-bending cases close to—but above—the critical closest approach.

A plot is not an accuracy metric. Save the actual arrays and numerical errors too.

---

# 7. Build the Gymnasium trajectory environment

Install RL dependencies when the dataset and reference gallery are ready:

~~~bash
python -m pip install gymnasium stable-baselines3 tensorboard
~~~

Create src/lab/trajectory_env.py.

A Gymnasium environment implements:
- __init__: configuration and spaces;
- reset(seed=None, options=None): initialize one training ray and return observation/info;
- step(action): generate the next state and return observation, reward, terminated, truncated, info.

One episode represents one complete scattering ray.

At reset:
- select a training r0 using the seeded environment RNG;
- load its reference trajectory for reward calculation;
- initialize u = 1/r0, v = 0, phi = 0;
- clear generated points and reward accumulators.

At each step:
- the policy proposes next u and v;
- phi advances by fixed delta_phi;
- the reward evaluates teacher tracking, ODE residual, and invariant;
- the proposed state becomes the next current state if valid.

Terminate on valid escape or a documented invalid-state failure. Truncate on a maximum angular horizon or maximum number of decisions without valid completion.

Do not let the policy earn success by setting u <= 0 immediately. Check that the path is plausible, the crossing is outward, the escape angle is within a valid range, and trajectory quality has not been ignored.

## 7.1 Spaces and normalization

Use a bounded observation space containing initial u0, current u and v, normalized angle, progress, and previous error. Use a two-dimensional continuous action space for normalized next u and v.

Calibrate bounds from training trajectories only and freeze them. Test that all normal observations belong to the declared Box. Do not silently clip observations to hide a bad scaling choice.

## 7.2 Diagnostics

Return these in info:
- r0;
- phi;
- proposed u and v;
- reference u and v used for reward;
- tracking error;
- ODE residual;
- invariant error;
- cumulative reward;
- termination reason;
- escape angle where available.

Reference values are for reward and diagnostics, not policy observation.

## 7.3 Environment outline

This is an interface outline, not a complete runnable environment. Connect it to the dataset and reward implementation before training.

~~~python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PhotonTrajectoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset, delta_phi, observation_low, observation_high):
        super().__init__()
        self.dataset = dataset
        self.delta_phi = float(delta_phi)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Select a training trajectory with self.np_random.
        # Initialize u, v, phi and reference access.
        raise NotImplementedError

    def step(self, action):
        # Map action to proposed next u and v.
        # Advance phi by the fixed delta_phi.
        # Evaluate reward, escape, failure and horizon conditions.
        raise NotImplementedError
~~~

This skeleton must not be used for training until reset and step are fully implemented.

---

# 8. Test the environment before training

Add tests for:
- reset reproducibility under a fixed seed;
- initial state equals [1/r0, 0];
- every observation belongs to observation_space;
- action shape and values are handled safely;
- non-finite state cannot count as success;
- phi advances by exactly delta_phi;
- generated points are stored in order;
- a closer-to-reference proposal receives better tracking reward;
- physically consistent transitions have lower ODE residual;
- invariant error is measured against the correct I0;
- escape requires a valid outward crossing;
- horizon exhaustion is truncation, not success.

Run Gymnasium's environment checker on a small test dataset before training.

---

# 9. Baselines: do not make RL win by default

A credible project must compare RL with simpler approaches.

Use these baselines:
1. **DOP853 reference** — target curve and high-accuracy numerical reference.
2. **RK4** — conventional numerical integration.
3. **Supervised MLP next-state predictor** — trained on the same trajectory dataset.
4. **Physics-only integrator** — the governing ODE solved with a fixed, documented numerical method.
5. **Simple policy baseline** — a constant/extrapolation policy used to check environment behavior, not as the main scientific competitor.

The supervised MLP is essential. This is a deterministic differential equation with abundant reference trajectories; supervised learning may solve the mapping more efficiently than RL. We should compare it honestly.

A PINN can be added later as a physics-informed comparison. Do not add it before the RL environment and supervised baseline are functioning.

---

# 10. Train an RL agent

PPO is one possible algorithm, not the research objective. The objective is RL-based sequential photon-trajectory generation.

Start with PPO because it supports continuous bounded actions and has a practical Stable-Baselines3 implementation. Once it is stable, consider SAC as an off-policy comparison. Do not assume PPO is best without experiments.

The policy is an MLP that maps the observation to two continuous actions representing the proposed next u and v in normalized coordinates.

Training must:
- use training trajectories only;
- use validation trajectories for reward and hyperparameter tuning;
- save checkpoints;
- log each reward term separately;
- track trajectory error and invariant drift;
- track valid escape rate and episode lengths;
- preserve seeds and configuration.

Create scripts/train_rl.py only after the environment tests pass. First run a short smoke test. Inspect total reward, each reward component, tracking error, ODE residual, invariant drift, escape success, invalid-state rate, and rendered curves.

If reward improves while curve accuracy gets worse, the reward is exploitable or misweighted. Stop and diagnose instead of simply training longer.

---

# 11. Evaluate generated curves

## 11.1 Held-out initial conditions

Evaluate on r0 values absent from training and validation. Include weak, moderate, and strong bending. Keep near-critical cases in the valid scattering domain and extend the range only after the first experiment is stable.

## 11.2 Metrics

For every held-out ray, record:

| Metric | Meaning |
|---|---|
| u error | Difference in inverse radius on a common angular grid |
| v error | Difference in derivative state |
| Cartesian curve error | Difference in x/y after consistent radius masking |
| escape-angle error | Difference in predicted and reference escape angle |
| deflection error | Difference in bending angle |
| ODE residual | Local differential-equation consistency |
| invariant drift | Change in conserved quantity |
| escape success | Whether a valid escaping path was generated |
| rollout length | Number of agent decisions |

Compare curves on the same angular grid. Interpolate on the common valid interval; do not compare arrays of different lengths by their indices.

Put the agent curve and DOP853 reference on the same axes and include an error plot. A visually convincing curve can still have a poor escape-angle prediction.

## 11.3 Generalization tests

Test:
- unseen r0 values inside the training range;
- values near the edges of the training range;
- values closer to the critical orbit only after extending and validating the training range.

The first version is about escaping scattering rays. Captured trajectories are a later extension requiring an incoming-ray initialization and a new terminal condition.

## 11.4 Ablations

Change one thing at a time:
- teacher-tracking reward versus physics-only reward;
- with and without invariant penalty;
- observation with and without initial u0;
- different fixed angular increments;
- PPO versus SAC;
- RL versus supervised MLP.

These experiments reveal what the policy learned, not merely whether it trained.

---

# 12. Interpretation and success criteria

A convincing result should show:
- the agent generates complete, smooth scattering curves;
- escape and deflection predictions are accurate;
- generated states remain consistent with the ODE;
- conserved-quantity drift is controlled;
- the policy generalizes to unseen closest approaches;
- results are repeatable across seeds;
- comparisons with numerical and supervised baselines are fair.

RL may be less accurate or less efficient than a supervised model or conventional integrator. That is a valid finding. The goal is a rigorous experiment explaining what sequential reward-driven learning can and cannot do for this deterministic differential equation.

---

# 13. Reproducibility and engineering

Keep the existing geodesic model as the single source of truth. Standardize imports to package imports such as from lab.geodesics import simulate_ray.

Before the repository grows:
- put tests under the configured tests/ directory;
- remove tracked Python bytecode and ignore __pycache__/ and *.pyc;
- keep generated datasets and figures under results/;
- avoid committing huge transient logs/checkpoints unless intentionally publishing selected artifacts;
- record the Git commit associated with every result;
- keep training, validation, and test ray IDs separate.

A dataset manifest should record:
- physics equation and normalization;
- reference solver and tolerances;
- angular grid / delta_phi;
- training, validation, and test r0 lists;
- random seed;
- data-generation version.

Every RL run should additionally record:
- Git commit and dataset manifest;
- observation and action definitions;
- normalization scales;
- reward components and weights;
- algorithm and hyperparameters;
- seed and training budget;
- held-out metrics.

---

# 14. Milestones and exact next steps

| Milestone | Deliverable | Completion condition |
|---:|---|---|
| 1. Protect physics | geodesic tests | DOP853 results and escape behavior are tested |
| 2. Dense reference | reference.py | dense u and v queries work |
| 3. Dataset | trajectory_data.py and manifest | disjoint train/validation/test rays exist |
| 4. Reference gallery | plots | weak, moderate, strong bending displayed |
| 5. Metrics | trajectory_metrics.py | state, curve, escape, deflection, ODE and invariant metrics work |
| 6. RL environment | trajectory_env.py | agent proposes next state at fixed delta_phi; tests pass |
| 7. Baselines | RK4 and supervised MLP | fair baseline results on identical cases |
| 8. RL smoke test | training script | finite rewards, valid rollouts, curves render |
| 9. RL experiments | ablations | reward and policy behavior are understood |
| 10. Generalization | held-out evaluation | unseen r0 tested with reproducible metrics |
| 11. Algorithm comparison | PPO/SAC if justified | supported by fair evaluation |
| 12. Extension | captured rays / PINN | added only after scattering task works |

## Immediate next session

Do not install PPO or create the RL environment yet.

1. Add tests protecting the current geodesic solver and escape event.
2. Create a dense DOP853 reference exposing both u and v.
3. Generate a small gallery of reference curves across several r0 values.
4. Numerically verify the conserved quantity along those trajectories.
5. Design the dataset format and train/validation/test split.

The next major implementation is **trajectory data generation**, not step-size control.

---

# References

- SciPy solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Sutton and Barto, Reinforcement Learning: An Introduction: http://incompleteideas.net/book/the-book-2nd.html

**Guiding principle:** the agent learns to generate the light path. Numerical solvers provide the physics reference and scientific controls; they are not the agent's action.
