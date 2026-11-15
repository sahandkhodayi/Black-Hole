# Presentation notes — Black Hole Lab

These notes are a speaker guide for a short university presentation. The
project combines a numerical-physics simulation with an experimental
reinforcement-learning baseline.

## Suggested title

**Learning to Track Photon Trajectories Around a Schwarzschild Black Hole**

Subtitle: A numerical reference solver and a teacher-guided PPO experiment.

## 60-second opening

"Light does not travel in a straight line near a massive object. In this
project, I simulate the path of photons around an idealized Schwarzschild
black hole. First, I solve the photon-orbit equation numerically using
SciPy's DOP853 integrator. Then I formulate a reinforcement-learning task:
a PPO agent predicts the next point of a photon trajectory, receives a reward
based on its difference from the numerical reference, and feeds its own
prediction into the next step. The goal is to test how well a learned policy
can track these trajectories—not to claim that the agent discovered
Einstein's equations."

## Slide 1 — Motivation and objective

- Gravitational curvature changes the paths of light.
- The Schwarzschild model gives an idealized, non-rotating, uncharged black
  hole.
- Numerical integration provides reference photon trajectories.
- Machine learning experiment: can a sequential policy approximate those
  trajectories?

Say explicitly that this is a simplified model, not a full astrophysical
simulation.

## Slide 2 — Physics

Use normalized units with Schwarzschild radius (r_s=1). Define

[
u(\phi)=\frac{1}{r(\phi)}.
]

The orbit equation in this project is

[
\frac{d^2u}{d\phi^2}+u=\frac{3}{2}u^2.
]

Let (v=du/d\phi), giving

[
u'=v,\qquad v'=\frac{3}{2}u^2-u.
]

At closest approach (r_0),

[
u(0)=\frac{1}{r_0},\qquad v(0)=0.
]

The event horizon is at (r=1); the photon sphere is at (r=1.5). The
reference solver integrates escaping rays until (u\) approaches zero.

## Slide 3 — Numerical reference and dataset

- SciPy `solve_ivp` with DOP853 integrates the first-order ODE.
- The dataset stores complete rays in packed arrays: `case_id`, `r0`,
  `phi`, `u`, `v`, and `offsets`.
- Splitting is performed by whole ray, not by individual samples.
- The tracked split manifest contains 21 training, 4 validation, and 5 test
  rays.

Why split by ray? Samples from one trajectory are strongly related. Putting
points from the same ray into both train and test would leak trajectory
information and make evaluation too optimistic.

## Slide 4 — RL formulation

The policy observation is

[
o_t=\left[\frac{r_0}{12},\frac{\phi_t}{3},
\frac{u_t}{0.5},\frac{v_t}{0.2}\right].
]

The continuous action is (a_t\in[-1,1]^2), decoded as

[
u_{t+1}^{(p)}=0.25(a_1+1),\qquad
v_{t+1}^{(p)}=0.2a_2.
]

The reference next state is used by the environment for reward, but it is not
included in the policy observation. Define normalized tracking errors

[
e_u=\frac{u_{t+1}^{(p)}-u_{t+1}^{(*)}}{0.5},\qquad
e_v=\frac{v_{t+1}^{(p)}-v_{t+1}^{(*)}}{0.2}.
]

The current reward is

[
R_t=\frac{1}{1+e_u^2+e_v^2}.
]

It lies in ((0,1]): a perfect next-state prediction earns 1, and larger
normalized errors earn less. The policy is autoregressive: its predicted
state becomes the current state for the next decision.

## Slide 5 — PPO and experiment design

PPO (Proximal Policy Optimization) is a policy-gradient algorithm suitable
for continuous actions. Its clipped objective discourages excessively large
policy updates.

Current stable configuration:
- MLP policy with hidden layers [64, 64];
- learning rate (10^{-4});
- rollout length 512;
- batch size 128;
- 10 epochs per update;
- discount factor 0.99;
- GAE lambda 0.95;
- clip range 0.15;
- entropy coefficient 0.005;
- target KL 0.03.

These are experimental settings, not a theoretically optimal configuration.
The training reward is episode return, i.e. the sum of per-step rewards.
Since trajectories contain different numbers of samples, reward values are
not directly comparable across rays of different lengths.

## Slide 6 — Results and interpretation

Use the actual plots and metrics from
[PRESENTATION_RESULTS.md](PRESENTATION_RESULTS.md) and the generated files in `outputs/rl_figures/`.

For the reported `stable_v1` run:
- the held-out test set contains five complete rays;
- mean test RMSE in (u) decreased from about 0.159 to 0.1039;
- mean test RMSE in (v) decreased from about 0.111 to 0.0679;
- this corresponds to roughly 35% and 39% reductions, respectively.

These baseline values are read approximately from the initial-versus-trained
bar chart; the trained values are calculated from the five per-ray rows
provided in the evaluation CSV. Replace approximate initial values with
exact initial `summary.json` values if available.

The trajectory overlay still shows a visible mismatch: some PPO predictions
remain nearly flat while the DOP853 inverse-radius curves approach zero.
Therefore, the accurate conclusion is that PPO improved average test RMSE
relative to its untrained initialization, but has not yet learned accurate
full-trajectory tracking.

## Slide 7 — Limitations and next work

1. The reward uses DOP853 teacher targets: this is teacher-guided RL.
2. The episode ends at the final reference sample; PPO does not decide whether
   a ray escapes or is captured.
3. The dataset and held-out test set are small.
4. One seed does not establish robustness.
5. DOP853 remains the numerical accuracy reference.
6. A supervised MLP baseline is an important future comparison; a PINN is a
   separate future direction.

Next technical work should diagnose whether the policy learns near-constant
actions, inspect per-step errors, and compare against simple baselines before
adding complexity to the reward.

## Likely questions

**Did the AI discover Einstein's equations?**  
No. The orbit equation is specified by the Schwarzschild model. PPO learns a
policy to approximate trajectories generated by a numerical solver.

**Why use RL instead of supervised learning?**  
The experiment studies sequential decisions with feedback: each prediction
becomes the next state. This does not prove RL is better. A supervised MLP is
a necessary future baseline.

**Does PPO see the correct future state?**  
No. The future DOP853 state is used internally to calculate reward and
evaluation metrics. It is not part of the observation.

**Why is the PPO curve not identical to DOP853?**  
The policy has limited data and training, and it feeds its own imperfect
predictions back into subsequent decisions. Errors can accumulate.

**Why is DOP853 better?**  
DOP853 numerically integrates the specified differential equation at tight
tolerances. PPO is an approximate learned policy, not a replacement for the
numerical solver.

**What does the reward mean?**  
It measures one-step normalized state tracking error. Reward 1 means a
perfect next-state prediction; lower values indicate larger error.

**Does episode termination mean the agent detected escape?**  
No. The episode ends at the last sample of the reference trajectory. Escape
detection is not learned in this version.

**How did you avoid data leakage?**  
Whole trajectories, rather than individual samples, were assigned to train,
validation, or test splits.

**What is the strongest result so far?**  
On five held-out rays, the trained stable_v1 policy reduced average RMSE
relative to its initial policy. The trajectory overlay still shows substantial
tracking error, so this is preliminary progress rather than a solved task.

## Presenter checklist

- [ ] Open the orbit/u/v comparison figure.
- [ ] Open the initial-versus-trained test RMSE figure.
- [ ] Know the exact Git commit and training seed.
- [ ] Keep the five-ray test-set limitation visible.
- [ ] Do not call the model an autonomous black-hole simulator or physics
      discovery system.
- [ ] Keep the source code and repository available for questions.
