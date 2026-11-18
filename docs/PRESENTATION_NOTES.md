# Presentation notes — Black Hole Lab

## Suggested title

**Can AI Learn Photon Dynamics Around a Black Hole Without Being Given the Equations?**

Subtitle: Comparing black-box interaction, neural-network learning, and numerical physics.

## The central research question

Can learning systems approximate the behaviour of photons around a Schwarzschild
black hole when the governing geodesic equation is not supplied directly to
the learner? How much information does each learning approach require, and
how well does it generalize to unseen initial conditions?

This is a research question, not a claim that AI proves general relativity.

## Opening (about 45 seconds)

"Light paths are curved by gravity. I first built a numerical reference
simulation for photon trajectories around an idealized Schwarzschild black
hole. The research question is whether AI can learn aspects of this behaviour
without being directly given the governing differential equation. I compare
different information settings: an interaction-based reinforcement-learning
experiment, a neural network trained on reference trajectories, and a
physics-informed approach. Finally, I test the learned predictions against
held-out numerical simulations and known physical behaviour."

Do not say that the equation-hidden RL experiment is complete until its
separate black-box environment has been implemented and evaluated.

## Slide 1 — Why this is interesting

- A black hole changes the paths of nearby light.
- The Schwarzschild solution is a controlled, idealized setting.
- Numerical integration gives a trusted reference for experiments.
- The ML question is whether a model can reproduce unseen trajectories
  without being given the governing equation as an input.

## Slide 2 — Reference physics

Use dimensionless units with Schwarzschild radius r_s = 1.

- Event horizon: r = 1.
- Photon sphere: r = 1.5.
- Inverse radius: u = 1/r.
- Independent variable: azimuthal angle phi.
- State: (u, v), where v = du/dphi.

The reference equation is

    u'' + u = 1.5u^2

or, as a first-order system,

    u' = v,  v' = 1.5u^2 - u.

The initial conditions at closest approach are

    u(0) = 1/r_0,  v(0) = 0.

SciPy DOP853 integrates this system and detects outward escape at u = 0.
This equation is disclosed to the audience and used by the reference solver.
It is not supplied as an input feature to the supervised MLP.

## Slide 3 — Three information regimes

| Method | What the learner receives | What it tests |
|---|---|---|
| Black-box RL (planned) | Interaction and sparse outcome feedback; no pointwise DOP853 targets | Can an agent discover useful behaviour from outcomes? |
| Supervised MLP (implemented) | DOP853 state-transition examples, but no governing equation input | Can a neural model generalize from examples? |
| PINN (future) | Differential-equation residual and initial/boundary conditions | Do explicit physics constraints improve learning? |

The existing PPO trajectory environment is a separate **teacher-guided PPO
baseline**: its reward compares each predicted next state with a DOP853 target.
It is not the equation-hidden black-box RL experiment. Never combine the
results or call them the same method.

## Slide 4 — Supervised neural model

The MLP learns the one-step map

    (r_0, phi, u, v, delta_phi) -> (u_next, v_next)

from DOP853 transitions. Complete trajectories are assigned to train,
validation, or test splits. The final test rollout is autoregressive: each
predicted state is fed back into the next model input.

The key visual is the held-out comparison:
- solid line: DOP853 reference;
- dashed line: MLP prediction;
- multiple closest-approach radii, including strong and weak bending.

A low one-step validation loss is not enough. Prediction errors can
accumulate during autoregressive rollout.

## Slide 5 — Black-box RL experiment (planned)

The agent must not receive the geodesic equation, pointwise DOP853 target
states, or a reward based on pointwise DOP853 error. The environment exposes
only a declared black-box interaction and sparse outcome feedback.

A defensible first target is learning a critical capture/escape boundary or
another observable property from repeated interactions. After training, the
learned result can be compared with the theoretical photon-sphere behaviour.
The experiment's observation, action, reward, and termination rules must be
documented before training.

This tests discovery of a physical property through interaction. It does not,
by itself, mean the agent has learned a complete general-purpose simulator.

## Slide 6 — Evidence, not just attractive plots

Show:
1. DOP853 reference trajectories for weak, moderate, and strong bending.
2. MLP autoregressive predictions on held-out rays.
3. Training and validation loss history.
4. A separate RL learning progression once the black-box experiment exists.
5. Quantitative checks in a technical appendix: held-out errors, deflection
   angles, critical behaviour, and physical consistency.

Use identical initial conditions and plot scales for comparisons. State the
dataset size, split counts, seed, solver tolerances, and actual training
duration. A visually similar curve is evidence, not proof by itself.

## Slide 7 — Limitations and honest conclusions

- DOP853 is the numerical reference, not experimental astronomical data.
- The supervised MLP learns from DOP853-generated examples.
- The current PPO environment is teacher-guided, not equation-hidden.
- The black-box RL experiment is a separate implementation milestone.
- A PINN explicitly uses the equation and must be described as physics-informed.
- The small tracked v2 dataset is a smoke-test dataset; use a larger frozen
  dataset for final generalization claims.
- No model has proved Einstein's equations.

## Likely questions

**Did AI discover Einstein's equations?**

No. The reference equation comes from the Schwarzschild model. The
experiments test how well different learning methods approximate selected
behaviour under explicitly stated information constraints.

**Does the supervised MLP know the equation?**

It is not given the equation as an input or loss. It learns from numerical
state-transition examples generated by DOP853.

**Is the existing PPO fully autonomous?**

No. The existing PPO reward uses the next DOP853 state. It is a teacher-guided
trajectory-tracking baseline. The equation-hidden black-box RL task is
separate and must not be represented as completed yet.

**Why use DOP853?**

It provides a high-accuracy numerical reference for the specified ODE. It
allows controlled generation of trajectories and held-out comparisons.

**What does a successful result mean?**

A model reproduces trajectories or physical observables on held-out initial
conditions, with measured error and disclosed assumptions. It does not mean
that the model proves the underlying theory.

## Presenter checklist

- [ ] Use the new presentation runbook to recreate outputs.
- [ ] Show the reference geometry and the MLP held-out comparison.
- [ ] Keep solid-reference and dashed-prediction styles consistent.
- [ ] Record the dataset, split manifest, seed, commit, and training duration.
- [ ] Clearly distinguish planned black-box RL from existing teacher-guided PPO.
- [ ] Do not claim independent physics discovery or proof of general relativity.
