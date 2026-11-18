# Black Hole Lab — presentation experiment protocol

## Research question

Can learning systems approximate photon trajectories around a Schwarzschild black hole when the governing geodesic equation is not supplied to the learner? How do black-box reinforcement learning, supervised neural regression, and physics-informed learning differ in the information they require and in their ability to generalize?

This is a comparison of learning methods, not a claim that AI proves general relativity.

## Scientific honesty: three different information regimes

### A. Black-box RL / system identification

The agent is not given the geodesic equation, DOP853 target trajectories, or next-state labels. It interacts with a black-box environment and receives only declared observations and outcome feedback.

The environment necessarily contains a physical process or external oracle. Therefore this is not "AI with no human setup" and not discovery from nothing. The environment interface, action space, reward, and experiment boundaries are designed by us. Document exactly what the agent can observe and what the reward reveals.

A valid first experiment is to let an agent choose controlled ray-launch parameters or propose local trajectory actions, and receive sparse outcome feedback from a black-box renderer/simulator. Do not secretly use pointwise DOP853 error as reward in this track. If the reward uses known reference states, label it teacher-guided imitation, not autonomous RL.

### B. Supervised neural surrogate

The MLP receives DOP853 state-transition examples. It is allowed reference data but is not given the governing equation. Train on complete training rays, select checkpoints on validation rays, and evaluate once on held-out test rays. The final evaluation must be autoregressive: feed each prediction into the next model input.

### C. Physics-informed neural network (PINN)

The PINN is explicitly given the differential-equation residual and boundary/initial conditions. It is therefore physics-informed, not equation-free discovery. Its value is to test whether physics constraints improve accuracy or reduce the amount of reference data required.

For the dimensionless Schwarzschild photon equation in this repository:

`u'' + u - 1.5u^2 = 0`

A PINN minimizes a residual such as

`R(phi) = d²u_theta/dphi² + u_theta - 1.5 u_theta²`

along with appropriate initial/boundary constraints. Compare data-only, physics-only (where well-posed), and combined losses carefully. The equation and its coefficient must be disclosed in the presentation.

### D. Existing teacher-guided PPO

The existing `PhotonTrajectoryEnv` gives reward by comparing each proposed next state to a DOP853 next state. Keep this as a clearly labelled teacher-guided PPO baseline. It is not the black-box RL experiment above. Do not merge results or call them the same method.

## Reference physics for the class

Use dimensionless units with Schwarzschild radius `r_s = 1`:

- Event horizon: `r = 1`.
- Photon sphere: `r = 1.5`.
- Inverse radius: `u = 1/r`.
- Independent variable: azimuthal angle `phi`.
- First-order state: `(u, v)`, where `v = du/dphi`.
- Geodesic equation: `u'' = 1.5u² - u`.
- Numerical reference: SciPy DOP853 with a documented event at `u = 0` for outward escape.

Explain that the photon sphere is an unstable circular photon orbit. Rays near the critical impact parameter can bend strongly and may require a larger integration-angle limit. A plot is not proof by itself: include numerical residuals, held-out tests, and solver tolerance checks in the technical appendix.

## Clean experiment and evidence plan

Do not delete source datasets, split manifests, code, or reference figures. Clean only generated run artifacts after confirming they are reproducible. Keep every final run under a unique name and save its config, seed, split IDs, checkpoint, logs, and figures together.

### PPO training-time snapshots

For each PPO track, use one fixed training configuration and save/evaluate the policy at these elapsed-time checkpoints:

| Elapsed training time | Evidence to capture |
|---|---|
| 1 minute | policy checkpoint, reward/episode history, same fixed visual test rays |
| 10 minutes | same artifacts and same test rays |
| 1 hour | same artifacts and same test rays |
| Longer, if useful | same protocol; do not change test cases |

Prefer one continuous run with checkpoint snapshots at 60, 600, and 3600 seconds. This makes the snapshots a learning progression rather than three unrelated experiments. If the trainer cannot save at exact wall-clock times, record the actual elapsed time and never label it as exact.

For fair visual comparison, keep the camera limits, initial rays, colors, and reference curves identical across snapshots. Show the policy path against DOP853, and include at least one ray near the photon sphere as well as weak-deflection rays. Save both the full figure and an individual high-resolution plot.

### Supervised MLP

- Record initial/untrained and validation-selected checkpoints.
- Plot training and validation loss versus epoch.
- Evaluate autoregressively on the untouched test rays.
- Include unseen closest approaches, not merely more samples from known rays.

### PINN

- Plot predicted trajectories against DOP853.
- Plot the learned differential-equation residual over the domain.
- Show training collocation points and any reference points separately.
- Run data-ablation experiments (for example, many reference points versus sparse points) while keeping the physics residual disclosed.
- Check initial conditions and behavior near the photon sphere; do not assume a low residual guarantees a physically correct global solution.

## Presentation figures

1. Schwarzschild geometry diagram: event horizon, photon sphere, and example rays.
2. DOP853 reference trajectories for weak, moderate, and strong bending.
3. Deflection angle versus closest-approach radius, with the critical region identified.
4. RL learning progression at 1 min, 10 min, and 1 hour.
5. RL versus DOP853 on identical held-out launch conditions.
6. MLP autoregressive rollout versus DOP853.
7. PINN trajectory and equation-residual plots.
8. Final comparison matrix listing what information each method was allowed to use.

## Required experiment manifest

Every result must record:
- git commit and branch;
- dataset and split-manifest hashes or immutable filenames;
- random seed;
- algorithm and full hyperparameters;
- training duration and actual timestep/epoch count;
- observation, action, reward, and termination definitions;
- what physics information and reference data the learner could access;
- exact held-out test cases;
- solver tolerances and event handling;
- output file paths.

## Claims we may and may not make

Allowed:
- "The model approximated held-out DOP853 trajectories under these test conditions."
- "The black-box agent learned a policy from the declared interaction feedback."
- "The PINN used the Schwarzschild geodesic residual as a physics constraint."

Not allowed without much stronger evidence:
- "AI independently proved Einstein's equations."
- "The agent learned without any human-provided assumptions."
- "A visually similar curve proves the learned model is physically correct."

## Current limitations to resolve before final runs

- The tracked v2 split contains only 30 rays (21 train, 4 validation, 5 test). That is a smoke-test scale, not a convincing final scientific dataset.
- The current supervised MLP guide defaults to the v2 dataset. Expand and freeze a larger, well-documented dataset before final training.
- The current PPO trainer is teacher-guided. A separate black-box RL environment and reward design are required for the equation-hidden experiment.
- A PINN implementation is a separate milestone; do not label the supervised MLP as a PINN.
- The final comparison must use exactly the same held-out initial conditions wherever the task definitions allow it.
