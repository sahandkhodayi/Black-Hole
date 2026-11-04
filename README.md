# Black Hole Lab

A computational project studying Schwarzschild photon trajectories and learning to generate gravitationally bent light paths with reinforcement learning.

## Start here

**[RL Photon-Trajectory Generation Guide](GUIDE_RL_TRAJECTORY_LEARNING.md)**

This is the active research and implementation guide. The goal is to train an RL agent to generate the photon's path sequentially: at each fixed angular increment, the agent predicts the next state of the ray. It is not an RL step-size optimizer.

### Research milestones

1. Protect the existing geodesic physics with tests.
2. Build a dense DOP853 reference exposing both u and v.
3. Generate a multi-trajectory dataset split by closest-approach parameter.
4. Visualize weak, moderate, and strong gravitational bending.
5. Build an RL environment where the agent predicts the next trajectory state.
6. Train and compare RL policies against numerical and supervised baselines.
7. Evaluate trajectory accuracy and generalization on unseen initial conditions.

## Other documentation

- [Previous RK4-to-PPO guide](GUIDE_FROM_RK4_TO_PPO.md) — retained for history; its opening notice explains why it is no longer the active plan.
- [Unified research textbook](guide.md) — background material; see its opening notice for the current objective.
- [Earlier research guide](DOCS/BLACK_HOLE_LAB_GUIDE.md) — retained as historical/background reference.

## Research question

> Can a reinforcement-learning agent learn to generate photon trajectories near a Schwarzschild black hole, generalize to unseen initial conditions, and reproduce gravitational light bending?

The existing geodesic ODE and DOP853 solver provide the physics reference. The RL agent's action is its proposed next trajectory state, not a numerical integration step size.

See the active guide for the mathematical formulation, reward design, dataset generation, environment, baselines, and evaluation plan.
