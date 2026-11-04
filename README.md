# Black Hole Lab

A computational project exploring Schwarzschild photon trajectories, numerical integration, adaptive step-size control, and reinforcement learning.

## Start here

**[Integrated implementation guide: RK4 to PPO](GUIDE_FROM_RK4_TO_PPO.md)**

This is the active, step-by-step guide. It starts from the repository's current RK4 implementation and proceeds through:

1. RK4 unit tests and convergence validation
2. Tests for the Schwarzschild geodesic model
3. A dense DOP853 reference and fixed-step RK4 runner
4. Escape-event localization and numerical convergence experiments
5. An adaptive RK4 step-doubling baseline
6. A Gymnasium environment for step-size control
7. Non-learning baselines, PPO training, and held-out evaluation

The guide keeps the explanation, implementation, tests, and acceptance criteria together in each milestone. Do not jump to PPO before the numerical baselines are validated.

## Additional documentation

- [Unified research and implementation textbook](guide.md) — broader project context and background material. The active implementation order is linked at the top.
- [Earlier research and implementation guide](DOCS/BLACK_HOLE_LAB_GUIDE.md) — retained for historical/background reference; its opening notice points to the current guide.

## Current research question

> Can a reinforcement-learning policy learn to choose numerical integration step sizes for Schwarzschild photon geodesics, balancing computational cost against numerical accuracy?

The agent controls the numerical integrator's step size. It does not control the photon or change the physical geodesic.

## Development

The project uses Python, NumPy, SciPy, and Matplotlib. Gymnasium and Stable-Baselines3 are introduced later in the RL milestone. See the integrated guide for the current setup instructions and exact milestone order.
