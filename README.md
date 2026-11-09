# Black Hole Lab

A numerical-physics project studying photon trajectories and gravitational light bending around a Schwarzschild black hole, with a later research goal of learning trajectory generation through reinforcement learning.

## Start here

**[Open the RL Learning Guidebook](GUIDEBOOK.md)**

The guidebook is the active project manual. It teaches the work in small milestones, asks you to implement the important pieces yourself, gives progressive hints, and defines what to inspect before moving on. It deliberately does not hand you a complete RL agent.

## Current repository capabilities

- Schwarzschild photon scattering ODE and DOP853 reference solver.
- Dense reference interpolation for arbitrary angles inside the integrated interval.
- Packed trajectory dataset generation and inspection.
- Reproducible whole-trajectory train/validation/test splitting.
- Numerical diagnostics, plots, weak-field comparison, and tests.

## Quick start

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
~~~

Generate and inspect a fresh dataset without overwriting the tracked example:

~~~powershell
python src/generate_data.py --count 30 --r-min 2.2 --r-max 12 --output data/photon_trajectories_v2.npz --manifest data/trajectory_manifest_v2.json
python src/inspect_data.py --data data/photon_trajectories_v2.npz --manifest data/trajectory_manifest_v2.json --show
python src/make_splits.py --data data/photon_trajectories_v2.npz --output data/split_manifest_v2.json --seed 42
~~~

## Previous guide

The former long README has been preserved at [docs/PROJECT_REFERENCE.md](docs/PROJECT_REFERENCE.md). Treat it as historical study material; use GUIDEBOOK.md for current instructions.

## Research scope

The target is sequential generation of escaping photon trajectories. The reference solver supplies ground-truth curves for evaluation and (in the first proposed experiment) teacher-guided reward. PPO is a candidate first algorithm, not the research question itself. The guidebook discusses the difference between teacher-guided trajectory imitation through RL and a policy that independently decides when a ray has escaped.

No RL environment, reward implementation, policy, or ML model is currently included in the repository. You will build those as learning milestones.
