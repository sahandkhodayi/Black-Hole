# Black Hole Lab

A numerical-physics and machine-learning prototype studying photon trajectories and gravitational light bending around a Schwarzschild black hole.

## Start here

**[Open the RL presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md)** for the working prototype's installation, timed training, evaluation, and plotting commands. The [RL Learning Guidebook](GUIDEBOOK.md) remains the deeper theory and research-design reference.

## Current repository capabilities

- Schwarzschild photon scattering ODE and DOP853 reference solver.
- Dense reference interpolation for arbitrary angles inside the integrated interval.
- Packed trajectory dataset generation and inspection.
- Reproducible whole-trajectory train/validation/test splitting.
- Numerical diagnostics, plots, weak-field comparison, and tests.
- Initial teacher-guided PPO environment, training, evaluation, and run-comparison scripts (presentation/rl-prototype branch).

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

## RL presentation prototype

The teacher-guided PPO trajectory-tracking prototype is available in `src/rl/`. See [the presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md) for timed training runs, evaluation, and plots, and [presentation notes](docs/PRESENTATION_NOTES.md) for the physics, RL formulation, slide outline, and likely questions. This is an experimental prototype; run it locally and report only measured results.

## Research scope

The target is sequential generation of escaping photon trajectories. The reference solver supplies ground-truth curves for evaluation and (in the first proposed experiment) teacher-guided reward. PPO is a candidate first algorithm, not the research question itself. The guidebook discusses the difference between teacher-guided trajectory imitation through RL and a policy that independently decides when a ray has escaped.

The first RL environment, PPO training script, evaluation script, and experiment comparison utility are included on the `presentation/rl-prototype` branch. They are a teacher-guided tracking baseline, not a claim of autonomous physics discovery.
