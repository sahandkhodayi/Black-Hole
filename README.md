# Black Hole Lab

A numerical-physics and machine-learning prototype studying photon trajectories and gravitational light bending around a Schwarzschild black hole.

## Start here

**[Presentation runbook](docs/PRESENTATION_RUNBOOK.md)** restores a clean run after deleting generated outputs and gives the exact PowerShell commands for data generation, training, and visual evaluation.

**[Presentation research protocol](docs/PRESENTATION_RESEARCH_PROTOCOL.md)** defines the equation-hidden RL, supervised MLP, PINN, and scientific evidence tracks, including reproducible training-time snapshots.

**[Start with supervised photon-dynamics learning](docs/SUPERVISED_DYNAMICS_GUIDE.md)** for the direct neural-network baseline and visual test-ray evaluation. The [PPO presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md) covers the separate reinforcement-learning experiment. For simple PNG figures without TensorBoard, see [Simple presentation plots](docs/SIMPLE_PRESENTATION_PLOTS.md). If PPO training is unstable, follow the [PPO stability update](docs/RL_STABILITY_UPDATE.md) and evaluate the validation-selected checkpoint. The [RL Learning Guidebook](GUIDEBOOK.md) remains the deeper theory and research-design reference.

## Current repository capabilities

- Schwarzschild photon scattering ODE and DOP853 reference solver.
- Dense reference interpolation for arbitrary angles inside the integrated interval.
- Packed trajectory dataset generation and inspection.
- Reproducible whole-trajectory train/validation/test splitting.
- Numerical diagnostics, plots, weak-field comparison, and tests.
- Supervised neural photon-dynamics model with whole-ray train/validation/test splits and autoregressive evaluation.
- Teacher-guided PPO environment, training, evaluation, run-comparison, and presentation-plot scripts.

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


### Compare neural architectures automatically

To compare ReLU, tanh, GELU, SiLU, and ELU networks across two widths and
two depths, run:

~~~powershell
python -m rl.sweep_dynamics --dataset data/photon_trajectories_v2.npz --splits data/split_manifest_v2.json --epochs 200 --patience 25 --seed 42 --run-name dynamics_architecture_sweep
~~~

The sweep trains 20 candidates and ranks them by **autoregressive validation
error** (each model consumes its own previous predictions). It does not use
the test split to choose a winner. Results are written to
`outputs/rl_runs/dynamics_architecture_sweep/leaderboard.csv` and
`leaderboard.json`; the winning checkpoint is identified in
`BEST_MODEL.txt`. Evaluate that winner on the test split only after model
selection is complete.
\n## Previous guide

The former long README has been preserved at [docs/PROJECT_REFERENCE.md](docs/PROJECT_REFERENCE.md). Treat it as historical study material; use GUIDEBOOK.md for current instructions.

## RL presentation prototype

The supervised next-state MLP is the recommended first learned dynamics baseline; see [its training and evaluation guide](docs/SUPERVISED_DYNAMICS_GUIDE.md). The teacher-guided PPO trajectory-tracking prototype remains available in `src/rl/` as a separate RL experiment. See [the presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md) for timed training runs, evaluation, and plots, and [presentation notes](docs/PRESENTATION_NOTES.md) for the slide script, physics, RL formulation, and likely questions. The current measured experiment is documented in [presentation results](docs/PRESENTATION_RESULTS.md). This is an experimental prototype; run it locally and report only measured results.

## Research scope

The target is sequential generation of escaping photon trajectories. The reference solver supplies ground-truth curves for evaluation and (in the first proposed experiment) teacher-guided reward. PPO is a candidate first algorithm, not the research question itself. The guidebook discusses the difference between teacher-guided trajectory imitation through RL and a policy that independently decides when a ray has escaped.

The RL environment, PPO training script, evaluation script, and experiment comparison utility are included on the default branch. They are a teacher-guided tracking baseline, not a claim of autonomous physics discovery.


## Live RL research dashboard

Watch PPO training in real time, inspect reward/loss/validation-error curves, compare a live validation ray against DOP853, and stop safely at any time: [Live training dashboard guide](docs/LIVE_TRAINING_DASHBOARD.md).
