# Black Hole Lab

A numerical-physics and machine-learning prototype studying photon trajectories and gravitational light bending around a Schwarzschild black hole.

## Start here

**[Open the RL presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md)** for installation, timed training, and evaluation. For simple PNG figures without TensorBoard, see [Simple presentation plots](docs/SIMPLE_PRESENTATION_PLOTS.md). If PPO training is unstable, follow the [PPO stability update](docs/RL_STABILITY_UPDATE.md) and evaluate the validation-selected checkpoint. The [RL Learning Guidebook](GUIDEBOOK.md) remains the deeper theory and research-design reference.

## Current repository capabilities

- Schwarzschild photon scattering ODE and DOP853 reference solver.
- Dense reference interpolation for arbitrary angles inside the integrated interval.
- Packed trajectory dataset generation and inspection.
- Reproducible whole-trajectory train/validation/test splitting.
- Numerical diagnostics, plots, weak-field comparison, and tests.
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

## Previous guide

The former long README has been preserved at [docs/PROJECT_REFERENCE.md](docs/PROJECT_REFERENCE.md). Treat it as historical study material; use GUIDEBOOK.md for current instructions.

## RL presentation prototype

The teacher-guided PPO trajectory-tracking prototype is available in `src/rl/`. See [the presentation quickstart](docs/RL_PRESENTATION_QUICKSTART.md) for timed training runs, evaluation, and plots, and [presentation notes](docs/PRESENTATION_NOTES.md) for the slide script, physics, RL formulation, and likely questions. The current measured experiment is documented in [presentation results](docs/PRESENTATION_RESULTS.md). This is an experimental prototype; run it locally and report only measured results.

## Research scope

The target is sequential generation of escaping photon trajectories. The reference solver supplies ground-truth curves for evaluation and (in the first proposed experiment) teacher-guided reward. PPO is a candidate first algorithm, not the research question itself. The guidebook discusses the difference between teacher-guided trajectory imitation through RL and a policy that independently decides when a ray has escaped.

The RL environment, PPO training script, evaluation script, and experiment comparison utility are included on the default branch. They are a teacher-guided tracking baseline, not a claim of autonomous physics discovery.


## Interactive classroom website

Explore a DOP853 photon orbit by changing its closest approach, animate the photon, and see the synchronized inverse-radius and derivative plots. The page shows how the second-order orbit equation becomes a first-order system, then lets you compare the PPO prediction with the numerical reference. PPO training snapshots refresh once per minute. See [the visual classroom guide](docs/ANALYTICS_DASHBOARD.md). The optional Matplotlib research console remains available from the CLI: [Live training dashboard guide](docs/LIVE_TRAINING_DASHBOARD.md).

## Parallel MLP + PPO classroom experiment

The presentation pipeline trains the selected supervised dynamics MLP (\`dynamics_architecture_sweep_relu_w128_d3\`) alongside the existing teacher-guided PPO trajectory model. It evaluates both autoregressively on the same held-out photon rays and writes per-ray metrics, comparison plots, and a plain-language analysis. The classroom demo loads saved checkpoints and does not train live.

Install and run from PowerShell:

~~~powershell
python -m pip install -e ".[dev,rl,ml]"
python -m presentation.run_parallel --run-name class_presentation --rl-seconds 600 --mlp-epochs 300 --seed 42
python -m presentation.classroom_demo --show
~~~

See [presentation pipeline instructions](PRESENTATION_PIPELINE_README.md) and [the detailed workflow](docs/PRESENTATION_CODE_WORKFLOW.md). The current PPO environment predicts the next \`(u, v)\` state with teacher-guided reward; it does not control the numerical solver's step size. The MLP is supervised rather than a PINN.



## Easy-to-read experiment results

Open the local results page after training or evaluating a model:

~~~powershell
uv pip install -e ".[rl]"
python -m bh_analytics.dashboard --root outputs
~~~

It opens the interactive classroom website. Trace photon paths from the differential equation, animate the result, watch the `u(φ)` and `v(φ)` state plots move together, and compare PPO's live validation path with DOP853. Start/stop PPO from the browser and inspect saved MLP, PPO, and paired comparison results below. No CSV files need to be prepared by hand. See [the visual classroom guide](docs/ANALYTICS_DASHBOARD.md).
