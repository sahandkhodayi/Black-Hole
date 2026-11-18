# Black Hole Lab — presentation quickstart

This guide is the shortest reliable route from a fresh clone to the PPO
presentation figures. It assumes the repository's tracked dataset and split
manifest are present.

## 1. What this experiment actually asks

Can a PPO policy sequentially approximate photon states from a Schwarzschild
black-hole scattering trajectory?

This is **teacher-guided sequential trajectory tracking**:
- DOP853 generates the numerical reference trajectory.
- The policy observes the initial radius, current angle, its own current
  predicted state, and the upcoming angular step size.
- The policy proposes the next inverse-radius state and its derivative.
- The environment compares that proposal with the next DOP853 state to
  calculate reward.
- The next policy observation contains the policy's own prediction, not the
  reference future state.

It is not autonomous discovery of general relativity, an independent solver
for the geodesic equation, or a learned capture/escape classifier.

## 2. Install and verify

PowerShell, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rl]"
python -m pytest -q
```

If you already have a working virtual environment, do not recreate it; just
install the updated editable package. Existing PPO checkpoints with the
previous four-value observation are incompatible with the current five-value
observation; start a fresh run.

## 3. Reproduce a fresh experiment

```powershell
python -m rl.train --seconds 600 --seed 42 --run-name presentation_stepaware_v2
```

The command saves artifacts under
`outputs/rl_runs/presentation_stepaware_v2/`:
- `initial_model.zip`: policy before learning;
- `final_model.zip`: last policy before the time limit;
- `best/best_model.zip`: checkpoint with the highest validation episode
  reward observed by EvalCallback;
- `evaluation/evaluations.npz`: validation reward history;
- `config.json`: seed, split IDs, observation schema, reward, and PPO settings.

## 4. Evaluate held-out test rays

```powershell
python -m rl.evaluate --model outputs/rl_runs/presentation_stepaware_v2/initial_model.zip --split test --output outputs/rl_runs/presentation_stepaware_v2/evaluation_initial
python -m rl.evaluate --model outputs/rl_runs/presentation_stepaware_v2/best/best_model.zip --split test --output outputs/rl_runs/presentation_stepaware_v2/evaluation_test
```

If `best/best_model.zip` was never created because training stopped before
the first evaluation, use `final_model.zip` and state that it is the final
checkpoint rather than a validation-selected checkpoint.

Each evaluation folder contains:
- `trajectory_comparison.png`: orbit, inverse-radius, and derivative panels;
- `metrics.csv`: one RMSE row per held-out ray;
- `summary.json`: aggregate mean and median RMSE.

## 5. Make the summary figures

```powershell
python -m rl.presentation_plots outputs/rl_runs/presentation_stepaware_v2
```

Figures are written to `outputs/rl_figures/`. Compare initial and trained
policies on the same test rays; do not tune hyperparameters against the test
split.

## 6. Read the results correctly

- Higher validation reward is better, but episode returns depend on ray
  sample counts. Use test RMSE as the main accuracy measure.
- Lower RMSE is better.
- Solid curves are DOP853; dashed curves are PPO.
- PPO is rolled out autoregressively: each predicted state is fed back into
  the next observation. Small one-step errors can therefore accumulate.
- DOP853 is the numerical reference and is expected to be more accurate on
  the trajectories it integrated.

For the interpretation, slide plan, physics, and likely questions, see
`PRESENTATION_NOTES.md`. For the live Matplotlib dashboard, see
`LIVE_TRAINING_DASHBOARD.md`.
