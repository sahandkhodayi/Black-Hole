# Black Hole Lab — presentation quickstart

This guide is the shortest reliable route from a fresh clone to the PPO
presentation figures. It assumes the repository's tracked dataset and split
manifest are present.

## 1. What this experiment actually asks

Can a PPO policy sequentially approximate photon states from a Schwarzschild
black-hole scattering trajectory?

This is **teacher-guided sequential trajectory tracking**:
- DOP853 generates the numerical reference trajectory.
- The policy observes the initial radius, current angle, and its own current
  predicted state.
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
pull the latest commit and install the updated editable package.

## 3. Reproduce the current stable experiment

Do not reuse an old run folder: the reward and training configuration have
changed over the project's iterations.

```powershell
python -m rl.train --seconds 600 --seed 42 --run-name presentation_run
```

The command saves artifacts under
`outputs/rl_runs/presentation_run/`:
- `initial_model.zip`: policy before learning;
- `final_model.zip`: last policy before the time limit;
- `best/best_model.zip`: checkpoint with the highest validation episode
  reward observed by EvalCallback;
- `evaluation/evaluations.npz`: validation reward history;
- `config.json`: seed, split IDs, reward definition, and PPO settings.

The best checkpoint is not guaranteed to have the lowest test RMSE. Select it
using validation only, then report test metrics once for the frozen choice.

## 4. Evaluate the same held-out test rays

```powershell
python -m rl.evaluate --model outputs/rl_runs/presentation_run/initial_model.zip --split test --output outputs/rl_runs/presentation_run/evaluation_initial
python -m rl.evaluate --model outputs/rl_runs/presentation_run/best/best_model.zip --split test --output outputs/rl_runs/presentation_run/evaluation_final
```

If `best/best_model.zip` was never created because training stopped before
the first evaluation, use `final_model.zip` and state that it is the final
checkpoint rather than a validation-selected checkpoint.

Each evaluation folder contains:
- `trajectory_comparison.png`: orbit, inverse-radius, and derivative panels;
- `metrics.csv`: one RMSE row per held-out ray;
- `summary.json`: aggregate mean and median RMSE.

## 5. Make the two summary figures

```powershell
python -m rl.presentation_plots outputs/rl_runs/presentation_run
```

Figures are written to `outputs/rl_figures/`:
- `01_training_progress.png`: validation episode reward during training;
- `02_test_error_comparison.png`: initial and trained policy RMSE on the
  same test rays.

For the trained bar to appear, the trained evaluation must be saved in
`evaluation_final` (as in the command above).

## 6. How to read the results

- Higher validation reward is better, but reward is the sum of one-step
  rewards over an episode. Rays have different sample counts, so compare
  reward curves cautiously and use test RMSE as the main accuracy measure.
- Lower RMSE is better.
- Solid curves are DOP853; dashed curves are PPO.
- PPO is rolled out autoregressively: each predicted state is fed back into
  the next observation. Small one-step errors can therefore accumulate.
- DOP853 is the reference solver and is expected to be more accurate on the
  trajectories it integrated.

## 7. Reproducibility checklist

Record the Git commit, dataset and split manifest, seed, actual timesteps,
wall-clock time, checkpoint name, and test metrics. Do not tune using the test
set. Do not describe one seed or five test rays as broad proof of
generalization.

For the interpretation, slide plan, limitations, and likely questions, see
[PRESENTATION_NOTES.md](PRESENTATION_NOTES.md). For the current observed
stable_v1 result, see [PRESENTATION_RESULTS.md](PRESENTATION_RESULTS.md).


## 8. Watch PPO learn live

For an interactive dashboard with episode reward, PPO losses, validation RMSE, and a live DOP853-versus-PPO ray, see [LIVE_TRAINING_DASHBOARD.md](LIVE_TRAINING_DASHBOARD.md). Start a long run with:

```powershell
python -m rl.train --seconds 3600 --seed 42 --run-name live_research_01 --live
```

Click **Stop & save** or press **Q** / **Esc** in the dashboard to stop early. Ctrl+C in the terminal also saves the current policy. The latest dashboard snapshot and CSV history are saved in the run folder.
