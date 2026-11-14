# Simple presentation plots (no TensorBoard needed)

You can make the figures for your presentation with one command. TensorBoard is
optional; this script reads the ordinary output files created by training and
evaluation and writes PNG images.

## 1. Train and evaluate

From the repository root (PowerShell):

```powershell
python -m pip install -e ".[dev,rl]"
python -m rl.train --seconds 600 --seed 42 --run-name demo
python -m rl.evaluate --model outputs/rl_runs/demo/initial_model.zip --split test --output outputs/rl_runs/demo/evaluation_initial
python -m rl.evaluate --model outputs/rl_runs/demo/final_model.zip --split test --output outputs/rl_runs/demo/evaluation_final
```

If you already trained, skip the training command and use your existing run
folder. Evaluation creates a trajectory figure, a per-ray CSV, and a summary
JSON for each model.

## 2. Make the simple plots

```powershell
python -m rl.presentation_plots outputs/rl_runs/demo
```

The images are saved in `outputs/rl_figures/`:

- `01_training_progress.png`: validation reward as training timesteps increase.
  The shaded region shows variation between validation episodes.
- `02_test_error_comparison.png`: initial vs trained policy RMSE on the same
  held-out test rays. Lower is better.

You can compare several training durations in one command:

```powershell
python -m rl.presentation_plots outputs/rl_runs/1min outputs/rl_runs/10min outputs/rl_runs/1hour
```

## 3. Understand the plots

- **Reward:** higher is better. A rising curve suggests the policy is learning
  to make predictions closer to the DOP853 reference. A noisy or flat curve is
  also a valid result; do not claim improvement if the data does not show it.
- **RMSE:** lower is better. It measures the average size of the prediction
  error on test rays that were held out from training.
- **Trajectory comparison:** open
  `evaluation_initial/trajectory_comparison.png` and
  `evaluation_final/trajectory_comparison.png`. Solid curves are DOP853;
  dashed curves are PPO.

The reference solver is expected to be more accurate than this learned policy.
This experiment evaluates teacher-guided trajectory tracking, not independent
discovery of the black-hole equations.
