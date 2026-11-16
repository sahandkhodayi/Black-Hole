# Live PPO research dashboard

The live dashboard is an optional Matplotlib window. It does not require
TensorBoard.

## Start a long experiment

From the repository root, in PowerShell:

```powershell
python -m rl.train --seconds 3600 --seed 42 --run-name live_research_01 --live
```

The time limit is one hour, but you can stop earlier from the dashboard.

## Dashboard panels

- **Episode reward:** completed training-episode returns and a five-episode
  moving average.
- **Optimization losses:** policy-gradient loss and value-function loss
  reported by Stable-Baselines3 after PPO updates.
- **Validation RMSE:** autoregressive validation errors for inverse radius
  (u=1/r) and derivative (v=du/dphi). Validation cases are never used
  for gradient updates.
- **Live validation ray:** one fixed validation ray, comparing the DOP853
  reference with the current PPO rollout in (u(phi)).

Validation is evaluated every 1,024 training steps by default. Change this
with `--live-eval-freq 2048` (larger values reduce evaluation overhead).

## Stop safely at any time

- Click **Stop & save** in the dashboard, or press **Q** / **Esc** while the
  dashboard window is focused.
- Alternatively press **Ctrl+C** in the terminal.
- The stop callback exits the current training loop as soon as it receives
  the request. The training script writes `final_model.zip`, updates
  `config.json`, and closes the environments.
- A periodic checkpoint is also written every 1,000 environment steps under
  `checkpoints/`. The best validation-reward model remains under `best/`.

A wall-clock limit is still active. For example, `--seconds 3600` provides
a one-hour upper bound; the dashboard stop button can finish the run earlier.

## Outputs

All outputs are saved in `outputs/rl_runs/<run-name>/`:

- `live_dashboard.png`: latest snapshot of the dashboard.
- `live_history.csv`: timestamped reward, loss, and validation metric
  samples for later plotting or spreadsheet analysis.
- `final_model.zip`: model at the moment training stopped.
- `checkpoints/`: periodic recovery checkpoints.
- `best/best_model.zip`: checkpoint selected by validation reward.
- `config.json`: experiment settings, elapsed time, timesteps, and whether
  the run was stopped early.

The dashboard is a monitoring tool, not a source of scientific truth.
Use the saved test split only for final held-out evaluation. Do not tune
hyperparameters against test results.

## Headless mode

Without `--live`, training continues without opening a GUI. It still saves
the final model and periodic checkpoints. Ctrl+C is handled as a request to
save the current policy.
