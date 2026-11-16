# Live PPO research dashboard

The live dashboard is an optional Matplotlib window. It does not require
TensorBoard.

## Start a long experiment

From the repository root, in PowerShell:

```powershell
python -m rl.train --seconds 3600 --seed 42 --run-name live_research_01 --live
```

The time limit is one hour, but you can stop earlier from the dashboard.

## Live research console

The main panel is an animated top-down view of a Schwarzschild photon ray.
It draws the event horizon at r/r_s = 1, the photon sphere at r/r_s = 1.5,
and the actual validation trajectory in Cartesian coordinates. The DOP853
reference is blue and the PPO rollout is dashed pink. The ray advances while
training collects environment steps; a new real validation rollout replaces
the displayed ray at each validation interval. The black-hole rings are a
visual guide, not a simulated accretion disk or a time-evolving spacetime.

Smaller panels show:

- **Episode reward:** completed training-episode returns and a five-episode
  moving average.
- **Optimization losses:** policy-gradient loss and value-function loss
  reported by Stable-Baselines3 after PPO updates.
- **Action distribution:** recent normalized PPO actions for u and v.
- **Validation RMSE:** autoregressive validation errors for u and v.
- **u(phi):** reference and predicted inverse-radius histories.
- **v(phi):** reference and predicted radial-derivative histories.

Validation cases are never used for gradient updates. All orbit and metric
panels are driven by actual dataset validation rays and policy predictions.

Validation is evaluated every 1,024 training steps by default. Change this
with `--live-eval-freq 2048` (larger values reduce evaluation overhead).

## Stop safely at any time

- Click **Pause** to freeze training at the current environment step; click
  **Resume** to continue. The space bar also toggles pause/resume while the
  dashboard window is focused.
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

- `live_dashboard.png`: final snapshot of the dashboard, written once when the run ends.
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
