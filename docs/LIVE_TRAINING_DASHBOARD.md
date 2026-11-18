# Live PPO research dashboard

The live dashboard is an optional Matplotlib window. It does not require
TensorBoard.

## Start a long experiment

From the repository root, in PowerShell:

```powershell
python -m rl.train --seconds 3600 --seed 42 --run-name live_research_01 --live --live-interval-seconds 60
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

The dashboard performs an initial validation report at the first PPO rollout
boundary, then re-evaluates validation rays and redraws the graphs approximately
once per wall-clock interval. The default is 60 seconds. Change it with
`--live-interval-seconds 30` or `--live-interval-seconds 120`. Updates happen
at PPO rollout boundaries, so a report can appear slightly after the exact
wall-clock mark.

## Stop safely at any time

- Click **Pause** to freeze training at the current environment step; click
  **Resume** to continue. The space bar also toggles pause/resume while the
  dashboard window is focused.
- Click **Quit & save** in the dashboard, or press **Q** / **Esc** while the
  dashboard window is focused. Closing the dashboard window also requests a safe stop.
- Alternatively press **Ctrl+C** in the terminal.
- The training script catches the interrupt, exits the PPO loop, writes
  `final_model.zip`, updates `config.json`, closes the environments, and
  saves the final dashboard snapshot as `live_dashboard.png`. The CSV history
  remains available as `live_history.csv`. A checkpoint is also written every
  1,000 environment steps. The best validation-reward model remains under
  `best/`.

A wall-clock limit is still active. For example, `--seconds 3600` provides
a one-hour upper bound; the dashboard stop button can finish the run earlier.

## Outputs

All outputs are saved in `outputs/rl_runs/<run-name>/`:

- `live_dashboard.png`: final snapshot of the dashboard, written once when the run ends.
- `live_history.csv`: timestamped reward, loss, and validation metric
  samples written on the configured reporting interval (60 seconds by default),
  plus the final model/dashboard outputs. This avoids per-rollout disk writes.
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
