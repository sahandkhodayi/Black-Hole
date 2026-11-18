# Standalone Python live dashboard

The standalone dashboard uses **PySide6 + PyQtGraph** for the desktop UI and
keeps the existing Stable-Baselines3 PPO engine and Schwarzschild dataset.
The GUI and trainer are separate processes:

- The GUI owns the Qt event loop and persistent plot objects.
- The trainer writes small JSON Lines telemetry records.
- Pause, resume, and stop commands travel through a small control file.
- The trainer retains its existing checkpoint, best-model, and final-model saves.

No Rust build is required.

## Windows PowerShell setup

Run these commands from the repository root:

```powershell
..venv\Scripts\Activate.ps1
python -m pip install -e ".[rl,gui]"
```

If this is a fresh environment, create and activate it first:

```powershell
py -3.11 -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[rl,gui]"
```

## Start the dashboard

```powershell
python -m rl.qt_dashboard --seconds 600
```

Optional arguments:

```powershell
python -m rl.qt_dashboard --seconds 300 --run-name qt_trial_01 --refresh-ms 200
```

The dashboard starts the trainer automatically. Do not start a second
`rl.train` command in another terminal for the same run.

## Controls and panels

- **Pause / Resume** pauses PPO updates through the trainer's control channel.
- **Stop & save** asks the trainer to stop cleanly; the training script writes
  `final_model.zip` and updates `config.json`.
- Closing the window also requests a clean stop and waits briefly for saving.
- Episode reward and its five-episode moving average update as episodes finish.
- Policy/value losses update after PPO rollouts.
- Validation RMSE and the first held-out validation ray update on the
  wall-clock report interval (60 seconds by default).

The five plots are created once. Incoming data updates their existing curves
with PyQtGraph's `setData()`; the dashboard does not clear and rebuild plots
on each refresh.

## Output files

Training artifacts are written under `outputs/rl_runs/`:

- `initial_model.zip`
- `final_model.zip`
- `best/best_model.zip` when the evaluation callback selects a best model
- `checkpoints/`
- `tensorboard/`
- `config.json`
- `dashboard.jsonl` and `dashboard.control` for GUI telemetry/control

Use a distinct `--run-name` for each experiment to avoid mixing results.

## Headless training remains available

The existing command still works without installing the GUI extras:

```powershell
python -m rl.train --seconds 600
```

The legacy Matplotlib dashboard remains available separately:

```powershell
python -m rl.train --live --seconds 600
```

Do not combine `--live` with the telemetry/control arguments; the Qt
dashboard launches the trainer and supplies those arguments itself.

## Important interpretation note

This dashboard is a monitoring interface, not evidence that the policy has
learned the physics. Rising reward alone is insufficient. Compare validation
RMSE and predicted curves against DOP853, and use held-out test rays with the
project's evaluation script before making accuracy claims.

The GUI process stays responsive while the trainer performs validation. The
trainer itself still performs its periodic validation in-process, so PPO
updates pause briefly during that calculation; moving validation to a third
worker process can be considered if profiling shows it is a bottleneck.
