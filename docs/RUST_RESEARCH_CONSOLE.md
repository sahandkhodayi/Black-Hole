# Native Rust research console

Black Hole Lab now has an optional native desktop frontend written in Rust.
Python remains the scientific and machine-learning backend.

## Architecture and compatibility

The Rust application does not reimplement or modify the physics or PPO code.
It launches the existing `python -m rl.train` process with `--console` and
reads compact JSON Lines telemetry from that run's output directory.

The existing Python components remain authoritative:

- Schwarzschild photon geodesic equation and DOP853 reference solver.
- Packed photon trajectory dataset and train/validation split manifest.
- Gymnasium `PhotonTrajectoryEnv`, including its observation/action mapping.
- Stable-Baselines3 PPO policy, network, optimizer settings, and reward.
- Existing checkpoints, validation reward selection, and evaluation scripts.

The bridge adds one callback that samples episode reward, PPO losses, and
autoregressive validation rollouts. It does not participate in gradients or
change environment transitions. The Rust frontend sends pause/resume/stop
commands through an atomic control-file replacement.

## Requirements

- Python 3.11 or newer with the project RL dependencies installed.
- Rust stable toolchain with Cargo.
- The project dataset and split manifest in the repository's normal locations.

Install the Python package from the repository root:

```powershell
python -m pip install -e ".[dev,rl]"
```

Install Rust from the official Rust toolchain installer if Cargo is not
available. The first Cargo build downloads and compiles the GUI dependencies.

## Run on Windows

From the repository root, activate the project's virtual environment and run:

```powershell
.venv\Scripts\Activate.ps1
cargo run --release --manifest-path tools/research-console/Cargo.toml
```

The application starts with the current directory as its project root. Check
the **Project root** field if you launched it from another directory. Check
**Python executable** and point it to the virtual-environment interpreter if
needed, for example:

```
C:\path\to\Black-Hole\.venv\Scripts\python.exe
```

You can also set `BLACKHOLE_PYTHON` before launching the Rust application.

On Linux/WSL, use the venv interpreter at `.venv/bin/python`. A desktop
display server is required for the native window; on WSL use WSLg or run the
application on native Windows.

## Use the console

1. Set the project root to the Black Hole Lab repository.
2. Set the Python executable to the interpreter where `.[dev,rl]` is installed.
3. Choose a run name, duration, seed, and validation frequency.
4. Select **Start PPO training**.
5. Use **Pause**, **Resume**, or **Stop & save** in the toolbar.

The console displays the DOP853 reference and PPO autoregressive prediction
for the first validation case, plus episode reward, policy/value losses, and
validation RMSE for u and v. Validation is evaluated at the configured
interval; it is not recomputed on every GUI frame.

Training stdout and stderr are captured in
`outputs/rl_runs/<run-name>/training.log`. Telemetry is appended to
`live_events.jsonl`. The control file is
`console_control.json`. Normal model, checkpoint, evaluation, and config
artifacts continue to be produced by the Python training script.

## CLI fallback

The existing Matplotlib dashboard remains available:

```powershell
python -m rl.train --seconds 120 --seed 42 --run-name matplotlib_demo --live
```

Headless training is unchanged:

```powershell
python -m rl.train --seconds 120 --seed 42 --run-name headless_demo
```

Do not pass `--live` and `--console` together. The native Rust application
selects `--console` automatically.

## Performance design

- Python emits one compact telemetry record per PPO rollout, plus a validation
  payload only at the configured evaluation interval.
- Validation curves are downsampled before being sent to the GUI.
- Rust reads only newly appended JSONL bytes, rather than repeatedly loading
  the complete history file.
- The Rust window renders independently from the Python training process.
- No Matplotlib figures are created in native-console mode.
- Training logs, checkpoints, and evaluation artifacts are written by the
  existing Python backend.

The frontend is a monitoring/control surface. It is not a separate solver and
does not change the meaning of the experiment's scientific results.
