# Supervised photon-dynamics learning

## Why this exists

The first learning experiment asked PPO to predict the next state of a known
DOP853 trajectory. That formulation is teacher-guided, but it makes a
deterministic regression problem unnecessarily difficult: PPO must discover
a smooth input-to-next-state mapping through policy-gradient updates and
delayed episode returns.

This project now includes a direct supervised baseline. It learns the
one-step map

`(r_0, phi_i, u_i, v_i, \Delta\phi_i) \rightarrow (u_{i+1}, v_{i+1})`

from DOP853 transitions. Whole trajectories—not individual rows—are split
between training, validation, and test, so neighbouring points from one ray
cannot leak across splits.

PPO remains available as a separate reinforcement-learning experiment. Do
not describe the supervised MLP as PPO or claim either model discovers
Einstein's equations. DOP853 remains the numerical reference.

## Install

PowerShell, from the repository root:

```powershell
python -m pip install -e ".[dev,rl]"
python -m pytest -q
```

The `rl` extra installs Stable-Baselines3 and its PyTorch dependency. A
CUDA-enabled PyTorch build is only useful if the installed PyTorch can see a
compatible NVIDIA driver; CPU is the safe default.

## Train

```powershell
python -m rl.train_dynamics --epochs 300 --patience 30 --seed 42 --run-name dynamics_supervised_v1
```

The trainer uses the existing `data/split_manifest_v2.json`:
- training rays supply gradient updates;
- validation rays select the best checkpoint and trigger early stopping;
- test rays are not used during training or model selection.

Artifacts are written to
`outputs/rl_runs/dynamics_supervised_v1/`:
- `best/best_model.pt`: validation-selected weights and preprocessing scale;
- `training_history.csv`: training and validation loss by epoch;
- `config.json`: split IDs, seed, feature schema, and best epoch.

The model uses a 3-hidden-layer MLP (128 GELU units per layer), AdamW,
gradient clipping, learning-rate reduction on validation plateaus, and
early stopping.

## Evaluate visually

```powershell
python -m rl.evaluate_dynamics --model outputs/rl_runs/dynamics_supervised_v1/best/best_model.pt --split test --output outputs/rl_runs/dynamics_supervised_v1/evaluation_test
Start-Process "outputs\rl_runs\dynamics_supervised_v1\evaluation_test\trajectory_comparison.png"
```

Evaluation is **autoregressive**: after the first state, each MLP prediction
becomes the input to the next step. This is deliberately harder than
teacher-forced one-step validation and exposes accumulated drift. The figure
overlays DOP853 (solid) and MLP (dashed) for the same held-out rays.

## What counts as success

1. The held-out MLP orbit should visually follow the DOP853 path across
   several different closest approaches.
2. The `u(\phi)` and `v(\phi)` curves should follow the reference shape,
   not merely start at the same point.
3. Autoregressive test performance must be reported separately from
   teacher-forced validation loss.
4. Compare against the initial/untrained model and a simple baseline before
   making claims about improvement.

A small one-step validation loss is not sufficient evidence: rollout error
can accumulate. If the curves still drift, inspect which radii and which
parts of each trajectory fail before increasing model size.
