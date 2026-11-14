# PPO stability update

The PPO prototype has been updated to use a bounded positive tracking reward and
more conservative policy updates.

## Important: start a fresh run

The reward definition changed. Do not resume or compare an old checkpoint as if
it had been trained with the new reward. Pull the latest code and start a new
run with a new name:

```powershell
git pull
python -m pip install -e ".[dev,rl]"
python -m pytest -q
python -m rl.train --seconds 600 --seed 42 --run-name stable_v1
```

The training script saves:
- `initial_model.zip`: untrained starting policy.
- `final_model.zip`: last policy produced before the time limit.
- `best/best_model.zip`: policy with the best validation reward during training.

**Evaluate the best checkpoint first**, not automatically the final checkpoint:

```powershell
python -m rl.evaluate --model outputs/rl_runs/stable_v1/best/best_model.zip --split test --output outputs/rl_runs/stable_v1/evaluation_best
```

Then compare initial and best on the same held-out test rays:

```powershell
python -m rl.evaluate --model outputs/rl_runs/stable_v1/initial_model.zip --split test --output outputs/rl_runs/stable_v1/evaluation_initial
python -m rl.presentation_plots outputs/rl_runs/stable_v1
```

For the plotting script to include both bars, copy the best evaluation summary
into the expected trained-policy location, or evaluate the best checkpoint
directly into that location:

```powershell
python -m rl.evaluate --model outputs/rl_runs/stable_v1/best/best_model.zip --split test --output outputs/rl_runs/stable_v1/evaluation_final
```

## What changed

- Reward is now `1 / (1 + normalized_squared_error)`, so it stays in (0, 1]
  and gives a higher reward to smaller prediction errors.
- PPO learning rate reduced to `1e-4`.
- Rollout length increased to `512`; batch size `128`.
- Smaller policy clip range `0.15`, modest entropy coefficient `0.005`,
  and `target_kl=0.03` to limit destructive policy updates.
- Checkpoint and validation frequency set to every 1,000 steps.

These are stability-oriented changes, not a guarantee that PPO will beat
DOP853. DOP853 remains the numerical reference. Judge success using the
held-out test RMSE and trajectory overlays, not training reward alone.
