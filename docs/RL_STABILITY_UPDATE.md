# PPO stability update

The PPO prototype uses a bounded positive tracking reward, conservative
policy updates, and an observation that includes the upcoming angular step.

## Important: start a fresh run

The observation schema changed from four to five values. Existing PPO
checkpoints were trained with the old observation shape and are **not
compatible** with this version. Do not resume an old checkpoint or compare
it as though it used the new environment. Start a fresh run with a new name:

```powershell
python -m pip install -e ".[dev,rl]"
python -m pytest -q
python -m rl.train --seconds 600 --seed 42 --run-name ppo_stepaware_v2
```

The training script saves:
- `initial_model.zip`: untrained starting policy.
- `final_model.zip`: last policy produced before the time limit.
- `best/best_model.zip`: policy with the best validation reward.

**Evaluate the best checkpoint first**, not automatically the final checkpoint:

```powershell
python -m rl.evaluate --model outputs/rl_runs/ppo_stepaware_v2/best/best_model.zip --split test --output outputs/rl_runs/ppo_stepaware_v2/evaluation_test
```

## Why the observation changed

The dataset samples rays at a nominal angular spacing, but the last interval
is shorter because the reference includes the exact DOP853 escape event.
Previously, the policy observed (r_0), (phi), (u), and (v), but not
the size of the next integration interval. It therefore had no direct input
that distinguished a normal sample transition from the shortened final
transition. The fifth observation is

`next_delta_phi / step_scale`

where `step_scale` is the largest positive interval in the loaded dataset.
The environment exposes this interval before every action. This is an
information correction, not a claim that PPO will become accurate by itself.

## Reward and optimizer

- Reward: `1 / (1 + normalized_squared_error)`, bounded in (0, 1].
- Learning rate: `1e-4`.
- Rollout length: `512`; batch size: `128`.
- Policy clip range: `0.15`; entropy coefficient: `0.005`.
- KL guard: `target_kl=0.03`.

These settings are stability-oriented, not a guarantee that PPO will beat
DOP853. Judge success using held-out trajectory RMSE, orbit overlays, and
physical diagnostics—not training reward alone. DOP853 remains the reference.
