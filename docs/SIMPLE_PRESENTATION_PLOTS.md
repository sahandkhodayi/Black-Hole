# Simple presentation plots (no TensorBoard required)

TensorBoard is optional. Training and evaluation save ordinary files, and the
plotting script turns them into PNG figures.

## 1. Train

```powershell
python -m pip install -e ".[dev,rl]"
python -m rl.train --seconds 600 --seed 42 --run-name demo
```

## 2. Evaluate initial and best policies on the same test rays

```powershell
python -m rl.evaluate --model outputs/rl_runs/demo/initial_model.zip --split test --output outputs/rl_runs/demo/evaluation_initial
python -m rl.evaluate --model outputs/rl_runs/demo/best/best_model.zip --split test --output outputs/rl_runs/demo/evaluation_final
```

If the best checkpoint was not created, evaluate `final_model.zip` and label
it accurately as the final checkpoint.

## 3. Create summary plots

```powershell
python -m rl.presentation_plots outputs/rl_runs/demo
```

Output:
- `outputs/rl_figures/01_training_progress.png`: validation episode return
  over training timesteps;
- `outputs/rl_figures/02_test_error_comparison.png`: initial versus trained
  mean RMSE on the same held-out test rays;
- each evaluation folder contains `trajectory_comparison.png`, with the
  orbit, inverse-radius state, and derivative state.

## 4. Interpretation

Higher episode return is better, but return is a sum across an episode and
ray lengths differ. Use held-out RMSE and the trajectory overlay as the main
accuracy evidence. Lower RMSE is better. Solid lines are DOP853 and dashed
lines are PPO.

The model is autoregressive: predicted states feed into later observations.
This makes the trajectory plot important because local errors can accumulate.
The experiment is teacher-guided tracking, not independent physics discovery.
