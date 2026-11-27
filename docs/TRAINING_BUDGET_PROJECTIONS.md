# Training budget projections

Open `presentation/training_budget_explorer.html` directly in a browser. It is self-contained, needs no server, CDN, or training run.

## What is measured

- **MLP:** 68.1 s actual training time; test mean normalized RMSE 0.04115075 on the five held-out rays. The MLP log reports best validation rollout score 0.00405923 at epoch 130 and early stopping at epoch 160.
- **PPO, short run:** 60.502 s actual; test mean normalized RMSE 0.30079889. Recomputed from the five per-ray test rows using the metric formula in `presentation.common.calculate_metrics`, with U_SCALE=0.5 and V_SCALE=0.2.
- **PPO, longer run:** 600.125 s actual; test mean normalized RMSE 0.28818041 from the saved class-presentation comparison summary.

## What is projected

- MLP values after its 68.1-second measured point are held flat as a **plateau assumption**, because the saved run early-stopped. They are not longer training measurements.
- PPO values after 600.125 seconds are extrapolated with a two-point power law fitted to the 60-second and 600-second results. The two PPO runs used different reward functions and PPO hyperparameters, so this is an illustrative scenario only. The 1-hour, 10-hour, and 10-day values are not actual model checkpoints or test results.

## Metric

Mean normalized RMSE across five held-out test rays:

`sqrt(0.5 * mean((du / 0.5)^2 + (dv / 0.2)^2))`

Lower is better. The five-ray test set is small; do not generalize these results beyond this experiment.

## Source artifacts

- `outputs/rl_runs/1min/config.json`
- `outputs/rl_runs/1min/evaluation_best/metrics.csv`
- `outputs/rl_runs/class_presentation_ppo/config.json`
- `outputs/presentation/class_presentation/comparison/summary.json`
- `outputs/presentation/class_presentation/training_logs/mlp.log`
- `src/presentation/common.py`

No checkpoints, datasets, or original outputs were deleted.
