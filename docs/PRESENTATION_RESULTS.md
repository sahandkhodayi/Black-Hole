# Current experimental result — stable_v1

This page records the result supplied from the local `stable_v1` run. It is
not a claim that the repository can reproduce the exact local artifacts
without the run folder.

## Experiment

- Task: teacher-guided autoregressive photon-trajectory tracking.
- Algorithm: PPO (Stable-Baselines3).
- Run name: `stable_v1`.
- Evaluation: five complete held-out test rays.
- Reference: DOP853.
- Metrics: RMSE in inverse radius (u=1/r) and derivative
  (v=du/d\phi). Lower is better.

## Initial versus trained policy

The uploaded comparison chart shows approximately:

| Metric | Initial policy | Trained policy | Relative reduction |
|---|---:|---:|---:|
| Mean RMSE (u) | 0.159 | 0.1039 | 34.7% |
| Mean RMSE (v) | 0.111 | 0.0679 | 38.8% |

The initial chart values are read approximately from the plotted bars. The
trained values below are calculated directly from the five per-ray CSV values.

## Trained policy: per-ray results

| Case ID | Closest approach (r_0/r_s) | RMSE (u) | RMSE (v) |
|---:|---:|---:|---:|
| 4 | 10.3103 | 0.10618 | 0.04269 |
| 27 | 2.5379 | 0.13077 | 0.13932 |
| 1 | 7.6069 | 0.09803 | 0.03317 |
| 13 | 4.2276 | 0.07691 | 0.07800 |
| 8 | 10.9862 | 0.10745 | 0.04649 |
| **Mean** | — | **0.10387** | **0.06793** |

## Interpretation

The trained policy improves average test RMSE relative to the untrained
initial policy. This is a positive result on the held-out rays.

However, the orbit and inverse-radius plots show that the predicted curves
still differ visibly from DOP853. Several PPO inverse-radius predictions
remain nearly flat rather than decreasing toward zero. Lower average RMSE
does not mean that the policy has learned the complete scattering dynamics.

The defensible claim is:

> In this initial five-ray held-out evaluation, PPO reduced mean inverse-radius
> RMSE by approximately 35% and mean derivative RMSE by approximately 39%
> relative to its initial policy. The predicted trajectories still show
> substantial mismatch from the DOP853 reference, so the result demonstrates
> preliminary learning rather than accurate trajectory reconstruction.

## What this result does not establish

- It does not show that PPO is more accurate than DOP853.
- It does not show generalization across the full physical parameter space.
- It does not show independent escape/capture detection.
- It does not show that RL is better than supervised learning.
- It does not establish robustness across multiple random seeds.

## Files to show

From the local run directory:
- `evaluation_initial/trajectory_comparison.png`
- `evaluation_final/trajectory_comparison.png`
- `evaluation_initial/summary.json`
- `evaluation_final/summary.json`
- `evaluation_final/metrics.csv`

Generated summary plots:
- `outputs/rl_figures/01_training_progress.png`
- `outputs/rl_figures/02_test_error_comparison.png`

For a final report, replace the approximate initial chart values with exact
values from `evaluation_initial/summary.json` and record the Git commit,
seed, actual timesteps, and elapsed time from the run's `config.json`.
