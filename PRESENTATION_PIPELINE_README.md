# Presentation pipeline patch for Black Hole Lab

Copy the folders in this patch into the root of your existing Black Hole Lab repository, keeping the folder structure. It adds a supervised dynamics MLP baseline, parallel MLP + PPO training, a held-out comparison, a plain-language analysis report, and a fast classroom demo that loads saved checkpoints.

## Quick start (PowerShell)

```powershell
python -m pip install -e ".[dev,rl,ml]"
python -m pytest -q
python -m presentation.run_parallel --run-name class_presentation --rl-seconds 600 --mlp-epochs 300 --seed 42
python -m presentation.classroom_demo --show
```

Or run `scripts/train_for_presentation.ps1` to train and evaluate, then `scripts/run_class_demo.ps1` during class.

Full workflow and explanations: `docs/PRESENTATION_CODE_WORKFLOW.md`.

## Important scientific scope

The current code's PPO environment predicts the next photon state `(u, v)` and uses DOP853 values to calculate teacher-guided reward. It does not control the ODE solver's step size. The MLP is trained on supervised one-step examples from the same dataset; both models are evaluated autoregressively on identical held-out rays.

The requested architecture name `dynamics_architecture_sweep_relu_w128_d3` is implemented as 3 hidden ReLU layers with width 128 and a linear two-value regression output. This patch retrains that architecture on the current project dataset. The uploaded source archive did not contain a trained MLP weights file, so it cannot reuse an older checkpoint unless that checkpoint is separately provided and its architecture/state-dict format is verified.

The MLP is supervised, not a PINN: the training loss is next-state prediction error. No differential-equation residual is included in the neural-network loss.

## Outputs

- `outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt`
- `outputs/rl_runs/class_presentation_ppo/best/best_model.zip` and `final_model.zip` (when available)
- `outputs/presentation/class_presentation/selected_ppo_model.zip` (selected by validation-only autoregressive error)
- `outputs/presentation/class_presentation/ppo_checkpoint_selection.json`
- `outputs/presentation/class_presentation/comparison/metrics_by_ray.csv`
- `outputs/presentation/class_presentation/comparison/01_mean_error_comparison.png`
- `outputs/presentation/class_presentation/comparison/02_error_by_test_ray.png`
- `outputs/presentation/class_presentation/comparison/03_example_trajectory_comparison.png`
- `outputs/presentation/class_presentation/comparison/analysis.md`

All comparison numbers are generated from the actual run. The patch does not ship invented result values.