# Black Hole Lab — parallel training and classroom run

This workflow adds a supervised dynamics MLP baseline alongside the existing PPO experiment. It is designed to be run **before** class. It does not train anything during the classroom demonstration.

## Important: what the current PPO code actually learns

The current `PhotonTrajectoryEnv` does **not** let PPO choose an ODE solver step size. It asks PPO to predict the next state `(u, v)` of a reference photon trajectory. The DOP853 next state is used internally to calculate reward; it is not included in the policy observation.

The supervised baseline uses the same observation/action contract:

- observation: `[r0/12, phi/3, current_u/0.5, current_v/0.2]`
- action 0 decodes to `next_u = 0.25 * (action[0] + 1)`
- action 1 decodes to `next_v = 0.2 * action[1]`

The selected MLP is `dynamics_architecture_sweep_relu_w128_d3`: three hidden layers, 128 units per layer, ReLU hidden activation. The output layer is linear, as in a standard regression MLP. At evaluation, predictions are clipped to the same valid action range as the PPO environment.

This is a **supervised MLP**, not a PINN: its loss is prediction error against reference examples; it does not add the differential-equation residual to the loss. If “IPNN” means a different architecture in your local code, do not relabel this MLP as a PINN.

## 1. Install dependencies and check the project

From the repository root in PowerShell:

```powershell
python -m pip install -e ".[dev,rl]"
python -m pytest -q
```

Keep the existing `data/photon_trajectories_v2.npz` and `data/split_manifest_v2.json`. The split manifest is by complete ray: 21 training rays, 4 validation rays, and 5 held-out test rays. Do not split individual samples from one ray across train and test.

If those version-2 data files are missing, create them first from the repository root:

```powershell
python src/generate_data.py --count 30 --r-min 2.2 --r-max 12 --output data/photon_trajectories_v2.npz --manifest data/trajectory_manifest_v2.json
python src/make_splits.py --data data/photon_trajectories_v2.npz --output data/split_manifest_v2.json --seed 42
```

This creates a reproducible reference dataset and a split by whole trajectory. Do not generate a new split after seeing test results; keep the same manifest for both learning methods.

## 2. Train both models at the same time

Recommended first presentation run (10-minute PPO budget):

```powershell
python -m presentation.run_parallel --run-name class_presentation --rl-seconds 600 --mlp-epochs 300 --seed 42
```

The command starts two independent Python processes:

1. **MLP:** trains the selected ReLU-128-3 architecture, saves the checkpoint with the lowest autoregressive error on the four validation rays, and stops when that validation score stops improving.
2. **PPO:** runs the existing time-limited `rl.train` script and saves checkpoints plus its configuration/logs.
3. **PPO checkpoint selection:** if both the validation-reward `best_model.zip` and `final_model.zip` exist, the runner evaluates both on the validation rays and keeps the one with lower autoregressive normalized RMSE. It never uses the test split for checkpoint selection.

The processes write separate logs under:

```text
outputs/presentation/class_presentation/training_logs/mlp.log
outputs/presentation/class_presentation/training_logs/ppo.log
```

Saved models:

```text
outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt
outputs/rl_runs/class_presentation_ppo/best/best_model.zip
outputs/presentation/class_presentation/selected_ppo_model.zip
outputs/presentation/class_presentation/ppo_checkpoint_selection.json
```

The selected PPO checkpoint is copied to a stable path for the classroom demo. It may come from `best_model.zip` or `final_model.zip`, depending on which performs better on validation rollouts. The comparison is automatically generated after both jobs finish.

You can run the jobs separately if needed:

```powershell
python -m presentation.train_mlp --epochs 300 --patience 30 --seed 42
python -m rl.train --seconds 600 --seed 42 --run-name class_presentation_ppo
```

## 3. What comparison gets generated?

The evaluator loads both saved checkpoints and tests them on the exact same five held-out rays. Both models run **autoregressively**: after the initial state, each model receives its own last prediction as the next input. This is stricter than scoring one-step predictions only, because errors can accumulate along the trajectory.

Output folder:

```text
outputs/presentation/class_presentation/comparison/
├── 01_mean_error_comparison.png
├── 02_error_by_test_ray.png
├── 03_example_trajectory_comparison.png
├── analysis.md
├── metrics_by_ray.csv
├── run_metadata.json
└── summary.json
```

How to read the results:

- **RMSE** measures prediction error; lower is better.
- `u = 1/r` is inverse distance from the black hole.
- `v = du/dphi` measures how the inverse distance changes with angle.
- A trajectory overlay shows whether the predicted curve follows DOP853.
- The “best” model can differ between the `u` and `v` metrics. Do not hide that by reporting only one number.
- The MLP is trained on one-step pairs whose current inputs are DOP853 reference states; at evaluation it receives its own previous prediction. PPO is trained through its sequential environment. This makes the test procedure the same, while the training signals remain different.

## 4. Fast classroom demonstration — no training

Before class, make sure both checkpoints exist. Then run:

```powershell
python -m presentation.classroom_demo --show
```

It loads both trained models, including the PPO checkpoint selected using validation-only rollouts, selects the first held-out test ray (case 4 in the current split), performs one short rollout, prints the two RMSE values, and opens a plot comparing DOP853, MLP, and PPO.

To show a different held-out ray:

```powershell
python -m presentation.classroom_demo --case-id 27 --show
```

This command does not call `.learn()` and does not train PPO. It only loads the saved policies and evaluates one ray. It should take seconds, not the time needed for RL training.

## 5. Suggested live explanation (3–5 minutes)

1. Show the reference curve and ask students what the two state variables represent.
2. Explain that MLP learned from examples, while PPO learned by trying a next-state action and receiving a reward.
3. Run `classroom_demo.py` and point to the solid reference curve and the two model predictions.
4. Show the printed RMSE table: smaller error means a closer prediction on this particular ray.
5. Show the mean-error chart from the full five-ray test. Explain that one ray can be easier than another, which is why we test several rays.
6. State the limitation: the experiment compares two ways of predicting the next state; it does not prove AI discovered relativity or that PPO is universally better.

## 6. Re-run the comparison without training

```powershell
python -m presentation.evaluate_pair `
  --mlp outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt `
  --ppo outputs/presentation/class_presentation/selected_ppo_model.zip `
  --split test `
  --output outputs/presentation/class_presentation/comparison
```

The parallel runner already chooses between the PPO validation-best and final checkpoints using validation rollout error. Use the copied `selected_ppo_model.zip` for the test comparison so that the test set remains untouched during selection.

## Scientific limits to say out loud

- There are only five held-out test rays in the current split. The comparison is preliminary.
- DOP853 is a high-accuracy numerical reference, not an exact symbolic solution.
- PPO uses teacher-guided rewards: reference next-state values determine the reward. The agent is not receiving the future reference values as observations.
- PPO's current action is a next-state prediction, not an adaptive solver step size.
- This MLP is data-supervised. It is not physics-informed unless the training objective explicitly includes a physics residual.