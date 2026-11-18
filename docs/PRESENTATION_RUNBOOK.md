# Presentation runbook — Black Hole Lab

This runbook rebuilds generated artifacts after deleting the local `outputs/`
folder and creates a clean, reproducible set of presentation figures.

## Important: what is already in Git

The repository tracks the reference dataset and split manifest:

- `data/photon_trajectories_v2.npz`
- `data/split_manifest_v2.json`

Deleting `outputs/` does not delete these files. Training and evaluation
recreate their output folders automatically. Do not delete the source data
or split manifest when cleaning generated artifacts.

## 1. Update the correct branch

PowerShell, from the repository root:

```powershell
git switch feature/black-hole-lab-overhaul
git pull origin feature/black-hole-lab-overhaul
```

If Git says the branch does not exist locally, fetch it first:

```powershell
git fetch origin
git switch --track origin/feature/black-hole-lab-overhaul
```

## 2. Install and test

Use the existing virtual environment if it works:

```powershell
python -m pip install -e ".[dev,rl]"
python -m pytest -q
```

Do not delete and recreate the virtual environment just because the
`outputs/` folder was deleted.

## 3. Optional: build a larger presentation dataset

The tracked v2 dataset is a small smoke-test set (30 rays). It is useful for
checking that the pipeline runs, but too small for strong generalization
claims. For a larger experiment, create a separate dataset so the tracked
smoke-test files remain untouched:

```powershell
python src/generate_data.py --count 120 --r-min 1.6 --r-max 12 --delta-phi 0.01 --max-angle 80 --seed 42 --output data/presentation_trajectories.npz --manifest data/presentation_trajectory_manifest.json
python src/make_splits.py --data data/presentation_trajectories.npz --output data/presentation_splits.json --seed 42
```

The lower closest-approach range samples stronger bending near the photon
sphere at `r=1.5 r_s`. Some very near-critical rays may need a large
integration angle and can remain unresolved; inspect the generation manifest
rather than silently treating unresolved cases as successful escapes.

If you want the quickest smoke test, skip this step and use the tracked v2
dataset and split manifest in the next commands.

## 4. Train the supervised neural baseline

For the larger presentation dataset:

```powershell
python -m rl.train_dynamics --dataset data/presentation_trajectories.npz --splits data/presentation_splits.json --epochs 300 --patience 30 --seed 42 --run-name presentation_mlp_v1
```

For the tracked smoke-test dataset instead:

```powershell
python -m rl.train_dynamics --dataset data/photon_trajectories_v2.npz --splits data/split_manifest_v2.json --epochs 300 --patience 30 --seed 42 --run-name smoke_mlp_v1
```

Training creates `outputs/rl_runs/<run-name>/` automatically. It stores the
best validation checkpoint, training history, and experiment config.

## 5. Make the held-out visual comparison

For the larger dataset:

```powershell
python -m rl.evaluate_dynamics --model outputs/rl_runs/presentation_mlp_v1/best/best_model.pt --dataset data/presentation_trajectories.npz --splits data/presentation_splits.json --split test --output outputs/rl_runs/presentation_mlp_v1/evaluation_test
Start-Process "outputs\rl_runs\presentation_mlp_v1\evaluation_test\trajectory_comparison.png"
```

The figure displays three representative held-out rays to keep it readable.
All test rays—not just the three drawn—are included in `metrics.csv`.

- Solid curves: DOP853 numerical reference.
- Dashed curves: autoregressive neural-network predictions.
- Each dashed prediction is fed back into the model for the next step.

## 6. What to show in the presentation

1. A DOP853 geometry/reference figure.
2. The supervised MLP held-out comparison figure.
3. The training/validation loss history.
4. A table or appendix with the full test-set metrics and unresolved-ray
   count from the data manifest.

Do not claim the MLP discovered the geodesic equation. It learns from
DOP853-generated examples without receiving the equation as an input.

## 7. The RL experiment is a separate milestone

The current `PhotonTrajectoryEnv` is teacher-guided PPO: DOP853 next-state
targets determine its reward. It must not be presented as equation-hidden
or autonomous RL.

The equation-hidden RL experiment requires a separate black-box task and
sparse outcome reward. A promising first scientific target is discovering
the capture/escape boundary or critical photon behaviour from interaction,
then comparing the learned result against the theoretical reference only
after training. That is not the same as giving PPO pointwise DOP853 targets.

Keep the two experiments separate in plots, configs, and presentation claims.
