# Black Hole Lab — post-training analytics dashboard

The analytics dashboard is a local browser-based GUI for inspecting experiment
artifacts after training. It complements the existing Matplotlib live PPO
training window: live monitoring stays in the training process, while this
dashboard loads saved runs and lets you compare and report them later.

It is intentionally read-only with respect to experiment outputs. It does not
load or execute model checkpoints, start training, overwrite CSV files, or
change experiment metadata. It reads JSON and CSV artifacts and displays
images already produced by the project.

## Start the dashboard

Run this from the repository root in PowerShell or a WSL shell:

```powershell
python -m bh_analytics.dashboard --root outputs
```

The server binds to `127.0.0.1:8765` and opens the browser automatically. If
your environment cannot launch a Windows browser from WSL, run:

```bash
python -m bh_analytics.dashboard --root outputs --no-browser
```

Then open `http://127.0.0.1:8765` manually in your browser. Stop the server
with `Ctrl+C` in the terminal.

The dashboard is local-only by default. Do not bind it to `0.0.0.0` on an
untrusted network: the browser interface exposes experiment CSVs and images
from the selected outputs directory.

## What it discovers

The loader adapts to the project's existing artifact formats; training code
does not need to be changed to use it.

| Output folder | Type | Data shown |
|---|---|---|
| `outputs/mlp_runs/<run>/` | Supervised MLP | `config.json`, `history.csv`, best epoch, validation rollout NRMSE, checkpoint presence |
| `outputs/rl_runs/<run>/` | PPO / RL | `config.json`, `live_history.csv` or evaluation reward history, all `evaluation_*` summaries and per-ray metrics |
| `outputs/presentation/<run>/comparison/` | Paired comparison | `summary.json`, `run_metadata.json`, `metrics_by_ray.csv`, comparison plots, existing `analysis.md` |

The experiment library lists each folder that meets the minimum artifact
requirements. Use **Refresh** after another training/evaluation process writes
new files.

## Dashboard tabs

- **Overview:** experiment type, status, summary metrics, configuration, and
  artifact counts. Paired presentation runs include MLP-vs-PPO comparison
  cards and separate u/v error comparisons.
- **Training:** MLP train/validation MSE and autoregressive validation error;
  PPO reward history and validation RMSE when those series were saved. The
  chart does not invent missing history.
- **Evaluation:** choose an available PPO evaluation split (`initial`, `final`,
  `best`, `validation`, or `test`) and inspect its summary, per-ray CSV, and
  saved trajectory comparison. Paired runs show both models' held-out metrics
  side by side, including each test ray.
- **Artifacts & data:** open existing PNG/SVG/JPEG figures and download raw CSV
  files. The dashboard keeps raw tables available rather than replacing them
  with screenshots.

## Generate a portable HTML report

From the selected experiment, press **Generate HTML report**. The dashboard
writes the report to:

```text
outputs/analytics/reports/<experiment-id>.html
```

The report embeds the experiment configuration, metric tables, compatible
training-history charts, and existing image artifacts. Images are embedded as
data URIs, so the report can be opened or shared without the original output
folder or a running dashboard. It contains the measured values from the run;
it does not claim that a model is accurate beyond the saved evaluation split.

You can also generate a report from the terminal:

```powershell
python -m bh_analytics.report --root outputs --experiment presentation/class_presentation
```

Use an ID shown in the dashboard's experiment library. Examples:

- `mlp/dynamics_architecture_sweep_relu_w128_d3`
- `rl/class_presentation_ppo`
- `presentation/class_presentation`

The destination defaults to `outputs/analytics/reports/`. To choose another
folder, pass `--output-dir PATH`.

## Current repository-specific notes

- The tracked MLP run saves `history.csv` with `train_mse`, `validation_mse`,
  and validation rollout error columns. Its configuration also records the
  selected best epoch and training settings.
- PPO runs can expose `live_history.csv`. Older runs without it can still
  expose evaluation rewards from `evaluation/evaluations.npz`; if neither is
  present, the Training tab says that no history was saved.
- PPO evaluation folders such as `evaluation_test/` contain `summary.json`,
  `metrics.csv`, and a trajectory plot. The dashboard prefers the test split
  for the run's headline metrics, while keeping other splits selectable.
- The paired presentation run contains the held-out MLP/PPO aggregate summary
  and per-ray rows. This is the best place to compare the two current learning
  approaches on identical test rays.
- A standalone MLP training folder does not necessarily contain its own
  autoregressive test-evaluation CSV. In that case the dashboard reports the
  training history and directs you to the paired comparison rather than
  inventing test metrics from the training loss.

## Troubleshooting

**No experiments appear:** confirm `--root` points to the `outputs` directory,
not the repository root or an individual run folder. Expected subfolders are
`mlp_runs`, `rl_runs`, and `presentation`.

**A chart says history is unavailable:** the selected run did not save a
compatible history file. Evaluation CSVs and saved figures can still be
inspected. This is different from a zero metric.

**The browser does not open:** use `--no-browser` and visit the local URL
manually. Ensure that the Python process remains running.

**A report has no images:** the report can only embed images that already
exist beneath the selected experiment folder. Generate the relevant
presentation/evaluation plots first, then press Refresh and regenerate.

## Dependencies and security

No extra Python dependency is needed beyond the project's base install. The
GUI uses Python's standard-library HTTP server and browser-native JavaScript
canvas charts. It does not load scripts or chart libraries from a CDN. Requests
are restricted to discovered experiment IDs and files safely resolved below
the configured outputs directory.
