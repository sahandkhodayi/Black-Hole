# Reading your Black Hole Lab results

Use this local website to start a PPO training run, watch its progress, stop and save it, and inspect previous MLP, PPO, and classroom-comparison results. Training runs in a separate backend process; the browser does not run the model itself.

## Open it

From the Black Hole Lab project folder, use the same Python environment you use for the project:

```powershell
python -m pip install -e ".[rl]"
python -m bh_analytics.dashboard --root outputs
```

Your browser opens at `http://127.0.0.1:8765`. Keep the terminal window open while using the page. Stop the dashboard with `Ctrl+C`.

If you are using WSL and the browser does not open automatically:

```bash
python -m bh_analytics.dashboard --root outputs --no-browser
```

Then open `http://127.0.0.1:8765` in your Windows browser.

## Start training from the website

At the top of the page, use **Train a PPO agent**:

1. Choose a run length (1, 5, 10, or 30 minutes) and a random seed.
2. Press **Start training**. The backend launches the existing PPO training script using the same Python environment as the dashboard.
3. The page checks the backend once every **60 seconds**. It does not poll every second. The reward, optimization-loss, and validation-error graphs refresh when a new snapshot is ready.
4. Press **Stop & save** to request a safe stop. The trainer finishes its current step, saves the final model, and keeps the best validation checkpoint.

The first graph can take a little time to appear because the trainer waits for a PPO rollout boundary before recording its first validation snapshot. Later snapshots are written at approximately one-minute intervals. Validation work also happens only at those intervals, so it does not run on every environment step.

The browser controls currently launch **PPO training**. The existing MLP training and paired MLP-vs-PPO presentation commands remain available through their documented command-line workflow. Only one browser-launched training run can be active at a time.

Training controls are enabled only when the dashboard is bound to localhost. This is intentional: the local training API is not an authenticated remote job server.

The page uses these local API routes:

- `POST /api/training/start` starts a PPO run.
- `GET /api/training/status` returns the latest saved history snapshot.
- `POST /api/training/stop` requests a safe save-and-stop.

## What you will see

The dashboard opens on the classroom MLP-vs-PPO comparison when it is available. This is the most useful first result because both models were tested on the same unseen photon paths.

- **Unseen test paths** tells you how many complete paths were used for the comparison.
- **Average error** tells you how closely a model's predicted path followed the DOP853 numerical reference. Smaller is better; zero would be a perfect match.
- **The bars** make the two average errors easy to compare.
- **Individual paths** shows the score for each test path when those details were saved.
- **Pictures** show the trajectory and error plots already produced by the project.

The result is only about this experiment and its test paths. It does not prove that one AI method is always better or that either model discovered Einstein's equations.

You can select another run on the left:

- **Neural network training** shows whether the MLP's training and validation errors improved.
- **PPO training and test** shows its saved test error and reward history when available.
- **MLP vs PPO comparison** shows both models side by side.

## Do I need to prepare CSV files?

**No. Do not create or move CSV files by hand.**

The dashboard reads results already saved by the training and evaluation commands. It can show the overall result from a saved `summary.json` even when a per-path CSV is not present. CSV files only add the optional per-path table or detailed training curve. A missing CSV must not hide the overall experiment result.

The usual output locations are:

- `outputs/mlp_runs/<run>/` for MLP training.
- `outputs/rl_runs/<run>/` for PPO training and its evaluations.
- `outputs/presentation/<run>/comparison/` for the paired comparison.

The dashboard checks those locations automatically. Press the refresh button after creating new results.

## Save a report

Choose an experiment and press **Save a shareable report**. A standalone HTML file is created under:

```text
outputs/analytics/reports/
```

The report includes the plain-language takeaway, key numbers, plots and any available per-path scores. You can open it later or send it to someone; it does not need the dashboard server to be running.

## Nothing appears?

1. Make sure you started the command from the Black Hole Lab project folder.
2. Check that `--root outputs` points to the folder that contains `mlp_runs`, `rl_runs`, and/or `presentation`.
3. If you just trained a model, run its evaluation step to create test results, then press refresh.

If the folder is empty, the page explains what it is looking for. It does not require a pre-made collection of CSV files or a manual dashboard setup.
