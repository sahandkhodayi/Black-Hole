"""Create simple, presentation-ready PNG plots from PPO run folders.

Example:
    python -m rl.presentation_plots outputs/rl_runs/1min outputs/rl_runs/10min
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(run_dir):
    path = run_dir / "evaluation" / "evaluations.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        steps = np.asarray(data["timesteps"], dtype=float)
        rewards = np.asarray(data["results"], dtype=float)
    return steps, rewards.mean(axis=1), rewards.std(axis=1)


def find_test_summary(run_dir):
    # Prefer explicit initial/final comparisons created by the quickstart.
    candidates = [
        ("Initial policy", run_dir / "evaluation_initial" / "summary.json"),
        ("Trained policy", run_dir / "evaluation_final" / "summary.json"),
        ("Best policy", run_dir / "evaluation_test" / "summary.json"),
    ]
    found = []
    for label, path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            found.append((label, summary))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="One or more outputs/rl_runs/<run-name> folders")
    parser.add_argument("--output", default="outputs/rl_figures")
    args = parser.parse_args()

    runs = [Path(item) for item in args.run_dirs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Figure 1: validation reward through training.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    history_count = 0
    for run in runs:
        history = load_history(run)
        if history is None:
            print(f"No validation history found for {run}; skipping reward curve.")
            continue
        steps, mean, std = history
        ax.plot(steps, mean, marker="o", markersize=3, label=run.name)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.16)
        history_count += 1
    if history_count:
        ax.set(title="PPO learning progress on validation rays",
               xlabel="Training timesteps", ylabel="Mean episode reward")
        ax.grid(alpha=0.25)
        ax.legend(title="Training run")
        fig.tight_layout()
        path = output / "01_training_progress.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved {path}")
    else:
        print("No reward plot created: train at least until the first evaluation.")
    plt.close(fig)

    # Figure 2: held-out test errors for initial/final/best policy.
    labels, rmse_u, rmse_v = [], [], []
    for run in runs:
        summaries = find_test_summary(run)
        for label, summary in summaries:
            labels.append(f"{run.name}\n{label}")
            rmse_u.append(float(summary["mean_rmse_u"]))
            rmse_v.append(float(summary["mean_rmse_v"]))

    if labels:
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.15), 5.5))
        ax.bar(x - width / 2, rmse_u, width, label="u = 1/r error")
        ax.bar(x + width / 2, rmse_v, width, label="du/dphi error")
        ax.set(title="PPO prediction error on held-out test rays",
               ylabel="Mean RMSE (lower is better)", xticks=x, xticklabels=labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output / "02_test_error_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)
    else:
        print("No test-error plot created. Run evaluate.py for initial and final models first.")

    print(f"\nPresentation figures folder: {output.resolve()}")


if __name__ == "__main__":
    main()
