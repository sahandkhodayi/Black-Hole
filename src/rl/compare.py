"""Plot validation reward histories from several wall-clock PPO runs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", default="outputs/rl_figures/training_progress.png")
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for raw in args.run_dirs:
        folder = Path(raw)
        path = folder / "evaluation" / "evaluations.npz"
        if not path.exists():
            print(f"Skipping {folder}: no evaluation history")
            continue
        with np.load(path, allow_pickle=False) as data:
            steps, rewards = np.asarray(data["timesteps"]), np.asarray(data["results"], dtype=float)
        ax.plot(steps, rewards.mean(axis=1), marker="o", label=folder.name)
        plotted += 1
    if not plotted:
        raise SystemExit("No evaluation histories found; train until the first evaluation.")
    ax.set(title="PPO validation reward during training", xlabel="Environment timesteps",
           ylabel="Mean validation episode reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
