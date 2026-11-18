"""Compare MLP architectures by autoregressive validation error.

The test split is never consulted during model selection. Each candidate is
trained on the same training rays and ranked on complete held-out validation
rays after feeding its own predictions back into the model.
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


ACTIVATIONS = ("relu", "tanh", "gelu", "silu", "elu")
WIDTHS = (64, 128)
DEPTHS = (2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-name", default="dynamics_architecture_sweep")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")

    root = Path("outputs/rl_runs") / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    results = []
    candidates = list(itertools.product(ACTIVATIONS, WIDTHS, DEPTHS))
    print(f"Comparing {len(candidates)} MLP configurations on validation rollouts.")
    print("Selection metric: normalized autoregressive state RMSE.")
    print("The test split is not used for model selection.")

    for index, (activation, width, depth) in enumerate(candidates, start=1):
        name = f"{args.run_name}_{activation}_w{width}_d{depth}"
        print(f"\\n[{index}/{len(candidates)}] {name}", flush=True)
        subprocess.run([
            sys.executable, "-m", "rl.train_dynamics",
            "--dataset", args.dataset, "--splits", args.splits,
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--seed", str(args.seed), "--device", args.device,
            "--run-name", name, "--activation", activation,
            "--width", str(width), "--depth", str(depth),
        ], check=True)
        eval_dir = Path("outputs/rl_runs") / name / "selection_validation"
        subprocess.run([
            sys.executable, "-m", "rl.evaluate_dynamics",
            "--model", str(Path("outputs/rl_runs") / name / "best" / "best_model.pt"),
            "--dataset", args.dataset, "--splits", args.splits,
            "--split", "validation", "--output", str(eval_dir),
            "--device", args.device,
        ], check=True)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        # Normalize each state component by its characteristic scale so u
        # does not dominate v merely because their numerical units differ.
        score = ((summary["mean_rmse_u"] / 0.5) ** 2 +
                 (summary["mean_rmse_v"] / 0.2) ** 2) ** 0.5
        row = {
            "name": name, "activation": activation, "width": width,
            "depth": depth, "validation_rmse_u": summary["mean_rmse_u"],
            "validation_rmse_v": summary["mean_rmse_v"],
            "normalized_rollout_score": score,
            "checkpoint": str(Path("outputs/rl_runs") / name / "best" / "best_model.pt"),
        }
        results.append(row)
        print(f"Validation rollout score: {score:.6f}", flush=True)

        with (root / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda item: item["normalized_rollout_score"]))

    ranked = sorted(results, key=lambda item: item["normalized_rollout_score"])
    best = ranked[0]
    report = {
        "selection": "lowest normalized autoregressive validation RMSE",
        "test_split_used_for_selection": False,
        "candidate_count": len(candidates),
        "best_model": best,
        "leaderboard": ranked,
    }
    (root / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    (root / "BEST_MODEL.txt").write_text(
        f"Best validation model: {best['name']}\\n"
        f"Checkpoint: {best['checkpoint']}\\n"
        f"Normalized validation rollout score: {best['normalized_rollout_score']:.8f}\\n",
        encoding="utf-8",
    )
    print("\\nBEST VALIDATION MODEL")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
