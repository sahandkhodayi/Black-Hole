"""Run pre-presentation training budgets and export measured/projection CSVs.

By default this runs 1 min, 10 min, and 1 hour for each model. This can take
several hours. Run it well before the presentation. 10-hour and 10-day entries
are extrapolations from at least two measured checkpoints, never measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_or_final(directory: Path, name: str):
    best = directory / "best" / "best_model.zip"
    final = directory / "final_model.zip"
    if best.is_file():
        return best
    if final.is_file():
        return final
    raise FileNotFoundError(f"{name} training produced neither {best} nor {final}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--output-root", default="outputs/budget_sweep")
    parser.add_argument("--budgets", nargs="+", type=int, default=[60, 600, 3600])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--models", nargs="+",
                        choices=["MLP", "PINN", "PPO", "Adaptive RL"],
                        default=["MLP", "PINN", "PPO", "Adaptive RL"])
    args = parser.parse_args()
    if any(value <= 0 for value in args.budgets):
        parser.error("all budgets must be positive seconds")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    measured = []

    for seconds in args.budgets:
        label = f"{seconds}s"
        run_root = root / label
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoints = {}
        if "MLP" in args.models:
            mlp_dir = run_root / "mlp"
            run([sys.executable, "-m", "presentation.train_mlp",
                 "--dataset", args.dataset, "--splits", args.splits,
                 "--output-dir", mlp_dir, "--seconds", seconds,
                 "--seed", args.seed, "--device", args.device])
            checkpoints["MLP"] = mlp_dir / "best_model.pt"
        if "PINN" in args.models:
            pinn_dir = run_root / "pinn"
            run([sys.executable, "-m", "presentation.train_pinn",
                 "--dataset", args.dataset, "--splits", args.splits,
                 "--output-dir", pinn_dir, "--seconds", seconds,
                 "--seed", args.seed, "--device", args.device])
            checkpoints["PINN"] = pinn_dir / "best_model.pt"
        if "PPO" in args.models:
            ppo_name = f"budget_{label}_ppo"
            run([sys.executable, "-m", "rl.train", "--dataset", args.dataset,
                 "--splits", args.splits, "--seconds", seconds,
                 "--seed", args.seed, "--run-name", ppo_name,
                 "--output-root", "outputs", "--device", args.device])
            checkpoints["PPO"] = checkpoint_or_final(Path("outputs/rl_runs") / ppo_name, "PPO")
        if "Adaptive RL" in args.models:
            adaptive_name = f"budget_{label}_adaptive"
            run([sys.executable, "-m", "rl.train_adaptive", "--dataset", args.dataset,
                 "--splits", args.splits, "--seconds", seconds,
                 "--seed", args.seed, "--run-name", adaptive_name,
                 "--output-root", "outputs", "--device", args.device])
            checkpoints["Adaptive RL"] = checkpoint_or_final(
                Path("outputs/adaptive_rl_runs") / adaptive_name, "Adaptive RL")

        required = {"MLP", "PINN", "PPO"}
        if not required.issubset(checkpoints):
            print(f"[budget] {label}: trained {sorted(checkpoints)}; full arena needs MLP, PINN, and PPO. Skipping paired scoring.")
            continue
        arena_dir = run_root / "validation_arena"
        command = [sys.executable, "-m", "presentation.evaluate_arena",
                   "--dataset", args.dataset, "--splits", args.splits,
                   "--mlp", checkpoints["MLP"], "--ppo", checkpoints["PPO"],
                   "--pinn", checkpoints["PINN"], "--output", arena_dir,
                   "--split", "validation", "--device", "cpu"]
        if "Adaptive RL" in checkpoints:
            command.extend(["--adaptive-ppo", checkpoints["Adaptive RL"]])
        run(command)
        summary_path = arena_dir / "summary.csv"
        with summary_path.open(newline="", encoding="utf-8") as handle:
            scores = {row["model"]: float(row["mean_normalized_rmse"])
                      for row in csv.DictReader(handle)}
        config_paths = {
            "MLP": run_root / "mlp" / "config.json",
            "PINN": run_root / "pinn" / "config.json",
            "PPO": Path("outputs/rl_runs") / f"budget_{label}_ppo" / "config.json",
            "Adaptive RL": Path("outputs/adaptive_rl_runs") / f"budget_{label}_adaptive" / "config.json",
        }
        for model_name, score in scores.items():
            if model_name not in checkpoints:
                continue
            config = read_json(config_paths[model_name])
            actual = config.get("elapsed_seconds", config.get("seconds_actual", seconds))
            measured.append({
                "model": model_name, "budget_label": label,
                "requested_seconds": seconds, "actual_training_seconds": float(actual),
                "validation_normalized_rmse": score, "status": "measured",
            })
        with (root / "budget_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
            writer.writeheader()
            writer.writerows(measured)
        print(f"[budget] saved measured checkpoints and validation scores for {label}")

    if not measured:
        raise RuntimeError("No complete paired budget evaluations were produced")
    projections = list(measured)
    for model_name in dict.fromkeys(row["model"] for row in measured):
        rows = [row for row in measured if row["model"] == model_name]
        rows = sorted(rows, key=lambda row: row["actual_training_seconds"])
        for seconds, label in ((36000, "10 hours"), (864000, "10 days")):
            row = {"model": model_name, "budget_label": label,
                   "requested_seconds": seconds, "actual_training_seconds": "",
                   "validation_normalized_rmse": "", "status": "insufficient_measured_points"}
            valid = [r for r in rows if r["actual_training_seconds"] > 0
                     and r["validation_normalized_rmse"] > 0]
            if len(valid) >= 2:
                slope, intercept = np.polyfit(
                    np.log([r["actual_training_seconds"] for r in valid]),
                    np.log([r["validation_normalized_rmse"] for r in valid]), 1)
                row["validation_normalized_rmse"] = float(np.exp(intercept + slope * np.log(seconds)))
                row["status"] = "projected"
            projections.append(row)
    with (root / "budget_projection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projections[0]))
        writer.writeheader()
        writer.writerows(projections)
    print(f"\nSaved measured data: {root / 'budget_measurements.csv'}")
    print(f"Saved clearly labelled extrapolations: {root / 'budget_projection.csv'}")


if __name__ == "__main__":
    main()
