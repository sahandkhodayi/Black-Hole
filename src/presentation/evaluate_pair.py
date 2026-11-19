"""Compare the supervised dynamics MLP and PPO on identical held-out rays."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from presentation.common import (
    autoregressive_rollout,
    calculate_metrics,
    extract_ray,
    load_dataset,
    read_split_ids,
)
from presentation.model import load_dynamics_mlp


def make_mlp_predictor(model: torch.nn.Module, device: str):
    def predict(obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            action = model(tensor).squeeze(0).cpu().numpy()
        return np.asarray(action, dtype=np.float32)
    return predict


def make_ppo_predictor(model):
    def predict(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(2)
    return predict


def _save_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No metrics rows to save")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_error_bars(summary: dict, output: Path) -> None:
    names = ["MLP", "PPO"]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    u_vals = [summary[name.lower()]["mean_rmse_u"] for name in names]
    v_vals = [summary[name.lower()]["mean_rmse_v"] for name in names]
    bars_u = ax.bar(x - width / 2, u_vals, width, label="Inverse-radius error (u)")
    bars_v = ax.bar(x + width / 2, v_vals, width, label="Derivative error (v)")
    ax.set_xticks(x, names)
    ax.set_ylabel("Mean RMSE on held-out rays (lower is better)")
    ax.set_title("MLP vs PPO — same unseen photon trajectories")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (bars_u, bars_v):
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "01_mean_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_per_ray(rows: list[dict], output: Path) -> None:
    case_ids = sorted({int(row["case_id"]) for row in rows})
    x = np.arange(len(case_ids))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in (
        (axes[0], "rmse_u", "Inverse-radius prediction error"),
        (axes[1], "rmse_v", "Derivative prediction error"),
    ):
        for offset, method in ((-width / 2, "MLP"), (width / 2, "PPO")):
            lookup = {int(row["case_id"]): float(row[metric]) for row in rows if row["model"] == method}
            vals = [lookup[cid] for cid in case_ids]
            ax.bar(x + offset, vals, width, label=method)
        ax.set_xticks(x, [str(cid) for cid in case_ids])
        ax.set_xlabel("Held-out ray ID")
        ax.set_title(title)
        ax.set_ylabel("RMSE (lower is better)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Error on each test ray — no ray was used for training")
    fig.tight_layout()
    fig.savefig(output / "02_error_by_test_ray.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_example(ray, results: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    orbit_ax, u_ax, v_ax = axes
    orbit_ax.add_patch(plt.Circle((0, 0), 1.0, color="black", label="Event horizon (r=1)"))
    orbit_ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, linestyle="--", color="gray", label="Photon sphere (r=1.5)"))

    def draw_orbit(phi, u, style, label):
        full_phi = np.concatenate((-phi[:0:-1], phi))
        full_u = np.concatenate((u[:0:-1], u))
        radius = np.full_like(full_u, np.inf, dtype=np.float64)
        np.divide(1.0, full_u, out=radius, where=full_u > 1e-12)
        visible = np.isfinite(radius) & (radius <= 20.0) & (radius >= 1.0)
        orbit_ax.plot(radius[visible] * np.cos(full_phi[visible]),
                      radius[visible] * np.sin(full_phi[visible]), style, linewidth=1.7, label=label)

    draw_orbit(ray.phi, ray.u, "k-", "DOP853 reference")
    palette = {"MLP": "#168aad", "PPO": "#e07a28"}
    styles = {"MLP": "--", "PPO": ":"}
    for method, result in results.items():
        color = palette[method]
        style = styles[method]
        draw_orbit(ray.phi, result["predicted_u"], style, method)
        u_ax.plot(ray.phi, result["reference_u"], "k-", alpha=0.65, linewidth=1.2)
        u_ax.plot(ray.phi, result["predicted_u"], style, color=color, label=method, linewidth=1.7)
        v_ax.plot(ray.phi, result["reference_v"], "k-", alpha=0.65, linewidth=1.2)
        v_ax.plot(ray.phi, result["predicted_v"], style, color=color, label=method, linewidth=1.7)

    orbit_ax.set_title(f"Photon path (case {ray.case_id}, r₀={ray.r0:.2f})")
    orbit_ax.set_xlabel("x / rₛ")
    orbit_ax.set_ylabel("y / rₛ")
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.set_xlim(-15, 15)
    orbit_ax.set_ylim(-15, 15)
    orbit_ax.legend(fontsize=7, frameon=False, loc="upper right")
    u_ax.set_title("Inverse radius: u = 1/r")
    u_ax.set_xlabel("Angle φ (radians)")
    u_ax.set_ylabel("u")
    v_ax.set_title("Derivative: v = du/dφ")
    v_ax.set_xlabel("Angle φ (radians)")
    v_ax.set_ylabel("v")
    for ax in (u_ax, v_ax):
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("DOP853 reference vs learned next-state predictions")
    fig.tight_layout()
    fig.savefig(output / "03_example_trajectory_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plain_language_report(summary: dict, output: Path, example_case: int) -> None:
    mlp = summary["mlp"]
    ppo = summary["ppo"]
    winner_u = "MLP" if mlp["mean_rmse_u"] < ppo["mean_rmse_u"] else "PPO"
    winner_v = "MLP" if mlp["mean_rmse_v"] < ppo["mean_rmse_v"] else "PPO"

    def relative_difference(a: float, b: float) -> float:
        # Positive means the second method has lower error than the first.
        if a <= 0:
            return 0.0
        return (a - b) / a * 100.0

    ppo_gain_u = relative_difference(mlp["mean_rmse_u"], ppo["mean_rmse_u"])
    ppo_gain_v = relative_difference(mlp["mean_rmse_v"], ppo["mean_rmse_v"])
    comparison_lines = []
    comparison_lines.append(
        f"- For u (inverse radius), **{winner_u}** has the lower average error "
        f"({mlp['mean_rmse_u']:.6g} for MLP vs {ppo['mean_rmse_u']:.6g} for PPO)."
    )
    comparison_lines.append(
        f"- For v (change of inverse radius with angle), **{winner_v}** has the lower average error "
        f"({mlp['mean_rmse_v']:.6g} for MLP vs {ppo['mean_rmse_v']:.6g} for PPO)."
    )
    if ppo_gain_u >= 0:
        comparison_lines.append(f"- PPO's average u error is {ppo_gain_u:.1f}% lower than MLP's.")
    else:
        comparison_lines.append(f"- PPO's average u error is {abs(ppo_gain_u):.1f}% higher than MLP's.")
    if ppo_gain_v >= 0:
        comparison_lines.append(f"- PPO's average v error is {ppo_gain_v:.1f}% lower than MLP's.")
    else:
        comparison_lines.append(f"- PPO's average v error is {abs(ppo_gain_v):.1f}% higher than MLP's.")

    text = f"""# Black Hole Lab — plain-language model comparison

## What was compared?

Two models were asked to predict the next point of the same photon trajectories:

- **MLP:** a supervised neural network trained from DOP853 examples. The selected architecture is `dynamics_architecture_sweep_relu_w128_d3` (three hidden layers, 128 neurons per layer, ReLU activation).
- **PPO:** a reinforcement-learning agent trained in the project's teacher-guided environment. It receives a reward when its next predicted state is close to the DOP853 reference.
- **DOP853:** the numerical reference used to create the trajectories. It is the target for this experiment, not an exact closed-form solution.

Both learned models were tested on the same {summary['num_test_rays']} complete rays. Those ray IDs were held out from training. Evaluation was **autoregressive**: after the first point, each model receives its own previous prediction as the next input. This is important because small mistakes can build up over a long path.

## Results

{chr(10).join(comparison_lines)}

![Mean error comparison](01_mean_error_comparison.png)

![Error by test ray](02_error_by_test_ray.png)

## What does the error mean?

- `u = 1/r` is inverse distance from the black hole. Lower u means farther away.
- `v = du/dφ` describes how u changes as the angle changes.
- **RMSE** is an average error measure. It is zero for a perfect match, and smaller is better.
- The plots show whether the predicted curve follows the reference curve. A low average error does not guarantee that every part of every path looks correct.

## A simple interpretation for the class

The MLP learns from examples: it sees many current states and the next correct state, then adjusts its weights to reduce prediction error.

PPO learns by trying actions: it proposes a next state, receives a reward based on how close that state is to the reference, and gradually adjusts its policy. PPO training takes longer, so we train it before the class and load the saved result during the live demonstration.

The model with the smaller RMSE is more accurate **on this test set and for this prediction task**. This experiment does not prove that one learning method is always better, and it does not mean that either model discovered Einstein's equations.

## Example ray

The trajectory plot uses test case `{example_case}`. The solid black line is the DOP853 reference; dashed and dotted lines are the MLP and PPO predictions.

![Example trajectory](03_example_trajectory_comparison.png)

## Important limitations

- The current dataset contains only {summary['num_test_rays']} held-out test rays. Treat the result as a small experiment, not broad proof of generalization.
- Both learning methods use the same small trajectory dataset and the same state/action scaling.
- The PPO reward is teacher-guided: DOP853's next state is used inside the environment to calculate reward. The future target is not placed in the policy observation.
- This current PPO task predicts the next `(u, v)` state. It does **not** choose the numerical solver's step size.
- DOP853 is a high-accuracy numerical reference, not an analytical exact solution.

## Reproducibility

See `run_metadata.json` for checkpoint paths, test IDs, and training configuration references. Keep the dataset and split manifest unchanged when reproducing this comparison.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")


def run_comparison(
    mlp_checkpoint: str | Path,
    ppo_checkpoint: str | Path,
    dataset_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    example_case: int | None = None,
    device: str = "cpu",
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    case_ids = read_split_ids(splits_path, split)

    mlp, mlp_payload = load_dynamics_mlp(str(mlp_checkpoint), device=device)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install the RL extra first: python -m pip install -e .[rl]") from exc
    ppo = PPO.load(str(ppo_checkpoint), device="cpu")

    predictors = {
        "MLP": make_mlp_predictor(mlp, device),
        "PPO": make_ppo_predictor(ppo),
    }
    rows: list[dict] = []
    per_model_results: dict[str, dict[int, dict[str, np.ndarray]]] = {"MLP": {}, "PPO": {}}
    rays = {case_id: extract_ray(data, case_id) for case_id in case_ids}

    for method, predictor in predictors.items():
        for case_id, ray in rays.items():
            result = autoregressive_rollout(ray, predictor)
            per_model_results[method][case_id] = result
            metrics = calculate_metrics(ray, result)
            rows.append({"model": method, **metrics})

    _save_metrics_csv(output / "metrics_by_ray.csv", rows)

    summary: dict = {"split": split, "num_test_rays": len(case_ids), "test_case_ids": case_ids}
    for method in ("MLP", "PPO"):
        method_rows = [row for row in rows if row["model"] == method]
        summary[method.lower()] = {
            f"mean_{metric}": float(np.mean([float(row[metric]) for row in method_rows]))
            for metric in ("rmse_u", "rmse_v", "mae_u", "mae_v", "normalized_rmse")
        }
        summary[method.lower()]["median_rmse_u"] = float(np.median([row["rmse_u"] for row in method_rows]))
        summary[method.lower()]["median_rmse_v"] = float(np.median([row["rmse_v"] for row in method_rows]))

    if example_case is None:
        example_case = case_ids[0]
    if example_case not in rays:
        raise ValueError(f"Example case {example_case} is not in the {split!r} split: {case_ids}")

    summary["example_case_id"] = int(example_case)
    summary["mlp_architecture"] = mlp_payload.get("architecture", {})
    summary["mlp_checkpoint"] = str(Path(mlp_checkpoint))
    summary["ppo_checkpoint"] = str(Path(ppo_checkpoint))
    summary["dataset"] = str(dataset_path)
    summary["splits"] = str(splits_path)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _plot_error_bars(summary, output)
    _plot_per_ray(rows, output)
    _plot_example(
        rays[example_case],
        {method: per_model_results[method][example_case] for method in ("MLP", "PPO")},
        output,
    )
    _plain_language_report(summary, output, example_case)

    metadata = {
        "mlp_checkpoint": str(Path(mlp_checkpoint)),
        "ppo_checkpoint": str(Path(ppo_checkpoint)),
        "dataset": str(dataset_path),
        "splits": str(splits_path),
        "split_evaluated": split,
        "test_case_ids": case_ids,
        "evaluation_mode": "autoregressive; each model receives its own previous prediction",
        "reference": "DOP853 data; future reference u/v are used for scoring, not policy observations",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


def select_ppo_checkpoint(
    candidates: dict[str, str | Path],
    dataset_path: str | Path,
    splits_path: str | Path,
    output_json: str | Path,
) -> Path:
    """Choose PPO checkpoint using validation rollout error, never test error."""
    from stable_baselines3 import PPO

    data = load_dataset(dataset_path)
    validation_ids = read_split_ids(splits_path, "validation")
    scores = []
    for label, raw_path in candidates.items():
        path = Path(raw_path)
        if not path.is_file():
            continue
        model = PPO.load(str(path), device="cpu")
        per_ray = []
        predictor = make_ppo_predictor(model)
        for case_id in validation_ids:
            ray = extract_ray(data, case_id)
            result = autoregressive_rollout(ray, predictor)
            per_ray.append(calculate_metrics(ray, result))
        scores.append({
            "label": label,
            "checkpoint": str(path),
            "mean_validation_normalized_rmse": float(
                np.mean([row["normalized_rmse"] for row in per_ray])
            ),
            "mean_validation_rmse_u": float(np.mean([row["rmse_u"] for row in per_ray])),
            "mean_validation_rmse_v": float(np.mean([row["rmse_v"] for row in per_ray])),
            "validation_case_ids": validation_ids,
        })

    if not scores:
        raise FileNotFoundError("None of the candidate PPO checkpoints exists")
    selected = min(scores, key=lambda item: item["mean_validation_normalized_rmse"])
    report = {
        "selection_metric": "mean autoregressive normalized RMSE on validation rays (lower is better)",
        "validation_case_ids": validation_ids,
        "candidates": scores,
        "selected": selected,
        "test_split_used_for_selection": False,
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return Path(selected["checkpoint"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp", required=True, help="Path to presentation MLP best_model.pt")
    parser.add_argument("--ppo", required=True, help="Path to PPO best_model.zip or final_model.zip")
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--example-case", type=int, default=None)
    parser.add_argument("--output", default="outputs/presentation/final_comparison")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    summary = run_comparison(
        args.mlp, args.ppo, args.dataset, args.splits, args.output,
        split=args.split, example_case=args.example_case, device=args.device,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nSaved plots and plain-language report to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()