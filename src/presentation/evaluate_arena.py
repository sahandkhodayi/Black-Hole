"""Precompute fair, saved-checkpoint comparisons for the browser Model Arena.

All learned models are evaluated on the same held-out complete rays. The MLP
and state-prediction PPO use autoregressive rollouts. The PINN predicts u(phi)
and its derivative directly. Adaptive PPO integrates the ODE with policy-chosen
RK4 steps, then is sampled on the same reference phi grid for scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from presentation.common import (
    autoregressive_rollout, calculate_metrics, extract_ray, load_dataset, read_split_ids,
)
from presentation.evaluate_pair import make_mlp_predictor, make_ppo_predictor
from presentation.model import load_dynamics_mlp
from presentation.pinn import load_pinn, predict_pinn
from rl.adaptive_environment import AdaptivePhotonEnv


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows to write: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(ray, pred_u, pred_v):
    result = {"predicted_u": np.asarray(pred_u), "predicted_v": np.asarray(pred_v)}
    return calculate_metrics(ray, result)


def _adaptive_rollout(model, env, case_id):
    obs, _ = env.reset(options={"case_id": int(case_id)})
    phis = [float(env.phi_now)]
    us = [float(env.u)]
    vs = [float(env.v)]
    steps = []
    policy_seconds = 0.0
    terminated = False
    while not terminated:
        started = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        policy_seconds += time.perf_counter() - started
        obs, _reward, terminated, _truncated, info = env.step(action)
        phis.append(float(info["phi"]))
        us.append(float(info["predicted_u"]))
        vs.append(float(info["predicted_v"]))
        steps.append(float(info["step_size"]))
        if len(steps) > 100000:
            raise RuntimeError("Adaptive policy exceeded 100,000 steps; possible non-progressing policy")
    return (np.asarray(phis), np.asarray(us), np.asarray(vs),
            len(steps), 4 * len(steps), policy_seconds)


def _plot_arena(prediction_rows, summary_rows, output, example_case):
    models = [row["model"] for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), 1.0, color="black", label="Event horizon"))
    ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, linestyle="--", color="gray", label="Photon sphere"))
    styles = {"MLP": ("--", "#168aad"), "PINN": ("-.", "#9b5de5"),
              "PPO": (":", "#e07a28"), "Adaptive RL": ("-", "#2a9d8f")}
    example = [row for row in prediction_rows if int(row["case_id"]) == int(example_case)]
    reference = [row for row in example if row["model"] == "DOP853"]
    if reference:
        rows = sorted(reference, key=lambda item: float(item["phi"]))
        phi = np.asarray([float(row["phi"]) for row in rows])
        u = np.asarray([float(row["reference_u"]) for row in rows])
        radius = np.divide(1.0, u, out=np.full_like(u, np.inf), where=u > 1e-12)
        valid = np.isfinite(radius) & (radius >= 1) & (radius <= 20)
        ax.plot(radius[valid] * np.cos(phi[valid]), radius[valid] * np.sin(phi[valid]),
                color="black", linewidth=2.2, label="DOP853 reference")
    for model in models:
        rows = sorted([row for row in example if row["model"] == model],
                      key=lambda item: float(item["phi"]))
        if not rows:
            continue
        phi = np.asarray([float(row["phi"]) for row in rows])
        u = np.asarray([float(row["predicted_u"]) for row in rows])
        radius = np.divide(1.0, u, out=np.full_like(u, np.inf), where=u > 1e-12)
        valid = np.isfinite(radius) & (radius >= 1) & (radius <= 20)
        style, color = styles.get(model, ("--", None))
        ax.plot(radius[valid] * np.cos(phi[valid]), radius[valid] * np.sin(phi[valid]),
                linestyle=style, color=color, linewidth=1.7, label=model)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(-15, 15), ylim=(-15, 15), xlabel="x / rₛ", ylabel="y / rₛ",
           title=f"Same held-out photon ray · case {example_case}")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    x = np.arange(len(summary_rows))
    values = [float(row["mean_normalized_rmse"]) for row in summary_rows]
    bars = ax.bar(x, values, color=[styles.get(name, ("", "#777"))[1] for name in models])
    ax.set_xticks(x, models)
    ax.set_ylabel("Mean normalized RMSE (lower is better)")
    ax.set_title("Held-out test accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "model_arena_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_arena(*, dataset_path, splits_path, mlp_path, ppo_path, pinn_path,
              adaptive_ppo_path=None, output_dir="outputs/presentation/model_arena",
              split="test", device="cpu", example_case=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    case_ids = read_split_ids(splits_path, split)
    if split != "test":
        raise ValueError("The public Model Arena only exports the held-out test split")
    rays = {cid: extract_ray(data, cid) for cid in case_ids}

    mlp, _ = load_dynamics_mlp(str(mlp_path), device=device)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install RL dependencies with python -m pip install -e '.[rl]'") from exc
    ppo = PPO.load(str(ppo_path), device="cpu")
    pinn, _ = load_pinn(pinn_path, device=device)
    adaptive = PPO.load(str(adaptive_ppo_path), device="cpu") if adaptive_ppo_path else None

    methods = {
        "MLP": lambda ray: autoregressive_rollout(ray, make_mlp_predictor(mlp, device)),
        "PPO": lambda ray: autoregressive_rollout(ray, make_ppo_predictor(ppo)),
    }
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    adaptive_env = AdaptivePhotonEnv(dataset_path, case_ids, randomize_cases=False) if adaptive else None

    for case_id, ray in rays.items():
        results = {}
        for name, fn in methods.items():
            started = time.perf_counter()
            result = fn(ray)
            elapsed = time.perf_counter() - started
            results[name] = result
            metrics = _metrics(ray, result["predicted_u"], result["predicted_v"])
            metric_rows.append({"model": name, **metrics,
                                "inference_seconds": elapsed,
                                "integration_steps": len(ray.phi) - 1,
                                "function_evaluations": None,
                                "checkpoint": str(mlp_path if name == "MLP" else ppo_path)})
        started = time.perf_counter()
        pinn_u, pinn_v = predict_pinn(pinn, ray.r0, ray.phi, device=device)
        pinn_elapsed = time.perf_counter() - started
        results["PINN"] = {"predicted_u": pinn_u, "predicted_v": pinn_v}
        metrics = _metrics(ray, pinn_u, pinn_v)
        metric_rows.append({"model": "PINN", **metrics,
                            "inference_seconds": pinn_elapsed,
                            "integration_steps": len(ray.phi) - 1,
                            "function_evaluations": None, "checkpoint": str(pinn_path)})

        if adaptive is not None:
            adaptive_env.reset(options={"case_id": int(case_id)})
            started = time.perf_counter()
            phis, us, vs, steps, nfev, policy_seconds = _adaptive_rollout(
                adaptive, adaptive_env, case_id)
            elapsed = time.perf_counter() - started
            pred_u = np.interp(ray.phi, phis, us)
            pred_v = np.interp(ray.phi, phis, vs)
            results["Adaptive RL"] = {"predicted_u": pred_u, "predicted_v": pred_v}
            metrics = _metrics(ray, pred_u, pred_v)
            metric_rows.append({"model": "Adaptive RL", **metrics,
                                "inference_seconds": policy_seconds,
                                "integration_steps": steps,
                                "function_evaluations": nfev,
                                "wall_seconds_including_integration": elapsed,
                                "checkpoint": str(adaptive_ppo_path)})

        for model_name, result in results.items():
            for i, phi in enumerate(ray.phi):
                prediction_rows.append({
                    "case_id": case_id, "r0": ray.r0, "phi": float(phi),
                    "model": model_name, "reference_u": float(ray.u[i]),
                    "reference_v": float(ray.v[i]),
                    "predicted_u": float(result["predicted_u"][i]),
                    "predicted_v": float(result["predicted_v"][i]),
                    "absolute_error_u": abs(float(result["predicted_u"][i] - ray.u[i])),
                    "absolute_error_v": abs(float(result["predicted_v"][i] - ray.v[i])),
                })
        for i, phi in enumerate(ray.phi):
            prediction_rows.append({
                "case_id": case_id, "r0": ray.r0, "phi": float(phi),
                "model": "DOP853", "reference_u": float(ray.u[i]),
                "reference_v": float(ray.v[i]), "predicted_u": float(ray.u[i]),
                "predicted_v": float(ray.v[i]), "absolute_error_u": 0.0,
                "absolute_error_v": 0.0,
            })

    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "metrics_by_ray.csv", metric_rows)
    summary_rows = []
    for name in dict.fromkeys(row["model"] for row in metric_rows):
        rows = [row for row in metric_rows if row["model"] == name]
        summary_rows.append({
            "model": name,
            "mean_rmse_u": float(np.mean([row["rmse_u"] for row in rows])),
            "mean_rmse_v": float(np.mean([row["rmse_v"] for row in rows])),
            "mean_normalized_rmse": float(np.mean([row["normalized_rmse"] for row in rows])),
            "worst_ray_normalized_rmse": float(np.max([row["normalized_rmse"] for row in rows])),
            "mean_inference_ms": float(1000 * np.mean([row["inference_seconds"] for row in rows])),
            "mean_integration_steps": float(np.mean([row["integration_steps"] for row in rows])),
            "mean_function_evaluations": (
                float(np.mean([row["function_evaluations"] for row in rows
                               if row["function_evaluations"] is not None]))
                if any(row["function_evaluations"] is not None for row in rows) else None),
        })
    _write_csv(output / "summary.csv", summary_rows)
    if example_case is None:
        example_case = case_ids[0]
    if int(example_case) not in rays:
        raise ValueError(f"example-case {example_case} is not in the held-out test split")
    metadata = {
        "schema_version": 1, "split": split, "test_case_ids": case_ids,
        "dataset": str(dataset_path), "splits": str(splits_path),
        "checkpoints": {"MLP": str(mlp_path), "PPO": str(ppo_path),
                        "PINN": str(pinn_path), "Adaptive RL": str(adaptive_ppo_path) if adaptive_ppo_path else None},
        "fairness": {
            "same_held_out_complete_rays": True,
            "same_reference_phi_samples_for_scoring": True,
            "mlp_and_ppo": "autoregressive; previous model output is fed back",
            "pinn": "direct u(phi) function and autograd derivative; no trajectory labels in PINN loss",
            "adaptive_rl": "policy selects RK4 step sizes; interpolated to the reference phi grid for scoring",
            "reference": "DOP853 numerical solution, not a closed-form exact solution",
        },
        "live_training": False,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _plot_arena(prediction_rows, summary_rows, output, example_case)
    print(json.dumps({"output": str(output), "models": summary_rows}, indent=2))
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--mlp", required=True)
    parser.add_argument("--ppo", required=True)
    parser.add_argument("--pinn", required=True)
    parser.add_argument("--adaptive-ppo", default=None)
    parser.add_argument("--output", default="outputs/presentation/model_arena")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--example-case", type=int, default=None)
    args = parser.parse_args()
    run_arena(dataset_path=args.dataset, splits_path=args.splits,
              mlp_path=args.mlp, ppo_path=args.ppo, pinn_path=args.pinn,
              adaptive_ppo_path=args.adaptive_ppo, output_dir=args.output,
              device=args.device, example_case=args.example_case)


if __name__ == "__main__":
    main()
