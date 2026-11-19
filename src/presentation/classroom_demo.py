"""Fast classroom demo: load saved MLP and PPO policies; do not train live."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401  (kept: required by the model loaders)

from presentation.common import autoregressive_rollout, calculate_metrics, extract_ray, load_dataset, read_split_ids
from presentation.evaluate_pair import make_mlp_predictor, make_ppo_predictor
from presentation.model import load_dynamics_mlp


def use_interactive_backend() -> bool:
    """Try to switch matplotlib to a GUI backend.

    Must be called BEFORE any figure is created. Some modules in the project
    may force the non-interactive "Agg" backend, which is why plt.show() used
    to do nothing and print a FigureCanvasAgg warning.
    """
    for name in ("QtAgg", "TkAgg", "MacOSX"):
        try:
            plt.switch_backend(name)
            return True
        except Exception:
            continue
    return False


def open_file(path: Path) -> None:
    """Open the saved image with the operating system's default viewer."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:  # never crash the demo just because of the viewer
        print(f"Could not open the image automatically: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp", default="outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3/best_model.pt")
    parser.add_argument("--ppo", default="outputs/presentation/class_presentation/selected_ppo_model.zip")
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--case-id", type=int, default=None, help="Defaults to the first held-out test ray")
    parser.add_argument("--output", default="outputs/presentation/classroom_demo.png")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    # Pick the backend first, before any figure exists.
    can_show = args.show and use_interactive_backend()

    for path in (args.mlp, args.ppo, args.dataset, args.splits):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Train both models first with: python -m presentation.run_parallel"
            )

    data = load_dataset(args.dataset)
    test_ids = read_split_ids(args.splits, "test")
    case_id = test_ids[0] if args.case_id is None else args.case_id
    if case_id not in test_ids:
        raise ValueError(f"case-id {case_id} is not in held-out test IDs {test_ids}")
    ray = extract_ray(data, case_id)

    mlp, _ = load_dynamics_mlp(args.mlp, device=args.device)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install RL dependencies: python -m pip install -e .[rl]") from exc
    ppo = PPO.load(args.ppo, device="cpu")

    # The policies are already trained. This only runs one short test trajectory.
    results = {
        "MLP": autoregressive_rollout(ray, make_mlp_predictor(mlp, args.device)),
        "PPO": autoregressive_rollout(ray, make_ppo_predictor(ppo)),
    }
    metrics = {name: calculate_metrics(ray, result) for name, result in results.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    orbit_ax, u_ax, v_ax = axes
    orbit_ax.add_patch(plt.Circle((0, 0), 1.0, color="black", label="Event horizon"))
    orbit_ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, linestyle="--", color="gray", label="Photon sphere"))

    def draw_orbit(phi, u, style, label, color):
        full_phi = np.concatenate((-phi[:0:-1], phi))
        full_u = np.concatenate((u[:0:-1], u))
        radius = np.full_like(full_u, np.inf, dtype=np.float64)
        np.divide(1.0, full_u, out=radius, where=full_u > 1e-12)
        valid = np.isfinite(radius) & (radius >= 1.0) & (radius <= 20.0)
        orbit_ax.plot(radius[valid] * np.cos(full_phi[valid]), radius[valid] * np.sin(full_phi[valid]),
                      style, color=color, linewidth=1.8, label=label)

    draw_orbit(ray.phi, ray.u, "-", "DOP853 reference", "black")
    colors = {"MLP": "#168aad", "PPO": "#e07a28"}
    styles = {"MLP": "--", "PPO": ":"}
    for method, result in results.items():
        color, style = colors[method], styles[method]
        draw_orbit(ray.phi, result["predicted_u"], style, method, color)
        u_ax.plot(ray.phi, ray.u, "k-", linewidth=1.2, alpha=0.55)
        u_ax.plot(ray.phi, result["predicted_u"], style, color=color, linewidth=1.8, label=method)
        v_ax.plot(ray.phi, ray.v, "k-", linewidth=1.2, alpha=0.55)
        v_ax.plot(ray.phi, result["predicted_v"], style, color=color, linewidth=1.8, label=method)

    orbit_ax.set_title(f"One photon path (r₀={ray.r0:.2f} rₛ)")
    orbit_ax.set_xlabel("x / rₛ")
    orbit_ax.set_ylabel("y / rₛ")
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.set_xlim(-15, 15)
    orbit_ax.set_ylim(-15, 15)
    orbit_ax.legend(fontsize=7, frameon=False)
    u_ax.set_title("u = 1/r")
    u_ax.set_xlabel("Angle φ (radians)")
    u_ax.set_ylabel("Inverse radius")
    u_ax.legend(frameon=False)
    v_ax.set_title("v = du/dφ")
    v_ax.set_xlabel("Angle φ (radians)")
    v_ax.set_ylabel("Derivative")
    v_ax.legend(frameon=False)
    for ax in (u_ax, v_ax):
        ax.grid(alpha=0.25)
    fig.suptitle(f"Black Hole Lab — classroom comparison | test ray {case_id}")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print("\nBLACK HOLE LAB — CLASSROOM DEMO")
    print(f"Test ray: {case_id} | closest approach r0 = {ray.r0:.4f} r_s | samples = {len(ray.phi)}")
    print("No training is running. Both saved models were loaded and evaluated on the same ray.")
    print("\nRMSE (lower is better)")
    print(f"{'Model':<8} {'u = 1/r':>14} {'v = du/dphi':>16}")
    for method in ("MLP", "PPO"):
        row = metrics[method]
        print(f"{method:<8} {row['rmse_u']:>14.6f} {row['rmse_v']:>16.6f}")
    print(f"\nSaved figure: {output.resolve()}")

    if args.show:
        if can_show:
            plt.show()
        else:
            # No GUI backend available: open the saved PNG instead of warning.
            plt.close(fig)
            print("No interactive matplotlib backend found; opening the saved image instead.")
            open_file(output)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()