"""Evaluate a PPO policy on complete rays and save presentation-ready diagnostics.

The policy is rolled forward autoregressively: each predicted state becomes the
current state for the next decision. DOP853 targets are used only for metrics
and reward during environment interaction, never as policy observations.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from stable_baselines3 import PPO

from rl.environment import PhotonTrajectoryEnv


def evaluate_case(env, model, case_id):
    """Roll out one ray and return reference/predicted states on the same grid."""
    obs, reset_info = env.reset(options={"case_id": case_id})
    ref_u = [float(env.target_u[0])]
    ref_v = [float(env.target_v[0])]
    pred_u = [float(env.u)]
    pred_v = [float(env.v)]
    phis = [float(env.phi[0])]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ref_u.append(info["reference_u"])
        ref_v.append(info["reference_v"])
        pred_u.append(info["predicted_u"])
        pred_v.append(info["predicted_v"])
        phis.append(info["phi"])

    arrays = {
        "phi": np.asarray(phis),
        "reference_u": np.asarray(ref_u),
        "reference_v": np.asarray(ref_v),
        "predicted_u": np.asarray(pred_u),
        "predicted_v": np.asarray(pred_v),
    }
    return reset_info, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    split_manifest = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    if args.split not in split_manifest or not split_manifest[args.split]:
        raise ValueError(f"Split {args.split!r} is missing or empty in {args.splits}")
    case_ids = [int(value) for value in split_manifest[args.split]]
    env = PhotonTrajectoryEnv(args.dataset, case_ids, randomize_cases=False)
    model = PPO.load(args.model)
    out = Path(args.output) if args.output else Path(args.model).parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, (orbit_ax, u_ax, v_ax) = plt.subplots(1, 3, figsize=(17, 5.8))
    orbit_ax.add_patch(Circle((0, 0), 1, color="black", label="Event horizon"))
    orbit_ax.add_patch(Circle((0, 0), 1.5, fill=False, linestyle="--",
                              color="black", label="Photon sphere"))

    for case_id in case_ids:
        reset_info, result = evaluate_case(env, model, case_id)
        phi = result["phi"]
        ref_u, ref_v = result["reference_u"], result["reference_v"]
        pred_u, pred_v = result["predicted_u"], result["predicted_v"]
        rmse_u = float(np.sqrt(np.mean((pred_u - ref_u) ** 2)))
        rmse_v = float(np.sqrt(np.mean((pred_v - ref_v) ** 2)))
        rows.append({
            "case_id": case_id,
            "r0": reset_info["r0"],
            "samples": len(phi),
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
            "escape_angle_reference": float(phi[-1]),
        })

        # Mirror the incoming half of the orbit to show the full scattering path.
        full_phi = np.concatenate((-phi[:0:-1], phi))
        ref_full_u = np.concatenate((ref_u[:0:-1], ref_u))
        pred_full_u = np.concatenate((pred_u[:0:-1], pred_u))
        radius_ref = np.full_like(ref_full_u, np.inf)
        radius_pred = np.full_like(pred_full_u, np.inf)
        np.divide(1.0, ref_full_u, out=radius_ref, where=ref_full_u > 0)
        np.divide(1.0, pred_full_u, out=radius_pred, where=pred_full_u > 0)
        visible_ref = (ref_full_u > 0) & (radius_ref <= 30)
        visible_pred = (pred_full_u > 0) & (radius_pred <= 30)

        ref_line, = orbit_ax.plot(
            radius_ref[visible_ref] * np.cos(full_phi[visible_ref]),
            radius_ref[visible_ref] * np.sin(full_phi[visible_ref]),
            "-", linewidth=1.5, label=f"DOP853, case {case_id}",
        )
        color = ref_line.get_color()
        orbit_ax.plot(
            radius_pred[visible_pred] * np.cos(full_phi[visible_pred]),
            radius_pred[visible_pred] * np.sin(full_phi[visible_pred]),
            "--", linewidth=1.5, color=color, label=f"PPO, case {case_id}",
        )
        u_ax.plot(phi, ref_u, "-", linewidth=1.5, color=color,
                  label=f"Reference {case_id}")
        u_ax.plot(phi, pred_u, "--", linewidth=1.5, color=color,
                  label=f"PPO {case_id}")
        v_ax.plot(phi, ref_v, "-", linewidth=1.5, color=color,
                  label=f"Reference {case_id}")
        v_ax.plot(phi, pred_v, "--", linewidth=1.5, color=color,
                  label=f"PPO {case_id}")

    mean_u = float(np.mean([row["rmse_u"] for row in rows]))
    mean_v = float(np.mean([row["rmse_v"] for row in rows]))
    orbit_ax.set(title="Photon orbit: DOP853 vs PPO",
                 xlabel="x / r_s", ylabel="y / r_s",
                 xlim=(-15, 15), ylim=(-15, 15))
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.legend(fontsize=6, ncol=2)
    u_ax.set(title=f"Inverse radius (mean RMSE={mean_u:.4f})",
             xlabel="phi (radians)", ylabel="u = 1/r")
    v_ax.set(title=f"Radial derivative (mean RMSE={mean_v:.4f})",
             xlabel="phi (radians)", ylabel="v = du/dphi")
    for axis in (u_ax, v_ax):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=2)
    fig.suptitle(f"PPO autoregressive rollout on {args.split} rays", fontsize=14)
    fig.tight_layout()
    figure = out / "trajectory_comparison.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics_path = out / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "num_rays": len(rows),
        "mean_rmse_u": mean_u,
        "mean_rmse_v": mean_v,
        "median_rmse_u": float(np.median([row["rmse_u"] for row in rows])),
        "median_rmse_v": float(np.median([row["rmse_v"] for row in rows])),
        "figure": str(figure),
        "metrics": str(metrics_path),
        "interpretation": "PPO is evaluated autoregressively; DOP853 is the reference.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
