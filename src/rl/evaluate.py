"""Evaluate a PPO policy on held-out rays and save comparison plots/metrics."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from rl.environment import PhotonTrajectoryEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    split = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    case_ids = [int(x) for x in split[args.split]]
    env, model = PhotonTrajectoryEnv(args.dataset, case_ids, randomize_cases=False), PPO.load(args.model)
    out = Path(args.output) if args.output else Path(args.model).parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, (orbit_ax, state_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
    orbit_ax.add_patch(Circle((0, 0), 1, color="black", label="Event horizon"))
    orbit_ax.add_patch(Circle((0, 0), 1.5, fill=False, linestyle="--", label="Photon sphere"))
    for case_id in case_ids:
        obs, reset_info = env.reset(options={"case_id": case_id})
        ref_u, ref_v = [float(env.target_u[0])], [float(env.target_v[0])]
        pred_u, pred_v = [float(env.u)], [float(env.v)]
        phis, done = [float(env.phi[0])], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            ref_u.append(info["reference_u"]); ref_v.append(info["reference_v"])
            pred_u.append(info["predicted_u"]); pred_v.append(info["predicted_v"])
            phis.append(info["phi"])
        ref_u, pred_u = np.asarray(ref_u), np.asarray(pred_u)
        ref_v, pred_v, phi = np.asarray(ref_v), np.asarray(pred_v), np.asarray(phis)
        rows.append({"case_id": case_id, "r0": reset_info["r0"], "samples": len(phi),
                     "rmse_u": float(np.sqrt(np.mean((pred_u-ref_u)**2))),
                     "rmse_v": float(np.sqrt(np.mean((pred_v-ref_v)**2))),
                     "escape_angle_reference": float(phi[-1])})
        full_phi = np.concatenate((-phi[:0:-1], phi))
        for values, style, label in (
            (np.concatenate((ref_u[:0:-1], ref_u)), "-", f"reference r0={reset_info['r0']:.2f}"),
            (np.concatenate((pred_u[:0:-1], pred_u)), "--", f"PPO r0={reset_info['r0']:.2f}")):
            radius = np.full_like(values, np.inf)
            np.divide(1.0, values, out=radius, where=values > 0)
            visible = (values > 0) & (radius <= 30)
            orbit_ax.plot(radius[visible]*np.cos(full_phi[visible]),
                          radius[visible]*np.sin(full_phi[visible]), style, linewidth=1.3, label=label)
        state_ax.plot(phi, ref_u, "-", alpha=0.5)
        state_ax.plot(phi, pred_u, "--", label=f"case {case_id}")
    orbit_ax.set(title="DOP853 reference (solid) vs PPO (dashed)", xlabel="x / r_s", ylabel="y / r_s")
    orbit_ax.set_aspect("equal"); orbit_ax.set_xlim(-15, 15); orbit_ax.set_ylim(-15, 15)
    orbit_ax.legend(fontsize=7, ncol=2)
    state_ax.set(title="Inverse-radius state tracking", xlabel="phi (radians)", ylabel="u = 1/r")
    state_ax.grid(alpha=0.25); state_ax.legend(fontsize=8)
    fig.tight_layout()
    figure = out / "trajectory_comparison.png"
    fig.savefig(figure, dpi=180); plt.close(fig)
    metrics = out / "metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"split": args.split, "num_rays": len(rows),
               "mean_rmse_u": float(np.mean([r["rmse_u"] for r in rows])),
               "mean_rmse_v": float(np.mean([r["rmse_v"] for r in rows])),
               "figure": str(figure), "metrics": str(metrics)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
