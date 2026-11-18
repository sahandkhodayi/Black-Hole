"""Autoregressively evaluate the supervised dynamics model on held-out rays."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from rl.dynamics_model import (
    load_checkpoint,
    load_packed_dataset,
    make_transition_arrays,
    predict_next,
)


def read_ids(path, key):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [int(value) for value in manifest.get(key, [])]
    if not ids:
        raise ValueError(f"Split {key!r} is empty")
    return ids


def rollout(data, case_id, model, step_scale, device):
    positions = {int(value): i for i, value in enumerate(data["case_id"])}
    pos = positions[int(case_id)]
    start, end = map(int, data["offsets"][pos:pos + 2])
    phi = data["phi"][start:end].astype(float)
    ref_u = data["u"][start:end].astype(float)
    ref_v = data["v"][start:end].astype(float)
    r0 = float(data["r0"][pos])
    pred_u = [float(ref_u[0])]
    pred_v = [float(ref_v[0])]

    for index in range(len(phi) - 1):
        next_state = predict_next(
            model,
            (pred_u[-1], pred_v[-1]),
            r0=r0,
            phi=float(phi[index]),
            delta_phi=float(phi[index + 1] - phi[index]),
            step_scale=step_scale,
            device=device,
        )
        pred_u.append(float(next_state[0]))
        pred_v.append(float(next_state[1]))

    return {
        "phi": phi,
        "reference_u": ref_u,
        "reference_v": ref_v,
        "predicted_u": np.asarray(pred_u),
        "predicted_v": np.asarray(pred_v),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    data = load_packed_dataset(args.dataset)
    case_ids = read_ids(args.splits, args.split)
    # The checkpoint stores the exact angular scale used during training.
    model, step_scale, _ = load_checkpoint(args.model, device=args.device)
    out = Path(args.output) if args.output else Path(args.model).parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    # Keep every test ray in the metrics, but show only three representative
    # radii in the presentation figure to avoid an unreadable legend.
    positions = {int(value): i for i, value in enumerate(data["case_id"])}
    ordered_ids = sorted(case_ids, key=lambda cid: float(data["r0"][positions[int(cid)]]))
    if len(ordered_ids) <= 3:
        plot_ids = ordered_ids
    else:
        picks = np.linspace(0, len(ordered_ids) - 1, 3).round().astype(int)
        plot_ids = [ordered_ids[index] for index in sorted(set(picks.tolist()))]

    fig, (orbit_ax, u_ax, v_ax) = plt.subplots(
        1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    orbit_ax.add_patch(Circle((0, 0), 1, color="black", label="Event horizon"))
    orbit_ax.add_patch(Circle((0, 0), 1.5, fill=False, linestyle=":",
                              linewidth=1.8, color="black", label="Photon sphere"))
    rows = []

    for case_id in case_ids:
        result = rollout(data, case_id, model, step_scale, args.device)
        phi = result["phi"]
        ru, rv = result["reference_u"], result["reference_v"]
        pu, pv = result["predicted_u"], result["predicted_v"]
        rmse_u = float(np.sqrt(np.mean((pu - ru) ** 2)))
        rmse_v = float(np.sqrt(np.mean((pv - rv) ** 2)))
        pos = positions[int(case_id)]
        rows.append({
            "case_id": case_id,
            "r0": float(data["r0"][pos]),
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
        })

        if case_id not in plot_ids:
            continue
        color = f"C{plot_ids.index(case_id)}"
        full_phi = np.concatenate((-phi[:0:-1], phi))
        for values, style in ((ru, "-"), (pu, "--")):
            full_u = np.concatenate((values[:0:-1], values))
            radius = np.full_like(full_u, np.inf)
            np.divide(1.0, full_u, out=radius, where=full_u > 0)
            visible = (full_u > 0) & (radius <= 30)
            orbit_ax.plot(
                radius[visible] * np.cos(full_phi[visible]),
                radius[visible] * np.sin(full_phi[visible]),
                style, linewidth=2.2, color=color,
            )
        u_ax.plot(phi, ru, "-", color=color, linewidth=2.0)
        u_ax.plot(phi, pu, "--", color=color, linewidth=2.0)
        v_ax.plot(phi, rv, "-", color=color, linewidth=2.0)
        v_ax.plot(phi, pv, "--", color=color, linewidth=2.0)

    from matplotlib.lines import Line2D
    ray_handles = [
        Line2D([0], [0], color=f"C{i}", linewidth=2.4,
               label=f"Ray r₀={float(data['r0'][positions[int(cid)]]):.1f} rₛ")
        for i, cid in enumerate(plot_ids)
    ]
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label="DOP853 reference"),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="--", label="AI prediction"),
    ]
    orbit_ax.set(
        title="Photon paths around the black hole",
        xlabel="x / rₛ", ylabel="y / rₛ",
        xlim=(-15, 15), ylim=(-15, 15),
    )
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.legend(handles=[
        Line2D([0], [0], color="black", linewidth=5, label="Event horizon"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle=":", label="Photon sphere"),
        *ray_handles, *style_handles,
    ], fontsize=8, loc="upper right", framealpha=0.95)
    u_ax.set(title="Inverse radius", xlabel="φ (radians)", ylabel="u = 1/r")
    v_ax.set(title="Radial derivative", xlabel="φ (radians)", ylabel="du/dφ")
    for axis in (u_ax, v_ax):
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    orbit_ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Learned photon dynamics — unseen {args.split} trajectories",
        fontsize=16, fontweight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Solid: numerical reference     Dashed: neural-network rollout",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure = out / "trajectory_comparison.png"
    fig.savefig(figure, dpi=240, bbox_inches="tight")
    plt.close(fig)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Aggregate per-ray rollout errors only after every selected ray has
    # been evaluated. These are test-set diagnostics, not training targets.
    mean_u = float(np.mean([row["rmse_u"] for row in rows]))
    mean_v = float(np.mean([row["rmse_v"] for row in rows]))
    summary = {
        "model": "supervised MLP",
        "split": args.split,
        "num_rays": len(rows),
        "mean_rmse_u": mean_u,
        "mean_rmse_v": mean_v,
        "figure": str(figure),
        "metrics": str(out / "metrics.csv"),
        "rollout": "autoregressive; each predicted state is fed into the next step",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
