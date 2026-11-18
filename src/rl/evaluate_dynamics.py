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

    fig, (orbit_ax, u_ax, v_ax) = plt.subplots(1, 3, figsize=(17, 5.8))
    orbit_ax.add_patch(Circle((0, 0), 1, color="black", label="Event horizon"))
    orbit_ax.add_patch(Circle((0, 0), 1.5, fill=False, linestyle="--",
                              color="black", label="Photon sphere"))
    rows = []

    for case_id in case_ids:
        result = rollout(data, case_id, model, step_scale, args.device)
        phi = result["phi"]
        ru, rv = result["reference_u"], result["reference_v"]
        pu, pv = result["predicted_u"], result["predicted_v"]
        rmse_u = float(np.sqrt(np.mean((pu - ru) ** 2)))
        rmse_v = float(np.sqrt(np.mean((pv - rv) ** 2)))
        rows.append({
            "case_id": case_id,
            "r0": float(data["r0"][{int(cid): i for i, cid in enumerate(data["case_id"])}[int(case_id)]]),
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
        })

        full_phi = np.concatenate((-phi[:0:-1], phi))
        for values, style in ((ru, "-"), (pu, "--")):
            full_u = np.concatenate((values[:0:-1], values))
            radius = np.full_like(full_u, np.inf)
            np.divide(1.0, full_u, out=radius, where=full_u > 0)
            visible = (full_u > 0) & (radius <= 30)
            orbit_ax.plot(radius[visible] * np.cos(full_phi[visible]),
                          radius[visible] * np.sin(full_phi[visible]),
                          style, linewidth=1.6, color=f"C{len(rows)-1}",
                          label=f"{'DOP853' if style == '-' else 'MLP'} {case_id}")
        color = f"C{len(rows)-1}"
        u_ax.plot(phi, ru, "-", color=color, linewidth=1.5)
        u_ax.plot(phi, pu, "--", color=color, linewidth=1.5)
        v_ax.plot(phi, rv, "-", color=color, linewidth=1.5)
        v_ax.plot(phi, pv, "--", color=color, linewidth=1.5)

    mean_u = float(np.mean([row["rmse_u"] for row in rows]))
    mean_v = float(np.mean([row["rmse_v"] for row in rows]))
    orbit_ax.set(title="Photon orbit: DOP853 vs supervised MLP",
                 xlabel="x / r_s", ylabel="y / r_s", xlim=(-15, 15), ylim=(-15, 15))
    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.legend(fontsize=7, ncol=2)
    u_ax.set(title=f"Inverse radius (mean RMSE={mean_u:.4f})",
             xlabel="phi (radians)", ylabel="u = 1/r")
    v_ax.set(title=f"Radial derivative (mean RMSE={mean_v:.4f})",
             xlabel="phi (radians)", ylabel="v = du/dphi")
    for axis in (u_ax, v_ax):
        axis.grid(alpha=0.25)
    fig.suptitle(f"Supervised MLP autoregressive rollout on {args.split} rays")
    fig.tight_layout()
    figure = out / "trajectory_comparison.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
