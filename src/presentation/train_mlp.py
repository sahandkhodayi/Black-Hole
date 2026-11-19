"""Train the selected supervised dynamics MLP on the project's ray dataset."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from presentation.common import (
    autoregressive_rollout,
    build_supervised_examples,
    calculate_metrics,
    extract_ray,
    load_dataset,
    read_split_ids,
)
from presentation.model import DynamicsMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: DynamicsMLP,
    config: dict,
    epoch: int,
    val_loss: float,
    val_rollout_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "name": "dynamics_architecture_sweep_relu_w128_d3",
                "input_dim": 4,
                "width": config["width"],
                "depth": config["depth"],
                "output_dim": 2,
                "hidden_activation": "ReLU",
                "output_activation": "Linear; clipped to [-1, 1] at rollout",
            },
            "normalization": {
                "observation": ["r0/12", "phi/3", "u/0.5", "v/0.2"],
                "action": ["2*(u_next/0.5)-1", "v_next/0.2"],
            },
            "epoch": int(epoch),
            "validation_mse": float(val_loss),
            "validation_rollout_normalized_rmse": float(val_rollout_score),
            "config": config,
        },
        path,
    )


def validation_rollout_score(
    model: DynamicsMLP,
    dataset: dict[str, np.ndarray],
    case_ids: list[int],
    device: str,
) -> tuple[float, float, float]:
    """Score validation using the same autoregressive rollout as final tests."""
    model.eval()

    def predict(obs: np.ndarray) -> np.ndarray:
        batch = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            return model(batch).squeeze(0).cpu().numpy()

    per_ray = []
    for case_id in case_ids:
        ray = extract_ray(dataset, case_id)
        rollout = autoregressive_rollout(ray, predict)
        per_ray.append(calculate_metrics(ray, rollout))
    return (
        float(np.mean([row["normalized_rmse"] for row in per_ray])),
        float(np.mean([row["rmse_u"] for row in per_ray])),
        float(np.mean([row["rmse_v"] for row in per_ray])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--output-dir", default="outputs/mlp_runs/dynamics_architecture_sweep_relu_w128_d3")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    if args.width < 1 or args.depth < 1 or args.learning_rate <= 0:
        parser.error("width, depth, and learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    dataset = load_dataset(args.dataset)
    train_ids = read_split_ids(args.splits, "train")
    validation_ids = read_split_ids(args.splits, "validation")
    train_x, train_y = build_supervised_examples(dataset, train_ids)
    val_x, val_y = build_supervised_examples(dataset, validation_ids)

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DynamicsMLP(width=args.width, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    history_path = output_dir / "history.csv"
    config = {
        "run_name": output_dir.name,
        "task": "supervised one-step photon state prediction",
        "architecture_name": "dynamics_architecture_sweep_relu_w128_d3",
        "width": args.width,
        "depth": args.depth,
        "hidden_activation": "ReLU",
        "output_activation": "Linear; clipped to [-1, 1] at rollout",
        "input_features": ["r0/12", "phi/3", "current_u/0.5", "current_v/0.2"],
        "targets": ["2*(next_u/0.5)-1", "next_v/0.2"],
        "train_case_ids": train_ids,
        "validation_case_ids": validation_ids,
        "checkpoint_selection": "lowest autoregressive normalized RMSE on complete validation rays",
        "train_transitions": int(len(train_x)),
        "validation_transitions": int(len(val_x)),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "device": device,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    best_val = float("inf")
    best_rollout_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict] = []
    started = time.perf_counter()
    interrupted = False

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sum = 0.0
            train_count = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_sum += float(loss.detach().cpu()) * len(batch_x)
                train_count += len(batch_x)

            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    loss = loss_fn(model(batch_x), batch_y)
                    val_sum += float(loss.cpu()) * len(batch_x)
                    val_count += len(batch_x)

            train_loss = train_sum / max(1, train_count)
            val_loss = val_sum / max(1, val_count)
            val_rollout_score, val_rollout_u, val_rollout_v = validation_rollout_score(
                model, dataset, validation_ids, device
            )
            row = {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": val_loss,
                "validation_rollout_normalized_rmse": val_rollout_score,
                "validation_rollout_rmse_u": val_rollout_u,
                "validation_rollout_rmse_v": val_rollout_v,
            }
            rows.append(row)
            save_checkpoint(last_path, model, config, epoch, val_loss, val_rollout_score)

            # Select the MLP by the same kind of autoregressive metric used in
            # the final comparison, not merely by one-step teacher-forced loss.
            if val_rollout_score < best_rollout_score:
                best_val = val_loss
                best_rollout_score = val_rollout_score
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(best_path, model, config, epoch, val_loss, val_rollout_score)
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"[MLP] epoch={epoch:03d}/{args.epochs} "
                    f"train_mse={train_loss:.6g} val_mse={val_loss:.6g} "
                    f"val_rollout={val_rollout_score:.6g} best_epoch={best_epoch}",
                    flush=True,
                )
            if stale_epochs >= args.patience:
                print(f"[MLP] early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("[MLP] interrupted; the latest and best checkpoints have been saved.", flush=True)
    finally:
        elapsed = time.perf_counter() - started
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch", "train_mse", "validation_mse",
                    "validation_rollout_normalized_rmse",
                    "validation_rollout_rmse_u", "validation_rollout_rmse_v",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        config.update({
            "epochs_completed": len(rows),
            "best_epoch": best_epoch,
            "validation_mse_at_best_rollout_checkpoint": best_val if np.isfinite(best_val) else None,
            "best_validation_rollout_normalized_rmse": (
                best_rollout_score if np.isfinite(best_rollout_score) else None
            ),
            "elapsed_seconds": elapsed,
            "interrupted": interrupted,
            "best_checkpoint": str(best_path),
        })
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if not best_path.exists():
        raise RuntimeError("MLP training did not produce a best checkpoint")
    print(
        f"[MLP] done: {best_path} | best_epoch={best_epoch} "
        f"val_rollout={best_rollout_score:.6g} | elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
