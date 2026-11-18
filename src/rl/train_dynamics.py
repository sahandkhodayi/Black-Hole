"""Train a neural photon-dynamics surrogate from DOP853 transitions.

Unlike PPO, this is deterministic supervised regression: every training
example maps a known state and angular interval to the next DOP853 state.
Whole trajectories are separated by the existing train/validation manifest.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rl.dynamics_model import (
    PhotonDynamicsMLP,
    load_packed_dataset,
    make_transition_arrays,
    save_checkpoint,
)


def read_split(path, key):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [int(value) for value in manifest.get(key, [])]
    if not ids:
        raise ValueError(f"Split {key!r} is empty in {path}")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu", "silu", "elu"], default="gelu")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="dynamics_supervised")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.width, args.depth) <= 0:
        parser.error("epochs, batch-size, and patience must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    ) else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot access a CUDA device")

    data = load_packed_dataset(args.dataset)
    train_ids = read_split(args.splits, "train")
    val_ids = read_split(args.splits, "validation")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation case IDs overlap")

    x_train, y_train, step_scale = make_transition_arrays(data, train_ids)
    x_val, y_val, val_step_scale = make_transition_arrays(data, val_ids)
    if not np.isclose(step_scale, val_step_scale):
        raise ValueError("Unexpected inconsistent angular-step scale")

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model = PhotonDynamicsMLP(width=args.width, activation=args.activation, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    loss_fn = torch.nn.MSELoss()
    run_dir = Path("outputs/rl_runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0

    print(f"Device: {device}")
    print(f"Training rays: {len(train_ids)} ({len(x_train):,} transitions)")
    print(f"Validation rays: {len(val_ids)} ({len(x_val):,} transitions)")
    print("Training a supervised next-state model; no PPO reward is involved.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(xb)
            train_count += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t))
        train_loss = train_sum / max(train_count, 1)
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                run_dir / "best" / "best_model.pt", model,
                step_scale=step_scale, width=args.width, activation=args.activation,
                depth=args.depth, best_epoch=best_epoch,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} | train MSE {train_loss:.6f} | "
                f"validation MSE {val_loss:.6f}"
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    with (run_dir / "training_history.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_mse,validation_mse,learning_rate\n")
        for row in history:
            handle.write(
                f"{row['epoch']},{row['train_mse']},{row['validation_mse']},"
                f"{row['learning_rate']}\n"
            )
    config = {
        "algorithm": "supervised MLP regression",
        "architecture": {"width": args.width, "depth": args.depth, "activation": args.activation},
        "task": "one-step photon state prediction from DOP853 transitions",
        "seed": args.seed,
        "device": device,
        "train_case_ids": train_ids,
        "validation_case_ids": val_ids,
        "train_transitions": len(x_train),
        "validation_transitions": len(x_val),
        "step_scale": step_scale,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "early_stopping_patience": args.patience,
        "input_schema": ["r0/12", "phi/3", "current_u/0.5", "current_v/0.2", "delta_phi/step_scale"],
        "target_schema": ["next_u/0.5", "next_v/0.2"],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {run_dir / 'best' / 'best_model.pt'}")
    print(f"Training history: {run_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
