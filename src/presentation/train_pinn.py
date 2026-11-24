"""Train a PINN using only the Schwarzschild ODE and initial conditions."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from presentation.common import extract_ray, load_dataset, read_split_ids
from presentation.pinn import PhotonPINN, physics_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--output-dir", default="outputs/pinn_runs/pinn_baseline")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.seconds <= 0 or args.steps < 1 or args.batch_size < 8:
        parser.error("seconds and steps must be positive; batch-size must be >= 8")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.set_num_threads(1)

    data = load_dataset(args.dataset)
    train_ids = read_split_ids(args.splits, "train")
    val_ids = read_split_ids(args.splits, "validation")
    train_rays = [extract_ray(data, cid) for cid in train_ids]
    val_rays = [extract_ray(data, cid) for cid in val_ids]
    train_r0 = np.asarray([ray.r0 for ray in train_rays], dtype=np.float32)
    train_end = np.asarray([ray.phi[-1] for ray in train_rays], dtype=np.float32)
    val_r0 = np.asarray([ray.r0 for ray in val_rays], dtype=np.float32)
    val_end = np.asarray([ray.phi[-1] for ray in val_rays], dtype=np.float32)

    model = PhotonPINN(args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_model.pt"
    config = {
        "algorithm": "PINN",
        "physics_equation": "d2u/dphi2 + u - 1.5*u**2 = 0",
        "initial_conditions": ["u(0)=1/r0", "du/dphi(0)=0"],
        "training_mode": "physics residual and initial conditions only; no trajectory labels",
        "width": args.width, "depth": args.depth,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "seconds_requested": args.seconds, "steps_requested": args.steps,
        "seed": args.seed, "device": device,
        "train_case_ids": train_ids, "validation_case_ids": val_ids,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def validation_loss():
        model.eval()
        losses = []
        with torch.enable_grad():
            for r0_value, end_value in zip(val_r0, val_end):
                phi = torch.linspace(0.0, float(end_value), 128, device=device).reshape(-1, 1)
                r0 = torch.full_like(phi, float(r0_value))
                u, v, residual = physics_residual(model, r0, phi)
                initial = torch.tensor([[float(r0_value)]], device=device)
                phi0 = torch.zeros_like(initial)
                u0, v0, _ = physics_residual(model, initial, phi0)
                losses.append(float((residual.square().mean() + (u0 - 1.0 / initial).square().mean()
                                     + v0.square().mean()).detach().cpu()))
        return float(np.mean(losses))

    started = time.monotonic()
    rows = []
    best = float("inf")
    step = 0
    while step < args.steps and time.monotonic() - started < args.seconds:
        model.train()
        indices = np.random.randint(0, len(train_rays), size=args.batch_size)
        r0_np = train_r0[indices]
        end_np = train_end[indices]
        phi_np = np.random.random(args.batch_size).astype(np.float32) * end_np
        r0 = torch.as_tensor(r0_np, device=device).reshape(-1, 1)
        phi = torch.as_tensor(phi_np, device=device).reshape(-1, 1)
        u, v, residual = physics_residual(model, r0, phi)
        phi0 = torch.zeros_like(r0)
        u0, v0, _ = physics_residual(model, r0, phi0)
        target_u0 = 1.0 / r0
        loss_physics = residual.square().mean()
        loss_initial = (u0 - target_u0).square().mean() + v0.square().mean()
        loss = loss_physics + 10.0 * loss_initial
        if not torch.isfinite(loss):
            raise FloatingPointError("PINN loss became non-finite; training stopped")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step == 1 or step % 100 == 0:
            val = validation_loss()
            row = {"step": step, "train_loss": float(loss.detach().cpu()),
                   "physics_loss": float(loss_physics.detach().cpu()),
                   "initial_condition_loss": float(loss_initial.detach().cpu()),
                   "validation_physics_loss": val,
                   "elapsed_seconds": time.monotonic() - started}
            rows.append(row)
            if val < best:
                best = val
                torch.save({"state_dict": model.state_dict(), "config": config,
                            "step": step, "validation_physics_loss": best}, best_path)
            print(f"[PINN] step={step} train={row['train_loss']:.5g} val_physics={val:.5g}", flush=True)

    elapsed = time.monotonic() - started
    if not best_path.exists():
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "step": step, "validation_physics_loss": validation_loss()}, best_path)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "step", "train_loss", "physics_loss", "initial_condition_loss",
            "validation_physics_loss", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    config.update({"steps_completed": step, "elapsed_seconds": elapsed,
                   "best_validation_physics_loss": best,
                   "checkpoint": str(best_path)})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[PINN] saved {best_path}; steps={step}; elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
