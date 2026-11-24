"""Train PPO to choose the integration step size for photon geodesics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl.adaptive_environment import AdaptivePhotonEnv
from rl.callbacks import WallClockLimitCallback


def read_ids(path, key):
    ids = [int(x) for x in json.loads(Path(path).read_text(encoding="utf-8"))[key]]
    if not ids:
        raise ValueError(f"Split {key!r} is empty")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="adaptive_ppo")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--min-step", type=float, default=0.002)
    parser.add_argument("--max-step", type=float, default=0.04)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    train_ids = read_ids(args.splits, "train")
    val_ids = read_ids(args.splits, "validation")
    train_env = Monitor(AdaptivePhotonEnv(
        args.dataset, train_ids, min_step=args.min_step, max_step=args.max_step))
    eval_env = Monitor(AdaptivePhotonEnv(
        args.dataset, val_ids, min_step=args.min_step, max_step=args.max_step,
        randomize_cases=False))
    output = Path(args.output_root) / "adaptive_rl_runs" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    model = PPO(
        "MlpPolicy", train_env, learning_rate=5e-5, n_steps=1024,
        batch_size=128, n_epochs=5, gamma=0.99, gae_lambda=0.95,
        clip_range=0.10, ent_coef=0.002, target_kl=0.015,
        max_grad_norm=0.5, clip_range_vf=0.2,
        policy_kwargs={"net_arch": [64, 64]}, seed=args.seed,
        device=args.device, verbose=1,
        tensorboard_log=str(output / "tensorboard"),
    )
    config = {
        "algorithm": "PPO", "task": "teacher-guided adaptive RK4 step-size control",
        "action": f"step size in [{args.min_step}, {args.max_step}] radians",
        "reward": "negative normalized DOP853 state error minus small step-count penalty",
        "seconds_requested": args.seconds, "seed": args.seed,
        "train_case_ids": train_ids, "validation_case_ids": val_ids,
        "learning_rate": 5e-5, "n_steps": 1024, "batch_size": 128,
        "n_epochs": 5, "clip_range": 0.10, "target_kl": 0.015,
        "max_grad_norm": 0.5, "clip_range_vf": 0.2,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    callbacks = [
        WallClockLimitCallback(args.seconds),
        CheckpointCallback(save_freq=1000, save_path=str(output / "checkpoints"),
                           name_prefix="adaptive_ppo"),
        EvalCallback(eval_env, best_model_save_path=str(output / "best"),
                     log_path=str(output / "evaluation"), eval_freq=1000,
                     n_eval_episodes=min(4, len(val_ids)), deterministic=True),
    ]
    started = time.monotonic()
    try:
        model.learn(total_timesteps=10_000_000, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("Interrupted; saving the current adaptive policy.")
    finally:
        model.save(output / "final_model")
        config["seconds_actual"] = time.monotonic() - started
        config["timesteps_actual"] = int(model.num_timesteps)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        train_env.close()
        eval_env.close()
    print(f"Saved adaptive PPO run: {output}")


if __name__ == "__main__":
    main()
