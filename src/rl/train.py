"""Train PPO on the teacher-guided photon trajectory environment."""
import argparse
import json
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from rl.callbacks import WallClockLimitCallback
from rl.environment import PhotonTrajectoryEnv


def read_ids(path, key):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [int(x) for x in manifest[key]]
    if not ids:
        raise ValueError(f"Split {key!r} is empty")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    name = args.run_name or f"ppo_{int(args.seconds)}s_seed{args.seed}"
    run_dir = Path("outputs/rl_runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = read_ids(args.splits, "train"), read_ids(args.splits, "validation")
    train_env = Monitor(PhotonTrajectoryEnv(args.dataset, train_ids, randomize_cases=True))
    eval_env = Monitor(PhotonTrajectoryEnv(args.dataset, val_ids, randomize_cases=False))
    model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=256,
                batch_size=64, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, policy_kwargs={"net_arch": [64, 64]},
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=args.seed, device=args.device, verbose=1)
    model.save(run_dir / "initial_model")
    callbacks = [
        WallClockLimitCallback(args.seconds),
        CheckpointCallback(save_freq=500, save_path=str(run_dir / "checkpoints"),
                           name_prefix="ppo_photon"),
        EvalCallback(eval_env, best_model_save_path=str(run_dir / "best"),
                     log_path=str(run_dir / "evaluation"), eval_freq=500,
                     n_eval_episodes=min(4, len(val_ids)), deterministic=True, verbose=1),
    ]
    config = {
        "algorithm": "PPO", "task": "teacher-guided sequential photon trajectory tracking",
        "seconds_requested": args.seconds, "seed": args.seed,
        "dataset": args.dataset, "splits": args.splits,
        "train_case_ids": train_ids, "validation_case_ids": val_ids,
        "observation": ["r0/12", "phi/3", "current_u/0.5", "current_v/0.2"],
        "action": ["u in [0, 0.5]", "v in [-0.2, 0.2]"],
        "reward": "1 - 4 * (((u_pred-u_ref)/0.5)^2 + ((v_pred-v_ref)/0.2)^2)",
        "policy": "MlpPolicy", "network": [64, 64], "learning_rate": 3e-4,
        "n_steps": 256, "batch_size": 64, "n_epochs": 5,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        model.learn(total_timesteps=10_000_000, callback=callbacks,
                    progress_bar=False, tb_log_name=name)
    finally:
        elapsed = time.monotonic() - started
        model.save(run_dir / "final_model")
        config["seconds_actual"] = elapsed
        config["timesteps_actual"] = int(model.num_timesteps)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        train_env.close()
        eval_env.close()
    print(f"Saved {run_dir}; elapsed={elapsed:.1f}s; timesteps={model.num_timesteps}")


if __name__ == "__main__":
    main()
