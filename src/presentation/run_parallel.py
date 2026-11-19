"""Train the supervised MLP and PPO simultaneously, then compare them."""
from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _spawn(label: str, command: list[str], log_path: Path, env: dict[str, str]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    print(f"[{label}] starting: {' '.join(command)}")
    print(f"[{label}] log: {log_path}")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return process, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/photon_trajectories_v2.npz")
    parser.add_argument("--splits", default="data/split_manifest_v2.json")
    parser.add_argument("--run-name", default="class_presentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl-seconds", type=float, default=600,
                        help="Wall-clock training budget for PPO; default is 10 minutes")
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-patience", type=int, default=30)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ppo-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--example-case", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if args.rl_seconds <= 0 or args.mlp_epochs <= 0 or args.mlp_patience <= 0:
        parser.error("training time, MLP epochs, and patience must be positive")
    for required in (args.dataset, args.splits):
        if not Path(required).is_file():
            raise FileNotFoundError(f"Required input is missing: {required}")

    mlp_name = "dynamics_architecture_sweep_relu_w128_d3"
    mlp_dir = Path("outputs/mlp_runs") / mlp_name
    ppo_name = f"{args.run_name}_ppo"
    ppo_dir = Path("outputs/rl_runs") / ppo_name
    log_dir = Path("outputs/presentation") / args.run_name / "training_logs"
    comparison_dir = Path(args.output) if args.output else Path("outputs/presentation") / args.run_name / "comparison"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the two CPU jobs from each creating a large thread pool at the same time.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    mlp_cmd = [
        sys.executable, "-m", "presentation.train_mlp",
        "--dataset", args.dataset,
        "--splits", args.splits,
        "--output-dir", str(mlp_dir),
        "--epochs", str(args.mlp_epochs),
        "--patience", str(args.mlp_patience),
        "--batch-size", str(args.mlp_batch_size),
        "--seed", str(args.seed),
        "--device", args.mlp_device,
        "--torch-threads", "1",
    ]
    ppo_cmd = [
        sys.executable, "-m", "rl.train",
        "--dataset", args.dataset,
        "--splits", args.splits,
        "--seconds", str(args.rl_seconds),
        "--seed", str(args.seed),
        "--run-name", ppo_name,
        "--device", args.ppo_device,
    ]

    jobs = {}
    started = time.perf_counter()
    try:
        jobs["MLP"] = _spawn("MLP", mlp_cmd, log_dir / "mlp.log", env)
        jobs["PPO"] = _spawn("PPO", ppo_cmd, log_dir / "ppo.log", env)
        last_report = 0.0
        while any(process.poll() is None for process, _ in jobs.values()):
            time.sleep(2.0)
            now = time.perf_counter()
            if now - last_report >= 15.0:
                states = [f"{name}: {'running' if proc.poll() is None else f'exit {proc.returncode}'}"
                          for name, (proc, _) in jobs.items()]
                print("[parallel] " + " | ".join(states), flush=True)
                last_report = now
    except KeyboardInterrupt:
        print("\n[parallel] stop requested; sending interrupt to training jobs...", flush=True)
        for process, _ in jobs.values():
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except Exception:
                    process.terminate()
        for process, _ in jobs.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
    finally:
        for process, log in jobs.values():
            if process.poll() is None:
                process.terminate()
            log.close()

    elapsed = time.perf_counter() - started
    results = {name: process.returncode for name, (process, _) in jobs.items()}
    print(f"\n[parallel] finished in {elapsed:.1f}s; exit codes: {results}")
    print(f"[parallel] MLP log: {log_dir / 'mlp.log'}")
    print(f"[parallel] PPO log: {log_dir / 'ppo.log'}")
    failed = {name: code for name, code in results.items() if code not in (0, None)}
    if failed:
        details = ", ".join(f"{name} exit={code}" for name, code in failed.items())
        raise RuntimeError(
            f"A training process failed ({details}). No comparison was run to avoid "
            "accidentally evaluating an old checkpoint. Inspect the logs above."
        )

    mlp_checkpoint = mlp_dir / "best_model.pt"
    ppo_best = ppo_dir / "best" / "best_model.zip"
    ppo_final = ppo_dir / "final_model.zip"

    if not mlp_checkpoint.is_file():
        raise RuntimeError(
            f"MLP checkpoint was not created: {mlp_checkpoint}. Check {log_dir / 'mlp.log'}"
        )
    candidates = {}
    if ppo_best.is_file():
        candidates["validation_reward_best"] = ppo_best
    if ppo_final.is_file():
        candidates["final_iteration"] = ppo_final
    if not candidates:
        raise RuntimeError(
            f"PPO checkpoint was not created: {ppo_best} or {ppo_final}. Check {log_dir / 'ppo.log'}"
        )

    from presentation.evaluate_pair import select_ppo_checkpoint

    presentation_run_dir = Path("outputs/presentation") / args.run_name
    selection_report = presentation_run_dir / "ppo_checkpoint_selection.json"
    selected_ppo = select_ppo_checkpoint(
        candidates,
        args.dataset,
        args.splits,
        selection_report,
    )
    # Give the class demo a stable checkpoint path regardless of whether the
    # validation winner was best_model.zip or final_model.zip.
    selected_ppo_copy = presentation_run_dir / "selected_ppo_model.zip"
    selected_ppo_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_ppo, selected_ppo_copy)
    print(f"[selection] PPO checkpoint selected on validation rollout error: {selected_ppo}")
    print(f"[selection] stable copy: {selected_ppo_copy}")

    if not args.skip_evaluation:
        command = [
            sys.executable, "-m", "presentation.evaluate_pair",
            "--mlp", str(mlp_checkpoint),
            "--ppo", str(selected_ppo_copy),
            "--dataset", args.dataset,
            "--splits", args.splits,
            "--split", "test",
            "--output", str(comparison_dir),
        ]
        if args.example_case is not None:
            command.extend(["--example-case", str(args.example_case)])
        print("\n[comparison] evaluating both models on the same held-out test rays...")
        subprocess.run(command, check=True, env=env)

    print("\nPresentation assets:")
    print(f"  MLP checkpoint: {mlp_checkpoint}")
    print(f"  PPO checkpoint: {selected_ppo_copy}")
    print(f"  PPO selection report: {selection_report}")
    print(f"  Comparison folder: {comparison_dir}")
    print("  Class demo: python -m presentation.classroom_demo")
    if any(code not in (0, None) for code in results.values()):
        raise SystemExit("One training process exited with an error. Inspect its log before presenting.")


if __name__ == "__main__":
    main()
