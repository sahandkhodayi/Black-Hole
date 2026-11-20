"""Discover and normalize Black Hole Lab experiment artifacts.

The loader is deliberately read-only. It understands the existing MLP, PPO,
and paired-presentation output layouts without requiring a training rerun or
changing the files already written by those tools.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

MAX_CSV_ROWS = 10_000
MAX_ARTIFACTS = 40
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _coerce_cell(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    if number.is_integer() and not any(char in text.lower() for char in (".", "e")):
        return int(number)
    return number


def read_csv(path: Path, *, limit: int = MAX_CSV_ROWS) -> list[dict[str, Any]]:
    """Read a CSV as JSON-safe rows, returning an empty list for bad files."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                rows.append({key: _coerce_cell(value) for key, value in row.items() if key})
    except (OSError, UnicodeError, csv.Error):
        return []
    return rows


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return ""


def _iter_files(folder: Path, suffixes: set[str]):
    """Walk an experiment once, skipping bulky non-analysis artifact folders."""
    skipped = {"checkpoints", "tensorboard", "target", "__pycache__", ".git"}
    for current, directories, filenames in os.walk(folder, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in skipped)
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.suffix.lower() in suffixes:
                yield path


def _artifact_entries(root: Path, folder: Path) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if not folder.exists():
        return found
    for path in _iter_files(folder, _IMAGE_SUFFIXES):
        if not path.is_file():
            continue
        relative = _relative(root, path)
        if not relative:
            continue
        found.append({"name": path.name, "path": relative, "kind": "image"})
        if len(found) >= MAX_ARTIFACTS:
            break
    return found


def _csv_entries(root: Path, folder: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not folder.exists():
        return found
    for path in _iter_files(folder, {".csv"}):
        relative = _relative(root, path)
        if not relative:
            continue
        rows = read_csv(path)
        found.append({"name": path.name, "path": relative, "rows": len(rows)})
        if len(found) >= 40:
            break
    return found


def _rl_history(run_dir: Path) -> list[dict[str, Any]]:
    live = read_csv(run_dir / "live_history.csv")
    if live:
        return live

    # Older PPO runs may have evaluation rewards but no live CSV history.
    archive = run_dir / "evaluation" / "evaluations.npz"
    if archive.is_file():
        try:
            import numpy as np

            with np.load(archive, allow_pickle=False) as data:
                steps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
                rewards = np.asarray(data["results"], dtype=float)
            if rewards.ndim == 1:
                rewards = rewards[:, None]
            return [
                {
                    "timesteps": int(step),
                    "reward_mean": float(np.mean(values)),
                    "reward_std": float(np.std(values)),
                }
                for step, values in zip(steps, rewards, strict=False)
                if math.isfinite(float(step)) and np.all(np.isfinite(values))
            ]
        except (OSError, ValueError, KeyError, ImportError):
            return []
    return []


def _load_evaluations(root: Path, run_dir: Path) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for folder in sorted(run_dir.glob("evaluation_*")):
        if not folder.is_dir():
            continue
        name = folder.name.removeprefix("evaluation_")
        summary = _read_json(folder / "summary.json")
        metrics = read_csv(folder / "metrics.csv")
        artifacts = _artifact_entries(root, folder)
        if summary or metrics or artifacts:
            evaluations.append({
                "name": name,
                "summary": summary,
                "metrics": metrics,
                "artifacts": artifacts,
            })
    # Stable semantic order: test first, then best, final, validation, initial.
    priority = {"test": 0, "best": 1, "final": 2, "validation": 3, "initial": 4}
    evaluations.sort(key=lambda item: (priority.get(item["name"], 10), item["name"]))
    return evaluations


def _make_mlp(root: Path, folder: Path) -> dict[str, Any]:
    config = _read_json(folder / "config.json")
    history = read_csv(folder / "history.csv")
    best_epoch = config.get("best_epoch")
    rollout_values = [
        row.get("validation_rollout_normalized_rmse")
        for row in history
        if isinstance(row.get("validation_rollout_normalized_rmse"), (int, float))
    ]
    best_rollout = min(rollout_values) if rollout_values else config.get(
        "best_validation_rollout_normalized_rmse"
    )
    summary = {
        "epochs_completed": config.get("epochs_completed", len(history)),
        "best_epoch": best_epoch,
        "best_validation_rollout_normalized_rmse": best_rollout,
        "elapsed_seconds": config.get("elapsed_seconds"),
        "train_transitions": config.get("train_transitions"),
        "validation_transitions": config.get("validation_transitions"),
    }
    evaluations: list[dict[str, Any]] = []
    artifacts = _artifact_entries(root, folder)
    csv_files = _csv_entries(root, folder)
    checkpoint = folder / "best_model.pt"
    return {
        "id": f"mlp/{folder.name}",
        "name": folder.name,
        "kind": "mlp",
        "label": "Supervised MLP",
        "path": _relative(root, folder),
        "status": "completed" if checkpoint.exists() else "artifacts available",
        "config": config,
        "summary": summary,
        "history": history,
        "evaluations": evaluations,
        "metrics": [],
        "comparison": {},
        "artifacts": artifacts,
        "csv_files": csv_files,
        "notes": [],
    }


def _make_rl(root: Path, folder: Path) -> dict[str, Any]:
    config = _read_json(folder / "config.json")
    evaluations = _load_evaluations(root, folder)
    preferred = next((item for item in evaluations if item["name"] == "test"), None)
    if preferred is None and evaluations:
        preferred = evaluations[0]
    summary = preferred["summary"] if preferred else {}
    metrics = preferred["metrics"] if preferred else []
    notes: list[str] = []
    if not evaluations:
        notes.append("No evaluation_* folder with metrics.csv or summary.json was found.")
    if not (folder / "live_history.csv").is_file() and not (folder / "evaluation" / "evaluations.npz").is_file():
        notes.append("No saved training-curve history was found for this run.")
    checkpoint_exists = (folder / "final_model.zip").exists() or (folder / "best" / "best_model.zip").exists()
    return {
        "id": f"rl/{folder.name}",
        "name": folder.name,
        "kind": "rl",
        "label": str(config.get("algorithm", "PPO")),
        "path": _relative(root, folder),
        "status": "completed" if checkpoint_exists else "artifacts available",
        "config": config,
        "summary": summary,
        "history": _rl_history(folder),
        "evaluations": evaluations,
        "metrics": metrics,
        "comparison": {},
        "artifacts": _artifact_entries(root, folder),
        "csv_files": _csv_entries(root, folder),
        "notes": notes,
    }


def _make_presentation(root: Path, folder: Path) -> dict[str, Any]:
    comparison_dir = folder / "comparison"
    summary = _read_json(comparison_dir / "summary.json")
    metadata = _read_json(comparison_dir / "run_metadata.json")
    rows = read_csv(comparison_dir / "metrics_by_ray.csv")
    notes: list[str] = []
    if not summary:
        notes.append("The comparison summary is missing or could not be read.")
    if not rows:
        notes.append("No per-ray metrics were found in metrics_by_ray.csv.")
    return {
        "id": f"presentation/{folder.name}",
        "name": folder.name,
        "kind": "presentation",
        "label": "MLP vs PPO comparison",
        "path": _relative(root, folder),
        "status": "evaluated" if summary and rows else "artifacts available",
        "config": metadata,
        "summary": summary,
        "history": [],
        "evaluations": [],
        "metrics": rows,
        "comparison": {"mlp": summary.get("mlp", {}), "ppo": summary.get("ppo", {})},
        "artifacts": _artifact_entries(root, folder),
        "csv_files": _csv_entries(root, folder),
        "notes": notes,
        "analysis_markdown": _read_text(comparison_dir / "analysis.md"),
    }


def _read_text(path: Path, limit: int = 50_000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except (OSError, UnicodeError):
        return ""


def discover_experiments(outputs_root: str | Path) -> list[dict[str, Any]]:
    """Discover MLP, PPO, and paired-presentation runs below an outputs dir."""
    root = Path(outputs_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    experiments: list[dict[str, Any]] = []
    for kind, builder in (("mlp_runs", _make_mlp), ("rl_runs", _make_rl)):
        base = root / kind
        if base.is_dir():
            for folder in sorted(base.iterdir(), key=lambda item: item.name.lower()):
                if folder.is_dir() and (folder / "config.json").is_file():
                    experiments.append(builder(root, folder))
    presentation_root = root / "presentation"
    if presentation_root.is_dir():
        for folder in sorted(presentation_root.iterdir(), key=lambda item: item.name.lower()):
            if folder.is_dir() and (folder / "comparison" / "summary.json").is_file():
                experiments.append(_make_presentation(root, folder))
    return sorted(experiments, key=lambda item: (item["kind"], item["name"].lower()))


def get_experiment(outputs_root: str | Path, experiment_id: str) -> dict[str, Any] | None:
    """Return one experiment by its discovered ID; never accept an arbitrary path."""
    for experiment in discover_experiments(outputs_root):
        if experiment["id"] == experiment_id:
            return experiment
    return None


def resolve_artifact(outputs_root: str | Path, relative_path: str) -> Path:
    """Resolve a requested artifact while preventing traversal outside outputs."""
    root = Path(outputs_root).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact path escapes the outputs directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(relative_path)
    if candidate.suffix.lower() not in _IMAGE_SUFFIXES | {".csv", ".md", ".json", ".html"}:
        raise ValueError("This file type is not served by the analytics dashboard")
    return candidate


def experiment_index(outputs_root: str | Path) -> list[dict[str, Any]]:
    """Compact experiment list for the dashboard sidebar."""
    items = []
    for item in discover_experiments(outputs_root):
        config = item["config"]
        summary = item["summary"]
        items.append({
            "id": item["id"],
            "name": item["name"],
            "kind": item["kind"],
            "label": item["label"],
            "status": item["status"],
            "algorithm": config.get("algorithm"),
            "epochs": summary.get("epochs_completed"),
            "timesteps": config.get("timesteps_actual"),
            "test_rays": summary.get("num_rays", summary.get("num_test_rays")),
            "elapsed_seconds": summary.get("elapsed_seconds", config.get("seconds_actual")),
        })
    return items
