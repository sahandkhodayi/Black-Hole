"""Create portable, self-contained HTML experiment reports."""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bh_analytics.core import get_experiment, resolve_artifact


def _display(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.6g}"
    return str(value)


def _chart(rows: list[dict[str, Any]], x_key: str, keys: list[str], title: str) -> str:
    """Build a small self-contained SVG chart from finite numeric CSV values."""
    series = []
    for key in keys:
        points = [(row.get(x_key), row.get(key)) for row in rows]
        points = [(x, y) for x, y in points if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        if points:
            series.append((key, points))
    if not series:
        return f'<p class="muted">No compatible values were saved for {html.escape(title)}.</p>'
    all_points = [point for _, points in series for point in points]
    xmin = min(x for x, _ in all_points); xmax = max(x for x, _ in all_points)
    ymin = min(y for _, y in all_points); ymax = max(y for _, y in all_points)
    colors = ["#63d5e8", "#a99bff", "#f1bd70"]
    def xy(x: float, y: float) -> tuple[float, float]:
        px = 58 + (x - xmin) / (xmax - xmin or 1) * 700
        py = 255 - (y - ymin) / (ymax - ymin or 1) * 220
        return px, py
    paths = []
    legends = []
    for index, (name, points) in enumerate(series):
        color = colors[index % len(colors)]
        coords = [xy(float(x), float(y)) for x, y in points]
        commands = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
        paths.append(f'<path d="{commands}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legends.append(f'<text x="{65 + index * 220}" y="24" fill="{color}" font-size="12">{html.escape(name)}</text>')
    grid = "".join(f'<line x1="58" y1="{35 + i * 55}" x2="758" y2="{35 + i * 55}" stroke="#2b3850"/><text x="8" y="{39 + i * 55}" fill="#9caec8" font-size="10">{ymax - (ymax-ymin)*i/4:.3g}</text>' for i in range(5))
    return (f'<svg viewBox="0 0 780 290" role="img" aria-label="{html.escape(title)}">'
            f'{grid}{"".join(legends)}{"".join(paths)}'
            f'<text x="400" y="282" text-anchor="middle" fill="#9caec8" font-size="11">{html.escape(x_key)}</text></svg>')


def _embedded_images(root: Path, artifacts: list[dict[str, str]]) -> str:
    figures = []
    for artifact in artifacts[:16]:
        try:
            path = resolve_artifact(root, artifact["path"])
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if not mime.startswith("image/"):
                continue
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            source = f"data:{mime};base64,{payload}"
            label = html.escape(artifact["name"])
            figures.append(f'<figure><img src="{source}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        except (OSError, ValueError, KeyError):
            continue
    return "".join(figures) or '<p class="muted">No saved image artifacts were found.</p>'


def render_report(experiment: dict[str, Any], outputs_root: str | Path) -> str:
    """Render a beginner-friendly, self-contained report from whatever results exist."""
    root = Path(outputs_root).expanduser().resolve()
    kind = experiment["kind"]
    summary = experiment.get("summary") or {}
    history = experiment.get("history") or []
    metrics = experiment.get("metrics") or []
    comparison = experiment.get("comparison") or {}
    cards: list[tuple[str, Any, str]] = []
    charts = ""
    conclusion = ""
    comparison_bars = ""

    if kind == "presentation":
        mlp = comparison.get("mlp") or summary.get("mlp") or {}
        ppo = comparison.get("ppo") or summary.get("ppo") or {}
        mlp_error = mlp.get("mean_normalized_rmse")
        ppo_error = ppo.get("mean_normalized_rmse")
        rays = summary.get("num_test_rays", summary.get("num_rays"))
        cards = [
            ("Unseen test paths", rays, "Paths kept out of training"),
            ("MLP average error", mlp_error, "Lower is better"),
            ("PPO average error", ppo_error, "Lower is better"),
        ]
        if isinstance(mlp_error, (int, float)) and isinstance(ppo_error, (int, float)):
            winner = "MLP" if mlp_error <= ppo_error else "PPO"
            ratio = max(mlp_error, ppo_error) / max(min(mlp_error, ppo_error), 1e-12)
            loser = "PPO" if winner == "MLP" else "MLP"
            conclusion = (
                f"{winner} followed the reference paths more closely in this test. "
                f"Its average error was about {ratio:.2g} times smaller than {loser}'s."
            )
            maximum = max(mlp_error, ppo_error, 1e-12)
            comparison_bars = (
                f'<div class="bar"><span>MLP</span><div><i style="width:{mlp_error/maximum*100:.2f}%;background:#62c7de"></i></div><b>{mlp_error:.4g}</b></div>'
                f'<div class="bar"><span>PPO</span><div><i style="width:{ppo_error/maximum*100:.2f}%;background:#ab9af4"></i></div><b>{ppo_error:.4g}</b></div>'
            )
        else:
            conclusion = "The saved results include a paired comparison. Some aggregate error values were not recorded."
        explanation = (
            "Each model predicted the next point along photon paths that were not used for training. "
            "We compared those predictions with the DOP853 numerical reference. An error of zero is a perfect match; lower is better. "
            "This small test does not prove that one AI method is always better or that either model discovered Einstein's equations."
        )
        charts = '<p class="muted">There is no single training curve for a paired comparison. The important result is how both models performed on the same unseen paths.</p>'
        heading = "Which model followed the paths more accurately?"
    elif kind == "mlp":
        cards = [
            ("Training rounds", summary.get("epochs_completed"), "Rounds completed"),
            ("Best round", summary.get("best_epoch"), "Selected using validation data"),
            ("Best validation error", summary.get("best_validation_rollout_normalized_rmse"), "Lower is better"),
            ("Training time", summary.get("elapsed_seconds"), "Seconds"),
        ]
        charts = _chart(history, "epoch", ["train_mse", "validation_mse"], "Training progress")
        explanation = (
            "The network learns from examples of photon motion. Training error measures its fit on examples it learns from; "
            "validation error checks performance on separate examples. A separate held-out test is needed to judge full-path predictions."
        )
        conclusion = "This run records how the neural network learned. It is not, by itself, a comparison against PPO."
        heading = "How did the neural network learn?"
    else:
        cards = [
            ("Test paths", summary.get("num_rays"), "Complete photon paths"),
            ("Average distance error", summary.get("mean_rmse_u"), "Lower is better"),
            ("Average change error", summary.get("mean_rmse_v"), "Lower is better"),
        ]
        charts = _chart(history, "timesteps", ["reward_mean", "episode_reward"], "Training reward")
        explanation = (
            "PPO learns by trying predictions and receiving feedback. The test errors show how far its predictions were from "
            "the numerical reference on the saved test paths. Smaller errors are better."
        )
        conclusion = "This report summarizes the saved PPO training and evaluation results."
        heading = "How did PPO perform?"

    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(_display(value))}</strong><small>{html.escape(note)}</small></div>'
        for label, value, note in cards
    )

    if metrics:
        if kind == "presentation":
            columns = [("case_id", "Test path"), ("model", "Model"), ("normalized_rmse", "Overall error"), ("rmse_u", "Distance error"), ("rmse_v", "Change error")]
        else:
            columns = [("case_id", "Test path"), ("rmse_u", "Distance error"), ("rmse_v", "Change error")]
        present_columns = [(key, label) for key, label in columns if key in metrics[0]]
        header = "".join(f"<th>{html.escape(label)}</th>" for _, label in present_columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_display(row.get(key)))}</td>" for key, _ in present_columns) + "</tr>"
            for row in metrics[:200]
        )
        table = f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
    else:
        table = '<p class="muted">Individual test-path scores were not saved. The overall summary above is still available.</p>'

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    figures = _embedded_images(root, experiment.get("artifacts", []))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Black Hole Lab — {html.escape(experiment["name"])}</title>
<style>
:root{{--bg:#0b1220;--panel:#142238;--line:#2b3c54;--text:#edf3fb;--muted:#a5b2c5;--blue:#62c7de;--violet:#ab9af4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 Inter,"Segoe UI",Arial,sans-serif}}
main{{max-width:1000px;margin:auto;padding:38px 24px 65px}}.eyebrow{{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.15em}}
h1{{font-size:clamp(28px,5vw,42px);line-height:1.15;margin:9px 0}}h2{{font-size:20px;margin:0 0 12px}}p{{max-width:800px}}.muted,small{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}}
.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px}}.card span,.card small{{display:block;color:var(--muted);font-size:11px}}.card strong{{display:block;font-size:24px;margin:7px 0;overflow-wrap:anywhere}}
section{{margin:30px 0}}.bar{{display:grid;grid-template-columns:52px minmax(80px,1fr) 85px;gap:12px;align-items:center;margin:15px 0;font-size:13px}}.bar>div{{height:15px;border-radius:20px;background:#26384f;overflow:hidden}}.bar i{{height:100%;display:block;border-radius:20px}}
svg{{width:100%;height:auto;background:var(--panel);border-radius:12px;margin-top:12px}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}}th{{color:var(--blue)}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:9px 12px;color:var(--muted);font-size:11px}}footer{{border-top:1px solid var(--line);margin-top:42px;padding-top:15px;color:var(--muted);font-size:11px}}
</style></head><body><main>
<p class="eyebrow">BLACK HOLE LAB · EXPERIMENT RESULTS</p>
<h1>{html.escape(experiment["name"].replace("_", " ").replace("-", " "))}</h1>
<p class="muted">{html.escape(experiment.get("label", kind))} · Created {generated}</p>
<section><h2>At a glance</h2><div class="grid">{card_html}</div></section>
<section class="panel"><p class="eyebrow">THE TAKEAWAY</p><h2>{html.escape(heading)}</h2><p>{html.escape(conclusion)}</p>{comparison_bars}</section>
<section><h2>What do these numbers mean?</h2><p>{html.escape(explanation)}</p></section>
<section><h2>Learning progress</h2>{charts}</section>
<section><h2>Scores for each test path</h2>{table}<p class="muted">Lower error means the predicted path stayed closer to the numerical reference.</p></section>
<section><h2>Saved pictures</h2><div class="gallery">{figures}</div></section>
<footer>Generated by Black Hole Lab. This report uses saved results only; it does not train a model or modify experiment files.</footer>
</main></body></html>'''

def generate_report(outputs_root: str | Path, experiment_id: str, output_dir: str | Path | None = None) -> Path:
    experiment = get_experiment(outputs_root, experiment_id)
    if experiment is None:
        raise KeyError(f"Unknown experiment ID: {experiment_id}")
    root = Path(outputs_root).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "analytics" / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in experiment_id)
    target = target_dir / f"{safe_name.replace('/', '_')}.html"
    target.write_text(render_report(experiment, root), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone Black Hole Lab HTML experiment report.")
    parser.add_argument("--root", default="outputs", help="Outputs directory")
    parser.add_argument("--experiment", required=True, help="Experiment ID shown in the dashboard")
    parser.add_argument("--output-dir", default=None, help="Report directory (defaults to outputs/analytics/reports)")
    args = parser.parse_args()
    path = generate_report(args.root, args.experiment, args.output_dir)
    print(f"Report generated: {path.resolve()}")


if __name__ == "__main__":
    main()
