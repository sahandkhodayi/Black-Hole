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
    """Render a standalone report; all charts and saved images are embedded."""
    root = Path(outputs_root).expanduser().resolve()
    kind = experiment["kind"]
    config = experiment.get("config") or {}
    summary = experiment.get("summary") or {}
    history = experiment.get("history") or []
    metrics = experiment.get("metrics") or []
    comparison = experiment.get("comparison") or {}

    if kind == "mlp":
        cards = [("Epochs", summary.get("epochs_completed")), ("Best epoch", summary.get("best_epoch")),
                 ("Best validation rollout NRMSE", summary.get("best_validation_rollout_normalized_rmse")),
                 ("Training time (s)", summary.get("elapsed_seconds"))]
        charts = _chart(history, "epoch", ["train_mse", "validation_mse"], "Training and validation MSE")
        charts += _chart(history, "epoch", ["validation_rollout_normalized_rmse"], "Autoregressive validation NRMSE")
    elif kind == "rl":
        cards = [("Algorithm", config.get("algorithm", "PPO")), ("Timesteps", config.get("timesteps_actual")),
                 ("Training time (s)", config.get("seconds_actual")), ("Test rays", summary.get("num_rays")),
                 ("Mean RMSE u", summary.get("mean_rmse_u")), ("Mean RMSE v", summary.get("mean_rmse_v"))]
        charts = _chart(history, "timesteps", ["episode_reward", "reward_mean"], "Training reward")
        charts += _chart(history, "timesteps", ["validation_rmse_u", "validation_rmse_v"], "Validation RMSE")
    else:
        cards = [("Held-out test rays", summary.get("num_test_rays"))]
        for model in ("mlp", "ppo"):
            for variable in ("u", "v"):
                cards.append((f"{model.upper()} mean RMSE {variable}", (comparison.get(model) or {}).get(f"mean_rmse_{variable}")))
        charts = '<p class="muted">This paired comparison has no single training history. Compare each model on the same held-out rays below.</p>'

    card_html = "".join(f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(_display(value))}</strong></div>' for label, value in cards)
    columns = list(metrics[0]) if metrics else []
    table = ""
    if columns:
        header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
        body = "".join("<tr>" + "".join(f'<td>{html.escape(_display(row.get(column)))}</td>' for column in columns) + "</tr>" for row in metrics[:500])
        table = f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
    else:
        table = '<p class="muted">No per-ray metrics CSV was found for this experiment.</p>'

    analysis = experiment.get("analysis_markdown", "")
    notes = experiment.get("notes", [])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Black Hole Lab — {html.escape(experiment['name'])}</title>
<style>
:root{{--bg:#090d18;--panel:#121a2b;--line:#293650;--text:#edf3ff;--muted:#9caec8;--cyan:#63d5e8}}*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(ellipse at 70% -15%,#1f3150 0,transparent 45%),var(--bg);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 24px 70px}}h1{{font-size:clamp(28px,5vw,46px);line-height:1.1;margin:10px 0}}h2{{font-size:19px;margin:0 0 16px}}.muted,figcaption{{color:var(--muted)}}section{{margin:30px 0}}.meta{{color:var(--muted);font-size:12px;margin:15px 0 28px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card,.panel,figure{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}.card span{{display:block;color:var(--muted);font-size:12px}}.card strong{{display:block;font-size:22px;margin-top:6px;overflow-wrap:anywhere}}svg{{width:100%;height:auto;display:block;background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:14px 0}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}figure{{margin:0;padding:10px}}figure img{{width:100%;height:auto;display:block;border-radius:8px}}figcaption{{font-size:12px;padding:8px 4px 2px}}.scroll{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--cyan);background:#182238}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;color:#d2ddf0}}footer{{margin-top:45px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px}}
</style></head><body><main><div style="color:var(--cyan);font-size:12px;letter-spacing:.16em;text-transform:uppercase">Black Hole Lab · Experiment report</div>
<h1>{html.escape(experiment['name'])}</h1><div class="muted">{html.escape(experiment.get('label', kind))} · {html.escape(experiment.get('status', 'unknown'))}</div>
<div class="meta">ID: {html.escape(experiment['id'])} · {len(experiment.get('artifacts', []))} image artifacts · Generated {generated}</div>
<section><h2>Experiment summary</h2><div class="grid">{card_html}</div></section>
<section><h2>Training and validation history</h2>{charts}</section>
<section><h2>Evaluation metrics</h2>{table}</section>
<section><h2>Saved figures</h2><div class="gallery">{_embedded_images(root, experiment.get('artifacts', []))}</div></section>
{f'<section><h2>Existing analysis notes</h2><pre>{html.escape(analysis)}</pre></section>' if analysis else ''}
<section><h2>Experiment configuration</h2><pre>{html.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre></section>
{f'<section><h2>Data notes</h2><ul>{"".join(f"<li>{html.escape(note)}</li>" for note in notes)}</ul></section>' if notes else ''}
<footer>Generated by Black Hole Lab Analytics. Values are read from saved run artifacts; this report does not retrain models or alter experiment outputs.</footer></main></body></html>'''


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
