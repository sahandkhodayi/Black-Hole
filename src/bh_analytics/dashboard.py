"""Local browser-based experiment dashboard for Black Hole Lab."""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

from lab.geodesics import simulate_ray
from bh_analytics.core import discover_experiments, experiment_index, get_experiment, resolve_artifact
from bh_analytics.report import generate_report
from bh_analytics.training import TrainingManager

_TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"
_PHYSICS_TEMPLATE = Path(__file__).parent / "templates" / "physics_explorer.html"


def _simulate_photon_payload(closest_approach: object) -> dict:
    """Run a bounded, validated Schwarzschild scattering simulation for the explorer."""
    try:
        radius0 = float(closest_approach)
    except (TypeError, ValueError) as exc:
        raise ValueError("closest_approach must be a number.") from exc
    if not math.isfinite(radius0) or not 1.8 <= radius0 <= 15.0:
        raise ValueError("closest_approach must be between 1.8 and 15 Schwarzschild radii.")

    ray = simulate_ray(closest_approach=radius0, max_angle=40.0, max_step=0.02)
    radius = ray.radius
    visible = np.isfinite(radius) & (radius >= 1.0) & (radius <= 24.0)
    phi = ray.phi[visible]
    visible_radius = radius[visible]
    return {
        "closest_approach": radius0,
        "phi": phi.tolist(),
        "u": ray.u[visible].tolist(),
        "radius": visible_radius.tolist(),
        "x": (visible_radius * np.cos(phi)).tolist(),
        "y": (visible_radius * np.sin(phi)).tolist(),
        "escaped": ray.escaped,
        "escape_angle": ray.escape_angle,
        "deflection_degrees": ray.deflection_degrees,
        "nfev": ray.nfev,
    }


def make_handler(outputs_root: Path, *, allow_training: bool = True):
    root = outputs_root.resolve()
    training = TrainingManager(root)

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "BlackHoleLabAnalytics/0.2"

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, value) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                try:
                    self._send(200, _TEMPLATE.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    self._json(500, {"error": "Dashboard template is missing."})
                return
            if parsed.path == "/physics":
                try:
                    self._send(200, _PHYSICS_TEMPLATE.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    self._json(500, {"error": "Physics Explorer template is missing."})
                return
            if parsed.path.startswith("/assets/"):
                name = parsed.path.rsplit("/", 1)[-1]
                allowed = {
                    "dashboard.css": "text/css; charset=utf-8",
                    "dashboard-core.js": "text/javascript; charset=utf-8",
                    "dashboard-ui.js": "text/javascript; charset=utf-8",
                    "physics-explorer.css": "text/css; charset=utf-8",
                    "physics-explorer.js": "text/javascript; charset=utf-8",
                }
                if name not in allowed:
                    self._json(404, {"error": "Asset not found."})
                    return
                asset_path = Path(__file__).parent / "assets" / name
                try:
                    self._send(200, asset_path.read_bytes(), allowed[name])
                except OSError:
                    self._json(404, {"error": "Asset not found."})
                return
            if parsed.path == "/api/training/status":
                if not allow_training:
                    self._json(403, {"error": "Browser training is disabled when the server is exposed beyond this computer."})
                else:
                    self._json(200, training.status())
                return
            if parsed.path == "/api/experiments":
                self._json(200, {"experiments": experiment_index(root)})
                return
            if parsed.path == "/api/experiment":
                experiment_id = query.get("id", [""])[0]
                experiment = get_experiment(root, experiment_id)
                if experiment is None:
                    self._json(404, {"error": "Experiment not found."})
                else:
                    self._json(200, experiment)
                return
            if parsed.path == "/artifact":
                relative = query.get("path", [""])[0]
                try:
                    path = resolve_artifact(root, relative)
                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    if path.suffix.lower() == ".html":
                        raise ValueError("HTML reports must be opened through the report endpoint")
                    if path.suffix.lower() == ".csv":
                        content_type = "text/csv; charset=utf-8"
                    elif path.suffix.lower() in {".json", ".md"}:
                        content_type = "text/plain; charset=utf-8"
                    self._send(200, path.read_bytes(), content_type)
                except FileNotFoundError:
                    self._json(404, {"error": "Artifact not found."})
                except (ValueError, OSError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            if parsed.path == "/report":
                relative = query.get("path", [""])[0]
                try:
                    path = resolve_artifact(root, relative)
                    reports_root = (root / "analytics" / "reports").resolve()
                    path.resolve().relative_to(reports_root)
                    if path.suffix.lower() != ".html":
                        raise ValueError("Only generated HTML reports can be opened here")
                    self._send(200, path.read_bytes(), "text/html; charset=utf-8")
                except FileNotFoundError:
                    self._json(404, {"error": "Report not found."})
                except (ValueError, OSError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "Unknown endpoint."})

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed_path = urlparse(self.path).path
            if parsed_path == "/api/physics/simulate":
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    self._json(403, {"error": "Cross-origin simulation requests are not allowed."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 8_192:
                        raise ValueError("Invalid request size.")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise ValueError("The request body must be a JSON object.")
                    self._json(200, _simulate_photon_payload(body.get("closest_approach", 5.0)))
                except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                    self._json(400, {"error": str(exc)})
                except (RuntimeError, FloatingPointError) as exc:
                    self._json(500, {"error": f"Numerical integration failed: {exc}"})
                return

            if parsed_path in {"/api/training/start", "/api/training/stop"}:
                if not allow_training:
                    self._json(403, {"error": "Browser training is disabled when the server is exposed beyond this computer."})
                    return
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    self._json(403, {"error": "Cross-origin training requests are not allowed."})
                    return
                try:
                    if parsed_path == "/api/training/start":
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 8_192:
                            raise ValueError("Invalid request size.")
                        body = json.loads(self.rfile.read(length).decode("utf-8"))
                        self._json(202, training.start(body))
                    else:
                        self._json(202, training.stop())
                except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                    self._json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self._json(409, {"error": str(exc)})
                except OSError as exc:
                    self._json(500, {"error": f"Training request failed: {exc}"})
                return

            if parsed_path != "/api/report":
                self._json(404, {"error": "Unknown endpoint."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8_192:
                    raise ValueError("Invalid request size.")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("The request body must be a JSON object.")
                experiment_id = str(body.get("id", ""))
                path = generate_report(root, experiment_id)
                relative = path.resolve().relative_to(root).as_posix()
                self._json(201, {"path": relative, "url": "/report?path=" + quote(relative, safe="/")})
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
            except (ValueError, UnicodeError, json.JSONDecodeError, OSError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"[analytics] {self.address_string()} - {fmt % args}")

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs", help="Path to the project's outputs directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; default is local-only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        parser.error(f"Outputs directory does not exist: {root}")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        training_allowed = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        training_allowed = False
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(root, allow_training=training_allowed),
    )
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"Black Hole Lab Analytics is running at {url}")
    print(f"Reading experiments from: {root}")
    print("Browser training: enabled (localhost only)" if training_allowed
          else "Browser training: disabled for non-localhost bindings")
    print("Press Ctrl+C to stop the dashboard.")
    if not args.no_browser and args.host in {"127.0.0.1", "localhost"}:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
