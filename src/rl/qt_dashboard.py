"""Standalone PySide6 + PyQtGraph dashboard for PPO training.

Run from the repository root:
    python -m rl.qt_dashboard --seconds 600

The dashboard launches the existing training module as a child process. All
plot items are created once and updated with setData(); no axes are cleared.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg


class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, seconds=600, run_name=None, refresh_ms=250):
        super().__init__()
        self.setWindowTitle("Black Hole Lab | Live PPO Dashboard")
        self.resize(1440, 900)
        self.seconds = int(seconds)
        self.run_name = run_name
        self._read_offset = 0
        self._partial = b""
        self._timesteps = []
        self._rewards = []
        self._reward_episode = []
        self._policy_losses = []
        self._value_losses = []
        self._rmse_u = []
        self._rmse_v = []
        self._steps_for_loss = []
        self._steps_for_validation = []
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_process_output)
        self._process.finished.connect(self._process_finished)

        self.run_dir = Path("outputs/rl_runs") / (run_name or "qt_dashboard_run")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.run_dir / "dashboard.jsonl"
        self.control_path = self.run_dir / "dashboard.control"
        self.control_path.write_text("resume", encoding="utf-8")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)
        header = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Starting training…")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.steps_label = QtWidgets.QLabel("0 steps")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0 s")
        header.addWidget(self.status_label)
        header.addStretch(1)
        header.addWidget(self.steps_label)
        header.addSpacing(18)
        header.addWidget(self.elapsed_label)
        layout.addLayout(header)

        controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.stop_button = QtWidgets.QPushButton("Stop & save")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_training)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        pg.setConfigOptions(antialias=True, background="#101722", foreground="#d8e1ee")
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        self.reward_plot = self.graphics.addPlot(row=0, col=0, title="Episode reward")
        self.reward_plot.showGrid(x=True, y=True, alpha=0.2)
        self.reward_curve = self.reward_plot.plot(pen=pg.mkPen("#54c7ec", width=2))
        self.reward_mean_curve = self.reward_plot.plot(pen=pg.mkPen("#f4bd50", width=2))

        self.loss_plot = self.graphics.addPlot(row=0, col=1, title="PPO optimization losses")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self.policy_curve = self.loss_plot.plot(pen=pg.mkPen("#54c7ec", width=2), name="Policy")
        self.value_curve = self.loss_plot.plot(pen=pg.mkPen("#f4bd50", width=2), name="Value")
        self.loss_plot.addLegend()

        self.error_plot = self.graphics.addPlot(row=1, col=0, title="Validation RMSE (lower is better)")
        self.error_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rmse_u_curve = self.error_plot.plot(pen=pg.mkPen("#54c7ec", width=2), symbol="o")
        self.rmse_v_curve = self.error_plot.plot(pen=pg.mkPen("#f4bd50", width=2), symbol="o")
        self.error_plot.addLegend()

        self.trajectory_plot = self.graphics.addPlot(row=1, col=1, title="Validation ray: inverse radius u(φ)")
        self.trajectory_plot.showGrid(x=True, y=True, alpha=0.2)
        self.reference_curve = self.trajectory_plot.plot(
            pen=pg.mkPen("#54c7ec", width=2), name="DOP853 reference")
        self.prediction_curve = self.trajectory_plot.plot(
            pen=pg.mkPen("#ff6b8a", width=2, style=QtCore.Qt.PenStyle.DashLine),
            name="PPO prediction")
        self.trajectory_plot.addLegend()

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.log)

        self._started = QtCore.QElapsedTimer()
        self._started.start()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll_telemetry)
        self._timer.start(max(100, int(refresh_ms)))
        self._launch_training()

    def _launch_training(self):
        args = ["-m", "rl.train", "--seconds", str(self.seconds),
                "--telemetry-path", str(self.telemetry_path),
                "--control-path", str(self.control_path)]
        if self.run_name:
            args.extend(["--run-name", self.run_name])
        self.log.appendPlainText("Launching: " + sys.executable + " " + " ".join(args))
        self._process.start(sys.executable, args)
        if not self._process.waitForStarted(5000):
            self.status_label.setText("Could not start training process")
            self.log.appendPlainText(self._process.errorString())

    def _read_process_output(self):
        output = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output.strip():
            self.log.appendPlainText(output.rstrip())

    def _process_finished(self, code, _status):
        self.status_label.setText("Training finished" if code == 0 else f"Training exited with code {code}")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.log.appendPlainText(f"Training process exited with code {code}.")

    def _toggle_pause(self):
        paused = self.pause_button.text() == "Pause"
        self.control_path.write_text("pause" if paused else "resume", encoding="utf-8")
        self.pause_button.setText("Resume" if paused else "Pause")
        self.status_label.setText("Paused" if paused else "Running")

    def _stop_training(self):
        self.control_path.write_text("stop", encoding="utf-8")
        self.status_label.setText("Stopping — saving model…")
        self.stop_button.setEnabled(False)

    def _poll_telemetry(self):
        if not self.telemetry_path.exists():
            self.elapsed_label.setText(f"Elapsed: {self._started.elapsed() // 1000} s")
            return
        with self.telemetry_path.open("rb") as stream:
            stream.seek(self._read_offset)
            chunk = stream.read()
            self._read_offset = stream.tell()
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()
        for raw in lines:
            if not raw:
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            self._consume(record)
        self.elapsed_label.setText(f"Elapsed: {self._started.elapsed() // 1000} s")

    def _consume(self, record):
        event = record.get("event")
        step = int(record.get("timesteps", 0))
        self.steps_label.setText(f"{step:,} steps")
        if event == "status":
            self.status_label.setText(str(record.get("status", "Running")).capitalize())
        elif event == "episode":
            self._reward_episode.append(float(record.get("reward", 0.0)))
            self._rewards.append(float(record.get("reward", 0.0)))
            self._timesteps.append(step)
            self.reward_curve.setData(self._timesteps, self._rewards)
            if len(self._rewards) >= 5:
                import numpy as np
                smooth = np.convolve(self._rewards, np.ones(5) / 5, mode="valid")
                self.reward_mean_curve.setData(self._timesteps[4:], smooth)
        elif event == "metrics":
            self._steps_for_loss.append(step)
            self._policy_losses.append(record.get("policy_loss"))
            self._value_losses.append(record.get("value_loss"))
            self.policy_curve.setData(self._steps_for_loss, self._policy_losses)
            self.value_curve.setData(self._steps_for_loss, self._value_losses)
        elif event == "validation":
            self._steps_for_validation.append(step)
            self._rmse_u.append(record.get("rmse_u"))
            self._rmse_v.append(record.get("rmse_v"))
            self.rmse_u_curve.setData(self._steps_for_validation, self._rmse_u)
            self.rmse_v_curve.setData(self._steps_for_validation, self._rmse_v)
            ray = record.get("ray") or {}
            phi = ray.get("phi", [])
            if phi:
                self.reference_curve.setData(phi, ray.get("reference_u", []))
                self.prediction_curve.setData(phi, ray.get("predicted_u", []))
                self.trajectory_plot.setTitle(
                    f"Validation case {ray.get('case_id')} | r₀={ray.get('r0', 0):.3f} rₛ")
            self.log.appendPlainText(
                f"Validation @ {step:,}: RMSE u={record.get('rmse_u', float('nan')):.5f}, "
                f"v={record.get('rmse_v', float('nan')):.5f}")
        elif event == "warning":
            self.log.appendPlainText("WARNING: " + str(record.get("message", "")))

    def closeEvent(self, event):
        if self._process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.control_path.write_text("stop", encoding="utf-8")
            if not self._process.waitForFinished(5000):
                self._process.terminate()
                if not self._process.waitForFinished(2000):
                    self._process.kill()
                    self._process.waitForFinished(1000)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--refresh-ms", type=int, default=250)
    args = parser.parse_args()
    if args.seconds <= 0 or args.refresh_ms <= 0:
        parser.error("--seconds and --refresh-ms must be positive")
    app = QtWidgets.QApplication(sys.argv)
    window = TrainingWindow(args.seconds, args.run_name, args.refresh_ms)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
