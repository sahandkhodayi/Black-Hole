"""Live Black Hole Lab research console for Stable-Baselines3 PPO.

The orbit panel is driven by real validation rays from the dataset. DOP853
is the reference; PPO is the current policy's autoregressive rollout.
"""
from collections import deque
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LiveTrainingDashboard(BaseCallback):
    def __init__(self, run_dir, eval_env, validation_ids, *, eval_freq=2048,
                 refresh_freq=1024, window=100, enabled=True, verbose=0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.eval_env = eval_env
        self.validation_ids = [int(x) for x in validation_ids]
        self.eval_freq = max(1, int(eval_freq))
        self.refresh_freq = max(1, int(refresh_freq))
        self.episode_rewards = deque(maxlen=window)
        self.reward_steps, self.reward_values = [], []
        self.loss_steps, self.policy_losses, self.value_losses = [], [], []
        self.eval_steps, self.rmse_u, self.rmse_v = [], [], []
        self.action_samples = deque(maxlen=5000)
        self.last_eval_step = -self.eval_freq
        self.last_dashboard_refresh = -self.refresh_freq
        self.stop_requested = False
        self.paused = False
        self.enabled = bool(enabled)
        self.figure = None
        self.axes = None
        self.pause_button = None
        self.started = None
        self.history_path = self.run_dir / "live_history.csv"
        self.orbit_frame = 1
        self.orbit_lines = None
        self.orbit_marker = None
        self.latest_ray = None

    def _on_training_start(self):
        self.started = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
            from matplotlib.patches import Circle
            plt.ion()
            # A smaller canvas reduces GUI rendering cost, especially on laptops.
            self.figure = plt.figure(figsize=(13, 8), dpi=85, constrained_layout=False)
            self.figure.canvas.manager.set_window_title("Black Hole Lab | Live Research Console")
            grid = self.figure.add_gridspec(
                4, 3, left=.055, right=.98, top=.91, bottom=.13,
                hspace=.48, wspace=.34
            )
            self.axes = np.empty((4, 3), dtype=object)
            self.axes[0, 0] = self.figure.add_subplot(grid[0:3, 0:2])
            self.axes[0, 1] = self.axes[0, 0]
            self.axes[1, 0] = self.axes[0, 0]
            self.axes[1, 1] = self.axes[0, 0]
            self.axes[2, 0] = self.axes[0, 0]
            self.axes[2, 1] = self.axes[0, 0]
            self.axes[0, 2] = self.figure.add_subplot(grid[0, 2])
            self.axes[1, 2] = self.figure.add_subplot(grid[1, 2])
            self.axes[2, 2] = self.figure.add_subplot(grid[2, 2])
            self.axes[3, 0] = self.figure.add_subplot(grid[3, 0])
            self.axes[3, 1] = self.figure.add_subplot(grid[3, 1])
            self.axes[3, 2] = self.figure.add_subplot(grid[3, 2])

            pause_ax = self.figure.add_axes([.30, .035, .16, .055])
            stop_ax = self.figure.add_axes([.53, .035, .20, .055])
            self.pause_button = Button(pause_ax, "Pause")
            self.stop_button = Button(stop_ax, "Stop & save")
            self.pause_button.on_clicked(self._toggle_pause)
            self.stop_button.on_clicked(self._request_stop)
            self.figure.canvas.mpl_connect("key_press_event", self._on_key)
            self.figure.canvas.mpl_connect("close_event", self._on_window_close)
            self._refresh()
            plt.show(block=False)

        if not self.history_path.exists():
            with self.history_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([
                    "timesteps", "elapsed_seconds", "episode_reward",
                    "policy_loss", "value_loss", "validation_rmse_u",
                    "validation_rmse_v", "action_0_mean", "action_1_mean"
                ])

    def _request_stop(self, _event=None):
        self.stop_requested = True

    def _toggle_pause(self, _event=None):
        self.paused = not self.paused
        if self.pause_button is not None:
            self.pause_button.label.set_text("Resume" if self.paused else "Pause")

    def _on_window_close(self, _event=None):
        self.stop_requested = True

    def _on_key(self, event):
        if event.key == " ":
            self._toggle_pause()
        elif event.key in {"q", "escape"}:
            self._request_stop()

    def _evaluate_validation(self):
        env = self.eval_env.unwrapped
        errors_u, errors_v = [], []
        first_ray = None
        for case_id in self.validation_ids:
            obs, _ = env.reset(options={"case_id": case_id})
            ref_u = [float(env.target_u[0])]
            pred_u = [float(env.u)]
            ref_v = [float(env.target_v[0])]
            pred_v = [float(env.v)]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ref_u.append(info["reference_u"])
                pred_u.append(info["predicted_u"])
                ref_v.append(info["reference_v"])
                pred_v.append(info["predicted_v"])
            errors_u.append(float(np.sqrt(np.mean((np.asarray(pred_u)-ref_u)**2))))
            errors_v.append(float(np.sqrt(np.mean((np.asarray(pred_v)-ref_v)**2))))
            if first_ray is None:
                # env.phi is the complete ray angle array; retain the actual
                # validation case ID and closest-approach radius for display.
                first_ray = (np.asarray(env.phi).copy(), np.asarray(ref_u),
                             np.asarray(pred_u), np.asarray(ref_v), np.asarray(pred_v),
                             float(env.r0), int(case_id))
        return float(np.mean(errors_u)), float(np.mean(errors_v)), first_ray

    @staticmethod
    def _orbit_xy(phi, u):
        radius = np.divide(1.0, np.asarray(u, dtype=float),
                           out=np.full_like(np.asarray(u, dtype=float), 15.0),
                           where=np.asarray(u, dtype=float) > 1e-5)
        radius = np.clip(radius, 0.0, 15.0)
        return radius * np.cos(phi), radius * np.sin(phi)

    def _draw_orbit(self, ax):
        from matplotlib.patches import Circle
        ax.set_facecolor("#070b16")
        # Schematic photon sphere and event horizon in the project's
        # normalized Schwarzschild units (r_s = 1).
        ax.add_patch(Circle((0, 0), 1.5, fill=False, edgecolor="#e8b75a",
                            linestyle="--", linewidth=1.5, alpha=.85,
                            label="Photon sphere (1.5 r_s)"))
        ax.add_patch(Circle((0, 0), 1.0, facecolor="#020308",
                            edgecolor="#f0a34a", linewidth=2.0,
                            label="Event horizon (1 r_s)"))
        for radius, alpha in [(1.12, .25), (1.28, .16), (1.43, .10)]:
            ax.add_patch(Circle((0, 0), radius, fill=False,
                                edgecolor="#e58b42", linewidth=5,
                                alpha=alpha))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_xlabel("x / r_s")
        ax.set_ylabel("y / r_s")
        ax.set_title("LIVE PHOTON ORBIT  |  Schwarzschild geometry", loc="left",
                     color="white", fontsize=12, fontweight="bold")
        ax.grid(alpha=.12, color="white")
        ax.tick_params(colors="#aab4c5")
        for spine in ax.spines.values():
            spine.set_color("#344054")

    def _refresh(self):
        if not self.enabled or self.figure is None:
            return
        import matplotlib.pyplot as plt
        orbit_ax = self.axes[0, 0]
        reward_ax, loss_ax, action_ax = self.axes[0, 2], self.axes[1, 2], self.axes[2, 2]
        rmse_ax, u_ax, v_ax = self.axes[3, 0], self.axes[3, 1], self.axes[3, 2]
        for ax in [orbit_ax, reward_ax, loss_ax, action_ax, rmse_ax, u_ax, v_ax]:
            ax.clear()

        self._draw_orbit(orbit_ax)
        self.orbit_lines = None
        self.orbit_marker = None
        if self.latest_ray is not None:
            phi, ref_u, pred_u, ref_v, pred_v, r0, case_id = self.latest_ray
            ref_x, ref_y = self._orbit_xy(phi[:len(ref_u)], ref_u)
            pred_x, pred_y = self._orbit_xy(phi[:len(pred_u)], pred_u)
            ref_line, = orbit_ax.plot([], [], color="#50c8ff", linewidth=2.0,
                                      label="DOP853 reference")
            pred_line, = orbit_ax.plot([], [], color="#ff6b8a", linewidth=1.8,
                                       linestyle="--", label="PPO policy")
            marker, = orbit_ax.plot([], [], "o", color="#ffffff",
                                    markeredgecolor="#ffcf70", markersize=7,
                                    label="Current ray position")
            self.orbit_lines = (ref_line, pred_line, ref_x, ref_y, pred_x, pred_y)
            self.orbit_marker = marker
            self.orbit_frame = 1
            orbit_ax.legend(loc="upper right", fontsize=8, facecolor="#111827",
                            labelcolor="white")
            orbit_ax.text(.02, .02, f"Validation case {case_id} | r₀ = {r0:.3f} r_s",
                          transform=orbit_ax.transAxes, color="white", fontsize=9)
            self._animate_orbit()
        else:
            orbit_ax.text(.5, .5, "Waiting for first validation rollout...",
                          transform=orbit_ax.transAxes, color="white",
                          ha="center", va="center")

        if self.reward_steps:
            reward_ax.plot(self.reward_steps, self.reward_values, alpha=.35,
                           label="Episode return")
            if len(self.reward_values) >= 5:
                smooth = np.convolve(self.reward_values, np.ones(5)/5, mode="valid")
                reward_ax.plot(self.reward_steps[4:], smooth, linewidth=2,
                               label="5-episode mean")
            reward_ax.legend(fontsize=7)
        reward_ax.set(title="Episode reward", xlabel="Steps", ylabel="Return")
        reward_ax.grid(alpha=.2)

        if self.loss_steps:
            loss_ax.plot(self.loss_steps, self.policy_losses, label="Policy")
            loss_ax.plot(self.loss_steps, self.value_losses, label="Value")
            loss_ax.legend(fontsize=7)
        loss_ax.set(title="PPO optimization losses", xlabel="Steps", ylabel="Loss")
        loss_ax.grid(alpha=.2)

        if self.action_samples:
            samples = np.asarray(self.action_samples)
            action_ax.hist(samples[:, 0], bins=24, alpha=.65, label="u action")
            action_ax.hist(samples[:, 1], bins=24, alpha=.55, label="v action")
            action_ax.legend(fontsize=7)
        action_ax.set(title="Recent action distribution", xlabel="Normalized action")
        action_ax.grid(alpha=.2)

        if self.eval_steps:
            rmse_ax.plot(self.eval_steps, self.rmse_u, marker="o", label="RMSE u")
            rmse_ax.plot(self.eval_steps, self.rmse_v, marker="o", label="RMSE v")
            rmse_ax.legend(fontsize=7)
        rmse_ax.set(title="Validation error ↓", xlabel="Steps", ylabel="RMSE")
        rmse_ax.grid(alpha=.2)

        if self.latest_ray is not None:
            phi, ref_u, pred_u, ref_v, pred_v, _, _ = self.latest_ray
            u_ax.plot(phi[:len(ref_u)], ref_u, label="DOP853")
            u_ax.plot(phi[:len(pred_u)], pred_u, "--", label="PPO")
            u_ax.legend(fontsize=7)
            v_ax.plot(phi[:len(ref_v)], ref_v, label="DOP853")
            v_ax.plot(phi[:len(pred_v)], pred_v, "--", label="PPO")
            v_ax.legend(fontsize=7)
        u_ax.set(title="Inverse radius u(φ)", xlabel="φ (rad)", ylabel="u")
        v_ax.set(title="Radial derivative du/dφ", xlabel="φ (rad)", ylabel="v")
        u_ax.grid(alpha=.2)
        v_ax.grid(alpha=.2)

        state = ("STOP REQUESTED — saving" if self.stop_requested else
                 "PAUSED" if self.paused else "RUNNING")
        elapsed = time.monotonic() - self.started if self.started else 0
        self.figure.suptitle(
            f"BLACK HOLE LAB  |  {state}  |  {self.num_timesteps:,} steps  |  {elapsed:.0f}s",
            fontsize=15, fontweight="bold")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(.001)
        # Do not write a PNG during training: disk rendering is expensive.

    def _animate_orbit(self):
        if self.orbit_lines is None or self.latest_ray is None:
            return
        ref_line, pred_line, ref_x, ref_y, pred_x, pred_y = self.orbit_lines
        count = min(self.orbit_frame, len(ref_x), len(pred_x))
        ref_line.set_data(ref_x[:count], ref_y[:count])
        pred_line.set_data(pred_x[:count], pred_y[:count])
        if count:
            self.orbit_marker.set_data([pred_x[count-1]], [pred_y[count-1]])
        self.orbit_frame = min(self.orbit_frame + 1, max(len(ref_x), len(pred_x)))

    def _write_history(self, episode_reward=None, policy_loss=None, value_loss=None,
                       val_u=None, val_v=None):
        actions = np.asarray(self.action_samples) if self.action_samples else np.empty((0, 2))
        means = actions.mean(axis=0) if len(actions) else [np.nan, np.nan]
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                self.num_timesteps, time.monotonic() - self.started,
                episode_reward, policy_loss, value_loss, val_u, val_v,
                means[0], means[1]
            ])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode and "r" in episode:
                reward = float(episode["r"])
                self.episode_rewards.append(reward)
                self.reward_steps.append(self.num_timesteps)
                self.reward_values.append(reward)
        actions = self.locals.get("actions")
        if actions is not None:
            for action in np.asarray(actions).reshape(-1, 2):
                self.action_samples.append(action.copy())

        # Animate the ray occasionally instead of forcing a GUI redraw every
        # 25 environment steps. The full dashboard is refreshed separately.
        if (self.enabled and self.figure is not None
                and self.num_timesteps % max(128, self.refresh_freq // 4) == 0):
            self._animate_orbit()
            self.figure.canvas.draw_idle()
            import matplotlib.pyplot as plt
            plt.pause(.001)
        while self.enabled and self.paused and not self.stop_requested:
            import matplotlib.pyplot as plt
            self.figure.canvas.flush_events()
            plt.pause(.1)
        if self.stop_requested:
            return False
        return True

    def _on_rollout_start(self):
        values = getattr(self.model.logger, "name_to_value", {})
        policy = values.get("train/policy_gradient_loss")
        value = values.get("train/value_loss")
        if policy is not None and value is not None:
            self.loss_steps.append(self.num_timesteps)
            self.policy_losses.append(float(policy))
            self.value_losses.append(float(value))

        val_u = val_v = None
        if self.enabled and self.num_timesteps - self.last_eval_step >= self.eval_freq:
            val_u, val_v, self.latest_ray = self._evaluate_validation()
            self.eval_steps.append(self.num_timesteps)
            self.rmse_u.append(val_u)
            self.rmse_v.append(val_v)
            self.last_eval_step = self.num_timesteps

        latest_reward = self.reward_values[-1] if self.reward_values else None
        self._write_history(latest_reward, policy, value, val_u, val_v)
        # Rebuild the graph panels only periodically (or immediately after a
        # fresh validation result), not after every PPO rollout.
        if (val_u is not None
                or self.num_timesteps - self.last_dashboard_refresh >= self.refresh_freq):
            self._refresh()
            self.last_dashboard_refresh = self.num_timesteps

    def _on_training_end(self):
        self._refresh()
        if self.enabled and self.figure is not None:
            # One final snapshot is useful for the presentation; save only once.
            self.figure.savefig(self.run_dir / "live_dashboard.png", dpi=100)
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self.figure)
