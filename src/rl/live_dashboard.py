"""Interactive Matplotlib dashboard for PPO training.

Press the in-window "Stop & save" button or Q to stop safely. The callback
returns False at the next environment step; train.py then saves final_model.zip.
"""
from collections import deque
import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LiveTrainingDashboard(BaseCallback):
    def __init__(self, run_dir, eval_env, validation_ids, *, eval_freq=1024,
                 window=100, enabled=True, verbose=0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.eval_env = eval_env
        self.validation_ids = [int(x) for x in validation_ids]
        self.eval_freq = max(1, int(eval_freq))
        self.episode_rewards = deque(maxlen=window)
        self.reward_steps, self.reward_values = [], []
        self.loss_steps, self.policy_losses, self.value_losses = [], [], []
        self.eval_steps, self.rmse_u, self.rmse_v = [], [], []
        self.last_eval_step = -self.eval_freq
        self.stop_requested = False
        self.enabled = bool(enabled)
        self.figure = None
        self.axes = None
        self.started = None
        self.history_path = self.run_dir / "live_history.csv"

    def _on_training_start(self):
        self.started = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.enabled:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
            plt.ion()
            self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.figure.canvas.manager.set_window_title("Black Hole Lab — Live PPO")
            self.figure.subplots_adjust(bottom=0.13, hspace=0.32, wspace=0.28)
            button_ax = self.figure.add_axes([0.39, 0.025, 0.22, 0.055])
            self.stop_button = Button(button_ax, "Stop & save")
            self.stop_button.on_clicked(self._request_stop)
            self.figure.canvas.mpl_connect("key_press_event", self._on_key)
            self.figure.suptitle("Black Hole Lab | PPO training dashboard")
            self._refresh()
            plt.show(block=False)

        if not self.history_path.exists():
            with self.history_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    ["timesteps", "elapsed_seconds", "episode_reward",
                     "policy_loss", "value_loss", "validation_rmse_u",
                     "validation_rmse_v"])

    def _request_stop(self, _event=None):
        self.stop_requested = True

    def _on_key(self, event):
        if event.key in {"q", "escape"}:
            self._request_stop()

    def _evaluate_validation(self):
        errors_u, errors_v = [], []
        first_ray = None
        for case_id in self.validation_ids:
            obs, _ = self.eval_env.reset(options={"case_id": case_id})
            ref_u = [float(self.eval_env.target_u[0])]
            pred_u = [float(self.eval_env.u)]
            ref_v = [float(self.eval_env.target_v[0])]
            pred_v = [float(self.eval_env.v)]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ref_u.append(info["reference_u"])
                pred_u.append(info["predicted_u"])
                ref_v.append(info["reference_v"])
                pred_v.append(info["predicted_v"])
            errors_u.append(float(np.sqrt(np.mean((np.asarray(pred_u)-ref_u)**2))))
            errors_v.append(float(np.sqrt(np.mean((np.asarray(pred_v)-ref_v)**2))))
            if first_ray is None:
                first_ray = (np.asarray(ref_u), np.asarray(pred_u))
        return float(np.mean(errors_u)), float(np.mean(errors_v)), first_ray

    def _refresh(self):
        if not self.enabled or self.figure is None:
            return
        import matplotlib.pyplot as plt
        reward_ax, loss_ax, error_ax, trajectory_ax = self.axes
        for ax in self.axes.flat:
            ax.clear()

        if self.reward_steps:
            reward_ax.plot(self.reward_steps, self.reward_values, alpha=.35,
                           label="Episode reward")
            if len(self.reward_values) >= 5:
                smooth = np.convolve(self.reward_values,
                                     np.ones(5)/5, mode="valid")
                reward_ax.plot(self.reward_steps[4:], smooth, linewidth=2,
                               label="5-episode mean")
            reward_ax.legend()
        reward_ax.set(title="Episode reward", xlabel="Timesteps", ylabel="Return")
        reward_ax.grid(alpha=.25)

        if self.loss_steps:
            loss_ax.plot(self.loss_steps, self.policy_losses, label="Policy loss")
            loss_ax.plot(self.loss_steps, self.value_losses, label="Value loss")
            loss_ax.legend()
        loss_ax.set(title="Optimization losses", xlabel="Timesteps", ylabel="Loss")
        loss_ax.grid(alpha=.25)

        if self.eval_steps:
            error_ax.plot(self.eval_steps, self.rmse_u, marker="o", label="RMSE u")
            error_ax.plot(self.eval_steps, self.rmse_v, marker="o", label="RMSE v")
            error_ax.legend()
        error_ax.set(title="Validation error (lower is better)",
                     xlabel="Timesteps", ylabel="RMSE")
        error_ax.grid(alpha=.25)

        if getattr(self, "latest_ray", None) is not None:
            ref_u, pred_u = self.latest_ray
            phi = np.arange(len(ref_u))
            trajectory_ax.plot(phi, ref_u, label="DOP853 reference")
            trajectory_ax.plot(phi, pred_u, "--", label="PPO prediction")
            trajectory_ax.legend()
        trajectory_ax.set(title="Live validation ray: inverse radius",
                          xlabel="Sample index", ylabel="u = 1/r")
        trajectory_ax.grid(alpha=.25)

        state = "STOP REQUESTED — saving after current step" if self.stop_requested else "RUNNING"
        elapsed = time.monotonic() - self.started if self.started else 0
        self.figure.suptitle(
            f"Black Hole Lab | {state} | {self.num_timesteps:,} steps | {elapsed:.0f}s")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)
        self.figure.savefig(self.run_dir / "live_dashboard.png", dpi=140)

    def _write_history(self, episode_reward=None, policy_loss=None, value_loss=None,
                       val_u=None, val_v=None):
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                self.num_timesteps, time.monotonic() - self.started,
                episode_reward, policy_loss, value_loss, val_u, val_v])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode and "r" in episode:
                reward = float(episode["r"])
                self.episode_rewards.append(reward)
                self.reward_steps.append(self.num_timesteps)
                self.reward_values.append(reward)
        # Pump GUI events during collection so Stop/Q works mid-rollout.
        if self.enabled and self.figure is not None and self.num_timesteps % 25 == 0:
            self.figure.canvas.flush_events()
        if self.stop_requested:
            return False
        return True

    def _on_rollout_start(self):
        # SB3 calls this after the previous PPO update, so logger losses are
        # available here (the first rollout naturally has no losses yet).
        values = getattr(self.model.logger, "name_to_value", {})
        policy = values.get("train/policy_gradient_loss")
        value = values.get("train/value_loss")
        if policy is not None and value is not None:
            self.loss_steps.append(self.num_timesteps)
            self.policy_losses.append(float(policy))
            self.value_losses.append(float(value))

        val_u = val_v = None
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            val_u, val_v, self.latest_ray = self._evaluate_validation()
            self.eval_steps.append(self.num_timesteps)
            self.rmse_u.append(val_u)
            self.rmse_v.append(val_v)
            self.last_eval_step = self.num_timesteps

        latest_reward = self.reward_values[-1] if self.reward_values else None
        self._write_history(latest_reward, policy, value, val_u, val_v)
        self._refresh()

    def _on_training_end(self):
        self._refresh()
        if self.enabled:
            import matplotlib.pyplot as plt
            if self.figure is not None:
                plt.ioff()
                plt.close(self.figure)
