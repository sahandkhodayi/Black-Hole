"""Wall-clock callback for short presentation experiments."""
import time
from stable_baselines3.common.callbacks import BaseCallback


class WallClockLimitCallback(BaseCallback):
    def __init__(self, seconds, verbose=0):
        super().__init__(verbose)
        self.seconds = float(seconds)

    def _on_training_start(self):
        self.started = time.monotonic()

    def _on_step(self):
        elapsed = time.monotonic() - self.started
        self.logger.record("time/wall_clock_seconds", elapsed)
        return elapsed < self.seconds
