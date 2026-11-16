import json
import time
from collections import deque

import numpy as np

from rl.console_telemetry import ConsoleTelemetryCallback


def make_callback(tmp_path):
    return ConsoleTelemetryCallback(tmp_path, eval_env=None, validation_ids=[1])


def test_sample_downsamples_and_preserves_endpoints(tmp_path):
    callback = make_callback(tmp_path)
    values = np.arange(1000, dtype=float)

    sampled = callback._sample(values, 100)

    assert len(sampled) == 100
    assert sampled[0] == 0.0
    assert sampled[-1] == 999.0


def test_action_histogram_counts_recent_actions(tmp_path):
    callback = make_callback(tmp_path)
    callback.action_samples = deque(
        [np.asarray([-1.0, 1.0]), np.asarray([0.0, 0.0]),
         np.asarray([0.5, -0.5])],
        maxlen=2048,
    )

    histogram_u = callback._action_histogram(0)
    histogram_v = callback._action_histogram(1)

    assert len(histogram_u) == 24
    assert len(histogram_v) == 24
    assert sum(point[1] for point in histogram_u) == 3
    assert sum(point[1] for point in histogram_v) == 3


def test_emit_writes_one_valid_jsonl_record(tmp_path):
    callback = make_callback(tmp_path)
    callback.started = time.monotonic()
    callback.num_timesteps = 17
    events_path = tmp_path / "events.jsonl"

    with events_path.open("w", encoding="utf-8") as handle:
        callback._events_handle = handle
        callback._emit({"type": "metrics", "episode_reward": 12.5})
        callback._events_handle = None

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["timesteps"] == 17
    assert event["episode_reward"] == 12.5
