"""Small, dependency-free helpers for live experiment reporting."""


def report_is_due(last_report_time, now, interval_seconds):
    """Return True when a wall-clock report is due.

    last_report_time may be None to request an immediate first report.
    """
    interval = float(interval_seconds)
    if interval <= 0:
        raise ValueError("interval_seconds must be positive")
    return last_report_time is None or float(now) - float(last_report_time) >= interval
