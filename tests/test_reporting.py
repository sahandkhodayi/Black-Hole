from rl.reporting import report_is_due
import pytest


def test_report_is_due_immediately_before_first_report():
    assert report_is_due(None, 12.0, 60.0) is True


def test_report_is_due_only_after_wall_clock_interval():
    assert report_is_due(100.0, 159.9, 60.0) is False
    assert report_is_due(100.0, 160.0, 60.0) is True


def test_report_interval_must_be_positive():
    with pytest.raises(ValueError, match="interval_seconds must be positive"):
        report_is_due(0.0, 10.0, 0.0)
