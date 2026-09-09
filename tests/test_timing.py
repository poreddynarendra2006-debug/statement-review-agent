"""Stage timings feed the dashboard and the monitoring view, so they must be
recorded even when a stage fails."""

import time

import pytest

from agents.timing import TOTAL_KEY, Timings, timed


def test_stage_records_a_duration():
    timings = Timings()
    with timings.stage("validation"):
        time.sleep(0.01)

    recorded = dict(timings.stages)
    assert "validation" in recorded
    assert recorded["validation"] >= 0.01


def test_durations_are_never_negative():
    timings = Timings()
    with timings.stage("instant"):
        pass
    assert all(seconds >= 0 for _, seconds in timings.stages)
    assert timings.total >= 0


def test_stage_is_recorded_even_when_it_raises():
    """A failed run must still show where the time went before it failed."""
    timings = Timings()

    with pytest.raises(ValueError):
        with timings.stage("anomaly"):
            raise ValueError("component blew up")

    assert "anomaly" in dict(timings.stages)


def test_stages_keep_their_order():
    timings = Timings()
    for name in ("ingest", "validation", "trend", "anomaly"):
        with timings.stage(name):
            pass

    assert [name for name, _ in timings.stages] == [
        "ingest", "validation", "trend", "anomaly",
    ]


def test_repeated_stage_is_summed_not_overwritten():
    """The pipeline runs some stages once per company."""
    timings = Timings()
    timings.record("validation", 0.10)
    timings.record("validation", 0.25)

    assert timings.to_dict()["validation"] == pytest.approx(0.35)


def test_to_dict_includes_the_total():
    timings = Timings()
    with timings.stage("trend"):
        pass

    payload = timings.to_dict()
    assert TOTAL_KEY in payload
    assert payload[TOTAL_KEY] >= payload["trend"]


def test_slowest_identifies_the_worst_stage():
    timings = Timings()
    timings.record("fast", 0.01)
    timings.record("slow", 0.90)
    timings.record("medium", 0.20)

    assert timings.slowest() == ("slow", 0.90)


def test_slowest_is_none_before_anything_runs():
    assert Timings().slowest() is None


def test_summary_is_readable():
    timings = Timings()
    timings.record("validation", 0.42)
    summary = timings.summary()

    assert "reviewed in" in summary
    assert "validation" in summary


def test_timed_decorator_records_when_given_timings():
    @timed
    def run_stage(value, timings=None):
        return value * 2

    timings = Timings()
    assert run_stage(21, timings=timings) == 42
    assert "run_stage" in dict(timings.stages)


def test_timed_decorator_is_a_no_op_without_timings():
    @timed
    def run_stage(value, timings=None):
        return value * 2

    assert run_stage(21) == 42
