"""Stage timing for the review pipeline.

Two things depend on this. The dashboard shows the reviewer how long a review
took, which is the whole business case in one number - seconds against the
hours the same work takes by hand. And the monitoring view needs per-stage
durations to show where time actually goes.

Usage::

    timings = Timings()

    with timings.stage("validation"):
        results = run_all_validations(records)

    timings.total          # 0.42
    timings.to_dict()      # {"validation": 0.41, "_total": 0.42}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

#: Key used for the whole-run duration, prefixed so it cannot collide with a
#: stage name.
TOTAL_KEY = "_total"


class Timings:
    """Collects how long each stage of a review took, in seconds.

    Durations are wall clock, measured with a monotonic counter so a system
    clock change mid-run cannot produce a negative figure.
    """

    def __init__(self) -> None:
        self._stages: List[Tuple[str, float]] = []
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block of work and record it under ``name``.

        The duration is recorded even when the block raises, so a failed run
        still shows where the time went before it failed.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._stages.append((name, time.perf_counter() - started))

    def record(self, name: str, seconds: float) -> None:
        """Record a duration measured elsewhere."""
        self._stages.append((name, float(seconds)))

    @property
    def total(self) -> float:
        """Seconds since this Timings object was created."""
        return time.perf_counter() - self._started

    @property
    def stages(self) -> List[Tuple[str, float]]:
        """Stage durations in the order they were recorded."""
        return list(self._stages)

    def slowest(self) -> Tuple[str, float] | None:
        """The stage that took longest, or ``None`` if nothing ran yet."""
        return max(self._stages, key=lambda pair: pair[1], default=None)

    def to_dict(self) -> Dict[str, float]:
        """Durations rounded for display and storage.

        A stage timed more than once - which happens when the pipeline runs
        per company - is summed rather than overwritten.

        Microsecond resolution, because a fast stage rounded to four places
        reports 0.0, and a stage that ran is never a stage that took no time.
        """
        out: Dict[str, float] = {}
        for name, seconds in self._stages:
            out[name] = round(out.get(name, 0.0) + seconds, 6)
        out[TOTAL_KEY] = round(self.total, 6)
        return out

    def summary(self) -> str:
        """One line for logs: ``reviewed in 1.42s (validation 0.81s slowest)``."""
        slowest = self.slowest()
        if slowest is None:
            return f"reviewed in {self.total:.2f}s"
        name, seconds = slowest
        return f"reviewed in {self.total:.2f}s ({name} {seconds:.2f}s slowest)"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Timings {self.summary()}>"


def timed(func: Any) -> Any:
    """Decorator recording a function's duration onto a passed-in Timings.

    The wrapped function must accept a ``timings`` keyword argument. Useful
    for stage functions that are called from more than one place.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timings: Timings | None = kwargs.get("timings")
        if timings is None:
            return func(*args, **kwargs)
        with timings.stage(func.__name__):
            return func(*args, **kwargs)

    return wrapper
