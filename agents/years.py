"""What counts as a real reporting year.

The Validation rule accepts any year from 1900 to 2100, so a statement dated
2099 passed every check - and then stretched the review period to "FY2016 -
FY2099". A statement cannot cover a year that has not begun.

The latest accepted year is next year rather than this one: a fiscal year is
often named for the calendar year it ends in, so in September a company can
already be reporting against FY2027.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

EARLIEST_VALID_YEAR = 1900


def latest_valid_year(today: Optional[date] = None) -> int:
    """The latest year a statement can cover: next year, by fiscal naming."""
    return (today or date.today()).year + 1


def is_plausible_year(value: Any, today: Optional[date] = None) -> bool:
    """True for a year a real statement could cover. Anything unreadable is not."""
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return False
    return EARLIEST_VALID_YEAR <= year <= latest_valid_year(today)
