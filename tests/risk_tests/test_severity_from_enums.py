"""Severity has to mean the same thing whether it arrives as text or an enum.

The risk engine read severity with `str(value).upper()`. For the plain strings
the existing tests use, that is correct. The anomaly agent reports severity as
an enum, and `str()` on one gives "Severity.HIGH", which matched nothing in the
score table and quietly fell back to LOW.

So every anomaly was scored as LOW and described as LOW in the report, while
the anomalies table on the next page showed HIGH - the TL spotted the two
descriptions disagreeing, and the disagreement was the visible part of a
scoring error underneath.

The existing tests could not catch it: they pass dictionaries with string
severities, which is the one case that always worked. These pass enums.
"""

from enum import Enum

import pytest

from finsight.core.models import Severity
from risk_reporting.risk_engine import calculate_risk


class Anomaly:
    """The shape the anomaly agent actually hands over."""

    def __init__(self, severity, company="Vantage Corporation", year=2017):
        self.severity = severity
        self.company = company
        self.year = year
        self.anomaly_type = "revenue_profit_mismatch"
        self.explanation = "Revenue rose sharply while cash did not follow."


def score_for(severity):
    return calculate_risk({"anomaly": [Anomaly(severity)]}).score


# --- the bug itself ----------------------------------------------------------


def test_an_enum_severity_is_read_as_that_severity():
    assert score_for(Severity.HIGH) == score_for("HIGH")


@pytest.mark.parametrize("level", [s.value for s in Severity])
def test_every_level_matches_its_own_text(level):
    # The anomaly agent grades LOW, MEDIUM and HIGH; it has no CRITICAL of its
    # own, so whatever it does grade has to survive the journey intact.
    assert score_for(Severity(level)) == score_for(level)


def test_a_high_anomaly_outscores_a_low_one():
    # This is what failed in the report: both scored the same, because both
    # were read as LOW.
    assert score_for(Severity.HIGH) > score_for(Severity.LOW)


def test_the_reason_text_names_the_real_severity():
    result = calculate_risk({"anomaly": [Anomaly(Severity.HIGH)]})

    reason = result.contributors[0].reason
    assert "(HIGH)" in reason, "the breakdown described a HIGH anomaly as something else"
    assert "SEVERITY" not in reason.upper().replace("SEVERITY.", ""), reason


def test_the_breakdown_agrees_with_the_finding_it_came_from():
    findings = [Anomaly(Severity.HIGH), Anomaly(Severity.MEDIUM, year=2018)]

    result = calculate_risk({"anomaly": findings})

    described = [c.reason for c in result.contributors]
    assert any("(HIGH)" in r for r in described)
    assert any("(MEDIUM)" in r for r in described)


# --- and that it holds for anything else shaped like an enum -----------------


def test_a_plain_enum_works_too():
    class OtherSeverity(Enum):
        HIGH = "HIGH"

    assert score_for(OtherSeverity.HIGH) == score_for("HIGH")


def test_an_unknown_severity_still_falls_back_quietly():
    class Odd(Enum):
        WHATEVER = "NOT_A_SEVERITY"

    assert score_for(Odd.WHATEVER) == score_for("LOW")


def test_nothing_at_all_is_not_an_error():
    assert calculate_risk({"anomaly": [Anomaly(None)]}).score == score_for("LOW")
