"""A ceiling on what the statistical detectors may add to the score.

The anomaly model works to an alert budget: it returns roughly 5% of the rows
it is given, whatever those rows contain. So it reports findings on a clean
file exactly as it does on a broken one, and grades plenty of them HIGH.

While severity was being misread as LOW that never showed. Reading it properly
made clean books score 84 CRITICAL against a defective file's 100 - a number
that no longer told a reviewer anything. Holding each statistical source to a
ceiling restores the difference: clean 47 MEDIUM, defective 77 CRITICAL.

Failed accounting identities have no ceiling. Those are errors of fact, and
enough of them should reach 100 on their own.
"""

import pytest

from risk_reporting.risk_engine import STATISTICAL_POINT_CAPS, calculate_risk


class Anomaly:
    def __init__(self, i, severity="HIGH"):
        self.company, self.year = "Acme", 2000 + (i % 25)
        self.severity, self.anomaly_type = severity, "revenue_profit_mismatch"
        self.explanation = "Revenue rose while cash did not follow."


class Deviation:
    def __init__(self, i):
        self.company, self.year, self.metric = "Acme", 2000 + (i % 25), f"metric_{i}"
        self.deviation_percent = 400.0
        self.material_deviation = True


class Failure:
    def __init__(self, i, severity="CRITICAL"):
        self.rule_id, self.rule_name = f"R{i}", "Balance sheet does not balance"
        self.year, self.status, self.severity = 2000 + (i % 25), "FAIL", severity
        self.message = "assets do not equal liabilities plus equity"


def score(**outputs):
    return calculate_risk(outputs).score


# --- the ceiling holds -------------------------------------------------------


@pytest.mark.parametrize("source", sorted(STATISTICAL_POINT_CAPS))
def test_a_statistical_source_has_a_ceiling(source):
    maker = {"anomaly": Anomaly, "deviation": Deviation}[source]
    flood = score(**{source: [maker(i) for i in range(500)]})

    assert flood < 100, f"{source} findings alone should never reach 100"


def test_a_flood_of_anomalies_cannot_reach_critical_on_its_own():
    # 5% of a 480-row file is about 24 findings; 500 is far past anything real.
    assert score(anomaly=[Anomaly(i) for i in range(500)]) < 75


def test_more_anomalies_still_never_lower_the_score():
    previous = 0
    for count in (1, 5, 20, 100, 400):
        current = score(anomaly=[Anomaly(i) for i in range(count)])
        assert current >= previous
        previous = current


# --- while errors of fact do not --------------------------------------------


def test_failed_checks_have_no_ceiling():
    assert score(validation=[Failure(i) for i in range(200)]) == 100


def test_one_failed_check_outweighs_a_flood_of_anomalies_eventually():
    anomalies = [Anomaly(i) for i in range(500)]

    with_failures = score(anomaly=anomalies, validation=[Failure(i) for i in range(40)])
    anomalies_only = score(anomaly=anomalies)

    assert with_failures > anomalies_only, "real errors must be able to move the score"


def test_each_source_is_held_separately():
    # One source at its ceiling must not consume another's allowance.
    both = score(anomaly=[Anomaly(i) for i in range(500)],
                 deviation=[Deviation(i) for i in range(500)])
    one = score(anomaly=[Anomaly(i) for i in range(500)])

    assert both > one


# --- and severity still means something inside the ceiling -------------------


def test_high_anomalies_outscore_low_ones_under_the_cap():
    high = score(anomaly=[Anomaly(i, "HIGH") for i in range(20)])
    low = score(anomaly=[Anomaly(i, "LOW") for i in range(20)])

    assert high > low


def test_nothing_at_all_scores_zero():
    assert score(anomaly=[]) == 0
