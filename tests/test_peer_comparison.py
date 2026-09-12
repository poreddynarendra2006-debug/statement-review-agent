"""Peer comparison: does it find the company that stands apart, and stay quiet otherwise?

The risk with a comparison agent is noise. Flagging a third of the dataset
would be worse than not running it, so several of these tests are about what
it does *not* say.
"""

import csv
from pathlib import Path

import pytest

from analysis.peer_comparison import (
    MIN_PEERS,
    PeerFinding,
    compare_with_peers,
)

DATA = Path(__file__).resolve().parent.parent / "data"
CLEAN = DATA / "dummy_statements_clean.csv"


def company(name, year=2023, **fields):
    """One company-year with ordinary figures, overridden as a test needs."""
    base = dict(
        company=name, year=year, category="TECH",
        revenue=1000.0, gross_profit=400.0, operating_income=200.0,
        net_income=100.0, total_assets=2000.0, total_liabilities=800.0,
        shareholder_equity=1200.0, current_assets=600.0, current_liabilities=300.0,
    )
    base.update(fields)
    return base


def peer_group(size=8, **fields):
    """A set of similar companies, so one odd member has something to stand out from.

    Their figures vary a few percent, the way real peers do. Identical peers
    would leave the group no spread at all, which is a different case - and one
    test below covers it deliberately.
    """
    group = []
    for i in range(size):
        row = company(f"Peer {i}", **fields)
        drift = 1 + (i - size / 2) * 0.02
        for name in ("net_income", "gross_profit", "operating_income", "current_assets"):
            if isinstance(row.get(name), float):
                row[name] = round(row[name] * drift, 2)
        group.append(row)
    return group


def metrics_for(findings, name):
    return [f for f in findings if f.metric == name]


# --- when it should stay quiet ---------------------------------------------


def test_too_few_companies_means_no_comparison():
    # Four companies is not a peer group: one of them is always the extreme.
    records = peer_group(MIN_PEERS - 1) + [company("Odd One", net_income=-400.0)]

    assert compare_with_peers(records) == []


def test_a_company_in_line_with_its_peers_is_not_reported():
    records = peer_group(8)

    assert compare_with_peers(records) == []


def test_no_records_is_answered_with_no_findings():
    assert compare_with_peers([]) == []


def test_a_group_that_agrees_exactly_produces_nothing():
    # Identical figures leave the group no spread of its own. Nobody stands
    # apart from a unanimous group they are part of.
    records = [company(f"Peer {i}") for i in range(10)]

    assert compare_with_peers(records) == []


def test_one_company_away_from_a_unanimous_group_is_still_reported():
    # With no spread to measure against, the agreed figure itself is the
    # yardstick - otherwise a group that agrees closely could hide an outlier.
    records = [company(f"Peer {i}") for i in range(8)] + [company("Odd One", net_income=-900.0)]

    findings = compare_with_peers(records)

    assert {f.company for f in findings} == {"Odd One"}


def test_different_years_are_not_compared_with_each_other():
    # One company per year, so no year has a peer group, even though the
    # figures differ wildly across years.
    records = [company(f"Solo {y}", year=y, net_income=100.0 * (y - 2015))
               for y in range(2016, 2024)]

    assert compare_with_peers(records) == []


# --- when it should speak up ------------------------------------------------


def test_the_company_far_outside_the_group_is_reported():
    records = peer_group(8) + [company("Outlier Ltd", net_income=-900.0)]

    findings = compare_with_peers(records)

    reported = {f.company for f in findings}
    assert reported == {"Outlier Ltd"}, "only the company standing apart is named"


def test_a_finding_carries_the_figures_behind_it():
    records = peer_group(8) + [company("Outlier Ltd", net_income=-900.0)]

    finding = metrics_for(compare_with_peers(records), "net_profit_margin")[0]

    assert isinstance(finding, PeerFinding)
    assert finding.company == "Outlier Ltd"
    assert finding.year == 2023
    assert finding.value == pytest.approx(-0.9)
    assert finding.peer_median == pytest.approx(0.1, abs=0.01)
    assert finding.peer_count == 9
    assert finding.direction == "below"
    assert finding.source == "peer"
    assert "-90.0%" in finding.issue, "the figure a reviewer would check is in the sentence"
    assert "9 peers" in finding.evidence
    assert finding.peer_low <= finding.peer_median <= finding.peer_high


def test_being_far_above_the_group_is_reported_too():
    # Unusual is unusual in both directions: a margin nobody else approaches
    # is as much worth a question as a loss.
    records = peer_group(8) + [company("High Flyer", net_income=900.0)]

    finding = metrics_for(compare_with_peers(records), "net_profit_margin")[0]

    assert (finding.company, finding.direction) == ("High Flyer", "above")


RANK = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def test_severity_rises_with_distance():
    nearer = peer_group(8) + [company("Odd", current_assets=600.0, current_liabilities=95.0)]
    further = peer_group(8) + [company("Odd", current_assets=600.0, current_liabilities=8.0)]

    near_finding = metrics_for(compare_with_peers(nearer), "current_ratio")[0]
    far_finding = metrics_for(compare_with_peers(further), "current_ratio")[0]

    assert far_finding.distance > near_finding.distance
    assert RANK[far_finding.severity] >= RANK[near_finding.severity]


@pytest.mark.parametrize("liabilities, expected", [
    (210.0, "MEDIUM"),    # outside the group, but not by much
    (190.0, "HIGH"),
    (170.0, "CRITICAL"),  # nothing like its peers
])
def test_how_far_outside_decides_the_severity(liabilities, expected):
    records = peer_group(8) + [company("Odd", current_assets=600.0,
                                       current_liabilities=liabilities)]

    finding = metrics_for(compare_with_peers(records), "current_ratio")[0]

    assert finding.severity == expected


def test_the_worst_finding_comes_first():
    records = peer_group(8) + [
        company("Slightly Odd", current_assets=600.0, current_liabilities=95.0),
        company("Very Odd", net_income=-5000.0),
    ]

    findings = compare_with_peers(records)

    assert findings == sorted(findings, key=lambda f: -f.distance)


# --- how the peer group is chosen -------------------------------------------


def test_companies_are_compared_within_their_own_industry():
    # The same margin is ordinary among banks and extreme among software firms.
    banks = [company(f"Bank {i}", category="BANKING", net_income=20.0) for i in range(6)]
    tech = [company(f"Tech {i}", category="TECH", net_income=300.0) for i in range(6)]
    records = banks + tech + [company("Thin Margin Tech", category="TECH", net_income=15.0)]

    findings = compare_with_peers(records)

    assert {f.company for f in findings} == {"Thin Margin Tech"}
    assert all(f.peer_group.startswith("TECH") for f in findings)


def test_a_small_industry_is_compared_against_the_whole_year_and_says_so():
    tech = peer_group(8, category="TECH")
    records = tech + [company("Lonely Bank", category="BANKING", net_income=-900.0)]

    findings = compare_with_peers(records)

    assert {f.company for f in findings} == {"Lonely Bank"}
    assert all(f.widened for f in findings)
    assert all("the whole of 2023" in f.issue for f in findings)


# --- reading whatever the agents hand over -----------------------------------


def test_records_may_be_objects_rather_than_dictionaries():
    class Row:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    records = [Row(**row) for row in peer_group(8) + [company("Outlier Ltd", net_income=-900.0)]]

    assert {f.company for f in compare_with_peers(records)} == {"Outlier Ltd"}


@pytest.mark.parametrize("missing", [
    {"revenue": None},
    {"revenue": 0.0},
    {"net_income": None},
    {"shareholder_equity": 0.0},
    {"current_liabilities": None},
])
def test_a_measure_it_cannot_compute_is_skipped_not_guessed(missing):
    records = peer_group(8) + [company("Gappy Ltd", **missing)]

    findings = compare_with_peers(records)

    assert all(f.company != "Gappy Ltd" or f.value is not None for f in findings)


def test_a_record_with_no_company_or_year_is_ignored():
    records = peer_group(8) + [company("", net_income=-900.0), company("No Year", year=None)]

    assert compare_with_peers(records) == []


def test_findings_serialise_for_the_api():
    records = peer_group(8) + [company("Outlier Ltd", net_income=-900.0)]

    data = compare_with_peers(records)[0].to_dict()

    assert data["company"] == "Outlier Ltd"
    assert data["source"] == "peer"
    assert data["finding"] == data["issue"], "the risk engine reads `finding`"


def test_nothing_it_says_accuses_anyone():
    # A statistical difference is a reason to look, never an allegation.
    records = peer_group(8) + [company("Outlier Ltd", net_income=-900.0)]

    words = " ".join(f"{f.issue} {f.evidence}" for f in compare_with_peers(records)).lower()

    for accusation in ("fraud", "fraudulent", "manipulat", "falsif", "misconduct", "illegal"):
        assert accusation not in words


# --- against the real dataset ------------------------------------------------


def read_csv(path):
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [{k.strip().lower().replace(" ", "_"): number(v) for k, v in row.items()}
                for row in csv.DictReader(handle)]


@pytest.mark.skipif(not CLEAN.exists(), reason="the dummy dataset is not present")
def test_it_stays_quiet_on_most_of_the_clean_dataset():
    # Calibration, stated as a test: a comparison agent that reports a third of
    # the dataset is noise a reviewer will ignore. Measured at 7%.
    records = read_csv(CLEAN)

    findings = compare_with_peers(records)

    flagged = {(f.company, f.year) for f in findings}
    share = len(flagged) / len(records)
    assert 0.01 <= share <= 0.10, f"{share:.1%} of company-years flagged"
    assert all(f.peer_count >= MIN_PEERS for f in findings)
