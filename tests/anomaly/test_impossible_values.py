"""Records with impossible figures are reported whatever the alert budget allows.

The detector keeps its findings to about 5% of a file, so an ordinary upload is
not buried in borderline cases. Records far from the rest (CERTAIN_ROBUST_Z)
were already exempt. But impossible is not the same as extreme: net income
above revenue, or negative total assets, can look unremarkable beside the rest
of a small file.

The TL's retest planted ten anomalies in a file of about fifty rows, and the
report showed two. These pin the behaviour: every impossible record surfaces,
and ordinary files report exactly what they did before.
"""

from __future__ import annotations

import pytest

from finsight.analysis.anomaly_detection import _impossible_values

try:
    from scripts.check_agents import DATA, as_plain, load_records
    from api.dependencies import Record, build_orchestrator
except Exception:  # pragma: no cover - the pipeline is not installed in this checkout
    build_orchestrator = None


ORDINARY = dict(revenue=1000.0, gross_profit=400.0, net_income=120.0,
                total_assets=2000.0, total_liabilities=900.0, shareholder_equity=1100.0)


def record(**changes):
    return {**ORDINARY, **changes}


# --- the rules ---------------------------------------------------------------


def test_an_ordinary_record_is_possible():
    assert not _impossible_values(record())


@pytest.mark.parametrize("changes, why", [
    (dict(total_assets=-2000.0), "negative total assets"),
    (dict(total_assets=0.0), "no assets at all"),
    (dict(net_income=1500.0), "net income above revenue"),
    (dict(gross_profit=1800.0), "gross profit above revenue"),
    (dict(total_liabilities=20000.0), "liabilities ten times assets"),
    (dict(shareholder_equity=-10000.0), "equity deeply below minus assets"),
])
def test_each_impossible_figure_is_recognised(changes, why):
    assert _impossible_values(record(**changes)), why


def test_the_equity_name_the_pipeline_actually_sends_is_read():
    # The pipeline sends shareholder_equity; total_equity is only a fallback.
    assert _impossible_values(record(shareholder_equity=-10000.0))
    assert _impossible_values({**record(), "shareholder_equity": None, "total_equity": -10000.0})


@pytest.mark.parametrize("changes, why", [
    (dict(net_income=-400.0), "a loss"),
    (dict(net_income=950.0), "a 95% margin is unusual, not impossible"),
    (dict(shareholder_equity=-500.0), "negative equity smaller than assets"),
    (dict(total_liabilities=4000.0), "leverage of 2x assets"),
    (dict(revenue=0.0, net_income=50.0), "no revenue - nothing to compare against"),
])
def test_unusual_is_not_impossible(changes, why):
    assert not _impossible_values(record(**changes)), why


def test_missing_and_blank_figures_are_not_treated_as_impossible():
    assert not _impossible_values({})
    assert not _impossible_values({"revenue": None, "total_assets": float("nan")})
    assert not _impossible_values({"revenue": "n/a", "total_assets": "unknown"})


# --- through the pipeline ----------------------------------------------------


needs_pipeline = pytest.mark.skipif(build_orchestrator is None, reason="the pipeline is not installed")


def _planted_file():
    base = [as_plain(r) for r in load_records(DATA / "dummy_statements_clean.csv")[:40]]
    template = {k: v for k, v in base[0].items() if k not in ("company", "year")}
    revenue, assets = template["revenue"], template["total_assets"]

    def plant(name, **changes):
        return {**template, "company": name, "year": 2023, **changes}

    impossible = [
        plant("Impossible01", net_income=revenue * 1.5),
        plant("Impossible02", total_liabilities=assets * 10),
        plant("Impossible03", total_assets=-assets),
        plant("Impossible04", gross_profit=revenue * 1.8),
        plant("Impossible05", shareholder_equity=-assets * 5),
    ]
    rows = base + impossible
    records = [Record(**{k: v for k, v in row.items() if v is not None}) for row in rows]
    return records, [row["company"] for row in impossible]


@needs_pipeline
def test_every_impossible_record_surfaces_despite_the_budget():
    records, planted = _planted_file()

    result = build_orchestrator().run(records)

    reported = {finding.company for finding in result.anomalies}
    missing = [company for company in planted if company not in reported]
    assert not missing, f"the alert budget still hid {missing}"


@needs_pipeline
@pytest.mark.parametrize("dataset, expected", [
    ("dummy_statements_clean.csv", 24),
    ("dummy_statements_defective.csv", 24),
])
def test_ordinary_files_report_what_they_did_before(dataset, expected):
    result = build_orchestrator().run(load_records(DATA / dataset))

    assert len(result.anomalies) == expected
