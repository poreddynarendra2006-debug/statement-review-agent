"""The planner must run what the data supports, and say why it skipped the rest."""

import pytest

from agents.planner import (
    MIN_COMPANIES_FOR_PEER_ANALYSIS,
    Plan,
    Planner,
    Requirement,
    describe,
)
from agents.timing import Timings


@pytest.fixture
def planner():
    """A planner with one tool per requirement, each returning a marker."""
    p = Planner()
    p.register("validation", lambda r: "validated",
               requires=Requirement.STATEMENT_DETAIL,
               description="Five accounting identity checks")
    p.register("trend", lambda r: "trended",
               requires=Requirement.MULTIPLE_PERIODS,
               description="Year-on-year movement and ratios")
    p.register("recurring", lambda r: "recurred",
               requires=Requirement.RECURRENCE_WINDOW)
    p.register("peer_anomaly", lambda r: "compared",
               requires=Requirement.PEER_GROUP)
    p.register("always", lambda r: "ran", requires=Requirement.NONE)
    return p


# --- describing the submission ------------------------------------------


def test_describe_reads_the_submission(financial_records):
    facts = describe(financial_records)
    assert facts["records"] == 4
    assert facts["companies"] == 1
    assert facts["periods"] == 4
    assert facts["year_range"] == (2020, 2023)
    assert facts["has_statement_detail"] is True


def test_describe_notices_missing_statement_detail(kaggle_only_records):
    facts = describe(kaggle_only_records)
    assert facts["has_statement_detail"] is False
    assert facts["records_with_statement_detail"] == 0


# --- selection ----------------------------------------------------------


def test_full_records_select_everything_but_peer_analysis(planner, financial_records):
    plan = planner.plan(financial_records)

    assert "validation" in plan.selected
    assert "trend" in plan.selected
    assert "recurring" in plan.selected
    assert "always" in plan.selected
    # One company is not a peer group.
    assert "peer_anomaly" in plan.skipped


def test_accounting_checks_are_skipped_without_statement_detail(planner, kaggle_only_records):
    """The case that matters: Kaggle rows must not fail the accounting rules."""
    plan = planner.plan(kaggle_only_records)

    assert "validation" not in plan.selected
    assert "validation" in plan.skipped
    assert "balance sheet" in plan.skipped["validation"]
    # Everything the data does support still runs.
    assert "trend" in plan.selected


def test_single_period_skips_trend_analysis(planner, financial_records):
    plan = planner.plan(financial_records[:1])

    assert "trend" in plan.skipped
    assert "at least 2 periods" in plan.skipped["trend"]
    assert "validation" in plan.selected, "one period is still checkable"


def test_two_periods_allow_trend_but_not_recurrence(planner, financial_records):
    plan = planner.plan(financial_records[:2])

    assert "trend" in plan.selected
    assert "recurring" in plan.skipped


def test_peer_analysis_needs_a_peer_group(planner, financial_records):
    """Enough companies, and the peer tool becomes available."""
    from dataclasses import replace

    many = [
        replace(record, company=f"Company {i}")
        for i in range(MIN_COMPANIES_FOR_PEER_ANALYSIS)
        for record in financial_records[:2]
    ]
    plan = planner.plan(many)
    assert "peer_anomaly" in plan.selected


def test_empty_submission_skips_everything_with_a_reason(planner):
    plan = planner.plan([])

    assert plan.selected == []
    assert set(plan.skipped) == set(planner.tools)
    assert all("no records" in reason for reason in plan.skipped.values())


def test_only_restricts_consideration(planner, financial_records):
    plan = planner.plan(financial_records, only=["validation", "always"])

    assert set(plan.selected) == {"validation", "always"}
    assert "trend" not in plan.selected and "trend" not in plan.skipped


def test_unknown_tool_is_rejected(planner, financial_records):
    with pytest.raises(KeyError, match="unknown tool"):
        planner.plan(financial_records, only=["nonexistent"])


# --- execution ----------------------------------------------------------


def test_run_executes_only_the_selected_tools(planner, kaggle_only_records):
    results, plan = planner.run(kaggle_only_records)

    assert "validation" not in results
    assert results["trend"] == "trended"
    assert set(results) == set(plan.selected)


def test_a_failing_agent_does_not_stop_the_others(planner, financial_records):
    """A broken trend agent must not take the accounting checks down with it."""
    def explode(records):
        raise RuntimeError("component blew up")

    planner.register("trend", explode, requires=Requirement.MULTIPLE_PERIODS)
    results, plan = planner.run(financial_records)

    assert "trend" not in plan.selected
    assert "RuntimeError" in plan.skipped["trend"]
    assert results["validation"] == "validated"


def test_run_records_timings_when_given_them(planner, financial_records):
    timings = Timings()
    planner.run(financial_records, timings=timings)

    recorded = dict(timings.stages)
    assert "validation" in recorded
    assert "trend" in recorded


# --- reporting ----------------------------------------------------------


def test_tool_schemas_describe_every_registered_tool(planner):
    schemas = planner.tool_schemas()

    assert len(schemas) == len(planner.tools)
    validation = next(s for s in schemas if s["name"] == "validation")
    assert validation["requires"] == "statement_detail"
    assert "accounting" in validation["description"]


def test_registering_the_same_name_replaces_it(planner, financial_records):
    planner.register("always", lambda r: "replaced")
    results, _ = planner.run(financial_records)
    assert results["always"] == "replaced"


def test_plan_serialises_for_logging(planner, kaggle_only_records):
    payload = planner.plan(kaggle_only_records).to_dict()

    assert "selected" in payload and "skipped" in payload
    assert payload["facts"]["companies"] == 1


def test_plan_summary_is_readable(planner, kaggle_only_records):
    summary = planner.plan(kaggle_only_records).summary()
    assert summary.startswith("ran ")
    assert "skipped" in summary


def test_empty_plan_summary_does_not_crash():
    assert Plan().summary() == "ran nothing"


# --- run options --------------------------------------------------------


def test_options_reach_only_the_tools_that_accept_them(financial_records):
    seen = {}

    def validation(records, materiality=0.05):
        seen["validation"] = materiality
        return "ok"

    def anomaly(records):
        seen["anomaly"] = "called without options"
        return "ok"

    planner = Planner()
    planner.register("validation", validation, requires=Requirement.STATEMENT_DETAIL)
    planner.register("anomaly", anomaly)
    _, plan = planner.run(financial_records, options={"materiality": 0.01})

    assert seen == {"validation": 0.01, "anomaly": "called without options"}
    assert plan.skipped == {}, "a tool that ignores options must not be marked as failed"


def test_without_options_tools_keep_their_defaults(financial_records):
    planner = Planner()
    planner.register("validation", lambda records, materiality=0.05: materiality,
                     requires=Requirement.STATEMENT_DETAIL)
    results, _ = planner.run(financial_records)
    assert results["validation"] == 0.05
