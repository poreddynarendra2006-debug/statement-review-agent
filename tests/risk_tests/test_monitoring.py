import pytest

from database import database, monitoring


def payload(company="Acme", score=42, elapsed=3.5, mode="model", timings=None, coverage=None):
    return {
        "company": company, "period": "FY2020 - FY2023", "currency": "USD", "record_count": 10,
        "risk_score": score, "risk_result": {"score": score, "risk_level": "MEDIUM", "badge_color": "yellow", "summary": "x", "contributors": []},
        "validation_results": [], "failed_validations": [], "yoy_results": [], "ratio_results": [], "forecasts": [],
        "evaluations": [], "deviations": [], "material_deviations": [], "anomalies": [], "recurring_issues": [], "findings": [],
        "ai_summary": "x", "review_mode": mode,
        "coverage": coverage or {"selected": ["validation"], "skipped": {}, "facts": {}},
        "timings": timings or {"fast": 1.0, "slow": 4.0}, "elapsed_seconds": elapsed,
        "warnings": [], "security_flags": [],
    }


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "monitor.sqlite"))
    database.init_db()


def test_run_summary_empty_is_zero(temp_db):
    result = monitoring.run_summary()
    assert result["total_reviews"] == 0
    assert result["average_elapsed_seconds"] == 0
    assert result["slowest_elapsed_seconds"] == 0
    assert result["average_risk_score"] == 0
    assert result["count_by_review_mode"] == {"model": 0, "heuristic": 0, "none": 0}


def test_stage_performance_slowest_first(temp_db):
    database.save_review(payload("A", timings={"fast": 1.0, "slow": 4.0}))
    database.save_review(payload("B", timings={"fast": 2.0, "slow": 8.0}))
    stages = monitoring.stage_performance()
    assert stages[0]["stage"] == "slow"
    assert stages[0]["max_seconds"] == 8.0


def test_agent_coverage_counts_skipped_agent(temp_db):
    coverage = {"selected": ["validation"], "skipped": {"forecast": "model unavailable"}, "facts": {}}
    database.save_review(payload(coverage=coverage))
    result = monitoring.agent_coverage()
    assert result["forecast"]["ran"] == 0
    assert result["forecast"]["skipped"] == 1
    assert result["forecast"]["skip_reasons"]["model unavailable"] == 1
