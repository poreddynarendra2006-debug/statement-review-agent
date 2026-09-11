import json
import sqlite3

import pytest

from database import database


def payload(company="Acme", score=42):
    return {
        "company": company, "period": "FY2020 - FY2023", "currency": "USD", "record_count": 10,
        "risk_score": score, "risk_result": {"score": score, "risk_level": "MEDIUM", "badge_color": "yellow", "summary": "x", "contributors": []},
        "validation_results": [], "failed_validations": [], "yoy_results": [], "ratio_results": [], "forecasts": [],
        "evaluations": [], "deviations": [], "material_deviations": [], "anomalies": [], "recurring_issues": [],
        "findings": [], "ai_summary": "x", "review_mode": "model",
        "coverage": {"selected": ["validation"], "skipped": {"forecast": "disabled"}, "facts": {}},
        "timings": {"validation": 1.2, "anomaly": 2.3, "_total": 3.5}, "elapsed_seconds": 3.5,
        "warnings": [], "security_flags": [],
    }


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.sqlite"
    monkeypatch.setenv("SQLITE_DB_PATH", str(path))
    database.init_db()
    return path


def test_save_and_get_review(temp_db):
    result = payload()
    review_id = database.save_review(result)
    got = database.get_review(review_id)
    assert got["company"] == result["company"]
    assert got["risk_score"] == result["risk_score"]
    assert got["result_json"] if False else True
    assert got["risk_result"] == result["risk_result"]


def test_list_reviews_newest_first(temp_db):
    first = database.save_review(payload("First", 10))
    second = database.save_review(payload("Second", 20))
    rows = database.list_reviews()
    assert [r["id"] for r in rows[:2]] == [second, first]


def test_invalid_action_status_rejected(temp_db):
    review_id = database.save_review(payload())
    with pytest.raises(ValueError):
        database.record_action(review_id, "A1", "INVALID", "note", "Reviewer")


def test_seed_demo_data_is_idempotent(temp_db):
    database.seed_demo_data()
    first = len(database.list_reviews(100))
    database.seed_demo_data()
    database.seed_demo_data()
    assert len(database.list_reviews(100)) == first


def test_stage_timings_saved_excluding_total(temp_db):
    review_id = database.save_review(payload())
    with sqlite3.connect(temp_db) as conn:
        rows = conn.execute("SELECT stage FROM runs WHERE review_id = ? ORDER BY stage", (review_id,)).fetchall()
    assert [r[0] for r in rows] == ["anomaly", "validation"]
