"""The sole SQL-writing layer for review persistence."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DB_ENV = "SQLITE_DB_PATH"
DEFAULT_DB = "finsight_review.db"
VALID_ACTIONS = {"VERIFIED", "NEEDS_INVESTIGATION", "DISMISSED"}


def _path() -> str:
    return os.environ.get(DB_ENV, DEFAULT_DB)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            company TEXT NOT NULL,
            period TEXT,
            record_count INTEGER,
            risk_score INTEGER,
            risk_level TEXT,
            review_mode TEXT,
            elapsed_seconds REAL,
            findings_count INTEGER NOT NULL,
            failed_checks_count INTEGER NOT NULL,
            filename TEXT,
            owner_id INTEGER,
            result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviewer_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            finding_ref TEXT,
            status TEXT NOT NULL,
            note TEXT,
            reviewer TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(review_id) REFERENCES reviews(id)
        );
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            stage TEXT NOT NULL,
            seconds REAL,
            FOREIGN KEY(review_id) REFERENCES reviews(id),
            UNIQUE(review_id, stage)
        );
        """)
        _add_missing_columns(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Bring an older database up to the current columns.

    A database created before a column existed is not recreated by
    CREATE TABLE IF NOT EXISTS, so the column is added here instead. Without
    it, a deployment that kept its file would fail every insert.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(reviews)")}
    if "filename" not in existing:
        conn.execute("ALTER TABLE reviews ADD COLUMN filename TEXT")
    # The account that ran the review. Empty for the demo reviews, which every
    # reviewer sees; set for uploads, which only their owner sees.
    if "owner_id" not in existing:
        conn.execute("ALTER TABLE reviews ADD COLUMN owner_id INTEGER")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_review(result: dict) -> int:
    init_db()
    result = result or {}
    risk = result.get("risk_result") or {}
    timings = result.get("timings") or {}
    created_at = _now()
    payload = json.dumps(result, ensure_ascii=False, default=str)
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO reviews
            (created_at, company, period, record_count, risk_score, risk_level,
             review_mode, elapsed_seconds, findings_count, failed_checks_count,
             filename, owner_id, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (created_at, result.get("company", ""), result.get("period"),
             result.get("record_count"), risk.get("score", result.get("risk_score")),
             risk.get("risk_level"), result.get("review_mode"), result.get("elapsed_seconds"),
             len(result.get("findings") or []), len(result.get("failed_validations") or []),
             result.get("filename") or "", result.get("owner_id"), payload),
        )
        review_id = int(cur.lastrowid)
        for stage, seconds in timings.items():
            if stage == "_total":
                continue
            conn.execute(
                "INSERT OR IGNORE INTO runs (review_id, created_at, stage, seconds) VALUES (?, ?, ?, ?)",
                (review_id, created_at, str(stage), seconds),
            )
        return review_id


def get_review(review_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    if row is None:
        return None
    data = json.loads(row["result_json"])
    data["id"] = row["id"]
    data["created_at"] = row["created_at"]
    data["owner_id"] = row["owner_id"]
    return data


def list_reviews(limit: int = 50, owner_id: int | None = None) -> list[dict[str, Any]]:
    """Newest first. Given an owner, only their reviews and the shared demo ones.

    Filtered in SQL rather than afterwards, so a reviewer asking for 20 gets
    20 of their own, not 20 of everyone's with most of them removed.
    """
    init_db()
    columns = """id, created_at, company, period, record_count, risk_score,
                 risk_level, review_mode, elapsed_seconds, findings_count,
                 failed_checks_count, filename"""
    with _connect() as conn:
        if owner_id is None:
            rows = conn.execute(
                f"SELECT {columns} FROM reviews ORDER BY id DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT {columns} FROM reviews
                    WHERE owner_id = ? OR owner_id IS NULL
                    ORDER BY id DESC LIMIT ?""",
                (int(owner_id), max(0, int(limit))),
            ).fetchall()
    return [dict(row) for row in rows]


def record_action(review_id: int, finding_ref: Any, status: str, note: str, reviewer: str) -> int:
    if status not in VALID_ACTIONS:
        raise ValueError(f"Invalid action status: {status}")
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO reviewer_actions
            (review_id, finding_ref, status, note, reviewer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (review_id, json.dumps(finding_ref, ensure_ascii=False, default=str), status,
             note, reviewer, _now()),
        )
        return int(cur.lastrowid)


def get_actions(review_id: int) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, review_id, finding_ref, status, note, reviewer, created_at FROM reviewer_actions WHERE review_id = ? ORDER BY id",
            (review_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["finding_ref"] = json.loads(item["finding_ref"])
        except (TypeError, json.JSONDecodeError):
            pass
        result.append(item)
    return result


def _demo(company: str, period: str, score: int, level: str, mode: str, elapsed: float, timings: dict[str, float]) -> dict:
    return {
        "company": company, "period": period, "currency": "USD", "record_count": 1200,
        "risk_score": score,
        "risk_result": {"score": score, "risk_level": level, "badge_color": "red" if level == "CRITICAL" else "orange",
                        "summary": "Example review with potential inconsistencies.", "contributors": []},
        "validation_results": [], "failed_validations": [], "yoy_results": [], "ratio_results": [],
        "forecasts": [], "evaluations": [], "deviations": [], "material_deviations": [], "anomalies": [],
        "recurring_issues": [], "findings": [], "ai_summary": "Example monitoring review.",
        "review_mode": mode,
        "coverage": {"selected": ["validation", "trend", "anomaly"], "skipped": {},
                     "facts": {"records": 1200, "companies": 1, "periods": 4}},
        "timings": timings, "elapsed_seconds": elapsed, "warnings": [], "security_flags": []
    }


def seed_demo_data() -> None:
    init_db()
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    if count:
        return
    demos = [
        _demo("Northstar Manufacturing", "FY2022 - FY2025", 28, "MEDIUM", "model", 4.8,
               {"validation": 1.2, "anomaly": 2.1, "report": 1.0}),
        _demo("BlueRiver Retail", "FY2023 - FY2025", 61, "HIGH", "heuristic", 7.3,
               {"validation": 1.8, "anomaly": 3.7, "trend": 1.1}),
    ]
    for demo in demos:
        save_review(demo)


def _list_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Internal read helper used by monitoring; SQL remains in this module."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT r.stage, r.seconds FROM runs r
               JOIN reviews v ON v.id = r.review_id
               WHERE v.id IN (SELECT id FROM reviews ORDER BY id DESC LIMIT ?)""",
            (max(0, int(limit)),),
        ).fetchall()
    return [dict(row) for row in rows]


def _list_result_json(limit: int = 50) -> list[str]:
    """Internal read helper used by monitoring; SQL remains in this module."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT result_json FROM reviews ORDER BY id DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    return [row["result_json"] for row in rows]
