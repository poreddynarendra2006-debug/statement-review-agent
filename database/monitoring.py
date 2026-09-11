"""JSON-ready monitoring data derived from the SQLite database."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .database import _connect, _list_result_json, _list_runs, init_db


def run_summary(limit: int = 50) -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT elapsed_seconds, risk_score, review_mode FROM reviews ORDER BY id DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    elapsed = [r["elapsed_seconds"] for r in rows if r["elapsed_seconds"] is not None]
    scores = [r["risk_score"] for r in rows if r["risk_score"] is not None]
    modes = Counter(r["review_mode"] for r in rows)
    return {
        "total_reviews": len(rows),
        "average_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else 0,
        "slowest_elapsed_seconds": max(elapsed) if elapsed else 0,
        "average_risk_score": sum(scores) / len(scores) if scores else 0,
        "count_by_review_mode": {mode: modes.get(mode, 0) for mode in ("model", "heuristic", "none")},
    }


def stage_performance(limit: int = 50) -> list[dict[str, Any]]:
    rows = _list_runs(limit)
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["stage"], []).append(row["seconds"])
    output = [{"stage": stage, "average_seconds": sum(vals) / len(vals),
               "max_seconds": max(vals), "times_run": len(vals)}
              for stage, vals in groups.items()]
    return sorted(output, key=lambda x: x["max_seconds"], reverse=True)


def recent_reviews(limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT company, period, risk_score, elapsed_seconds, created_at FROM reviews ORDER BY id DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    return [dict(row) for row in rows]


def agent_coverage(limit: int = 50) -> dict[str, Any]:
    import json

    agents: dict[str, dict[str, Any]] = {}
    for payload in _list_result_json(limit):
        result = json.loads(payload)
        coverage = result.get("coverage") or {}
        selected = coverage.get("selected") or []
        skipped = coverage.get("skipped") or {}
        for agent in selected:
            item = agents.setdefault(str(agent), {"ran": 0, "skipped": 0, "skip_reasons": {}})
            item["ran"] += 1
        for agent, reason in skipped.items():
            item = agents.setdefault(str(agent), {"ran": 0, "skipped": 0, "skip_reasons": {}})
            item["skipped"] += 1
            item["skip_reasons"][str(reason)] = item["skip_reasons"].get(str(reason), 0) + 1
    return agents

