import io

import pytest

reportlab = pytest.importorskip("reportlab")

from reports.report_generator import DISCLAIMER, generate_report


def _result(full=True):
    if not full:
        return {
            "company": "Empty Co", "period": "FY2024", "currency": "USD", "record_count": 0,
            "risk_score": None, "risk_result": None, "validation_results": [], "failed_validations": [],
            "yoy_results": [], "ratio_results": [], "forecasts": [], "evaluations": [], "deviations": [],
            "material_deviations": [], "anomalies": [], "recurring_issues": [], "findings": [],
            "ai_summary": "", "review_mode": "none", "coverage": {"selected": [], "skipped": {}, "facts": {}},
            "timings": {}, "elapsed_seconds": 0, "warnings": [], "security_flags": [],
        }
    failed = {"rule_id": "R1", "rule_name": "Revenue check", "year": 2024, "status": "FAIL",
              "expected": 100, "actual": 90, "difference": 10, "severity": "HIGH",
              "evidence": "Ledger mismatch", "formula": "a-b", "message": "Potential inconsistency"}
    return {
        "company": "Acme Corp", "period": "FY2020 - FY2023", "currency": "USD", "record_count": 100,
        "risk_score": 61,
        "risk_result": {"score": 61, "risk_level": "HIGH", "badge_color": "orange",
                         "summary": "Potential inconsistencies require review.", "contributors": [
                             {"reason": "Revenue check failed", "points": 40, "source_agent": "validation", "finding_reference": {"rule_id": "R1"}},
                             {"reason": "Anomaly", "points": 21, "source_agent": "anomaly", "finding_reference": {"record_id": "A1"}},
                         ]},
        "validation_results": [failed], "failed_validations": [failed], "yoy_results": [], "ratio_results": [],
        "forecasts": [], "evaluations": [], "deviations": [],
        "material_deviations": [{"company": "Acme Corp", "year": 2023, "metric": "Revenue", "actual": 80, "forecast": 100,
                                 "deviation": -20, "deviation_percent": -20, "direction": "DOWN", "material_deviation": True,
                                 "materiality_threshold": 10}],
        "anomalies": [{"company": "Acme Corp", "year": 2023, "anomaly_type": "Outlier", "score": .9,
                       "severity": "HIGH", "confidence": .95, "record_id": "A1", "explanation": "Potential inconsistency"}],
        "recurring_issues": [], "findings": [], "ai_summary": "Review summary.", "review_mode": "heuristic",
        "coverage": {"selected": ["validation", "anomaly"], "skipped": {"forecast": "No forecast model configured"},
                     "facts": {"records": 100, "companies": 1, "periods": 4}},
        "timings": {"validation": 1.2, "_total": 2.0}, "elapsed_seconds": 2.0, "warnings": [], "security_flags": [],
    }


def _extract(pdf_bytes):
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_full_result_produces_pdf():
    pdf = generate_report(_result(), "Reviewer")
    assert pdf and pdf.startswith(b"%PDF")


def test_empty_result_produces_pdf():
    pdf = generate_report(_result(False))
    assert pdf and pdf.startswith(b"%PDF")


def test_skipped_agents_and_reasons_appear():
    text = _extract(generate_report(_result()))
    assert "forecast" in text
    assert "No forecast model configured" in text


def test_disclaimer_appears():
    text = _extract(generate_report(_result()))
    assert DISCLAIMER in text
