import json
from dataclasses import dataclass

from risk_reporting.risk_engine import RiskScoreResult, calculate_risk


@dataclass
class ValidationResult:
    rule_id: str
    rule_name: str
    year: int
    status: str
    expected: float
    actual: float
    difference: float
    severity: str
    evidence: str
    formula: str
    message: str


@dataclass
class AnomalyFinding:
    company: str
    year: int
    anomaly_type: str
    score: float
    severity: str
    confidence: float
    record_id: str
    relevant_features: list
    deviations: dict
    explanation: str
    recommendation: str
    model_name: str
    percentile: float


@dataclass
class DeviationRecord:
    company: str
    year: int
    metric: str
    actual: float
    forecast: float
    deviation: float
    deviation_percent: float
    direction: str
    material_deviation: bool
    materiality_threshold: float


def test_clean_input_scores_zero():
    result = calculate_risk({
        "validation": [],
        "anomaly": [],
        "deviation": [],
        "ratio": [],
    })
    assert result.score == 0
    assert result.risk_level == "LOW"
    assert result.contributors == []


def test_one_critical_scores_higher_than_one_low():
    low = AnomalyFinding("Acme", 2025, "small", 0.1, "LOW", .8, "L1", [], {}, "small", "", "m", 10)
    critical = ValidationResult("R1", "Critical mismatch", 2025, "FAIL", 100, 0, 100, "CRITICAL", "x", "a-b", "critical")
    assert calculate_risk({"anomaly": [low]}).score < calculate_risk({"validation": [critical]}).score


def test_115_medium_differs_from_3_critical():
    medium = [
        AnomalyFinding("Acme", 2025, "medium", .5, "MEDIUM", .8, f"M{i}", [], {}, "medium", "", "m", 50)
        for i in range(115)
    ]
    critical = [
        ValidationResult(f"R{i}", "Critical", 2025, "FAIL", 100, 0, 100, "CRITICAL", "x", "a-b", "critical")
        for i in range(3)
    ]
    medium_result = calculate_risk({"anomaly": medium})
    critical_result = calculate_risk({"validation": critical})
    assert medium_result.score != critical_result.score
    assert critical_result.score > medium_result.score


def test_pass_and_skipped_validations_do_not_contribute():
    items = [
        ValidationResult("R1", "Pass", 2025, "PASS", 100, 100, 0, "CRITICAL", "x", "a=b", "ok"),
        ValidationResult("R2", "Skipped", 2025, "SKIPPED", 100, 0, 0, "CRITICAL", "x", "a=b", "skip"),
    ]
    result = calculate_risk({"validation": items})
    assert result.score == 0
    assert result.contributors == []


def test_non_material_deviation_does_not_contribute():
    item = DeviationRecord("Acme", 2025, "revenue", 110, 100, 10, 10, "UP", False, 20)
    result = calculate_risk({"deviation": [item]})
    assert result.score == 0


def test_contributor_points_sum_to_total():
    findings = [
        AnomalyFinding("Acme", 2025, "a", .5, "HIGH", .9, "A1", [], {}, "a", "", "m", 90),
        AnomalyFinding("Acme", 2025, "b", .4, "MEDIUM", .9, "A2", [], {}, "b", "", "m", 80),
        DeviationRecord("Acme", 2025, "revenue", 150, 100, 50, 50, "UP", True, 20),
    ]
    result = calculate_risk({"anomaly": findings[:2], "deviation": [findings[2]]})
    assert sum(c.points for c in result.contributors) == result.score
    assert all(c.reason and c.source_agent and c.finding_reference is not None for c in result.contributors)


def test_result_serialises_to_json():
    finding = AnomalyFinding("Acme", 2025, "outlier", .9, "HIGH", .95, "A1", ["revenue"], {"revenue": 4}, "outlier", "review", "model", 99)
    result = calculate_risk({"anomaly": [finding]})
    payload = result.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["score"] == result.score
    assert decoded["contributors"][0]["source_agent"] == "anomaly"
    assert isinstance(result, RiskScoreResult)


def test_legacy_plain_dicts_still_work():
    result = calculate_risk({
        "validation": [{
            "finding": "Balance sheet mismatch",
            "severity": "HIGH",
            "evidence": "Difference = 5 Cr",
        }]
    })
    assert result.score > 0
    assert result.contributors[0].finding_reference["finding"] == "Balance sheet mismatch"


def _critical_anomalies(count):
    return [
        AnomalyFinding("Acme", 2025, "critical", .99, "CRITICAL", .99,
                       f"C{i}", [], {}, "critical finding", "review", "model", 99)
        for i in range(count)
    ]


def _critical_failures(count):
    """Failed accounting identities - errors of fact, so they carry no ceiling.

    The tests below are about the engine itself: that it clamps at 100, that
    severity beats sheer count, that adding a finding never lowers the score.
    They used anomalies simply as a convenient finding. Anomalies are now held
    to a ceiling of their own, because a budgeted detector returns findings on
    any file and an uncapped one made clean books read CRITICAL - so they can
    no longer reach 100 alone, and these properties are stated with a source
    that can. The ceiling itself is tested in test_statistical_caps.py.
    """
    return [
        ValidationResult(f"R{i}", "Critical mismatch", 2025, "FAIL", 100, 0, 100,
                         "CRITICAL", "x", "a-b", "critical")
        for i in range(count)
    ]


def test_1000_critical_findings_are_clamped_to_100():
    result = calculate_risk({"validation": _critical_failures(1000)})
    assert result.score == 100


def test_100_critical_findings_are_clamped_to_100():
    result = calculate_risk({"validation": _critical_failures(100)})
    assert result.score == 100


def test_score_is_always_within_zero_and_100():
    for count in [0, 1, 3, 10, 100, 1000, 5000]:
        result = calculate_risk({"anomaly": _critical_anomalies(count)})
        assert 0 <= result.score <= 100


def test_clamped_contributor_points_sum_exactly_to_100():
    result = calculate_risk({"validation": _critical_failures(1000)})
    assert result.score == 100
    assert sum(c.points for c in result.contributors) == result.score


def test_three_critical_still_outscore_115_medium():
    medium = [
        ValidationResult(f"M{i}", "Medium mismatch", 2025, "FAIL", 50, 40, 10,
                         "MEDIUM", "x", "a-b", "medium")
        for i in range(115)
    ]
    critical = _critical_failures(3)
    medium_score = calculate_risk({"validation": medium}).score
    critical_score = calculate_risk({"validation": critical}).score
    assert critical_score > medium_score


def test_adding_a_finding_never_lowers_score():
    previous = 0
    for count in range(1, 151):
        score = calculate_risk({"anomaly": _critical_anomalies(count)}).score
        assert score >= previous
        previous = score
