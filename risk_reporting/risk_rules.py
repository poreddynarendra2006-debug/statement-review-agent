"""Risk severity weights and score bands."""

SEVERITY_SCORES = {
    "NONE": 0,
    "LOW": 4,
    "MEDIUM": 7,
    "HIGH": 18,
    "CRITICAL": 40,
}


def get_risk_level(score: int) -> str:
    """Map a 0-100 risk score to a human-readable risk level."""
    score = max(0, min(100, int(score)))
    if score < 25:
        return "LOW"
    if score < 50:
        return "MEDIUM"
    if score < 75:
        return "HIGH"
    return "CRITICAL"
