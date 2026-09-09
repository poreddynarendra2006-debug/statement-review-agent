# `analysis/`

**Owner:** Validation (Member 3), Trend (Member 4), Anomaly (Member 5), Risk (Member 8)

All the calculations. No AI model may be imported into this folder except the anomaly model.

## Files that belong here

| File | What it does |
|:--|:--|
| `validation.py` | Member 3 — the five accounting checks. Emits ValidationResult. |
| `yoy_analysis.py` | Member 4 — year-on-year movement. Emits YoYResult. |
| `ratios.py` | Member 4 — liquidity, leverage, profitability, returns. |
| `anomaly_detection.py` | Member 5 — contradictions and the trained outlier model. Emits AnomalyFinding. |
| `recurring_issues.py` | Member 5 — problems repeating across years. |
| `risk_scoring.py` | Member 8 — the explainable 0-100 score. |

## Contract

`ValidationResult`, `YoYResult`, `AnomalyFinding` — three separate owners share this folder, so keep to your own files.

---

*Delete this README once the folder has real files in it.*
