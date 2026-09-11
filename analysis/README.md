# `analysis/`

**Owners:** Trend (its files below), Orchestration & API (`anomaly_detection.py`, `recurring_issues.py`)

| File | Owner |
|:--|:--|
| `__init__.py`, `trend.py`, `yoy_analysis.py`, `ratio_analysis.py`, `forecasting.py`, `deviation_analysis.py`, `data_mapping.py`, `trend_agent.py`, `visualization.py` | Trend - the app calls `run_trend_analysis` from `trend.py` |
| `anomaly_detection.py` | Orchestration & API - connects the Anomaly agent in `finsight/` to the app |
| `recurring_issues.py` | Orchestration & API - finds the same problem for the same company in 3 or more years, from the other agents' findings |

Trend's settings live in `config/`. Trend's tests live in `tests/trend/`.

Only change your own files in this folder. Anomaly's code goes in `finsight/`, and Validation's in `validation_agent/` - not here.
