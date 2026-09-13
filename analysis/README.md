# `analysis/`

Trend analysis, and the agents that work across other agents' findings.

| File | What it does | Role |
|:--|:--|:--|
| `trend.py` | Entry point, `run_trend_analysis` | Trend |
| `yoy_analysis.py`, `ratio_analysis.py` | Year-on-year change and financial ratios | Trend |
| `forecasting.py`, `deviation_analysis.py` | Backtested forecasts and materiality-graded deviations | Trend |
| `data_mapping.py`, `trend_agent.py`, `visualization.py` | Column mapping, the agent itself, charts | Trend |
| `anomaly_detection.py` | Connects the Anomaly agent in `finsight/` to the pipeline | Orchestration & API |
| `forensic_flags.py` | Forensic red-flag rules used alongside the anomaly model | Anomaly |
| `recurring_issues.py` | The same finding for the same company in 3 or more years | Orchestration & API |
| `peer_comparison.py` | Company-years far outside their industry on margins, returns, liquidity and leverage | Orchestration & API |

Trend's settings are in `config/`.
