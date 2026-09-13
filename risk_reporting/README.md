# `risk_reporting/`

**Owner:** Risk & Reporting

The explainable 0-100 risk score, broken down by the findings that drove it.

| File | What it does |
|:--|:--|
| `risk_engine.py` | `calculate_risk` combines failed checks, anomalies and material deviations into one score |
| `sample_data/risk_rules.py` | Points per severity, and the score bands for each risk level |
| `sample_data/agent_outputs.json` | Sample agent output for demonstrating the engine |

Severity outweighs volume, and statistical findings are capped so they can raise the score but not dominate it.
