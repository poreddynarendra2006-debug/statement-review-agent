# `tests/`

Each role's tests are in their own folder. Tests directly in this folder belong to Orchestration & API and cover the orchestrator, API, sign-in, deployment configuration and the checks between components.

| Folder | Role |
|:--|:--|
| `ingestion_tests/` | Data Ingestion |
| `validation_tests/` | Validation |
| `trend/` | Trend |
| `anomaly/` | Anomaly |
| `evidence_review/` | Evidence & Review |
| `risk_tests/` | Risk & Reporting |

`conftest.py` here is shared by the whole suite. CI runs everything on every push.

```bash
python -m pytest
```
