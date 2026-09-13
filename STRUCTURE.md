# Repository structure

Team 33 · AuditLens, the financial statement review agent

Each part of the review pipeline lives in its own folder, with its tests in a matching folder under `tests/`. The table shows what each folder does and which role built it.

| Folder | What it does | Role | Tests |
|:--|:--|:--|:--|
| `extraction/` | Reads CSV and Excel statements into one record format | Data Ingestion | `tests/ingestion_tests/` |
| `validation_agent/` | Accounting identities and data-quality checks | Validation | `tests/validation_tests/` |
| `analysis/` | Trend analysis, plus the anomaly connector, recurring issues and peer comparison | Trend; Orchestration & API | `tests/trend/`, `tests/` |
| `config/` | Trend settings | Trend | `tests/trend/` |
| `finsight/` | Anomaly detection: Isolation Forest, robust z-scores, explanations | Anomaly | `tests/anomaly/` |
| `evidence_agent/` | Builds the evidence packet from verified findings | Evidence & Review | `tests/evidence_review/` |
| `review_agent/` | Writes the review narrative from the evidence | Evidence & Review | `tests/evidence_review/` |
| `risk_reporting/` | Explainable 0-100 risk score | Risk & Reporting | `tests/risk_tests/` |
| `database/` | Saved reviews, reviewer actions, monitoring | Risk & Reporting | `tests/risk_tests/` |
| `reports/` | PDF review report | Risk & Reporting | `tests/risk_tests/` |
| `ui/` | The reviewer's screens | Frontend | `tests/test_ui_api_contract.py` |
| `agents/` | Orchestrator, planner, guardrails, data quality, groundedness | Orchestration & API | `tests/` |
| `api/` | REST API and sign-in | Orchestration & API | `tests/` |
| `training/` | Fine-tuning the review model, and its measured results | Evidence & Review; Orchestration & API | - |
| `scripts/` | Dataset generator and the manual agent check | Orchestration & API | - |
| `data/` | Sample datasets and the planted answer key | Data Ingestion; Validation | `tests/test_fixture_integrity.py` |
| `docs/` | Design, deployment, model performance, roadmap | Everyone | - |

Deployment files (`Dockerfile`, `entrypoint.sh`, `.github/workflows/`) belong to Orchestration & API.

---

## How a review flows through the folders

`api/` receives the upload, `extraction/` reads it, and `agents/orchestrator.py` runs every agent that has enough data: `validation_agent/`, `analysis/` (trend, recurring, peer) and `finsight/` (anomaly). `evidence_agent/` and `review_agent/` turn the findings into a narrative, `risk_reporting/` scores it, and `database/` and `reports/` save it and produce the PDF.

## Shared data contracts

Every agent exchanges the same objects, so a change to one of them affects the whole pipeline:

| Object | Defined in |
|:--|:--|
| `FinancialRecord` | `extraction/pipeline.py` |
| `ValidationResult` | `validation_agent/models.py` |
| `AnomalyFinding` | `finsight/core/models.py` |
| `AnalysisResult` | `agents/orchestrator.py` |

New fields are added with a default value, so existing agents keep working.

## Running the tests

```bash
python -m pytest
```
