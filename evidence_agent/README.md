# `evidence_agent/`

**Owner:** Evidence & Review

Upload your Evidence Agent package **into this folder**:

- `__init__.py`
- `agent.py`
- `models.py`

The app finds it through `agents/evidence_agent.py`, which calls
`compile_all_findings(result)` and reads the packet's `validation_findings`,
`trend_findings`, `anomaly_findings` and `recurring_issues`.

## Please don't

- Upload the `FinSight_Evidence_Review_Submission` wrapper folder - the files
  must sit directly in here, or the imports will not find them
- Upload `__pycache__/`
- Change anything outside this folder and `review_agent/`; your tests belong in
  `tests/evidence_review/`, where they already are
