# `review_agent/`

**Owner:** Evidence & Review

Upload your Review Agent package **into this folder**:

- `__init__.py`
- `agent.py`
- `models.py`
- `prompt.py`

Your own `README.md` can replace this file.

The app finds it through `agents/review_agent.py`, which calls
`generate_review(packet)` and reads `summary` and `generation_mode` from what
comes back.

## Please don't

- Upload the `FinSight_Evidence_Review_Submission` wrapper folder - the files
  must sit directly in here, or the imports will not find them
- Upload `__pycache__/`
- Change anything outside this folder and `evidence_agent/`; your tests belong
  in `tests/evidence_review/`, where they already are
