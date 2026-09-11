# `agents/`

**Owners:** Orchestration & API, Evidence & Review

| File | Owner |
|:--|:--|
| `__init__.py`, `orchestrator.py`, `planner.py`, `guardrails.py`, `timing.py` | Orchestration & API - never replace these |
| `evidence_agent.py` | Evidence & Review - `compile_all_findings(result)` returns the list of findings |
| `review_agent.py` | Evidence & Review - `write_review(result)` returns the summary text |

Both Evidence & Review functions receive the whole review result: `result.failed_validations`, `result.material_deviations`, `result.anomalies`, `result.screened_documents`, `result.security_flags`, and so on. `write_review` can also return `(text, mode)`, where `mode` is `"model"` or `"heuristic"`.

## Evidence & Review: upload only

`evidence_agent.py`, `review_agent.py`, and any helper module of yours with a name that doesn't already exist here (for example `prompts.py`). **Never upload an `__init__.py` into this folder.**

## Never upload

trained model files or checkpoint folders (`.bin`, `.safetensors`, `.pt`) - GitHub rejects files over 100 MB; share a link instead. Also `__pycache__/`.

Tests go in `tests/evidence_review/`.
