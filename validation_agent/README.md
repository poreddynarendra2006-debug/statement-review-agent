# `validation_agent/`

**Owner:** Validation

Rule-based checks: accounting identities such as assets = liabilities + equity, and data-quality checks.

| File | What it does |
|:--|:--|
| `validator.py` | Entry point, `run_all_validations` |
| `accounting.py` | The accounting identities |
| `checks.py` | Schema, missing values, duplicates and domain checks |
| `rules.py` | Rule definitions and limits |
| `models.py` | `ValidationResult` |

The data-quality checks join the pipeline through `agents/data_quality.py`.
