# `database/`

**Owner:** Risk & Reporting

Saved reviews, reviewer actions and monitoring. The app uses `database.database` and `database.monitoring`.

## Upload exactly these files

`__init__.py`, `database.py`, `monitoring.py`

Review data is written only through these functions. Reviewer accounts are kept separately by `api/auth.py`, in their own tables in the same file.

## Never upload

any `.db` file, `__pycache__/`

Tests go in `tests/risk_tests/`.
