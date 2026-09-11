# `database/`

**Owner:** Risk & Reporting

Saved reviews, reviewer actions and monitoring. The app uses `database.database` and `database.monitoring`.

## Upload exactly these files

`__init__.py`, `database.py`, `monitoring.py`

Nothing outside this folder writes SQL; everything goes through these functions.

## Never upload

any `.db` file, `__pycache__/`

Tests go in `tests/risk_tests/`.
