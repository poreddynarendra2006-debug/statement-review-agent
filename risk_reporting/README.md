# `risk_reporting/`

**Owner:** Risk & Reporting

The explainable 0-100 risk score. The app calls `calculate_risk` from `risk_engine.py`.

## Upload exactly these

What is inside your `risk_reporting/` folder - not the folder itself:

- `__init__.py`, `risk_engine.py`
- the `sample_data/` folder: `__init__.py`, `agent_outputs.json`, `risk_rules.py`

Your `database/` and `reports/` files go in those folders. Tests go in `tests/risk_tests/`.

## Never upload

the extra `risk_engine.py` from your top-level folder, `requirements.txt`, any `.db` file, `__pycache__/`
