# `validation_agent/`

**Owner:** Validation

The accounting identity and ratio checks. The app calls `run_all_validations` from here.

## Upload exactly these files

The files inside your `validation_agent/validation_agent/` folder - not the folder itself:

`__init__.py`, `accounting.py`, `checks.py`, `models.py`, `rules.py`, `validator.py`

## Never upload

`data/`, `outputs/`, `synthetic_data/`, `evaluate.py`, `validation_agent.zip`, `.env.example`, `DATASET_ANALYSIS.md`, `README.md`, `requirements.txt`, `__pycache__/`

Tests go in `tests/validation_tests/`.
