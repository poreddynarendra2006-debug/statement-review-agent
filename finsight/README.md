# `finsight/`

**Owner:** Anomaly

The anomaly detection package. The app reaches it through `analysis/anomaly_detection.py`, which Orchestration & API maintains.

## Upload exactly these

What is inside your `finsight/` folder - not the folder itself:

- `__init__.py`, `__main__.py`
- the `analysis/`, `core/` and `data/` folders

The `data/` folder **inside** `finsight/` is code (`loader.py`), so upload it. The separate top-level `data/` folder with CSV files is not.

## Never upload

`models/` (`.joblib` files - the model trains on each upload), the top-level `data/` folder, `findings_*.json`, `run_*.json`, `run_*.txt`, `console_*.txt`, `.pytest_cache/`, `pytest.ini`, `requirements.txt`, `README.md`, `.gitignore`, `__pycache__/`

Tests go in `tests/anomaly/`.
