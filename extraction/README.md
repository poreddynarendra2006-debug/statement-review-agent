# `extraction/`

**Owner:** Data Ingestion

Reads an uploaded statement file and maps its columns onto `FinancialRecord`. The app calls `ingest_financial_statement` from here.

## Upload exactly these files

The files inside your `src/ingestion/` folder - not the folder itself:

`__init__.py`, `cleaner.py`, `cli.py`, `config.py`, `mapper.py`, `pipeline.py`, `reader.py`, `reporter.py`, `schema.py`, `validator.py`

Excel uploads are converted to CSV by the API before they reach this code, so it only needs to read CSV.

## Never upload

`data/`, `requirements.txt`, `README.md`, `.gitignore`, `__pycache__/`

Tests go in `tests/ingestion_tests/`.
