# `extraction/`

**Owner:** Data Ingestion

Reads an uploaded statement and maps its columns onto the common `FinancialRecord` format. The pipeline calls `ingest_financial_statement`.

| File | What it does |
|:--|:--|
| `pipeline.py` | Entry point and the `FinancialRecord` definition |
| `reader.py` | Reads the file |
| `mapper.py` | Maps the file's column names onto standard fields |
| `cleaner.py` | Normalises values: numbers, blanks, currency formatting |
| `validator.py` | Rejects files that are not financial statements |
| `schema.py`, `config.py` | Field definitions and settings |
| `reporter.py`, `cli.py` | Ingestion report and a command-line entry point |

Excel files are converted to CSV by `api/spreadsheets.py` before they reach this code. Column mapping runs without any external service: `mapper.py` can optionally ask a hosted model for help, but that path stays off unless an API key is configured, and by team decision none is.
