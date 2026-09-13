# `data/`

Datasets used for development, tests and the demo.

| File | What it is |
|:--|:--|
| `dummy_statements_clean.csv` | 480 company-years of synthetic statements with no planted errors |
| `dummy_statements_defective.csv` | The same companies with errors planted for testing |
| `dummy_statements_labels.json` | The answer key: which errors were planted, and where |
| `kaggle_financial_statements.csv` | Real financial statements of major companies, 2009-2023, from Kaggle |

The synthetic files come from `scripts/generate_dummy_statements.py`. No client data is stored in this repository.
