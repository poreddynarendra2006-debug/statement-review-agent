# `tests/`

**Owner of the files directly in this folder:** Orchestration & API

Each role has its own folder in here. **Never upload test files straight into `tests/`.** A file with the same name replaces the existing one - `conftest.py` in this folder is shared by every test in the project, and replacing it breaks them all.

## Your folder

| Role | Folder | Files from your project's `tests/` folder |
|:--|:--|:--|
| Data Ingestion | `ingestion_tests/` | `__init__.py` and your 7 `test_*.py` files |
| Validation | `validation_tests/` | `__init__.py` and your 6 `test_*.py` files |
| Trend | `trend/` | `conftest.py` and your 6 `test_*.py` files |
| Anomaly | `anomaly/` | `__init__.py` and your 10 `test_*.py` files |
| Evidence & Review | `evidence_review/` | your `test_*.py` files, and `conftest.py` if you have one |
| Risk & Reporting | `risk_tests/` | `__init__.py` and your 4 `test_*.py` files |

Your own `conftest.py` belongs in **your** folder. It applies to your tests only, alongside the shared one here.

## Running everything

```bash
python -m pytest
```
