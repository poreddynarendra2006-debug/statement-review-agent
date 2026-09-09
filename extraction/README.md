# `extraction/`

**Owner:** Data Ingestion (Member 2)

Reads statement files and converts them into one common record format.

## Files that belong here

| File | What it does |
|:--|:--|
| `normalizer.py` | Defines FinancialRecord and maps any column names onto it. THE contract for the whole project. |
| `csv_loader.py` | Loads CSV files. |
| `excel_loader.py` | Loads .xlsx files. |
| `pdf_parser.py` | Extracts tables from PDF statements. |

## Contract

`FinancialRecord` — every other component consumes this. Do not rename or remove a field without telling the group.

---

*Delete this README once the folder has real files in it.*
