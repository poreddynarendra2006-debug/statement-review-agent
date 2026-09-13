# `reports/`

**Owner:** Risk & Reporting

The downloadable PDF review report.

| File | What it does |
|:--|:--|
| `report_generator.py` | Builds the PDF from a saved review: summary, risk breakdown, failed checks, deviations and anomalies, recurring issues, peer comparison, coverage |

Served by `GET /reviews/{id}/report.pdf`.
