# `database/`

**Owner:** Risk & Reporting

Storage for saved reviews, reviewer actions and run monitoring, in SQLite.

| File | What it does |
|:--|:--|
| `database.py` | Saves and reads reviews, and records reviewer decisions on findings |
| `monitoring.py` | Run statistics: reviews run, durations, risk levels, stage timings |

The database file is set by `SQLITE_DB_PATH`. Reviewer accounts are kept by `api/auth.py`, in their own tables in the same file.
