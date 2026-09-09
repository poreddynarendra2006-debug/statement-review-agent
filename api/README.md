# `api/`

**Owner:** Orchestration & API (Member 1)

Exposes the review engine over HTTP so other software can call it.

## Files that belong here

| File | What it does |
|:--|:--|
| `main.py` | FastAPI application. POST /review and GET /health. |
| `models.py` | Request and response schemas. |

## Contract

`POST /review` — agree the exact response fields with the Frontend before building it.

---

*Delete this README once the folder has real files in it.*
