# `api/`

**Owner:** Orchestration & API

The REST API. It serves the reviewer's screens from the same address, so there is one service to deploy.

| File | What it does |
|:--|:--|
| `main.py` | The FastAPI application and every endpoint |
| `models.py` | Request and response schemas |
| `auth.py` | Reviewer accounts, sign-in and tokens (PBKDF2-SHA256 password hashes) |
| `dependencies.py` | Wires each agent into the orchestrator |
| `spreadsheets.py` | Converts Excel uploads to CSV for ingestion |

## Endpoints

| Method | Path | Purpose |
|:--|:--|:--|
| `POST` | `/auth/register`, `/auth/login`, `/auth/logout` | Accounts and sign-in |
| `GET` | `/auth/me` | The signed-in account |
| `POST` | `/review/upload` | Upload a CSV or Excel statement and review it |
| `POST` | `/review` | Review records sent as JSON |
| `GET` | `/reviews`, `/reviews/{id}` | Review history and a saved review |
| `GET` | `/reviews/{id}/report.pdf` | The PDF report |
| `GET`, `POST` | `/reviews/{id}/actions` | Reviewer decisions on findings |
| `GET` | `/monitoring/summary`, `/monitoring/recent`, `/monitoring/stages`, `/monitoring/coverage` | Run statistics |
| `GET` | `/health` | Liveness check and which components are installed |

Full request and response schemas are at `/docs` while the service runs.
