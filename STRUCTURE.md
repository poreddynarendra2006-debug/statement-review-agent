# Where your files go

Team 33 · Financial Statement Review Agent

Every role has its own folder for code and its own folder inside `tests/` for tests. Each folder already exists and has a `README.md` listing the exact files that go in it. **Upload only into your own folders.**

---

## Your folders

| Role | Code goes in | Tests go in |
|:--|:--|:--|
| Data Ingestion | `extraction/` | `tests/ingestion_tests/` |
| Validation | `validation_agent/` | `tests/validation_tests/` |
| Trend | `analysis/` and `config/` | `tests/trend/` |
| Anomaly | `finsight/` | `tests/anomaly/` |
| Evidence & Review | `agents/evidence_agent.py`, `agents/review_agent.py` | `tests/evidence_review/` |
| Risk & Reporting | `risk_reporting/`, `database/`, `reports/` | `tests/risk_tests/` |
| Frontend | `ui/` | - |
| Orchestration & API | `agents/` (its own files), `api/`, deployment files | files directly in `tests/` |

Folder names follow the code, not the role, because the app imports these exact names. Renaming a folder stops the app finding your component.

---

## Never replace these shared files

They belong to Orchestration & API, and everyone's code or tests depend on them:

- `tests/conftest.py` and every file directly inside `tests/` - **never upload tests straight into `tests/`**
- `agents/__init__.py`, `agents/orchestrator.py`, `agents/planner.py`, `agents/guardrails.py`, `agents/timing.py`
- everything in `api/`
- `analysis/anomaly_detection.py` (connects the Anomaly agent to the app)
- `Dockerfile`, `entrypoint.sh`, `requirements.txt`, `pyproject.toml`

On GitHub, uploading a file with the same name into the same folder **replaces** the old file - it does not merge them. That is how a shared file gets lost.

---

## Uploading on the GitHub website

1. Open the repository on github.com
2. Click into **your** folder, for example `tests/validation_tests/`
3. Click **Add file → Upload files**
4. Drag in the **files inside** your folder - not the folder itself. Dragging a folder named `finsight` into `finsight/` creates `finsight/finsight/`, and the app will not find it.
5. Write a short message saying what it is, then click **Commit changes**

---

## Never upload

- datasets (`.csv`, `.xlsx`) - the shared ones are already in `data/`
- output folders, charts, `findings*.json`, console logs
- trained model files (`.joblib`, `.bin`, `.safetensors`, checkpoint folders) - GitHub rejects files over 100 MB; share a link instead
- `__pycache__/`, `.pytest_cache/`, `.venv/`, `.env`, any `.db` file
- your own `README.md`, `requirements.txt`, `pytest.ini` or `.gitignore`

---

## Rules

1. **Do not rename or delete a field** in a shared object (`FinancialRecord`, `ValidationResult`, `DeviationRecord`, `AnomalyFinding`, `AnalysisResult`) without telling the group. Everyone else's code breaks when you do.
2. **Adding a field is fine** - give it a default value so nothing downstream breaks.
3. **Upload under your own name.** The commit history shows who built what.
4. **Say what you uploaded** in the group chat, so it can be checked in the full app.

---

## What happens after you upload

1. Your files are run inside the full app with everyone else's
2. Anything broken goes back to whoever owns it, with the exact fix
3. Once everything runs cleanly, the app is packaged and deployed
