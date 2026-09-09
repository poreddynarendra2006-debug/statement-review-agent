# Where your files go

Team 33 · FinSight AI · Financial Statement Review Agent

Each folder below has its own `README.md` listing the exact files expected in it and who owns them. **Put your files in the folder for your component** — not at the top level.

---

## The folders

| Folder | Owner | Holds |
|:--|:--|:--|
| `extraction/` | Data Ingestion | Reading CSV, Excel and PDF files into one format |
| `analysis/` | Validation, Trend, Anomaly, Risk | All the calculations |
| `agents/` | Orchestration, Evidence & Review | Pipeline sequencing and the AI review layer |
| `api/` | Orchestration & API | The HTTP interface |
| `ui/` | Frontend | The screens |
| `reports/` | Risk & Reporting | PDF report generation |
| `database/` | Risk & Reporting | Storage |
| `utils/` | Shared | Logging, config, formatting |
| `data/` | Data Ingestion | Sample datasets |
| `tests/` | Everyone | Tests for your own component |
| `docs/` | Everyone | Design documents |

**Folders are named after the code, not after roles.** If they were named `orchestration/`, `frontend/` and so on, Python could not import across them and everything would have to be renamed before the project ran. Your role is written in each folder's README instead.

Three folders are shared by more than one person — `analysis/`, `agents/` and `utils/`. In those, **only touch your own files.**

---

## Uploading without using git

If you are not comfortable with git, use the GitHub website:

1. Open the repository on github.com
2. Click into the folder your file belongs in — for example `analysis/`
3. Click **Add file → Upload files**
4. Drag your file in
5. Write a short message saying what it is, then click **Commit changes**

That is the whole process. You do not need to install anything.

**To upload a whole folder at once:** drag the folder onto the upload page and GitHub keeps the structure.

---

## Rules

1. **Do not upload** `.env`, any `.db` file, or `__pycache__` folders. The `.gitignore` blocks most of these, but the website upload can bypass it — so check before you drag.
2. **Do not rename or delete a field** in a shared object (`FinancialRecord`, `ValidationResult`, `YoYResult`, `AnomalyFinding`, `AnalysisResult`) without telling the group. Everyone else's code breaks when you do.
3. **Adding a field is fine** — give it a default value so nothing downstream breaks.
4. **Upload under your own name.** The evaluation scores teamwork, and the commit history is the evidence.
5. **Say what you uploaded** in the group chat, so the person integrating knows it arrived.

---

## What happens after you upload

1. Everyone uploads their component
2. The whole pipeline is run end to end and checked
3. Anything broken goes back to whoever owns it
4. Once it runs cleanly, it is packaged and deployed to AWS

Until every folder has real files in it, the project cannot be run as a whole — so upload as soon as your part works on your own machine, rather than waiting until it is perfect.
