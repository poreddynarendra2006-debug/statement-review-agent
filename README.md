# FinSight AI

**An evidence-driven financial statement review agent.**

Upload a company's financial statements. The system checks the arithmetic, compares this year against last, finds figures that contradict each other or look unusual, scores the overall risk, and writes a review report explaining what it found — with every number traceable back to a calculation.

Built for the **Cognizant NPN AI & Analytics 2027** campus hackathon, use case #2, by Team 33.

---

## The problem

Before financial statements can be trusted, someone has to review them by hand: add up the totals, compare against the prior year, reconcile figures that appear on more than one statement, explain what moved, and write up their observations. It takes hours per company, it is repetitive, and tired people miss things.

This tool does the checking in seconds, so the reviewer spends their time on judgement instead of arithmetic.

---

## The design principle

> **The language model never performs arithmetic.**

Every figure is computed in ordinary Python, which returns the same answer every time. Only once a number has been calculated and checked is it handed to the AI, whose job is to explain the findings in clear English.

This is deliberate. Language models are known to produce numbers that read convincingly and are wrong. In a review context that is worse than having no tool at all. Keeping the model away from the arithmetic means every figure in the report can be traced to a calculation we can show.

We also **measure** it: after the AI writes its narrative, every number in the text is checked against the computed evidence. Anything that does not match is a hallucination, and it is reported as a metric rather than hoped away.

---

## How it works

```
   Reviewer uploads a financial statement
                  |
             Orchestrator
                  |
    +-------------+-------------+
    |             |             |
 Validation     Trend       Anomaly
   Agent        Agent        Agent
    |             |             |
    +-------------+-------------+
                  |
        Evidence Agent  -- verified figures only
                  |
         Review Agent   -- the AI writes here, and only here
                  |
          Risk Assessment
                  |
            Final Report
```

---

## Repository layout

| Folder | Contents |
|:--|:--|
| `extraction/` | Reading CSV, Excel and PDF statements into one common format |
| `analysis/` | Validation rules, year-on-year trends, ratios, anomaly detection, risk scoring |
| `agents/` | Pipeline orchestration and the AI review layer |
| `api/` | HTTP interface — `POST /review`, `GET /health` |
| `ui/` | The reviewer's screens |
| `reports/` | PDF report generation |
| `database/` | Storage for reviews, findings and reviewer actions |
| `utils/` | Logging, configuration, formatting |
| `data/` | Sample datasets |
| `tests/` | Automated tests |
| `docs/` | Design and deployment documentation |

Each folder has a `README.md` naming its owner and the files expected in it. See **[STRUCTURE.md](STRUCTURE.md)** for where to put your files and how to upload them.

---

## Documentation

| Document | What it covers |
|:--|:--|
| **[docs/HLD.md](docs/HLD.md)** | Architecture, data contracts, design decisions and the alternatives rejected |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | AWS setup, deployment, rollback, demo-day runbook |
| **[STRUCTURE.md](STRUCTURE.md)** | Where each file belongs, and how to upload without using git |

---

## Running it

Once the components are in place:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest -v
uvicorn api.main:app --reload     # then open http://localhost:8000
```

**No AI API key is required.** Without one the application falls back to an offline reviewer and remains fully functional — which also means no data leaves the machine.

---

## Built with

Python · pandas · numpy · scikit-learn · FastAPI · HTML/CSS/JavaScript · ReportLab · SQLite · Docker · AWS

---

## Status

Under active development during the build window, **8–14 September 2026**. Components are being contributed folder by folder; the project runs end to end once every folder has its files.

---

## A note on scope

This tool reports possible inconsistencies and recommends that a qualified person review them. It does not allege fraud, and it does not give investment advice. The final judgement always rests with the human reviewer.
