# `agents/`

**Owner:** Orchestration (Member 1), Evidence & Review (Member 6)

Runs the pipeline, and turns verified figures into readable review comments.

## Files that belong here

| File | What it does |
|:--|:--|
| `orchestrator.py` | Member 1 — sequences every stage, assembles AnalysisResult, records timings. |
| `evidence_agent.py` | Member 6 — collects verified figures into the evidence packet. |
| `review_agent.py` | Member 6 — calls the AI provider and writes the narrative. |
| `prompts.py` | Member 6 — all prompt templates live here, nowhere else. |

## Contract

`AnalysisResult` — the single object the UI and the API both consume.

---

*Delete this README once the folder has real files in it.*
