# `agents/`

**Owner:** Orchestration & API

The pipeline's control layer: which agents run, in what order, and the checks applied around them.

| File | What it does |
|:--|:--|
| `orchestrator.py` | Runs a review end to end and assembles the single `AnalysisResult` the API, screens and PDF all read |
| `planner.py` | Decides which agents have enough data to run, and records why others were skipped |
| `guardrails.py` | Screens uploaded document text for instruction-like content before any agent sees it |
| `data_quality.py` | Data-quality failures: negative values, duplicate company-years, blank required fields, impossible years |
| `groundedness.py` | Checks every number in the written review against the computed evidence |
| `years.py` | What counts as a real reporting year |
| `timing.py` | Per-stage timings for monitoring |
| `evidence_agent.py` | Connects the Evidence Agent in `evidence_agent/` to the pipeline |
| `review_agent.py` | Connects the Review Agent in `review_agent/` to the pipeline |
