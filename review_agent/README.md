# `review_agent/`

**Owner:** Evidence & Review

Writes the review narrative from the evidence packet. It never calculates a figure: every number comes from the agents, and `agents/groundedness.py` checks each one afterwards.

| File | What it does |
|:--|:--|
| `agent.py` | `generate_review(packet)` returns the summary and how it was written |
| `prompt.py` | Prepares a bounded, screened context from the evidence packet |
| `models.py` | The review output structure |

No hosted AI service is called. The pipeline reaches it through `agents/review_agent.py`.
