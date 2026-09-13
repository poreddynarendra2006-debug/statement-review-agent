# `evidence_agent/`

**Owner:** Evidence & Review

Builds the evidence packet: the verified findings from every agent, grouped by type, which the Review Agent writes from.

| File | What it does |
|:--|:--|
| `agent.py` | `compile_all_findings(result)` assembles the packet |
| `models.py` | The evidence packet and finding structures |

The pipeline reaches it through `agents/evidence_agent.py`.
