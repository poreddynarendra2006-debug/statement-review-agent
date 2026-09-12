"""Where the Review agent plugs in.

The Review role's package writes its narrative from an evidence packet, so this
builds the packet, asks for the review, and hands back the summary text with the
mode it was written in - which is what the interface shows the reviewer.
"""

from __future__ import annotations

from typing import Any, Tuple

# Module level on purpose: see agents/evidence_agent.py.
import review_agent as reviewer
from evidence_agent import compile_all_findings as build_packet

#: Their mode names, in this system's terms. Anything else is treated as
#: heuristic, which is the honest answer for text assembled from templates.
MODES = {
    "llm": "model",
    "model": "model",
    "deterministic_fallback": "heuristic",
    "heuristic": "heuristic",
}


def write_review(result: Any) -> Tuple[str, str]:
    """The review narrative, and whether a model or templates wrote it."""
    # write_review first, so a later rename on their side needs no change here.
    generate = getattr(reviewer, "write_review", None) or reviewer.generate_review
    review = generate(build_packet(result))

    if isinstance(review, str):
        return review, "heuristic"

    summary = getattr(review, "summary", "") or ""
    mode = str(getattr(review, "generation_mode", "") or "")
    return summary, MODES.get(mode, "heuristic")
