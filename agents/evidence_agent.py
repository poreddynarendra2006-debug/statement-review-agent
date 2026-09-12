"""Where the Evidence agent plugs in.

The Evidence role's package returns one EvidencePacket holding its findings
grouped by source. The rest of this system reads a single flat list of finding
dictionaries, so this turns one into the other and nothing more.

Kept here rather than in their package so their code stays theirs: they can
rename or restructure, and only this file changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Module level on purpose: if the Evidence role's package is not installed this
# import fails, and the API reports the component as missing instead of wiring
# a connector that raises on every review.
from evidence_agent import compile_all_findings as build_packet

#: The packet's groups, in the order a reviewer should meet them. Recurring is
#: listed too, so it flows through the moment their package starts collecting
#: it; until then the group is simply absent.
GROUPS = (
    "validation_findings",
    "trend_findings",
    "anomaly_findings",
    "recurring_issues",
)

#: Groups whose items are recurrences of an earlier finding.
RECURRING_GROUPS = frozenset({"recurring_issues", "recurring_findings"})


def compile_all_findings(result: Any) -> List[Dict[str, Any]]:
    """Every finding the Evidence agent assembled, as plain dictionaries."""
    packet = build_packet(result)

    # Already a list of findings: nothing to unpack.
    if isinstance(packet, list):
        return [_as_dict(item) for item in packet]

    findings: List[Dict[str, Any]] = []
    for group in GROUPS:
        for item in getattr(packet, group, None) or []:
            finding = _as_dict(item)
            if group in RECURRING_GROUPS:
                # A recurring issue carries the agent that first raised it, so
                # in one flat list it would read as a validation or trend
                # finding. Name it for what it is, and keep the origin.
                origin = finding.get("source")
                if origin and origin != "recurring":
                    finding["origin"] = origin
                finding["source"] = "recurring"
            findings.append(finding)
    return findings


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(item))
