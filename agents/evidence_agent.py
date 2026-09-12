"""Where the Evidence agent plugs in.

The Evidence role's package returns one EvidencePacket holding its findings
grouped by source. The rest of this system reads a single flat list of finding
dictionaries, so this turns one into the other and nothing more.

Kept here rather than in their package so their code stays theirs: they can
rename or restructure, and only this file changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: The packet's groups, in the order a reviewer should meet them. Recurring is
#: listed too, so it flows through the moment their package starts collecting
#: it; until then the group is simply absent.
GROUPS = (
    "validation_findings",
    "trend_findings",
    "anomaly_findings",
    "recurring_findings",
)


def compile_all_findings(result: Any) -> List[Dict[str, Any]]:
    """Every finding the Evidence agent assembled, as plain dictionaries."""
    from evidence_agent import compile_all_findings as build_packet

    packet = build_packet(result)

    # Already a list of findings: nothing to unpack.
    if isinstance(packet, list):
        return [_as_dict(item) for item in packet]

    findings: List[Dict[str, Any]] = []
    for group in GROUPS:
        for item in getattr(packet, group, None) or []:
            findings.append(_as_dict(item))
    return findings


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(item))
