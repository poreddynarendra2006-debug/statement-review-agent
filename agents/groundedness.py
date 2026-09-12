"""Checks that every number in the written review came from the evidence.

The design principle is that the model never performs arithmetic. This is the
part that proves it held, rather than assuming it did.

It matters because we measured the alternative. Fine-tuning our own reviewer
model on 12 September 2026 produced this, on a validation example:

    input  : revenue: 105150 | cost of revenue: 74670
    written: "With revenue of USD 105,150 and cost of revenue of USD 75,670..."

The prose is right, the conclusion is right, and one digit of a figure is
invented. A reviewer reading that would have no way of knowing. So every number
in the narrative is checked against the figures the engine actually computed,
and anything unsupported is reported rather than hoped away.

Two kinds of number are accepted:

- **Quoted** - the figure appears in the evidence, allowing for formatting:
  ``105,150`` for ``105150``, ``FY2022`` for ``2022``.
- **Derived** - a difference or sum of two figures in the evidence, which is how
  a review sentence legitimately says "overstated by 25,730".

Anything else is unsupported, and named.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

#: A run of digits, possibly with thousands separators and a decimal part.
TOKEN = re.compile(r"-?\d[\d,]*\.?\d*")

#: Two figures may be added or subtracted to reach a third before it counts as
#: derived. Beyond that the claim stops being checkable.
DERIVATION_TOLERANCE = 0.51

#: Parts of the result that are not evidence: the narrative being checked, the
#: warnings that quote it, and the verdict of the check itself.
NOT_EVIDENCE = frozenset({"ai_summary", "warnings", "groundedness", "review_mode"})

#: Above this many evidence figures, derivation is skipped: the pairwise search
#: is quadratic, and with tens of thousands of figures almost any number would
#: be "derivable" anyway, which would make the check meaningless as well as slow.
MAX_FIGURES_FOR_DERIVATION = 400


@dataclass
class GroundednessReport:
    """What the check found, in a form the interface and the API can show."""

    numbers_written: int = 0
    supported: int = 0
    unsupported: List[str] = field(default_factory=list)

    @property
    def share_supported(self) -> float:
        """1.0 when nothing was written, which is vacuously grounded."""
        if not self.numbers_written:
            return 1.0
        return round(self.supported / self.numbers_written, 4)

    @property
    def is_grounded(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numbers_written": self.numbers_written,
            "supported": self.supported,
            "unsupported": list(self.unsupported),
            "share_supported": self.share_supported,
            "is_grounded": self.is_grounded,
        }


def numbers_in(text: str) -> Set[str]:
    """Every number in a piece of text, normalised.

    A comma is a thousands separator only when what follows it is exactly three
    digits. Otherwise it separates a list, so ``years: 2022,2023,2024`` reads as
    three years rather than one twelve-digit number - a mistake that made an
    earlier version of this check report correct output as invented.
    """
    found: Set[str] = set()
    for token in TOKEN.findall(text or ""):
        token = token.rstrip(",.")
        parts = token.split(",")
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            found.add(_normalise(token.replace(",", "")))
        else:
            for part in parts:
                part = part.strip(".")
                if part:
                    found.add(_normalise(part))
    return {n for n in found if n}


def _normalise(value: str) -> str:
    """Trailing zeros and a trailing point are formatting, not a different number."""
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def evidence_numbers(result: Any) -> Set[str]:
    """Every figure the engine computed for this review.

    Taken from the serialised result, so it covers each agent's findings
    without this module having to know their shapes. The counts a narrative
    naturally quotes - "147 failed checks", "59 companies" - are added too,
    since those are facts about the review rather than invented figures.
    """
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    # The narrative is part of the result by the time this runs, so it must be
    # excluded: reading it back as its own evidence would pass any figure.
    # Warnings quote it, and the verdict of a previous check is not evidence.
    evidence = {k: v for k, v in payload.items() if k not in NOT_EVIDENCE}
    figures = numbers_in(json.dumps(evidence, default=str))

    for name in ("findings", "failed_validations", "validation_results", "anomalies",
                 "deviations", "material_deviations", "recurring_issues",
                 "peer_findings", "companies", "yoy_results", "ratio_results"):
        value = payload.get(name)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            figures.add(str(len(value)))
    return figures


def _derivable(value: str, figures: Iterable[float]) -> bool:
    """True when two evidence figures add or subtract to this one."""
    try:
        target = abs(float(value))
    except ValueError:
        return False
    pool = list(figures)
    for i, first in enumerate(pool):
        for second in pool[i + 1:]:
            if (abs(abs(first - second) - target) < DERIVATION_TOLERANCE
                    or abs(abs(first + second) - target) < DERIVATION_TOLERANCE):
                return True
    return False


def check(narrative: str, result: Any) -> GroundednessReport:
    """Check a written review against the figures behind it."""
    written = numbers_in(narrative)
    if not written:
        return GroundednessReport()

    figures = evidence_numbers(result)
    missing = sorted(n for n in written if n not in figures)

    if missing and len(figures) <= MAX_FIGURES_FOR_DERIVATION:
        pool = []
        for figure in figures:
            try:
                pool.append(float(figure))
            except ValueError:
                pass
        missing = [n for n in missing if not _derivable(n, pool)]

    return GroundednessReport(
        numbers_written=len(written),
        supported=len(written) - len(missing),
        unsupported=missing,
    )
