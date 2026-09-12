"""Compares each company against its peers in the same submission.

The other agents judge a company against itself: its own accounting identities,
its own history, its own forecast. That misses the question an auditor asks
early on - *is this normal for a company like this one?* A 4% net margin is
unremarkable for a retailer and startling for a software firm.

So this agent puts every company-year beside the others in the same industry
and year, and reports the ones sitting far outside the pack.

Two decisions matter:

**Who counts as a peer.** Companies in the same `category` and the same year.
Comparing 2016 against 2023, or a bank against a manufacturer, would produce
findings that say more about the grouping than the company. Where an industry
has too few companies in a year, the whole year is used instead and the
finding says so, because a stated weaker comparison beats a silent one.

**How far is far.** How far outside the middle half of the group a company
sits, measured in interquartile ranges - the textbook rule for extreme values
in data that is not symmetric, which financial ratios never are. A standard
deviation would be inflated by the very companies we are looking for, and a
symmetric rule would report half the healthy companies in a skewed group.
Three interquartile ranges beyond the middle half is the floor, and a finding
must also clear a relative gap, so a tightly packed group does not generate
findings over rounding. Measured on the clean dataset, that reports about 7%
of company-years; most of the pack is never mentioned.

Nothing here is an accusation. Being unlike one's peers is a reason to look,
which is exactly what the output says.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Below this many companies, "unusual among peers" is not a claim worth
#: making: with four companies, one of them is always the extreme one.
MIN_PEERS = 5

#: How far outside the middle half of the peer group, in interquartile ranges,
#: before a company-year is reported. 3.0 is the conventional line for an
#: extreme value; measured on the clean dataset it reports 7% of company-years,
#: against 14% at 1.5 and 4% at 4.0.
MIN_DISTANCE = 3.0
HIGH_DISTANCE = 4.5
CRITICAL_DISTANCE = 6.0

#: A finding must also be this far from the median in relative terms, so a
#: group that agrees closely does not produce findings over noise.
MIN_RELATIVE_GAP = 0.20

#: Stand-in spread for a group whose members all report the same figure, as a
#: share of that figure. With no spread of its own, the only sensible question
#: is how far off the agreed value a company is; at 5%, one sitting 15% away
#: is reported.
UNANIMOUS_SPREAD = 0.05


@dataclass
class PeerFinding:
    """One company-year sitting far outside its peer group on one measure."""

    company: str
    year: int
    metric: str
    value: float
    peer_median: float
    peer_low: float               #: 25th percentile of the peer group
    peer_high: float              #: 75th percentile of the peer group
    distance: float               #: interquartile ranges beyond the middle half
    direction: str                #: above | below
    severity: str                 #: MEDIUM | HIGH | CRITICAL
    peer_group: str               #: the industry and year compared against
    peer_count: int               #: companies in that comparison
    issue: str                    #: one readable line
    evidence: str                 #: the figures behind it
    source: str = "peer"
    #: Set when the industry had too few companies and the whole year was used.
    widened: bool = False

    @property
    def finding(self) -> str:
        """Alias the risk engine reads when scoring a generic finding."""
        return self.issue

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["finding"] = self.issue
        return data


# ---------------------------------------------------------------------------
# The measures compared
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measure:
    """A ratio worth comparing across companies, and how to read it."""

    name: str
    label: str
    numerator: str
    denominator: str
    as_percent: bool = True

    def of(self, record: Any) -> Optional[float]:
        """The ratio for one record, or None when it cannot be computed."""
        top = _number(_field(record, self.numerator))
        bottom = _number(_field(record, self.denominator))
        if top is None or bottom is None or bottom == 0:
            return None
        return top / bottom


#: Ratios rather than amounts, so a large company and a small one can be
#: compared at all. Revenue of 60,000 against 600 says nothing about health.
MEASURES: Tuple[Measure, ...] = (
    Measure("net_profit_margin", "Net profit margin", "net_income", "revenue"),
    Measure("gross_margin", "Gross margin", "gross_profit", "revenue"),
    Measure("operating_margin", "Operating margin", "operating_income", "revenue"),
    Measure("return_on_assets", "Return on assets", "net_income", "total_assets"),
    Measure("return_on_equity", "Return on equity", "net_income", "shareholder_equity"),
    Measure("current_ratio", "Current ratio", "current_assets", "current_liabilities",
            as_percent=False),
    Measure("debt_to_equity", "Debt to equity", "total_liabilities", "shareholder_equity",
            as_percent=False),
)


# ---------------------------------------------------------------------------
# Reading records, whether they arrive as objects or dicts
# ---------------------------------------------------------------------------


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _text(value: Any, default: str = "") -> str:
    return str(value).strip() if value not in (None, "") else default


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _quartiles(values: Sequence[float]) -> Tuple[float, float]:
    ordered = sorted(values)
    half = len(ordered) // 2
    lower = ordered[:half]
    upper = ordered[half + 1:] if len(ordered) % 2 else ordered[half:]
    return _median(lower or ordered), _median(upper or ordered)


def _group_spread(values: Sequence[float], low: float, high: float, middle: float) -> float:
    """The yardstick a distance is measured in, for this group.

    Normally the interquartile range. Two fallbacks matter, because a peer
    group of similar companies often agrees closely:

    - More than half the group sharing a value collapses the middle half to
      nothing, so the median absolute deviation is used instead - doubled,
      since a spread of the middle half is about twice the MAD.
    - Every member reporting the same figure leaves no spread at all. Then the
      only sensible yardstick is the agreed figure itself, so a share of it is
      used and a company well away from unanimity is still reported.

    0.0 means there is genuinely nothing to compare against.
    """
    iqr = high - low
    if iqr > 0:
        return iqr
    mad = _median([abs(value - middle) for value in values])
    if mad > 0:
        return 2 * mad
    return abs(middle) * UNANIMOUS_SPREAD


def _distance_outside(value: float, low: float, high: float, spread: float) -> float:
    """How far past the middle half a value sits, in units of that spread."""
    if spread <= 0:
        return 0.0
    if value > high:
        return (value - high) / spread
    if value < low:
        return (low - value) / spread
    return 0.0


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _peer_groups(records: Sequence[Any]) -> List[Tuple[str, int, bool, List[Any]]]:
    """Comparable sets: same industry and year where possible, else same year.

    Returns (industry label, year, widened, records). `widened` is True when
    the industry was too small on its own and the year's whole set was used,
    so a finding can say what it was compared against.
    """
    by_year: Dict[int, List[Any]] = defaultdict(list)
    by_year_industry: Dict[Tuple[int, str], List[Any]] = defaultdict(list)

    for record in records:
        year = _number(_field(record, "year"))
        company = _text(_field(record, "company"))
        if year is None or not company:
            continue
        year = int(year)
        by_year[year].append(record)
        by_year_industry[(year, _text(_field(record, "category"), "All industries").upper())].append(record)

    groups: List[Tuple[str, int, bool, List[Any]]] = []
    for (year, industry), members in sorted(by_year_industry.items()):
        if _companies(members) >= MIN_PEERS:
            groups.append((industry, year, False, members))
            continue
        # Too few in this industry: compare against the year instead, and say so.
        whole_year = by_year[year]
        if _companies(whole_year) >= MIN_PEERS:
            groups.append(("All industries", year, True, whole_year))

    # A widened group is added once per small industry; keep one copy per year.
    seen: set[Tuple[str, int]] = set()
    unique: List[Tuple[str, int, bool, List[Any]]] = []
    for industry, year, widened, members in groups:
        if (industry, year) in seen:
            continue
        seen.add((industry, year))
        unique.append((industry, year, widened, members))
    return unique


def _companies(records: Iterable[Any]) -> int:
    return len({_text(_field(r, "company")) for r in records} - {""})


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _severity(distance: float) -> str:
    if distance >= CRITICAL_DISTANCE:
        return "CRITICAL"
    if distance >= HIGH_DISTANCE:
        return "HIGH"
    return "MEDIUM"


def _format(value: float, measure: Measure) -> str:
    if measure.as_percent:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def compare_with_peers(records: Sequence[Any]) -> List[PeerFinding]:
    """Report company-years sitting far outside their peer group.

    Args:
        records: The submission, one object or dict per company-year.

    Returns:
        One finding per company-year and measure that stands apart, worst
        first. An empty list when there are too few companies to compare,
        which is the honest answer rather than a weak one.
    """
    findings: List[PeerFinding] = []

    for industry, year, widened, members in _peer_groups(records):
        group_label = f"{industry} {year}"
        for measure in MEASURES:
            valued = [(record, measure.of(record)) for record in members]
            valued = [(record, value) for record, value in valued if value is not None]
            if _companies(record for record, _ in valued) < MIN_PEERS:
                continue

            values = [value for _, value in valued]
            middle = _median(values)
            low, high = _quartiles(values)
            spread = _group_spread(values, low, high, middle)
            if spread <= 0:
                continue  # the group agrees on zero; there is nothing to compare

            for record, value in valued:
                distance = _distance_outside(value, low, high, spread)
                if distance < MIN_DISTANCE:
                    continue
                # Also require a real gap, so a tightly packed group does not
                # report findings over the third decimal place.
                gap = value - middle
                scale = max(abs(middle), abs(high - low), 1e-9)
                if abs(gap) / scale < MIN_RELATIVE_GAP:
                    continue

                company = _text(_field(record, "company"))
                direction = "above" if gap > 0 else "below"
                severity = _severity(distance)
                compared = ("the whole of " + str(year)) if widened else group_label
                issue = (
                    f"{measure.label} {_format(value, measure)} is far {direction} the "
                    f"{_format(middle, measure)} typical of peers in {compared}"
                )
                evidence = (
                    f"FY{year}: {company} {measure.label.lower()} {_format(value, measure)}; "
                    f"{_companies(m for m, _ in valued)} peers median {_format(middle, measure)}, "
                    f"middle half {_format(low, measure)} to {_format(high, measure)}; "
                    f"{distance:.1f} interquartile ranges outside that"
                )
                findings.append(PeerFinding(
                    company=company,
                    year=year,
                    metric=measure.name,
                    value=round(value, 6),
                    peer_median=round(middle, 6),
                    peer_low=round(low, 6),
                    peer_high=round(high, 6),
                    distance=round(distance, 2),
                    direction=direction,
                    severity=severity,
                    peer_group=group_label,
                    peer_count=_companies(m for m, _ in valued),
                    issue=issue,
                    evidence=evidence,
                    widened=widened,
                ))

    findings.sort(key=lambda f: (-f.distance, f.company, f.year, f.metric))
    return findings
