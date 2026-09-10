"""Decides which analysis agents to run for a given submission.

A fixed pipeline runs everything every time, then produces empty or misleading
output for whatever the data could not support. Year-on-year analysis on a
single period is meaningless; the five accounting checks cannot run on records
that carry no balance sheet; peer comparison against two companies is noise.

So each agent is registered as a *tool* with a stated requirement, and the
planner selects the applicable ones and records why it skipped the rest. That
record matters as much as the selection: a reviewer who sees no trend findings
needs to know whether there were none, or whether the trend agent never ran.

Nothing here computes a figure. The planner chooses what runs; the agents do
the work.

Usage::

    planner = Planner()
    planner.register("validation", run_validations,
                     requires=Requirement.STATEMENT_DETAIL,
                     description="Five accounting identity checks")

    decision = planner.plan(records)
    decision.selected     # ["validation", "anomaly"]
    decision.skipped      # {"trend": "needs at least 2 periods, found 1"}
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

#: A company-year needs these populated before the accounting checks mean
#: anything. Kaggle rows carry none of them.
STATEMENT_FIELDS = (
    "cost_of_revenue",
    "operating_expenses",
    "pre_tax_income",
    "taxes",
    "total_assets",
    "total_liabilities",
    "beginning_cash",
    "ending_cash",
)

#: Below this many companies, "unusual against peers" is not a claim worth
#: making.
MIN_COMPANIES_FOR_PEER_ANALYSIS = 5

#: Two periods to compare one movement; three before a trend is a trend.
MIN_PERIODS_FOR_TREND = 2
MIN_PERIODS_FOR_RECURRENCE = 3


class Requirement(Enum):
    """What an agent needs before it is worth running."""

    NONE = "none"
    STATEMENT_DETAIL = "statement_detail"
    MULTIPLE_PERIODS = "multiple_periods"
    RECURRENCE_WINDOW = "recurrence_window"
    PEER_GROUP = "peer_group"


@dataclass
class Tool:
    """One registered agent."""

    name: str
    run: Callable[..., Any]
    requires: Requirement = Requirement.NONE
    description: str = ""

    def schema(self) -> Dict[str, str]:
        """Tool definition, in the shape a model would be shown."""
        return {
            "name": self.name,
            "description": self.description or f"Run the {self.name} agent.",
            "requires": self.requires.value,
        }


@dataclass
class Plan:
    """What the planner decided, and why."""

    selected: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"selected": self.selected, "skipped": self.skipped, "facts": self.facts}

    def summary(self) -> str:
        ran = ", ".join(self.selected) or "nothing"
        if not self.skipped:
            return f"ran {ran}"
        return f"ran {ran}; skipped {', '.join(self.skipped)}"


def describe(records: Sequence[Any]) -> Dict[str, Any]:
    """Establish the facts the planner reasons over.

    Kept separate from the decision so the facts can be logged, shown in the
    interface, and tested without going near the agents.
    """
    companies = {getattr(r, "company", None) for r in records}
    companies.discard(None)
    years = sorted({getattr(r, "year", None) for r in records} - {None})

    def has_detail(record: Any) -> bool:
        return all(getattr(record, f, None) is not None for f in STATEMENT_FIELDS)

    with_detail = sum(1 for r in records if has_detail(r))

    return {
        "records": len(records),
        "companies": len(companies),
        "periods": len(years),
        "year_range": (years[0], years[-1]) if years else None,
        "records_with_statement_detail": with_detail,
        "has_statement_detail": with_detail > 0,
    }


class Planner:
    """Registry of analysis agents, and the decision about which to run."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # -- registration ------------------------------------------------------

    def register(self, name: str, run: Callable[..., Any],
                 requires: Requirement = Requirement.NONE,
                 description: str = "") -> None:
        """Add an agent. Registering the same name twice replaces it."""
        self._tools[name] = Tool(name=name, run=run, requires=requires,
                                 description=description)

    @property
    def tools(self) -> List[str]:
        return list(self._tools)

    def tool_schemas(self) -> List[Dict[str, str]]:
        """Every registered tool, as definitions."""
        return [t.schema() for t in self._tools.values()]

    # -- planning ----------------------------------------------------------

    def _blocker(self, requirement: Requirement, facts: Dict[str, Any]) -> Optional[str]:
        """Why a requirement is not met, or None when it is."""
        if requirement is Requirement.NONE:
            return None

        if requirement is Requirement.STATEMENT_DETAIL:
            if not facts["has_statement_detail"]:
                return ("no record carries balance sheet or P&L detail, so the "
                        "accounting identities cannot be evaluated")
            return None

        if requirement is Requirement.MULTIPLE_PERIODS:
            if facts["periods"] < MIN_PERIODS_FOR_TREND:
                return (f"needs at least {MIN_PERIODS_FOR_TREND} periods to compare, "
                        f"found {facts['periods']}")
            return None

        if requirement is Requirement.RECURRENCE_WINDOW:
            if facts["periods"] < MIN_PERIODS_FOR_RECURRENCE:
                return (f"needs at least {MIN_PERIODS_FOR_RECURRENCE} periods to call "
                        f"something recurring, found {facts['periods']}")
            return None

        if requirement is Requirement.PEER_GROUP:
            if facts["companies"] < MIN_COMPANIES_FOR_PEER_ANALYSIS:
                return (f"needs at least {MIN_COMPANIES_FOR_PEER_ANALYSIS} companies "
                        f"for peer comparison, found {facts['companies']}")
            return None

        return None

    def plan(self, records: Sequence[Any], only: Optional[Iterable[str]] = None) -> Plan:
        """Choose which agents to run against ``records``.

        Args:
            records: The submission, as FinancialRecord-like objects.
            only: Restrict consideration to these tools. Unknown names raise.
        """
        facts = describe(records)
        plan = Plan(facts=facts)

        names = list(self._tools) if only is None else list(only)
        for name in names:
            if name not in self._tools:
                raise KeyError(f"unknown tool: {name!r}. registered: {sorted(self._tools)}")

        if not records:
            plan.skipped = {n: "no records were submitted" for n in names}
            return plan

        for name in names:
            blocker = self._blocker(self._tools[name].requires, facts)
            if blocker is None:
                plan.selected.append(name)
            else:
                plan.skipped[name] = blocker

        return plan

    # -- execution ---------------------------------------------------------

    @staticmethod
    def _call(tool: Tool, records: Sequence[Any], options: Optional[Dict[str, Any]]) -> Any:
        """Run one tool, passing only the options its function accepts.

        The materiality setting matters to validation and trend but means
        nothing to anomaly detection. Filtering by signature lets one set of
        run options go to every agent without each having to accept them all.
        """
        if not options:
            return tool.run(records)
        try:
            params = inspect.signature(tool.run).parameters
        except (TypeError, ValueError):
            return tool.run(records)
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        usable = {k: v for k, v in options.items() if takes_any or k in params}
        return tool.run(records, **usable)

    def run(self, records: Sequence[Any], timings: Any = None,
            only: Optional[Iterable[str]] = None,
            options: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], Plan]:
        """Plan, then run the selected agents.

        A failing agent is recorded as skipped rather than allowed to bring
        down the whole review - a broken trend agent should not stop the
        accounting checks from reporting.

        Returns:
            The output of each agent that ran, and the plan describing what
            ran and what did not.
        """
        plan = self.plan(records, only=only)
        results: Dict[str, Any] = {}

        for name in list(plan.selected):
            tool = self._tools[name]
            try:
                if timings is not None and hasattr(timings, "stage"):
                    with timings.stage(name):
                        results[name] = self._call(tool, records, options)
                else:
                    results[name] = self._call(tool, records, options)
            except Exception as exc:  # noqa: BLE001 - deliberate isolation
                plan.selected.remove(name)
                plan.skipped[name] = f"agent raised {type(exc).__name__}: {exc}"

        return results, plan
