"""Runs the review pipeline and assembles the single result everything reads.

This module owns two things.

:class:`AnalysisResult` is the one object the interface, the API and the PDF
report all consume. Every component produces its own shaped output; this is
where those become one thing, so nothing downstream needs to know which
component produced what.

:class:`ReviewOrchestrator` sequences the run. It does not compute anything -
the agents do that. Its job is order, assembly, timing, and making sure a
component that is missing or broken degrades the review rather than ending it.

Components are injected rather than imported, so the pipeline can run before
they all exist, and so a component can be swapped without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .guardrails import prepare_document_text
from .planner import Planner, Requirement
from .timing import Timings

#: Stage names, in the order they run. Used for timings and the coverage strip.
STAGES = (
    "validation",
    "trend",
    "anomaly",
    "recurring",
    "peer",
    "evidence",
    "review",
    "risk",
)


@dataclass
class AnalysisResult:
    """Everything one review produced.

    The interface, the API and the PDF report all read this and nothing else.
    Fields default to empty rather than None so a partial review still renders:
    a missing component means an empty list, never an exception downstream.
    """

    company: str = "Unknown"
    #: Every company in the submission. `company` summarises them in one line
    #: ("60 companies"); this is the list behind it, for filters and headings.
    companies: List[str] = field(default_factory=list)
    #: The uploaded file this review came from, when it came from one.
    filename: str = ""
    period: str = ""
    currency: str = "$"
    record_count: int = 0
    #: Materiality the review was graded at. None means each agent's own default.
    materiality: Optional[float] = None

    validation_results: List[Any] = field(default_factory=list)
    yoy_results: List[Any] = field(default_factory=list)
    ratio_results: List[Any] = field(default_factory=list)
    forecasts: List[Any] = field(default_factory=list)
    evaluations: List[Any] = field(default_factory=list)
    deviations: List[Any] = field(default_factory=list)
    anomalies: List[Any] = field(default_factory=list)
    recurring_issues: List[Any] = field(default_factory=list)

    findings: List[Any] = field(default_factory=list)
    ai_summary: str = ""
    review_mode: str = "none"          # model | heuristic | none
    risk_result: Optional[Any] = None

    #: Which agents ran, which did not, and why. Rendered as the coverage strip.
    coverage: Dict[str, Any] = field(default_factory=dict)
    #: Seconds per stage, plus the total.
    timings: Dict[str, float] = field(default_factory=dict)
    #: Anything the reviewer should know about the run itself.
    warnings: List[str] = field(default_factory=list)
    #: Instruction-like text found in uploaded documents and neutralised.
    security_flags: List[Dict[str, Any]] = field(default_factory=list)
    #: Uploaded document text after screening, ready to place in a prompt.
    #: Only this version may reach a model - never the raw text.
    screened_documents: List[str] = field(default_factory=list)

    # -- derived views -----------------------------------------------------

    @property
    def failed_validations(self) -> List[Any]:
        return [v for v in self.validation_results if getattr(v, "status", "") == "FAIL"]

    @property
    def skipped_validations(self) -> List[Any]:
        return [v for v in self.validation_results if getattr(v, "status", "") == "SKIPPED"]

    @property
    def material_deviations(self) -> List[Any]:
        """Movements large enough to need explaining.

        Materiality lives on DeviationRecord rather than on the YoY result,
        so this is the field to read when asking whether a change matters.
        """
        return [d for d in self.deviations if getattr(d, "material_deviation", False)]

    @property
    def elapsed_seconds(self) -> float:
        return self.timings.get("_total", 0.0)

    @property
    def risk_score(self) -> Optional[int]:
        return getattr(self.risk_result, "score", None)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the API, the interface and storage."""

        def dump(items: Sequence[Any]) -> List[Any]:
            return [i.to_dict() if hasattr(i, "to_dict") else i for i in items]

        return {
            "company": self.company,
            "companies": list(self.companies),
            "filename": self.filename,
            "period": self.period,
            "currency": self.currency,
            "record_count": self.record_count,
            "materiality": self.materiality,
            "risk_score": self.risk_score,
            "risk_result": self.risk_result.to_dict()
            if hasattr(self.risk_result, "to_dict") else self.risk_result,
            "validation_results": dump(self.validation_results),
            "failed_validations": dump(self.failed_validations),
            "yoy_results": dump(self.yoy_results),
            "ratio_results": dump(self.ratio_results),
            "forecasts": dump(self.forecasts),
            "evaluations": dump(self.evaluations),
            "deviations": dump(self.deviations),
            "material_deviations": dump(self.material_deviations),
            "anomalies": dump(self.anomalies),
            "recurring_issues": dump(self.recurring_issues),
            "findings": dump(self.findings),
            "ai_summary": self.ai_summary,
            "review_mode": self.review_mode,
            "coverage": self.coverage,
            "timings": self.timings,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": self.warnings,
            "security_flags": self.security_flags,
            "screened_documents": self.screened_documents,
        }


class ReviewOrchestrator:
    """Sequences a review across whichever components are available.

    Every component is optional. Supplying none produces an empty but valid
    result rather than an error, which is what lets the pipeline be built and
    demonstrated before every component exists.
    """

    def __init__(
        self,
        *,
        validation: Optional[Callable[[Sequence[Any]], List[Any]]] = None,
        trend: Optional[Callable[[Sequence[Any]], Any]] = None,
        anomaly: Optional[Callable[[Sequence[Any]], List[Any]]] = None,
        recurring: Optional[Callable[[Sequence[Any]], List[Any]]] = None,
        peer: Optional[Callable[[Sequence[Any]], List[Any]]] = None,
        evidence: Optional[Callable[..., List[Any]]] = None,
        review: Optional[Callable[..., Any]] = None,
        risk: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._components = {
            "validation": validation,
            "trend": trend,
            "anomaly": anomaly,
            "recurring": recurring,
            "peer": peer,
        }
        self._evidence = evidence
        self._review = review
        self._risk = risk

        self._planner = Planner()
        requirements = {
            # Always runs: the agent itself skips each check whose inputs are
            # missing, and its ratio checks work on data with no balance sheet.
            "validation": Requirement.NONE,
            "trend": Requirement.MULTIPLE_PERIODS,
            "anomaly": Requirement.ENOUGH_RECORDS,
            "recurring": Requirement.RECURRENCE_WINDOW,
            "peer": Requirement.PEER_GROUP,
        }
        descriptions = {
            "validation": "Accounting identity and ratio checks",
            "trend": "Year-on-year movement, ratios and forecasting",
            "anomaly": "Statistical and model-based outlier detection",
            "recurring": "Issues repeating across several periods",
            "peer": "Comparison against other companies in the dataset",
        }
        for name, fn in self._components.items():
            if fn is not None:
                self._planner.register(name, fn, requires=requirements[name],
                                       description=descriptions[name])

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _identify(records: Sequence[Any]) -> tuple[str, str, str, List[str]]:
        companies = sorted(str(c) for c in {getattr(r, "company", None) for r in records} - {None})
        years = sorted({getattr(r, "year", None) for r in records} - {None})

        if not companies:
            company = "Unknown"
        elif len(companies) == 1:
            company = companies[0]
        else:
            company = f"{len(companies)} companies"

        if not years:
            period = ""
        elif len(years) == 1:
            period = f"FY{years[0]}"
        else:
            period = f"FY{years[0]} - FY{years[-1]}"

        currency = next((getattr(r, "currency", None) for r in records
                         if getattr(r, "currency", None)), "$")
        return company, period, currency, companies

    def _screen_documents(self, texts: Sequence[str],
                          result: AnalysisResult) -> List[str]:
        """Strip instruction-like text out of anything from an uploaded file.

        Document content is data, never instruction. This runs before any text
        reaches a prompt, so every route into the model is covered rather than
        only the PDF one.
        """
        safe: List[str] = []
        for text in texts:
            prepared, check = prepare_document_text(text)
            safe.append(prepared)
            if check.is_suspicious:
                result.security_flags.append(check.to_dict())
        if result.security_flags:
            result.warnings.append(
                f"{len(result.security_flags)} passage(s) of uploaded text contained "
                "instruction-like content and were neutralised before analysis."
            )
        return safe

    @staticmethod
    def _unpack_trend(output: Any, result: AnalysisResult) -> None:
        """Spread the trend agent's several outputs onto the result.

        It returns five kinds of thing, so it may hand back a dict of lists or
        a single list of YoY results. Both are accepted.
        """
        if output is None:
            return
        if isinstance(output, dict):
            result.yoy_results = list(output.get("yoy", output.get("yoy_results", [])))
            result.ratio_results = list(output.get("ratios", output.get("ratio_results", [])))
            result.forecasts = list(output.get("forecasts", []))
            result.evaluations = list(output.get("evaluations", []))
            result.deviations = list(output.get("deviations", []))
        else:
            result.yoy_results = list(output)

    # -- the pipeline ------------------------------------------------------

    def run(self, records: Sequence[Any],
            document_texts: Optional[Sequence[str]] = None,
            materiality: Optional[float] = None) -> AnalysisResult:
        """Review a set of records and return everything the run produced.

        Args:
            records: The submission, one object per company-year.
            document_texts: Free text from uploaded files, screened before use.
            materiality: Passed to every agent that accepts it (validation,
                trend), so the interface slider re-grades findings. None leaves
                each agent on its own default.
        """
        timings = Timings()
        result = AnalysisResult(materiality=materiality)

        if not records:
            result.warnings.append("No records were supplied, so nothing was reviewed.")
            result.timings = timings.to_dict()
            result.coverage = self._planner.plan([]).to_dict()
            return result

        result.record_count = len(records)
        (result.company, result.period,
         result.currency, result.companies) = self._identify(records)

        if document_texts:
            with timings.stage("guardrails"):
                result.screened_documents = self._screen_documents(document_texts, result)

        # Analysis agents, each run only where the data supports it.
        options = {"materiality": materiality} if materiality is not None else None
        outputs, plan = self._planner.run(records, timings=timings, options=options)
        result.coverage = plan.to_dict()

        result.validation_results = list(outputs.get("validation") or [])
        self._unpack_trend(outputs.get("trend"), result)
        result.anomalies = list(outputs.get("anomaly") or [])
        result.recurring_issues = list(outputs.get("recurring") or [])
        if outputs.get("peer"):
            result.anomalies.extend(outputs["peer"])

        for name, reason in plan.skipped.items():
            result.warnings.append(f"{name} did not run: {reason}")

        # Evidence, then narrative, then risk. Each guarded so a failure
        # downstream never discards the findings already computed.
        if self._evidence is not None:
            with timings.stage("evidence"):
                try:
                    result.findings = list(self._evidence(result) or [])
                except Exception as exc:  # noqa: BLE001
                    result.warnings.append(f"Evidence assembly failed: {exc}")

        if self._review is not None:
            with timings.stage("review"):
                try:
                    narrative = self._review(result)
                    if isinstance(narrative, tuple):
                        result.ai_summary, result.review_mode = narrative
                    else:
                        result.ai_summary = narrative or ""
                        result.review_mode = "model"
                except Exception as exc:  # noqa: BLE001
                    result.warnings.append(f"AI review unavailable, continuing without it: {exc}")
                    result.review_mode = "none"

        if self._risk is not None:
            with timings.stage("risk"):
                try:
                    result.risk_result = self._risk(result)
                except Exception as exc:  # noqa: BLE001
                    result.warnings.append(f"Risk scoring failed: {exc}")

        result.timings = timings.to_dict()
        return result
