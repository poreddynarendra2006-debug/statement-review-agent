"""Request and response schemas for the review API.

These are deliberately separate from the internal dataclasses. The internal
objects change as components evolve; this is the contract we publish to
anything outside the application, and it should change only when we mean it to.

Everything is typed rather than a bare dict, so FastAPI generates accurate
documentation at /docs and rejects malformed requests before they reach the
orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness. The container orchestrator polls this."""

    status: str = Field("ok", description="'ok' when the service can serve requests.")
    version: str = Field(..., description="Application version.")
    review_engine: str = Field(
        ..., description="'ready' when the orchestrator is constructed."
    )
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which analysis components are wired in. A component that is "
                    "absent is skipped, not an error.",
    )


class RecordIn(BaseModel):
    """One company-year submitted for review.

    Only company and year are required. Every figure is optional because the
    two data sources carry different fields, and a missing figure causes the
    affected checks to be skipped rather than to fail.
    """

    model_config = {"extra": "allow"}       # unknown columns are preserved

    company: str
    year: int
    currency: Optional[str] = "$"

    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    pre_tax_income: Optional[float] = None
    taxes: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None

    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    shareholder_equity: Optional[float] = None
    cash: Optional[float] = None
    debt: Optional[float] = None

    beginning_cash: Optional[float] = None
    ending_cash: Optional[float] = None
    cash_flow_operating: Optional[float] = None
    cash_flow_investing: Optional[float] = None
    cash_flow_financing: Optional[float] = None


class ReviewRequest(BaseModel):
    """A review submission."""

    records: List[RecordIn] = Field(
        ..., min_length=1,
        description="One entry per company-year. At least one required.",
    )
    document_texts: List[str] = Field(
        default_factory=list,
        description="Free text extracted from uploaded documents. Screened for "
                    "instruction-like content before anything reaches the model.",
    )
    materiality: Optional[float] = Field(
        None, ge=0, le=1,
        description="Proportion of the benchmark figure treated as material. "
                    "Defaults to the engine's own threshold.",
    )


class CoverageResponse(BaseModel):
    """Which agents ran, which did not, and why.

    The interface renders this directly. Without it, an empty findings list is
    ambiguous - the reviewer cannot tell whether nothing was found or whether
    the check never ran.
    """

    selected: List[str] = Field(default_factory=list)
    skipped: Dict[str, str] = Field(
        default_factory=dict,
        description="Agent name to the reason it did not run.",
    )
    facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="What the planner observed: record count, companies, periods.",
    )


class SecurityFlag(BaseModel):
    """Instruction-like content found in an uploaded document and neutralised."""

    is_suspicious: bool
    categories: List[str] = Field(default_factory=list)
    detections: List[Dict[str, Any]] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """Everything one review produced.

    Field names match AnalysisResult.to_dict() so the interface and the API
    return the same shape and the front end needs only one mapping.
    """

    company: str
    period: str
    currency: str
    record_count: int

    review_id: Optional[int] = Field(
        None,
        description="Id of the saved review, for the PDF and reviewer actions. "
                    "None when review history isn't installed or saving failed.",
    )
    materiality: Optional[float] = Field(
        None, description="Materiality the review was graded at. None means the agents' defaults.",
    )

    risk_score: Optional[int] = None
    risk_result: Optional[Dict[str, Any]] = None

    validation_results: List[Dict[str, Any]] = Field(default_factory=list)
    failed_validations: List[Dict[str, Any]] = Field(default_factory=list)
    yoy_results: List[Dict[str, Any]] = Field(default_factory=list)
    ratio_results: List[Dict[str, Any]] = Field(default_factory=list)
    forecasts: List[Dict[str, Any]] = Field(default_factory=list)
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    deviations: List[Dict[str, Any]] = Field(default_factory=list)
    material_deviations: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    recurring_issues: List[Dict[str, Any]] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)

    ai_summary: str = ""
    review_mode: str = Field(
        "none",
        description="'model' when the trained model wrote the narrative, "
                    "'heuristic' for the offline fallback, 'none' if neither ran.",
    )

    coverage: CoverageResponse = Field(default_factory=CoverageResponse)
    timings: Dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    security_flags: List[SecurityFlag] = Field(default_factory=list)
    screened_documents: List[str] = Field(
        default_factory=list,
        description="Uploaded document text after screening, framed as untrusted content. "
                    "Only this version is ever given to a model.",
    )


class ErrorResponse(BaseModel):
    """A failed request. Says what went wrong and what to do about it."""

    error: str = Field(..., description="What went wrong.")
    detail: Optional[str] = Field(None, description="How to fix it.")


class ActionRequest(BaseModel):
    """A reviewer's decision on one finding."""

    finding_ref: str = Field(..., min_length=1,
                             description="Identifies the finding, e.g. its rule_id and year.")
    status: Literal["VERIFIED", "NEEDS_INVESTIGATION", "DISMISSED"]
    note: str = Field("", max_length=2000)
    reviewer: str = Field("", max_length=120)


class ActionCreated(BaseModel):
    """The reviewer action that was just recorded."""

    action_id: int
