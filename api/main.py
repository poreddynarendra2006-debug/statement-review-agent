"""The review engine as an HTTP service.

Two endpoints:

    GET  /health   liveness, and which components are wired in
    POST /review   submit statements, get findings back

Interactive documentation is generated at /docs.

The endpoint is deliberately usable before every component exists. A component
that is missing is reported as skipped in the coverage block rather than
causing a failure, so this can be integrated against from the first day.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import (
    VERSION,
    component_status,
    get_orchestrator,
    to_records,
)
from .models import (
    ErrorResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("api")

app = FastAPI(
    title="FinSight AI - Financial Statement Review",
    description=(
        "Reviews financial statements and returns findings with the evidence "
        "behind them.\n\n"
        "Every figure is computed deterministically. The language model writes "
        "the narrative and never performs arithmetic, so each number in a "
        "response traces back to a calculation."
    ),
    version=VERSION,
    docs_url="/docs",
)

# The workspace runs on a different port, so it is a cross-origin caller.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback to a caller.

    The detail belongs in the logs; the caller gets something actionable.
    """
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="The review could not be completed.",
            detail="An unexpected error occurred. Check the service logs.",
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Liveness check, polled by the container orchestrator.

    Reports 'ok' whenever the service can accept a request. Components that
    are not wired in do not make the service unhealthy - they make individual
    checks skip, which is a different thing.
    """
    return HealthResponse(
        status="ok",
        version=VERSION,
        review_engine="ready" if get_orchestrator() else "unavailable",
        components=component_status(),
    )


@app.post(
    "/review",
    response_model=ReviewResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["review"],
)
def review(request: ReviewRequest) -> Dict[str, Any]:
    """Review a set of financial records.

    Send one entry per company-year. Figures that are absent cause the checks
    depending on them to be skipped, with the reason reported in `coverage` -
    a record with no balance sheet detail is reviewed for everything else
    rather than rejected.

    Any `document_texts` are screened for instruction-like content before
    reaching the model. Anything found is neutralised and reported in
    `security_flags`.
    """
    records = to_records(request.records)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one record is required.",
        )

    logger.info("reviewing %d record(s)", len(records))
    result = get_orchestrator().run(records, document_texts=request.document_texts)
    logger.info("reviewed in %.2fs, risk %s", result.elapsed_seconds, result.risk_score)

    return result.to_dict()


@app.get("/", include_in_schema=False)
def root() -> Dict[str, Any]:
    """Point a browser at something useful rather than a 404."""
    return {
        "service": "FinSight AI - Financial Statement Review",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
    }
