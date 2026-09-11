"""The review engine as an HTTP service.

Accounts
    POST /auth/register                 create a reviewer account
    POST /auth/login                    sign in; returns a token
    GET  /auth/me                       the signed-in account
    POST /auth/logout                   end the session

Review, history and monitoring need "Authorization: Bearer <token>" unless
AUTH_REQUIRED=false. The service endpoints and the screens stay public.

Review
    POST /review                        submit records as JSON
    POST /review/upload                 upload a CSV or Excel statement file

History and reports (need Risk & Reporting)
    GET  /reviews                       recent saved reviews
    GET  /reviews/{id}                  one saved review
    GET  /reviews/{id}/report.pdf       the PDF audit report
    GET  /reviews/{id}/actions          reviewer actions on a review
    POST /reviews/{id}/actions          mark a finding verified / to investigate / dismissed

Monitoring (needs Risk & Reporting)
    GET  /monitoring/summary | /monitoring/stages | /monitoring/coverage | /monitoring/recent

Service
    GET  /health
    GET  /docs                          interactive documentation

When ui/ holds the front end, its pages are served from / on the same port.

Every endpoint works before every component exists. A missing analysis agent
shows up as skipped in the review's coverage; a missing Ingestion or Risk &
Reporting component makes only the endpoints that need it answer 503.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from agents.orchestrator import ReviewOrchestrator

from .auth import prepare_accounts, require_user
from .auth import router as auth_router
from .dependencies import (
    ALLOWED_UPLOAD_SUFFIXES,
    MAX_UPLOAD_BYTES,
    UI_DIR,
    VERSION,
    Reporting,
    component_status,
    full_result,
    get_ingestion,
    get_orchestrator,
    get_reporting,
    ingestion_warnings,
    prepare_storage,
    records_from_ingestion,
    to_records,
)
from .models import (
    ActionCreated,
    ActionRequest,
    ErrorResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
    UserResponse,
)
from .spreadsheets import NoTableFound, excel_to_csv

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    prepare_storage()
    prepare_accounts()
    yield


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
    lifespan=lifespan,
)

# Same-origin in production, where the screens are served from this app. CORS
# only matters when the front end runs from a separate dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)

#: On every endpoint that reads or writes review data.
SIGNED_IN = [Depends(require_user)]

UNAVAILABLE = {503: {"model": ErrorResponse, "description": "The component this needs is not installed yet."}}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback to a caller. The detail belongs in the logs."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="The request could not be completed.",
            detail="An unexpected error occurred. Check the service logs.",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_review(orchestrator: ReviewOrchestrator, reporting: Optional[Reporting],
                records: List[Any], document_texts: Optional[List[str]] = None,
                materiality: Optional[float] = None,
                extra_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run a review, save it when storage is available, and return the payload."""
    logger.info("reviewing %d record(s)", len(records))
    result = orchestrator.run(records, document_texts=document_texts, materiality=materiality)
    payload = result.to_dict()
    if extra_warnings:
        payload["warnings"] = [*extra_warnings, *payload["warnings"]]

    payload["review_id"] = None
    if reporting is not None:
        try:
            payload["review_id"] = reporting.database.save_review(payload)
        except Exception:  # noqa: BLE001 - a saved copy is a convenience, the review is the point
            logger.exception("review completed but could not be saved")
            payload["warnings"] = [*payload["warnings"],
                                   "The review completed but could not be saved to history."]

    logger.info("reviewed in %.2fs, risk %s, saved as %s",
                result.elapsed_seconds, result.risk_score, payload["review_id"])
    return payload


def require_reporting(reporting: Optional[Reporting] = Depends(get_reporting)) -> Reporting:
    if reporting is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review history, reports and monitoring are not available yet: "
                   "the Risk & Reporting component is not installed.",
        )
    return reporting


def require_monitoring(reporting: Reporting = Depends(require_reporting)) -> Any:
    if reporting.monitoring is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring is not available yet: the monitoring module is not installed.",
        )
    return reporting.monitoring


def _saved_review(reporting: Reporting, review_id: int) -> Dict[str, Any]:
    review = reporting.database.get_review(review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No saved review with id {review_id}.")
    return review


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Liveness check, polled by the container host.

    'ok' whenever the service can accept requests. A component that is not
    installed doesn't make the service unhealthy; it makes individual checks
    skip or individual endpoints answer 503.
    """
    return HealthResponse(status="ok", version=VERSION, review_engine="ready",
                          components=component_status())


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


@app.post("/review", response_model=ReviewResponse, dependencies=SIGNED_IN,
          responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          tags=["review"])
def review(request: ReviewRequest,
           orchestrator: ReviewOrchestrator = Depends(get_orchestrator),
           reporting: Optional[Reporting] = Depends(get_reporting)) -> Dict[str, Any]:
    """Review records sent as JSON.

    One entry per company-year. Figures that are absent cause the checks
    depending on them to be skipped, with the reason in `coverage`.

    `materiality` re-grades findings. `document_texts` are screened for
    instruction-like content before reaching the model; anything found is
    neutralised and reported in `security_flags`.
    """
    return _run_review(orchestrator, reporting, to_records(request.records),
                       document_texts=request.document_texts,
                       materiality=request.materiality)


def _as_uploaded(message: str, temp_path: str, name: str) -> str:
    """Ingestion names the temporary copy; the user knows the file by its own name."""
    return message.replace(temp_path, name).replace(Path(temp_path).name, name)


@app.post("/review/upload", response_model=ReviewResponse, dependencies=SIGNED_IN,
          responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
                     422: {"model": ErrorResponse}, **UNAVAILABLE},
          tags=["review"])
async def review_upload(
    file: UploadFile = File(..., description="Financial statements as CSV or Excel."),
    materiality: Optional[float] = Form(None, description="Between 0 and 1. Omit for the default."),
    orchestrator: ReviewOrchestrator = Depends(get_orchestrator),
    ingest: Optional[Callable[..., Any]] = Depends(get_ingestion),
    reporting: Optional[Reporting] = Depends(get_reporting),
) -> Dict[str, Any]:
    """Upload a statement file and review it.

    The file goes through Data Ingestion, which maps its columns onto the
    standard fields, then through the same review as POST /review.
    """
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix == ".xls":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"'{name}' is an older .xls workbook, which can't be read. "
                   "Save it as .xlsx or CSV and upload it again.",
        )
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"'{name}' is not a supported file type. "
                   f"Upload a CSV or Excel file ({', '.join(ALLOWED_UPLOAD_SUFFIXES)}).",
        )
    if materiality is not None and not 0 <= materiality <= 1:
        raise HTTPException(status_code=422,
                            detail="Materiality must be between 0 and 1, for example 0.05 for 5%.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{name}' is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    if not content.strip():
        raise HTTPException(status_code=422,
                            detail=f"'{name}' is empty.")
    if ingest is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File upload is not available yet: the Data Ingestion component is "
                   "not installed. Send records as JSON to POST /review instead.",
        )

    # Ingestion reads CSV only, so a workbook is handed over as the table it holds.
    if suffix == ".xlsx":
        try:
            content = await run_in_threadpool(excel_to_csv, content)
        except NoTableFound as exc:
            raise HTTPException(status_code=422,
                                detail=f"'{name}' has no sheet with a table of column headings "
                                       "and rows under them.") from exc
        except Exception as exc:  # noqa: BLE001 - a corrupt or renamed file is the uploader's to fix
            logger.warning("could not open %s as a workbook: %s", name, exc)
            raise HTTPException(status_code=422,
                                detail=f"'{name}' could not be opened as an Excel workbook.") from exc
        suffix = ".csv"

    # Ingestion reads from a path, so the upload is written to a temporary
    # file that is always removed.
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        try:
            ingested = await run_in_threadpool(ingest, path)
        except Exception as exc:  # noqa: BLE001 - report as a bad file, not a server fault
            logger.exception("ingestion failed for %s", name)
            raise HTTPException(
                status_code=422,
                detail=f"'{name}' could not be read as financial statements: "
                       f"{_as_uploaded(str(exc), path, name)}",
            ) from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if getattr(ingested, "status", "success") == "error":
        message = _as_uploaded(getattr(ingested, "message", "") or "", path, name)
        raise HTTPException(
            status_code=422,
            detail=message or f"'{name}' does not look like a financial statement file.",
        )
    records = records_from_ingestion(ingested)
    if not records:
        raise HTTPException(status_code=422,
                            detail=f"No company-year records were found in '{name}'.")

    warnings = [f"Ingestion: {w}" for w in ingestion_warnings(ingested)]
    return await run_in_threadpool(_run_review, orchestrator, reporting, records,
                                   None, materiality, warnings)


# ---------------------------------------------------------------------------
# History, reports, reviewer actions
# ---------------------------------------------------------------------------


@app.get("/reviews", dependencies=SIGNED_IN, responses=UNAVAILABLE, tags=["history"])
def list_reviews(limit: int = Query(20, ge=1, le=200),
                 reporting: Reporting = Depends(require_reporting)) -> List[Dict[str, Any]]:
    """Recent saved reviews, newest first."""
    return reporting.database.list_reviews(limit=limit)


@app.get("/reviews/{review_id}", dependencies=SIGNED_IN, responses={404: {"model": ErrorResponse}, **UNAVAILABLE},
         tags=["history"])
def get_review(review_id: int, reporting: Reporting = Depends(require_reporting)) -> Dict[str, Any]:
    """One saved review, including the full result."""
    return _saved_review(reporting, review_id)


@app.get("/reviews/{review_id}/report.pdf", response_class=Response, dependencies=SIGNED_IN,
         responses={200: {"content": {"application/pdf": {}}},
                    404: {"model": ErrorResponse}, **UNAVAILABLE},
         tags=["reports"])
def review_report(review_id: int,
                  reviewer_name: str = Query("", max_length=120,
                                             description="Printed on the sign-off block."),
                  reporting: Reporting = Depends(require_reporting),
                  account: Optional[UserResponse] = Depends(require_user)) -> Response:
    """Download the PDF audit report for a saved review."""
    reviewer_name = reviewer_name or (account.name if account else "")
    if reporting.report is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="PDF reports are not available yet: the report generator is not installed.")
    result = full_result(_saved_review(reporting, review_id))
    pdf = reporting.report.generate_report(result, reviewer_name=reviewer_name)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="finsight-review-{review_id}.pdf"'},
    )


@app.get("/reviews/{review_id}/actions", dependencies=SIGNED_IN, responses={404: {"model": ErrorResponse}, **UNAVAILABLE},
         tags=["history"])
def list_actions(review_id: int, reporting: Reporting = Depends(require_reporting)) -> List[Dict[str, Any]]:
    """Every reviewer action recorded against a review."""
    _saved_review(reporting, review_id)
    return reporting.database.get_actions(review_id)


@app.post("/reviews/{review_id}/actions", response_model=ActionCreated, dependencies=SIGNED_IN,
          status_code=status.HTTP_201_CREATED,
          responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, **UNAVAILABLE},
          tags=["history"])
def add_action(review_id: int, action: ActionRequest,
               reporting: Reporting = Depends(require_reporting),
               account: Optional[UserResponse] = Depends(require_user)) -> ActionCreated:
    """Record a reviewer's decision on a finding - the human in the loop.

    When someone is signed in, the action is recorded under their account's
    name, whatever the request says.
    """
    _saved_review(reporting, review_id)
    reviewer = account.name if account else action.reviewer
    try:
        action_id = reporting.database.record_action(
            review_id, action.finding_ref, action.status, action.note, reviewer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ActionCreated(action_id=action_id)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


@app.get("/monitoring/summary", dependencies=SIGNED_IN, responses=UNAVAILABLE, tags=["monitoring"])
def monitoring_summary(limit: int = Query(50, ge=1, le=500),
                       monitoring: Any = Depends(require_monitoring)) -> Dict[str, Any]:
    """Reviews run, average and slowest duration, average risk, review modes."""
    return monitoring.run_summary(limit=limit)


@app.get("/monitoring/stages", dependencies=SIGNED_IN, responses=UNAVAILABLE, tags=["monitoring"])
def monitoring_stages(limit: int = Query(50, ge=1, le=500),
                      monitoring: Any = Depends(require_monitoring)) -> List[Dict[str, Any]]:
    """Time spent in each pipeline stage, slowest first."""
    return monitoring.stage_performance(limit=limit)


@app.get("/monitoring/coverage", dependencies=SIGNED_IN, responses=UNAVAILABLE, tags=["monitoring"])
def monitoring_coverage(limit: int = Query(50, ge=1, le=500),
                        monitoring: Any = Depends(require_monitoring)) -> Dict[str, Any]:
    """How often each agent ran or was skipped, and why."""
    return monitoring.agent_coverage(limit=limit)


@app.get("/monitoring/recent", dependencies=SIGNED_IN, responses=UNAVAILABLE, tags=["monitoring"])
def monitoring_recent(limit: int = Query(10, ge=1, le=100),
                      monitoring: Any = Depends(require_monitoring)) -> List[Dict[str, Any]]:
    """The latest reviews with their risk score and duration."""
    return monitoring.recent_reviews(limit=limit)


# ---------------------------------------------------------------------------
# The reviewer's screens
# ---------------------------------------------------------------------------


def mount_frontend(target: FastAPI, directory: str | os.PathLike[str]) -> bool:
    """Serve the front end from / when its index.html is present.

    Mounted after every API route, so /review, /reviews and the rest keep
    precedence over any file of the same name.
    """
    folder = Path(directory)
    if not (folder / "index.html").is_file():
        return False
    target.mount("/", StaticFiles(directory=str(folder), html=True), name="ui")
    logger.info("serving the front end from %s", folder)
    return True


if not mount_frontend(app, UI_DIR):

    @app.get("/", include_in_schema=False)
    def root() -> Dict[str, Any]:
        """Until the front end is added, point a browser somewhere useful."""
        return {
            "service": "FinSight AI - Financial Statement Review",
            "version": VERSION,
            "docs": "/docs",
            "health": "/health",
        }
