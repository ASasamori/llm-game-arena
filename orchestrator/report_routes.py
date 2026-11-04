"""Browser and terminal presentation of authenticated, recorded run evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from orchestrator.exports import ExportNotFound, export_database

from orchestrator.reports import (
    ReportNotFound,
    ReportRequestError,
    build_run_report,
    render_text_report,
)

_TEMPLATE = Path(__file__).with_name("static") / "run_report.html"
_PRIVATE_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def report_links(request: Request, run_id: str) -> dict[str, str]:
    """URLs carry identifiers only; credentials stay in Authorization headers."""
    data_url = str(request.url_for("get_run_report", run_id=run_id))
    return {
        "report_url": str(request.url_for("run_report_viewer", run_id=run_id)),
        "text_report_url": data_url + "?format=text",
        "report_data_url": data_url,
        "export_url": str(request.url_for("export_run", run_id=run_id)),
    }


def _validate_uuid(value: str, field: str) -> None:
    try:
        UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be a UUID") from exc


def create_report_router(
    authenticate: Callable[..., str],
    enforce_rate_limit: Callable[[str], Any],
    session_factory: Callable[[], Session],
) -> APIRouter:
    """Inject gateway dependencies without adding a circular module import."""
    router = APIRouter()

    @router.get(
        "/v1/runs/{run_id}/export",
        name="export_run",
        tags=["runs"],
        summary="Download all recorded evidence for one run as a ZIP",
    )
    def export_run(run_id: str, api_key: str = Depends(authenticate)) -> FileResponse:
        enforce_rate_limit(api_key)
        _validate_uuid(run_id, "run_id")
        directory = Path(tempfile.mkdtemp(prefix="arena-download-"))
        try:
            with session_factory() as session:
                path = directory / "run.zip"
                export_database(session, path, run_id=run_id)
        except Exception as exc:
            shutil.rmtree(directory, ignore_errors=True)
            if isinstance(exc, ExportNotFound):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"arena-{run_id}.zip",
            headers=_PRIVATE_HEADERS,
            background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
        )

    @router.get(
        "/reports",
        response_class=HTMLResponse,
        include_in_schema=False,
        name="reports_home",
    )
    @router.get(
        "/runs/{run_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
        name="run_report_viewer",
    )
    def run_report_viewer(run_id: str | None = None) -> HTMLResponse:
        # This is a data-free shell. All run information needs an authenticated
        # fetch; opening a browser URL never puts an API key in its history.
        if run_id is not None:
            _validate_uuid(run_id, "run_id")
        nonce = secrets.token_urlsafe(24)
        html = _TEMPLATE.read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
        return HTMLResponse(
            html,
            headers={
                **_PRIVATE_HEADERS,
                "Content-Security-Policy": (
                    f"default-src 'none'; script-src 'nonce-{nonce}'; "
                    "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                    "img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                    "frame-ancestors 'none'"
                ),
            },
        )

    @router.get(
        "/v1/runs/{run_id}/report",
        name="get_run_report",
        tags=["runs"],
        summary="Inspect recorded decisions in JSON or terminal text",
        description=(
            "Paginated episode and transition evidence. Summaries and responses "
            "are recorded model output; unavailable reasoning is not inferred. "
            "Use format=text for a control-character-safe ASCII terminal report."
        ),
        responses={
            200: {
                "content": {
                    "application/json": {"schema": {"type": "object"}},
                    "text/plain": {"schema": {"type": "string"}},
                }
            },
            400: {"description": "Invalid identifier or pagination"},
            401: {"description": "API key required"},
            404: {"description": "Run or selected episode not found"},
        },
    )
    def get_run_report(
        run_id: str,
        request: Request,
        format: Literal["json", "text"] = "json",
        episode_id: str | None = None,
        episode_limit: int = 20,
        episode_offset: int = 0,
        step_limit: int = 20,
        step_offset: int = 0,
        width: int = 100,
        include_prompt: bool = False,
        api_key: str = Depends(authenticate),
    ) -> Response:
        enforce_rate_limit(api_key)
        _validate_uuid(run_id, "run_id")
        if episode_id is not None:
            _validate_uuid(episode_id, "episode_id")
        if not 60 <= width <= 160:
            raise HTTPException(status_code=400, detail="width must be 60..160")
        session = session_factory()
        try:
            try:
                report = build_run_report(
                    session,
                    run_id,
                    episode_id=episode_id,
                    episode_limit=episode_limit,
                    episode_offset=episode_offset,
                    step_limit=step_limit,
                    step_offset=step_offset,
                )
            except ReportNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ReportRequestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            report["links"] = report_links(request, run_id)
            if format == "text":
                return PlainTextResponse(
                    render_text_report(
                        report, width=width, include_prompt=include_prompt
                    ),
                    headers={
                        **_PRIVATE_HEADERS,
                        "Content-Disposition": f'inline; filename="run-{run_id}.txt"',
                    },
                )
            return JSONResponse(report, headers=_PRIVATE_HEADERS)
        finally:
            session.close()

    return router
