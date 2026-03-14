from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from app.logging_config import configure_logging

configure_logging()

from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import HTMLResponse

from app.config import Settings, get_settings
from app.orchestration import NewsroomOrchestrator, build_default_orchestrator
from app.qa_player import build_qa_player_feed, load_qa_player_html
from app.telemetry import configure_telemetry
from app.schemas import (
    HealthStatus,
    HourlyPlaylistScriptsRequest,
    HourlyPlaylistScriptsResponse,
    QAPlayerFeed,
    RadioRundown,
    RadioRundownRequest,
    TeamUpdateBatchRequest,
    TeamUpdateBatchResponse,
    TeamUpdateReport,
    TeamUpdateReportRequest,
    UsageTelemetrySnapshot,
)
from app.usage_dashboard import load_usage_dashboard_html

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    orchestrator: NewsroomOrchestrator | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = settings or get_settings()
        resolved_settings.model_dump()
        app.state.settings = resolved_settings
        app.state.telemetry = configure_telemetry(resolved_settings)
        app.state.orchestrator = orchestrator or build_default_orchestrator(resolved_settings)
        logger.info("Application started")
        try:
            yield
        finally:
            if app.state.orchestrator is not None and hasattr(app.state.orchestrator, "close"):
                await app.state.orchestrator.close()
                logger.info("Application adapters closed")
            if app.state.telemetry is not None and hasattr(app.state.telemetry, "close"):
                app.state.telemetry.close()
                logger.info("Telemetry closed")

    app = FastAPI(title="T4L Radio Agency", lifespan=lifespan)
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.state.telemetry = None

    def _get_orchestrator() -> NewsroomOrchestrator:
        if app.state.orchestrator is None:
            resolved_settings = app.state.settings or get_settings()
            app.state.settings = resolved_settings
            app.state.orchestrator = build_default_orchestrator(resolved_settings)
        return app.state.orchestrator

    @app.get("/healthz", response_model=HealthStatus)
    async def healthz() -> HealthStatus:
        resolved_settings = app.state.settings or get_settings()
        return HealthStatus(
            status="ok",
            config_ready=True,
            models=resolved_settings.model_summary(),
            external_services=resolved_settings.readiness_summary(),
        )

    @app.post("/orchestrations/radio-rundown", response_model=RadioRundown)
    async def create_radio_rundown(request: RadioRundownRequest) -> RadioRundown:
        logger.info("Received radio rundown request", extra={"teams": request.teams})
        return await _get_orchestrator().run_radio_rundown(request)

    @app.post("/orchestrations/team-update-report", response_model=TeamUpdateReport)
    async def create_team_update_report(request: TeamUpdateReportRequest) -> TeamUpdateReport:
        logger.info("Received team update report request", extra={"team": request.team})
        return await _get_orchestrator().run_team_update_report(request)

    @app.post("/orchestrations/team-update-reports", response_model=TeamUpdateBatchResponse)
    async def create_team_update_reports(
        request: TeamUpdateBatchRequest,
    ) -> TeamUpdateBatchResponse:
        logger.info("Received team update batch request", extra={"teams": request.teams})
        return await _get_orchestrator().run_team_update_reports(request)

    @app.post("/orchestrations/hourly-playlist-scripts", response_model=HourlyPlaylistScriptsResponse)
    async def create_hourly_playlist_scripts(
        request: HourlyPlaylistScriptsRequest,
    ) -> HourlyPlaylistScriptsResponse:
        logger.info("Received hourly playlist scripts request")
        return await _get_orchestrator().run_hourly_playlist_scripts(request)

    @app.get("/qa/player", response_class=HTMLResponse)
    async def qa_player() -> HTMLResponse:
        return HTMLResponse(load_qa_player_html())

    @app.get("/qa/player-feed", response_model=QAPlayerFeed)
    def qa_player_feed(batch: str | None = Query(default=None)) -> QAPlayerFeed:
        resolved_settings = app.state.settings or get_settings()
        return build_qa_player_feed(resolved_settings, batch=batch)

    @app.get("/observability/usage", response_class=HTMLResponse)
    async def usage_dashboard() -> HTMLResponse:
        return HTMLResponse(load_usage_dashboard_html())

    @app.get("/observability/usage/live", response_model=UsageTelemetrySnapshot)
    async def usage_dashboard_live(
        from_: datetime | None = Query(default=None, alias="from"),
        to_: datetime | None = Query(default=None, alias="to"),
        run_id: str | None = Query(default=None),
    ) -> UsageTelemetrySnapshot:
        telemetry = app.state.telemetry
        if telemetry is None or not hasattr(telemetry, "dashboard_snapshot"):
            return UsageTelemetrySnapshot(
                status="ok",
                telemetry_enabled=False,
                export_configured=False,
                generated_at=datetime.now(UTC),
                applied_filters={
                    "from": from_.isoformat() if from_ is not None else None,
                    "to": to_.isoformat() if to_ is not None else None,
                    "run_id": run_id.strip() if run_id else None,
                },
            )
        return UsageTelemetrySnapshot.model_validate(
            telemetry.dashboard_snapshot(
                from_time=from_,
                to_time=to_,
                run_id=run_id.strip() if run_id else None,
            )
        )

    return app


app = create_app()
