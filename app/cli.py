# This file defines the command-line interface (CLI) for the application using the Typer library.
#  It includes a command to run the radio show rundown generation process, which accepts various parameters.

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, TypeVar

import typer

from app.constants import DEFAULT_SCRIPT_LANGUAGE, NFL_TEAMS
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.orchestration import build_default_orchestrator
from app.telemetry import TelemetryManager, configure_telemetry
from app.schemas import (
    HourlyPlaylistScriptsRequest,
    RadioRundownRequest,
    TeamUpdateBatchRequest,
    TeamUpdateReportRequest,
)
from app.adapters import GeminiTTSBatchAdapter
from app.newsroom.helpers import build_tts_batch_process_request

configure_logging()

logger = logging.getLogger(__name__)

app = typer.Typer(help="Generate a structured NFL radio show rundown.")
RunResultT = TypeVar("RunResultT")


def _build_runtime() -> tuple[object, TelemetryManager, object]:
    settings = get_settings()
    if isinstance(settings, Settings):
        telemetry = configure_telemetry(settings)
    else:
        telemetry = TelemetryManager(enabled=False)
    orchestrator = build_default_orchestrator(settings)
    return settings, telemetry, orchestrator


async def _close_runtime(orchestrator: object, telemetry: TelemetryManager) -> None:
    try:
        if hasattr(orchestrator, "close"):
            await orchestrator.close()
    finally:
        telemetry.close()


async def _run_with_runtime_cleanup(
    orchestrator: object,
    telemetry: TelemetryManager,
    awaitable: Awaitable[RunResultT],
) -> RunResultT:
    try:
        return await awaitable
    finally:
        await _close_runtime(orchestrator, telemetry)


@app.callback()
def main() -> None:
    """CLI entrypoint."""


@app.command("run")
def run_rundown(
    lookback_hours: int = typer.Option(24, min=1, max=168),
    target_duration_minutes: int = typer.Option(60, min=15, max=240),
    max_segments: int = typer.Option(8, min=1, max=16),
    teams: list[str] | None = typer.Option(
        None,
        "--team",
        help="Optional team code filter. Repeat --team to limit the run to specific teams, for example --team ARI --team MIN.",
    ),
    output_json: Path | None = typer.Option(None),
) -> None:
    request = RadioRundownRequest(
        lookback_hours=lookback_hours,
        target_duration_minutes=target_duration_minutes,
        max_segments=max_segments,
        teams=teams,
    )
    _settings, telemetry, orchestrator = _build_runtime()
    rundown = asyncio.run(
        _run_with_runtime_cleanup(
            orchestrator,
            telemetry,
            orchestrator.run_radio_rundown(request),
        )
    )

    typer.echo(
        f"Run {rundown.run_id} | {rundown.target_duration_minutes} minute show | "
        f"{len(rundown.segments)} primary segments"
    )
    if request.teams:
        typer.echo(f"Teams: {', '.join(request.teams)}")
        typer.echo("")
    typer.echo("")
    for segment in rundown.segments:
        teams = ", ".join(segment.teams) if segment.teams else "League-wide"
        typer.echo(
            f"{segment.rank}. {segment.headline} [{teams}] | "
            f"{segment.recommended_duration_seconds // 60}m "
            f"{segment.recommended_duration_seconds % 60:02d}s | "
            f"{segment.recommended_word_count} words"
        )
        typer.echo(f"   Angle: {segment.editorial_angle}")
        typer.echo(f"   Summary: {segment.summary}")
        typer.echo(f"   Sources: {', '.join(str(item.url) for item in segment.source_articles)}")
        typer.echo("")

    if output_json is not None:
        output_json.write_text(rundown.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"Wrote JSON to {output_json}")


@app.command("team-update")
def run_team_update(
    teams: list[str] | None = typer.Option(
        None,
        "--team",
        help=(
            "Optional team filter. Repeat --team or pass one bracketed list, for example "
            '--team MIN --team BAL or --team "[NYJ, ARI, BAL]". If omitted, all 32 teams run.'
        ),
    ),
    lookback_minutes: int = typer.Option(60, min=1, max=10080),
    output_json: Path | None = typer.Option(None),
) -> None:
    team_codes = _parse_team_update_targets(teams)
    _settings, telemetry, orchestrator = _build_runtime()
    reports = asyncio.run(
        _run_with_runtime_cleanup(
            orchestrator,
            telemetry,
            orchestrator.run_team_update_reports(
                TeamUpdateBatchRequest(
                    teams=team_codes,
                    lookback_minutes=lookback_minutes,
                )
            )
        )
    )
    payload = reports.model_dump(mode="json")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    typer.echo(serialized)

    if output_json is not None:
        output_json.write_text(serialized, encoding="utf-8")
        typer.echo(f"Wrote JSON to {output_json}", err=True)


@app.command("write-scripts")
def run_write_scripts(
    playlist_id: str | None = typer.Option(
        None,
        "--playlist-id",
        help="Optional saved playlist id. If omitted, the latest saved hourly playlist is used.",
    ),
    language: str = typer.Option(
        DEFAULT_SCRIPT_LANGUAGE,
        "--language",
        help="Script language. Supported values: en-US, de-DE.",
    ),
    no_tts: bool = typer.Option(
        False,
        "--no-tts",
        help="Skip Gemini TTS batch rendering and return scripts only.",
    ),
    output_json: Path | None = typer.Option(None),
) -> None:
    _settings, telemetry, orchestrator = _build_runtime()
    response = asyncio.run(
        _run_with_runtime_cleanup(
            orchestrator,
            telemetry,
            orchestrator.run_hourly_playlist_scripts(
                HourlyPlaylistScriptsRequest(
                    playlist_id=playlist_id,
                    language=language,
                    enable_tts=not no_tts,
                )
            )
        )
    )
    serialized = json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)
    typer.echo(serialized)

    if output_json is not None:
        output_json.write_text(serialized, encoding="utf-8")
        typer.echo(f"Wrote JSON to {output_json}", err=True)


@app.command("process-tts")
def run_process_tts(
    batch_id: str = typer.Option(
        ...,
        "--batch-id",
        help="Batch ID from a previous TTS batch creation.",
    ),
) -> None:
    """Process an already-created TTS batch: download audio from Gemini and upload to Supabase Storage."""
    settings = get_settings()
    tts_batch = GeminiTTSBatchAdapter(
        endpoint_url=str(settings.gemini_tts_batch_url),
        process_timeout_seconds=settings.gemini_tts_batch_timeout_seconds,
    )
    request = build_tts_batch_process_request(batch_id=batch_id, settings=settings)
    result = asyncio.run(tts_batch.process_batch(request))
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _parse_team_update_targets(raw_teams: list[str] | None) -> list[str]:
    if not raw_teams:
        return list(NFL_TEAMS.keys())

    alias_map = {
        "NJY": "NYJ",
    }
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_team in raw_teams:
        cleaned = raw_team.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        for part in cleaned.split(","):
            token = part.strip().strip("'\"")
            if not token:
                continue
            code = alias_map.get(token.upper(), token.upper())
            validated = TeamUpdateReportRequest(team=code).team
            if validated not in seen:
                seen.add(validated)
                normalized.append(validated)

    return normalized or list(NFL_TEAMS.keys())
