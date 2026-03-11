# This file defines the command-line interface (CLI) for the application using the Typer library.
#  It includes a command to run the radio show rundown generation process, which accepts various parameters.

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

from app.constants import DEFAULT_SCRIPT_LANGUAGE, NFL_TEAMS
from app.config import get_settings
from app.logging_config import configure_logging
from app.orchestration import build_default_orchestrator
from app.schemas import (
    HourlyPlaylistScriptsRequest,
    RadioRundownRequest,
    TeamUpdateBatchRequest,
    TeamUpdateReportRequest,
)

configure_logging()

logger = logging.getLogger(__name__)

app = typer.Typer(help="Generate a structured NFL radio show rundown.")


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
    settings = get_settings()
    orchestrator = build_default_orchestrator(settings)
    rundown = asyncio.run(orchestrator.run_radio_rundown(request))

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
    settings = get_settings()
    orchestrator = build_default_orchestrator(settings)
    reports = asyncio.run(
        orchestrator.run_team_update_reports(
            TeamUpdateBatchRequest(
                teams=team_codes,
                lookback_minutes=lookback_minutes,
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
    settings = get_settings()
    orchestrator = build_default_orchestrator(settings)
    response = asyncio.run(
        orchestrator.run_hourly_playlist_scripts(
            HourlyPlaylistScriptsRequest(
                playlist_id=playlist_id,
                language=language,
                enable_tts=not no_tts,
            )
        )
    )
    serialized = json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)
    typer.echo(serialized)

    if output_json is not None:
        output_json.write_text(serialized, encoding="utf-8")
        typer.echo(f"Wrote JSON to {output_json}", err=True)


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
