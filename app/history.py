# This file contains local SQLite persistence for generated team update reports.

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

from app.schemas import (
    HourlyPlaylist,
    HourlyPlaylistHistoryEntry,
    HourlyStoryScriptHistoryEntry,
    RadioStoryScript,
    TeamUpdateHistoryEntry,
    TeamUpdateReport,
)


class TeamUpdateHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS team_update_history (
                    id TEXT PRIMARY KEY,
                    batch_run_id TEXT NOT NULL,
                    team_code TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    production_status TEXT NOT NULL DEFAULT 'tracked_only',
                    production_rank INTEGER,
                    playlist_id TEXT,
                    source_story_ids TEXT NOT NULL,
                    source_urls TEXT NOT NULL,
                    source_group_ids TEXT NOT NULL,
                    report_json TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_team_update_history_team_generated_at
                ON team_update_history(team_code, generated_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly_playlist_history (
                    id TEXT PRIMARY KEY,
                    batch_run_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    lookback_minutes INTEGER NOT NULL,
                    selected_count INTEGER NOT NULL,
                    playlist_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly_story_script_history (
                    id TEXT PRIMARY KEY,
                    script_run_id TEXT NOT NULL,
                    playlist_id TEXT NOT NULL,
                    batch_run_id TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en-US',
                    team_code TEXT NOT NULL,
                    playlist_rank INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    script_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "team_update_history", "batch_run_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                connection,
                "team_update_history",
                "production_status",
                "TEXT NOT NULL DEFAULT 'tracked_only'",
            )
            self._ensure_column(connection, "team_update_history", "production_rank", "INTEGER")
            self._ensure_column(connection, "team_update_history", "playlist_id", "TEXT")
            self._ensure_column(
                connection,
                "hourly_story_script_history",
                "language",
                "TEXT NOT NULL DEFAULT 'en-US'",
            )

    def list_recent_report_ready_entries(
        self,
        *,
        team_code: str,
        generated_after: datetime,
    ) -> list[TeamUpdateHistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    batch_run_id,
                    team_code,
                    generated_at,
                    status,
                    production_status,
                    production_rank,
                    playlist_id,
                    source_story_ids,
                    source_urls,
                    source_group_ids,
                    report_json
                FROM team_update_history
                WHERE team_code = ? AND status = 'report_ready' AND generated_at >= ?
                ORDER BY generated_at DESC
                """,
                (team_code, generated_after.isoformat()),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def list_batch_entries(self, *, batch_run_id: str) -> list[TeamUpdateHistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    batch_run_id,
                    team_code,
                    generated_at,
                    status,
                    production_status,
                    production_rank,
                    playlist_id,
                    source_story_ids,
                    source_urls,
                    source_group_ids,
                    report_json
                FROM team_update_history
                WHERE batch_run_id = ?
                ORDER BY team_code ASC
                """,
                (batch_run_id,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def save_report(
        self,
        *,
        report: TeamUpdateReport,
        batch_run_id: str,
        source_story_ids: list[str],
        source_urls: list[str],
        source_group_ids: list[str],
        production_status: str = "tracked_only",
        production_rank: int | None = None,
        playlist_id: str | None = None,
    ) -> str:
        record_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO team_update_history (
                    id,
                    batch_run_id,
                    team_code,
                    generated_at,
                    status,
                    production_status,
                    production_rank,
                    playlist_id,
                    source_story_ids,
                    source_urls,
                    source_group_ids,
                    report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    batch_run_id,
                    report.team,
                    report.generated_at.isoformat(),
                    report.status,
                    production_status,
                    production_rank,
                    playlist_id,
                    json.dumps(source_story_ids),
                    json.dumps(source_urls),
                    json.dumps(source_group_ids),
                    json.dumps(report.report.model_dump(mode="json")) if report.report is not None else None,
                ),
            )
        return record_id

    def mark_reports_put_to_production(
        self,
        *,
        batch_run_id: str,
        playlist_id: str,
        ranked_team_codes: list[tuple[int, str]],
    ) -> None:
        with self._connect() as connection:
            for production_rank, team_code in ranked_team_codes:
                connection.execute(
                    """
                    UPDATE team_update_history
                    SET production_status = 'put_to_production',
                        production_rank = ?,
                        playlist_id = ?
                    WHERE batch_run_id = ? AND team_code = ?
                    """,
                    (production_rank, playlist_id, batch_run_id, team_code),
                )

    def save_hourly_playlist(
        self,
        *,
        batch_run_id: str,
        generated_at: datetime,
        lookback_minutes: int,
        playlist: HourlyPlaylist,
    ) -> str:
        playlist_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO hourly_playlist_history (
                    id,
                    batch_run_id,
                    generated_at,
                    lookback_minutes,
                    selected_count,
                    playlist_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    playlist_id,
                    batch_run_id,
                    generated_at.isoformat(),
                    lookback_minutes,
                    playlist.selected_count,
                    json.dumps(playlist.model_dump(mode="json")),
                ),
            )
        return playlist_id

    def list_hourly_playlists(self) -> list[HourlyPlaylistHistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, batch_run_id, generated_at, lookback_minutes, selected_count, playlist_json
                FROM hourly_playlist_history
                ORDER BY generated_at DESC
                """
            ).fetchall()
        return [self._playlist_row_to_entry(row) for row in rows]

    def get_latest_hourly_playlist(self) -> HourlyPlaylistHistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, batch_run_id, generated_at, lookback_minutes, selected_count, playlist_json
                FROM hourly_playlist_history
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._playlist_row_to_entry(row) if row is not None else None

    def get_hourly_playlist(self, *, playlist_id: str) -> HourlyPlaylistHistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, batch_run_id, generated_at, lookback_minutes, selected_count, playlist_json
                FROM hourly_playlist_history
                WHERE id = ?
                LIMIT 1
                """,
                (playlist_id,),
            ).fetchone()
        return self._playlist_row_to_entry(row) if row is not None else None

    def list_produced_reports_for_playlist(
        self,
        *,
        playlist_id: str,
        batch_run_id: str,
    ) -> list[TeamUpdateHistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    batch_run_id,
                    team_code,
                    generated_at,
                    status,
                    production_status,
                    production_rank,
                    playlist_id,
                    source_story_ids,
                    source_urls,
                    source_group_ids,
                    report_json
                FROM team_update_history
                WHERE batch_run_id = ?
                  AND playlist_id = ?
                  AND production_status = 'put_to_production'
                ORDER BY production_rank ASC, team_code ASC
                """,
                (batch_run_id, playlist_id),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def save_story_scripts(
        self,
        *,
        script_run_id: str,
        playlist_id: str,
        batch_run_id: str,
        generated_at: datetime,
        scripts: list[RadioStoryScript],
    ) -> None:
        with self._connect() as connection:
            for script in scripts:
                connection.execute(
                    """
                    INSERT INTO hourly_story_script_history (
                        id,
                        script_run_id,
                        playlist_id,
                        batch_run_id,
                        language,
                        team_code,
                        playlist_rank,
                        generated_at,
                        duration_seconds,
                        script_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        script_run_id,
                        playlist_id,
                        batch_run_id,
                        script.language,
                        script.team,
                        script.playlist_rank,
                        generated_at.isoformat(),
                        script.duration_seconds,
                        json.dumps(script.model_dump(mode="json")),
                    ),
                )

    def list_story_scripts(
        self,
        *,
        playlist_id: str | None = None,
        script_run_id: str | None = None,
    ) -> list[HourlyStoryScriptHistoryEntry]:
        query = """
            SELECT id, script_run_id, playlist_id, batch_run_id, language, team_code, playlist_rank, generated_at,
                   duration_seconds, script_json
            FROM hourly_story_script_history
        """
        params: list[str] = []
        clauses: list[str] = []
        if playlist_id is not None:
            clauses.append("playlist_id = ?")
            params.append(playlist_id)
        if script_run_id is not None:
            clauses.append("script_run_id = ?")
            params.append(script_run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY generated_at DESC, playlist_rank ASC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._script_row_to_entry(row) for row in rows]

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        ddl: str,
    ) -> None:
        column_names = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in column_names:
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> TeamUpdateHistoryEntry:
        return TeamUpdateHistoryEntry(
            id=str(row["id"]),
            batch_run_id=str(row["batch_run_id"] or ""),
            team_code=str(row["team_code"]),
            generated_at=datetime.fromisoformat(str(row["generated_at"])),
            status=str(row["status"]),
            production_status=str(row["production_status"] or "tracked_only"),
            production_rank=row["production_rank"],
            playlist_id=row["playlist_id"],
            source_story_ids=json.loads(str(row["source_story_ids"])),
            source_urls=json.loads(str(row["source_urls"])),
            source_group_ids=json.loads(str(row["source_group_ids"])),
            report_json=json.loads(row["report_json"]) if row["report_json"] is not None else None,
        )

    @staticmethod
    def _playlist_row_to_entry(row: sqlite3.Row) -> HourlyPlaylistHistoryEntry:
        return HourlyPlaylistHistoryEntry(
            id=str(row["id"]),
            batch_run_id=str(row["batch_run_id"]),
            generated_at=datetime.fromisoformat(str(row["generated_at"])),
            lookback_minutes=int(row["lookback_minutes"]),
            selected_count=int(row["selected_count"]),
            playlist_json=json.loads(str(row["playlist_json"])),
        )

    @staticmethod
    def _script_row_to_entry(row: sqlite3.Row) -> HourlyStoryScriptHistoryEntry:
        return HourlyStoryScriptHistoryEntry(
            id=str(row["id"]),
            script_run_id=str(row["script_run_id"]),
            playlist_id=str(row["playlist_id"]),
            batch_run_id=str(row["batch_run_id"]),
            language=str(row["language"] or "en-US"),
            team_code=str(row["team_code"]),
            playlist_rank=int(row["playlist_rank"]),
            generated_at=datetime.fromisoformat(str(row["generated_at"])),
            duration_seconds=int(row["duration_seconds"]),
            script_json=json.loads(str(row["script_json"])),
        )
