from __future__ import annotations

import json
import logging
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from agents import set_trace_processors
from agents.tracing import TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    ResponseSpanData,
    SpanData,
)
from agents.tracing.spans import Span
from agents.tracing.traces import Trace
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.constants import WORKFLOW_NAME
from app.config import Settings

logger = logging.getLogger(__name__)
_ACTIVE_TELEMETRY_MANAGER: "TelemetryManager | None" = None

_REQUESTS_METRIC = "llm_requests_total"
_INPUT_TOKENS_METRIC = "llm_input_tokens_total"
_OUTPUT_TOKENS_METRIC = "llm_output_tokens_total"
_CACHED_INPUT_TOKENS_METRIC = "llm_cached_input_tokens_total"
_REASONING_TOKENS_METRIC = "llm_reasoning_tokens_total"
_ESTIMATED_COST_METRIC = "llm_estimated_cost_usd_total"
_ACTIVE_RUNS_METRIC = "llm_active_runs"
_DEFAULT_EXPORT_INTERVAL_MILLIS = 5000

_OPENAI_PRICE_MAP: dict[str, dict[str, float]] = {
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
}
_MODEL_PRICE_FALLBACKS = {
    "gpt-5.4": "gpt-5.2",
}
_GOOGLE_BATCH_PRICE_MAP: dict[str, dict[str, float]] = {
    "gemini-2.5-pro-preview-tts": {"input": 0.50, "cached_input": 0.0, "output": 10.00},
    "gemini-2.5-flash-preview-tts": {"input": 0.25, "cached_input": 0.0, "output": 5.00},
}


@dataclass(frozen=True)
class NormalizedUsageRecord:
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class UsageEventRecord:
    occurred_at: datetime
    workflow: str
    run_id: str
    stage: str
    agent_name: str
    model: str
    provider: str
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class ActiveRunRecord:
    run_id: str
    workflow: str
    stage: str
    started_at: datetime


class UsageSnapshotStore:
    def __init__(self, recent_event_limit: int = 100, db_path: Path | None = None) -> None:
        self._db_path = db_path
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_db()
        self._lock = Lock()
        self._recent_events: deque[UsageEventRecord] = deque(maxlen=recent_event_limit)
        self._totals = self._empty_counter()
        self._by_agent: dict[str, dict[str, float | int]] = {}
        self._by_model: dict[str, dict[str, float | int]] = {}
        self._active_runs: dict[str, ActiveRunRecord] = {}

    def record_trace_start(self, trace_attrs: dict[str, str]) -> None:
        run_id = trace_attrs.get("run_id")
        if not run_id:
            return
        if self._db_path is not None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO usage_active_runs (run_id, workflow, stage, started_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        workflow = excluded.workflow,
                        stage = excluded.stage,
                        started_at = excluded.started_at
                    """,
                    (
                        run_id,
                        trace_attrs.get("workflow", WORKFLOW_NAME),
                        trace_attrs.get("stage", "unknown"),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                connection.commit()
        with self._lock:
            self._active_runs[run_id] = ActiveRunRecord(
                run_id=run_id,
                workflow=trace_attrs.get("workflow", WORKFLOW_NAME),
                stage=trace_attrs.get("stage", "unknown"),
                started_at=datetime.now(UTC),
            )

    def record_trace_end(self, trace_attrs: dict[str, str]) -> None:
        run_id = trace_attrs.get("run_id")
        if not run_id:
            return
        if self._db_path is not None:
            with self._connect() as connection:
                connection.execute("DELETE FROM usage_active_runs WHERE run_id = ?", (run_id,))
                connection.commit()
        with self._lock:
            self._active_runs.pop(run_id, None)

    def record_usage(
        self,
        *,
        trace_attrs: dict[str, str],
        metric_attrs: dict[str, str],
        usage: NormalizedUsageRecord,
    ) -> None:
        event = UsageEventRecord(
            occurred_at=datetime.now(UTC),
            workflow=trace_attrs.get("workflow", WORKFLOW_NAME),
            run_id=trace_attrs.get("run_id", "unknown"),
            stage=trace_attrs.get("stage", "unknown"),
            agent_name=metric_attrs.get("agent_name", "unknown"),
            model=metric_attrs.get("model", "unknown"),
            provider=metric_attrs.get("provider", "openai"),
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
        )
        if self._db_path is not None:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO usage_events (
                        occurred_at,
                        workflow,
                        run_id,
                        stage,
                        agent_name,
                        model,
                        provider,
                        requests,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens,
                        reasoning_tokens,
                        total_tokens,
                        estimated_cost_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.occurred_at.isoformat(),
                        event.workflow,
                        event.run_id,
                        event.stage,
                        event.agent_name,
                        event.model,
                        event.provider,
                        event.requests,
                        event.input_tokens,
                        event.output_tokens,
                        event.cached_input_tokens,
                        event.reasoning_tokens,
                        event.total_tokens,
                        event.estimated_cost_usd,
                    ),
                )
                connection.commit()
        with self._lock:
            self._recent_events.append(event)
            self._add_usage(self._totals, usage)
            self._add_usage(self._bucket(self._by_agent, event.agent_name), usage)
            self._add_usage(self._bucket(self._by_model, event.model), usage)

    def snapshot(
        self,
        *,
        enabled: bool,
        export_configured: bool,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if self._db_path is not None:
            return self._db_snapshot(
                enabled=enabled,
                export_configured=export_configured,
                from_time=from_time,
                to_time=to_time,
                run_id=run_id,
            )
        with self._lock:
            filtered_events = [
                event
                for event in self._recent_events
                if self._matches_event_filters(event, from_time=from_time, to_time=to_time, run_id=run_id)
            ]
            filtered_totals = self._empty_counter()
            filtered_by_agent: dict[str, dict[str, float | int]] = {}
            filtered_by_model: dict[str, dict[str, float | int]] = {}
            for event in filtered_events:
                usage = NormalizedUsageRecord(
                    requests=event.requests,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    cached_input_tokens=event.cached_input_tokens,
                    reasoning_tokens=event.reasoning_tokens,
                    total_tokens=event.total_tokens,
                    estimated_cost_usd=event.estimated_cost_usd,
                )
                self._add_usage(filtered_totals, usage)
                self._add_usage(self._bucket(filtered_by_agent, event.agent_name), usage)
                self._add_usage(self._bucket(filtered_by_model, event.model), usage)

            active_runs = sorted(
                [
                    run
                    for run in self._active_runs.values()
                    if self._matches_active_run_filters(run, from_time=from_time, to_time=to_time, run_id=run_id)
                ],
                key=lambda item: item.started_at,
                reverse=True,
            )
            active_runs_by_stage: dict[str, int] = {}
            for run in active_runs:
                active_runs_by_stage[run.stage] = active_runs_by_stage.get(run.stage, 0) + 1

            return {
                "status": "ok",
                "telemetry_enabled": enabled,
                "export_configured": export_configured,
                "generated_at": datetime.now(UTC),
                "totals": self._counter_payload(filtered_totals),
                "active_runs": [
                    {
                        "run_id": run.run_id,
                        "workflow": run.workflow,
                        "stage": run.stage,
                        "started_at": run.started_at,
                    }
                    for run in active_runs
                ],
                "active_runs_by_stage": active_runs_by_stage,
                "usage_by_agent": self._sorted_breakdowns(filtered_by_agent),
                "usage_by_model": self._sorted_breakdowns(filtered_by_model),
                "recent_events": [
                    {
                        "occurred_at": event.occurred_at,
                        "workflow": event.workflow,
                        "run_id": event.run_id,
                        "stage": event.stage,
                        "agent_name": event.agent_name,
                        "model": event.model,
                        "provider": event.provider,
                        "requests": event.requests,
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "cached_input_tokens": event.cached_input_tokens,
                        "reasoning_tokens": event.reasoning_tokens,
                        "total_tokens": event.total_tokens,
                        "estimated_cost_usd": event.estimated_cost_usd,
                    }
                    for event in reversed(filtered_events)
                ],
                "applied_filters": self._filter_payload(from_time=from_time, to_time=to_time, run_id=run_id),
            }

    def list_historical_runs(
        self,
        *,
        enabled: bool,
        export_configured: bool,
        limit: int = 200,
    ) -> dict[str, Any]:
        if self._db_path is not None:
            return self._db_historical_runs(enabled=enabled, export_configured=export_configured, limit=limit)

        with self._lock:
            active_run_ids = set(self._active_runs.keys())
            grouped: dict[str, dict[str, Any]] = {}
            for event in self._recent_events:
                if event.run_id in active_run_ids:
                    continue
                row = grouped.get(event.run_id)
                if row is None or event.occurred_at > row["finished_at"]:
                    grouped[event.run_id] = {
                        "run_id": event.run_id,
                        "workflow": event.workflow,
                        "finished_at": event.occurred_at,
                    }
            runs = sorted(grouped.values(), key=lambda item: item["finished_at"], reverse=True)[:limit]
            return {
                "status": "ok",
                "telemetry_enabled": enabled,
                "export_configured": export_configured,
                "generated_at": datetime.now(UTC),
                "runs": runs,
            }

    def _db_snapshot(
        self,
        *,
        enabled: bool,
        export_configured: bool,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        event_where_sql, event_params = self._event_where_clause(from_time=from_time, to_time=to_time, run_id=run_id)
        run_where_sql, run_params = self._active_run_where_clause(from_time=from_time, to_time=to_time, run_id=run_id)
        with self._connect() as connection:
            total_row = connection.execute(
                f"""
                SELECT
                    COALESCE(SUM(requests), 0) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_events
                {event_where_sql}
                """,
                event_params,
            ).fetchone()
            active_rows = connection.execute(
                f"""
                SELECT run_id, workflow, stage, started_at
                FROM usage_active_runs
                {run_where_sql}
                ORDER BY started_at DESC
                """,
                run_params,
            ).fetchall()
            agent_rows = connection.execute(
                f"""
                SELECT
                    agent_name AS key,
                    COALESCE(SUM(requests), 0) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_events
                {event_where_sql}
                GROUP BY agent_name
                ORDER BY total_tokens DESC, input_tokens DESC
                """,
                event_params,
            ).fetchall()
            model_rows = connection.execute(
                f"""
                SELECT
                    model AS key,
                    COALESCE(SUM(requests), 0) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                    COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_events
                {event_where_sql}
                GROUP BY model
                ORDER BY total_tokens DESC, input_tokens DESC
                """,
                event_params,
            ).fetchall()
            event_rows = connection.execute(
                f"""
                SELECT
                    occurred_at,
                    workflow,
                    run_id,
                    stage,
                    agent_name,
                    model,
                    provider,
                    requests,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    reasoning_tokens,
                    total_tokens,
                    estimated_cost_usd
                FROM usage_events
                {event_where_sql}
                ORDER BY occurred_at DESC
                LIMIT 100
                """,
                event_params,
            ).fetchall()

        active_runs = [
            {
                "run_id": row["run_id"],
                "workflow": row["workflow"],
                "stage": row["stage"],
                "started_at": datetime.fromisoformat(row["started_at"]),
            }
            for row in active_rows
        ]
        active_runs_by_stage: dict[str, int] = {}
        for run in active_runs:
            active_runs_by_stage[run["stage"]] = active_runs_by_stage.get(run["stage"], 0) + 1

        return {
            "status": "ok",
            "telemetry_enabled": enabled,
            "export_configured": export_configured,
            "generated_at": datetime.now(UTC),
            "totals": self._counter_payload(dict(total_row) if total_row is not None else self._empty_counter()),
            "active_runs": active_runs,
            "active_runs_by_stage": active_runs_by_stage,
            "usage_by_agent": [self._counter_payload(dict(row)) | {"key": row["key"]} for row in agent_rows],
            "usage_by_model": [self._counter_payload(dict(row)) | {"key": row["key"]} for row in model_rows],
            "recent_events": [
                {
                    "occurred_at": datetime.fromisoformat(row["occurred_at"]),
                    "workflow": row["workflow"],
                    "run_id": row["run_id"],
                    "stage": row["stage"],
                    "agent_name": row["agent_name"],
                    "model": row["model"],
                    "provider": row["provider"],
                    "requests": row["requests"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "cached_input_tokens": row["cached_input_tokens"],
                    "reasoning_tokens": row["reasoning_tokens"],
                    "total_tokens": row["total_tokens"],
                    "estimated_cost_usd": row["estimated_cost_usd"],
                }
                for row in event_rows
            ],
            "applied_filters": self._filter_payload(from_time=from_time, to_time=to_time, run_id=run_id),
        }

    def _db_historical_runs(
        self,
        *,
        enabled: bool,
        export_configured: bool,
        limit: int,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    usage_events.run_id AS run_id,
                    usage_events.workflow AS workflow,
                    MAX(usage_events.occurred_at) AS finished_at
                FROM usage_events
                LEFT JOIN usage_active_runs
                    ON usage_active_runs.run_id = usage_events.run_id
                WHERE usage_active_runs.run_id IS NULL
                GROUP BY usage_events.run_id, usage_events.workflow
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "status": "ok",
            "telemetry_enabled": enabled,
            "export_configured": export_configured,
            "generated_at": datetime.now(UTC),
            "runs": [
                {
                    "run_id": row["run_id"],
                    "workflow": row["workflow"],
                    "finished_at": datetime.fromisoformat(row["finished_at"]),
                }
                for row in rows
            ],
        }

    def _initialize_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    requests INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_active_runs (
                    run_id TEXT PRIMARY KEY,
                    workflow TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    started_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_events_occurred_at
                ON usage_events(occurred_at DESC)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._db_path is None:
            raise RuntimeError("SQLite connection requested for in-memory usage store.")
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _filter_payload(
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        run_id: str | None,
    ) -> dict[str, str | None]:
        return {
            "from": from_time.isoformat() if from_time is not None else None,
            "to": to_time.isoformat() if to_time is not None else None,
            "run_id": run_id.strip() if run_id else None,
        }

    @staticmethod
    def _matches_event_filters(
        event: UsageEventRecord,
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        run_id: str | None,
    ) -> bool:
        if run_id and event.run_id != run_id:
            return False
        if from_time and event.occurred_at < from_time:
            return False
        if to_time and event.occurred_at > to_time:
            return False
        return True

    @staticmethod
    def _matches_active_run_filters(
        run: ActiveRunRecord,
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        run_id: str | None,
    ) -> bool:
        if run_id and run.run_id != run_id:
            return False
        if from_time and run.started_at < from_time:
            return False
        if to_time and run.started_at > to_time:
            return False
        return True

    @staticmethod
    def _event_where_clause(
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        run_id: str | None,
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        if from_time is not None:
            clauses.append("occurred_at >= ?")
            params.append(from_time.isoformat())
        if to_time is not None:
            clauses.append("occurred_at <= ?")
            params.append(to_time.isoformat())
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else "", tuple(params))

    @staticmethod
    def _active_run_where_clause(
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        run_id: str | None,
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        if from_time is not None:
            clauses.append("started_at >= ?")
            params.append(from_time.isoformat())
        if to_time is not None:
            clauses.append("started_at <= ?")
            params.append(to_time.isoformat())
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else "", tuple(params))

    def _bucket(self, container: dict[str, dict[str, float | int]], key: str) -> dict[str, float | int]:
        if key not in container:
            container[key] = self._empty_counter()
        return container[key]

    def _sorted_breakdowns(self, buckets: dict[str, dict[str, float | int]]) -> list[dict[str, float | int | str]]:
        rows = [
            {"key": key, **self._counter_payload(values)}
            for key, values in buckets.items()
        ]
        return sorted(rows, key=lambda row: (row["total_tokens"], row["input_tokens"]), reverse=True)

    @staticmethod
    def _empty_counter() -> dict[str, float | int]:
        return {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    @staticmethod
    def _add_usage(counter: dict[str, float | int], usage: NormalizedUsageRecord) -> None:
        counter["requests"] += usage.requests
        counter["input_tokens"] += usage.input_tokens
        counter["output_tokens"] += usage.output_tokens
        counter["cached_input_tokens"] += usage.cached_input_tokens
        counter["reasoning_tokens"] += usage.reasoning_tokens
        counter["total_tokens"] += usage.total_tokens
        counter["estimated_cost_usd"] += usage.estimated_cost_usd

    @staticmethod
    def _counter_payload(counter: dict[str, float | int]) -> dict[str, float | int]:
        return {
            "requests": int(counter["requests"]),
            "input_tokens": int(counter["input_tokens"]),
            "output_tokens": int(counter["output_tokens"]),
            "cached_input_tokens": int(counter["cached_input_tokens"]),
            "reasoning_tokens": int(counter["reasoning_tokens"]),
            "total_tokens": int(counter["total_tokens"]),
            "estimated_cost_usd": float(
                Decimal(str(counter["estimated_cost_usd"])).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            ),
        }


class UsageMetricsRecorder:
    def __init__(self, meter_provider: MeterProvider) -> None:
        meter = meter_provider.get_meter("t4l.telemetry", version="1.0")
        self._requests = meter.create_counter(_REQUESTS_METRIC)
        self._input_tokens = meter.create_counter(_INPUT_TOKENS_METRIC)
        self._output_tokens = meter.create_counter(_OUTPUT_TOKENS_METRIC)
        self._cached_input_tokens = meter.create_counter(_CACHED_INPUT_TOKENS_METRIC)
        self._reasoning_tokens = meter.create_counter(_REASONING_TOKENS_METRIC)
        self._estimated_cost = meter.create_counter(_ESTIMATED_COST_METRIC, unit="USD")
        self._active_runs = meter.create_up_down_counter(_ACTIVE_RUNS_METRIC)

    def record_usage(self, usage: NormalizedUsageRecord, attrs: dict[str, str]) -> None:
        self._requests.add(usage.requests, attrs)
        self._input_tokens.add(usage.input_tokens, attrs)
        self._output_tokens.add(usage.output_tokens, attrs)
        self._cached_input_tokens.add(usage.cached_input_tokens, attrs)
        self._reasoning_tokens.add(usage.reasoning_tokens, attrs)
        self._estimated_cost.add(usage.estimated_cost_usd, attrs)

    def increment_active_runs(self, attrs: dict[str, str]) -> None:
        self._active_runs.add(1, attrs)

    def decrement_active_runs(self, attrs: dict[str, str]) -> None:
        self._active_runs.add(-1, attrs)


@dataclass
class TelemetryManager:
    enabled: bool
    export_configured: bool = False
    processor: "UsageTelemetryProcessor | None" = None
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None
    metrics_recorder: UsageMetricsRecorder | None = None
    snapshot_store: UsageSnapshotStore | None = None

    def close(self) -> None:
        global _ACTIVE_TELEMETRY_MANAGER
        if self.processor is not None:
            set_trace_processors([])
        if self.tracer_provider is not None:
            try:
                self.tracer_provider.shutdown()
            except Exception:
                logger.warning("Unable to shut down OTLP trace exporter cleanly.", exc_info=True)
        if self.meter_provider is not None:
            try:
                self.meter_provider.shutdown()
            except Exception:
                logger.warning("Unable to shut down OTLP metric exporter cleanly.", exc_info=True)
        if _ACTIVE_TELEMETRY_MANAGER is self:
            _ACTIVE_TELEMETRY_MANAGER = None

    def dashboard_snapshot(
        self,
        *,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if self.snapshot_store is None:
            return {
                "status": "ok",
                "telemetry_enabled": self.enabled,
                "export_configured": self.export_configured,
                "generated_at": datetime.now(UTC),
                "totals": UsageSnapshotStore._counter_payload(UsageSnapshotStore._empty_counter()),
                "active_runs": [],
                "active_runs_by_stage": {},
                "usage_by_agent": [],
                "usage_by_model": [],
                "recent_events": [],
                "applied_filters": UsageSnapshotStore._filter_payload(
                    from_time=from_time,
                    to_time=to_time,
                    run_id=run_id,
                ),
            }
        return self.snapshot_store.snapshot(
            enabled=self.enabled,
            export_configured=self.export_configured,
            from_time=from_time,
            to_time=to_time,
            run_id=run_id,
        )

    def dashboard_comparison(
        self,
        *,
        left_run_id: str,
        right_run_id: str,
    ) -> dict[str, Any]:
        left_snapshot = self.dashboard_snapshot(run_id=left_run_id.strip())
        right_snapshot = self.dashboard_snapshot(run_id=right_run_id.strip())
        left_totals = left_snapshot["totals"]
        right_totals = right_snapshot["totals"]
        return {
            "status": "ok",
            "telemetry_enabled": self.enabled,
            "export_configured": self.export_configured,
            "generated_at": datetime.now(UTC),
            "left": left_snapshot,
            "right": right_snapshot,
            "delta": {
                "requests": int(right_totals["requests"]) - int(left_totals["requests"]),
                "input_tokens": int(right_totals["input_tokens"]) - int(left_totals["input_tokens"]),
                "output_tokens": int(right_totals["output_tokens"]) - int(left_totals["output_tokens"]),
                "cached_input_tokens": int(right_totals["cached_input_tokens"]) - int(
                    left_totals["cached_input_tokens"]
                ),
                "reasoning_tokens": int(right_totals["reasoning_tokens"]) - int(left_totals["reasoning_tokens"]),
                "total_tokens": int(right_totals["total_tokens"]) - int(left_totals["total_tokens"]),
                "estimated_cost_usd": float(right_totals["estimated_cost_usd"]) - float(
                    left_totals["estimated_cost_usd"]
                ),
            },
        }

    def dashboard_historical_runs(self, *, limit: int = 200) -> dict[str, Any]:
        if self.snapshot_store is None:
            return {
                "status": "ok",
                "telemetry_enabled": self.enabled,
                "export_configured": self.export_configured,
                "generated_at": datetime.now(UTC),
                "runs": [],
            }
        return self.snapshot_store.list_historical_runs(
            enabled=self.enabled,
            export_configured=self.export_configured,
            limit=limit,
        )

    def record_external_usage(
        self,
        *,
        workflow: str,
        run_id: str,
        stage: str,
        agent_name: str,
        provider: str,
        model: str,
        requests: int,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        reasoning_tokens: int = 0,
        batch_mode: bool = False,
    ) -> None:
        if not self.enabled or self.snapshot_store is None:
            return

        usage = NormalizedUsageRecord(
            requests=max(1, requests),
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            cached_input_tokens=max(0, cached_input_tokens),
            reasoning_tokens=max(0, reasoning_tokens),
            total_tokens=max(0, total_tokens or (input_tokens + output_tokens)),
            estimated_cost_usd=estimate_cost_usd(
                provider=provider,
                model=model,
                input_tokens=max(0, input_tokens),
                cached_input_tokens=max(0, cached_input_tokens),
                output_tokens=max(0, output_tokens),
                batch_mode=batch_mode,
            ),
        )
        trace_attrs = {
            "project": self.processor._settings.telemetry_project_name if self.processor is not None else "unknown",
            "environment": (
                self.processor._settings.telemetry_environment if self.processor is not None else "unknown"
            ),
            "instance_id": (
                self.processor._settings.resolved_telemetry_instance_id()
                if self.processor is not None
                else "unknown"
            ),
            "workflow": workflow,
            "run_id": run_id,
            "stage": stage,
            "provider": provider,
            "agent_name": agent_name,
            "model": model,
        }
        metric_attrs = {
            key: trace_attrs[key]
            for key in ("project", "environment", "instance_id", "workflow", "stage", "agent_name", "model", "provider")
        }
        self.snapshot_store.record_usage(trace_attrs=trace_attrs, metric_attrs=metric_attrs, usage=usage)
        if self.metrics_recorder is not None:
            self.metrics_recorder.record_usage(usage, metric_attrs)
        if self.tracer_provider is not None:
            tracer = self.tracer_provider.get_tracer("t4l.telemetry", "1.0")
            span = tracer.start_span(f"external_usage:{agent_name}", attributes=trace_attrs)
            span.set_attribute("llm.requests", usage.requests)
            span.set_attribute("llm.input_tokens", usage.input_tokens)
            span.set_attribute("llm.output_tokens", usage.output_tokens)
            span.set_attribute("llm.cached_input_tokens", usage.cached_input_tokens)
            span.set_attribute("llm.reasoning_tokens", usage.reasoning_tokens)
            span.set_attribute("llm.total_tokens", usage.total_tokens)
            span.set_attribute("llm.estimated_cost_usd", usage.estimated_cost_usd)
            span.end()


class UsageTelemetryProcessor(TracingProcessor):
    def __init__(
        self,
        *,
        settings: Settings,
        tracer_provider: TracerProvider,
        metrics_recorder: UsageMetricsRecorder,
        snapshot_store: UsageSnapshotStore,
    ) -> None:
        self._settings = settings
        self._tracer = tracer_provider.get_tracer("t4l.telemetry", "1.0")
        self._metrics = metrics_recorder
        self._snapshot_store = snapshot_store
        self._root_spans: dict[str, Any] = {}
        self._otel_spans: dict[tuple[str, str], Any] = {}
        self._span_parent_ids: dict[tuple[str, str], str | None] = {}
        self._agent_names: dict[tuple[str, str], str] = {}
        self._trace_info: dict[str, dict[str, str]] = {}

    def on_trace_start(self, trace: Trace) -> None:
        try:
            trace_attrs = self._trace_attributes(trace)
            root_span = self._tracer.start_span(trace.name or WORKFLOW_NAME, attributes=trace_attrs)
            self._root_spans[trace.trace_id] = root_span
            self._trace_info[trace.trace_id] = trace_attrs
            self._snapshot_store.record_trace_start(trace_attrs)
            self._metrics.increment_active_runs(self._metric_attrs_from_trace_attrs(trace_attrs))
        except Exception:
            logger.warning("Telemetry trace start failed.", exc_info=True)

    def on_trace_end(self, trace: Trace) -> None:
        try:
            trace_attrs = self._trace_info.get(trace.trace_id) or self._trace_attributes(trace)
            self._snapshot_store.record_trace_end(trace_attrs)
            self._metrics.decrement_active_runs(self._metric_attrs_from_trace_attrs(trace_attrs))
            root_span = self._root_spans.pop(trace.trace_id, None)
            if root_span is not None:
                root_span.end()
            self._trace_info.pop(trace.trace_id, None)
        except Exception:
            logger.warning("Telemetry trace end failed.", exc_info=True)

    def on_span_start(self, span: Span[Any]) -> None:
        try:
            span_data = span.span_data
            span_key = (span.trace_id, span.span_id)
            self._span_parent_ids[span_key] = span.parent_id
            parent_context = self._parent_context(span)
            attrs = self._span_attributes(span, span_data)
            otel_span = self._tracer.start_span(
                self._span_name(span_data),
                context=parent_context,
                attributes=attrs,
            )
            self._otel_spans[span_key] = otel_span
            if isinstance(span_data, AgentSpanData):
                self._agent_names[span_key] = span_data.name
        except Exception:
            logger.warning("Telemetry span start failed.", exc_info=True)

    def on_span_end(self, span: Span[Any]) -> None:
        span_key = (span.trace_id, span.span_id)
        try:
            span_data = span.span_data
            otel_span = self._otel_spans.pop(span_key, None)
            if otel_span is None:
                return

            if span.error:
                otel_span.set_attribute("llm.error", str(span.error))

            usage_payload, usage_model, usage_input, usage_output = _usage_payload_for_span_data(span_data)
            if usage_payload is not None:
                usage = normalize_usage_record(usage_payload, model=usage_model)
                if usage is not None:
                    metric_attrs = self._usage_metric_attributes(span, span_data)
                    trace_attrs = self._usage_trace_attributes(span, span_data)
                    self._snapshot_store.record_usage(
                        trace_attrs=trace_attrs,
                        metric_attrs=metric_attrs,
                        usage=usage,
                    )
                    self._metrics.record_usage(usage, metric_attrs)
                    otel_span.set_attribute("llm.requests", usage.requests)
                    otel_span.set_attribute("llm.input_tokens", usage.input_tokens)
                    otel_span.set_attribute("llm.output_tokens", usage.output_tokens)
                    otel_span.set_attribute("llm.cached_input_tokens", usage.cached_input_tokens)
                    otel_span.set_attribute("llm.reasoning_tokens", usage.reasoning_tokens)
                    otel_span.set_attribute("llm.total_tokens", usage.total_tokens)
                    otel_span.set_attribute("llm.estimated_cost_usd", usage.estimated_cost_usd)
                    for key, value in trace_attrs.items():
                        if key not in {"span_type", "span_id", "parent_span_id"}:
                            otel_span.set_attribute(key, value)
                if self._settings.telemetry_capture_content:
                    _set_if_present(otel_span, "llm.input", _serialize_payload(usage_input))
                    _set_if_present(otel_span, "llm.output", _serialize_payload(usage_output))
            elif self._settings.telemetry_capture_content and isinstance(span_data, FunctionSpanData):
                _set_if_present(otel_span, "llm.tool.input", _serialize_payload(span_data.input))
                _set_if_present(otel_span, "llm.tool.output", _serialize_payload(span_data.output))

            otel_span.end()
        except Exception:
            logger.warning("Telemetry span end failed.", exc_info=True)
        finally:
            self._span_parent_ids.pop(span_key, None)
            self._agent_names.pop(span_key, None)

    def shutdown(self) -> None:
        self.force_flush()

    def force_flush(self) -> None:
        for span in list(self._otel_spans.values()):
            try:
                span.end()
            except Exception:
                logger.debug("Ignoring telemetry span flush failure.", exc_info=True)
        self._otel_spans.clear()
        self._span_parent_ids.clear()
        self._agent_names.clear()
        self._root_spans.clear()
        self._trace_info.clear()

    def _parent_context(self, span: Span[Any]):
        if span.parent_id is not None:
            parent_span = self._otel_spans.get((span.trace_id, span.parent_id))
            if parent_span is not None:
                return otel_trace.set_span_in_context(parent_span)
        root_span = self._root_spans.get(span.trace_id)
        if root_span is not None:
            return otel_trace.set_span_in_context(root_span)
        return None

    def _trace_attributes(self, trace: Trace) -> dict[str, str]:
        metadata = dict(getattr(trace, "metadata", {}) or {})
        return {
            "project": self._settings.telemetry_project_name,
            "environment": self._settings.telemetry_environment,
            "instance_id": self._settings.resolved_telemetry_instance_id(),
            "workflow": trace.name or WORKFLOW_NAME,
            "run_id": getattr(trace, "group_id", None) or trace.trace_id,
            "stage": str(metadata.get("stage", "unknown")),
            "provider": "openai",
        }

    def _span_attributes(self, span: Span[Any], span_data: SpanData) -> dict[str, str]:
        attrs = {
            **self._trace_info.get(span.trace_id, {}),
            "span_type": span_data.type,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
        }
        if span.parent_id is not None:
            attrs["parent_span_id"] = span.parent_id
        agent_name = self._agent_name_for_span(span, span_data)
        if agent_name:
            attrs["agent_name"] = agent_name
        model = _model_name_for_span_data(span_data)
        if model:
            attrs["model"] = model
        return attrs

    def _usage_trace_attributes(self, span: Span[Any], span_data: SpanData) -> dict[str, str]:
        attrs = self._span_attributes(span, span_data)
        attrs.setdefault("agent_name", self._nearest_agent_name(span) or "unknown")
        attrs.setdefault("model", _model_name_for_span_data(span_data) or "unknown")
        return attrs

    def _usage_metric_attributes(self, span: Span[Any], span_data: SpanData) -> dict[str, str]:
        return self._metric_attrs_from_trace_attrs(self._usage_trace_attributes(span, span_data))

    def _metric_attrs_from_trace_attrs(self, attrs: dict[str, str]) -> dict[str, str]:
        metric_keys = (
            "project",
            "environment",
            "instance_id",
            "workflow",
            "stage",
            "agent_name",
            "model",
            "provider",
        )
        return {key: attrs[key] for key in metric_keys if key in attrs}

    def _agent_name_for_span(self, span: Span[Any], span_data: SpanData) -> str:
        if isinstance(span_data, AgentSpanData):
            return span_data.name
        return self._nearest_agent_name(span) or "unknown"

    def _nearest_agent_name(self, span: Span[Any]) -> str | None:
        parent_id = span.parent_id
        while parent_id is not None:
            key = (span.trace_id, parent_id)
            agent_name = self._agent_names.get(key)
            if agent_name:
                return agent_name
            parent_id = self._span_parent_ids.get(key)
        return None

    def _span_name(self, span_data: SpanData) -> str:
        if isinstance(span_data, AgentSpanData):
            return f"agent:{span_data.name}"
        if isinstance(span_data, GenerationSpanData):
            return f"generation:{span_data.model or 'unknown'}"
        if isinstance(span_data, FunctionSpanData):
            return f"tool:{span_data.name}"
        return span_data.type


def configure_telemetry(settings: Settings) -> TelemetryManager:
    global _ACTIVE_TELEMETRY_MANAGER
    if _ACTIVE_TELEMETRY_MANAGER is not None:
        _ACTIVE_TELEMETRY_MANAGER.close()
    snapshot_store = UsageSnapshotStore(
        db_path=settings.team_update_history_sqlite_path.parent / "usage_telemetry.sqlite3"
    )
    if not settings.telemetry_enabled:
        manager = TelemetryManager(enabled=False, export_configured=False, snapshot_store=snapshot_store)
        _ACTIVE_TELEMETRY_MANAGER = manager
        return manager

    resource = Resource.create(
        {
            "service.name": settings.telemetry_project_name,
            "service.instance.id": settings.resolved_telemetry_instance_id(),
            "deployment.environment": settings.telemetry_environment,
            "service.version": "0.1.0",
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    meter_provider: MeterProvider
    if settings.otel_exporter_otlp_endpoint:
        headers = _parse_otlp_headers(settings.otel_exporter_otlp_headers)
        trace_endpoint = _signal_endpoint(settings.otel_exporter_otlp_endpoint, "traces")
        metric_endpoint = _signal_endpoint(settings.otel_exporter_otlp_endpoint, "metrics")

        trace_exporter = OTLPSpanExporter(endpoint=trace_endpoint, headers=headers)
        metric_exporter = OTLPMetricExporter(endpoint=metric_endpoint, headers=headers)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=_DEFAULT_EXPORT_INTERVAL_MILLIS,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        export_configured = True
    else:
        logger.info("Telemetry export endpoint is unset. Running with local in-memory usage dashboard only.")
        meter_provider = MeterProvider(resource=resource)
        export_configured = False

    metrics_recorder = UsageMetricsRecorder(meter_provider)
    processor = UsageTelemetryProcessor(
        settings=settings,
        tracer_provider=tracer_provider,
        metrics_recorder=metrics_recorder,
        snapshot_store=snapshot_store,
    )
    set_trace_processors([processor])
    manager = TelemetryManager(
        enabled=True,
        export_configured=export_configured,
        processor=processor,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        metrics_recorder=metrics_recorder,
        snapshot_store=snapshot_store,
    )
    _ACTIVE_TELEMETRY_MANAGER = manager
    return manager


def get_active_telemetry_manager() -> TelemetryManager | None:
    return _ACTIVE_TELEMETRY_MANAGER


def normalize_usage_record(usage: Any, *, model: str | None = None) -> NormalizedUsageRecord | None:
    if usage is None:
        return None

    raw = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
    if not raw:
        return None

    input_tokens = _int_value(raw.get("input_tokens"))
    output_tokens = _int_value(raw.get("output_tokens"))
    cached_input_tokens = _int_value(_nested_value(raw, "input_tokens_details", "cached_tokens"))
    reasoning_tokens = _int_value(_nested_value(raw, "output_tokens_details", "reasoning_tokens"))
    total_tokens = _int_value(raw.get("total_tokens")) or (input_tokens + output_tokens)
    estimated_cost = estimate_cost_usd(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )

    return NormalizedUsageRecord(
        requests=max(1, _int_value(raw.get("requests")) or 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
    )


def estimate_cost_usd(
    *,
    provider: str = "openai",
    model: str | None,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    batch_mode: bool = False,
) -> float:
    pricing = _pricing_for_model(provider=provider, model=model, batch_mode=batch_mode)
    if pricing is None:
        return 0.0
    non_cached_input_tokens = max(0, input_tokens - cached_input_tokens)
    total_cost = (
        (Decimal(non_cached_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing["input"]))
        + (Decimal(cached_input_tokens) / Decimal(1_000_000)) * Decimal(str(pricing["cached_input"]))
        + (Decimal(output_tokens) / Decimal(1_000_000)) * Decimal(str(pricing["output"]))
    )
    return float(total_cost.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def _pricing_key(model: str | None) -> str | None:
    if not model:
        return None
    for prefix, exact in _MODEL_PRICE_FALLBACKS.items():
        if model.startswith(prefix):
            return exact
    for key in sorted(_OPENAI_PRICE_MAP, key=len, reverse=True):
        if model.startswith(key):
            return key
    return None


def _pricing_for_model(
    *,
    provider: str,
    model: str | None,
    batch_mode: bool,
) -> dict[str, float] | None:
    if provider == "openai":
        price_key = _pricing_key(model)
        if price_key is None:
            return None
        return _OPENAI_PRICE_MAP[price_key]
    if provider == "google":
        if not model:
            return None
        for key in sorted(_GOOGLE_BATCH_PRICE_MAP, key=len, reverse=True):
            if model.startswith(key):
                return _GOOGLE_BATCH_PRICE_MAP[key]
        return None
    return None


def _signal_endpoint(base_endpoint: str, signal: str) -> str:
    trimmed = base_endpoint.rstrip("/")
    if trimmed.endswith(f"/v1/{signal}"):
        return trimmed
    parsed = urlsplit(trimmed)
    if parsed.path.endswith("/v1/traces") or parsed.path.endswith("/v1/metrics"):
        return trimmed.rsplit("/", 1)[0] + f"/{signal}"
    return f"{trimmed}/v1/{signal}"


def _parse_otlp_headers(raw_headers: str | None) -> dict[str, str] | None:
    if raw_headers is None or not raw_headers.strip():
        return None
    headers: dict[str, str] = {}
    for item in raw_headers.split(","):
        chunk = item.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers or None


def _nested_value(container: dict[str, Any], key: str, nested_key: str) -> Any:
    nested = container.get(key) or {}
    if isinstance(nested, dict):
        return nested.get(nested_key)
    return getattr(nested, nested_key, None)


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _model_name_for_span_data(span_data: SpanData) -> str | None:
    if isinstance(span_data, GenerationSpanData):
        return span_data.model
    if isinstance(span_data, ResponseSpanData):
        response = span_data.response
        return getattr(response, "model", None)
    if isinstance(span_data, AgentSpanData):
        return None
    return getattr(span_data, "model", None)


def _usage_payload_for_span_data(
    span_data: SpanData,
) -> tuple[Any | None, str | None, Any | None, Any | None]:
    if isinstance(span_data, GenerationSpanData):
        return span_data.usage, span_data.model, span_data.input, span_data.output
    if isinstance(span_data, ResponseSpanData):
        response = span_data.response
        usage = getattr(response, "usage", None)
        output = getattr(response, "output", None)
        model = getattr(response, "model", None)
        return usage, model, span_data.input, output
    return None, None, None, None


def _serialize_payload(payload: Any) -> str | None:
    if payload in (None, "", [], {}):
        return None
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(payload)


def _set_if_present(span: Any, key: str, value: str | None) -> None:
    if value:
        span.set_attribute(key, value)
