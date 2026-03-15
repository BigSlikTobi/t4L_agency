from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from agents.tracing import get_trace_provider
from agents.tracing.span_data import AgentSpanData, GenerationSpanData, ResponseSpanData
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import Settings
from app.telemetry import (
    TelemetryManager,
    UsageTelemetryProcessor,
    UsageMetricsRecorder,
    UsageSnapshotStore,
    configure_telemetry,
    estimate_cost_usd,
    normalize_usage_record,
)


class FakeMetricsRecorder:
    def __init__(self) -> None:
        self.usage_calls: list[tuple[object, dict[str, str]]] = []
        self.active_run_deltas: list[tuple[int, dict[str, str]]] = []

    def record_usage(self, usage, attrs: dict[str, str]) -> None:
        self.usage_calls.append((usage, attrs))

    def increment_active_runs(self, attrs: dict[str, str]) -> None:
        self.active_run_deltas.append((1, attrs))

    def decrement_active_runs(self, attrs: dict[str, str]) -> None:
        self.active_run_deltas.append((-1, attrs))


def build_settings(**overrides) -> Settings:
    payload = {
        "_env_file": None,
        "openai_api_key": "sk-test",
        "supabase_news_feed_url": "https://example.com/feed",
        "supabase_article_lookup_url": "https://example.com/article-lookup",
        "supabase_function_auth_token": "supabase-token",
        "telemetry_enabled": False,
    }
    payload.update(overrides)
    return Settings(**payload)


def test_normalize_usage_record_fills_missing_details() -> None:
    usage = normalize_usage_record(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        },
        model="gpt-5-mini-2025-08-07",
    )

    assert usage is not None
    assert usage.cached_input_tokens == 0
    assert usage.reasoning_tokens == 0
    assert usage.total_tokens == 150
    assert usage.estimated_cost_usd > 0


def test_usage_processor_attributes_nested_generation_to_parent_agent() -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    metrics = FakeMetricsRecorder()
    processor = UsageTelemetryProcessor(
        settings=build_settings(telemetry_enabled=True),
        tracer_provider=tracer_provider,
        metrics_recorder=metrics,
        snapshot_store=UsageSnapshotStore(),
    )

    trace = SimpleNamespace(
        trace_id="trace-1",
        name="NFL Radio Agency",
        group_id="run-1",
        metadata={"stage": "team_update_batch_agent"},
    )
    processor.on_trace_start(trace)

    agent_span = SimpleNamespace(
        trace_id="trace-1",
        span_id="agent-1",
        parent_id=None,
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=AgentSpanData(name="Team Update Batch Agent"),
    )
    generation_span = SimpleNamespace(
        trace_id="trace-1",
        span_id="gen-1",
        parent_id="agent-1",
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=GenerationSpanData(
            model="gpt-5-mini-2025-08-07",
            usage={
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        ),
    )

    processor.on_span_start(agent_span)
    processor.on_span_start(generation_span)
    processor.on_span_end(generation_span)
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)

    assert metrics.usage_calls
    usage, attrs = metrics.usage_calls[0]
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 10
    assert usage.reasoning_tokens == 5
    assert attrs["agent_name"] == "Team Update Batch Agent"
    assert attrs["stage"] == "team_update_batch_agent"
    assert attrs["model"] == "gpt-5-mini-2025-08-07"
    assert "run_id" not in attrs
    assert "trace_id" not in attrs

    span_attributes = [span.attributes for span in exporter.get_finished_spans()]
    generation_attributes = next(attrs for attrs in span_attributes if attrs.get("span_type") == "generation")
    assert generation_attributes["llm.input_tokens"] == 100
    assert generation_attributes["agent_name"] == "Team Update Batch Agent"
    assert generation_attributes["run_id"] == "run-1"


def test_usage_processor_records_response_span_usage() -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    metrics = FakeMetricsRecorder()
    processor = UsageTelemetryProcessor(
        settings=build_settings(telemetry_enabled=True),
        tracer_provider=tracer_provider,
        metrics_recorder=metrics,
        snapshot_store=UsageSnapshotStore(),
    )

    trace = SimpleNamespace(
        trace_id="trace-2",
        name="NFL Radio Agency",
        group_id="run-2",
        metadata={"stage": "team_update_batch_agent"},
    )
    processor.on_trace_start(trace)

    agent_span = SimpleNamespace(
        trace_id="trace-2",
        span_id="agent-1",
        parent_id=None,
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=AgentSpanData(name="Team Update Batch Agent"),
    )
    response_span = SimpleNamespace(
        trace_id="trace-2",
        span_id="resp-1",
        parent_id="agent-1",
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=ResponseSpanData(
            response=SimpleNamespace(
                model="gpt-5-mini-2025-08-07",
                usage=SimpleNamespace(
                    input_tokens=88,
                    output_tokens=22,
                    total_tokens=110,
                    input_tokens_details=SimpleNamespace(cached_tokens=8),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                ),
                output=[{"type": "message"}],
            ),
            input="hello",
        ),
    )

    processor.on_span_start(agent_span)
    processor.on_span_start(response_span)
    processor.on_span_end(response_span)
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)

    assert metrics.usage_calls
    usage, attrs = metrics.usage_calls[0]
    assert usage.input_tokens == 88
    assert usage.output_tokens == 22
    assert usage.cached_input_tokens == 8
    assert usage.reasoning_tokens == 3
    assert attrs["agent_name"] == "Team Update Batch Agent"
    assert attrs["model"] == "gpt-5-mini-2025-08-07"
    assert "run_id" not in attrs


def test_configure_telemetry_returns_disabled_manager_when_flag_off() -> None:
    manager = configure_telemetry(build_settings())

    assert isinstance(manager, TelemetryManager)
    assert manager.enabled is False


def test_configure_telemetry_supports_local_dashboard_without_exporter() -> None:
    manager = configure_telemetry(build_settings(telemetry_enabled=True))

    assert manager.enabled is True
    assert manager.snapshot_store is not None
    snapshot = manager.dashboard_snapshot()
    assert snapshot["telemetry_enabled"] is True
    assert snapshot["export_configured"] is False
    manager.close()


def test_configure_telemetry_initializes_exporters(monkeypatch) -> None:
    created = {"metric": 0, "trace": 0}

    class FakeMetricExporter:
        _preferred_temporality = None
        _preferred_aggregation = None

        def __init__(self, *args, **kwargs):
            created["metric"] += 1

        def export(self, *args, **kwargs):
            return None

        def force_flush(self, *args, **kwargs):
            return True

        def shutdown(self, *args, **kwargs):
            return True

    class FakeTraceExporter:
        def __init__(self, *args, **kwargs):
            created["trace"] += 1

        def export(self, *args, **kwargs):
            return None

        def shutdown(self, *args, **kwargs):
            return True

    monkeypatch.setattr("app.telemetry.OTLPMetricExporter", FakeMetricExporter)
    monkeypatch.setattr("app.telemetry.OTLPSpanExporter", FakeTraceExporter)

    manager = configure_telemetry(
        build_settings(
            telemetry_enabled=True,
            otel_exporter_otlp_endpoint="http://localhost:4318",
        )
    )

    assert manager.enabled is True
    assert created == {"metric": 1, "trace": 1}
    manager.close()


def test_configure_telemetry_replaces_existing_trace_processors() -> None:
    first = configure_telemetry(build_settings(telemetry_enabled=True))
    second = configure_telemetry(build_settings())

    processors = get_trace_provider()._multi_processor._processors

    assert first is not second
    assert first.tracer_provider is not None
    assert second.enabled is False
    assert processors == ()
    second.close()


def test_usage_processor_snapshot_tracks_recent_events_and_active_runs() -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider()
    snapshot_store = UsageSnapshotStore()
    processor = UsageTelemetryProcessor(
        settings=build_settings(telemetry_enabled=True),
        tracer_provider=tracer_provider,
        metrics_recorder=UsageMetricsRecorder(meter_provider),
        snapshot_store=snapshot_store,
    )

    trace = SimpleNamespace(
        trace_id="trace-snap",
        name="NFL Radio Agency",
        group_id="run-snap",
        metadata={"stage": "team_update_batch_agent"},
    )
    agent_span = SimpleNamespace(
        trace_id="trace-snap",
        span_id="agent-snap",
        parent_id=None,
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=AgentSpanData(name="Team Update Batch Agent"),
    )
    generation_span = SimpleNamespace(
        trace_id="trace-snap",
        span_id="gen-snap",
        parent_id="agent-snap",
        trace_metadata={"stage": "team_update_batch_agent"},
        error=None,
        span_data=GenerationSpanData(
            model="gpt-5-mini-2025-08-07",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    )

    processor.on_trace_start(trace)
    snapshot_during_run = snapshot_store.snapshot(enabled=True, export_configured=False)
    assert len(snapshot_during_run["active_runs"]) == 1

    processor.on_span_start(agent_span)
    processor.on_span_start(generation_span)
    processor.on_span_end(generation_span)
    processor.on_span_end(agent_span)
    processor.on_trace_end(trace)

    snapshot = snapshot_store.snapshot(enabled=True, export_configured=False)
    assert snapshot["totals"]["total_tokens"] == 15
    assert snapshot["usage_by_agent"][0]["key"] == "Team Update Batch Agent"
    assert snapshot["recent_events"][0]["run_id"] == "run-snap"
    assert snapshot["active_runs"] == []


def test_usage_snapshot_store_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "usage_telemetry.sqlite3"
    writer = UsageSnapshotStore(db_path=db_path)
    reader = UsageSnapshotStore(db_path=db_path)

    trace_attrs = {
        "workflow": "NFL Radio Agency",
        "run_id": "run-shared",
        "stage": "team_update_batch_agent",
        "provider": "openai",
    }
    metric_attrs = {
        "agent_name": "Team Update Batch Agent",
        "model": "gpt-5-mini-2025-08-07",
        "provider": "openai",
    }

    writer.record_trace_start(trace_attrs)
    snapshot_during_run = reader.snapshot(enabled=True, export_configured=False)
    assert len(snapshot_during_run["active_runs"]) == 1

    writer.record_usage(
        trace_attrs=trace_attrs,
        metric_attrs=metric_attrs,
        usage=normalize_usage_record(
            {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            model="gpt-5-mini-2025-08-07",
        ),
    )
    writer.record_trace_end(trace_attrs)

    snapshot = reader.snapshot(enabled=True, export_configured=False)
    assert snapshot["totals"]["total_tokens"] == 20
    assert snapshot["usage_by_model"][0]["key"] == "gpt-5-mini-2025-08-07"
    assert snapshot["recent_events"][0]["run_id"] == "run-shared"
    assert snapshot["active_runs"] == []


def test_usage_snapshot_store_filters_by_run_id_and_time(tmp_path: Path) -> None:
    db_path = tmp_path / "usage_telemetry.sqlite3"
    store = UsageSnapshotStore(db_path=db_path)

    store.record_usage(
        trace_attrs={
            "workflow": "NFL Radio Agency",
            "run_id": "run-a",
            "stage": "team_update_batch_agent",
            "provider": "openai",
        },
        metric_attrs={
            "agent_name": "Batch Agent",
            "model": "gpt-5-mini-2025-08-07",
            "provider": "openai",
        },
        usage=normalize_usage_record(
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            model="gpt-5-mini-2025-08-07",
        ),
    )
    cutoff = datetime.now(UTC)
    store.record_usage(
        trace_attrs={
            "workflow": "NFL Radio Agency",
            "run_id": "run-b",
            "stage": "hourly_script_batch_agent",
            "provider": "openai",
        },
        metric_attrs={
            "agent_name": "Script Agent",
            "model": "gpt-5.1-2025-11-13",
            "provider": "openai",
        },
        usage=normalize_usage_record(
            {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            model="gpt-5.1-2025-11-13",
        ),
    )

    snapshot_by_run = store.snapshot(enabled=True, export_configured=False, run_id="run-b")
    assert snapshot_by_run["totals"]["total_tokens"] == 30
    assert snapshot_by_run["recent_events"][0]["run_id"] == "run-b"
    assert snapshot_by_run["applied_filters"]["run_id"] == "run-b"

    snapshot_by_time = store.snapshot(enabled=True, export_configured=False, from_time=cutoff)
    assert snapshot_by_time["totals"]["total_tokens"] == 30
    assert snapshot_by_time["recent_events"][0]["run_id"] == "run-b"
    assert snapshot_by_time["applied_filters"]["from"] is not None


def test_estimate_cost_uses_family_pricing() -> None:
    cost = estimate_cost_usd(
        model="gpt-5.1-2025-11-13",
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=1_000_000,
    )

    assert cost == 11.25


def test_estimate_cost_supports_google_tts_batch_pricing() -> None:
    cost = estimate_cost_usd(
        provider="google",
        model="gemini-2.5-pro-preview-tts",
        input_tokens=1_000_000,
        cached_input_tokens=0,
        output_tokens=1_000_000,
        batch_mode=True,
    )

    assert cost == 10.5


def test_telemetry_manager_records_external_usage(tmp_path: Path) -> None:
    manager = configure_telemetry(
        build_settings(
            telemetry_enabled=True,
            team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3",
        )
    )

    manager.record_external_usage(
        workflow="NFL Radio Agency",
        run_id="script-run-1",
        stage="gemini_tts_batch",
        agent_name="Gemini TTS Batch",
        provider="google",
        model="gemini-2.5-pro-preview-tts",
        requests=1,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        batch_mode=True,
    )

    snapshot = manager.dashboard_snapshot(run_id="script-run-1")
    assert snapshot["totals"]["total_tokens"] == 150
    assert snapshot["usage_by_agent"][0]["key"] == "Gemini TTS Batch"
    assert snapshot["usage_by_model"][0]["key"] == "gemini-2.5-pro-preview-tts"
    assert snapshot["recent_events"][0]["provider"] == "google"
    manager.close()


def test_telemetry_manager_builds_run_comparison(tmp_path: Path) -> None:
    manager = configure_telemetry(
        build_settings(
            telemetry_enabled=True,
            team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3",
        )
    )

    manager.record_external_usage(
        workflow="NFL Radio Agency",
        run_id="run-a",
        stage="team_update_batch_agent",
        agent_name="Batch Agent",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        requests=1,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    manager.record_external_usage(
        workflow="NFL Radio Agency",
        run_id="run-b",
        stage="team_update_batch_agent",
        agent_name="Batch Agent",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        requests=2,
        input_tokens=140,
        output_tokens=30,
        total_tokens=170,
    )

    comparison = manager.dashboard_comparison(left_run_id="run-a", right_run_id="run-b")
    assert comparison["left"]["applied_filters"]["run_id"] == "run-a"
    assert comparison["right"]["applied_filters"]["run_id"] == "run-b"
    assert comparison["delta"]["requests"] == 1
    assert comparison["delta"]["total_tokens"] == 50
    manager.close()


def test_telemetry_manager_lists_historical_runs_by_finish_time(tmp_path: Path) -> None:
    manager = configure_telemetry(
        build_settings(
            telemetry_enabled=True,
            team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3",
        )
    )

    manager.record_external_usage(
        workflow="NFL Radio Agency",
        run_id="run-a",
        stage="team_update_batch_agent",
        agent_name="Batch Agent",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        requests=1,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )
    manager.record_external_usage(
        workflow="NFL Radio Agency",
        run_id="run-b",
        stage="team_update_batch_agent",
        agent_name="Batch Agent",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        requests=1,
        input_tokens=120,
        output_tokens=25,
        total_tokens=145,
    )

    runs = manager.dashboard_historical_runs(limit=10)
    assert [item["run_id"] for item in runs["runs"]] == ["run-b", "run-a"]
    assert runs["runs"][0]["finished_at"] >= runs["runs"][1]["finished_at"]
    manager.close()
