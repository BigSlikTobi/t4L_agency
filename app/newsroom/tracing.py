# This file defines the tracing configuration for the newsroom workflow, 
# allowing for structured logging and monitoring of the different stages of the process.

from __future__ import annotations

from agents import RunConfig


def build_run_config(
    run_id: str,
    *,
    stage: str,
    metadata: dict[str, str] | None = None,
) -> RunConfig:
    return RunConfig(
        workflow_name="NFL Radio Agency",
        group_id=run_id,
        trace_metadata={"stage": stage, **(metadata or {})},
    )
