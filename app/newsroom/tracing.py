# This file defines the tracing configuration for the newsroom workflow, 
# allowing for structured logging and monitoring of the different stages of the process.

from __future__ import annotations

from agents import RunConfig

from app.constants import WORKFLOW_NAME


def build_run_config(
    run_id: str,
    *,
    stage: str,
    metadata: dict[str, str] | None = None,
) -> RunConfig:
    return RunConfig(
        workflow_name=WORKFLOW_NAME,
        group_id=run_id,
        trace_metadata={"stage": stage, **(metadata or {})},
    )
