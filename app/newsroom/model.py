# This file defines the model settings for the agents used in the newsroom workflow, 
# allowing for configuration of parameters such as temperature and max tokens based on the application's settings.

from __future__ import annotations

from agents import ModelSettings

from app.config import Settings


def build_model_settings(
    settings: Settings,
    *,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
) -> ModelSettings:
    return ModelSettings(
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
