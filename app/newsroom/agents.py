# This file defines the agents used in the newsroom workflow, including their configurations and tools.

from __future__ import annotations

from agents import Agent
from agents.tool import FunctionTool

from app.config import Settings
from app.newsroom.model import build_model_settings
from app.newsroom.prompts import get_prompt
from app.schemas import (
    ArticleDigestAgentResult,
    HourlyPlaylist,
    HourlyPlaylistSelection,
    HourlyPlaylistScriptsBatchAgentResult,
    RadioRundownDraft,
    RadioStoryScriptDraft,
    TeamAnalysisResult,
    TeamUpdateBatchAgentResult,
    TeamUpdatePackage,
)


def build_article_data_agent(
    settings: Settings,
    *,
    article_lookup_tool: FunctionTool,
) -> Agent:
    return Agent(
        name="Article Data Agent",
        instructions=get_prompt("article_data_agent"),
        model=settings.agent_model("article_data_agent"),
        model_settings=build_model_settings(
            settings,
            tool_choice="required",
            parallel_tool_calls=False,
        ),
        tools=[article_lookup_tool],
        output_type=ArticleDigestAgentResult,
    )


def build_team_news_agent(
    settings: Settings,
    *,
    article_data_tool: FunctionTool,
) -> Agent:
    return Agent(
        name="Team News Agent",
        instructions=get_prompt("team_news_agent"),
        model=settings.agent_model("team_news_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[article_data_tool],
        output_type=TeamAnalysisResult,
    )


def build_rundown_orchestrator_agent(
    settings: Settings,
    *,
    team_news_tool: FunctionTool,
) -> Agent:
    return Agent(
        name="Rundown Orchestrator Agent",
        instructions=get_prompt("rundown_orchestrator_agent"),
        model=settings.agent_model("rundown_orchestrator_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[team_news_tool],
        output_type=RadioRundownDraft,
    )


def build_team_update_agent(
    settings: Settings,
    *,
    article_data_tool: FunctionTool,
) -> Agent:
    return Agent(
        name="Team Update Agent",
        instructions=get_prompt("team_update_agent"),
        model=settings.agent_model("team_update_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[article_data_tool],
        output_type=TeamUpdatePackage,
    )


def build_team_update_batch_agent(
    settings: Settings,
    *,
    team_update_tool: FunctionTool,
) -> Agent:
    return Agent(
        name="Team Update Batch Agent",
        instructions=get_prompt("team_update_batch_agent"),
        model=settings.agent_model("team_update_batch_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[team_update_tool],
        output_type=TeamUpdateBatchAgentResult,
    )


def build_hourly_playlist_orchestrator_agent(
    settings: Settings,
) -> Agent:
    return Agent(
        name="Hourly Playlist Orchestrator Agent",
        instructions=get_prompt("hourly_playlist_orchestrator_agent"),
        model=settings.agent_model("hourly_playlist_orchestrator_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=False),
        tools=[],
        output_type=HourlyPlaylistSelection,
    )


def build_radio_script_writer_agent(
    settings: Settings,
    *,
    article_data_tool: FunctionTool,
    prompt_name: str = "radio_script_writer_agent",
    agent_name: str = "Radio Script Writer Agent",
) -> Agent:
    return Agent(
        name=agent_name,
        instructions=get_prompt(prompt_name),
        model=settings.agent_model("radio_script_writer_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[article_data_tool],
        output_type=RadioStoryScriptDraft,
    )


def build_hourly_script_batch_agent(
    settings: Settings,
    *,
    radio_script_tool: FunctionTool,
    prompt_name: str = "hourly_script_batch_agent",
    agent_name: str = "Hourly Script Batch Agent",
) -> Agent:
    return Agent(
        name=agent_name,
        instructions=get_prompt(prompt_name),
        model=settings.agent_model("hourly_script_batch_agent"),
        model_settings=build_model_settings(settings, parallel_tool_calls=True),
        tools=[radio_script_tool],
        output_type=HourlyPlaylistScriptsBatchAgentResult,
    )
