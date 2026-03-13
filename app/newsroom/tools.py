# This file defines the tools used by the agents in the newsroom workflow,
# including their configurations and how they process inputs and outputs.

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TypeVar

from agents import Agent, Runner, function_tool
from agents.tool import FunctionTool
from agents.tool_context import ToolContext

from app.adapters import SupabaseArticleLookupAdapter
from app.constants import DEFAULT_SCRIPT_LANGUAGE, ScriptPersona
from app.newsroom.helpers import build_radio_script_input, coerce_output
from app.schemas import (
    ArticleContentLookupToolResponse,
    ArticleDigestAgentResult,
    RadioStoryScriptDraft,
    RadioNarrativeContext,
    RadioStoryScriptToolInput,
    RadioStoryScriptWriterInput,
    SourceArticleRef,
    StoryResearchToolInput,
    TeamAnalysisResult,
    TeamNewsStoryInput,
    TeamNewsToolInput,
    TeamUpdateCandidate,
    TeamUpdatePackage,
    TeamUpdateTeamInput,
)

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel")

_DIGEST_CACHE_MAX_SIZE = 512


class _BoundedCache(OrderedDict):
    """LRU dict that evicts the oldest entry when full."""

    def __init__(self, maxsize: int = _DIGEST_CACHE_MAX_SIZE) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key: str, value: dict) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            self.popitem(last=False)


async def _run_nested_agent(
    tool_context: ToolContext,
    *,
    agent: Agent,
    agent_input: str,
    output_schema: type[TModel],
    max_turns: int,
) -> dict:
    result = await Runner.run(
        agent,
        agent_input,
        context=tool_context.context,
        run_config=tool_context.run_config,
        max_turns=max_turns,
        auto_previous_response_id=True,
    )
    output = coerce_output(result.final_output, output_schema)
    return output.model_dump(mode="json")


def build_article_lookup_tool(adapter: SupabaseArticleLookupAdapter) -> FunctionTool:
    @function_tool(
        name_override="lookup_article_content",
        description_override=(
            "Look up one exact article URL in Supabase and return the stored article content "
            "and metadata."
        ),
        strict_mode=True,
    )
    async def lookup_article_content(url: str) -> dict:
        """Look up stored article content for one exact NFL article URL."""

        response = await adapter.lookup_article(url)
        return ArticleContentLookupToolResponse.model_validate(response).model_dump(mode="json")

    return lookup_article_content


def build_article_data_tool(agent: Agent) -> FunctionTool:
    _digest_cache: _BoundedCache = _BoundedCache()

    @function_tool(
        name_override="digest_article_data",
        description_override=(
            "Digest one exact NFL article URL using stored Supabase article content and return "
            "a structured summary, key facts, and confidence."
        ),
        strict_mode=True,
    )
    async def digest_article_data(
        tool_context: ToolContext,
        story_id: str,
        url: str,
        title: str,
        source_name: str,
        category: str | None = None,
        team_mentions: list[str] | None = None,
    ) -> dict:
        cached = _digest_cache.get(url)
        if cached is not None:
            logger.debug("Article digest cache hit for %s", url)
            return cached

        tool_input = StoryResearchToolInput(
            story_id=story_id,
            url=url,
            title=title,
            source_name=source_name,
            category=category,
            team_mentions=team_mentions or [],
        )
        result = await _run_nested_agent(
            tool_context,
            agent=agent,
            agent_input=tool_input.model_dump_json(),
            output_schema=ArticleDigestAgentResult,
            max_turns=3,
        )
        _digest_cache[url] = result
        return result

    return digest_article_data


def build_team_news_tool(agent: Agent) -> FunctionTool:
    @function_tool(
        name_override="analyze_team_news",
        description_override=(
            "Analyze one NFL team's last-24-hours stories and return scored candidates for every team "
            "story."
        ),
        strict_mode=True,
    )
    async def analyze_team_news(
        tool_context: ToolContext,
        team_code: str,
        team_name: str,
        stories: list[TeamNewsStoryInput],
    ) -> dict:
        tool_input = TeamNewsToolInput(
            team_code=team_code,
            team_name=team_name,
            stories=stories,
        )
        return await _run_nested_agent(
            tool_context,
            agent=agent,
            agent_input=tool_input.model_dump_json(),
            output_schema=TeamAnalysisResult,
            max_turns=16,
        )

    return analyze_team_news


def build_team_update_tool(agent: Agent) -> FunctionTool:
    @function_tool(
        name_override="build_team_update_package",
        description_override=(
            "Build one 180-second team update narrative package for a single team from its "
            "pre-filtered candidate stories."
        ),
        strict_mode=True,
    )
    async def build_team_update_package(
        tool_context: ToolContext,
        team_code: str,
        team_name: str,
        lookback_minutes: int,
        candidate_stories: list[TeamUpdateCandidate],
    ) -> dict:
        tool_input = TeamUpdateTeamInput(
            team_code=team_code,
            team_name=team_name,
            lookback_minutes=lookback_minutes,
            candidate_stories=candidate_stories,
        )
        return await _run_nested_agent(
            tool_context,
            agent=agent,
            agent_input=tool_input.model_dump_json(),
            output_schema=TeamUpdatePackage,
            max_turns=8,
        )

    return build_team_update_package


def build_radio_story_script_tool(
    agent: Agent,
    *,
    personas: list[ScriptPersona],
    language: str = DEFAULT_SCRIPT_LANGUAGE,
) -> FunctionTool:
    persona_by_name = {
        str(persona["name"]): persona
        for persona in personas
    }

    @function_tool(
        name_override="build_radio_story_script",
        description_override=(
            "Write one single-anchor radio script for one selected hourly playlist story after "
            "choosing a persona by name."
        ),
        strict_mode=True,
    )
    async def build_radio_story_script(
        team: str,
        playlist_rank: int,
        headline: str,
        continuity: str,
        production_reason: str,
        story_synopsis: str,
        source_articles: list[SourceArticleRef],
        persona_name: str,
        narrative_context: RadioNarrativeContext,
    ) -> dict:
        tool_input = RadioStoryScriptToolInput(
            team=team,
            playlist_rank=playlist_rank,
            headline=headline,
            continuity=continuity,
            production_reason=production_reason,
            story_synopsis=story_synopsis,
            source_articles=source_articles,
            persona_name=persona_name,
            narrative_context=narrative_context,
        )
        persona = persona_by_name.get(tool_input.persona_name)
        if persona is None:
            raise ValueError(f"Unknown persona {tool_input.persona_name!r}")

        writer_input = RadioStoryScriptWriterInput(
            team=tool_input.team,
            playlist_rank=tool_input.playlist_rank,
            language=language,
            headline=tool_input.headline,
            continuity=tool_input.continuity,
            production_reason=tool_input.production_reason,
            story_synopsis=tool_input.story_synopsis,
            persona_name=tool_input.persona_name,
            persona_backstory=str(persona["backstory"]),
            persona_specialty=str(persona["specialty"]),
            voice_name=str(persona["voice_name"]),
            dialect=str(persona["dialect"]),
            source_articles=tool_input.source_articles,
            narrative_context=tool_input.narrative_context,
        )
        result = await Runner.run(
            agent,
            build_radio_script_input(writer_input),
            max_turns=8,
            auto_previous_response_id=True,
        )
        draft = coerce_output(result.final_output, RadioStoryScriptDraft)
        return draft.model_dump(mode="json")

    return build_radio_story_script
