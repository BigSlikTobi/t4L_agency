# This file defines the tools used by the agents in the newsroom workflow,
# including their configurations and how they process inputs and outputs.

from __future__ import annotations

from agents import Agent, Runner, function_tool
from agents.tool import FunctionTool

from app.adapters import SupabaseArticleLookupAdapter
from app.constants import DEFAULT_SCRIPT_LANGUAGE, ScriptPersona
from app.newsroom.helpers import build_radio_script_input, coerce_output, extract_json_output
from app.schemas import (
    ArticleContentLookupToolResponse,
    RadioStoryScriptDraft,
    RadioStoryScriptToolInput,
    RadioStoryScriptWriterInput,
    SourceArticleRef,
    StoryResearchToolInput,
    TeamNewsToolInput,
    TeamUpdateTeamInput,
)


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
    return agent.as_tool(
        tool_name="digest_article_data",
        tool_description=(
            "Digest one exact NFL article URL using stored Supabase article content and return "
            "a structured summary, key facts, and confidence."
        ),
        parameters=StoryResearchToolInput,
        custom_output_extractor=extract_json_output,
        max_turns=3,
    )


def build_team_news_tool(agent: Agent) -> FunctionTool:
    return agent.as_tool(
        tool_name="analyze_team_news",
        tool_description=(
            "Analyze one NFL team's last-24-hours stories and return scored candidates for every team "
            "story."
        ),
        parameters=TeamNewsToolInput,
        custom_output_extractor=extract_json_output,
        max_turns=16,
    )


def build_team_update_tool(agent: Agent) -> FunctionTool:
    return agent.as_tool(
        tool_name="build_team_update_package",
        tool_description=(
            "Build one 180-second team update narrative package for a single team from its "
            "pre-filtered candidate stories."
        ),
        parameters=TeamUpdateTeamInput,
        custom_output_extractor=extract_json_output,
        max_turns=8,
    )


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
        )
        result = await Runner.run(
            agent,
            build_radio_script_input(writer_input),
            max_turns=8,
        )
        draft = coerce_output(result.final_output, RadioStoryScriptDraft)
        return draft.model_dump(mode="json")

    return build_radio_story_script
