# This file defines the tools used by the agents in the newsroom workflow,
# including their configurations and how they process inputs and outputs.

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import TypeVar

from agents import Agent, RunConfig, Runner, function_tool
from agents.tool import FunctionTool
from agents.tool_context import ToolContext
from openai import AsyncOpenAI

from app.adapters import SupabaseArticleLookupAdapter
from app.config import Settings
from app.constants import DEFAULT_SCRIPT_LANGUAGE, ScriptPersona, WORKFLOW_NAME
from app.newsroom.helpers import (
    build_radio_script_input,
    coerce_output,
    coerce_response_output,
    truncate_article_content,
)
from app.newsroom.prompts import get_prompt
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
    TeamUpdateAgentResult,
    TeamUpdateCandidate,
    TeamUpdateGateResult,
    TeamUpdateTeamInput,
)

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel")

_DIGEST_CACHE_MAX_SIZE = 512
_ARTICLE_DIGEST_TOKEN_BUDGETS: tuple[int, ...] = (800, 1200)
_TEAM_UPDATE_GATE_TOKEN_BUDGETS: tuple[int, ...] = (400, 800)


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


def _record_direct_api_usage(response: object, *, model: str, agent_name: str, run_id: str = "", stage: str = "") -> None:
    """Record usage from a direct OpenAI API call into the telemetry system."""
    from app.telemetry import get_active_telemetry_manager, normalize_usage_record

    manager = get_active_telemetry_manager()
    if manager is None or manager.snapshot_store is None:
        return
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return
    normalized = normalize_usage_record(usage_obj, model=model)
    if normalized is None:
        return
    manager.snapshot_store.record_usage(
        trace_attrs={"workflow": WORKFLOW_NAME, "run_id": run_id, "stage": stage},
        metric_attrs={"agent_name": agent_name, "model": model, "provider": "openai"},
        usage=normalized,
    )
    if manager.metrics_recorder is not None:
        manager.metrics_recorder.record_usage(
            normalized,
            {"agent_name": agent_name, "model": model, "provider": "openai"},
        )


async def _parse_structured_response(
    client: AsyncOpenAI,
    *,
    model: str,
    instructions: str,
    user_message: str,
    output_schema: type[TModel],
    token_budgets: tuple[int, ...],
    agent_name: str = "unknown",
    run_id: str = "",
    stage: str = "",
) -> TModel:
    last_error: ValueError | None = None

    for attempt_index, max_output_tokens in enumerate(token_budgets, start=1):
        response = await client.responses.parse(
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": user_message}],
            text_format=output_schema,
            reasoning={"effort": "minimal"},
            max_output_tokens=max_output_tokens,
            store=True,
        )
        _record_direct_api_usage(response, model=model, agent_name=agent_name, run_id=run_id, stage=stage)
        try:
            return coerce_response_output(response, output_schema)
        except ValueError as exc:
            last_error = exc
            incomplete_details = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete_details, "reason", None)
            if reason != "max_output_tokens" or attempt_index == len(token_budgets):
                raise
            next_budget = token_budgets[attempt_index]
            logger.warning(
                "Structured response for %s hit max_output_tokens=%s with model %s; retrying with %s",
                output_schema.__name__,
                max_output_tokens,
                model,
                next_budget,
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Structured response parsing failed unexpectedly for {output_schema.__name__}")


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


def build_article_data_tool(
    adapter: SupabaseArticleLookupAdapter,
    settings: Settings,
    *,
    prefetched_articles: dict[str, ArticleContentLookupToolResponse] | None = None,
) -> FunctionTool:
    _digest_cache: _BoundedCache = _BoundedCache()
    _prefetched = prefetched_articles if prefetched_articles is not None else {}
    _client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    _model = settings.agent_model("article_data_agent")
    _system_prompt = get_prompt("article_data_direct")

    @function_tool(
        name_override="digest_article_data",
        description_override=(
            "Digest one exact NFL article URL using stored Supabase article content and return "
            "a structured summary, key facts, and confidence."
        ),
        strict_mode=True,
    )
    async def digest_article_data(
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

        # Look up article: prefer prefetched, then fetch from adapter
        article_response = _prefetched.get(url)
        if article_response is None:
            raw = await adapter.lookup_article(url)
            article_response = ArticleContentLookupToolResponse.model_validate(raw)

        # Not found → return low-confidence result immediately (zero LLM tokens)
        if not article_response.found or article_response.article is None:
            result = ArticleDigestAgentResult(
                summary=f"Article not found in storage for {url}.",
                key_facts=[],
                confidence=0.1,
                content_status="missing",
            ).model_dump(mode="json")
            _digest_cache[url] = result
            return result

        article = article_response.article
        truncated_content = truncate_article_content(article.content)

        # Build user message with story metadata + truncated article content
        user_payload = {
            "story_id": story_id,
            "url": url,
            "title": title,
            "source_name": source_name,
            "category": category,
            "team_mentions": team_mentions or [],
            "article": {
                "header": article.header,
                "description": article.description,
                "content": truncated_content,
                "quotes": article.quotes,
            },
        }
        user_message = json.dumps(user_payload, separators=(",", ":"))

        # Direct API call — no agent overhead
        output = await _parse_structured_response(
            _client,
            model=_model,
            instructions=_system_prompt,
            user_message=user_message,
            output_schema=ArticleDigestAgentResult,
            token_budgets=_ARTICLE_DIGEST_TOKEN_BUDGETS,
            agent_name="Article Digest (direct)",
        )
        result = output.model_dump(mode="json")
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


def build_team_update_tool(agent: Agent, settings: Settings) -> tuple[FunctionTool, dict[str, dict]]:
    _gate_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    _gate_model = settings.agent_model("team_update_gate")
    _gate_prompt = get_prompt("team_update_gate")
    _report_stash: dict[str, dict] = {}

    @function_tool(
        name_override="build_team_update_package",
        description_override=(
            "Build one 180-second team update narrative package for a single team from its "
            "pre-filtered candidate stories, or return no_update when those candidates do not "
            "support a verifiable on-air update."
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

        # Phase 1: Lightweight gate decision via nano
        gate_payload = {
            "team_code": tool_input.team_code,
            "team_name": tool_input.team_name,
            "candidates": [
                {
                    "story_id": c.story_id,
                    "title": c.title,
                    "source_name": c.source_name,
                    "category": c.category,
                    "framing": c.framing,
                    "continuity": c.continuity,
                    "group_update_count": len(c.recent_group_updates),
                }
                for c in tool_input.candidate_stories
            ],
        }
        gate_message = json.dumps(gate_payload, separators=(",", ":"))

        gate_result = await _parse_structured_response(
            _gate_client,
            model=_gate_model,
            instructions=_gate_prompt,
            user_message=gate_message,
            output_schema=TeamUpdateGateResult,
            token_budgets=_TEAM_UPDATE_GATE_TOKEN_BUDGETS,
            agent_name="Team Update Gate (direct)",
        )
        logger.info(
            "Team update gate decision for %s: %s",
            tool_input.team_code,
            gate_result.decision,
        )

        if gate_result.decision == "no_update":
            result = TeamUpdateAgentResult(
                status="no_update",
                skip_reason=gate_result.skip_reason or "Gate: material not airworthy",
            ).model_dump(mode="json")
            _report_stash[tool_input.team_code] = result
            return result

        # Phase 2: Full package building via mini agent
        result = await _run_nested_agent(
            tool_context,
            agent=agent,
            agent_input=tool_input.model_dump_json(),
            output_schema=TeamUpdateAgentResult,
            max_turns=8,
        )
        _report_stash[tool_input.team_code] = result
        return result

    return build_team_update_package, _report_stash


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


def _deterministic_gate(candidates: list[TeamUpdateCandidate]) -> TeamUpdateGateResult:
    """Pure-Python gate: go if any candidate is new or has fresh group activity."""
    for c in candidates:
        if c.framing == "new":
            return TeamUpdateGateResult(decision="go")
        if c.framing == "update" and len(c.recent_group_updates) > 0:
            return TeamUpdateGateResult(decision="go")
    return TeamUpdateGateResult(
        decision="no_update",
        skip_reason="Gate: no new stories and no fresh group activity",
    )


async def process_team_update(
    *,
    team_input: TeamUpdateTeamInput,
    team_update_agent: Agent,
    run_config: RunConfig,
) -> TeamUpdateAgentResult:
    """Run deterministic gate + agent for a single team.

    1. Deterministic Python gate checks framing and group activity.
    2. If gate → no_update: return immediately (zero LLM tokens).
    3. If gate → go: run the team_update_agent via Runner.run.
    """
    # Phase 1: deterministic gate
    gate_result = _deterministic_gate(team_input.candidate_stories)
    logger.info(
        "Team update gate decision for %s: %s",
        team_input.team_code,
        gate_result.decision,
    )

    if gate_result.decision == "no_update":
        return TeamUpdateAgentResult(
            status="no_update",
            skip_reason=gate_result.skip_reason or "Gate: material not airworthy",
        )

    # Phase 2: full package building via agent
    result = await Runner.run(
        team_update_agent,
        team_input.model_dump_json(),
        run_config=run_config,
        max_turns=max(6, len(team_input.candidate_stories) + 2),
        auto_previous_response_id=True,
    )
    return coerce_output(result.final_output, TeamUpdateAgentResult)
