from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from agents import Agent, RunConfig
from agents import Runner as SDKRunner
from agents.tool_context import ToolContext

from app.adapters import NewsFeedAdapter, SupabaseArticleLookupAdapter
from app.config import Settings
from app.constants import NFL_TEAMS
from app.newsroom.context import NewsroomRunContext
from app.newsroom.prompts import load_prompts
from app.newsroom.tools import build_article_data_tool
from app.newsroom.helpers import (
    build_selected_team_payloads,
    capture_selected_story_coverage,
    capture_source_usage,
    coerce_output,
    coerce_response_output,
    filter_stories_for_team,
    selected_teams,
)
from app.newsroom.workflow import AgentsWorkflow
from app.schemas import (
    ArticleDigestAgentResult,
    FeedStory,
    RadioRundown,
    RadioRundownRequest,
    SegmentCandidate,
    SourceArticleRef,
    TeamAnalysisResult,
    TeamUpdateGateResult,
    TeamStoryCandidate,
)


class FakeNewsFeedAdapter(NewsFeedAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def fetch_stories(self, lookback_hours: int) -> list[FeedStory]:
        self.calls += 1
        return [
            FeedStory(
                id="story-1",
                url="https://example.com/story-1",
                title=f"Story in last {lookback_hours} hours",
                source_name="ESPN",
                category="Breaking News",
                facts_count=4,
                entities=[
                    {"entity_type": "team", "entity_id": "ARI", "matched_name": "Arizona Cardinals"},
                    {"entity_type": "team", "entity_id": "MIN", "matched_name": "Minnesota Vikings"},
                ],
            )
        ]

    async def close(self) -> None:
        pass


class FakeArticleLookupAdapter(SupabaseArticleLookupAdapter):
    def __init__(self) -> None:
        pass

    async def close(self) -> None:
        pass


def build_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_function_auth_token="supabase-token",
    )


def build_context() -> NewsroomRunContext:
    return NewsroomRunContext(
        request=RadioRundownRequest(),
        run_id="run-1",
        generated_at=datetime.now(UTC),
    )


def build_workflow() -> AgentsWorkflow:
    return AgentsWorkflow(
        settings=build_settings(),
        news_feed=FakeNewsFeedAdapter(),
        article_lookup=FakeArticleLookupAdapter(),
    )


def test_agents_module_imports_sdk_symbols_at_top_level() -> None:
    assert SDKRunner is not None
    assert Agent is not None


def test_sdk_agents_are_real_agent_instances() -> None:
    workflow = build_workflow()

    assert isinstance(workflow.team_agent, Agent)
    assert isinstance(workflow.orchestrator_agent, Agent)
    assert workflow.article_lookup_tool.name == "lookup_article_content"
    assert workflow.article_data_tool.name == "digest_article_data"
    assert [tool.name for tool in workflow.team_agent.tools] == ["digest_article_data"]
    assert [tool.name for tool in workflow.orchestrator_agent.tools] == ["analyze_team_news"]


def test_article_data_direct_prompt_exists() -> None:
    from app.newsroom.prompts import load_prompts

    prompts = load_prompts()
    assert "article_data_direct" in prompts
    assert "article content provided below" in prompts["article_data_direct"]


@pytest.mark.asyncio
async def test_article_data_tool_direct_call_returns_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.schemas import ArticleContentLookupToolResponse, StoredArticleRecord

    settings = build_settings()

    class FakeLookup:
        async def lookup_article(self, url: str):
            return ArticleContentLookupToolResponse(
                requested_url=url,
                found=True,
                article=StoredArticleRecord(
                    url=url,
                    content="Article body with enough words for testing.",
                    header="Header",
                    description="Deck",
                    quotes=[],
                    group_id="group-1",
                ),
            )

    tool = build_article_data_tool(FakeLookup(), settings)

    captured: dict[str, object] = {}

    class FakeResponse:
        output_text = ""
        output_parsed = ArticleDigestAgentResult(
            summary="Direct summary",
            key_facts=["Fact 1"],
            confidence=0.9,
            content_status="full",
        )
        output = []

    async def fake_parse(**kwargs):
        captured["model"] = kwargs.get("model")
        captured["instructions"] = kwargs.get("instructions")
        captured["text_format"] = kwargs.get("text_format")
        captured["max_output_tokens"] = kwargs.get("max_output_tokens")
        captured["reasoning"] = kwargs.get("reasoning")
        return FakeResponse()

    monkeypatch.setattr("app.newsroom.tools.AsyncOpenAI", lambda **kw: SimpleNamespace(
        responses=SimpleNamespace(parse=fake_parse)
    ))

    # Rebuild tool after monkeypatch
    tool = build_article_data_tool(FakeLookup(), settings)

    context = ToolContext(
        context=build_context(),
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments="{}",
    )

    result = await tool.on_invoke_tool(
        context,
        json.dumps(
            {
                "story_id": "story-1",
                "url": "https://example.com/story-1",
                "title": "Story 1",
                "source_name": "ESPN",
                "category": "Breaking News",
                "team_mentions": ["ARI"],
            }
        ),
    )

    assert result["summary"] == "Direct summary"
    assert captured["model"] == settings.agent_model("article_data_agent")
    assert captured["text_format"] is ArticleDigestAgentResult
    assert captured["max_output_tokens"] == 800
    assert captured["reasoning"] == {"effort": "minimal"}


@pytest.mark.asyncio
async def test_article_data_tool_records_run_scoped_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.schemas import ArticleContentLookupToolResponse, StoredArticleRecord

    settings = build_settings()

    class FakeLookup:
        async def lookup_article(self, url: str):
            return ArticleContentLookupToolResponse(
                requested_url=url,
                found=True,
                article=StoredArticleRecord(
                    url=url,
                    content="Article body with enough words for telemetry attribution testing.",
                    header="Header",
                    description="Deck",
                    quotes=[],
                    group_id="group-1",
                ),
            )

    class FakeResponse:
        output_text = ""
        output_parsed = ArticleDigestAgentResult(
            summary="Direct summary",
            key_facts=["Fact 1"],
            confidence=0.9,
            content_status="full",
        )
        output = []

    async def fake_parse(**kwargs):
        return FakeResponse()

    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "app.newsroom.tools.AsyncOpenAI",
        lambda **kw: SimpleNamespace(responses=SimpleNamespace(parse=fake_parse)),
    )
    monkeypatch.setattr(
        "app.newsroom.tools._record_direct_api_usage",
        lambda response, **kwargs: recorded.update(kwargs),
    )

    tool = build_article_data_tool(FakeLookup(), settings)
    context = ToolContext(
        context=build_context(),
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments="{}",
        run_config=RunConfig(
            workflow_name="NFL Radio Agency",
            group_id="run-123",
            trace_metadata={"stage": "team_update_agent"},
        ),
    )

    result = await tool.on_invoke_tool(
        context,
        json.dumps(
            {
                "story_id": "story-1",
                "url": "https://example.com/story-1",
                "title": "Story 1",
                "source_name": "ESPN",
            }
        ),
    )

    assert result["summary"] == "Direct summary"
    assert recorded["agent_name"] == "Article Digest (direct)"
    assert recorded["run_id"] == "run-123"
    assert recorded["stage"] == "team_update_agent"


def test_coerce_response_output_prefers_parsed_payload_when_output_text_is_blank() -> None:
    response = SimpleNamespace(
        output_parsed=TeamUpdateGateResult(decision="go"),
        output_text="",
        output=[],
    )

    result = coerce_response_output(response, TeamUpdateGateResult)

    assert result.decision == "go"


def test_coerce_response_output_falls_back_to_message_content_when_output_text_is_blank() -> None:
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text='{"decision":"no_update","skip_reason":"No airworthy update"}',
                    )
                ],
            )
        ],
    )

    result = coerce_response_output(response, TeamUpdateGateResult)

    assert result.decision == "no_update"
    assert result.skip_reason == "No airworthy update"


def test_closed_world_rules_are_present_in_narrative_prompts() -> None:
    prompts = load_prompts()

    narrative_prompt_names = [
        "article_data_agent",
        "team_news_agent",
        "team_update_agent",
        "hourly_playlist_orchestrator_agent",
        "hourly_narrative_planner_agent",
        "radio_script_writer_agent",
        "radio_script_writer_agent_de_de",
        "rundown_orchestrator_agent",
    ]

    for name in narrative_prompt_names:
        prompt = prompts[name]
        assert "Closed-world rule:" in prompt
        assert "omit it rather than infer it" in prompt
        assert "do not invent one" in prompt
        assert "sourced descriptors, role, action, timing, and consequence" in prompt


def test_concept_explainer_rule_is_present_in_narrative_prompts() -> None:
    prompts = load_prompts()

    for name in [
        "article_data_agent",
        "team_news_agent",
        "team_update_agent",
        "hourly_playlist_orchestrator_agent",
        "hourly_narrative_planner_agent",
        "radio_script_writer_agent",
        "radio_script_writer_agent_de_de",
        "rundown_orchestrator_agent",
    ]:
        prompt = prompts[name]
        assert "safety, dead cap, cap hit, waived, practice squad, or IR" in prompt
        assert "timeless concept explanation" in prompt


def test_writer_prompt_requires_traceable_facts_and_explicit_names() -> None:
    prompt = load_prompts()["radio_script_writer_agent"]

    assert "every factual statement in the intro, body, and outro must be traceable" in prompt
    assert "use that explicit name instead of vague phrasing" in prompt
    assert 'You may add "what this means" lines only as timeless concept explanations' in prompt
    assert "Treat the narrative_context fields as hard direction" in prompt
    assert "use only a brief callback and then pivot quickly into the fresh angle" in prompt
    assert "Do not replay the prior segment's lead sentence pattern" in prompt
    assert "Use primary_angle and fresh_material_to_emphasize as the center of gravity" in prompt
    assert 'Use micro-imperfections sparingly. An occasional "look," "right," "I mean," or "you know"' in prompt
    assert "place one longer sentence followed by a short punchy sentence" in prompt
    assert "Give each line a clear emotional intent such as confident, amused, or skeptical" in prompt
    assert "director_notes should be a compact high-level performance note" in prompt
    assert "pace should contain only the pacing guidance" in prompt
    assert "must_hit should list the exact names, phrases, or facts" in prompt


def test_narrative_planner_prompt_requires_distinct_angles_for_related_stories() -> None:
    prompt = load_prompts()["hourly_narrative_planner_agent"]

    assert "same underlying event or storyline" in prompt
    assert "meaningfully different primary_angle" in prompt
    assert "facts_already_aired" in prompt
    assert "fresh_material_to_emphasize" in prompt


def test_german_script_prompts_are_present_and_localized() -> None:
    prompts = load_prompts()
    writer_prompt = prompts["radio_script_writer_agent_de_de"]
    batch_prompt = prompts["hourly_script_batch_agent_de_de"]

    assert "Write the following fields in natural German for a German radio audience" in writer_prompt
    assert "Keep proper names, team names, explicit quotes, and source-grounded NFL terminology accurate" in writer_prompt
    assert "use only a brief callback and then pivot quickly into the fresh angle" in writer_prompt
    assert 'Use micro-imperfections sparingly in natural German. An occasional "schau", "ganz ehrlich", "weisst du", or "oder?"' in writer_prompt
    assert "place one longer sentence followed by a short punchy sentence" in writer_prompt
    assert "Give each line a clear emotional intent such as selbstbewusst, leicht amuesiert, or skeptisch" in writer_prompt
    assert "Preserve the supplied anti-repetition direction" in batch_prompt
    assert "All returned direction fields and script copy must be in natural German" in batch_prompt


def test_model_settings_omit_temperature_and_max_tokens_by_default() -> None:
    workflow = build_workflow()

    assert workflow.team_agent.model_settings.temperature is None
    assert workflow.team_agent.model_settings.max_tokens is None
    assert workflow.orchestrator_agent.model_settings.temperature is None
    assert workflow.orchestrator_agent.model_settings.max_tokens is None


def test_agents_workflow_uses_agent_specific_model_overrides() -> None:
    settings = Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_function_auth_token="supabase-token",
        openai_model_article_data_agent="article-model",
        openai_model_team_news_agent="team-model",
        openai_model_rundown_orchestrator_agent="rundown-model",
    )

    workflow = AgentsWorkflow(
        settings=settings,
        news_feed=FakeNewsFeedAdapter(),
        article_lookup=FakeArticleLookupAdapter(),
    )

    assert workflow.team_agent.model == "team-model"
    assert workflow.orchestrator_agent.model == "rundown-model"


def test_agent_output_coercion_accepts_valid_dict() -> None:
    payload = {
        "team": "ARI",
        "scored_stories": [
            {
                "story_id": "story-1",
                "url": "https://example.com/story-1",
                "title": "Headline source",
                "source_name": "ESPN",
                "category": "Breaking News",
                "headline": "Headline",
                "summary": "Summary",
                "key_points": ["Point 1"],
                "segment_idea": "Segment",
                "relevance_score": 0.91,
                "confidence": 0.75,
                "why_this_matters_now": "Because it matters.",
            }
        ],
    }

    result = coerce_output(payload, TeamAnalysisResult)
    assert result.team == "ARI"
    assert result.scored_stories[0].story_id == "story-1"


def test_agent_output_coercion_rejects_invalid_dict() -> None:
    with pytest.raises(ValueError):
        coerce_output(
            {
                "team": "ARI",
                "scored_stories": [
                    {
                        "story_id": "story-1",
                        "url": "https://example.com/story-1",
                    }
                ],
            },
            TeamAnalysisResult,
        )


def test_filter_stories_for_team_uses_exact_entity_id() -> None:
    stories = [
        FeedStory(
            id="story-1",
            url="https://example.com/ari",
            title="Cardinals update",
            source_name="ESPN",
            entities=[{"entity_type": "team", "entity_id": "ARI", "matched_name": "Arizona Cardinals"}],
        ),
        FeedStory(
            id="story-2",
            url="https://example.com/atl",
            title="Falcons update",
            source_name="ESPN",
            entities=[{"entity_type": "team", "entity_id": "ATL", "matched_name": "Atlanta Falcons"}],
        ),
    ]

    result = filter_stories_for_team(stories, "ARI")

    assert [story.id for story in result] == ["story-1"]


def test_request_teams_are_normalized_and_deduplicated() -> None:
    request = RadioRundownRequest(teams=["ari", "MIN", "ari"])

    assert request.teams == ["ARI", "MIN"]


def test_request_teams_reject_unknown_codes() -> None:
    with pytest.raises(ValueError):
        RadioRundownRequest(teams=["NOPE"])


def test_selected_team_payloads_include_only_exact_team_matches() -> None:
    stories = [
        FeedStory(
            id="story-1",
            url="https://example.com/ari",
            title="Arizona update",
            source_name="ESPN",
            entities=[{"entity_type": "team", "entity_id": "ARI", "matched_name": "Arizona Cardinals"}],
        ),
        FeedStory(
            id="story-2",
            url="https://example.com/min",
            title="Minnesota update",
            source_name="Yahoo",
            entities=[{"entity_type": "team", "entity_id": "MIN", "matched_name": "Minnesota Vikings"}],
        ),
    ]

    payloads = build_selected_team_payloads(
        RadioRundownRequest(teams=["ARI"]),
        stories,
    )

    assert payloads == [
        {
            "team_code": "ARI",
            "team_name": "Arizona Cardinals",
            "stories": [
                {
                    "story_id": "story-1",
                    "url": "https://example.com/ari",
                    "title": "Arizona update",
                    "source_name": "ESPN",
                    "category": None,
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_run_newsroom_uses_single_top_level_runner_call(monkeypatch) -> None:
    workflow = build_workflow()
    context = build_context()
    calls: list[tuple[str, str | None, int | None, bool | None]] = []

    async def fake_runner(agent, message, *, context=None, run_config=None, max_turns=None, **_kwargs):
        calls.append(
            (
                agent.name,
                getattr(run_config, "group_id", None),
                max_turns,
                _kwargs.get("auto_previous_response_id"),
            )
        )
        assert "selected_teams" in message
        return SimpleNamespace(
            final_output=RadioRundown(
                run_id="",
                generated_at=datetime.now(UTC),
                lookback_hours=24,
                target_duration_minutes=60,
                segments=[],
                backup_segments=[],
                warnings=[],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_rundown.Runner.run", fake_runner)

    result = await workflow.run_newsroom(context)

    assert result.target_duration_minutes == 60
    assert len(context.feed_stories) == 1
    assert calls == [("Rundown Orchestrator Agent", "run-1", 36, True)]


@pytest.mark.asyncio
async def test_run_newsroom_limits_selected_team_payloads(monkeypatch) -> None:
    workflow = build_workflow()
    context = NewsroomRunContext(
        request=RadioRundownRequest(teams=["ARI", "MIN"]),
        run_id="run-1",
        generated_at=datetime.now(UTC),
    )
    captured_message = ""

    async def fake_runner(agent, message, **_kwargs):
        nonlocal captured_message
        captured_message = message
        return SimpleNamespace(
            final_output=RadioRundown(
                run_id="",
                generated_at=datetime.now(UTC),
                lookback_hours=24,
                target_duration_minutes=60,
                segments=[],
                backup_segments=[],
                warnings=[],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_rundown.Runner.run", fake_runner)

    await workflow.run_newsroom(context)

    assert '"team_code":"ARI"' in captured_message
    assert '"team_code":"MIN"' in captured_message
    assert '"team_code":"ATL"' not in captured_message


def test_capture_source_usage_builds_digest_placeholders_from_feed_refs() -> None:
    context = build_context()
    context.feed_stories = [
        FeedStory(
            id="story-1",
            url="https://example.com/story-1",
            title="Story 1",
            source_name="ESPN",
            category="Breaking News",
            entities=[{"entity_type": "team", "entity_id": "ARI", "matched_name": "Arizona Cardinals"}],
        )
    ]
    draft = RadioRundown(
        run_id="",
        generated_at=datetime.now(UTC),
        lookback_hours=24,
        target_duration_minutes=60,
        segments=[
            SegmentCandidate(
                rank=1,
                headline="Headline",
                editorial_angle="Angle",
                teams=["ARI"],
                story_priority=5,
                recommended_duration_seconds=300,
                recommended_word_count=750,
                summary="Summary",
                key_points=["Point"],
                segment_idea="Idea",
                source_articles=[
                    SourceArticleRef(
                        story_id="story-1",
                        url="https://example.com/story-1",
                        title="Story 1",
                        source_name="ESPN",
                    )
                ],
                confidence=0.8,
            )
        ],
        backup_segments=[],
        warnings=[],
    )

    capture_source_usage(context, draft)

    assert context.researched_urls == {"https://example.com/story-1"}
    assert [digest.story_id for digest in context.article_digests] == ["story-1"]


def test_capture_selected_story_coverage_marks_all_team_stories() -> None:
    context = build_context()
    selected_teams = [
        {
            "team_code": "ARI",
            "team_name": "Arizona Cardinals",
            "stories": [
                {
                    "id": "story-1",
                    "url": "https://example.com/story-1",
                    "title": "Story 1",
                    "source_name": "ESPN",
                    "category": "Breaking News",
                    "facts_count": 3,
                    "entities": [
                        {
                            "entity_type": "team",
                            "entity_id": "ARI",
                            "matched_name": "Arizona Cardinals",
                        }
                    ],
                }
            ],
        },
        {
            "team_code": "MIN",
            "team_name": "Minnesota Vikings",
            "stories": [
                {
                    "id": "story-1",
                    "url": "https://example.com/story-1",
                    "title": "Story 1",
                    "source_name": "ESPN",
                    "category": "Breaking News",
                    "facts_count": 3,
                    "entities": [
                        {
                            "entity_type": "team",
                            "entity_id": "ARI",
                            "matched_name": "Arizona Cardinals",
                        }
                    ],
                },
                {
                    "id": "story-2",
                    "url": "https://example.com/story-2",
                    "title": "Story 2",
                    "source_name": "Yahoo",
                    "category": "Analysis",
                    "facts_count": 2,
                    "entities": [
                        {
                            "entity_type": "team",
                            "entity_id": "MIN",
                            "matched_name": "Minnesota Vikings",
                        }
                    ],
                },
            ],
        },
    ]

    capture_selected_story_coverage(context, selected_teams)

    assert context.researched_urls == {
        "https://example.com/story-1",
        "https://example.com/story-2",
    }
    assert [digest.story_id for digest in context.article_digests] == ["story-1", "story-2"]


def test_selected_teams_defaults_to_full_league() -> None:
    result = selected_teams(RadioRundownRequest())

    assert result == NFL_TEAMS
