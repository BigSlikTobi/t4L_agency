from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.newsroom.context import NewsroomRunContext
from app.orchestration import NewsroomOrchestrator
from app.schemas import (
    ArticleDigest,
    FeedStory,
    HourlyPlaylist,
    HourlyPlaylistItem,
    HourlyPlaylistScriptsRequest,
    HourlyPlaylistScriptsRunResult,
    RadioRundown,
    RadioStoryScript,
    RadioRundownRequest,
    SegmentCandidate,
    SourceArticleRef,
    SourceCoverage,
    TeamUpdateBatchRequest,
    TeamUpdateBatchResponse,
    TeamUpdatePackage,
    TeamUpdateReport,
    TeamUpdateReportRequest,
    TeamUpdateTopic,
)

class FakeAgents:
    async def run_newsroom(self, context: NewsroomRunContext):
        context.feed_stories = [
            FeedStory(
                id="story-1",
                url="https://example.com/vikings",
                title="Vikings center retires",
                source_name="ESPN",
                category="Breaking News",
                facts_count=5,
                entities=[{"entity_type": "team", "entity_id": "MIN", "matched_name": "Minnesota Vikings"}],
            ),
            FeedStory(
                id="story-2",
                url="https://example.com/shared",
                title="League reacts to blockbuster trade",
                source_name="NFL Network",
                category="Analysis",
                facts_count=6,
                entities=[
                    {"entity_type": "team", "entity_id": "MIN", "matched_name": "Minnesota Vikings"},
                    {"entity_type": "team", "entity_id": "HOU", "matched_name": "Houston Texans"},
                ],
            ),
        ]
        context.article_digests = [
            ArticleDigest(
                story_id="story-1",
                team_mentions=["MIN"],
                url="https://example.com/vikings",
                title="Vikings center retires",
                source_name="ESPN",
                category="Breaking News",
                summary="Summary for story-1",
                key_facts=["Fact for story-1"],
                confidence=0.75,
            ),
            ArticleDigest(
                story_id="story-2",
                team_mentions=["MIN", "HOU"],
                url="https://example.com/shared",
                title="League reacts to blockbuster trade",
                source_name="NFL Network",
                category="Analysis",
                summary="Summary for story-2",
                key_facts=["Fact for story-2"],
                confidence=0.8,
            ),
        ]
        context.researched_urls = {
            "https://example.com/vikings",
            "https://example.com/shared",
        }
        return RadioRundown(
            run_id=context.run_id,
            generated_at=datetime.now(UTC),
            lookback_hours=24,
            target_duration_minutes=60,
            segments=[
                SegmentCandidate(
                    rank=1,
                    headline="Shared blockbuster story",
                    editorial_angle="Two-team angle",
                    teams=["MIN", "HOU"],
                    story_priority=5,
                    recommended_duration_seconds=900,
                    recommended_word_count=2250,
                    summary="Shared summary",
                    key_points=["Point A", "Point B"],
                    segment_idea="Lead with the shared story",
                    source_articles=[
                        SourceArticleRef(
                            story_id="story-3",
                            url="https://example.com/shared",
                            title="League reacts to blockbuster trade",
                            source_name="NFL Network",
                        )
                    ],
                    confidence=0.9,
                ),
                SegmentCandidate(
                    rank=2,
                    headline="Vikings retirements",
                    editorial_angle="MIN angle",
                    teams=["MIN"],
                    story_priority=4,
                    recommended_duration_seconds=600,
                    recommended_word_count=1500,
                    summary="Vikings summary",
                    key_points=["MIN point"],
                    segment_idea="Discuss Vikings line impact",
                    source_articles=[
                        SourceArticleRef(
                            story_id="story-1",
                            url="https://example.com/vikings",
                            title="Vikings center retires",
                            source_name="ESPN",
                        )
                    ],
                    confidence=0.82,
                ),
            ],
            backup_segments=[],
            source_coverage=SourceCoverage(
                total_feed_stories=0,
                researched_articles=0,
                digested_articles=0,
                teams_with_stories=0,
                teams_without_stories=0,
            ),
            warnings=[],
        )


class FakeTeamUpdates:
    async def run_team_update(self, context):
        context.feed_stories = [
            FeedStory(
                id="story-1",
                url="https://example.com/vikings",
                title="Vikings update",
                source_name="ESPN",
                category="Breaking News",
                facts_count=2,
                entities=[{"entity_type": "team", "entity_id": "MIN", "matched_name": "Minnesota Vikings"}],
            )
        ]
        context.candidate_stories = [
            {
                "story_id": "story-1",
                "url": "https://example.com/vikings",
                "title": "Vikings update",
                "source_name": "ESPN",
                "category": "Breaking News",
                "team_code": "MIN",
                "group_id": "group-1",
                "framing": "new",
                "recent_group_updates": [],
            }
        ]
        return TeamUpdateReport(
            run_id=context.run_id,
            generated_at=context.generated_at,
            team=context.request.team,
            lookback_minutes=context.request.lookback_minutes,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Vikings update package",
                topics=[
                    TeamUpdateTopic(
                        headline="Vikings topic",
                        editorial_angle="Angle",
                        recommended_duration_seconds=180,
                        framing="new",
                        continuity="new_story",
                        talking_points=["Point"],
                        source_articles=[],
                    )
                ],
                source_articles=[],
            ),
            coverage={
                "total_feed_stories": 1,
                "team_feed_stories": 1,
                "candidate_stories": 1,
                "new_candidates": 1,
                "update_candidates": 0,
                "prior_reports_considered": 0,
            },
            warnings=[],
        )

    async def run_team_update_batch(self, *, run_id, generated_at, request):
        teams = request.teams or ["ARI", "MIN"]
        return TeamUpdateBatchResponse(
            run_id=run_id,
            generated_at=generated_at,
            lookback_minutes=request.lookback_minutes,
            reports=[
                TeamUpdateReport(
                    run_id=run_id,
                    generated_at=generated_at,
                    team=team,
                    lookback_minutes=request.lookback_minutes,
                    status="no_update",
                    coverage={},
                    warnings=[],
                )
                for team in teams
            ],
            hourly_playlist=HourlyPlaylist(
                selected_count=1 if teams else 0,
                items=[
                    HourlyPlaylistItem(
                        rank=1,
                        team=teams[0],
                        headline="Top headline",
                        framing="new",
                        continuity="new_story",
                        production_reason="Top report.",
                        source_articles=[],
                    )
                ]
                if teams
                else [],
            ),
        )

    async def run_hourly_playlist_scripts(self, *, script_run_id, generated_at, request):
        return HourlyPlaylistScriptsRunResult(
            script_run_id=script_run_id,
            playlist_id=request.playlist_id or "playlist-latest",
            batch_run_id="batch-run-1",
            language=request.language,
            generated_at=generated_at,
            scripts=[
                RadioStoryScript(
                    team="ARI",
                    playlist_rank=1,
                    language=request.language,
                    headline="Top headline",
                    continuity="new_story",
                    persona_name="Mason Reed",
                    persona_backstory="Drive-time breaker with Northeast instincts.",
                    persona_specialty="breaking news and urgency",
                    voice_name="Charon",
                    dialect="a light Mid-Atlantic edge",
                    duration_seconds=180,
                    slug="ari-top-headline",
                    audio_profile="Confident breaking-news sports anchor.",
                    scene="Top-of-hour NFL update.",
                    director_notes="Lead quickly and clearly.",
                    pace="Urgent but controlled.",
                    warmth="Close but crisp.",
                    must_hit=["Geno Smith", "Raiders paying most of the salary"],
                    pronunciations=[
                        {
                            "term": "Tom Pelissero",
                            "guide": "PEH-lih-SARE-oh",
                        }
                    ],
                    tts_prompt="Audio Profile: Confident anchor.\nScene: Top-of-hour NFL update.\nDirector's Notes: Lead quickly and clearly.\nScript: Intro Body Outro",
                    intro="Intro",
                    body="Body",
                    outro="Outro",
                    source_articles=[],
                )
            ],
            tts_batch={
                "batch_id": "batches/abc123",
                "status": "JOB_STATE_SUCCEEDED",
                "processed_count": 1,
                "failed_count": 0,
                "token_usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "reported_item_count": 1,
                },
                "local_usage_summary_file": "/tmp/usage.json",
                "usage_summary_path": "gemini-tts-batch/batches/abc123/usage_summary.json",
                "usage_summary_public_url": "https://example.com/usage.json",
                "manifest_path": "gemini-tts-batch/batches/abc123/manifest.json",
                "manifest_public_url": "https://example.com/manifest.json",
                "items": [
                    {
                        "id": "ari-top-headline",
                        "storage_path": "gemini-tts-batch/batches/abc123/ari-top-headline.mp3",
                        "mime_type": "audio/mpeg",
                        "source_mime_type": "audio/wav",
                        "public_url": "https://example.com/ari-top-headline.mp3",
                        "token_usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 0,
                            "output_tokens": 20,
                            "total_tokens": 30,
                        },
                    }
                ],
                "failures": [],
            },
        )


def build_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_function_auth_token="supabase-token",
        supabase_storage_key="storage-key",
    )


@pytest.mark.asyncio
async def test_orchestration_returns_agent_rundown_directly_and_preserves_metadata() -> None:
    orchestrator = NewsroomOrchestrator(
        settings=build_settings(),
        agents=FakeAgents(),
    )

    rundown = await orchestrator.run_radio_rundown(RadioRundownRequest())

    assert sum(segment.recommended_duration_seconds for segment in rundown.segments) == 1500
    assert rundown.segments[0].headline == "Shared blockbuster story"
    assert rundown.source_coverage.teams_without_stories + rundown.source_coverage.teams_with_stories == 32
    assert rundown.lookback_hours == 24
    assert rundown.target_duration_minutes == 60


@pytest.mark.asyncio
async def test_orchestration_runs_team_update_flow() -> None:
    orchestrator = NewsroomOrchestrator(
        settings=build_settings(),
        agents=FakeAgents(),
        team_updates=FakeTeamUpdates(),
    )

    report = await orchestrator.run_team_update_report(TeamUpdateReportRequest(team="MIN"))

    assert report.status == "report_ready"
    assert report.team == "MIN"
    assert report.report is not None
    assert report.report.total_duration_seconds == 180


@pytest.mark.asyncio
async def test_orchestration_runs_team_update_batch_flow() -> None:
    orchestrator = NewsroomOrchestrator(
        settings=build_settings(),
        agents=FakeAgents(),
        team_updates=FakeTeamUpdates(),
    )

    batch_response = await orchestrator.run_team_update_reports(
        TeamUpdateBatchRequest(teams=["ARI", "MIN"], lookback_minutes=60)
    )

    assert len(batch_response.reports) == 2
    assert batch_response.reports[0].run_id == batch_response.reports[1].run_id
    assert [report.team for report in batch_response.reports] == ["ARI", "MIN"]
    assert batch_response.hourly_playlist.selected_count == 1


@pytest.mark.asyncio
async def test_orchestration_runs_hourly_playlist_script_flow() -> None:
    orchestrator = NewsroomOrchestrator(
        settings=build_settings(),
        agents=FakeAgents(),
        team_updates=FakeTeamUpdates(),
    )

    response = await orchestrator.run_hourly_playlist_scripts(HourlyPlaylistScriptsRequest())

    assert response.language == "en-US"
    assert response.voice_name == "Charon"
    assert len(response.items) == 1
    assert response.items[0].id == "ari-top-headline"
    assert response.items[0].title == "Top headline"
    assert response.items[0].direction.pace == "Urgent but controlled."
    assert response.items[0].direction.must_hit == [
        "Geno Smith",
        "Raiders paying most of the salary",
    ]
    assert response.items[0].direction.pronunciations[0].guide == "PEH-lih-SARE-oh"
    assert response.tts_batch is not None
    assert response.tts_batch.items[0].public_url == "https://example.com/ari-top-headline.mp3"


@pytest.mark.asyncio
async def test_orchestration_preserves_hourly_playlist_script_language() -> None:
    orchestrator = NewsroomOrchestrator(
        settings=build_settings(),
        agents=FakeAgents(),
        team_updates=FakeTeamUpdates(),
    )

    response = await orchestrator.run_hourly_playlist_scripts(
        HourlyPlaylistScriptsRequest(language="de-DE")
    )

    assert response.language == "de-DE"
    assert response.tts_batch is not None
