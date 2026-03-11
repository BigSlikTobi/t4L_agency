from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.history import TeamUpdateHistoryStore
from app.newsroom.context import TeamUpdateRunContext
from app.newsroom.helpers import build_hourly_script_batch_input
from app.newsroom.workflow import TeamUpdateWorkflow
from app.schemas import (
    ArticleContentLookupToolResponse,
    FeedStory,
    HourlyPlaylist,
    HourlyPlaylistItem,
    HourlyPlaylistScriptsRequest,
    HourlyPlaylistScriptsResponse,
    RadioStoryScript,
    RadioStoryScriptInput,
    SourceArticleRef,
    StoredArticleRecord,
    StoryGroupUpdate,
    StoryGroupUpdatesToolResponse,
    TeamUpdateBatchRequest,
    TeamUpdateBatchAgentResult,
    TeamUpdateBatchAgentReport,
    TeamUpdateBatchResponse,
    TeamUpdatePackage,
    TeamUpdateReport,
    TeamUpdateReportRequest,
    TeamUpdateTopic,
    TTSBatchResult,
)
from app.config import Settings
from app.newsroom.workflow import _normalize_radio_story_scripts


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_story_group_updates_url="https://example.com/story-group-updates",
        supabase_function_auth_token="supabase-token",
        supabase_storage_key="storage-key",
        team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3",
    )


def make_story(
    *,
    story_id: str = "story-1",
    url: str = "https://example.com/story-1",
    title: str = "Roster update",
    team_code: str = "MIN",
) -> FeedStory:
    return FeedStory(
        id=story_id,
        url=url,
        title=title,
        source_name="ESPN",
        category="Breaking News",
        facts_count=3,
        entities=[{"entity_type": "team", "entity_id": team_code, "matched_name": "Minnesota Vikings"}],
    )


class FakeNewsFeedAdapter:
    def __init__(self, stories: list[FeedStory]) -> None:
        self.stories = stories
        self.calls: list[int] = []

    async def fetch_stories(self, lookback_hours: int) -> list[FeedStory]:
        self.calls.append(lookback_hours)
        return self.stories


class FakeArticleLookupAdapter:
    def __init__(self, responses: dict[str, ArticleContentLookupToolResponse]) -> None:
        self.responses = responses

    async def lookup_article(self, url: str) -> ArticleContentLookupToolResponse:
        return self.responses[url]


class FakeStoryGroupUpdatesAdapter:
    def __init__(self, responses: dict[str, StoryGroupUpdatesToolResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    async def fetch_recent_updates(
        self,
        *,
        group_id: str,
        lookback_minutes: int,
    ) -> StoryGroupUpdatesToolResponse:
        self.calls.append((group_id, lookback_minutes))
        return self.responses[group_id]


class FakeGeminiTTSBatchAdapter:
    def __init__(
        self,
        *,
        create_response: dict | None = None,
        status_responses: list[dict] | None = None,
        process_response: dict | None = None,
    ) -> None:
        self.create_calls: list[dict] = []
        self.status_calls: list[dict] = []
        self.process_calls: list[dict] = []
        self.create_response = create_response or {
            "batch_id": "batches/abc123",
            "status": "JOB_STATE_PENDING",
            "created_at": "2026-03-11T10:00:00Z",
            "updated_at": "2026-03-11T10:00:00Z",
            "output_file_id": None,
            "error_file_id": None,
            "error": None,
            "model_name": "gemini-2.5-pro-preview-tts",
            "total_items": 1,
            "input_file_name": "files/123",
            "local_input_file": "/tmp/input.jsonl",
            "supported_generation_methods": ["batchGenerateContent"],
        }
        self.status_responses = status_responses or [
            {
                "batch_id": "batches/abc123",
                "status": "JOB_STATE_SUCCEEDED",
                "created_at": "2026-03-11T10:00:00Z",
                "updated_at": "2026-03-11T10:05:00Z",
                "output_file_id": "files/out",
                "error_file_id": None,
                "error": None,
            }
        ]
        self.process_response = process_response or {
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
                    "id": "min-headline",
                    "storage_path": "gemini-tts-batch/batches/abc123/min-headline.mp3",
                    "mime_type": "audio/mpeg",
                    "source_mime_type": "audio/wav",
                    "public_url": "https://example.com/min-headline.mp3",
                    "token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 0,
                        "output_tokens": 20,
                        "total_tokens": 30,
                    },
                }
            ],
            "failures": [],
        }

    async def create_batch(self, request):
        self.create_calls.append(request.model_dump(mode="json"))
        return type("CreateResponse", (), self.create_response)()

    async def fetch_batch_status(self, request):
        self.status_calls.append(request.model_dump(mode="json"))
        response = self.status_responses[min(len(self.status_calls) - 1, len(self.status_responses) - 1)]
        return type("StatusResponse", (), response)()

    async def process_batch(self, request):
        self.process_calls.append(request.model_dump(mode="json"))
        return TTSBatchResult.model_validate(self.process_response)


def make_article_lookup_response(
    *,
    url: str = "https://example.com/story-1",
    group_id: str = "group-1",
    updated_at: datetime | None = None,
) -> ArticleContentLookupToolResponse:
    return ArticleContentLookupToolResponse(
        requested_url=url,
        found=True,
        article=StoredArticleRecord(
            url=url,
            cite_url=url,
            header="Header",
            content="Body",
            description="Deck",
            author="Reporter",
            category="Breaking News",
            quotes=[],
            group_id=group_id,
            updated_at=updated_at,
        ),
    )


def make_group_updates_response(
    *,
    group_id: str = "group-1",
    member_identifier: str = "story-2",
) -> StoryGroupUpdatesToolResponse:
    return StoryGroupUpdatesToolResponse(
        group_id=group_id,
        lookback_minutes=60,
        updates=[
            StoryGroupUpdate(
                member_identifier=member_identifier,
                added_at=datetime.now(UTC),
            )
        ],
    )


def build_workflow(
    tmp_path: Path,
    *,
    stories: list[FeedStory],
    article_lookup_responses: dict[str, ArticleContentLookupToolResponse],
    group_update_responses: dict[str, StoryGroupUpdatesToolResponse],
    tts_batch: FakeGeminiTTSBatchAdapter | None = None,
) -> TeamUpdateWorkflow:
    settings = build_settings(tmp_path)
    return TeamUpdateWorkflow(
        settings=settings,
        news_feed=FakeNewsFeedAdapter(stories),
        article_lookup=FakeArticleLookupAdapter(article_lookup_responses),
        story_group_updates=FakeStoryGroupUpdatesAdapter(group_update_responses),
        history_store=TeamUpdateHistoryStore(settings.team_update_history_sqlite_path),
        tts_batch=tts_batch or FakeGeminiTTSBatchAdapter(),
    )


def build_context(team: str = "MIN", lookback_minutes: int = 60) -> TeamUpdateRunContext:
    return TeamUpdateRunContext(
        request=TeamUpdateReportRequest(team=team, lookback_minutes=lookback_minutes),
        run_id="update-run-1",
        generated_at=datetime.now(UTC),
    )


def make_topic(
    *,
    headline: str = "Topic",
    editorial_angle: str = "Angle",
    framing: str = "new",
    continuity: str = "new_story",
    what_changed: str | None = None,
    source_articles: list[SourceArticleRef] | None = None,
) -> TeamUpdateTopic:
    return TeamUpdateTopic(
        headline=headline,
        editorial_angle=editorial_angle,
        recommended_duration_seconds=180,
        framing=framing,
        continuity=continuity,
        what_changed=what_changed,
        talking_points=["Point"],
        source_articles=source_articles or [],
    )


def make_team_update_report(
    *,
    team: str,
    lookback_minutes: int,
    status: str = "no_update",
    report: TeamUpdatePackage | None = None,
) -> TeamUpdateReport:
    return TeamUpdateReport(
        run_id=f"update-run-{team}",
        generated_at=datetime.now(UTC),
        team=team,
        lookback_minutes=lookback_minutes,
        status=status,
        report=report,
        coverage={},
        warnings=[],
    )


def make_radio_story_script(
    *,
    team: str = "MIN",
    playlist_rank: int = 1,
    continuity: str = "new_story",
    language: str = "en-US",
    persona_name: str = "Lena Torres",
    persona_backstory: str = "Phoenix night host turned NFL connector.",
    persona_specialty: str = "fan emotion and momentum swings",
    voice_name: str = "Puck",
    dialect: str = "a slight Arizona-Southwest lilt",
) -> RadioStoryScript:
    return RadioStoryScript(
        team=team,
        playlist_rank=playlist_rank,
        language=language,
        headline=f"{team} headline",
        continuity=continuity,
        persona_name=persona_name,
        persona_backstory=persona_backstory,
        persona_specialty=persona_specialty,
        voice_name=voice_name,
        dialect=dialect,
        duration_seconds=180,
        slug=f"{team.lower()}-headline",
        audio_profile="Warm FM host with bright energy and close-mic intimacy.",
        scene="Late-hour NFL update hitting a listener on the drive home.",
        director_notes="Stay warm, direct, and vivid. Build quick connection.",
        pace="Urgent but controlled.",
        warmth="Engaged and direct.",
        must_hit=["Headline", "Key fact"],
        pronunciations=[],
        tts_prompt="Audio Profile: Warm FM host.\nScene: Drive-home NFL update.\nDirector's Notes: Warm and direct.\nScript: Intro Body Outro",
        intro="Intro",
        body="Body",
        outro="Outro",
        source_articles=[],
    )


def test_team_update_request_normalizes_team() -> None:
    request = TeamUpdateReportRequest(team="min")

    assert request.team == "MIN"
    assert request.lookback_minutes == 60


def test_team_update_request_accepts_large_lookback_for_testing() -> None:
    request = TeamUpdateReportRequest(team="MIN", lookback_minutes=1440)

    assert request.lookback_minutes == 1440


def test_team_update_batch_request_defaults_to_all_teams_when_teams_absent() -> None:
    request = TeamUpdateBatchRequest()

    assert request.teams is None
    assert request.lookback_minutes == 60


def test_hourly_playlist_scripts_request_defaults_to_latest_playlist() -> None:
    request = HourlyPlaylistScriptsRequest()

    assert request.playlist_id is None
    assert request.language == "en-US"
    assert request.enable_tts is True


def test_hourly_playlist_scripts_request_accepts_de_de_language() -> None:
    request = HourlyPlaylistScriptsRequest(language="de-DE")

    assert request.language == "de-DE"


def test_hourly_playlist_scripts_request_accepts_enable_tts_override() -> None:
    request = HourlyPlaylistScriptsRequest(enable_tts=False)

    assert request.enable_tts is False


def test_team_update_workflow_uses_agent_specific_model_overrides(tmp_path: Path) -> None:
    settings = Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_story_group_updates_url="https://example.com/story-group-updates",
        supabase_function_auth_token="supabase-token",
        supabase_storage_key="storage-key",
        team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3",
        openai_model_article_data_agent="article-model",
        openai_model_team_update_agent="team-update-model",
        openai_model_team_update_batch_agent="batch-model",
        openai_model_hourly_playlist_orchestrator_agent="playlist-model",
        openai_model_radio_script_writer_agent="script-model",
        openai_model_hourly_script_batch_agent="script-batch-model",
    )

    workflow = TeamUpdateWorkflow(
        settings=settings,
        news_feed=FakeNewsFeedAdapter([]),
        article_lookup=FakeArticleLookupAdapter({}),
        story_group_updates=FakeStoryGroupUpdatesAdapter({}),
        history_store=TeamUpdateHistoryStore(settings.team_update_history_sqlite_path),
        tts_batch=FakeGeminiTTSBatchAdapter(),
    )

    assert workflow.article_data_agent.model == "article-model"
    assert workflow.team_update_agent.model == "team-update-model"
    assert workflow.team_update_batch_agent.model == "batch-model"
    assert workflow.hourly_playlist_orchestrator_agent.model == "playlist-model"
    assert workflow.radio_script_writer_agent.model == "script-model"
    assert workflow.hourly_script_batch_agent.model == "script-batch-model"
    assert workflow.script_generation_stacks["de-DE"].radio_script_writer_agent.model == "script-model"
    assert workflow.script_generation_stacks["de-DE"].hourly_script_batch_agent.model == "script-batch-model"


def test_radio_story_script_normalization_uses_playlist_rank_over_agent_team_label() -> None:
    selected_stories = [
        RadioStoryScriptInput(
            team="MIN",
            playlist_rank=1,
            headline="Minnesota headline",
            framing="new",
            continuity="new_story",
            production_reason="Lead",
            story_synopsis="Minnesota package Topic Angle",
            source_articles=[],
        ),
        RadioStoryScriptInput(
            team="LAC",
            playlist_rank=2,
            headline="Chargers headline",
            framing="new",
            continuity="new_story",
            production_reason="Second",
            story_synopsis="Chargers package Topic Angle",
            source_articles=[],
        ),
    ]

    scripts = [
        make_radio_story_script(team="MIN", playlist_rank=1),
        make_radio_story_script(team="MIN", playlist_rank=2),
    ]

    normalized = _normalize_radio_story_scripts(scripts, selected_stories)

    assert [script.team for script in normalized] == ["MIN", "LAC"]
    assert [script.playlist_rank for script in normalized] == [1, 2]
    assert [script.language for script in normalized] == ["en-US", "en-US"]


def test_radio_story_script_normalization_optimizes_text_for_tts() -> None:
    selected_stories = [
        RadioStoryScriptInput(
            team="MIN",
            playlist_rank=1,
            headline="Minnesota let’s roll",
            framing="new",
            continuity="new_story",
            production_reason="Lead",
            story_synopsis="Minnesota package Topic Angle",
            source_articles=[],
        )
    ]

    scripts = [
        RadioStoryScript(
            team="MIN",
            playlist_rank=1,
            headline="Minnesota let’s roll",
            continuity="new_story",
            persona_name="Lena Torres",
            persona_backstory="Phoenix night host — built for the drive home.",
            persona_specialty="fan emotion and momentum swings",
            voice_name="Puck",
            dialect="a slight Arizona-Southwest lilt",
            duration_seconds=180,
            slug="minnesota_let’s_roll",
            audio_profile="Warm FM host — close and bright.",
            scene="Late-hour update… right in the listener’s ear.",
            director_notes="Stay warm, direct, and don’t rush.",
            pace="Don’t rush.",
            warmth="Stay warm and direct.",
            must_hit=["Minnesota headline"],
            pronunciations=[
                {
                    "term": "Tagovailoa",
                    "guide": "TAH-go-vai-LOH-uh",
                }
            ],
            tts_prompt="ignored",
            intro="Let’s pick up right where we’ve been.",
            body="This is the update — and it matters.",
            outro="That’s the thread to watch tonight.",
            source_articles=[],
        )
    ]

    normalized = _normalize_radio_story_scripts(scripts, selected_stories)

    assert normalized[0].headline == "Minnesota let's roll"
    assert normalized[0].audio_profile == "Warm FM host - close and bright."
    assert normalized[0].scene == "Late-hour update... right in the listener's ear."
    assert normalized[0].intro == "Let's pick up right where we've been."
    assert normalized[0].body == "This is the update - and it matters."
    assert normalized[0].slug == "minnesota-let's-roll"
    assert normalized[0].pace == "Don't rush."
    assert normalized[0].warmth == "Stay warm and direct."
    assert normalized[0].must_hit == ["Minnesota headline"]
    assert normalized[0].pronunciations[0].term == "Tagovailoa"
    assert normalized[0].pronunciations[0].guide == "TAH-go-vai-LOH-uh"
    assert "\\u2019" not in normalized[0].tts_prompt
    assert "Pace: Don't rush." in normalized[0].tts_prompt
    assert "Warmth: Stay warm and direct." in normalized[0].tts_prompt
    assert "Must Hit: Minnesota headline" in normalized[0].tts_prompt
    assert "Pronunciations: Tagovailoa: TAH-go-vai-LOH-uh" in normalized[0].tts_prompt
    assert "Let's pick up right where we've been." in normalized[0].tts_prompt


def test_radio_story_script_normalization_rejects_persona_from_wrong_language_roster() -> None:
    selected_stories = [
        RadioStoryScriptInput(
            team="MIN",
            playlist_rank=1,
            headline="Minnesota headline",
            framing="new",
            continuity="new_story",
            production_reason="Lead",
            story_synopsis="Minnesota package Topic Angle",
            source_articles=[],
        )
    ]

    scripts = [
        make_radio_story_script(
            team="MIN",
            playlist_rank=1,
            language="de-DE",
            persona_name="Lena Torres",
        )
    ]

    with pytest.raises(ValueError, match="unknown persona"):
        _normalize_radio_story_scripts(scripts, selected_stories, language="de-DE")


def test_radio_story_script_normalization_extracts_structured_direction_from_director_notes() -> None:
    selected_stories = [
        RadioStoryScriptInput(
            team="NYJ",
            playlist_rank=1,
            headline="Jets shocker",
            framing="new",
            continuity="new_story",
            production_reason="Lead",
            story_synopsis="Jets shocker package",
            source_articles=[],
        )
    ]

    scripts = [
        RadioStoryScript(
            team="NYJ",
            playlist_rank=1,
            headline="Jets shocker",
            continuity="new_story",
            persona_name="Mason Reed",
            persona_backstory="Drive-time breaker with Northeast instincts.",
            persona_specialty="breaking news and urgency",
            voice_name="Charon",
            dialect="a light Mid-Atlantic edge",
            duration_seconds=180,
            slug="jets-shocker",
            audio_profile="Confident breaking-news sports anchor.",
            scene="Top-of-hour NFL update.",
            director_notes=(
                'Pace: urgent but controlled, no rambling. '
                'Warmth: engaged, speaking directly to Jets fans who are surprised by the move. '
                'Emphasis: punch "Geno Smith," "trade with the Raiders," "late-round pick swap," '
                'and "Raiders paying most of the salary." '
                'Intimacy: occasional second-person to pull the listener in, but keep it newsroom-sharp. '
                'Pronunciation: "Tom Pelissero" (PEH-lih-SARE-oh).'
            ),
            tts_prompt="ignored",
            intro="Intro",
            body="Body",
            outro="Outro",
            source_articles=[],
        )
    ]

    normalized = _normalize_radio_story_scripts(scripts, selected_stories)

    assert normalized[0].pace == "urgent but controlled, no rambling."
    assert "Jets fans" in normalized[0].warmth
    assert "occasional second-person" in normalized[0].warmth
    assert normalized[0].must_hit == [
        "Geno Smith",
        "trade with the Raiders",
        "late-round pick swap",
        "Raiders paying most of the salary",
    ]
    assert normalized[0].pronunciations[0].term == "Tom Pelissero"
    assert normalized[0].pronunciations[0].guide == "PEH-lih-SARE-oh"


def test_team_update_batch_request_normalizes_team_list() -> None:
    request = TeamUpdateBatchRequest(teams=["ari", "MIN", "ari"])

    assert request.teams == ["ARI", "MIN"]


def test_team_update_batch_agent_report_normalizes_full_team_name() -> None:
    report = TeamUpdateBatchAgentReport(
        team="Arizona Cardinals",
        lookback_minutes=180,
        status="no_update",
    )

    assert report.team == "ARI"


def test_team_update_workflow_builds_real_agent_instances(tmp_path: Path) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url)},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )

    assert workflow.article_lookup_tool.name == "lookup_article_content"
    assert workflow.article_data_tool.name == "digest_article_data"
    assert workflow.team_update_agent.name == "Team Update Agent"
    assert workflow.team_update_tool.name == "build_team_update_package"
    assert workflow.team_update_batch_agent.name == "Team Update Batch Agent"
    assert workflow.radio_script_writer_agent.name == "Radio Script Writer Agent"
    assert workflow.radio_story_script_tool.name == "build_radio_story_script"
    assert workflow.hourly_script_batch_agent.name == "Hourly Script Batch Agent"
    assert workflow.script_generation_stacks["de-DE"].radio_script_writer_agent.name == "Radio Script Writer Agent (de-DE)"
    assert workflow.script_generation_stacks["de-DE"].hourly_script_batch_agent.name == "Hourly Script Batch Agent (de-DE)"


def test_team_update_topic_requires_what_changed_for_updates() -> None:
    with pytest.raises(ValueError):
        make_topic(headline="Update topic", framing="update", continuity="follow_up_to_tracked_story")


def test_team_update_topic_requires_new_story_continuity_for_new_items() -> None:
    with pytest.raises(ValueError):
        make_topic(continuity="follow_up_to_tracked_story")


def test_team_update_report_ready_requires_report() -> None:
    with pytest.raises(ValueError):
        TeamUpdateReport(
            run_id="run-1",
            generated_at=datetime.now(UTC),
            team="MIN",
            lookback_minutes=60,
            status="report_ready",
            coverage={},
            warnings=[],
        )


def test_history_store_round_trips_report_ready_entries(tmp_path: Path) -> None:
    store = TeamUpdateHistoryStore(tmp_path / "history.sqlite3")
    report_ready = TeamUpdateReport(
        run_id="run-1",
        generated_at=datetime.now(UTC),
        team="MIN",
        lookback_minutes=60,
        status="report_ready",
        report=TeamUpdatePackage(
            total_duration_seconds=180,
            headline="Headline",
            topics=[make_topic()],
            source_articles=[],
        ),
        coverage={},
        warnings=[],
    )
    no_update = TeamUpdateReport(
        run_id="run-2",
        generated_at=datetime.now(UTC),
        team="MIN",
        lookback_minutes=60,
        status="no_update",
        coverage={},
        warnings=[],
    )

    store.save_report(
        report=report_ready,
        batch_run_id="batch-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
    )
    store.save_report(
        report=no_update,
        batch_run_id="batch-1",
        source_story_ids=[],
        source_urls=[],
        source_group_ids=[],
    )

    entries = store.list_recent_report_ready_entries(
        team_code="MIN",
        generated_after=datetime.now(UTC) - timedelta(days=1),
    )

    assert len(entries) == 1
    assert entries[0].source_group_ids == ["group-1"]


def test_history_store_round_trips_production_metadata_and_playlist_history(tmp_path: Path) -> None:
    store = TeamUpdateHistoryStore(tmp_path / "history.sqlite3")
    report_ready = TeamUpdateReport(
        run_id="run-1",
        generated_at=datetime.now(UTC),
        team="MIN",
        lookback_minutes=60,
        status="report_ready",
        report=TeamUpdatePackage(
            total_duration_seconds=180,
            headline="Headline",
            topics=[make_topic()],
            source_articles=[],
        ),
        coverage={},
        warnings=[],
    )

    store.save_report(
        report=report_ready,
        batch_run_id="batch-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id="playlist-1",
    )
    saved_playlist_id = store.save_hourly_playlist(
        batch_run_id="batch-1",
        generated_at=report_ready.generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Headline",
                    framing="new",
                    continuity="new_story",
                    production_reason="Top story.",
                    source_articles=[],
                )
            ],
        ),
    )

    entries = store.list_recent_report_ready_entries(
        team_code="MIN",
        generated_after=datetime.now(UTC) - timedelta(days=1),
    )
    playlists = store.list_hourly_playlists()

    assert entries[0].production_status == "put_to_production"
    assert entries[0].production_rank == 1
    assert entries[0].playlist_id == "playlist-1"
    assert playlists[0].id == saved_playlist_id
    assert playlists[0].selected_count == 1


def test_history_store_returns_latest_playlist_and_persists_story_scripts(tmp_path: Path) -> None:
    store = TeamUpdateHistoryStore(tmp_path / "history.sqlite3")
    older_generated_at = datetime.now(UTC) - timedelta(hours=1)
    newer_generated_at = datetime.now(UTC)
    older_playlist_id = store.save_hourly_playlist(
        batch_run_id="batch-older",
        generated_at=older_generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(selected_count=0, items=[]),
    )
    newer_playlist_id = store.save_hourly_playlist(
        batch_run_id="batch-newer",
        generated_at=newer_generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Headline",
                    framing="new",
                    continuity="new_story",
                    production_reason="Top story.",
                    source_articles=[],
                )
            ],
        ),
    )

    store.save_story_scripts(
        script_run_id="script-run-1",
        playlist_id=newer_playlist_id,
        batch_run_id="batch-newer",
        generated_at=newer_generated_at,
        scripts=[make_radio_story_script()],
    )

    latest_playlist = store.get_latest_hourly_playlist()
    explicit_playlist = store.get_hourly_playlist(playlist_id=older_playlist_id)
    scripts = store.list_story_scripts(playlist_id=newer_playlist_id)

    assert latest_playlist is not None
    assert latest_playlist.id == newer_playlist_id
    assert explicit_playlist is not None
    assert explicit_playlist.id == older_playlist_id
    assert scripts[0].script_run_id == "script-run-1"
    assert scripts[0].playlist_id == newer_playlist_id
    assert scripts[0].language == "en-US"


def test_history_store_reads_legacy_story_scripts_as_en_us(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE hourly_story_script_history (
                id TEXT PRIMARY KEY,
                script_run_id TEXT NOT NULL,
                playlist_id TEXT NOT NULL,
                batch_run_id TEXT NOT NULL,
                team_code TEXT NOT NULL,
                playlist_rank INTEGER NOT NULL,
                generated_at TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                script_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO hourly_story_script_history (
                id,
                script_run_id,
                playlist_id,
                batch_run_id,
                team_code,
                playlist_rank,
                generated_at,
                duration_seconds,
                script_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-script-1",
                "script-run-legacy",
                "playlist-legacy",
                "batch-legacy",
                "MIN",
                1,
                datetime.now(UTC).isoformat(),
                180,
                json.dumps(make_radio_story_script().model_dump(mode="json")),
            ),
        )

    store = TeamUpdateHistoryStore(db_path)
    scripts = store.list_story_scripts(script_run_id="script-run-legacy")

    assert scripts[0].language == "en-US"


def test_build_hourly_script_batch_input_includes_language_and_selected_personas(tmp_path: Path) -> None:
    store = TeamUpdateHistoryStore(tmp_path / "history.sqlite3")
    generated_at = datetime.now(UTC)
    playlist_id = store.save_hourly_playlist(
        batch_run_id="batch-run-1",
        generated_at=generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Headline",
                    framing="new",
                    continuity="new_story",
                    production_reason="Lead story.",
                    source_articles=[],
                )
            ],
        ),
    )
    playlist = store.get_hourly_playlist(playlist_id=playlist_id)
    assert playlist is not None

    payload = build_hourly_script_batch_input(
        playlist,
        [
            RadioStoryScriptInput(
                team="MIN",
                playlist_rank=1,
                headline="Headline",
                framing="new",
                continuity="new_story",
                production_reason="Lead story.",
                story_synopsis="Headline angle",
                source_articles=[],
            )
        ],
        language="de-DE",
        personas=[
            {
                "name": "Mika Brandt",
                "specialty": "breaking news, urgency, and franchise-defining developments",
                "voice_name": "Charon",
                "dialect": "ein leichter norddeutscher Nachrichtenrhythmus mit klarer Breaking-News-Schaerfe",
                "backstory": "unused in batch payload",
            }
        ],
    )

    assert '"language": "de-DE"' in payload
    assert '"name": "Mika Brandt"' in payload
    assert '"voice_name": "Charon"' in payload


@pytest.mark.asyncio
async def test_team_update_workflow_returns_no_update_when_team_has_no_feed_stories(
    tmp_path: Path,
) -> None:
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
    )

    report = await workflow.run_team_update(build_context())

    assert report.status == "no_update"
    assert report.report is None
    assert report.coverage.candidate_stories == 0


@pytest.mark.asyncio
async def test_team_update_workflow_returns_no_update_when_group_has_no_recent_changes(
    tmp_path: Path,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-1")},
        group_update_responses={
            "group-1": StoryGroupUpdatesToolResponse(group_id="group-1", lookback_minutes=60, updates=[])
        },
    )

    report = await workflow.run_team_update(build_context())

    assert report.status == "no_update"
    assert report.coverage.team_feed_stories == 1
    assert report.coverage.candidate_stories == 0


@pytest.mark.asyncio
async def test_team_update_workflow_falls_back_to_article_updated_at_when_group_updates_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={
            story.url: make_article_lookup_response(
                url=story.url,
                group_id="group-1",
                updated_at=datetime.now(UTC),
            )
        },
        group_update_responses={
            "group-1": StoryGroupUpdatesToolResponse(group_id="group-1", lookback_minutes=60, updates=[])
        },
    )

    async def fake_runner(agent, message, **_kwargs):
        assert story.url in message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Fallback headline",
                topics=[
                    make_topic(
                        headline="Fallback topic",
                        source_articles=[
                            SourceArticleRef(
                                story_id=story.id,
                                url=story.url,
                                title=story.title,
                                source_name=story.source_name,
                            )
                        ],
                    )
                ],
                source_articles=[
                    SourceArticleRef(
                        story_id=story.id,
                        url=story.url,
                        title=story.title,
                        source_name=story.source_name,
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.status == "report_ready"
    assert any("Falling back to article updated_at" in warning for warning in report.warnings)


@pytest.mark.asyncio
async def test_team_update_workflow_marks_new_candidate_without_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-1")},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )
    captured_message = ""

    async def fake_runner(agent, message, **_kwargs):
        nonlocal captured_message
        captured_message = message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Minnesota update",
                topics=[
                    make_topic(
                        headline="Roster update",
                        editorial_angle="Fresh angle",
                        source_articles=[
                            SourceArticleRef(
                                story_id=story.id,
                                url=story.url,
                                title=story.title,
                                source_name=story.source_name,
                            )
                        ],
                    )
                ],
                source_articles=[
                    SourceArticleRef(
                        story_id=story.id,
                        url=story.url,
                        title=story.title,
                        source_name=story.source_name,
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.status == "report_ready"
    assert report.report is not None
    assert report.coverage.new_candidates == 1
    assert '"framing": "new"' in captured_message


@pytest.mark.asyncio
async def test_team_update_workflow_marks_update_candidate_from_prior_group_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-1")},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )
    prior_report = TeamUpdateReport(
        run_id="prior",
        generated_at=datetime.now(UTC) - timedelta(hours=2),
        team="MIN",
        lookback_minutes=60,
        status="report_ready",
        report=TeamUpdatePackage(
            total_duration_seconds=180,
            headline="Old headline",
            topics=[make_topic(headline="Old topic")],
            source_articles=[],
        ),
        coverage={},
        warnings=[],
    )
    workflow._history_store.save_report(
        report=prior_report,
        batch_run_id="prior-batch",
        source_story_ids=["old-story"],
        source_urls=["https://example.com/old-story"],
        source_group_ids=["group-1"],
    )
    captured_message = ""

    async def fake_runner(agent, message, **_kwargs):
        nonlocal captured_message
        captured_message = message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Updated headline",
                topics=[
                    make_topic(
                        headline="Updated topic",
                        framing="update",
                        continuity="follow_up_to_tracked_story",
                        what_changed="A related story was added to the same group in the last hour.",
                        source_articles=[
                            SourceArticleRef(
                                story_id=story.id,
                                url=story.url,
                                title=story.title,
                                source_name=story.source_name,
                            )
                        ],
                    )
                ],
                source_articles=[
                    SourceArticleRef(
                        story_id=story.id,
                        url=story.url,
                        title=story.title,
                        source_name=story.source_name,
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.status == "report_ready"
    assert report.coverage.update_candidates == 1
    assert '"framing": "update"' in captured_message


@pytest.mark.asyncio
async def test_team_update_workflow_marks_update_candidate_from_prior_url_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story(url="https://example.com/repeat")
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-2")},
        group_update_responses={"group-2": make_group_updates_response(group_id="group-2")},
    )
    prior_report = TeamUpdateReport(
        run_id="prior",
        generated_at=datetime.now(UTC) - timedelta(hours=2),
        team="MIN",
        lookback_minutes=60,
        status="report_ready",
        report=TeamUpdatePackage(
            total_duration_seconds=180,
            headline="Old headline",
            topics=[make_topic(headline="Old topic")],
            source_articles=[],
        ),
        coverage={},
        warnings=[],
    )
    workflow._history_store.save_report(
        report=prior_report,
        batch_run_id="prior-batch",
        source_story_ids=["other-story"],
        source_urls=[story.url],
        source_group_ids=["unrelated-group"],
    )

    async def fake_runner(agent, message, **_kwargs):
        assert '"matched_by": "url"' in message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Repeated URL headline",
                topics=[
                    make_topic(
                        headline="Repeated URL topic",
                        framing="update",
                        continuity="follow_up_to_tracked_story",
                        what_changed="The same source URL has new same-hour related activity.",
                        source_articles=[],
                    )
                ],
                source_articles=[],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.coverage.update_candidates == 1


@pytest.mark.asyncio
async def test_team_update_workflow_marks_update_candidate_from_produced_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-1")},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )
    prior_report = TeamUpdateReport(
        run_id="prior",
        generated_at=datetime.now(UTC) - timedelta(hours=2),
        team="MIN",
        lookback_minutes=60,
        status="report_ready",
        report=TeamUpdatePackage(
            total_duration_seconds=180,
            headline="Produced headline",
            topics=[make_topic(headline="Produced topic")],
            source_articles=[],
        ),
        coverage={},
        warnings=[],
    )
    workflow._history_store.save_report(
        report=prior_report,
        batch_run_id="prior-batch",
        source_story_ids=["old-story"],
        source_urls=["https://example.com/old-story"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id="playlist-1",
    )

    async def fake_runner(agent, message, **_kwargs):
        assert '"continuity": "follow_up_to_produced_story"' in message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Produced follow-up",
                topics=[
                    make_topic(
                        headline="Produced update topic",
                        framing="update",
                        continuity="follow_up_to_produced_story",
                        what_changed="New same-group reporting advances an already aired storyline.",
                    )
                ],
                source_articles=[],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.status == "report_ready"
    assert report.report is not None
    assert report.report.topics[0].continuity == "follow_up_to_produced_story"


@pytest.mark.asyncio
async def test_team_update_workflow_filters_to_requested_team_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    min_story = make_story(story_id="story-1", team_code="MIN")
    atl_story = FeedStory(
        id="story-2",
        url="https://example.com/story-2",
        title="Falcons story",
        source_name="Yahoo",
        category="Breaking News",
        facts_count=2,
        entities=[{"entity_type": "team", "entity_id": "ATL", "matched_name": "Atlanta Falcons"}],
    )
    workflow = build_workflow(
        tmp_path,
        stories=[min_story, atl_story],
        article_lookup_responses={min_story.url: make_article_lookup_response(url=min_story.url, group_id="group-1")},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )

    async def fake_runner(agent, message, **_kwargs):
        assert "Falcons story" not in message
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Team-only headline",
                topics=[make_topic()],
                source_articles=[],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    report = await workflow.run_team_update(build_context())

    assert report.coverage.team_feed_stories == 1


@pytest.mark.asyncio
async def test_team_update_end_to_end_persists_then_frames_followup_as_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    settings = build_settings(tmp_path)
    history_store = TeamUpdateHistoryStore(settings.team_update_history_sqlite_path)
    news_feed = FakeNewsFeedAdapter([story])
    article_lookup = FakeArticleLookupAdapter(
        {story.url: make_article_lookup_response(url=story.url, group_id="group-1")}
    )
    group_updates = FakeStoryGroupUpdatesAdapter({"group-1": make_group_updates_response(group_id="group-1")})
    workflow = TeamUpdateWorkflow(
        settings=settings,
        news_feed=news_feed,
        article_lookup=article_lookup,
        story_group_updates=group_updates,
        history_store=history_store,
        tts_batch=FakeGeminiTTSBatchAdapter(),
    )
    framings: list[str] = []

    async def fake_runner(agent, message, **_kwargs):
        framing = "update" if '"framing": "update"' in message else "new"
        framings.append(framing)
        return SimpleNamespace(
            final_output=TeamUpdatePackage(
                total_duration_seconds=180,
                headline=f"{framing} headline",
                topics=[
                    make_topic(
                        headline=f"{framing} topic",
                        framing=framing,
                        continuity=(
                            "follow_up_to_tracked_story" if framing == "update" else "new_story"
                        ),
                        what_changed=(
                            "A same-group follow-up item arrived in the last hour."
                            if framing == "update"
                            else None
                        ),
                        source_articles=[
                            SourceArticleRef(
                                story_id=story.id,
                                url=story.url,
                                title=story.title,
                                source_name=story.source_name,
                            )
                        ],
                    )
                ],
                source_articles=[
                    SourceArticleRef(
                        story_id=story.id,
                        url=story.url,
                        title=story.title,
                        source_name=story.source_name,
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    first_report = await workflow.run_team_update(build_context())
    second_report = await workflow.run_team_update(build_context())

    assert first_report.status == "report_ready"
    assert second_report.status == "report_ready"
    assert framings == ["new", "update"]


@pytest.mark.asyncio
async def test_team_update_batch_uses_single_top_level_runner_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    story = make_story()
    workflow = build_workflow(
        tmp_path,
        stories=[story],
        article_lookup_responses={story.url: make_article_lookup_response(url=story.url, group_id="group-1")},
        group_update_responses={"group-1": make_group_updates_response(group_id="group-1")},
    )
    calls: list[tuple[str, str | None]] = []

    async def fake_runner(agent, message, *, run_config=None, **_kwargs):
        calls.append((agent.name, getattr(run_config, "group_id", None)))
        if agent.name == "Team Update Batch Agent":
            assert "selected_teams" in message
            return SimpleNamespace(
                final_output=TeamUpdateBatchAgentResult(
                    reports=[
                        TeamUpdateBatchAgentReport(
                            team="MIN",
                            lookback_minutes=60,
                            status="report_ready",
                            report=TeamUpdatePackage(
                                total_duration_seconds=180,
                                headline="Batch headline",
                                topics=[
                                    make_topic(
                                        headline="Batch topic",
                                        source_articles=[
                                            SourceArticleRef(
                                                story_id=story.id,
                                                url=story.url,
                                                title=story.title,
                                                source_name=story.source_name,
                                            )
                                        ],
                                    )
                                ],
                                source_articles=[
                                    SourceArticleRef(
                                        story_id=story.id,
                                        url=story.url,
                                        title=story.title,
                                        source_name=story.source_name,
                                    )
                                ],
                            ),
                        )
                    ]
                )
            )
        assert agent.name == "Hourly Playlist Orchestrator Agent"
        return SimpleNamespace(
            final_output=HourlyPlaylist(
                selected_count=1,
                items=[
                    HourlyPlaylistItem(
                        rank=1,
                        team="MIN",
                        headline="Batch headline",
                        framing="new",
                        continuity="new_story",
                        production_reason="Strongest ready report in the hour.",
                        source_articles=[],
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    batch_response = await workflow.run_team_update_batch(
        run_id="batch-run-1",
        generated_at=datetime.now(UTC),
        request=TeamUpdateBatchRequest(teams=["MIN"], lookback_minutes=60),
    )

    assert len(batch_response.reports) == 1
    assert batch_response.hourly_playlist.selected_count == 1
    assert calls == [
        ("Team Update Batch Agent", "batch-run-1"),
        ("Hourly Playlist Orchestrator Agent", "batch-run-1"),
    ]


@pytest.mark.asyncio
async def test_team_update_batch_persists_all_reports_and_marks_playlist_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    min_story = make_story(team_code="MIN")
    atl_story = make_story(
        story_id="story-2",
        url="https://example.com/story-2",
        title="Falcons update",
        team_code="ATL",
    )
    workflow = build_workflow(
        tmp_path,
        stories=[min_story, atl_story],
        article_lookup_responses={
            min_story.url: make_article_lookup_response(url=min_story.url, group_id="group-1"),
            atl_story.url: make_article_lookup_response(url=atl_story.url, group_id="group-2"),
        },
        group_update_responses={
            "group-1": make_group_updates_response(group_id="group-1"),
            "group-2": StoryGroupUpdatesToolResponse(group_id="group-2", lookback_minutes=60, updates=[]),
        },
    )

    async def fake_runner(agent, message, **_kwargs):
        if agent.name == "Team Update Batch Agent":
            return SimpleNamespace(
                final_output=TeamUpdateBatchAgentResult(
                    reports=[
                        TeamUpdateBatchAgentReport(
                            team="MIN",
                            lookback_minutes=60,
                            status="report_ready",
                            report=TeamUpdatePackage(
                                total_duration_seconds=180,
                                headline="MIN package",
                                topics=[make_topic(headline="MIN topic")],
                                source_articles=[
                                    SourceArticleRef(
                                        story_id=min_story.id,
                                        url=min_story.url,
                                        title=min_story.title,
                                        source_name=min_story.source_name,
                                    )
                                ],
                            ),
                        ),
                        TeamUpdateBatchAgentReport(
                            team="ATL",
                            lookback_minutes=60,
                            status="no_update",
                        ),
                    ]
                )
            )

        assert '"team": "MIN"' in message
        assert '"team": "ATL"' not in message
        assert '"eligible_reports"' in message
        return SimpleNamespace(
            final_output=HourlyPlaylist(
                selected_count=1,
                items=[
                    HourlyPlaylistItem(
                        rank=1,
                        team="MIN",
                        headline="MIN package",
                        framing="new",
                        continuity="new_story",
                        production_reason="Best ready report in the batch.",
                        source_articles=[],
                    )
                ],
            )
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    batch_response = await workflow.run_team_update_batch(
        run_id="batch-run-1",
        generated_at=datetime.now(UTC),
        request=TeamUpdateBatchRequest(teams=["MIN", "ATL"], lookback_minutes=60),
    )

    stored_entries = workflow._history_store.list_batch_entries(batch_run_id="batch-run-1")
    stored_by_team = {entry.team_code: entry for entry in stored_entries}
    playlists = workflow._history_store.list_hourly_playlists()

    assert batch_response.hourly_playlist.selected_count == 1
    assert len(stored_entries) == 2
    assert stored_by_team["MIN"].production_status == "put_to_production"
    assert stored_by_team["MIN"].production_rank == 1
    assert stored_by_team["ATL"].production_status == "tracked_only"
    assert playlists[0].selected_count == 1


@pytest.mark.asyncio
async def test_hourly_playlist_scripts_use_latest_saved_playlist_and_persist_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_batch = FakeGeminiTTSBatchAdapter()
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
        tts_batch=tts_batch,
    )
    generated_at = datetime.now(UTC)
    playlist_id = workflow._history_store.save_hourly_playlist(
        batch_run_id="batch-run-1",
        generated_at=generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Playlist headline",
                    framing="update",
                    continuity="follow_up_to_produced_story",
                    production_reason="Strong lead for the hour.",
                    source_articles=[
                        SourceArticleRef(
                            story_id="story-1",
                            url="https://example.com/story-1",
                            title="Title",
                            source_name="ESPN",
                        )
                    ],
                )
            ],
        ),
    )
    workflow._history_store.save_report(
        report=TeamUpdateReport(
            run_id="batch-run-1",
            generated_at=generated_at,
            team="MIN",
            lookback_minutes=60,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Playlist headline",
                topics=[
                    make_topic(
                        headline="Topic",
                        framing="update",
                        continuity="follow_up_to_produced_story",
                        what_changed="New same-group development.",
                        source_articles=[
                            SourceArticleRef(
                                story_id="story-1",
                                url="https://example.com/story-1",
                                title="Title",
                                source_name="ESPN",
                            )
                        ],
                    )
                ],
                source_articles=[],
            ),
            coverage={},
            warnings=[],
        ),
        batch_run_id="batch-run-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id=playlist_id,
    )

    async def fake_runner(agent, message, **_kwargs):
        assert agent.name == "Hourly Script Batch Agent"
        assert '"playlist_id": "' in message
        assert '"persona_roster": [' in message
        assert '"voice_name": "Charon"' in message
        assert '"continuity": "follow_up_to_produced_story"' in message
        assert '"production_reason": "Strong lead for the hour."' in message
        return SimpleNamespace(
            final_output={
                "scripts": [
                    make_radio_story_script(
                        team="MIN",
                        playlist_rank=1,
                        continuity="follow_up_to_produced_story",
                    ).model_dump(mode="json")
                ]
            }
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    response = await workflow.run_hourly_playlist_scripts(
        script_run_id="script-run-1",
        generated_at=generated_at,
        request=HourlyPlaylistScriptsRequest(),
    )

    saved_scripts = workflow._history_store.list_story_scripts(script_run_id="script-run-1")

    assert response.playlist_id == playlist_id
    assert len(response.scripts) == 1
    assert response.language == "en-US"
    assert response.scripts[0].continuity == "follow_up_to_produced_story"
    assert response.scripts[0].language == "en-US"
    assert response.tts_batch is not None
    assert response.tts_batch.items[0].public_url == "https://example.com/min-headline.mp3"
    assert saved_scripts[0].playlist_id == playlist_id
    assert saved_scripts[0].script_run_id == "script-run-1"
    assert saved_scripts[0].language == "en-US"
    assert tts_batch.create_calls[0]["items"][0]["id"] == "min-headline"
    assert tts_batch.create_calls[0]["items"][0]["voice_name"] == "Puck"
    assert tts_batch.create_calls[0]["items"][0]["script"]["body"] == "Body"
    assert tts_batch.status_calls[0]["batch_id"] == "batches/abc123"
    assert tts_batch.process_calls[0]["supabase"]["bucket"] == "audio"


@pytest.mark.asyncio
async def test_hourly_playlist_scripts_support_de_de_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_batch = FakeGeminiTTSBatchAdapter(
        process_response={
            "batch_id": "batches/de123",
            "status": "JOB_STATE_SUCCEEDED",
            "processed_count": 1,
            "failed_count": 0,
            "token_usage": {
                "input_tokens": 11,
                "cached_input_tokens": 0,
                "output_tokens": 21,
                "total_tokens": 32,
                "reported_item_count": 1,
            },
            "local_usage_summary_file": "/tmp/usage-de.json",
            "usage_summary_path": "gemini-tts-batch/batches/de123/usage_summary.json",
            "usage_summary_public_url": "https://example.com/usage-de.json",
            "manifest_path": "gemini-tts-batch/batches/de123/manifest.json",
            "manifest_public_url": "https://example.com/manifest-de.json",
            "items": [
                {
                    "id": "min-headline",
                    "storage_path": "gemini-tts-batch/batches/de123/min-headline.mp3",
                    "mime_type": "audio/mpeg",
                    "source_mime_type": "audio/wav",
                    "public_url": "https://example.com/min-headline-de.mp3",
                    "token_usage": {
                        "input_tokens": 11,
                        "cached_input_tokens": 0,
                        "output_tokens": 21,
                        "total_tokens": 32,
                    },
                }
            ],
            "failures": [],
        }
    )
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
        tts_batch=tts_batch,
    )
    generated_at = datetime.now(UTC)
    playlist_id = workflow._history_store.save_hourly_playlist(
        batch_run_id="batch-run-1",
        generated_at=generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Playlist headline",
                    framing="update",
                    continuity="follow_up_to_produced_story",
                    production_reason="Strong lead for the hour.",
                    source_articles=[],
                )
            ],
        ),
    )
    workflow._history_store.save_report(
        report=TeamUpdateReport(
            run_id="batch-run-1",
            generated_at=generated_at,
            team="MIN",
            lookback_minutes=60,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Playlist headline",
                topics=[
                    make_topic(
                        headline="Thema",
                        framing="update",
                        continuity="follow_up_to_produced_story",
                        what_changed="Neue Entwicklung.",
                        source_articles=[],
                    )
                ],
                source_articles=[],
            ),
            coverage={},
            warnings=[],
        ),
        batch_run_id="batch-run-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id=playlist_id,
    )

    async def fake_runner(agent, message, **_kwargs):
        assert agent.name == "Hourly Script Batch Agent (de-DE)"
        assert '"language": "de-DE"' in message
        assert '"name": "Mika Brandt"' in message
        return SimpleNamespace(
            final_output={
                "scripts": [
                    make_radio_story_script(
                        team="MIN",
                        playlist_rank=1,
                        continuity="follow_up_to_produced_story",
                        language="de-DE",
                        persona_name="Mika Brandt",
                        persona_backstory=(
                            "Mika hat sich im deutschen Sportfunk damit einen Namen gemacht."
                        ),
                        persona_specialty=(
                            "breaking news, urgency, and franchise-defining developments"
                        ),
                        voice_name="Charon",
                        dialect=(
                            "ein leichter norddeutscher Nachrichtenrhythmus mit klarer Breaking-News-Schaerfe"
                        ),
                    ).model_dump(mode="json")
                ]
            }
        )

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    response = await workflow.run_hourly_playlist_scripts(
        script_run_id="script-run-de",
        generated_at=generated_at,
        request=HourlyPlaylistScriptsRequest(language="de-DE"),
    )

    saved_scripts = workflow._history_store.list_story_scripts(script_run_id="script-run-de")

    assert response.playlist_id == playlist_id
    assert response.language == "de-DE"
    assert response.scripts[0].language == "de-DE"
    assert response.scripts[0].persona_name == "Mika Brandt"
    assert response.tts_batch is not None
    assert response.tts_batch.items[0].public_url == "https://example.com/min-headline-de.mp3"
    assert saved_scripts[0].language == "de-DE"


@pytest.mark.asyncio
async def test_hourly_playlist_scripts_skip_tts_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_batch = FakeGeminiTTSBatchAdapter()
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
        tts_batch=tts_batch,
    )
    generated_at = datetime.now(UTC)
    playlist_id = workflow._history_store.save_hourly_playlist(
        batch_run_id="batch-run-1",
        generated_at=generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Playlist headline",
                    framing="new",
                    continuity="new_story",
                    production_reason="Lead item.",
                    source_articles=[],
                )
            ],
        ),
    )
    workflow._history_store.save_report(
        report=TeamUpdateReport(
            run_id="batch-run-1",
            generated_at=generated_at,
            team="MIN",
            lookback_minutes=60,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Playlist headline",
                topics=[make_topic()],
                source_articles=[],
            ),
            coverage={},
            warnings=[],
        ),
        batch_run_id="batch-run-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id=playlist_id,
    )

    async def fake_runner(_agent, _message, **_kwargs):
        return SimpleNamespace(final_output={"scripts": [make_radio_story_script().model_dump(mode="json")]})

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    response = await workflow.run_hourly_playlist_scripts(
        script_run_id="script-run-no-tts",
        generated_at=generated_at,
        request=HourlyPlaylistScriptsRequest(enable_tts=False),
    )

    assert response.tts_batch is None
    assert tts_batch.create_calls == []
    assert tts_batch.status_calls == []
    assert tts_batch.process_calls == []


@pytest.mark.asyncio
async def test_hourly_playlist_scripts_raise_on_terminal_tts_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_batch = FakeGeminiTTSBatchAdapter(
        status_responses=[
            {
                "batch_id": "batches/failed",
                "status": "JOB_STATE_FAILED",
                "created_at": "2026-03-11T10:00:00Z",
                "updated_at": "2026-03-11T10:05:00Z",
                "output_file_id": None,
                "error_file_id": "files/error",
                "error": "upstream failure",
            }
        ]
    )
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
        tts_batch=tts_batch,
    )
    generated_at = datetime.now(UTC)
    playlist_id = workflow._history_store.save_hourly_playlist(
        batch_run_id="batch-run-1",
        generated_at=generated_at,
        lookback_minutes=60,
        playlist=HourlyPlaylist(
            selected_count=1,
            items=[
                HourlyPlaylistItem(
                    rank=1,
                    team="MIN",
                    headline="Playlist headline",
                    framing="new",
                    continuity="new_story",
                    production_reason="Lead item.",
                    source_articles=[],
                )
            ],
        ),
    )
    workflow._history_store.save_report(
        report=TeamUpdateReport(
            run_id="batch-run-1",
            generated_at=generated_at,
            team="MIN",
            lookback_minutes=60,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Playlist headline",
                topics=[make_topic()],
                source_articles=[],
            ),
            coverage={},
            warnings=[],
        ),
        batch_run_id="batch-run-1",
        source_story_ids=["story-1"],
        source_urls=["https://example.com/story-1"],
        source_group_ids=["group-1"],
        production_status="put_to_production",
        production_rank=1,
        playlist_id=playlist_id,
    )

    async def fake_runner(_agent, _message, **_kwargs):
        return SimpleNamespace(final_output={"scripts": [make_radio_story_script().model_dump(mode="json")]})

    monkeypatch.setattr("app.newsroom.workflow_team_update.Runner.run", fake_runner)

    with pytest.raises(ValueError, match="Final status: JOB_STATE_FAILED"):
        await workflow.run_hourly_playlist_scripts(
            script_run_id="script-run-failed-tts",
            generated_at=generated_at,
            request=HourlyPlaylistScriptsRequest(),
        )


@pytest.mark.asyncio
async def test_hourly_playlist_scripts_raise_when_no_saved_playlist_exists(
    tmp_path: Path,
) -> None:
    workflow = build_workflow(
        tmp_path,
        stories=[],
        article_lookup_responses={},
        group_update_responses={},
    )

    with pytest.raises(ValueError, match="No saved hourly playlist"):
        await workflow.run_hourly_playlist_scripts(
            script_run_id="script-run-1",
            generated_at=datetime.now(UTC),
            request=HourlyPlaylistScriptsRequest(),
        )


def test_team_update_cli_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOrchestrator:
        async def run_team_update_reports(self, request):
            teams = request.teams or ["ARI", "MIN"]
            return TeamUpdateBatchResponse(
                run_id="batch-run-1",
                generated_at=datetime.now(UTC),
                lookback_minutes=request.lookback_minutes,
                reports=[
                    make_team_update_report(team=team, lookback_minutes=request.lookback_minutes)
                    for team in teams
                ],
                hourly_playlist=HourlyPlaylist(selected_count=0, items=[]),
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["team-update", "--team", "MIN"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reports"][0]["team"] == "MIN"
    assert payload["reports"][0]["status"] == "no_update"
    assert payload["hourly_playlist"]["selected_count"] == 0


def test_team_update_cli_command_accepts_bracketed_team_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_teams: list[str] = []

    class FakeOrchestrator:
        async def run_team_update_reports(self, request):
            nonlocal captured_teams
            captured_teams = request.teams or []
            return TeamUpdateBatchResponse(
                run_id="batch-run-1",
                generated_at=datetime.now(UTC),
                lookback_minutes=request.lookback_minutes,
                reports=[],
                hourly_playlist=HourlyPlaylist(selected_count=0, items=[]),
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["team-update", "--team", "[NJY, ARI, BAL]"])

    assert result.exit_code == 0
    assert captured_teams == ["NYJ", "ARI", "BAL"]


def test_team_update_cli_command_defaults_to_all_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_teams: list[str] = []

    class FakeOrchestrator:
        async def run_team_update_reports(self, request):
            nonlocal captured_teams
            captured_teams = request.teams or []
            return TeamUpdateBatchResponse(
                run_id="batch-run-1",
                generated_at=datetime.now(UTC),
                lookback_minutes=request.lookback_minutes,
                reports=[],
                hourly_playlist=HourlyPlaylist(selected_count=0, items=[]),
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["team-update"])

    assert result.exit_code == 0
    assert len(captured_teams) == 32


def test_team_update_cli_command_prints_batch_response_for_multiple_teams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrchestrator:
        async def run_team_update_reports(self, request):
            teams = request.teams or ["ARI", "MIN"]
            return TeamUpdateBatchResponse(
                run_id="batch-run-1",
                generated_at=datetime.now(UTC),
                lookback_minutes=request.lookback_minutes,
                reports=[
                    make_team_update_report(team=team, lookback_minutes=request.lookback_minutes)
                    for team in teams
                ],
                hourly_playlist=HourlyPlaylist(
                    selected_count=1,
                    items=[
                        HourlyPlaylistItem(
                            rank=1,
                            team="ARI",
                            headline="Arizona lead",
                            framing="new",
                            continuity="new_story",
                            production_reason="Lead item.",
                            source_articles=[],
                        )
                    ],
                ),
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["team-update", "--team", "ARI", "--team", "MIN"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [item["team"] for item in payload["reports"]] == ["ARI", "MIN"]
    assert payload["hourly_playlist"]["items"][0]["team"] == "ARI"


def test_write_scripts_cli_command_uses_latest_playlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrchestrator:
        async def run_hourly_playlist_scripts(self, request):
            assert request.playlist_id is None
            assert request.language == "en-US"
            assert request.enable_tts is True
            return HourlyPlaylistScriptsResponse(
                language=request.language,
                voice_name="Puck",
                items=[
                    {
                        "id": "min-headline",
                        "title": "MIN headline",
                        "direction": {
                            "audio_profile": "Warm FM host with bright energy and close-mic intimacy.",
                            "scene": "Late-hour NFL update hitting a listener on the drive home.",
                            "director_notes": "Stay warm, direct, and vivid. Build quick connection.",
                            "pace": "Urgent but controlled.",
                            "warmth": "Engaged and direct.",
                            "must_hit": ["Headline", "Key fact"],
                            "pronunciations": [
                                {
                                    "term": "Tagovailoa",
                                    "guide": "TAH-go-vai-LOH-uh",
                                }
                            ],
                        },
                        "script": {
                            "intro": "Intro",
                            "body": "Body",
                            "outro": "Outro",
                        },
                    }
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
                            "id": "min-headline",
                            "storage_path": "gemini-tts-batch/batches/abc123/min-headline.mp3",
                            "mime_type": "audio/mpeg",
                            "source_mime_type": "audio/wav",
                            "public_url": "https://example.com/min-headline.mp3",
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

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["write-scripts"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["language"] == "en-US"
    assert payload["voice_name"] == "Puck"
    assert payload["items"][0]["id"] == "min-headline"
    assert payload["items"][0]["title"] == "MIN headline"
    assert payload["items"][0]["direction"]["must_hit"] == ["Headline", "Key fact"]
    assert payload["tts_batch"]["items"][0]["public_url"] == "https://example.com/min-headline.mp3"


def test_write_scripts_cli_command_accepts_playlist_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrchestrator:
        async def run_hourly_playlist_scripts(self, request):
            assert request.playlist_id == "playlist-123"
            return HourlyPlaylistScriptsResponse(
                language=request.language,
                voice_name="",
                items=[],
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["write-scripts", "--playlist-id", "playlist-123"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["language"] == "en-US"
    assert payload["voice_name"] == ""
    assert payload["items"] == []
    assert payload["tts_batch"] is None


def test_write_scripts_cli_command_accepts_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrchestrator:
        async def run_hourly_playlist_scripts(self, request):
            assert request.language == "de-DE"
            return HourlyPlaylistScriptsResponse(
                language=request.language,
                voice_name="Charon",
                items=[],
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["write-scripts", "--language", "de-DE"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["language"] == "de-DE"


def test_write_scripts_cli_command_accepts_no_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOrchestrator:
        async def run_hourly_playlist_scripts(self, request):
            assert request.enable_tts is False
            return HourlyPlaylistScriptsResponse(
                language=request.language,
                voice_name="",
                items=[],
                tts_batch=None,
            )

    monkeypatch.setattr("app.cli.get_settings", lambda: object())
    monkeypatch.setattr("app.cli.build_default_orchestrator", lambda _settings: FakeOrchestrator())

    result = CliRunner().invoke(cli_app, ["write-scripts", "--no-tts"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tts_batch"] is None
