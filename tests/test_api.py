from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.history import TeamUpdateHistoryStore
from app.main import create_app
from app.schemas import (
    HourlyPlaylist,
    HourlyPlaylistItem,
    HourlyPlaylistScriptsResponse,
    RadioRundown,
    RadioStoryScript,
    SegmentCandidate,
    SourceArticleRef,
    SourceCoverage,
    TeamUpdateBatchResponse,
    TeamUpdatePackage,
    TeamUpdateReport,
)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.last_request = None
        self.last_team_update_request = None
        self.last_team_update_batch_request = None
        self.last_hourly_playlist_scripts_request = None

    async def run_radio_rundown(self, _request):
        self.last_request = _request
        return RadioRundown(
            run_id="run-1",
            generated_at=datetime.now(UTC),
            lookback_hours=24,
            target_duration_minutes=60,
            segments=[
                SegmentCandidate(
                    rank=1,
                    headline="Lead",
                    editorial_angle="Angle",
                    teams=["MIN"],
                    story_priority=5,
                    recommended_duration_seconds=600,
                    recommended_word_count=1500,
                    summary="Summary",
                    key_points=["Point"],
                    segment_idea="Idea",
                    source_articles=[
                        SourceArticleRef(
                            story_id="story-1",
                            url="https://example.com/story-1",
                            title="Title",
                            source_name="ESPN",
                        )
                    ],
                    confidence=0.8,
                )
            ],
            backup_segments=[],
            source_coverage=SourceCoverage(
                total_feed_stories=1,
                researched_articles=1,
                digested_articles=1,
                teams_with_stories=1,
                teams_without_stories=31,
            ),
            warnings=[],
        )

    async def run_team_update_report(self, request):
        self.last_team_update_request = request
        return TeamUpdateReport(
            run_id="update-run-1",
            generated_at=datetime.now(UTC),
            team=request.team,
            lookback_minutes=request.lookback_minutes,
            status="report_ready",
            report=TeamUpdatePackage(
                total_duration_seconds=180,
                headline="Minnesota Vikings update",
                topics=[
                    {
                        "headline": "Roster move update",
                        "editorial_angle": "What changed in the last hour",
                        "recommended_duration_seconds": 180,
                        "framing": "new",
                        "continuity": "new_story",
                        "talking_points": ["Point 1"],
                        "source_articles": [
                            {
                                "story_id": "story-1",
                                "url": "https://example.com/story-1",
                                "title": "Title",
                                "source_name": "ESPN",
                            }
                        ],
                    }
                ],
                source_articles=[
                    SourceArticleRef(
                        story_id="story-1",
                        url="https://example.com/story-1",
                        title="Title",
                        source_name="ESPN",
                    )
                ],
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

    async def run_team_update_reports(self, request):
        self.last_team_update_batch_request = request
        teams = request.teams or ["ARI", "MIN"]
        return TeamUpdateBatchResponse(
            run_id="batch-run-1",
            generated_at=datetime.now(UTC),
            lookback_minutes=request.lookback_minutes,
            reports=[
                await self.run_team_update_report(
                    type("Req", (), {"team": team, "lookback_minutes": request.lookback_minutes})()
                )
                for team in teams
            ],
            hourly_playlist=HourlyPlaylist(
                selected_count=1 if teams else 0,
                items=[
                    HourlyPlaylistItem(
                        rank=1,
                        team=teams[0],
                        headline="Minnesota Vikings update",
                        framing="new",
                        continuity="new_story",
                        production_reason="Top report in the batch.",
                        source_articles=[
                            SourceArticleRef(
                                story_id="story-1",
                                url="https://example.com/story-1",
                                title="Title",
                                source_name="ESPN",
                            )
                        ],
                    )
                ]
                if teams
                else [],
            ),
        )

    async def run_hourly_playlist_scripts(self, request):
        self.last_hourly_playlist_scripts_request = request
        return HourlyPlaylistScriptsResponse(
            language=request.language,
            voice_name="Puck",
            items=[
                {
                    "id": "vikings-update",
                    "title": "Minnesota Vikings update",
                    "direction": {
                        "audio_profile": "Warm FM host with bright energy.",
                        "scene": "Drive-home NFL update.",
                        "director_notes": "Stay warm and direct.",
                        "pace": "Urgent but controlled.",
                        "warmth": "Engaged and listener-close.",
                        "must_hit": ["Geno Smith", "trade with the Raiders"],
                        "pronunciations": [
                            {
                                "term": "Tom Pelissero",
                                "guide": "PEH-lih-SARE-oh",
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
                        "id": "vikings-update",
                        "storage_path": "gemini-tts-batch/batches/abc123/vikings-update.mp3",
                        "mime_type": "audio/mpeg",
                        "source_mime_type": "audio/wav",
                        "public_url": "https://example.com/vikings-update.mp3",
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


def build_settings(*, team_update_history_sqlite_path: Path | None = None) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        supabase_news_feed_url="https://example.com/feed",
        supabase_article_lookup_url="https://example.com/article-lookup",
        supabase_function_auth_token="supabase-token",
        supabase_storage_key="storage-key",
        team_update_history_sqlite_path=team_update_history_sqlite_path or Path("./var/team_update_history.sqlite3"),
    )


def make_radio_story_script(
    *,
    team: str = "MIN",
    playlist_rank: int = 1,
    language: str = "en-US",
) -> RadioStoryScript:
    return RadioStoryScript(
        team=team,
        playlist_rank=playlist_rank,
        language=language,
        headline=f"{team} headline",
        continuity="new_story",
        persona_name="Lena Torres",
        persona_backstory="Phoenix night host turned NFL connector.",
        persona_specialty="fan emotion and momentum swings",
        voice_name="Puck",
        dialect="a slight Arizona-Southwest lilt",
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


def test_healthz() -> None:
    app = create_app(settings=build_settings(), orchestrator=FakeOrchestrator())
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["config_ready"] is True


def test_create_radio_rundown() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/radio-rundown", json={})
    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert orchestrator.last_request.teams is None


def test_create_radio_rundown_with_team_subset() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/radio-rundown", json={"teams": ["ari", "MIN"]})

    assert response.status_code == 200
    assert orchestrator.last_request.teams == ["ARI", "MIN"]


def test_create_team_update_report() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/team-update-report", json={"team": "min"})

    assert response.status_code == 200
    assert response.json()["run_id"] == "update-run-1"
    assert orchestrator.last_team_update_request.team == "MIN"


def test_create_team_update_reports_batch() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/team-update-reports", json={"teams": ["ari", "MIN"]})

    assert response.status_code == 200
    assert len(response.json()["reports"]) == 2
    assert response.json()["hourly_playlist"]["selected_count"] == 1
    assert orchestrator.last_team_update_batch_request.teams == ["ARI", "MIN"]


def test_create_hourly_playlist_scripts() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/hourly-playlist-scripts", json={})

    assert response.status_code == 200
    assert response.json()["language"] == "en-US"
    assert response.json()["voice_name"] == "Puck"
    assert response.json()["items"][0]["id"] == "vikings-update"
    assert response.json()["items"][0]["script"]["body"] == "Body"
    assert response.json()["items"][0]["direction"]["pace"] == "Urgent but controlled."
    assert response.json()["tts_batch"]["items"][0]["public_url"] == "https://example.com/vikings-update.mp3"
    assert orchestrator.last_hourly_playlist_scripts_request.playlist_id is None
    assert orchestrator.last_hourly_playlist_scripts_request.language == "en-US"
    assert orchestrator.last_hourly_playlist_scripts_request.enable_tts is True


def test_create_hourly_playlist_scripts_accepts_language() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/hourly-playlist-scripts", json={"language": "de-DE"})

    assert response.status_code == 200
    assert response.json()["language"] == "de-DE"
    assert orchestrator.last_hourly_playlist_scripts_request.language == "de-DE"


def test_create_hourly_playlist_scripts_accepts_enable_tts_override() -> None:
    orchestrator = FakeOrchestrator()
    app = create_app(settings=build_settings(), orchestrator=orchestrator)
    client = TestClient(app)
    response = client.post("/orchestrations/hourly-playlist-scripts", json={"enable_tts": False})

    assert response.status_code == 200
    assert orchestrator.last_hourly_playlist_scripts_request.enable_tts is False


def test_qa_player_page_loads() -> None:
    app = create_app(settings=build_settings(), orchestrator=FakeOrchestrator())
    client = TestClient(app)

    response = client.get("/qa/player")

    assert response.status_code == 200
    assert "QA Radio Player" in response.text
    assert "/qa/player-feed" in response.text


def test_qa_player_feed_prefers_latest_scripts_artifact(tmp_path: Path) -> None:
    settings = build_settings(team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3")
    scripts_path = settings.team_update_history_sqlite_path.parent / "scripts.json"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text(
        HourlyPlaylistScriptsResponse(
            language="en-US",
            voice_name="Puck",
            items=[
                {
                    "id": "vikings-update",
                    "title": "Minnesota Vikings update",
                    "direction": {
                        "audio_profile": "Warm FM host with bright energy.",
                        "scene": "Drive-home NFL update.",
                        "director_notes": "Stay warm and direct.",
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
                "token_usage": {},
                "items": [
                    {
                        "id": "vikings-update",
                        "storage_path": "gemini-tts-batch/batches/abc123/vikings-update.mp3",
                        "mime_type": "audio/mpeg",
                        "source_mime_type": "audio/wav",
                        "public_url": "https://example.com/vikings-update.mp3",
                        "token_usage": {},
                    }
                ],
                "failures": [],
            },
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    app = create_app(settings=settings, orchestrator=FakeOrchestrator())
    client = TestClient(app)

    response = client.get("/qa/player-feed")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "artifact"
    assert payload["has_audio"] is True
    assert payload["items"][0]["audio_url"] == "https://example.com/vikings-update.mp3"
    assert payload["items"][0]["script_text"] == "Intro Body Outro"


def test_qa_player_feed_supports_explicit_batch_override(tmp_path: Path) -> None:
    settings = build_settings(team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3")
    scripts_path = settings.team_update_history_sqlite_path.parent / "scripts.json"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text(
        HourlyPlaylistScriptsResponse(
            language="en-US",
            voice_name="Puck",
            items=[
                {
                    "id": "vikings-update",
                    "title": "Minnesota Vikings update",
                    "direction": {
                        "audio_profile": "Warm FM host with bright energy.",
                        "scene": "Drive-home NFL update.",
                        "director_notes": "Stay warm and direct.",
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
                "token_usage": {},
                "items": [
                    {
                        "id": "vikings-update",
                        "storage_path": "gemini-tts-batch/batches/abc123/vikings-update.mp3",
                        "mime_type": "audio/mpeg",
                        "source_mime_type": "audio/wav",
                        "public_url": "https://example.com/vikings-update.mp3",
                        "token_usage": {},
                    }
                ],
                "failures": [],
            },
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    app = create_app(settings=settings, orchestrator=FakeOrchestrator())
    client = TestClient(app)

    response = client.get("/qa/player-feed", params={"batch": "gemini-tts-batch/batches/override-id"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_audio"] is True
    assert (
        payload["items"][0]["audio_url"]
        == "https://example.com/storage/v1/object/public/audio/gemini-tts-batch/batches/override-id/vikings-update.mp3"
    )


def test_qa_player_feed_uses_requested_batch_manifest_items(tmp_path: Path) -> None:
    settings = build_settings(team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3")
    scripts_path = settings.team_update_history_sqlite_path.parent / "scripts.json"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text(
        HourlyPlaylistScriptsResponse(
            language="en-US",
            voice_name="Puck",
            items=[
                {
                    "id": "vikings-update",
                    "title": "Minnesota Vikings update",
                    "direction": {"audio_profile": "a", "scene": "b", "director_notes": "c"},
                    "script": {"intro": "Intro", "body": "Body", "outro": "Outro"},
                }
            ],
            tts_batch=None,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    app = create_app(settings=settings, orchestrator=FakeOrchestrator())
    client = TestClient(app)

    fake_manifest = {
        "batch_id": "batches/override-id",
        "status": "JOB_STATE_SUCCEEDED",
        "output_file_id": "files/batch-override-id",
        "processed_count": 2,
        "failed_count": 0,
        "token_usage": {},
        "items": [
            {
                "id": "dj-moore-to-bills-why-it-ended-in-chicago",
                "storage_path": "gemini-tts-batch/batches/override-id/dj-moore-to-bills-why-it-ended-in-chicago.mp3",
                "mime_type": "audio/mpeg",
                "source_mime_type": "audio/wav",
                "public_url": "https://example.com/storage/v1/object/public/audio/gemini-tts-batch/batches/override-id/dj-moore-to-bills-why-it-ended-in-chicago.mp3",
                "token_usage": {},
            },
            {
                "id": "cowboys-crosby-door-open",
                "storage_path": "gemini-tts-batch/batches/override-id/cowboys-crosby-door-open.mp3",
                "mime_type": "audio/mpeg",
                "source_mime_type": "audio/wav",
                "public_url": "https://example.com/storage/v1/object/public/audio/gemini-tts-batch/batches/override-id/cowboys-crosby-door-open.mp3",
                "token_usage": {},
            },
        ],
        "failures": [],
        "processed_at": "2026-03-13T16:21:53.441695+00:00",
    }

    with patch("app.qa_player.httpx.get", return_value=Mock(status_code=200, text="", raise_for_status=Mock())) as mocked_get:
        mocked_get.return_value.text = __import__("json").dumps(fake_manifest)
        response = client.get("/qa/player-feed", params={"batch": "gemini-tts-batch/batches/override-id"})

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["items"]] == [
        "dj-moore-to-bills-why-it-ended-in-chicago",
        "cowboys-crosby-door-open",
    ]
    assert payload["items"][0]["audio_url"].endswith("dj-moore-to-bills-why-it-ended-in-chicago.mp3")


def test_qa_player_feed_falls_back_to_latest_history_scripts(tmp_path: Path) -> None:
    settings = build_settings(team_update_history_sqlite_path=tmp_path / "team_update_history.sqlite3")
    store = TeamUpdateHistoryStore(settings.team_update_history_sqlite_path)
    store.save_story_scripts(
        script_run_id="script-run-1",
        playlist_id="playlist-1",
        batch_run_id="batch-1",
        generated_at=datetime.now(UTC),
        scripts=[
            make_radio_story_script(team="MIN", playlist_rank=1),
            make_radio_story_script(team="DET", playlist_rank=2),
        ],
    )
    app = create_app(settings=settings, orchestrator=FakeOrchestrator())
    client = TestClient(app)

    response = client.get("/qa/player-feed")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "history"
    assert payload["script_run_id"] == "script-run-1"
    assert payload["playlist_id"] == "playlist-1"
    assert payload["has_audio"] is False
    assert [item["team"] for item in payload["items"]] == ["MIN", "DET"]
    assert payload["items"][0]["script_text"] == "Intro Body Outro"
