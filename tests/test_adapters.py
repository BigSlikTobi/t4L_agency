from __future__ import annotations

import json

import httpx
import pytest

from app.adapters import (
    ExternalServiceError,
    GeminiTTSBatchAdapter,
    NewsFeedAdapter,
    StoryGroupUpdatesAdapter,
    SupabaseArticleLookupAdapter,
)
from app.schemas import (
    GeminiTTSBatchCreateRequest,
    GeminiTTSCredentials,
    GeminiTTSBatchProcessRequest,
    GeminiTTSSupabaseStorageConfig,
    GeminiTTSBatchStatusRequest,
)


@pytest.mark.asyncio
async def test_news_feed_adapter_sends_auth_headers() -> None:
    captured_headers: dict[str, str] = {}
    captured_request: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers["authorization"] = request.headers["Authorization"]
        captured_headers["apikey"] = request.headers["apikey"]
        captured_request["method"] = request.method
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "stories": [
                    {
                        "id": "story-1",
                        "url": "https://example.com/story-1",
                        "title": "Big move",
                        "source_name": "ESPN",
                        "category": "Breaking News",
                        "facts_count": 4,
                        "entities": [],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = NewsFeedAdapter("https://example.com/feed", "token-123")
        stories = await adapter.fetch_stories(24)
    finally:
        httpx.AsyncClient = original_client

    assert len(stories) == 1
    assert captured_headers["authorization"] == "Bearer token-123"
    assert captured_headers["apikey"] == "token-123"
    assert captured_request["method"] == "POST"
    assert captured_request["body"] == '{"lookback_hours":24}'


@pytest.mark.asyncio
async def test_news_feed_adapter_raises_on_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = NewsFeedAdapter("https://example.com/feed", "token-123")
        with pytest.raises(ExternalServiceError):
            await adapter.fetch_stories(24)
    finally:
        httpx.AsyncClient = original_client


@pytest.mark.asyncio
async def test_article_lookup_adapter_sends_auth_headers_and_body() -> None:
    captured_headers: dict[str, str] = {}
    captured_request: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers["authorization"] = request.headers["Authorization"]
        captured_headers["apikey"] = request.headers["apikey"]
        captured_request["method"] = request.method
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "requested_url": "https://example.com/article",
                "found": True,
                "article": {
                    "url": "https://example.com/article",
                    "cite_url": "https://example.com/cite",
                    "header": "Stored header",
                    "content": "Stored body",
                    "description": "Stored deck",
                    "author": "Reporter",
                    "category": "Breaking News",
                    "quotes": ["Quote 1"],
                    "embeddings_id": 6797,
                    "group_id": "2f81811e-ae4b-4c49-9355-b26c75ce2081",
                    "created_at": "2026-01-31T17:22:15.991+00:00",
                    "updated_at": "2026-03-08T10:09:43.63922+00:00",
                },
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = SupabaseArticleLookupAdapter(
            "https://example.com/article-lookup",
            "supabase-token",
            timeout_seconds=10.0,
        )
        result = await adapter.lookup_article("https://example.com/article")
    finally:
        httpx.AsyncClient = original_client

    assert result.found is True
    assert result.article is not None
    assert result.article.header == "Stored header"
    assert captured_headers["authorization"] == "Bearer supabase-token"
    assert captured_headers["apikey"] == "supabase-token"
    assert captured_request["method"] == "POST"
    assert captured_request["body"] == '{"url":"https://example.com/article"}'


@pytest.mark.asyncio
async def test_article_lookup_adapter_returns_not_found_payload() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"requested_url": "https://example.com/missing", "found": False})

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = SupabaseArticleLookupAdapter(
            "https://example.com/article-lookup",
            "supabase-token",
        )
        result = await adapter.lookup_article("https://example.com/missing")
    finally:
        httpx.AsyncClient = original_client

    assert result.requested_url == "https://example.com/missing"
    assert result.found is False
    assert result.article is None


@pytest.mark.asyncio
async def test_article_lookup_adapter_raises_on_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = SupabaseArticleLookupAdapter(
            "https://example.com/article-lookup",
            "supabase-token",
        )
        with pytest.raises(ExternalServiceError):
            await adapter.lookup_article("https://example.com/article")
    finally:
        httpx.AsyncClient = original_client


@pytest.mark.asyncio
async def test_story_group_updates_adapter_sends_auth_headers_and_body() -> None:
    captured_headers: dict[str, str] = {}
    captured_request: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers["authorization"] = request.headers["Authorization"]
        captured_headers["apikey"] = request.headers["apikey"]
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "group_id": "group-1",
                "lookback_minutes": 60,
                "updates": [
                    {
                        "member_identifier": "story-2",
                        "added_at": "2026-03-08T10:09:43.639220+00:00",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = StoryGroupUpdatesAdapter(
            "https://example.com/story-group-updates",
            "supabase-token",
        )
        result = await adapter.fetch_recent_updates(group_id="group-1", lookback_minutes=60)
    finally:
        httpx.AsyncClient = original_client

    assert result.group_id == "group-1"
    assert len(result.updates) == 1
    assert result.updates[0].member_identifier == "story-2"
    assert captured_headers["authorization"] == "Bearer supabase-token"
    assert captured_headers["apikey"] == "supabase-token"
    assert captured_request["body"] == '{"group_id":"group-1","lookback_minutes":60}'


@pytest.mark.asyncio
async def test_story_group_updates_adapter_raises_on_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = StoryGroupUpdatesAdapter(
            "https://example.com/story-group-updates",
            "supabase-token",
        )
        with pytest.raises(ExternalServiceError):
            await adapter.fetch_recent_updates(group_id="group-1", lookback_minutes=60)
    finally:
        httpx.AsyncClient = original_client


@pytest.mark.asyncio
async def test_gemini_tts_batch_adapter_sends_create_payload() -> None:
    captured_request: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
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
            },
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = GeminiTTSBatchAdapter("https://example.com/gemini-tts-batch")
        result = await adapter.create_batch(
            GeminiTTSBatchCreateRequest(
                model_name="gemini-2.5-pro-preview-tts",
                voice_name="Charon",
                items=[
                    {
                        "id": "story-1",
                        "title": "Story 1",
                        "voice_name": "Charon",
                        "direction": {"audio_profile": "Breaking anchor"},
                        "script": {"intro": "Intro", "body": "Body", "outro": "Outro"},
                    }
                ],
                credentials=GeminiTTSCredentials(gemini="gemini-key"),
            )
        )
    finally:
        httpx.AsyncClient = original_client

    assert result.batch_id == "batches/abc123"
    assert result.status == "JOB_STATE_PENDING"
    assert captured_request["method"] == "POST"
    assert captured_request["url"] == "https://example.com/gemini-tts-batch"
    assert captured_request["body"] == {
        "action": "create",
        "model_name": "gemini-2.5-pro-preview-tts",
        "voice_name": "Charon",
        "items": [
            {
                "id": "story-1",
                "title": "Story 1",
                "voice_name": "Charon",
                "direction": {
                    "audio_profile": "Breaking anchor",
                    "scene": "",
                    "director_notes": "",
                    "pace": "",
                    "warmth": "",
                    "must_hit": [],
                    "pronunciations": [],
                },
                "script": {
                    "intro": "Intro",
                    "body": "Body",
                    "outro": "Outro",
                },
            }
        ],
        "credentials": {"gemini": "gemini-key"},
    }


@pytest.mark.asyncio
async def test_gemini_tts_batch_adapter_processes_status_and_manifest_response() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "batch_id": "batches/abc123",
                    "status": "JOB_STATE_SUCCEEDED",
                    "created_at": "2026-03-11T10:00:00Z",
                    "updated_at": "2026-03-11T10:05:00Z",
                    "output_file_id": "files/out",
                    "error_file_id": None,
                    "error": None,
                },
            ),
            httpx.Response(
                200,
                json={
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
                            "id": "story-1",
                            "storage_path": "gemini-tts-batch/batches/abc123/story-1.mp3",
                            "mime_type": "audio/mpeg",
                            "source_mime_type": "audio/wav",
                            "public_url": "https://example.com/story-1.mp3",
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
            ),
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = GeminiTTSBatchAdapter("https://example.com/gemini-tts-batch")
        status = await adapter.fetch_batch_status(
            GeminiTTSBatchStatusRequest(
                batch_id="batches/abc123",
                credentials=GeminiTTSCredentials(gemini="gemini-key"),
            )
        )
        processed = await adapter.process_batch(
            GeminiTTSBatchProcessRequest(
                batch_id="batches/abc123",
                credentials=GeminiTTSCredentials(gemini="gemini-key"),
                supabase=GeminiTTSSupabaseStorageConfig(
                    url="https://project.supabase.co",
                    key="storage-key",
                    bucket="audio",
                    path_prefix="gemini-tts-batch",
                ),
            )
        )
    finally:
        httpx.AsyncClient = original_client

    assert status.status == "JOB_STATE_SUCCEEDED"
    assert processed.processed_count == 1
    assert processed.items[0].public_url == "https://example.com/story-1.mp3"
    assert processed.manifest_public_url == "https://example.com/manifest.json"


@pytest.mark.asyncio
async def test_gemini_tts_batch_adapter_raises_on_failure() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = GeminiTTSBatchAdapter("https://example.com/gemini-tts-batch")
        with pytest.raises(ExternalServiceError):
            await adapter.fetch_batch_status(
                GeminiTTSBatchStatusRequest(batch_id="batches/abc123")
            )
    finally:
        httpx.AsyncClient = original_client


@pytest.mark.asyncio
async def test_gemini_tts_batch_adapter_does_not_retry_wrapped_auth_failures() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            502,
            text=(
                "Failed to upload path.mp3 to Supabase: {'statusCode': 403, "
                "'error': Unauthorized, 'message': new row violates row-level security policy'}"
            ),
        )

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    class MockClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = MockClient
    try:
        adapter = GeminiTTSBatchAdapter("https://example.com/gemini-tts-batch")
        with pytest.raises(ExternalServiceError, match="Permanent upstream auth error"):
            await adapter.process_batch(
                GeminiTTSBatchProcessRequest(
                    batch_id="batches/abc123",
                    credentials=GeminiTTSCredentials(gemini="gemini-key"),
                    supabase=GeminiTTSSupabaseStorageConfig(
                        url="https://project.supabase.co",
                        key="bad-key",
                        bucket="audio",
                        path_prefix="gemini-tts-batch",
                    ),
                )
            )
    finally:
        httpx.AsyncClient = original_client

    assert calls == 1
