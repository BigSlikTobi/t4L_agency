# This file contains adapter classes for external services used by the application.
# It defines adapters for the Supabase news feed, stored article lookup edge functions,
# and recent group update lookups.

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.schemas import (
    ArticleContentLookupToolResponse,
    FeedStory,
    GeminiTTSBatchCreateRequest,
    GeminiTTSBatchJobStatus,
    GeminiTTSBatchProcessRequest,
    GeminiTTSBatchStatusRequest,
    StoryGroupUpdatesToolResponse,
    TTSBatchResult,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_MAX_ATTEMPTS = 3


class ExternalServiceError(RuntimeError):
    """Raised when an upstream integration fails."""


class _TransientHTTPError(ExternalServiceError):
    """Raised for retryable HTTP status codes (429, 5xx)."""


def _default_retry():
    """Shared retry policy: 3 attempts, exponential backoff 1→8s, retries on transient HTTP and transport errors."""
    return retry(
        retry=retry_if_exception_type((_TransientHTTPError, httpx.TransportError)),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def _check_transient(response: httpx.Response) -> None:
    """Raise _TransientHTTPError if response has a retryable status code."""
    lowered_body = response.text.lower()
    if response.status_code in _RETRYABLE_STATUS_CODES and (
        "unauthorized" in lowered_body
        or "row-level security" in lowered_body
        or "new row violates row-le" in lowered_body
        or "statuscode': 403" in lowered_body
        or '"statuscode": 403' in lowered_body
    ):
        raise ExternalServiceError(
            f"Permanent upstream auth error despite HTTP {response.status_code}: {response.text[:200]}"
        )
    if response.status_code in _RETRYABLE_STATUS_CODES:
        raise _TransientHTTPError(
            f"Transient HTTP {response.status_code}: {response.text[:200]}"
        )


class NewsFeedAdapter:
    def __init__(self, base_url: str, auth_token: str, timeout_seconds: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "apikey": auth_token,
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @_default_retry()
    async def fetch_stories(self, lookback_hours: int) -> list[FeedStory]:
        logger.info("Fetching news feed stories", extra={"lookback_hours": lookback_hours})
        response = await self._client.post("/", json={"lookback_hours": lookback_hours})

        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"News feed request failed with status {response.status_code}: {response.text}"
            )

        payload = response.json()
        stories = payload.get("stories", [])
        result = [FeedStory.model_validate(story) for story in stories]
        logger.info("Fetched news feed stories", extra={"count": len(result)})
        return result


class SupabaseArticleLookupAdapter:
    def __init__(self, base_url: str, auth_token: str, timeout_seconds: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "apikey": auth_token,
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @_default_retry()
    async def lookup_article(self, url: str) -> ArticleContentLookupToolResponse:
        logger.debug("Looking up article", extra={"url": url})
        response = await self._client.post("/", json={"url": url})

        if response.status_code == 404:
            return ArticleContentLookupToolResponse(requested_url=url, found=False, article=None)
        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"Article lookup request failed with status {response.status_code}: {response.text}"
            )

        return ArticleContentLookupToolResponse.model_validate(response.json())


class StoryGroupUpdatesAdapter:
    def __init__(self, base_url: str, auth_token: str, timeout_seconds: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "apikey": auth_token,
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    @_default_retry()
    async def fetch_recent_updates(
        self,
        *,
        group_id: str,
        lookback_minutes: int,
    ) -> StoryGroupUpdatesToolResponse:
        logger.debug("Fetching story group updates", extra={"group_id": group_id})
        response = await self._client.post(
            "/", json={"group_id": group_id, "lookback_minutes": lookback_minutes}
        )

        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"Story group updates request failed with status {response.status_code}: {response.text}"
            )

        return StoryGroupUpdatesToolResponse.model_validate(response.json())


class GeminiTTSBatchAdapter:
    def __init__(
        self,
        endpoint_url: str,
        timeout_seconds: float = 30.0,
        process_timeout_seconds: float = 300.0,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._client = httpx.AsyncClient(timeout=timeout_seconds)
        self._process_timeout = httpx.Timeout(process_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    @_default_retry()
    async def create_batch(self, request: GeminiTTSBatchCreateRequest) -> GeminiTTSBatchJobStatus:
        response = await self._client.post(self._endpoint_url, json=request.model_dump(mode="json"))
        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"TTS batch create request failed with status {response.status_code}: {response.text}"
            )
        return GeminiTTSBatchJobStatus.model_validate(response.json())

    @_default_retry()
    async def fetch_batch_status(self, request: GeminiTTSBatchStatusRequest) -> GeminiTTSBatchJobStatus:
        response = await self._client.post(self._endpoint_url, json=request.model_dump(mode="json"))
        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"TTS batch status request failed with status {response.status_code}: {response.text}"
            )
        return GeminiTTSBatchJobStatus.model_validate(response.json())

    @_default_retry()
    async def process_batch(self, request: GeminiTTSBatchProcessRequest) -> TTSBatchResult:
        response = await self._client.post(
            self._endpoint_url,
            json=request.model_dump(mode="json"),
            timeout=self._process_timeout,
        )
        _check_transient(response)
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"TTS batch process request failed with status {response.status_code}: {response.text}"
            )
        return TTSBatchResult.model_validate(response.json())
