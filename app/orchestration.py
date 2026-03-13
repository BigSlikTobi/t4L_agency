# This file contains the NewsroomOrchestrator class, which coordinates the workflow of generating a radio rundown 
# based on a given request. It uses the AgentsWorkflow to execute the various steps of the process, 
# and compiles the final RadioRundown output.

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.adapters import (
    GeminiTTSBatchAdapter,
    NewsFeedAdapter,
    StoryGroupUpdatesAdapter,
    SupabaseArticleLookupAdapter,
)
from app.config import Settings
from app.constants import NFL_TEAMS
from app.history import TeamUpdateHistoryStore
from app.newsroom.context import NewsroomRunContext, TeamUpdateRunContext
from app.newsroom.workflow import AgentsWorkflow, TeamUpdateWorkflow
from app.schemas import (
    HourlyPlaylistScriptsRequest,
    HourlyPlaylistScriptsResponse,
    HourlyPlaylistScriptsRunResult,
    RadioRundown,
    RadioRundownRequest,
    SourceCoverage,
    TeamUpdateBatchRequest,
    TeamUpdateBatchResponse,
    TeamUpdateReport,
    TeamUpdateReportRequest,
)


class NewsroomOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        agents: AgentsWorkflow,
        team_updates: TeamUpdateWorkflow | None = None,
        adapters: list | None = None,
    ) -> None:
        self._agents = agents
        self._team_updates = team_updates
        self._adapters = adapters or []

    async def close(self) -> None:
        for adapter in self._adapters:
            if hasattr(adapter, "close"):
                await adapter.close()

    async def run_radio_rundown(self, request: RadioRundownRequest) -> RadioRundown:
        context = NewsroomRunContext(
            request=request,
            run_id=str(uuid4()),
            generated_at=datetime.now(UTC),
        )
        draft = await self._agents.run_newsroom(context)
        segment_teams = {
            team
            for segment in draft.segments + draft.backup_segments
            for team in segment.teams
        }
        source_story_ids = {
            source.story_id
            for segment in draft.segments + draft.backup_segments
            for source in segment.source_articles
        }
        source_urls = {
            source.url
            for segment in draft.segments + draft.backup_segments
            for source in segment.source_articles
        }
        selected_team_count = len(request.teams or NFL_TEAMS)
        draft.run_id = context.run_id
        draft.generated_at = context.generated_at
        draft.lookback_hours = request.lookback_hours
        draft.target_duration_minutes = request.target_duration_minutes
        draft.source_coverage = SourceCoverage(
            total_feed_stories=len(context.feed_stories),
            researched_articles=max(len(context.researched_urls), len(source_urls)),
            digested_articles=max(len(context.article_digests), len(source_story_ids)),
            teams_with_stories=len(segment_teams),
            teams_without_stories=max(0, selected_team_count - len(segment_teams)),
        )
        draft.warnings = context.warnings
        return draft

    async def run_team_update_report(self, request: TeamUpdateReportRequest) -> TeamUpdateReport:
        if self._team_updates is None:
            raise RuntimeError("Team update workflow is not configured.")

        context = TeamUpdateRunContext(
            request=request,
            run_id=str(uuid4()),
            generated_at=datetime.now(UTC),
        )
        return await self._team_updates.run_team_update(context)

    async def run_team_update_reports(
        self,
        request: TeamUpdateBatchRequest,
    ) -> TeamUpdateBatchResponse:
        if self._team_updates is None:
            raise RuntimeError("Team update workflow is not configured.")

        run_id = str(uuid4())
        generated_at = datetime.now(UTC)
        return await self._team_updates.run_team_update_batch(
            run_id=run_id,
            generated_at=generated_at,
            request=request,
        )

    async def run_hourly_playlist_scripts(
        self,
        request: HourlyPlaylistScriptsRequest,
    ) -> HourlyPlaylistScriptsResponse:
        if self._team_updates is None:
            raise RuntimeError("Team update workflow is not configured.")

        script_run_id = str(uuid4())
        generated_at = datetime.now(UTC)
        result = await self._team_updates.run_hourly_playlist_scripts(
            script_run_id=script_run_id,
            generated_at=generated_at,
            request=request,
        )
        return HourlyPlaylistScriptsResponse.from_run_result(
            HourlyPlaylistScriptsRunResult.model_validate(result)
        )


def build_default_orchestrator(settings: Settings) -> NewsroomOrchestrator:
    news_feed = NewsFeedAdapter(
        base_url=str(settings.supabase_news_feed_url),
        auth_token=settings.supabase_function_auth_token.get_secret_value(),
        timeout_seconds=settings.news_timeout_seconds,
    )
    article_lookup = SupabaseArticleLookupAdapter(
        str(settings.supabase_article_lookup_url),
        settings.supabase_function_auth_token.get_secret_value(),
        timeout_seconds=settings.news_timeout_seconds,
    )
    tts_batch = GeminiTTSBatchAdapter(
        endpoint_url=str(settings.gemini_tts_batch_url),
        timeout_seconds=settings.news_timeout_seconds,
        process_timeout_seconds=settings.gemini_tts_batch_timeout_seconds,
    )
    adapters = [news_feed, article_lookup, tts_batch]

    team_update_workflow = None
    if settings.supabase_story_group_updates_url is not None:
        story_group_updates = StoryGroupUpdatesAdapter(
            str(settings.supabase_story_group_updates_url),
            settings.supabase_function_auth_token.get_secret_value(),
            timeout_seconds=settings.news_timeout_seconds,
        )
        adapters.append(story_group_updates)
        team_update_workflow = TeamUpdateWorkflow(
            settings=settings,
            news_feed=news_feed,
            article_lookup=article_lookup,
            story_group_updates=story_group_updates,
            history_store=TeamUpdateHistoryStore(settings.team_update_history_sqlite_path),
            tts_batch=tts_batch,
        )
    agents = AgentsWorkflow(
        settings=settings,
        news_feed=news_feed,
        article_lookup=article_lookup,
    )
    return NewsroomOrchestrator(
        settings=settings,
        agents=agents,
        team_updates=team_update_workflow,
        adapters=adapters,
    )
