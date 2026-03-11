# Team update workflow — handles single-team reports, batch updates,
# hourly playlist selection, and script generation.

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from agents import Agent, Runner
from agents.tool import FunctionTool

from app.adapters import (
    GeminiTTSBatchAdapter,
    NewsFeedAdapter,
    StoryGroupUpdatesAdapter,
    SupabaseArticleLookupAdapter,
)
from app.config import Settings
from app.constants import (
    DEFAULT_SCRIPT_LANGUAGE,
    NFL_TEAMS,
    SCRIPT_LANGUAGE_CONFIGS,
    ScriptPersona,
)
from app.history import TeamUpdateHistoryStore
from app.newsroom.agents import (
    build_article_data_agent,
    build_hourly_playlist_orchestrator_agent,
    build_hourly_script_batch_agent,
    build_radio_script_writer_agent,
    build_team_update_agent,
    build_team_update_batch_agent,
)
from app.newsroom.context import TeamUpdateRunContext
from app.newsroom.helpers import (
    build_tts_batch_create_request,
    build_tts_batch_process_request,
    build_hourly_script_batch_input,
    build_radio_script_story_inputs,
    build_hourly_playlist_input,
    build_team_update_batch_input,
    build_team_update_input,
    coerce_output,
    filter_stories_for_team,
)
from app.newsroom.tools import (
    build_article_data_tool,
    build_article_lookup_tool,
    build_radio_story_script_tool,
    build_team_update_tool,
)
from app.newsroom.tracing import build_run_config
from app.schemas import (
    ArticleContentLookupToolResponse,
    ArticleDigest,
    FeedStory,
    GeminiTTSBatchJobStatus,
    GeminiTTSBatchStatusRequest,
    GeminiTTSCredentials,
    HourlyPlaylist,
    HourlyPlaylistItem,
    HourlyPlaylistSelection,
    HourlyPlaylistScriptsBatchAgentResult,
    HourlyPlaylistScriptsRequest,
    HourlyPlaylistScriptsRunResult,
    RadioStoryScript,
    RadioStoryScriptDraft,
    ScriptDirectionPronunciation,
    StoryGroupUpdatesToolResponse,
    TeamUpdateBatchAgentResult,
    TeamUpdateBatchRequest,
    TeamUpdateBatchResponse,
    TeamUpdateCandidate,
    TeamUpdateHistoryEntry,
    TeamUpdatePackage,
    TeamUpdateReport,
    TeamUpdateReportRequest,
    TeamUpdateTeamInput,
)

logger = logging.getLogger(__name__)

_CONCURRENT_LOOKUP_LIMIT: Final[int] = 10
_TTS_TERMINAL_FAILURE_STATES: Final[frozenset[str]] = frozenset(
    {
        "JOB_STATE_CANCELLED",
        "JOB_STATE_CANCELLING",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_FAILED",
    }
)


@dataclass(frozen=True)
class ScriptGenerationStack:
    language: str
    personas: list[ScriptPersona]
    radio_script_writer_agent: Agent
    radio_story_script_tool: FunctionTool
    hourly_script_batch_agent: Agent


class TeamUpdateWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        news_feed: NewsFeedAdapter,
        article_lookup: SupabaseArticleLookupAdapter,
        story_group_updates: StoryGroupUpdatesAdapter,
        history_store: TeamUpdateHistoryStore,
        tts_batch: GeminiTTSBatchAdapter | None = None,
    ) -> None:
        self._settings = settings
        self._news_feed = news_feed
        self._article_lookup = article_lookup
        self._story_group_updates = story_group_updates
        self._history_store = history_store
        self._tts_batch = tts_batch
        self._lookup_semaphore = asyncio.Semaphore(_CONCURRENT_LOOKUP_LIMIT)
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())

        self.article_lookup_tool = build_article_lookup_tool(article_lookup)
        self.article_data_agent = build_article_data_agent(
            settings,
            article_lookup_tool=self.article_lookup_tool,
        )
        self.article_data_tool = build_article_data_tool(self.article_data_agent)
        self.team_update_agent = build_team_update_agent(
            settings,
            article_data_tool=self.article_data_tool,
        )
        self.team_update_tool = build_team_update_tool(self.team_update_agent)
        self.team_update_batch_agent = build_team_update_batch_agent(
            settings,
            team_update_tool=self.team_update_tool,
        )
        self.hourly_playlist_orchestrator_agent = build_hourly_playlist_orchestrator_agent(settings)
        self.script_generation_stacks = {
            language: self._build_script_generation_stack(settings, language=language)
            for language in SCRIPT_LANGUAGE_CONFIGS
        }
        default_script_stack = self.script_generation_stacks[DEFAULT_SCRIPT_LANGUAGE]
        self.radio_script_writer_agent = default_script_stack.radio_script_writer_agent
        self.radio_story_script_tool = default_script_stack.radio_story_script_tool
        self.hourly_script_batch_agent = default_script_stack.hourly_script_batch_agent

    async def run_team_update(self, context: TeamUpdateRunContext) -> TeamUpdateReport:
        logger.info(
            "Starting team update",
            extra={"run_id": context.run_id, "team": context.request.team},
        )
        await self._prepare_team_context(context)

        if not context.candidate_stories:
            logger.info(
                "No candidate stories for team",
                extra={"run_id": context.run_id, "team": context.request.team},
            )
            report = TeamUpdateReport(
                run_id=context.run_id,
                generated_at=context.generated_at,
                team=context.request.team,
                lookback_minutes=context.request.lookback_minutes,
                status="no_update",
                coverage=self._build_coverage(context),
                warnings=context.warnings,
            )
            self._history_store.save_report(
                report=report,
                batch_run_id=context.run_id,
                source_story_ids=[],
                source_urls=[],
                source_group_ids=[],
            )
            return report

        team_name = str(NFL_TEAMS[context.request.team]["name"])
        logger.info(
            "Running team update agent",
            extra={
                "run_id": context.run_id,
                "team": context.request.team,
                "candidate_count": len(context.candidate_stories),
            },
        )
        result = await Runner.run(
            self.team_update_agent,
            build_team_update_input(context, team_name, context.candidate_stories),
            context=context,
            run_config=build_run_config(
                context.run_id,
                stage="team_update_agent",
                metadata={
                    "team": context.request.team,
                    "candidate_story_count": str(len(context.candidate_stories)),
                },
            ),
            max_turns=max(6, len(context.candidate_stories) + 2),
        )
        package = coerce_output(result.final_output, TeamUpdatePackage)
        report = TeamUpdateReport(
            run_id=context.run_id,
            generated_at=context.generated_at,
            team=context.request.team,
            lookback_minutes=context.request.lookback_minutes,
            status="report_ready",
            report=package,
            coverage=self._build_coverage(context),
            warnings=context.warnings,
        )
        selected_sources = _selected_source_articles(report.report)
        self._persist_team_update_report(
            report=report,
            selected_sources=selected_sources,
            context=context,
            batch_run_id=context.run_id,
        )
        return report

    async def run_team_update_batch(
        self,
        *,
        run_id: str,
        generated_at: datetime,
        request: TeamUpdateBatchRequest,
    ) -> TeamUpdateBatchResponse:
        lookback_hours = max(1, math.ceil(request.lookback_minutes / 60))
        feed_stories = await self._news_feed.fetch_stories(lookback_hours)
        team_codes = request.teams or list(NFL_TEAMS.keys())

        logger.info(
            "Preparing team contexts for batch",
            extra={"run_id": run_id, "team_count": len(team_codes)},
        )
        team_contexts: list[TeamUpdateRunContext] = []
        selected_team_payloads: list[dict] = []
        for team_code in team_codes:
            context = TeamUpdateRunContext(
                request=TeamUpdateReportRequest(
                    team=team_code,
                    lookback_minutes=request.lookback_minutes,
                ),
                run_id=run_id,
                generated_at=generated_at,
            )
            context.feed_stories = list(feed_stories)
            await self._prepare_team_context(context, skip_feed_fetch=True)
            team_contexts.append(context)
            selected_team_payloads.append(
                TeamUpdateTeamInput(
                    team_code=team_code,
                    team_name=str(NFL_TEAMS[team_code]["name"]),
                    lookback_minutes=request.lookback_minutes,
                    candidate_stories=context.candidate_stories,
                ).model_dump(mode="json")
            )

        logger.info(
            "Running team update batch agent",
            extra={"run_id": run_id, "team_count": len(team_contexts)},
        )
        result = await Runner.run(
            self.team_update_batch_agent,
            build_team_update_batch_input(run_id, request, selected_team_payloads),
            run_config=build_run_config(
                run_id,
                stage="team_update_batch_agent",
                metadata={"team_count": str(len(team_contexts))},
            ),
            max_turns=max(8, len(team_contexts) + 4),
        )
        batch_output = coerce_output(result.final_output, TeamUpdateBatchAgentResult)

        context_by_team = {context.request.team: context for context in team_contexts}
        reports: list[TeamUpdateReport] = []
        for item in batch_output.reports:
            context = context_by_team[item.team]
            report = TeamUpdateReport(
                run_id=run_id,
                generated_at=generated_at,
                team=item.team,
                lookback_minutes=item.lookback_minutes,
                status=item.status,
                report=item.report,
                coverage=self._build_coverage(context),
                warnings=context.warnings,
            )
            selected_sources = _selected_source_articles(report.report) if report.report is not None else []
            self._persist_team_update_report(
                report=report,
                selected_sources=selected_sources,
                context=context,
                batch_run_id=run_id,
            )
            reports.append(report)
        batch_response = TeamUpdateBatchResponse(
            run_id=run_id,
            generated_at=generated_at,
            lookback_minutes=request.lookback_minutes,
            reports=reports,
            hourly_playlist=HourlyPlaylist(selected_count=0, items=[]),
        )
        batch_response.hourly_playlist = await self._build_hourly_playlist(batch_response)
        logger.info(
            "Batch run complete",
            extra={"run_id": run_id, "report_count": len(reports)},
        )
        return batch_response

    async def _prepare_team_context(
        self,
        context: TeamUpdateRunContext,
        *,
        skip_feed_fetch: bool = False,
    ) -> None:
        if not skip_feed_fetch:
            lookback_hours = max(1, math.ceil(context.request.lookback_minutes / 60))
            context.feed_stories = await self._news_feed.fetch_stories(lookback_hours)
        team_stories = filter_stories_for_team(context.feed_stories, context.request.team)
        context.team_feed_story_count = len(team_stories)
        context.prior_reports = self._history_store.list_recent_report_ready_entries(
            team_code=context.request.team,
            generated_after=context.generated_at - timedelta(days=7),
        )
        context.candidate_stories = await self._build_candidate_stories(
            team_code=context.request.team,
            stories=team_stories,
            prior_reports=context.prior_reports,
            lookback_minutes=context.request.lookback_minutes,
            context=context,
        )

    def _persist_team_update_report(
        self,
        *,
        report: TeamUpdateReport,
        selected_sources: list,
        context: TeamUpdateRunContext,
        batch_run_id: str,
    ) -> None:
        candidate_by_url = {candidate.url: candidate for candidate in context.candidate_stories}
        self._history_store.save_report(
            report=report,
            batch_run_id=batch_run_id,
            source_story_ids=list(dict.fromkeys(source.story_id for source in selected_sources)),
            source_urls=list(dict.fromkeys(source.url for source in selected_sources)),
            source_group_ids=list(
                dict.fromkeys(
                    candidate_by_url[source.url].group_id
                    for source in selected_sources
                    if source.url in candidate_by_url
                )
            ),
        )

    async def _build_candidate_stories(
        self,
        *,
        team_code: str,
        stories: list[FeedStory],
        prior_reports: list,
        lookback_minutes: int,
        context: TeamUpdateRunContext,
    ) -> list[TeamUpdateCandidate]:
        # Deduplicate stories by id before doing any I/O
        unique_stories: list[FeedStory] = []
        seen_story_ids: set[str] = set()
        for story in stories:
            if story.id not in seen_story_ids:
                seen_story_ids.add(story.id)
                unique_stories.append(story)

        if not unique_stories:
            return []

        # --- Phase 1: Parallel article lookups (bounded by semaphore) ---
        logger.info(
            "Looking up articles in parallel",
            extra={"team": team_code, "story_count": len(unique_stories)},
        )

        async def _lookup_article(story: FeedStory) -> tuple[FeedStory, ArticleContentLookupToolResponse]:
            async with self._lookup_semaphore:
                result = await self._article_lookup.lookup_article(story.url)
                return story, result

        lookup_results = await asyncio.gather(
            *[_lookup_article(story) for story in unique_stories],
            return_exceptions=True,
        )

        # Filter to stories with valid article data and group_id
        stories_with_articles: list[tuple[FeedStory, ArticleContentLookupToolResponse]] = []
        for result in lookup_results:
            if isinstance(result, BaseException):
                logger.warning("Article lookup failed", exc_info=result)
                continue
            story, article_lookup = result
            if not article_lookup.found or article_lookup.article is None:
                context.warnings.append(f"No stored article data found for {story.url}")
                continue
            if not article_lookup.article.group_id:
                context.warnings.append(f"Stored article is missing group_id for {story.url}")
                continue
            stories_with_articles.append((story, article_lookup))

        if not stories_with_articles:
            return []

        # --- Phase 2: Parallel group update fetches (bounded by semaphore) ---
        logger.info(
            "Fetching group updates in parallel",
            extra={"team": team_code, "article_count": len(stories_with_articles)},
        )

        async def _fetch_group_updates(
            story: FeedStory,
            article_lookup: ArticleContentLookupToolResponse,
        ) -> tuple[FeedStory, ArticleContentLookupToolResponse, StoryGroupUpdatesToolResponse]:
            async with self._lookup_semaphore:
                group_updates = await self._story_group_updates.fetch_recent_updates(
                    group_id=article_lookup.article.group_id,
                    lookback_minutes=lookback_minutes,
                )
                return story, article_lookup, group_updates

        group_results = await asyncio.gather(
            *[
                _fetch_group_updates(story, article_lookup)
                for story, article_lookup in stories_with_articles
            ],
            return_exceptions=True,
        )

        # --- Phase 3: Sequential candidate assembly (deterministic, no I/O) ---
        candidate_stories: list[TeamUpdateCandidate] = []
        for result in group_results:
            if isinstance(result, BaseException):
                logger.warning("Group update fetch failed", exc_info=result)
                continue
            story, article_lookup, group_updates = result

            recent_group_updates = list(group_updates.updates)
            if not recent_group_updates:
                fallback_update = _fallback_update_from_article_timestamp(
                    story=story,
                    article_updated_at=article_lookup.article.updated_at,
                    lookback_minutes=lookback_minutes,
                )
                if fallback_update is not None:
                    recent_group_updates.append(fallback_update)
                    context.warnings.append(
                        f"Falling back to article updated_at for group activity on {story.url}"
                    )
            if not recent_group_updates:
                continue

            matched_by = None
            previous_report_generated_at = None
            framing = "new"
            continuity = "new_story"
            matched_report = _find_matching_prior_report(
                prior_reports,
                group_id=article_lookup.article.group_id,
                story_id=story.id,
                url=story.url,
            )
            if matched_report is not None and article_lookup.article.group_id in matched_report.source_group_ids:
                framing = "update"
                matched_by = "group_id"
                previous_report_generated_at = matched_report.generated_at
            elif matched_report is not None and story.id in matched_report.source_story_ids:
                framing = "update"
                matched_by = "story_id"
                previous_report_generated_at = matched_report.generated_at
            elif matched_report is not None and story.url in matched_report.source_urls:
                framing = "update"
                matched_by = "url"
                previous_report_generated_at = matched_report.generated_at

            if matched_report is not None:
                continuity = (
                    "follow_up_to_produced_story"
                    if matched_report.production_status == "put_to_production"
                    else "follow_up_to_tracked_story"
                )

            context.researched_urls.add(story.url)
            context.article_digests.append(
                ArticleDigest(
                    story_id=story.id,
                    team_mentions=[team_code],
                    url=story.url,
                    title=story.title,
                    source_name=story.source_name,
                    category=story.category,
                    summary=(
                        f"Candidate {framing} story with {len(recent_group_updates)} recent group updates "
                        f"and continuity {continuity}."
                    ),
                    key_facts=[
                        f"group_id={article_lookup.article.group_id}",
                        f"recent_group_updates={len(recent_group_updates)}",
                        f"continuity={continuity}",
                    ],
                    confidence=0.6,
                )
            )
            candidate_stories.append(
                TeamUpdateCandidate(
                    story_id=story.id,
                    url=story.url,
                    title=story.title,
                    source_name=story.source_name,
                    category=story.category,
                    team_code=team_code,
                    group_id=article_lookup.article.group_id,
                    framing=framing,
                    continuity=continuity,
                    matched_by=matched_by,
                    previous_report_generated_at=previous_report_generated_at,
                    recent_group_updates=recent_group_updates,
                )
            )

        logger.info(
            "Candidate stories built",
            extra={"team": team_code, "candidate_count": len(candidate_stories)},
        )
        return candidate_stories

    @staticmethod
    def _build_coverage(context: TeamUpdateRunContext):
        return {
            "total_feed_stories": len(context.feed_stories),
            "team_feed_stories": context.team_feed_story_count,
            "candidate_stories": len(context.candidate_stories),
            "new_candidates": sum(1 for candidate in context.candidate_stories if candidate.framing == "new"),
            "update_candidates": sum(
                1 for candidate in context.candidate_stories if candidate.framing == "update"
            ),
            "prior_reports_considered": len(context.prior_reports),
        }

    async def _build_hourly_playlist(
        self,
        batch_response: TeamUpdateBatchResponse,
    ) -> HourlyPlaylist:
        eligible_reports = [report for report in batch_response.reports if report.status == "report_ready"]
        if not eligible_reports:
            empty_playlist = HourlyPlaylist(selected_count=0, items=[])
            self._history_store.save_hourly_playlist(
                batch_run_id=batch_response.run_id,
                generated_at=batch_response.generated_at,
                lookback_minutes=batch_response.lookback_minutes,
                playlist=empty_playlist,
            )
            return empty_playlist

        result = await Runner.run(
            self.hourly_playlist_orchestrator_agent,
            build_hourly_playlist_input(batch_response),
            run_config=build_run_config(
                batch_response.run_id,
                stage="hourly_playlist_orchestrator_agent",
                metadata={"eligible_report_count": str(len(eligible_reports))},
            ),
            max_turns=6,
        )
        playlist_selection = coerce_output(result.final_output, HourlyPlaylistSelection)
        playlist = _validate_hourly_playlist_selection(playlist_selection, eligible_reports)

        playlist_id = self._history_store.save_hourly_playlist(
            batch_run_id=batch_response.run_id,
            generated_at=batch_response.generated_at,
            lookback_minutes=batch_response.lookback_minutes,
            playlist=playlist,
        )
        self._history_store.mark_reports_put_to_production(
            batch_run_id=batch_response.run_id,
            playlist_id=playlist_id,
            ranked_team_codes=[(item.rank, item.team) for item in playlist.items],
        )
        return playlist

    async def run_hourly_playlist_scripts(
        self,
        *,
        script_run_id: str,
        generated_at: datetime,
        request: HourlyPlaylistScriptsRequest,
    ) -> HourlyPlaylistScriptsRunResult:
        script_stack = self.script_generation_stacks[request.language]
        playlist_entry = (
            self._history_store.get_hourly_playlist(playlist_id=request.playlist_id)
            if request.playlist_id is not None
            else self._history_store.get_latest_hourly_playlist()
        )
        if playlist_entry is None:
            raise ValueError("No saved hourly playlist is available for script generation.")

        produced_reports = self._history_store.list_produced_reports_for_playlist(
            playlist_id=playlist_entry.id,
            batch_run_id=playlist_entry.batch_run_id,
        )
        selected_stories = build_radio_script_story_inputs(playlist_entry, produced_reports)

        playlist_items = playlist_entry.playlist_json.get("items", [])
        if playlist_items and len(selected_stories) != len(playlist_items):
            raise ValueError(
                "Saved hourly playlist is missing one or more matching produced team reports."
            )

        if not selected_stories:
            return HourlyPlaylistScriptsRunResult(
                script_run_id=script_run_id,
                playlist_id=playlist_entry.id,
                batch_run_id=playlist_entry.batch_run_id,
                language=request.language,
                generated_at=generated_at,
                scripts=[],
            )

        result = await Runner.run(
            script_stack.hourly_script_batch_agent,
            build_hourly_script_batch_input(
                playlist_entry,
                selected_stories,
                language=request.language,
                personas=script_stack.personas,
            ),
            run_config=build_run_config(
                script_run_id,
                stage="hourly_script_batch_agent",
                metadata={
                    "language": request.language,
                    "playlist_id": playlist_entry.id,
                    "story_count": str(len(selected_stories)),
                },
            ),
            max_turns=max(6, len(selected_stories) + 2),
        )
        batch_output = coerce_output(result.final_output, HourlyPlaylistScriptsBatchAgentResult)
        scripts = _normalize_radio_story_scripts(
            batch_output.scripts,
            selected_stories,
            language=request.language,
            personas=script_stack.personas,
        )
        self._history_store.save_story_scripts(
            script_run_id=script_run_id,
            playlist_id=playlist_entry.id,
            batch_run_id=playlist_entry.batch_run_id,
            generated_at=generated_at,
            scripts=scripts,
        )
        tts_batch = None
        if request.enable_tts:
            tts_batch = await self._run_tts_batch(script_run_id=script_run_id, scripts=scripts)
        return HourlyPlaylistScriptsRunResult(
            script_run_id=script_run_id,
            playlist_id=playlist_entry.id,
            batch_run_id=playlist_entry.batch_run_id,
            language=request.language,
            generated_at=generated_at,
            scripts=scripts,
            tts_batch=tts_batch,
        )

    def _build_script_generation_stack(
        self,
        settings: Settings,
        *,
        language: str,
    ) -> ScriptGenerationStack:
        config = SCRIPT_LANGUAGE_CONFIGS[language]
        writer_agent_name = (
            "Radio Script Writer Agent"
            if language == DEFAULT_SCRIPT_LANGUAGE
            else f"Radio Script Writer Agent ({language})"
        )
        batch_agent_name = (
            "Hourly Script Batch Agent"
            if language == DEFAULT_SCRIPT_LANGUAGE
            else f"Hourly Script Batch Agent ({language})"
        )
        radio_script_writer_agent = build_radio_script_writer_agent(
            settings,
            article_data_tool=self.article_data_tool,
            prompt_name=str(config["writer_prompt"]),
            agent_name=writer_agent_name,
        )
        radio_story_script_tool = build_radio_story_script_tool(
            radio_script_writer_agent,
            personas=list(config["personas"]),
            language=language,
        )
        hourly_script_batch_agent = build_hourly_script_batch_agent(
            settings,
            radio_script_tool=radio_story_script_tool,
            prompt_name=str(config["batch_prompt"]),
            agent_name=batch_agent_name,
        )
        return ScriptGenerationStack(
            language=language,
            personas=list(config["personas"]),
            radio_script_writer_agent=radio_script_writer_agent,
            radio_story_script_tool=radio_story_script_tool,
            hourly_script_batch_agent=hourly_script_batch_agent,
        )

    async def _run_tts_batch(
        self,
        *,
        script_run_id: str,
        scripts: list[RadioStoryScript],
    ):
        if self._tts_batch is None:
            raise RuntimeError("TTS batch workflow is not configured.")

        create_request = build_tts_batch_create_request(scripts, settings=self._settings)
        batch_job = await self._tts_batch.create_batch(create_request)
        completed_batch = await self._wait_for_tts_batch(
            script_run_id=script_run_id,
            batch_id=batch_job.batch_id,
        )
        processed_batch = await self._tts_batch.process_batch(
            build_tts_batch_process_request(
                batch_id=completed_batch.batch_id,
                settings=self._settings,
            )
        )
        if processed_batch.processed_count <= 0:
            raise ValueError(
                f"TTS batch {processed_batch.batch_id} completed without any processed audio items."
            )
        return processed_batch

    async def _wait_for_tts_batch(
        self,
        *,
        script_run_id: str,
        batch_id: str,
    ) -> GeminiTTSBatchJobStatus:
        deadline = asyncio.get_running_loop().time() + self._settings.gemini_tts_batch_timeout_seconds
        while True:
            status = await self._tts_batch.fetch_batch_status(
                GeminiTTSBatchStatusRequest(
                    batch_id=batch_id,
                    credentials=self._build_tts_credentials(),
                )
            )
            logger.info(
                "Checked TTS batch status",
                extra={
                    "run_id": script_run_id,
                    "batch_id": batch_id,
                    "status": status.status,
                },
            )
            if status.status == "JOB_STATE_SUCCEEDED":
                return status
            if status.status in _TTS_TERMINAL_FAILURE_STATES:
                detail = f": {status.error}" if status.error else ""
                raise ValueError(
                    f"TTS batch {batch_id} did not succeed. Final status: {status.status}{detail}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"TTS batch {batch_id} did not finish within "
                    f"{self._settings.gemini_tts_batch_timeout_seconds} seconds."
                )
            await asyncio.sleep(self._settings.gemini_tts_batch_poll_interval_seconds)

    def _build_tts_credentials(self) -> GeminiTTSCredentials | None:
        if self._settings.gemini_api_key is None:
            return None
        return GeminiTTSCredentials(gemini=self._settings.gemini_api_key.get_secret_value())


# ---------------------------------------------------------------------------
# Module-level helper functions (private to this module)
# ---------------------------------------------------------------------------

def _find_matching_prior_report(
    prior_reports: list[TeamUpdateHistoryEntry],
    *,
    group_id: str | None = None,
    story_id: str | None = None,
    url: str | None = None,
) -> TeamUpdateHistoryEntry | None:
    for report in prior_reports:
        if group_id is not None and group_id in report.source_group_ids:
            return report
        if story_id is not None and story_id in report.source_story_ids:
            return report
        if url is not None and url in report.source_urls:
            return report
    return None


def _selected_source_articles(report: TeamUpdatePackage) -> list:
    selected_sources = list(report.source_articles)
    for topic in report.topics:
        selected_sources.extend(topic.source_articles)
    return selected_sources


def _normalize_radio_story_scripts(
    scripts: list[RadioStoryScriptDraft],
    selected_stories,
    *,
    language: str = DEFAULT_SCRIPT_LANGUAGE,
    personas: list[ScriptPersona] | None = None,
) -> list[RadioStoryScript]:
    expected_by_rank = {story.playlist_rank: story for story in selected_stories}
    script_by_rank: dict[int, RadioStoryScriptDraft] = {}
    resolved_personas = personas
    if resolved_personas is None:
        resolved_personas = list(SCRIPT_LANGUAGE_CONFIGS[language]["personas"])
    persona_by_name = {
        str(persona["name"]): persona
        for persona in resolved_personas
    }
    for script in scripts:
        if script.playlist_rank not in expected_by_rank:
            raise ValueError(
                f"Generated scripts contained unexpected playlist rank {script.playlist_rank}."
            )
        if script.playlist_rank in script_by_rank:
            raise ValueError(
                f"Generated scripts contained duplicate playlist rank {script.playlist_rank}."
            )
        script_by_rank[script.playlist_rank] = script

    if len(script_by_rank) != len(expected_by_rank):
        raise ValueError(
            "Generated scripts did not match the expected playlist story count."
        )

    normalized_scripts: list[RadioStoryScript] = []
    for story_input in selected_stories:
        matching_script = script_by_rank.get(story_input.playlist_rank)
        if matching_script is None:
            raise ValueError(
                f"Missing generated script for playlist rank {story_input.playlist_rank} "
                f"(team {story_input.team})."
            )
        persona = persona_by_name.get(matching_script.persona_name)
        if persona is None:
            raise ValueError(
                f"Generated scripts used unknown persona {matching_script.persona_name!r}."
            )
        audio_profile = _normalize_tts_text(matching_script.audio_profile)
        scene = _normalize_tts_text(matching_script.scene)
        director_notes = _normalize_tts_text(matching_script.director_notes)
        fallback_direction = _extract_direction_fields(director_notes)
        pace = _normalize_tts_text(matching_script.pace or str(fallback_direction["pace"]))
        warmth = _normalize_tts_text(matching_script.warmth or str(fallback_direction["warmth"]))
        must_hit = [
            normalized
            for item in matching_script.must_hit
            if (normalized := _normalize_tts_text(item))
        ] or list(fallback_direction["must_hit"])
        pronunciations = [
            ScriptDirectionPronunciation(
                term=_normalize_tts_text(item.term),
                guide=_normalize_tts_text(item.guide),
            )
            for item in matching_script.pronunciations
            if _normalize_tts_text(item.term) and _normalize_tts_text(item.guide)
        ] or list(fallback_direction["pronunciations"])
        intro = _normalize_tts_text(matching_script.intro)
        body = _normalize_tts_text(matching_script.body)
        outro = _normalize_tts_text(matching_script.outro)
        normalized_scripts.append(
            RadioStoryScript(
                team=story_input.team,
                playlist_rank=story_input.playlist_rank,
                language=language,
                headline=_normalize_tts_text(story_input.headline),
                continuity=story_input.continuity,
                persona_name=_normalize_tts_text(matching_script.persona_name),
                persona_backstory=_normalize_tts_text(str(persona["backstory"])),
                persona_specialty=_normalize_tts_text(str(persona["specialty"])),
                voice_name=_normalize_tts_text(str(persona["voice_name"])),
                dialect=_normalize_tts_text(str(persona["dialect"])),
                duration_seconds=matching_script.duration_seconds,
                slug=_normalize_slug(matching_script.slug),
                audio_profile=audio_profile,
                scene=scene,
                director_notes=director_notes,
                pace=pace,
                warmth=warmth,
                must_hit=must_hit,
                pronunciations=pronunciations,
                tts_prompt=_build_tts_prompt(
                    audio_profile=audio_profile,
                    scene=scene,
                    director_notes=director_notes,
                    pace=pace,
                    warmth=warmth,
                    must_hit=must_hit,
                    pronunciations=pronunciations,
                    intro=intro,
                    body=body,
                    outro=outro,
                ),
                intro=intro,
                body=body,
                outro=outro,
                source_articles=story_input.source_articles,
            )
        )

    if len({script.team for script in normalized_scripts}) != len(normalized_scripts):
        raise ValueError("Generated scripts must contain unique teams.")
    if len(normalized_scripts) != len(expected_by_rank):
        raise ValueError("Generated scripts did not match the expected playlist stories.")
    return normalized_scripts


_TTS_TRANSLATION_TABLE: Final[dict[int, str]] = str.maketrans(
    {
        0x2018: "'",
        0x2019: "'",
        0x201A: "'",
        0x201B: "'",
        0x201C: '"',
        0x201D: '"',
        0x201E: '"',
        0x2013: "-",
        0x2014: "-",
        0x2015: "-",
        0x2010: "-",
        0x2011: "-",
        0x2026: "...",
        0x00A0: " ",
    }
)

_DIRECTOR_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<label>pace|warmth|emphasis|intimacy|pronunciation)\s*:\s*(?P<value>.*?)(?=(?:\s+(?:pace|warmth|emphasis|intimacy|pronunciation)\s*:)|$)",
    re.IGNORECASE,
)


def _normalize_tts_text(value: str) -> str:
    normalized = value.translate(_TTS_TRANSLATION_TABLE)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    normalized_lines = [" ".join(line.split()) for line in normalized.split("\n")]
    return "\n".join(line for line in normalized_lines).strip()


def _extract_direction_fields(director_notes: str) -> dict[str, object]:
    sections = {
        match.group("label").lower(): _normalize_tts_text(match.group("value"))
        for match in _DIRECTOR_SECTION_PATTERN.finditer(director_notes)
    }
    emphasis = str(sections.get("emphasis", ""))
    warmth = str(sections.get("warmth", ""))
    intimacy = str(sections.get("intimacy", ""))
    pronunciation = str(sections.get("pronunciation", ""))
    warmth_parts = [part for part in [warmth, intimacy] if part]
    return {
        "pace": str(sections.get("pace", "")),
        "warmth": " ".join(warmth_parts),
        "must_hit": _extract_must_hit_phrases(emphasis),
        "pronunciations": _extract_pronunciations(pronunciation),
    }


def _extract_must_hit_phrases(value: str) -> list[str]:
    return [
        normalized.rstrip(".,;:")
        for item in re.findall(r'"([^"]+)"', value)
        if (normalized := _normalize_tts_text(item))
    ]


def _extract_pronunciations(value: str) -> list[ScriptDirectionPronunciation]:
    pronunciations: list[ScriptDirectionPronunciation] = []
    for raw_item in value.split(","):
        item = _normalize_tts_text(raw_item)
        if not item:
            continue
        match = re.match(r'^"(?P<term>[^"]+)"\s*\((?P<guide>[^)]+)\)[\.;:!?]*$', item)
        if match is None:
            continue
        term = _normalize_tts_text(match.group("term"))
        guide = _normalize_tts_text(match.group("guide"))
        if term and guide:
            pronunciations.append(ScriptDirectionPronunciation(term=term, guide=guide))
    return pronunciations


def _normalize_slug(value: str) -> str:
    normalized = _normalize_tts_text(value).lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-")


def _build_tts_prompt(
    *,
    audio_profile: str,
    scene: str,
    director_notes: str,
    pace: str,
    warmth: str,
    must_hit: list[str],
    pronunciations: list[ScriptDirectionPronunciation],
    intro: str,
    body: str,
    outro: str,
) -> str:
    pronunciation_text = (
        "; ".join(f"{item.term}: {item.guide}" for item in pronunciations)
        if pronunciations
        else "None"
    )
    must_hit_text = "; ".join(must_hit) if must_hit else "None"
    return (
        f"Audio Profile: {audio_profile}\n"
        f"Scene: {scene}\n"
        f"Director's Notes: {director_notes}\n"
        f"Pace: {pace}\n"
        f"Warmth: {warmth}\n"
        f"Must Hit: {must_hit_text}\n"
        f"Pronunciations: {pronunciation_text}\n"
        "Script:\n"
        f"Intro: {intro}\n"
        f"Body: {body}\n"
        f"Outro: {outro}"
    )


def _validate_hourly_playlist_selection(
    playlist: HourlyPlaylistSelection,
    eligible_reports: list[TeamUpdateReport],
) -> HourlyPlaylist:
    eligible_by_team = {report.team: report for report in eligible_reports}
    max_items = min(15, len(eligible_reports))
    min_items = len(eligible_reports) if len(eligible_reports) < 10 else 10

    if not (min_items <= playlist.selected_count <= max_items):
        raise ValueError(
            f"Hourly playlist must select between {min_items} and {max_items} eligible reports; "
            f"received {playlist.selected_count}."
        )

    normalized_items: list[HourlyPlaylistItem] = []
    for index, item in enumerate(playlist.items, start=1):
        report = eligible_by_team.get(item.team)
        if report is None or report.report is None:
            raise ValueError(f"Hourly playlist selected unknown or ineligible team {item.team}.")
        topic = report.report.topics[0]
        normalized_items.append(
            HourlyPlaylistItem(
                rank=index,
                team=report.team,
                headline=report.report.headline,
                framing=topic.framing,
                continuity=topic.continuity,
                production_reason=item.production_reason,
                source_articles=report.report.source_articles,
            )
        )
    return HourlyPlaylist(selected_count=len(normalized_items), items=normalized_items)


def _fallback_update_from_article_timestamp(
    *,
    story: FeedStory,
    article_updated_at,
    lookback_minutes: int,
):
    if article_updated_at is None:
        return None
    cutoff = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
    if article_updated_at < cutoff:
        return None
    return {
        "member_identifier": story.url,
        "added_at": article_updated_at,
    }
