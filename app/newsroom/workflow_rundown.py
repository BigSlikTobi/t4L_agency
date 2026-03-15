# Radio rundown workflow — orchestrates the full-league radio show rundown flow.

from __future__ import annotations

import logging
import os

from agents import Runner

from app.adapters import NewsFeedAdapter, SupabaseArticleLookupAdapter
from app.config import Settings
from app.newsroom.agents import (
    build_rundown_orchestrator_agent,
    build_team_news_agent,
)
from app.newsroom.context import NewsroomRunContext
from app.newsroom.helpers import (
    build_public_rundown,
    build_orchestrator_input,
    build_selected_team_payloads,
    capture_selected_story_coverage,
    capture_source_usage,
    coerce_output,
)
from app.newsroom.tools import (
    build_article_data_tool,
    build_article_lookup_tool,
    build_team_news_tool,
)
from app.newsroom.tracing import build_run_config
from app.schemas import RadioRundown, RadioRundownDraft

logger = logging.getLogger(__name__)


class AgentsWorkflow:
    def __init__(
        self,
        *,
        settings: Settings,
        news_feed: NewsFeedAdapter,
        article_lookup: SupabaseArticleLookupAdapter,
    ) -> None:
        self._news_feed = news_feed
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())

        self.article_lookup_tool = build_article_lookup_tool(article_lookup)
        self.article_data_tool = build_article_data_tool(article_lookup, settings)

        self.team_agent = build_team_news_agent(
            settings,
            article_data_tool=self.article_data_tool,
        )
        self.team_news_tool = build_team_news_tool(self.team_agent)

        self.orchestrator_agent = build_rundown_orchestrator_agent(
            settings,
            team_news_tool=self.team_news_tool,
        )

    async def run_newsroom(self, context: NewsroomRunContext) -> RadioRundown:
        logger.info("Starting newsroom run", extra={"run_id": context.run_id})
        context.feed_stories = await self._news_feed.fetch_stories(context.request.lookback_hours)
        context.researched_urls = set()
        context.article_digests = []

        selected_team_payloads = build_selected_team_payloads(context.request, context.feed_stories)
        if not selected_team_payloads:
            raise ValueError("No teams available for orchestration.")
        capture_selected_story_coverage(context, selected_team_payloads)

        logger.info(
            "Running orchestrator agent",
            extra={"run_id": context.run_id, "team_count": len(selected_team_payloads)},
        )
        result = await Runner.run(
            self.orchestrator_agent,
            build_orchestrator_input(context, selected_team_payloads),
            context=context,
            run_config=build_run_config(
                context.run_id,
                stage="orchestrator_agent",
                metadata={"team_count": str(len(selected_team_payloads))},
            ),
            max_turns=max(8, len(selected_team_payloads) + 4),
            auto_previous_response_id=True,
        )
        draft = coerce_output(result.final_output, RadioRundownDraft)
        public_rundown = build_public_rundown(draft=draft, request=context.request)
        capture_source_usage(context, public_rundown)
        logger.info(
            "Newsroom run complete",
            extra={"run_id": context.run_id, "segment_count": len(public_rundown.segments)},
        )
        return public_rundown
