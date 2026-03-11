# This file defines the context for a newsroom run, including the request details, generated feed stories, 
# researched URLs, article digests, and any warnings encountered during the process.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.schemas import (
    ArticleDigest,
    FeedStory,
    RadioRundownRequest,
    TeamUpdateCandidate,
    TeamUpdateHistoryEntry,
    TeamUpdateReportRequest,
)


@dataclass(slots=True)
class NewsroomRunContext:
    request: RadioRundownRequest
    run_id: str
    generated_at: datetime
    feed_stories: list[FeedStory] = field(default_factory=list)
    researched_urls: set[str] = field(default_factory=set)
    article_digests: list[ArticleDigest] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TeamUpdateRunContext:
    request: TeamUpdateReportRequest
    run_id: str
    generated_at: datetime
    feed_stories: list[FeedStory] = field(default_factory=list)
    researched_urls: set[str] = field(default_factory=set)
    article_digests: list[ArticleDigest] = field(default_factory=list)
    candidate_stories: list[TeamUpdateCandidate] = field(default_factory=list)
    prior_reports: list[TeamUpdateHistoryEntry] = field(default_factory=list)
    team_feed_story_count: int = 0
    warnings: list[str] = field(default_factory=list)
