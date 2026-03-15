# This file defines the Pydantic models (schemas) used throughout the application for data validation and serialization.
# It includes models for representing news stories, article digests, segment candidates, radio rundowns.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.constants import DEFAULT_SCRIPT_LANGUAGE, NFL_TEAMS

ScriptLanguage = Literal["en-US", "de-DE"]


class EntityMatch(BaseModel):
    entity_type: str
    entity_id: str
    matched_name: str


class FeedStory(BaseModel):
    id: str
    url: str
    title: str
    source_name: str
    category: str | None = None
    facts_count: int = 0
    entities: list[EntityMatch] = Field(default_factory=list)


class SourceArticleRef(BaseModel):
    story_id: str
    url: str
    title: str
    source_name: str


class ArticleDigest(BaseModel):
    story_id: str
    team_mentions: list[str] = Field(default_factory=list)
    url: str
    title: str
    source_name: str
    category: str | None = None
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ArticleDigestAgentResult(BaseModel):
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    content_status: Literal["full", "thin", "missing"] = "full"


class TeamNewsStoryInput(BaseModel):
    story_id: str
    url: str
    title: str
    source_name: str
    category: str | None = None


class TeamStoryScore(BaseModel):
    story_id: str
    headline: str
    segment_idea: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    why_this_matters_now: str


class TeamStoryCandidate(TeamStoryScore):
    pass


class TeamAnalysisResult(BaseModel):
    team: str
    scored_stories: list[TeamStoryScore] = Field(default_factory=list)


class SegmentCandidate(BaseModel):
    rank: int = Field(ge=1)
    headline: str
    editorial_angle: str
    teams: list[str] = Field(default_factory=list)
    story_priority: int = Field(ge=1, le=5)
    recommended_duration_seconds: int = Field(ge=0)
    recommended_word_count: int = Field(ge=0)
    summary: str
    key_points: list[str] = Field(default_factory=list)
    segment_idea: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class RundownSegmentDraft(BaseModel):
    rank: int = Field(ge=1)
    headline: str
    editorial_angle: str
    teams: list[str] = Field(default_factory=list)
    story_priority: int = Field(ge=1, le=5)
    recommended_duration_seconds: int = Field(ge=0)
    summary: str
    key_points: list[str] = Field(default_factory=list)
    segment_idea: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SourceCoverage(BaseModel):
    total_feed_stories: int = Field(default=0, ge=0)
    researched_articles: int = Field(default=0, ge=0)
    digested_articles: int = Field(default=0, ge=0)
    teams_with_stories: int = Field(default=0, ge=0)
    teams_without_stories: int = Field(default=0, ge=0)


class RadioRundown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lookback_hours: int = Field(ge=1, le=168)
    target_duration_minutes: int = Field(ge=15, le=240)
    segments: list[SegmentCandidate] = Field(default_factory=list)
    backup_segments: list[SegmentCandidate] = Field(default_factory=list)
    source_coverage: SourceCoverage = Field(default_factory=SourceCoverage)
    warnings: list[str] = Field(default_factory=list)


class RadioRundownDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[RundownSegmentDraft] = Field(default_factory=list)
    backup_segments: list[RundownSegmentDraft] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RadioRundownRequest(BaseModel):
    lookback_hours: int = Field(default=24, ge=1, le=168)
    target_duration_minutes: int = Field(default=60, ge=15, le=240)
    max_segments: int = Field(default=8, ge=1, le=16)
    teams: list[str] | None = None

    @field_validator("teams", mode="before")
    @classmethod
    def _coerce_single_team(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("teams")
    @classmethod
    def _normalize_teams(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        normalized: list[str] = []
        seen: set[str] = set()
        for team in value:
            code = str(team).strip().upper()
            if code not in NFL_TEAMS:
                raise ValueError(
                    f"Unknown team code `{code}`. Expected one of: {', '.join(sorted(NFL_TEAMS))}"
                )
            if code not in seen:
                seen.add(code)
                normalized.append(code)
        return normalized or None


class HealthStatus(BaseModel):
    status: Literal["ok"]
    config_ready: bool
    models: dict[str, str]
    external_services: dict[str, str]


class UsageTelemetryTotals(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageTelemetryBreakdown(BaseModel):
    key: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageTelemetryActiveRun(BaseModel):
    run_id: str
    workflow: str
    stage: str
    started_at: datetime


class UsageTelemetryEvent(BaseModel):
    occurred_at: datetime
    workflow: str
    run_id: str
    stage: str
    agent_name: str
    model: str
    provider: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageTelemetrySnapshot(BaseModel):
    status: Literal["ok"]
    telemetry_enabled: bool
    export_configured: bool
    generated_at: datetime
    totals: UsageTelemetryTotals = Field(default_factory=UsageTelemetryTotals)
    active_runs: list[UsageTelemetryActiveRun] = Field(default_factory=list)
    active_runs_by_stage: dict[str, int] = Field(default_factory=dict)
    usage_by_agent: list[UsageTelemetryBreakdown] = Field(default_factory=list)
    usage_by_model: list[UsageTelemetryBreakdown] = Field(default_factory=list)
    recent_events: list[UsageTelemetryEvent] = Field(default_factory=list)
    applied_filters: dict[str, str | None] = Field(default_factory=dict)


class StoredArticleRecord(BaseModel):
    url: str
    cite_url: str | None = None
    header: str | None = None
    content: str
    description: str | None = None
    author: str | None = None
    category: str | None = None
    quotes: list[str] = Field(default_factory=list)
    embeddings_id: int | None = None
    group_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ArticleContentLookupToolInput(BaseModel):
    url: str


class ArticleContentLookupToolResponse(BaseModel):
    requested_url: str
    found: bool
    article: StoredArticleRecord | None = None


class StoryResearchToolInput(BaseModel):
    story_id: str
    url: str
    title: str
    source_name: str
    category: str | None = None
    team_mentions: list[str] = Field(default_factory=list)


class TeamNewsToolInput(BaseModel):
    team_code: str
    team_name: str
    stories: list[TeamNewsStoryInput] = Field(default_factory=list)


class StoryGroupUpdate(BaseModel):
    member_identifier: str
    added_at: datetime


class StoryGroupUpdatesToolResponse(BaseModel):
    group_id: str
    lookback_minutes: int = Field(ge=1)
    updates: list[StoryGroupUpdate] = Field(default_factory=list)


class TeamUpdateCandidate(BaseModel):
    story_id: str
    url: str
    title: str
    source_name: str
    category: str | None = None
    team_code: str
    group_id: str
    framing: Literal["new", "update"]
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    matched_by: Literal["group_id", "story_id", "url"] | None = None
    previous_report_generated_at: datetime | None = None
    recent_group_updates: list[StoryGroupUpdate] = Field(default_factory=list)


class TeamUpdateTeamInput(BaseModel):
    team_code: str
    team_name: str
    lookback_minutes: int = Field(ge=1, le=10080)
    candidate_stories: list[TeamUpdateCandidate] = Field(default_factory=list)


class TeamUpdateTopic(BaseModel):
    headline: str
    editorial_angle: str
    recommended_duration_seconds: int = Field(ge=1)
    framing: Literal["new", "update"]
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    what_changed: str | None = None
    talking_points: list[str] = Field(default_factory=list)
    source_articles: list[SourceArticleRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_update_fields(self) -> "TeamUpdateTopic":
        if self.framing == "update" and not self.what_changed:
            raise ValueError("what_changed is required when framing=update")
        if self.framing == "update" and self.continuity == "new_story":
            raise ValueError("continuity cannot be new_story when framing=update")
        if self.framing == "new" and self.continuity != "new_story":
            raise ValueError("continuity must be new_story when framing=new")
        if self.framing == "new":
            self.what_changed = None
        return self


class TeamUpdatePackage(BaseModel):
    total_duration_seconds: int = Field(default=180, ge=180, le=180)
    headline: str
    topics: list[TeamUpdateTopic] = Field(default_factory=list)
    source_articles: list[SourceArticleRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_total_duration(self) -> "TeamUpdatePackage":
        if sum(topic.recommended_duration_seconds for topic in self.topics) != self.total_duration_seconds:
            raise ValueError("Topic durations must sum to total_duration_seconds")
        return self


class TeamUpdatePackageTransport(BaseModel):
    total_duration_seconds: int = Field(default=180, ge=1)
    headline: str
    topics: list[TeamUpdateTopic] = Field(default_factory=list)
    source_articles: list[SourceArticleRef] = Field(default_factory=list)


class TeamUpdateGateResult(BaseModel):
    decision: Literal["go", "no_update"]
    skip_reason: str | None = None


class TeamUpdateAgentResult(BaseModel):
    status: Literal["report_ready", "no_update"]
    report: TeamUpdatePackage | None = None
    skip_reason: str | None = None

    @model_validator(mode="after")
    def validate_report_status(self) -> "TeamUpdateAgentResult":
        if self.status == "report_ready" and self.report is None:
            raise ValueError("report is required when status=report_ready")
        if self.status == "no_update":
            self.report = None
        return self


class TeamUpdateCoverage(BaseModel):
    total_feed_stories: int = Field(default=0, ge=0)
    team_feed_stories: int = Field(default=0, ge=0)
    candidate_stories: int = Field(default=0, ge=0)
    new_candidates: int = Field(default=0, ge=0)
    update_candidates: int = Field(default=0, ge=0)
    prior_reports_considered: int = Field(default=0, ge=0)


class TeamUpdateReportRequest(BaseModel):
    team: str
    lookback_minutes: int = Field(default=60, ge=1, le=10080)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        raw = str(value).strip()
        code = raw.upper()
        if code in NFL_TEAMS:
            return code

        lowered = raw.casefold()
        for team_code, meta in NFL_TEAMS.items():
            if lowered == str(meta["name"]).casefold():
                return team_code
            if lowered in {alias.casefold() for alias in meta.get("aliases", [])}:
                return team_code

        raise ValueError(
            f"Unknown team identifier `{raw}`. Expected one of: {', '.join(sorted(NFL_TEAMS))}"
        )


class TeamUpdateBatchRequest(BaseModel):
    teams: list[str] | None = None
    lookback_minutes: int = Field(default=60, ge=1, le=10080)

    @field_validator("teams", mode="before")
    @classmethod
    def coerce_single_team(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("teams")
    @classmethod
    def normalize_teams(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for team in value:
            code = TeamUpdateReportRequest(team=team).team
            if code not in seen:
                seen.add(code)
                normalized.append(code)
        return normalized or None


class TeamUpdateBatchAgentReport(BaseModel):
    team: str
    lookback_minutes: int = Field(ge=1, le=10080)
    status: Literal["report_ready", "no_update"]
    report: TeamUpdatePackage | None = None

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team

    @model_validator(mode="after")
    def validate_report_status(self) -> "TeamUpdateBatchAgentReport":
        if self.status == "report_ready" and self.report is None:
            raise ValueError("report is required when status=report_ready")
        if self.status == "no_update":
            self.report = None
        return self


class TeamUpdateBatchAgentTransportReport(BaseModel):
    team: str
    lookback_minutes: int = Field(ge=1, le=10080)
    status: Literal["report_ready", "no_update"]
    report: TeamUpdatePackageTransport | None = None

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team

    @model_validator(mode="after")
    def validate_report_status(self) -> "TeamUpdateBatchAgentTransportReport":
        if self.status == "report_ready" and self.report is None:
            raise ValueError("report is required when status=report_ready")
        if self.status == "no_update":
            self.report = None
        return self


class TeamUpdateBatchAgentResult(BaseModel):
    reports: list[TeamUpdateBatchAgentReport] = Field(default_factory=list)


class TeamUpdateBatchAgentTransportResult(BaseModel):
    reports: list[TeamUpdateBatchAgentTransportReport] = Field(default_factory=list)


class TeamUpdateBatchAgentNanoReport(BaseModel):
    team: str
    lookback_minutes: int = Field(ge=1, le=10080)
    status: Literal["report_ready", "no_update"]

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class TeamUpdateBatchAgentNanoResult(BaseModel):
    reports: list[TeamUpdateBatchAgentNanoReport] = Field(default_factory=list)


class HourlyPlaylistItem(BaseModel):
    rank: int = Field(ge=1)
    team: str
    headline: str
    framing: Literal["new", "update"]
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    production_reason: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team

    @model_validator(mode="after")
    def validate_continuity(self) -> "HourlyPlaylistItem":
        if self.framing == "new" and self.continuity != "new_story":
            raise ValueError("continuity must be new_story when framing=new")
        if self.framing == "update" and self.continuity == "new_story":
            raise ValueError("continuity cannot be new_story when framing=update")
        return self


class HourlyPlaylistSelectionItem(BaseModel):
    rank: int = Field(ge=1)
    team: str
    production_reason: str

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class HourlyPlaylistSelection(BaseModel):
    selected_count: int = Field(ge=0, le=15)
    items: list[HourlyPlaylistSelectionItem] = Field(default_factory=list)


class HourlyPlaylist(BaseModel):
    selected_count: int = Field(ge=0, le=15)
    items: list[HourlyPlaylistItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> "HourlyPlaylist":
        if self.selected_count != len(self.items):
            raise ValueError("selected_count must match the number of items")
        if len(self.items) > 15:
            raise ValueError("hourly playlist cannot contain more than 15 items")

        ranks = [item.rank for item in self.items]
        if len(set(ranks)) != len(ranks):
            raise ValueError("hourly playlist ranks must be unique")
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("hourly playlist ranks must be sequential starting at 1")

        teams = [item.team for item in self.items]
        if len(set(teams)) != len(teams):
            raise ValueError("hourly playlist cannot contain duplicate teams")
        return self


class TeamUpdateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    team: str
    lookback_minutes: int = Field(ge=1, le=10080)
    status: Literal["report_ready", "no_update"]
    report: TeamUpdatePackage | None = None
    coverage: TeamUpdateCoverage = Field(default_factory=TeamUpdateCoverage)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_status(self) -> "TeamUpdateReport":
        if self.status == "report_ready" and self.report is None:
            raise ValueError("report is required when status=report_ready")
        if self.status == "no_update":
            self.report = None
        return self


class TeamUpdateBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lookback_minutes: int = Field(ge=1, le=10080)
    reports: list[TeamUpdateReport] = Field(default_factory=list)
    hourly_playlist: HourlyPlaylist = Field(
        default_factory=lambda: HourlyPlaylist(selected_count=0, items=[])
    )


class HourlyPlaylistScriptsRequest(BaseModel):
    playlist_id: str | None = None
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    enable_tts: bool = True


class RadioStoryScript(BaseModel):
    team: str
    playlist_rank: int = Field(ge=1)
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    headline: str
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    persona_name: str
    persona_backstory: str
    persona_specialty: str
    voice_name: str
    dialect: str
    duration_seconds: int = Field(ge=1, le=240)
    slug: str
    audio_profile: str
    scene: str
    director_notes: str
    pace: str = ""
    warmth: str = ""
    must_hit: list[str] = Field(default_factory=list)
    pronunciations: list["ScriptDirectionPronunciation"] = Field(default_factory=list)
    tts_prompt: str
    intro: str
    body: str
    outro: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class ScriptDirectionPronunciation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    guide: str


class ScriptDirection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_profile: str = ""
    scene: str = ""
    director_notes: str = ""
    pace: str = ""
    warmth: str = ""
    must_hit: list[str] = Field(default_factory=list)
    pronunciations: list[ScriptDirectionPronunciation] = Field(default_factory=list)


class ScriptSections(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intro: str = ""
    body: str = ""
    outro: str = ""


class TTSBatchItemTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TTSBatchAggregateTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reported_item_count: int = 0


class TTSBatchAudioItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    storage_path: str
    mime_type: str
    source_mime_type: str
    public_url: str
    token_usage: TTSBatchItemTokenUsage = Field(default_factory=TTSBatchItemTokenUsage)


class TTSBatchFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    error: str
    token_usage: TTSBatchItemTokenUsage = Field(default_factory=TTSBatchItemTokenUsage)


class TTSBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    status: str
    processed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    token_usage: TTSBatchAggregateTokenUsage = Field(default_factory=TTSBatchAggregateTokenUsage)
    local_usage_summary_file: str | None = None
    usage_summary_path: str | None = None
    usage_summary_public_url: str | None = None
    manifest_path: str | None = None
    manifest_public_url: str | None = None
    items: list[TTSBatchAudioItem] = Field(default_factory=list)
    failures: list[TTSBatchFailure] = Field(default_factory=list)


class GeminiTTSCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gemini: str


class GeminiTTSSupabaseStorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    key: str
    bucket: str
    path_prefix: str


class GeminiTTSBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    voice_name: str
    direction: ScriptDirection = Field(default_factory=ScriptDirection)
    script: ScriptSections = Field(default_factory=ScriptSections)


class GeminiTTSBatchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create"] = "create"
    model_name: str
    voice_name: str
    items: list[GeminiTTSBatchItem] = Field(default_factory=list)
    credentials: GeminiTTSCredentials | None = None


class GeminiTTSBatchStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["status"] = "status"
    batch_id: str
    credentials: GeminiTTSCredentials | None = None


class GeminiTTSBatchProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["process"] = "process"
    batch_id: str
    credentials: GeminiTTSCredentials | None = None
    supabase: GeminiTTSSupabaseStorageConfig


class GeminiTTSBatchJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    error: str | None = None
    model_name: str | None = None
    total_items: int | None = None
    input_file_name: str | None = None
    local_input_file: str | None = None
    supported_generation_methods: list[str] = Field(default_factory=list)


class HourlyPlaylistScriptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    direction: ScriptDirection = Field(default_factory=ScriptDirection)
    script: ScriptSections = Field(default_factory=ScriptSections)


class HourlyPlaylistScriptsRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_run_id: str
    playlist_id: str
    batch_run_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scripts: list[RadioStoryScript] = Field(default_factory=list)
    tts_batch: TTSBatchResult | None = None


class HourlyPlaylistScriptsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    voice_name: str = ""
    items: list[HourlyPlaylistScriptItem] = Field(default_factory=list)
    tts_batch: TTSBatchResult | None = None

    @classmethod
    def from_run_result(
        cls,
        result: "HourlyPlaylistScriptsRunResult",
    ) -> "HourlyPlaylistScriptsResponse":
        voice_name = result.scripts[0].voice_name if result.scripts else ""
        return cls(
            language=result.language,
            voice_name=voice_name,
            items=[
                HourlyPlaylistScriptItem(
                    id=script.slug,
                    title=script.headline,
                    direction=ScriptDirection(
                        audio_profile=script.audio_profile,
                        scene=script.scene,
                        director_notes=script.director_notes,
                        pace=script.pace,
                        warmth=script.warmth,
                        must_hit=script.must_hit,
                        pronunciations=script.pronunciations,
                    ),
                    script=ScriptSections(
                        intro=script.intro,
                        body=script.body,
                        outro=script.outro,
                    ),
                )
                for script in result.scripts
            ],
            tts_batch=result.tts_batch,
        )


class QAPlayerFeedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    playlist_rank: int = Field(ge=1)
    team: str = ""
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    voice_name: str = ""
    duration_seconds: int | None = Field(default=None, ge=1, le=240)
    audio_url: str | None = None
    intro: str = ""
    body: str = ""
    outro: str = ""
    script_text: str = ""


class QAPlayerBatchOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    generated_at: datetime
    script_run_id: str = ""
    playlist_id: str = ""
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    item_count: int = Field(default=0, ge=0)


class QAPlayerFeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["artifact", "history", "empty"]
    generated_at: datetime | None = None
    selected_batch: str = ""
    script_run_id: str = ""
    playlist_id: str = ""
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    has_audio: bool = False
    available_batches: list[QAPlayerBatchOption] = Field(default_factory=list)
    items: list[QAPlayerFeedItem] = Field(default_factory=list)


class RadioNarrativeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_kind: str = "team_story"
    segment_role: Literal["opener", "middle", "closer", "standalone"] = "standalone"
    hour_theme: str = ""
    hour_narrative_brief: str = ""
    story_thread: str = ""
    primary_angle: str = ""
    callback_budget: str = ""
    already_aired_summary: str = ""
    facts_already_aired: list[str] = Field(default_factory=list)
    fresh_material_to_emphasize: list[str] = Field(default_factory=list)
    previous_segment_headline: str = ""
    previous_segment_takeaway: str = ""
    handoff_in: str = ""
    handoff_out: str = ""
    next_segment_headline: str = ""
    next_segment_tease: str = ""


class RadioStoryScriptInput(BaseModel):
    team: str
    playlist_rank: int = Field(ge=1)
    headline: str
    framing: Literal["new", "update"]
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    production_reason: str
    story_synopsis: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)
    narrative_context: RadioNarrativeContext = Field(default_factory=RadioNarrativeContext)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class BatchPersonaOption(BaseModel):
    name: str
    specialty: str
    voice_name: str
    dialect: str


class HourlyScriptBatchInput(BaseModel):
    playlist_id: str
    batch_run_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    hour_narrative_brief: str = ""
    persona_roster: list[BatchPersonaOption] = Field(default_factory=list)
    selected_stories: list[RadioStoryScriptInput] = Field(default_factory=list)


class RadioStoryScriptToolInput(BaseModel):
    team: str
    playlist_rank: int = Field(ge=1)
    headline: str
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    production_reason: str
    story_synopsis: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)
    persona_name: str
    narrative_context: RadioNarrativeContext = Field(default_factory=RadioNarrativeContext)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class RadioStoryScriptWriterInput(BaseModel):
    team: str
    playlist_rank: int = Field(ge=1)
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    headline: str
    continuity: Literal["new_story", "follow_up_to_tracked_story", "follow_up_to_produced_story"]
    production_reason: str
    story_synopsis: str
    persona_name: str
    persona_backstory: str
    persona_specialty: str
    voice_name: str
    dialect: str
    source_articles: list[SourceArticleRef] = Field(default_factory=list)
    narrative_context: RadioNarrativeContext = Field(default_factory=RadioNarrativeContext)

    @field_validator("team")
    @classmethod
    def normalize_team(cls, value: str) -> str:
        return TeamUpdateReportRequest(team=value).team


class RadioStoryScriptDraft(BaseModel):
    playlist_rank: int = Field(ge=1)
    persona_name: str
    duration_seconds: int = Field(ge=1, le=240)
    slug: str
    audio_profile: str
    scene: str
    director_notes: str
    pace: str = ""
    warmth: str = ""
    must_hit: list[str] = Field(default_factory=list)
    pronunciations: list[ScriptDirectionPronunciation] = Field(default_factory=list)
    intro: str
    body: str
    outro: str


class HourlyPlaylistScriptsBatchAgentResult(BaseModel):
    scripts: list[RadioStoryScriptDraft] = Field(default_factory=list)


class HourlyNarrativePlannerInput(BaseModel):
    playlist_id: str
    batch_run_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    selected_stories: list[RadioStoryScriptInput] = Field(default_factory=list)
    prior_narrative_plan: HourlyNarrativePlan | None = None


class HourlyNarrativeSegmentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    playlist_rank: int = Field(ge=1)
    segment_kind: str = "team_story"
    segment_role: Literal["opener", "middle", "closer", "standalone"]
    narrative_focus: str
    story_thread: str = ""
    primary_angle: str = ""
    callback_budget: str = ""
    already_aired_summary: str = ""
    facts_already_aired: list[str] = Field(default_factory=list)
    fresh_material_to_emphasize: list[str] = Field(default_factory=list)
    handoff_in: str = ""
    handoff_out: str = ""
    previous_segment_headline: str = ""
    previous_segment_takeaway: str = ""
    next_segment_headline: str = ""
    next_segment_tease: str = ""


class HourlyNarrativePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour_theme: str
    hour_narrative_brief: str
    segments: list[HourlyNarrativeSegmentPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_segments(self) -> "HourlyNarrativePlan":
        if not self.segments:
            return self
        ranks = [segment.playlist_rank for segment in self.segments]
        if len(set(ranks)) != len(ranks):
            raise ValueError("hourly narrative plan ranks must be unique")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("hourly narrative plan ranks must be sequential starting at 1")
        return self


class TeamUpdateHistoryEntry(BaseModel):
    id: str
    batch_run_id: str
    team_code: str
    generated_at: datetime
    status: Literal["report_ready", "no_update"]
    production_status: Literal["tracked_only", "put_to_production"] = "tracked_only"
    production_rank: int | None = None
    playlist_id: str | None = None
    source_story_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    source_group_ids: list[str] = Field(default_factory=list)
    report_json: dict | None = None


class HourlyPlaylistHistoryEntry(BaseModel):
    id: str
    batch_run_id: str
    generated_at: datetime
    lookback_minutes: int = Field(ge=1, le=10080)
    selected_count: int = Field(ge=0, le=15)
    playlist_json: dict


class HourlyStoryScriptHistoryEntry(BaseModel):
    id: str
    script_run_id: str
    playlist_id: str
    batch_run_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    team_code: str
    playlist_rank: int = Field(ge=1)
    generated_at: datetime
    duration_seconds: int = Field(ge=1, le=240)
    script_json: dict


class HourlyStoryScriptBatchHistoryEntry(BaseModel):
    batch_run_id: str
    script_run_id: str
    playlist_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    generated_at: datetime
    item_count: int = Field(ge=0)


class HourlyNarrativePlanHistoryEntry(BaseModel):
    id: str
    script_run_id: str
    playlist_id: str
    batch_run_id: str
    language: ScriptLanguage = DEFAULT_SCRIPT_LANGUAGE
    generated_at: datetime
    narrative_plan_json: dict
