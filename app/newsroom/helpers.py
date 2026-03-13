# This file contains helper functions for the newsroom workflow, including team selection, story filtering, 
# payload construction, and output processing. It makes sure that the agents receive the correct input and that 
# the outputs are properly captured and validated.

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.constants import (
    DEFAULT_SCRIPT_LANGUAGE,
    DEFAULT_SPOKEN_WORDS_PER_MINUTE,
    NFL_TEAMS,
    ScriptPersona,
)
from app.newsroom.context import NewsroomRunContext, TeamUpdateRunContext
from app.schemas import (
    ArticleDigest,
    FeedStory,
    HourlyNarrativePlan,
    HourlyNarrativePlannerInput,
    GeminiTTSBatchCreateRequest,
    GeminiTTSCredentials,
    GeminiTTSBatchItem,
    GeminiTTSBatchProcessRequest,
    GeminiTTSSupabaseStorageConfig,
    HourlyScriptBatchInput,
    HourlyPlaylistHistoryEntry,
    HourlyPlaylistScriptsBatchAgentResult,
    RadioRundown,
    RadioNarrativeContext,
    RadioRundownDraft,
    RadioStoryScript,
    RadioStoryScriptWriterInput,
    RadioStoryScriptInput,
    RadioRundownRequest,
    RundownSegmentDraft,
    SegmentCandidate,
    TeamUpdateBatchResponse,
    TeamUpdateBatchRequest,
    TeamUpdateCandidate,
    TeamUpdateHistoryEntry,
    TeamUpdatePackage,
)

T = TypeVar("T")


def selected_teams(request: RadioRundownRequest) -> dict[str, dict[str, object]]:
    if request.teams is None:
        return NFL_TEAMS
    return {team_code: NFL_TEAMS[team_code] for team_code in request.teams}


def filter_stories_for_team(stories: list[FeedStory], team_code: str) -> list[FeedStory]:
    return [
        story
        for story in stories
        if any(
            entity.entity_type == "team" and entity.entity_id == team_code
            for entity in story.entities
        )
    ]


def team_mentions_for_story(story: FeedStory) -> list[str]:
    mentions: list[str] = []
    for entity in story.entities:
        if entity.entity_type == "team" and entity.entity_id not in mentions:
            mentions.append(entity.entity_id)
    return mentions


def build_selected_team_payloads(
    request: RadioRundownRequest,
    stories: list[FeedStory],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for team_code, meta in selected_teams(request).items():
        team_stories = filter_stories_for_team(stories, team_code)
        payloads.append(
            {
                "team_code": team_code,
                "team_name": str(meta["name"]),
                "stories": [
                    {
                        "story_id": story.id,
                        "url": story.url,
                        "title": story.title,
                        "source_name": story.source_name,
                        "category": story.category,
                    }
                    for story in team_stories
                ],
            }
        )
    return payloads


def build_orchestrator_input(
    context: NewsroomRunContext,
    selected_team_payloads: list[dict[str, Any]],
) -> str:
    payload = {
        "lookback_hours": context.request.lookback_hours,
        "target_duration_minutes": context.request.target_duration_minutes,
        "max_segments": context.request.max_segments,
        "selected_teams": selected_team_payloads,
    }
    return (
        "Create the final NFL radio rundown from these selected teams.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def capture_selected_story_coverage(
    context: NewsroomRunContext,
    selected_team_payloads: list[dict[str, Any]],
) -> None:
    story_map = {story.id: story for story in context.feed_stories}
    unique_stories: dict[str, FeedStory] = {}
    for team in selected_team_payloads:
        for raw_story in team["stories"]:
            story_id = str(raw_story.get("story_id") or raw_story.get("id") or "")
            if not story_id:
                continue
            story = story_map.get(story_id)
            if story is None:
                if "id" in raw_story and "entities" in raw_story:
                    story = FeedStory.model_validate(raw_story)
                else:
                    story = FeedStory(
                        id=story_id,
                        url=str(raw_story["url"]),
                        title=str(raw_story["title"]),
                        source_name=str(raw_story["source_name"]),
                        category=raw_story.get("category"),
                        entities=[],
                    )
            unique_stories[story.id] = story

    context.researched_urls = {story.url for story in unique_stories.values()}
    context.article_digests = [
        ArticleDigest(
            story_id=story.id,
            team_mentions=team_mentions_for_story(story),
            url=story.url,
            title=story.title,
            source_name=story.source_name,
            category=story.category,
            summary="Queued for team-level scoring.",
            key_facts=[],
            confidence=0.5,
        )
        for story in unique_stories.values()
    ]


def capture_source_usage(
    context: NewsroomRunContext,
    draft: RadioRundown,
) -> None:
    story_map = {story.id: story for story in context.feed_stories}
    seen_story_ids = {digest.story_id for digest in context.article_digests}

    for segment in draft.segments + draft.backup_segments:
        for source in segment.source_articles:
            context.researched_urls.add(source.url)
            if source.story_id in seen_story_ids:
                continue
            story = story_map.get(source.story_id)
            if story is None:
                continue
            context.article_digests.append(
                ArticleDigest(
                    story_id=story.id,
                    team_mentions=team_mentions_for_story(story),
                    url=story.url,
                    title=story.title,
                    source_name=story.source_name,
                    category=story.category,
                    summary="Referenced in final rundown.",
                    key_facts=[],
                    confidence=0.5,
                )
            )
            seen_story_ids.add(source.story_id)


def _compact_candidate(candidate: TeamUpdateCandidate) -> dict:
    data = candidate.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"recent_group_updates"},
    )
    data["recent_group_update_count"] = len(candidate.recent_group_updates)
    if candidate.recent_group_updates:
        data["latest_group_update_at"] = candidate.recent_group_updates[0].added_at.isoformat()
    return data


def build_team_update_input(
    context: TeamUpdateRunContext,
    team_name: str,
    candidate_stories: list[TeamUpdateCandidate],
) -> str:
    payload = {
        "team_code": context.request.team,
        "team_name": team_name,
        "lookback_minutes": context.request.lookback_minutes,
        "candidate_stories": [_compact_candidate(candidate) for candidate in candidate_stories],
    }
    return (
        "Create a 3-minute NFL team update report from these vetted candidate stories.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_team_update_batch_input(
    run_id: str,
    request: TeamUpdateBatchRequest,
    selected_team_payloads: list[dict[str, Any]],
) -> str:
    payload = {
        "lookback_minutes": request.lookback_minutes,
        "selected_teams": selected_team_payloads,
    }
    return (
        "Create team update reports for these selected teams in one batch.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_hourly_playlist_input(
    batch_response: TeamUpdateBatchResponse,
) -> str:
    eligible_reports: list[dict[str, Any]] = []
    for report in batch_response.reports:
        if report.status != "report_ready" or report.report is None or not report.report.topics:
            continue
        topic = report.report.topics[0]
        synopsis_parts = [topic.headline, topic.editorial_angle]
        if topic.what_changed:
            synopsis_parts.append(topic.what_changed)
        eligible_reports.append(
            {
                "team": report.team,
                "headline": report.report.headline,
                "framing": topic.framing,
                "continuity": topic.continuity,
                "synopsis": " ".join(part for part in synopsis_parts if part),
                "source_articles": [article.model_dump(mode="json", exclude_none=True) for article in report.report.source_articles],
            }
        )
    payload = {
        "lookback_minutes": batch_response.lookback_minutes,
        "eligible_reports": eligible_reports,
    }
    return (
        "Select the final hourly production playlist from these team update reports.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_hourly_script_batch_input(
    playlist: HourlyPlaylistHistoryEntry,
    selected_stories: list[RadioStoryScriptInput],
    *,
    language: str = DEFAULT_SCRIPT_LANGUAGE,
    personas: list[ScriptPersona],
    hour_narrative_brief: str = "",
) -> str:
    payload = HourlyScriptBatchInput(
        playlist_id=playlist.id,
        batch_run_id=playlist.batch_run_id,
        language=language,
        hour_narrative_brief=hour_narrative_brief,
        persona_roster=[
            {
                "name": str(persona["name"]),
                "specialty": str(persona["specialty"]),
                "voice_name": str(persona["voice_name"]),
                "dialect": str(persona["dialect"]),
            }
            for persona in personas
        ],
        selected_stories=selected_stories,
    ).model_dump(mode="json", exclude_defaults=True)
    return (
        "Write radio scripts for the selected hourly playlist stories.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_hourly_narrative_planner_input(
    playlist: HourlyPlaylistHistoryEntry,
    selected_stories: list[RadioStoryScriptInput],
    *,
    language: str = DEFAULT_SCRIPT_LANGUAGE,
    prior_narrative_plan: HourlyNarrativePlan | None = None,
) -> str:
    payload = HourlyNarrativePlannerInput(
        playlist_id=playlist.id,
        batch_run_id=playlist.batch_run_id,
        language=language,
        selected_stories=selected_stories,
        prior_narrative_plan=prior_narrative_plan,
    ).model_dump(mode="json", exclude_defaults=True)
    return (
        "Plan the narrative arc for this saved hourly playlist before script writing.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_radio_script_story_inputs(
    playlist: HourlyPlaylistHistoryEntry,
    produced_reports: list[TeamUpdateHistoryEntry],
) -> list[RadioStoryScriptInput]:
    playlist_items = {
        str(item["team"]): item
        for item in playlist.playlist_json.get("items", [])
    }
    story_inputs: list[RadioStoryScriptInput] = []
    for produced_report in produced_reports:
        playlist_item = playlist_items.get(produced_report.team_code)
        if playlist_item is None or produced_report.report_json is None:
            continue
        package = TeamUpdatePackage.model_validate(produced_report.report_json)
        source_articles = list(package.source_articles)
        topic = package.topics[0]
        source_articles.extend(topic.source_articles)
        deduped_source_articles = list(
            {
                (article.story_id, article.url): article
                for article in source_articles
            }.values()
        )
        synopsis_parts = [package.headline, topic.headline, topic.editorial_angle]
        if topic.what_changed:
            synopsis_parts.append(topic.what_changed)
        story_inputs.append(
            RadioStoryScriptInput(
                team=produced_report.team_code,
                playlist_rank=int(playlist_item["rank"]),
                headline=str(playlist_item["headline"]),
                framing=str(playlist_item["framing"]),
                continuity=str(playlist_item["continuity"]),
                production_reason=str(playlist_item["production_reason"]),
                story_synopsis=" ".join(part for part in synopsis_parts if part),
                source_articles=deduped_source_articles,
            )
        )
    return sorted(story_inputs, key=lambda item: item.playlist_rank)


def apply_hourly_narrative_plan(
    selected_stories: list[RadioStoryScriptInput],
    narrative_plan: HourlyNarrativePlan,
) -> list[RadioStoryScriptInput]:
    if not selected_stories:
        return []

    plan_by_rank = {
        segment.playlist_rank: segment
        for segment in narrative_plan.segments
    }
    if set(plan_by_rank) != {story.playlist_rank for story in selected_stories}:
        raise ValueError("Narrative plan ranks did not match the selected hourly playlist stories.")

    enriched_stories: list[RadioStoryScriptInput] = []
    for story in selected_stories:
        segment_plan = plan_by_rank[story.playlist_rank]
        enriched_stories.append(
            story.model_copy(
                update={
                    "narrative_context": RadioNarrativeContext(
                        segment_kind=segment_plan.segment_kind,
                        segment_role=segment_plan.segment_role,
                        hour_theme=narrative_plan.hour_theme,
                        hour_narrative_brief=narrative_plan.hour_narrative_brief,
                        story_thread=segment_plan.story_thread,
                        primary_angle=segment_plan.primary_angle,
                        callback_budget=segment_plan.callback_budget,
                        already_aired_summary=segment_plan.already_aired_summary,
                        facts_already_aired=segment_plan.facts_already_aired,
                        fresh_material_to_emphasize=segment_plan.fresh_material_to_emphasize,
                        previous_segment_headline=segment_plan.previous_segment_headline,
                        previous_segment_takeaway=segment_plan.previous_segment_takeaway,
                        handoff_in=segment_plan.handoff_in,
                        handoff_out=segment_plan.handoff_out,
                        next_segment_headline=segment_plan.next_segment_headline,
                        next_segment_tease=segment_plan.next_segment_tease,
                    ),
                    "story_synopsis": " ".join(
                        part
                        for part in [story.story_synopsis, segment_plan.narrative_focus]
                        if part
                    ),
                }
            )
        )
    return enriched_stories


def build_radio_script_input(
    story_input: RadioStoryScriptWriterInput,
) -> str:
    payload = story_input.model_dump(mode="json", exclude_defaults=True)
    return (
        "Write one single-anchor radio script for this hourly playlist story.\n\n"
        f"Input JSON:\n{json.dumps(payload, separators=(',', ':'))}"
    )


def build_tts_batch_create_request(
    scripts: list[RadioStoryScript],
    *,
    settings: Settings,
) -> GeminiTTSBatchCreateRequest:
    if not scripts:
        raise ValueError("Cannot create a TTS batch without scripts.")

    credentials = (
        GeminiTTSCredentials(gemini=settings.gemini_api_key.get_secret_value())
        if settings.gemini_api_key is not None
        else None
    )
    return GeminiTTSBatchCreateRequest(
        model_name=settings.gemini_tts_batch_model_name,
        voice_name=scripts[0].voice_name,
        items=[
            GeminiTTSBatchItem(
                id=script.slug,
                title=script.headline,
                voice_name=script.voice_name,
                direction={
                    "audio_profile": script.audio_profile,
                    "scene": script.scene,
                    "director_notes": script.director_notes,
                    "pace": script.pace,
                    "warmth": script.warmth,
                    "must_hit": script.must_hit,
                    "pronunciations": [item.model_dump(mode="json") for item in script.pronunciations],
                },
                script={
                    "intro": script.intro,
                    "body": script.body,
                    "outro": script.outro,
                },
            )
            for script in scripts
        ],
        credentials=credentials,
    )


def build_tts_batch_process_request(
    *,
    batch_id: str,
    settings: Settings,
) -> GeminiTTSBatchProcessRequest:
    credentials = (
        GeminiTTSCredentials(gemini=settings.gemini_api_key.get_secret_value())
        if settings.gemini_api_key is not None
        else None
    )
    return GeminiTTSBatchProcessRequest(
        batch_id=batch_id,
        credentials=credentials,
        supabase=GeminiTTSSupabaseStorageConfig(
            url=settings.resolved_supabase_storage_url(),
            key=settings.resolved_supabase_storage_key(),
            bucket=settings.supabase_storage_bucket,
            path_prefix=settings.supabase_storage_path_prefix,
        ),
    )


def build_public_rundown(
    *,
    draft: RadioRundownDraft,
    request: RadioRundownRequest,
) -> RadioRundown:
    return RadioRundown(
        lookback_hours=request.lookback_hours,
        target_duration_minutes=request.target_duration_minutes,
        segments=[_materialize_segment(segment) for segment in draft.segments],
        backup_segments=[_materialize_segment(segment) for segment in draft.backup_segments],
        warnings=draft.warnings,
    )


def _materialize_segment(segment: RundownSegmentDraft) -> SegmentCandidate:
    return SegmentCandidate(
        rank=segment.rank,
        headline=segment.headline,
        editorial_angle=segment.editorial_angle,
        teams=segment.teams,
        story_priority=segment.story_priority,
        recommended_duration_seconds=segment.recommended_duration_seconds,
        recommended_word_count=_recommended_word_count(segment.recommended_duration_seconds),
        summary=segment.summary,
        key_points=segment.key_points,
        segment_idea=segment.segment_idea,
        source_articles=segment.source_articles,
        confidence=segment.confidence,
    )


def _recommended_word_count(duration_seconds: int) -> int:
    return round((duration_seconds / 60) * DEFAULT_SPOKEN_WORDS_PER_MINUTE)


async def extract_json_output(run_result: Any) -> str:
    output = getattr(run_result, "final_output", run_result)
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    return json.dumps(output)


def coerce_output(payload: Any, schema: type[T]) -> T:
    if isinstance(payload, schema):
        return payload
    if isinstance(payload, BaseModel):
        try:
            normalized = payload.model_dump(mode="json")
            filtered = {
                field_name: normalized[field_name]
                for field_name in schema.model_fields
                if field_name in normalized
            }
            return schema.model_validate(filtered)
        except ValidationError as exc:
            raise ValueError(
                f"Agent returned invalid payload for {schema.__name__}: {exc}"
            ) from exc
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return schema.model_validate_json(stripped)
            except ValidationError as exc:
                raise ValueError(
                    f"Agent returned invalid JSON payload for {schema.__name__}: {exc}"
                ) from exc
        raise ValueError(f"Agent returned plain-text output for {schema.__name__}: {payload}")
    if isinstance(payload, dict):
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Agent returned invalid payload for {schema.__name__}: {exc}") from exc
    raise ValueError(f"Agent returned unsupported payload type for {schema.__name__}: {type(payload)!r}")
