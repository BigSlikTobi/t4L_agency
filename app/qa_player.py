from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.history import TeamUpdateHistoryStore
from app.schemas import (
    HourlyPlaylistScriptsResponse,
    QAPlayerBatchOption,
    QAPlayerFeed,
    QAPlayerFeedItem,
    RadioStoryScript,
    TTSBatchAudioItem,
)

logger = logging.getLogger(__name__)

QA_PLAYER_HTML_PATH = Path(__file__).with_name("qa_player.html")


def load_qa_player_html() -> str:
    return QA_PLAYER_HTML_PATH.read_text(encoding="utf-8")


def build_qa_player_feed(settings: Settings, batch: str | None = None) -> QAPlayerFeed:
    store = TeamUpdateHistoryStore(settings.team_update_history_sqlite_path)
    available_batches = _load_available_batches(store)
    selected_batch = _resolve_selected_history_batch(batch, available_batches)

    if selected_batch:
        return _with_batch_metadata(
            _load_history_feed(store, batch_run_id=selected_batch),
            selected_batch=selected_batch,
            available_batches=available_batches,
        )

    if batch and not selected_batch:
        batch_feed = _load_batch_manifest_feed(settings, batch)
        if batch_feed is not None:
            return _with_batch_metadata(batch_feed, available_batches=available_batches)

    scripts_artifact_path = settings.team_update_history_sqlite_path.parent / "scripts.json"
    artifact_feed = _load_artifact_feed(scripts_artifact_path)
    if artifact_feed is not None:
        if batch and not selected_batch:
            return _with_batch_metadata(
                _apply_batch_override(artifact_feed, settings, batch),
                available_batches=available_batches,
            )
        return _with_batch_metadata(artifact_feed, available_batches=available_batches)

    history_feed = _load_history_feed(store)
    if batch and not selected_batch:
        history_feed = _apply_batch_override(history_feed, settings, batch)
        return _with_batch_metadata(history_feed, available_batches=available_batches)
    return _with_batch_metadata(
        history_feed,
        selected_batch=selected_batch or history_feed.selected_batch,
        available_batches=available_batches,
    )


def _load_artifact_feed(path: Path) -> QAPlayerFeed | None:
    if not path.exists():
        return None

    try:
        response = HourlyPlaylistScriptsResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Unable to parse QA scripts artifact", extra={"path": str(path)}, exc_info=True)
        return None

    audio_urls = {}
    if response.tts_batch is not None:
        audio_urls = {item.id: item.public_url for item in response.tts_batch.items}

    items = [
        QAPlayerFeedItem(
            id=item.id,
            title=item.title,
            playlist_rank=index + 1,
            language=response.language,
            voice_name=response.voice_name,
            audio_url=audio_urls.get(item.id),
            intro=item.script.intro,
            body=item.script.body,
            outro=item.script.outro,
            script_text=_join_script_text(item.script.intro, item.script.body, item.script.outro),
        )
        for index, item in enumerate(response.items)
    ]
    return QAPlayerFeed(
        source="artifact",
        language=response.language,
        has_audio=any(item.audio_url for item in items),
        items=items,
    )


def _load_history_feed(
    store: TeamUpdateHistoryStore,
    *,
    batch_run_id: str | None = None,
) -> QAPlayerFeed:
    entries = (
        store.get_story_scripts_for_batch(batch_run_id=batch_run_id)
        if batch_run_id
        else store.get_latest_story_scripts()
    )
    if not entries:
        return QAPlayerFeed(source="empty")

    items: list[QAPlayerFeedItem] = []
    for entry in entries:
        script = RadioStoryScript.model_validate(entry.script_json)
        items.append(
            QAPlayerFeedItem(
                id=script.slug,
                title=script.headline,
                playlist_rank=script.playlist_rank,
                team=script.team,
                language=script.language,
                voice_name=script.voice_name,
                duration_seconds=script.duration_seconds,
                intro=script.intro,
                body=script.body,
                outro=script.outro,
                script_text=_join_script_text(script.intro, script.body, script.outro),
            )
        )

    latest = entries[0]
    return QAPlayerFeed(
        source="history",
        generated_at=latest.generated_at,
        selected_batch=latest.batch_run_id,
        script_run_id=latest.script_run_id,
        playlist_id=latest.playlist_id,
        language=items[0].language,
        has_audio=False,
        items=items,
    )


def _join_script_text(intro: str, body: str, outro: str) -> str:
    return " ".join(part.strip() for part in [intro, body, outro] if part.strip())


def _apply_batch_override(feed: QAPlayerFeed, settings: Settings, batch: str) -> QAPlayerFeed:
    normalized_batch = _normalize_batch_path(batch, settings)
    if not normalized_batch or not feed.items:
        return feed

    storage_base_url = settings.resolved_supabase_storage_url().rstrip("/")
    bucket = settings.supabase_storage_bucket
    updated_items: list[QAPlayerFeedItem] = []

    for item in feed.items:
        filename = _audio_filename(item)
        audio_url = (
            f"{storage_base_url}/storage/v1/object/public/{bucket}/{normalized_batch}/{filename}"
        )
        updated_items.append(
            item.model_copy(update={"audio_url": audio_url})
        )

    return feed.model_copy(
        update={
            "has_audio": True,
            "items": updated_items,
        }
    )


def _load_batch_manifest_feed(settings: Settings, batch: str) -> QAPlayerFeed | None:
    normalized_batch = _normalize_batch_path(batch, settings)
    if not normalized_batch:
        return None

    manifest_url = (
        f"{settings.resolved_supabase_storage_url().rstrip('/')}/storage/v1/object/public/"
        f"{settings.supabase_storage_bucket}/{normalized_batch}/manifest.json"
    )

    try:
        response = httpx.get(manifest_url, timeout=10.0)
        response.raise_for_status()
        manifest_payload = json.loads(response.text)
        manifest_items = [
            TTSBatchAudioItem.model_validate(item)
            for item in manifest_payload.get("items", [])
        ]
    except Exception:
        logger.warning(
            "Unable to load QA batch manifest",
            extra={"batch": normalized_batch, "manifest_url": manifest_url},
            exc_info=True,
        )
        return None

    items = [
        QAPlayerFeedItem(
            id=item.id,
            title=_title_from_item_id(item.id),
            playlist_rank=index + 1,
            audio_url=item.public_url,
            script_text="",
        )
        for index, item in enumerate(manifest_items)
    ]
    return QAPlayerFeed(
        source="artifact",
        has_audio=bool(items),
        items=items,
    )


def _normalize_batch_path(batch: str, settings: Settings) -> str:
    candidate = str(batch).strip().strip("/")
    if not candidate:
        return ""

    path_prefix = settings.supabase_storage_path_prefix.strip("/")
    bucket_prefix = f"{settings.supabase_storage_bucket.strip('/')}/"
    if candidate.startswith(bucket_prefix):
        candidate = candidate[len(bucket_prefix) :]

    if candidate.startswith(path_prefix) and not candidate.startswith(f"{path_prefix}/"):
        return candidate

    if candidate.startswith("batches/"):
        return f"{path_prefix}/{candidate}"

    if candidate.startswith("http://") or candidate.startswith("https://"):
        path = urlsplit(candidate).path.strip("/")
        marker = f"storage/v1/object/public/{settings.supabase_storage_bucket.strip('/')}/"
        index = path.find(marker)
        if index >= 0:
            resolved = path[index + len(marker) :]
            return resolved.strip("/")

    return candidate


def _audio_filename(item: QAPlayerFeedItem) -> str:
    if item.audio_url:
        path = urlsplit(item.audio_url).path
        filename = path.rsplit("/", maxsplit=1)[-1].strip()
        if filename:
            return filename
    return f"{item.id}.mp3"


def _title_from_item_id(item_id: str) -> str:
    words = [part for part in item_id.replace("_", "-").split("-") if part]
    if not words:
        return item_id
    return " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words)


def _load_available_batches(store: TeamUpdateHistoryStore) -> list[QAPlayerBatchOption]:
    return [
        QAPlayerBatchOption(
            batch_id=entry.batch_run_id,
            generated_at=entry.generated_at,
            script_run_id=entry.script_run_id,
            playlist_id=entry.playlist_id,
            language=entry.language,
            item_count=entry.item_count,
        )
        for entry in store.list_story_script_batches()
    ]


def _resolve_selected_history_batch(
    batch: str | None,
    available_batches: list[QAPlayerBatchOption],
) -> str:
    normalized_batch = str(batch or "").strip()
    if not normalized_batch:
        return ""

    available_batch_ids = {entry.batch_id for entry in available_batches}
    if normalized_batch in available_batch_ids:
        return normalized_batch
    return ""


def _with_batch_metadata(
    feed: QAPlayerFeed,
    *,
    selected_batch: str = "",
    available_batches: list[QAPlayerBatchOption] | None = None,
) -> QAPlayerFeed:
    return feed.model_copy(
        update={
            "selected_batch": selected_batch or feed.selected_batch,
            "available_batches": available_batches or [],
        }
    )
