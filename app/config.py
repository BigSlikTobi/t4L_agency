# This file contains the Settings class for managing application configuration.
# It uses Pydantic for validation and supports loading from environment variables and a .env.

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    gemini_api_key: SecretStr | None = None
    supabase_news_feed_url: AnyHttpUrl
    supabase_article_lookup_url: AnyHttpUrl
    supabase_story_group_updates_url: AnyHttpUrl | None = None
    gemini_tts_batch_url: AnyHttpUrl = (
        "https://us-central1-tackle4loss-888b5.cloudfunctions.net/gemini-tts-batch"
    )
    supabase_function_auth_token: SecretStr
    supabase_storage_url: AnyHttpUrl | None = None
    supabase_storage_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_storage_bucket: str = "audio"
    supabase_storage_path_prefix: str = "gemini-tts-batch"
    team_update_history_sqlite_path: Path = Path("./var/team_update_history.sqlite3")
    gemini_tts_batch_model_name: str = "gemini-2.5-pro-preview-tts"
    gemini_tts_batch_poll_interval_seconds: float = 10.0
    gemini_tts_batch_timeout_seconds: float = 900.0

    openai_model_chief_editor: str = "gpt-5-mini-2025-08-07"
    openai_model_team_analyst: str = "gpt-5-mini-2025-08-07"
    openai_model_story_researcher: str = "gpt-5-mini-2025-08-07"
    openai_model_article_data_agent: str = "gpt-5-mini-2025-08-07"
    openai_model_team_news_agent: str = "gpt-5-2025-08-07"
    openai_model_rundown_orchestrator_agent: str = "gpt-5.2-2025-12-11"
    openai_model_team_update_agent: str = "gpt-5-mini-2025-08-07"
    openai_model_team_update_batch_agent: str = "gpt-5-mini-2025-08-07"
    openai_model_hourly_playlist_orchestrator_agent: str = "gpt-5.2-2025-12-11"
    openai_model_radio_script_writer_agent: str = "gpt-5.1-2025-11-13"
    openai_model_hourly_narrative_planner_agent: str = "gpt-5.4-2026-03-05"
    openai_model_hourly_script_batch_agent: str = "gpt-5-mini-2025-08-07"
    team_update_batch_chunk_size: int = 8
    script_batch_chunk_size: int = 5

    openai_temperature: float | None = None
    openai_max_tokens: int | None = None

    news_timeout_seconds: float = 15.0
    telemetry_enabled: bool = False
    telemetry_project_name: str = "t4l-radio-agency"
    telemetry_environment: str = "local"
    telemetry_instance_id: str = ""
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None
    telemetry_capture_content: bool = False

    @model_validator(mode="after")
    def validate_supabase_urls(self) -> "Settings":
        if str(self.supabase_article_lookup_url) == str(self.supabase_news_feed_url):
            raise ValueError("SUPABASE_ARTICLE_LOOKUP_URL must differ from SUPABASE_NEWS_FEED_URL")
        return self

    def readiness_summary(self) -> dict[str, str]:
        return {
            "supabase_news_feed": str(self.supabase_news_feed_url),
            "supabase_article_lookup": str(self.supabase_article_lookup_url),
            "supabase_story_group_updates": (
                str(self.supabase_story_group_updates_url)
                if self.supabase_story_group_updates_url is not None
                else "unconfigured"
            ),
            "gemini_tts_batch": str(self.gemini_tts_batch_url),
            "supabase_storage_url": self.resolved_supabase_storage_url(),
            "supabase_storage_bucket": self.supabase_storage_bucket,
            "supabase_storage_path_prefix": self.supabase_storage_path_prefix,
            "team_update_history_sqlite_path": str(self.team_update_history_sqlite_path),
            "telemetry_enabled": str(self.telemetry_enabled).lower(),
            "telemetry_project_name": self.telemetry_project_name,
            "telemetry_environment": self.telemetry_environment,
            "telemetry_instance_id": self.resolved_telemetry_instance_id(),
            "otel_exporter_otlp_endpoint": self.otel_exporter_otlp_endpoint or "unconfigured",
        }

    def model_summary(self) -> dict[str, str]:
        return self.agent_models()

    def agent_models(self) -> dict[str, str]:
        return {
            "article_data_agent": self.openai_model_article_data_agent or self.openai_model_story_researcher,
            "team_news_agent": self.openai_model_team_news_agent or self.openai_model_team_analyst,
            "rundown_orchestrator_agent": (
                self.openai_model_rundown_orchestrator_agent or self.openai_model_chief_editor
            ),
            "team_update_agent": self.openai_model_team_update_agent or self.openai_model_team_analyst,
            "team_update_batch_agent": (
                self.openai_model_team_update_batch_agent or self.openai_model_chief_editor
            ),
            "hourly_playlist_orchestrator_agent": (
                self.openai_model_hourly_playlist_orchestrator_agent or self.openai_model_chief_editor
            ),
            "radio_script_writer_agent": (
                self.openai_model_radio_script_writer_agent or self.openai_model_team_analyst
            ),
            "hourly_narrative_planner_agent": (
                self.openai_model_hourly_narrative_planner_agent or self.openai_model_chief_editor
            ),
            "hourly_script_batch_agent": (
                self.openai_model_hourly_script_batch_agent or self.openai_model_chief_editor
            ),
        }

    def agent_model(self, agent_name: str) -> str:
        try:
            return self.agent_models()[agent_name]
        except KeyError as exc:
            raise ValueError(f"Unknown agent name: {agent_name}") from exc

    def resolved_supabase_storage_url(self) -> str:
        if self.supabase_storage_url is not None:
            return str(self.supabase_storage_url).rstrip("/")

        parsed = urlsplit(str(self.supabase_news_feed_url))
        return f"{parsed.scheme}://{parsed.netloc}"

    def resolved_supabase_storage_key(self) -> str:
        if self.supabase_storage_key is not None:
            return self.supabase_storage_key.get_secret_value()
        if self.supabase_service_role_key is not None:
            return self.supabase_service_role_key.get_secret_value()
        raise ValueError(
            "TTS batch processing requires SUPABASE_STORAGE_KEY or SUPABASE_SERVICE_ROLE_KEY "
            "for Supabase Storage uploads."
        )

    def resolved_telemetry_instance_id(self) -> str:
        if self.telemetry_instance_id.strip():
            return self.telemetry_instance_id.strip()
        return os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "local-instance"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
