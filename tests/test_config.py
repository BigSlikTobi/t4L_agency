from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def build_settings_kwargs() -> dict[str, str]:
    return {
        "_env_file": None,
        "openai_api_key": "sk-test",
        "supabase_news_feed_url": "https://example.com/feed",
        "supabase_article_lookup_url": "https://example.com/article-lookup",
        "supabase_function_auth_token": "supabase-token",
    }


def test_settings_require_distinct_supabase_function_urls() -> None:
    settings = Settings(**build_settings_kwargs())

    assert str(settings.supabase_news_feed_url) == "https://example.com/feed"
    assert str(settings.supabase_article_lookup_url) == "https://example.com/article-lookup"
    assert settings.supabase_story_group_updates_url is None
    assert str(settings.gemini_tts_batch_url) == (
        "https://us-central1-tackle4loss-888b5.cloudfunctions.net/gemini-tts-batch"
    )
    assert settings.gemini_tts_batch_model_name == "gemini-2.5-pro-preview-tts"
    assert settings.openai_model_team_update_agent == "gpt-5-mini-2025-08-07"
    assert settings.openai_model_team_update_batch_agent == "gpt-5-mini-2025-08-07"
    assert settings.openai_model_radio_script_writer_agent == "gpt-5.3-chat-latest"
    assert settings.gemini_tts_batch_poll_interval_seconds == 10.0
    assert settings.gemini_tts_batch_timeout_seconds == 900.0
    assert settings.resolved_supabase_storage_url() == "https://example.com"
    assert settings.supabase_storage_bucket == "audio"
    assert settings.supabase_storage_path_prefix == "gemini-tts-batch"


def test_settings_reject_identical_supabase_function_urls() -> None:
    with pytest.raises(ValidationError):
        Settings(**{**build_settings_kwargs(), "supabase_article_lookup_url": "https://example.com/feed"})


def test_agent_models_fall_back_to_shared_role_models() -> None:
    settings = Settings(
        **{
            **build_settings_kwargs(),
            "openai_model_chief_editor": "chief-model",
            "openai_model_team_analyst": "analyst-model",
            "openai_model_story_researcher": "research-model",
            "openai_model_article_data_agent": "",
            "openai_model_team_news_agent": "",
            "openai_model_rundown_orchestrator_agent": "",
            "openai_model_team_update_gate": "",
            "openai_model_team_update_agent": "",
            "openai_model_team_update_batch_agent": "",
            "openai_model_hourly_playlist_orchestrator_agent": "",
            "openai_model_radio_script_writer_agent": "",
            "openai_model_hourly_script_batch_agent": "",
        }
    )

    assert settings.agent_model("article_data_agent") == "research-model"
    assert settings.agent_model("team_news_agent") == "analyst-model"
    assert settings.agent_model("rundown_orchestrator_agent") == "chief-model"
    assert settings.agent_model("team_update_gate") == "analyst-model"
    assert settings.agent_model("team_update_agent") == "analyst-model"
    assert settings.agent_model("team_update_batch_agent") == "chief-model"
    assert settings.agent_model("hourly_playlist_orchestrator_agent") == "chief-model"
    assert settings.agent_model("radio_script_writer_agent") == "analyst-model"
    assert settings.agent_model("hourly_script_batch_agent") == "chief-model"


def test_agent_models_prefer_specific_agent_overrides() -> None:
    settings = Settings(
        **{
            **build_settings_kwargs(),
            "openai_model_chief_editor": "chief-model",
            "openai_model_team_analyst": "analyst-model",
            "openai_model_story_researcher": "research-model",
            "openai_model_article_data_agent": "",
            "openai_model_team_news_agent": "",
            "openai_model_rundown_orchestrator_agent": "",
            "openai_model_team_update_agent": "team-update-model",
            "openai_model_team_update_batch_agent": "",
            "openai_model_hourly_playlist_orchestrator_agent": "",
            "openai_model_radio_script_writer_agent": "script-model",
            "openai_model_hourly_script_batch_agent": "",
        }
    )

    assert settings.agent_model("team_update_agent") == "team-update-model"
    assert settings.agent_model("radio_script_writer_agent") == "script-model"
    assert settings.model_summary()["team_news_agent"] == "analyst-model"


def test_settings_prefer_explicit_storage_overrides() -> None:
    settings = Settings(
        **{
            **build_settings_kwargs(),
            "supabase_storage_url": "https://storage.example.com",
            "supabase_storage_key": "storage-key",
            "supabase_storage_bucket": "tts-audio",
            "supabase_storage_path_prefix": "radio-hourly",
        }
    )

    assert settings.resolved_supabase_storage_url() == "https://storage.example.com"
    assert settings.resolved_supabase_storage_key() == "storage-key"
    assert settings.supabase_storage_bucket == "tts-audio"
    assert settings.supabase_storage_path_prefix == "radio-hourly"


def test_settings_accept_service_role_key_for_storage() -> None:
    settings = Settings(
        **{
            **build_settings_kwargs(),
            "supabase_service_role_key": "service-role-key",
        }
    )

    assert settings.resolved_supabase_storage_key() == "service-role-key"


def test_settings_require_explicit_storage_or_service_role_key() -> None:
    settings = Settings(**build_settings_kwargs())

    with pytest.raises(ValueError, match="SUPABASE_STORAGE_KEY or SUPABASE_SERVICE_ROLE_KEY"):
        settings.resolved_supabase_storage_key()
