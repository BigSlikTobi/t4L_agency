# This file manages the loading and retrieval of prompts for the agents in the newsroom workflow.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


PROMPTS_PATH = Path(__file__).with_name("prompts.yml")
REQUIRED_PROMPTS = {
    "article_data_agent",
    "team_news_agent",
    "team_update_agent",
    "team_update_batch_agent",
    "hourly_playlist_orchestrator_agent",
    "radio_script_writer_agent",
    "radio_script_writer_agent_de_de",
    "hourly_script_batch_agent",
    "hourly_script_batch_agent_de_de",
    "rundown_orchestrator_agent",
}


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, str]:
    raw = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt file must be a mapping: {PROMPTS_PATH}")

    prompts: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Prompt file contains non-string entry for key {key!r}")
        prompts[key] = value.strip()

    missing = sorted(REQUIRED_PROMPTS - prompts.keys())
    if missing:
        raise ValueError(f"Prompt file is missing required prompts: {', '.join(missing)}")
    return prompts


def get_prompt(name: str) -> str:
    prompts = load_prompts()
    try:
        return prompts[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt name: {name}") from exc
