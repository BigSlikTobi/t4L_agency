# Centralized logging configuration for the T4L Radio Agency.
# Import and call configure_logging() at application startup (main.py or cli.py) to enable
# structured JSON logging across all modules.

from __future__ import annotations

import logging
import json
import sys
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Compact JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra fields passed via `extra={...}`
        for key in ("run_id", "team", "teams", "lookback_hours", "count",
                     "team_count", "story_count", "article_count",
                     "candidate_count", "segment_count", "report_count",
                     "url", "group_id", "eligible_report_count"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured JSON logging for the entire application."""
    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JSONFormatter)
           for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
