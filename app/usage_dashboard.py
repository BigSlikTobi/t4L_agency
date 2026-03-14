from __future__ import annotations

from pathlib import Path

USAGE_DASHBOARD_HTML_PATH = Path(__file__).with_name("usage_dashboard.html")


def load_usage_dashboard_html() -> str:
    return USAGE_DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
