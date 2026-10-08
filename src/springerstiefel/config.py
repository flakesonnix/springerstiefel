"""Configuration: environment variables with sane defaults."""

import os
from dataclasses import dataclass

HEY_BASE = "https://hey.bild.de"
DEFAULT_EXPERIENCE_ID = "a5d82531-015a-46d0-9547-47602fe9b03e"

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:155.0) "
        "Gecko/20100101 Firefox/155.0"
    ),
    "Origin": HEY_BASE,
    "Referer": HEY_BASE + "/",
}


@dataclass
class Settings:
    """All tunables, overridable via environment (see README)."""

    base_url: str = HEY_BASE
    experience_id: str = DEFAULT_EXPERIENCE_ID
    timeout: float = 120.0
    max_history: int = 30
    max_tool_chars: int = 4000
    max_chunk_chars: int = 6000
    max_chunks: int = 5
    tool_retry: bool = True


def load_settings() -> Settings:
    """Build settings from the environment."""
    return Settings(
        experience_id=os.environ.get("HEY_EXPERIENCE_ID", DEFAULT_EXPERIENCE_ID),
        timeout=float(os.environ.get("HEY_TIMEOUT", "120")),
        max_history=int(os.environ.get("HEY_MAX_HISTORY", "30")),
        max_tool_chars=int(os.environ.get("HEY_MAX_TOOL_CHARS", "4000")),
        max_chunk_chars=int(os.environ.get("HEY_MAX_CHUNK_CHARS", "6000")),
        max_chunks=int(os.environ.get("HEY_MAX_CHUNKS", "5")),
        tool_retry=os.environ.get("HEY_TOOL_RETRY", "1") == "1",
    )


#: Process-wide settings; tests may patch its fields via monkeypatch.
settings = load_settings()
