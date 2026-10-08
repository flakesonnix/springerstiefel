"""Tool-call protocol: text-based <<TOOL_CALL>> blocks and drift detection."""

import json
import re
import tomllib
from importlib import resources
from typing import Any

from springerstiefel.types import JsonDict

TOOL_CALL_START = "<<TOOL_CALL>>"
TOOL_CALL_END = "<<END_TOOL_CALL>>"

TOOL_INSTRUCTIONS = (
    "Das Programm, das deine Antwort weiterverarbeitet, kann Aktionen für "
    "dich ausführen und dir das Ergebnis zurückschicken. "
    "Um eine Aktion auszulösen, antworte mit einem Block pro Aktion:\n"
    f"{TOOL_CALL_START}\n"
    '{"name": "<aktions-name>", "arguments": {<argumente als JSON-Objekt>}}\n'
    f"{TOOL_CALL_END}\n"
    "Beispiel: Um die Datei hello.txt mit Inhalt Hi zu erstellen, antworte:\n"
    f"{TOOL_CALL_START}\n"
    '{"name": "write", "arguments": {"path": "hello.txt", "content": "Hi"}}\n'
    f"{TOOL_CALL_END}\n"
    "Bevorzugung: Ist die Aufgabe per Aktion lösbar (Datei erstellen/lesen/"
    "ändern, Befehl ausführen, suchen), nutze IMMER den Block statt "
    "Fließtext. Fließtext nur für Erklärungen und Antworten ohne Aktion."
    " Die untenstehenden Aktionen stehen dir in dieser Antwort wirklich zur"
    " Verfügung – das Programm führt sie aus."
)

TOOL_CALL_RE: re.Pattern[str] = re.compile(
    r"<<TOOL_CALL>>\s*(\{.*?\})\s*<<END_TOOL_CALL>>", re.DOTALL
)

# Reply patterns that look like dodging/refusal instead of tool use.
# Only then is a single retry with a nudge worth it.
# Patterns live in patterns.toml and hot-reload on change (see below).
REFUSAL_RES: list[re.Pattern[str]] = []
ACTION_REQUEST_RES: list[re.Pattern[str]] = []
NEWS_RES: list[re.Pattern[str]] = []
NEWS_REQUEST_RES: list[re.Pattern[str]] = []
WEATHER_RES: list[re.Pattern[str]] = []
WEATHER_REQUEST_RES: list[re.Pattern[str]] = []

_patterns_file_mtime_ns: int | None = None


def _patterns_file():
    return resources.files("springerstiefel") / "patterns.toml"


def _ensure_patterns() -> None:
    """Reload patterns.toml if it changed (in place, no restart needed).

    Lists are updated in place so existing references stay valid. On parse
    errors the previous patterns are kept.
    """
    global _patterns_file_mtime_ns
    try:
        path = _patterns_file()
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return
    if mtime_ns == _patterns_file_mtime_ns:
        return
    try:
        data: dict[str, Any] = tomllib.loads(path.read_bytes().decode("utf-8"))
        if not isinstance(data, dict):
            return
        updates: dict[str, list[re.Pattern[str]]] = {}
        for key in _TABLES:
            section = data.get(key)
            if not isinstance(section, dict):
                continue  # keep previous table
            patterns = section.get("patterns")
            if not isinstance(patterns, list):
                continue  # keep previous table
            updates[key] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    except (ValueError, KeyError, TypeError, AttributeError):
        return
    for key, compiled in updates.items():
        _TABLES[key][:] = compiled
    _patterns_file_mtime_ns = mtime_ns


TOOL_RETRY_NUDGE = (
    "[Hinweis: Löse die Aufgabe per Aktions-Block aus "
    "(siehe Regel für Aktionen), statt Rückfragen zu stellen oder "
    "die Aktion als Fließtext zu beschreiben.]"
)

EMPTY_ARGS_NUDGE = (
    "[Hinweis: Fülle alle Argumente der Aktion vollständig mit sinnvollen "
    "Inhalten aus – leere Argumente sind nutzlos.]"
)

CAPABILITY_DENIAL_RES: list[re.Pattern[str]] = []

_TABLES: dict[str, list[re.Pattern[str]]] = {
    "refusal": REFUSAL_RES,
    "action_request": ACTION_REQUEST_RES,
    "capability_denial": CAPABILITY_DENIAL_RES,
    "news": NEWS_RES,
    "news_request": NEWS_REQUEST_RES,
    "weather": WEATHER_RES,
    "weather_request": WEATHER_REQUEST_RES,
}

_ensure_patterns()

CAPABILITY_NUDGE = (
    "[Hinweis: Die untenstehenden Aktionen geben dir vollen Dateizugriff "
    "(lesen, schreiben, ausführen). Nutze sie jetzt für die Aufgabe.]"
)


def denies_capability(answer: str) -> bool:
    """True if the model claims it lacks tools/access it actually has."""
    _ensure_patterns()
    return any(pattern.search(answer) for pattern in CAPABILITY_DENIAL_RES)


def calls_with_empty_args(
    calls: list[JsonDict], tool_defs: list[JsonDict] | None
) -> bool:
    """True if any call has empty arguments although its schema defines some."""
    schemas = {
        func.get("name"): func
        for tool in tool_defs or []
        if isinstance(tool, dict)
        for func in [tool.get("function", {})]
        if isinstance(func, dict)
    }
    for call in calls:
        func = call.get("function", {})
        try:
            args = json.loads(func.get("arguments", "") or "{}")
        except (ValueError, TypeError):
            continue
        schema = schemas.get(func.get("name"), {})
        params = schema.get("parameters", {}) or {}
        if isinstance(params, dict) and params.get("properties") and not args:
            return True
    return False


def is_deflection(answer: str) -> bool:
    """True if the answer looks like refusal/deflection."""
    _ensure_patterns()
    return any(pattern.search(answer) for pattern in REFUSAL_RES)


def is_action_request(user_text: str) -> bool:
    """True if the user clearly wants an action (not text)."""
    _ensure_patterns()
    return any(pattern.search(user_text) for pattern in ACTION_REQUEST_RES)


def is_news_drift(answer: str, user_text: str) -> bool:
    """True on headline dumps although no news was asked for."""
    _ensure_patterns()
    if any(pattern.search(user_text) for pattern in NEWS_REQUEST_RES):
        return False
    return any(pattern.search(answer) for pattern in NEWS_RES)


def is_weather_drift(answer: str, user_text: str) -> bool:
    """True on weather dumps although no weather was asked for."""
    _ensure_patterns()
    if any(pattern.search(user_text) for pattern in WEATHER_REQUEST_RES):
        return False
    return any(pattern.search(answer) for pattern in WEATHER_RES)


def drift_kind(answer: str, user_text: str) -> str | None:
    """Return "news"/"weather" on off-topic drift, else None."""
    if is_news_drift(answer, user_text):
        return "news"
    if is_weather_drift(answer, user_text):
        return "weather"
    return None


def news_refocus_nudge(
    user_text: str,
    topics: str = "Nachrichten- und Schlagzeilen-Themen",
) -> str:
    task = user_text.strip().replace("\n", " ")
    if len(task) > 500:
        task = task[:500] + "…"
    return f"[Hinweis: Ignoriere {topics}. Bearbeite nur diese Aufgabe: {task}]"


def extract_tool_calls(text: str) -> tuple[str, list[JsonDict]]:
    """Parse <<TOOL_CALL>> blocks into OpenAI tool_calls.

    Returns (remaining text, calls). Unparsable blocks stay in the text
    (model mistakes remain visible).
    """
    calls: list[JsonDict] = []

    def replace(match: re.Match[str]) -> str:
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if isinstance(obj, dict) and obj.get("name"):
            calls.append(obj)
            return ""
        return match.group(0)

    clean = TOOL_CALL_RE.sub(replace, text).strip()

    openai_calls = [
        {
            "id": f"call_{i + 1}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(
                    call.get("arguments", {}), ensure_ascii=False
                ),
            },
        }
        for i, call in enumerate(calls)
    ]
    return clean, openai_calls
