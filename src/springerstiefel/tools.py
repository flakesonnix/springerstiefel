"""Tool-call protocol: text-based <<TOOL_CALL>> blocks and drift detection."""

import json
import re

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
)

TOOL_CALL_RE: re.Pattern[str] = re.compile(
    r"<<TOOL_CALL>>\s*(\{.*?\})\s*<<END_TOOL_CALL>>", re.DOTALL
)

# Reply patterns that look like dodging/refusal instead of tool use.
# Only then is a single retry with a nudge worth it.
REFUSAL_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"wobei soll ich",
        r"wobei kann ich",
        r"womit kann ich",
        r"wie kann ich.*helfen",
        r"lassen sie mich wissen",
        r"kann .* nicht",
        r"kann .* keine",
        r"in diesem schritt",
        r"meinen sie",
        r"leider kann",
        r"brauche .* (mehr|weitere|genauere)",
        r"nennen sie (bitte )?(pfad|datei|inhalt)",
        r"sagen sie (mir )?(bitte )?(was|welche)",
        r"soll ich",
        r"möchten sie, dass ich",
        r"wenn sie möchten",
        r"sagen sie bescheid",
        r"geben sie bescheid",
        r"darf ich",
        r"welche datei",
        r"how can i help",
        r"should i",
        r"shall i",
        r"do you want me to",
        r"would you like me to",
        r"let me know if",
        r"i can (add|create|write|make|generate|implement|provide)",
        r"i can('|no)t",
        r"i('| a)m not able to",
    )
]

# Requests that clearly want an action (not text) – the proxy escalates
# those to a mandatory tone.
ACTION_REQUEST_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"erstelle .* datei",
        r"erstellen .* datei",
        r"schreibe .* (datei|programm|code)",
        r"lege .* datei an",
        r"speichere .* (in|als|unter)",
        r"führe .* aus",
        r"kompiliere",
        r"create .* file",
        r"write .* (file|program|code)",
        r"save .* (to|as|in)",
        r"\brun\b.*(test|build|command)",
    )
]

# Markers for drifting into news mode (BILD grounding with citations).
NEWS_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\[bild_\d",
        r"schlagzeilen",
        r"bildplus",
        r"bild berichtet",
        r"themenkontext",
    )
]

# Signals that the user actually wants news – then no retry.
NEWS_REQUEST_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"nachricht",
        r"schlagzeile",
        r"\bnews\b",
        r"aktuell",
        r"\bbild\b",
        r"zeitung",
        r"was gibt es neues",
    )
]

# Markers for drifting into weather mode without BILD citations.
WEATHER_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwetter\b",
        r"wetterbericht",
        r"wettervorhersage",
        r"regenwahrscheinlichkeit",
        r"\bwind aus\b",
        r"\bböen\b",
        r"\bbedeckt\b",
        r"niederschlag",
        r"°c",
    )
]

# Signals that the user actually wants weather – then no retry.
WEATHER_REQUEST_RES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwetter",
        r"wettervorhersage",
        r"wetterbericht",
        r"\bregen\b",
        r"temperatur",
        r"\bklima\b",
        r"\bweather\b",
        r"forecast",
    )
]

TOOL_RETRY_NUDGE = (
    "[Hinweis: Löse die Aufgabe per Aktions-Block aus "
    "(siehe Regel für Aktionen), statt Rückfragen zu stellen oder "
    "die Aktion als Fließtext zu beschreiben.]"
)


def is_deflection(answer: str) -> bool:
    """True if the answer looks like refusal/deflection."""
    return any(pattern.search(answer) for pattern in REFUSAL_RES)


def is_action_request(user_text: str) -> bool:
    """True if the user clearly wants an action (not text)."""
    return any(pattern.search(user_text) for pattern in ACTION_REQUEST_RES)


def is_news_drift(answer: str, user_text: str) -> bool:
    """True on headline dumps although no news was asked for."""
    if any(pattern.search(user_text) for pattern in NEWS_REQUEST_RES):
        return False
    return any(pattern.search(answer) for pattern in NEWS_RES)


def is_weather_drift(answer: str, user_text: str) -> bool:
    """True on weather dumps although no weather was asked for."""
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
