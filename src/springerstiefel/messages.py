"""Transcript building: embed history, system prompt and tools in one message.

Hey_ accepts a single message per request, so the OpenAI history becomes a
transcript placed before the current user message.
"""

import json

from fastapi import HTTPException

from springerstiefel import language, tools
from springerstiefel.config import settings
from springerstiefel.types import FinalParts, JsonDict, Message, ToolChoice


def prompt_language(messages: list[Message]) -> str:
    """Detect the prompt language from the last user message ("en"/"de")."""
    return language.detect_language(
        message_text(messages[last_user_index(messages)])
    )


#: Standing tone note: terse prompts are orders, not chat offers. Calm
#: wording on purpose (shouting trips the embedded-instruction guardrail).
TONE_NOTE = (
    "[Umgangston]\nDer Benutzer schreibt kurz und direkt – das sind "
    "verbindliche Anweisungen, keine Gesprächsangebote. Keine Vorstellung, "
    "keine Meta-Kommentare über dich selbst, direkt zur Sache."
)


def message_text(message: Message) -> str:
    """Extract readable text from an OpenAI message (str or parts)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def last_user_index(messages: list[Message]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    raise HTTPException(400, "No user message supplied")


def last_user_text(messages: list[Message]) -> str:
    """Take the last user message (Hey_ knows only a single message)."""
    return message_text(messages[last_user_index(messages)])


def _message_parts(
    messages: list[Message],
    tool_defs: list[JsonDict] | None,
    tool_choice: ToolChoice,
) -> tuple[str | None, list[str], str, str | None]:
    idx = last_user_index(messages)

    system_parts = [
        message_text(m) for m in messages if m.get("role") == "system"
    ]
    # Everything except the system prompt becomes transcript. If turns came
    # after the last user message (tool round-trip), nothing is repeated –
    # instead we ask to continue.
    tail = [m for m in messages[idx + 1:] if m.get("role") != "system"]
    if tail:
        transcript_msgs = [
            m for m in messages if m.get("role") != "system"
        ][-settings.max_history:]
        current = (
            "Setze die Bearbeitung fort. "
            "Das neueste Werkzeugergebnis steht am Ende des Verlaufs."
        )
    else:
        transcript_msgs = [
            m for m in messages[:idx] if m.get("role") != "system"
        ][-settings.max_history:]
        current = message_text(messages[idx])

    system_section = (
        "[Systemanweisung]\n" + "\n".join(system_parts) if system_parts else None
    )

    lines: list[str] = []
    for m in transcript_msgs:
        role = m.get("role")
        if role == "user":
            lines.append(f"Benutzer: {message_text(m)}")
        elif role == "assistant":
            text = message_text(m)
            if text:
                lines.append(f"Assistent: {text}")
            for call in m.get("tool_calls") or []:
                func = call.get("function", {})
                lines.append(
                    f"Assistent (Werkzeugaufruf {func.get('name')}: "
                    f"{func.get('arguments')})"
                )
        elif role == "tool":
            name = m.get("name") or m.get("tool_call_id") or "?"
            result = message_text(m)
            # Tool outputs (files, shell) are the biggest prompt bloat –
            # truncate, keep the gist.
            if len(result) > settings.max_tool_chars:
                result = (
                    result[:settings.max_tool_chars]
                    + "\n[… Ausgabe gekürzt …]"
                )
            lines.append(f"Werkzeugergebnis ({name}): {result}")

    tool_section = None
    if tool_defs and tool_choice != "none":
        tool_section = (
            "[Verfügbare Aktionen – als JSON-Schema]\n"
            + json.dumps(tool_defs, ensure_ascii=False, indent=2)
            + "\n\n[Regel für Aktionen]\n"
            + tools.TOOL_INSTRUCTIONS
        )
        user_texts = " ".join(
            message_text(m) for m in messages if m.get("role") == "user"
        )
        if (
            tool_choice == "required"
            or (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
            )
            or tools.is_action_request(user_texts)
        ):
            tool_section += (
                "\nWICHTIG: Diese Aufgabe ist per Aktion zu lösen – "
                "antworte mit Aktions-Blöcken statt Fließtext."
            )

    return system_section, lines, current, tool_section


def build_hey_message(
    messages: list[Message],
    tool_defs: list[JsonDict] | None = None,
    tool_choice: ToolChoice = None,
) -> str:
    """Embed history + system prompt + tool definitions in one Hey_ message.

    tool_choice="none" hides the tool section; "required", a specific
    function, or an explicit action request marks the call as mandatory.
    """
    system_section, lines, current, tool_section = _message_parts(
        messages, tool_defs, tool_choice
    )
    sections = [
        s
        for s in (
            system_section,
            "[Bisheriger Verlauf]\n" + "\n".join(lines) if lines else None,
            TONE_NOTE,
            f"Aktuelle Anweisung:\n{current}",
            tool_section,
        )
        if s
    ]
    text = "\n\n".join(sections)
    if prompt_language(messages) == "en":
        text += "\n\n" + language.LANGUAGE_DIRECTIVE
    return text


def split_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    """Split transcript lines into chunks (whole lines, max. max_chars)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


def render_intermediate(
    system_section: str | None,
    lines: list[str],
    part: int,
    total: int,
    lang: str = "de",
) -> str:
    head = f"[Gesprächsverlauf Teil {part}/{total} – lies ihn, antworte noch nicht]\n"
    sections = [
        s
        for s in (system_section, head + "\n".join(lines))
        if s
    ]
    if lang == "en":
        return "\n\n".join(sections) + (
            "\n\nSummarize in 2-3 sentences what matters for what follows. "
            "Reply with only the summary."
        )
    return "\n\n".join(sections) + (
        "\n\nFasse in 2–3 Sätzen zusammen, was für die weitere Bearbeitung "
        "wichtig ist. Antworte nur mit der Zusammenfassung."
    )


def render_final(parts: FinalParts, summaries: list[str]) -> str:
    sections = [
        s
        for s in (
            parts["system"],
            (
                "[Zusammenfassung früherer Verlaufsteile]\n"
                + "\n".join(f"- {s}" for s in summaries)
            )
            if summaries
            else None,
            (
                "[Bisheriger Verlauf (letzter Teil)]\n"
                + "\n".join(parts["lines"])
            )
            if parts["lines"]
            else None,
            TONE_NOTE,
            f"Aktuelle Anweisung:\n{parts['current']}",
            parts["tools"],
        )
        if s
    ]
    text = "\n\n".join(sections)
    if parts.get("lang") == "en":
        text += "\n\n" + language.LANGUAGE_DIRECTIVE
    return text


def build_hey_jobs(
    messages: list[Message],
    tool_defs: list[JsonDict] | None,
    tool_choice: ToolChoice,
) -> tuple[list[str], FinalParts | None]:
    """Build the FIFO job list for oversized prompts.

    Returns (intermediate jobs, final parts|None). Small histories yield
    ([], None) – then the normal single request runs. Otherwise: work the
    intermediate jobs (summary chain), render the final with summaries. At
    most the newest settings.max_chunks history chunks are considered.
    """
    system_section, lines, current, tool_section = _message_parts(
        messages, tool_defs, tool_choice
    )
    chunks = split_lines(lines, settings.max_chunk_chars)
    if len(chunks) > settings.max_chunks:
        chunks = chunks[-settings.max_chunks:]
    if len(chunks) <= 1:
        return [], None
    total = len(chunks)
    lang = prompt_language(messages)
    intermediates = [
        render_intermediate(system_section, chunk, i + 1, total, lang)
        for i, chunk in enumerate(chunks[:-1])
    ]
    final_parts: FinalParts = {
        "system": system_section,
        "lines": chunks[-1],
        "current": current,
        "tools": tool_section,
        "lang": lang,
    }
    return intermediates, final_parts
