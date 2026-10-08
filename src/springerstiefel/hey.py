"""Hey_ backend client: conversations, SSE chat, retries, summaries."""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx

from springerstiefel import tools
from springerstiefel.config import (
    BROWSER_HEADERS,
    DEFAULT_EXPERIENCE_ID,
    settings,
)
from springerstiefel.types import HeyEvent, JsonDict, Message

#: Citation markers like [bild_0_1], [web_2], [bild_0_0:image_0].
CITATION_RE = re.compile(r"\[(bild|web|image)(?:_\d+)+(?:\:[^\]]*)?\]")

#: Resolved sources by citation index: {"bild_0_1": {"title": ..., "url": ...}}.
Source = dict[str, str]
Sources = dict[str, Source]


def extract_message_text(message: Message) -> str:
    """Get the answer text from a Hey_ message object.

    Hey_ returns plain `content`, or a JSON envelope
    `{"answer": ..., "suggestions": ...}` (string or dict, in `content`
    or `parsed`). Only `answer` counts here.
    """
    content = message.get("content")
    if isinstance(content, dict):
        answer = content.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") and '"answer"' in stripped:
            try:
                envelope = json.loads(stripped)
            except json.JSONDecodeError:
                envelope = None
            if isinstance(envelope, dict):
                answer = envelope.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return answer
        if stripped:
            return content
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer
    return ""


def extract_sources(delta: JsonDict) -> Sources:
    """Pull {index: {title, url}} out of a tool delta's sources list."""
    found: Sources = {}
    sources = delta.get("sources") or []
    if not isinstance(sources, list):
        return found
    for source in sources:
        if not isinstance(source, dict):
            continue
        index = source.get("index")
        url = source.get("url")
        if not index or not url:
            continue
        found[str(index)] = {
            "title": str(source.get("title") or source.get("name") or url),
            "url": str(url),
        }
    return found


def resolve_sources(text: str, sources: Sources) -> list[tuple[str, str, str]]:
    """Match citation markers in text order against known sources.

    Returns [(marker, title, url)]; image markers (bild_0_0:image_0) fall
    back to their parent article. Unknown markers are skipped.
    """
    resolved: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for match in CITATION_RE.finditer(text):
        marker = match.group(0)[1:-1]
        if marker in seen:
            continue
        entry = sources.get(marker)
        if entry is None and ":" in marker:
            entry = sources.get(marker.split(":")[0])
        if entry is None:
            continue
        seen.add(marker)
        resolved.append((marker, entry["title"], entry["url"]))
    return resolved


def format_sources(resolved: list[tuple[str, str, str]]) -> str:
    lines = [f"[{marker}] [{title}]({url})" for marker, title, url in resolved]
    return "\n\nQuellen:\n" + "\n".join(lines)


def append_sources(text: str, sources: Sources) -> str:
    """Append a Quellen section with resolved citation links, if any."""
    resolved = resolve_sources(text, sources)
    if not resolved:
        return text
    return text + format_sources(resolved)


def pick_experience(items: list[JsonDict]) -> str:
    """Pick a usable chat experience ID from /api/home items.

    Prefers the HEY_EXPERIENCE_SLUG match, then any enabled chat
    experience, then any experience with an ID. Falls back to default.
    """
    slug = settings.experience_slug.strip().lower()
    with_ids = [i for i in items if isinstance(i, dict) and i.get("experienceId")]
    if slug:
        for item in with_ids:
            if slug in str(item.get("slug", "")).lower():
                return str(item["experienceId"])
    for item in with_ids:
        if item.get("interactionMode", "chat") == "chat" and not item.get(
            "isTextInputDisabled", False
        ):
            return str(item["experienceId"])
    if with_ids:
        return str(with_ids[0]["experienceId"])
    return DEFAULT_EXPERIENCE_ID


# One session = one HTTP client + one Hey_ conversation.
# A proxy turn shares exactly one session (saves one conversation POST
# ~170ms per extra call and keeps the turn server-side in one conversation).
HeySession = tuple[httpx.AsyncClient, str]

_shared_transport: httpx.AsyncHTTPTransport | None = None


def shared_transport() -> httpx.AsyncHTTPTransport:
    """Process-wide pooled transport (TCP/TLS reuse across turns).

    Per-turn clients keep their own cookie jar for isolation; only the
    connection pool is shared.
    """
    global _shared_transport
    if _shared_transport is None:
        _shared_transport = _PooledTransport()
    return _shared_transport


class _PooledTransport(httpx.AsyncHTTPTransport):
    """Transport whose pool outlives single clients.

    Per-turn clients are closed after their turn; closing must not tear
    down the shared pool, so aclose is a no-op (process lifetime).
    """


@dataclass
class TurnState:
    """Last turn of this client: messages, conversation, cookies, timestamp."""

    messages: list[Message] = field(default_factory=list)
    conversation_id: str = ""
    cookies: dict[str, str] = field(default_factory=dict)
    at: float = 0.0


def _jar_cookies(client: httpx.AsyncClient) -> dict[str, str]:
    try:
        return {
            cookie.name: cookie.value
            for cookie in client.cookies.jar
            if cookie.value is not None
        }
    except Exception:
        return {}


class HeyClient:
    """Stateful client for the Hey_ website backend."""

    def __init__(self, experience_id: str | None = None) -> None:
        self._experience_id = experience_id or settings.experience_id
        self._resolved_experience_id: str | None = None
        self._last_turn: TurnState | None = None

    async def resolve_experience_id(self, client: httpx.AsyncClient) -> str:
        """Return the experience ID for new conversations.

        Explicit override (constructor/env) wins without any request.
        Otherwise pick a usable chat experience from GET /api/home,
        falling back to the last known default. Resolved once per client.
        """
        if self._experience_id:
            return self._experience_id
        if self._resolved_experience_id is not None:
            return self._resolved_experience_id
        try:
            response = await client.get("/api/home", params={"page": 1, "limit": 15})
            response.raise_for_status()
            items = response.json().get("data") or []
            self._resolved_experience_id = pick_experience(items)
        except (httpx.HTTPError, ValueError, AttributeError):
            self._resolved_experience_id = DEFAULT_EXPERIENCE_ID
        return self._resolved_experience_id

    async def warmup(self) -> None:
        """Resolve the experience eagerly (server startup).

        Saves one GET /api/home on the first turn (and its TTFT), and
        primes the pooled transport. Never raises – worst case the first
        turn resolves lazily as before.
        """
        try:
            client = httpx.AsyncClient(
                transport=shared_transport(),
                base_url=settings.base_url,
                headers=BROWSER_HEADERS,
                timeout=settings.timeout,
            )
            try:
                await self.resolve_experience_id(client)
            finally:
                await client.aclose()
        except Exception:
            pass

    @asynccontextmanager
    async def session(self) -> AsyncIterator[HeySession]:
        async with httpx.AsyncClient(
            transport=shared_transport(),
            base_url=settings.base_url,
            headers=BROWSER_HEADERS,
            timeout=settings.timeout,
        ) as client:
            experience_id = await self.resolve_experience_id(client)
            conv = await client.post(
                "/api/conversations",
                json={"experienceId": experience_id},
            )
            conv.raise_for_status()
            yield client, conv.json()["conversationId"]

    @asynccontextmanager
    async def turn_session(
        self, messages: list[Message]
    ) -> AsyncIterator[tuple[httpx.AsyncClient, str, bool]]:
        """Yield (client, conversation_id, reused) for one proxy turn.

        When this turn's history strictly extends the previous turn's (the
        normal agent-loop shape) and that turn is fresh, its conversation
        and cookies are reused – saving one conversation POST. Otherwise a
        new conversation is opened. Disable via HEY_REUSE_CONVERSATION=0.
        """
        now = time.monotonic()
        prev = self._last_turn
        if (
            settings.reuse_conversation
            and prev is not None
            and now - prev.at < settings.reuse_ttl
            and len(messages) > len(prev.messages)
            and messages[: len(prev.messages)] == prev.messages
        ):
            client = httpx.AsyncClient(
                transport=shared_transport(),
                base_url=settings.base_url,
                headers=BROWSER_HEADERS,
                timeout=settings.timeout,
                cookies=prev.cookies,
            )
            try:
                yield client, prev.conversation_id, True
            finally:
                self._last_turn = TurnState(
                    list(messages),
                    prev.conversation_id,
                    _jar_cookies(client),
                    time.monotonic(),
                )
                await client.aclose()
            return
        async with self.session() as (client, conversation_id):
            try:
                yield client, conversation_id, False
            finally:
                self._last_turn = TurnState(
                    list(messages),
                    conversation_id,
                    _jar_cookies(client),
                    time.monotonic(),
                )

    async def _chat_stream(
        self, client: httpx.AsyncClient, conversation_id: str, message: str
    ) -> AsyncIterator[HeyEvent]:
        async with client.stream(
            "POST",
            "/api/chat",
            json={"message": message, "source": "custom"},
            headers={
                "Accept": "text/event-stream",
                "x-conversation-id": conversation_id,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield ("content", delta["content"])
                found = extract_sources(delta)
                if found:
                    yield ("sources", found)
                full = choice.get("message") or {}
                text = extract_message_text(full)
                if text:
                    yield ("final", text)

    async def events(
        self, message: str, session: HeySession | None = None
    ) -> AsyncIterator[HeyEvent]:
        """Run the Hey_ flow.

        Yields ("content", str) | ("sources", dict) | ("final", str) |
        ("done", None). "sources" maps citation indexes to title/url and can
        arrive before the text that references them.
        """
        if session is None:
            async with self.session() as (client, conversation_id):
                async for event in self._chat_stream(client, conversation_id, message):
                    yield event
        else:
            async for event in self._chat_stream(session[0], session[1], message):
                yield event
        yield ("done", None)

    async def full_text_with_sources(
        self, message: str, session: HeySession | None = None
    ) -> tuple[str, Sources]:
        """Collect answer text plus any citation sources seen on the way."""
        parts: list[str] = []
        final: str | None = None
        by_index: Sources = {}
        async for kind, value in self.events(message, session):
            if kind == "content":
                parts.append(value)
            elif kind == "final":
                final = value
            elif kind == "sources":
                by_index.update(value)
        text = final if final is not None else "".join(parts)
        return text, by_index

    async def full_text(
        self, message: str, session: HeySession | None = None
    ) -> str:
        """Collect the Hey_ answer into one string (for non-streaming)."""
        text, _ = await self.full_text_with_sources(message, session)
        return text

    async def summarize(
        self, jobs: list[str], session: HeySession | None = None
    ) -> list[str]:
        """Summarize intermediate jobs – in parallel, order preserved.

        Jobs are independent (only their summaries feed the final), so the
        chunk phase costs max instead of sum.
        """
        if len(jobs) <= 1:
            return [await self.full_text(job, session) for job in jobs]
        return list(
            await asyncio.gather(
                *(self.full_text(job, session) for job in jobs)
            )
        )

    async def answer(
        self,
        message: str,
        tool_defs: list[JsonDict] | None,
        user_text: str = "",
        session: HeySession | None = None,
    ) -> tuple[str, Sources]:
        """Fetch the Hey_ answer plus citation sources.

        Retry budget: at most 2 extra calls. An explicit action request
        without any tool call always retries once (objective criterion, no
        phrase matching); deflection and news/weather drift retry with
        their nudges. Sources always come from the accepted attempt.
        (Disable retries via HEY_TOOL_RETRY=0.)
        """
        answer, sources = await self.full_text_with_sources(message, session)
        if not tool_defs or not settings.tool_retry:
            return answer, sources
        for _ in range(2):
            _, calls = tools.extract_tool_calls(answer)
            if calls:
                if not tools.calls_with_empty_args(calls, tool_defs):
                    return answer, sources
                nudge = tools.EMPTY_ARGS_NUDGE
            elif tools.denies_capability(answer):
                nudge = tools.CAPABILITY_NUDGE
            elif tools.is_action_request(user_text):
                nudge = tools.TOOL_RETRY_NUDGE
            elif tools.is_deflection(answer):
                nudge = tools.TOOL_RETRY_NUDGE
            else:
                kind = tools.drift_kind(answer, user_text)
                if not kind:
                    return answer, sources
                topics = (
                    "Nachrichten- und Schlagzeilen-Themen"
                    if kind == "news"
                    else "Wetter-Themen"
                )
                nudge = tools.news_refocus_nudge(user_text, topics)
            answer, sources = await self.full_text_with_sources(
                f"{message}\n\n{nudge}", session
            )
        return answer, sources
