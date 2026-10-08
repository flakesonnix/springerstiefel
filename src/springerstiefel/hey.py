"""Hey_ backend client: conversations, SSE chat, retries, summaries."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from springerstiefel import tools
from springerstiefel.config import BROWSER_HEADERS, settings
from springerstiefel.types import HeyEvent, JsonDict, Message


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


# One session = one HTTP client + one Hey_ conversation.
# A proxy turn shares exactly one session (saves one conversation POST
# ~170ms per extra call and keeps the turn server-side in one conversation).
HeySession = tuple[httpx.AsyncClient, str]


class HeyClient:
    """Stateful client for the Hey_ website backend."""

    def __init__(self, experience_id: str | None = None) -> None:
        self.experience_id = experience_id or settings.experience_id

    @asynccontextmanager
    async def session(self) -> AsyncIterator[HeySession]:
        async with httpx.AsyncClient(
            base_url=settings.base_url,
            headers=BROWSER_HEADERS,
            timeout=settings.timeout,
        ) as client:
            conv = await client.post(
                "/api/conversations",
                json={"experienceId": self.experience_id},
            )
            conv.raise_for_status()
            yield client, conv.json()["conversationId"]

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
                full = choice.get("message") or {}
                text = extract_message_text(full)
                if text:
                    yield ("final", text)

    async def events(
        self, message: str, session: HeySession | None = None
    ) -> AsyncIterator[HeyEvent]:
        """Run the Hey_ flow. Yields ("content", str) | ("final", str) | ("done", None).

        "content": streaming delta. "final": full text from the closing
        object (duplicate of the deltas – prefer when present).
        Without a session, one is opened per call.
        """
        if session is None:
            async with self.session() as (client, conversation_id):
                async for event in self._chat_stream(client, conversation_id, message):
                    yield event
        else:
            async for event in self._chat_stream(session[0], session[1], message):
                yield event
        yield ("done", None)

    async def full_text(
        self, message: str, session: HeySession | None = None
    ) -> str:
        """Collect the Hey_ answer into one string (for non-streaming)."""
        parts: list[str] = []
        final: str | None = None
        async for kind, value in self.events(message, session):
            if kind == "content":
                parts.append(value)
            elif kind == "final":
                final = value
        return final if final is not None else "".join(parts)

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
    ) -> str:
        """Fetch the Hey_ answer, with one retry each on deflection and drift.

        - Deflection (refusal/counter-question despite tools): once with nudge.
        - Off-topic drift (news/weather dump although neither was asked for):
          once refocused on the task.
        (Disable via HEY_TOOL_RETRY=0.)
        """
        answer = await self.full_text(message, session)
        if not tool_defs or not settings.tool_retry:
            return answer
        _, calls = tools.extract_tool_calls(answer)
        if calls:
            return answer
        if tools.is_deflection(answer):
            second = await self.full_text(
                f"{message}\n\n{tools.TOOL_RETRY_NUDGE}", session
            )
            _, calls = tools.extract_tool_calls(second)
            if calls or not tools.drift_kind(second, user_text):
                return second
            answer = second
        kind = tools.drift_kind(answer, user_text)
        if kind:
            topics = (
                "Nachrichten- und Schlagzeilen-Themen"
                if kind == "news"
                else "Wetter-Themen"
            )
            return await self.full_text(
                f"{message}\n\n{tools.news_refocus_nudge(user_text, topics)}",
                session,
            )
        return answer
