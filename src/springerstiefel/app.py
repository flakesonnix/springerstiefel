"""FastAPI app: OpenAI-compatible HTTP layer (:8787)."""

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from springerstiefel import hey as hey_module
from springerstiefel import messages, tools
from springerstiefel.hey import HeyClient, HeySession
from springerstiefel.messages import message_text
from springerstiefel.types import JsonDict, Message, ToolChoice

hey = HeyClient()

DEBUG = os.environ.get("HEY_DEBUG", "0") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await hey.warmup()
    yield


app = FastAPI(title="springerstiefel", lifespan=lifespan)


def debug_log(**fields: object) -> None:
    if DEBUG:
        line = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"[hey-proxy] {line}", flush=True)


@app.get("/v1/models")
async def models() -> JsonDict:
    return {
        "object": "list",
        "data": [
            {
                "id": "hey",
                "object": "model",
                "owned_by": "bild",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
) -> Response:
    body: JsonDict = await request.json()

    request_messages: list[Message] = body.get("messages", [])
    stream: bool = body.get("stream", False)
    tool_defs: list[JsonDict] | None = body.get("tools") or None
    tool_choice: ToolChoice = body.get("tool_choice", "auto")

    if not request_messages:
        raise HTTPException(400, "No messages supplied")

    user_text = " ".join(
        message_text(m) for m in request_messages if m.get("role") == "user"
    )

    async def resolve_text(session: HeySession) -> str:
        """Single request or chunk queue (FIFO, summary chain).

        Small histories take the normal path. Otherwise older history parts
        are summarized in parallel and the summaries feed the final request.
        All calls share the turn session (one client, one Hey_ conversation).
        """
        intermediates, final_parts = messages.build_hey_jobs(
            request_messages, tool_defs, tool_choice
        )
        if final_parts is None:
            return messages.build_hey_message(
                request_messages, tool_defs, tool_choice
            )
        summaries = await hey.summarize(intermediates, session)
        return messages.render_final(final_parts, summaries)

    def completion_message(answer: str) -> tuple[Message, str | None]:
        """Build (message, finish_reason) – with tool calls if present."""
        if tool_defs:
            clean, tool_calls = tools.extract_tool_calls(answer)
            if tool_calls:
                return (
                    {
                        "role": "assistant",
                        "content": clean or None,
                        "tool_calls": tool_calls,
                    },
                    "tool_calls",
                )
        return ({"role": "assistant", "content": answer}, "stop")

    def chunk(
        content: str | None = None,
        tool_calls: list[JsonDict] | None = None,
        role: str | None = None,
        finish: str | None = None,
    ) -> str:
        delta: JsonDict = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        payload = {
            "id": "hey-proxy",
            "object": "chat.completion.chunk",
            "model": "hey",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish}
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    try:
        started = time.perf_counter()
        if not stream:
            async with hey.turn_session(request_messages) as (client, cid, reused):
                session_ms = (time.perf_counter() - started) * 1000
                text = await resolve_text((client, cid))
                answer, sources = await hey.answer(
                    text, tool_defs, user_text, (client, cid)
                )
            debug_log(
                mode="non-stream",
                in_messages=len(request_messages),
                tools=len(tool_defs or []),
                hey_chars=len(text),
                reused=reused,
                session_ms=f"{session_ms:.0f}",
                total_ms=f"{(time.perf_counter() - started) * 1000:.0f}",
                finish="n/a",
            )
            message, finish = completion_message(
                hey_module.append_sources(answer, sources)
            )
            return JSONResponse({
                "id": "hey-proxy",
                "object": "chat.completion",
                "model": "hey",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish}
                ],
            })

        # With tools: collect first (tool calls can't stream live), then emit
        # chunks. Without tools: live passthrough. Each streaming response
        # opens its own turn session (generators start after return – never
        # reuse anything from the endpoint scope).
        if tool_defs:
            async def generate_buffered() -> AsyncIterator[str]:
                started = time.perf_counter()
                async with hey.turn_session(request_messages) as (client, cid, reused):
                    session_ms = (time.perf_counter() - started) * 1000
                    text = await resolve_text((client, cid))
                    yield chunk(role="assistant")
                    answer, sources = await hey.answer(
                        text, tool_defs, user_text, (client, cid)
                    )
                    debug_log(
                        mode="stream-buffered",
                        in_messages=len(request_messages),
                        tools=len(tool_defs or []),
                        hey_chars=len(text),
                        reused=reused,
                        session_ms=f"{session_ms:.0f}",
                        total_ms=f"{(time.perf_counter() - started) * 1000:.0f}",
                    )
                    message, finish = completion_message(
                        hey_module.append_sources(answer, sources)
                    )
                    if message.get("content"):
                        yield chunk(content=message["content"])
                    for i, call in enumerate(message.get("tool_calls") or []):
                        indexed = dict(call, index=i)
                        yield chunk(tool_calls=[indexed])
                    yield chunk(finish=finish)
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_buffered(), media_type="text/event-stream"
            )

        async def generate() -> AsyncIterator[str]:
            started = time.perf_counter()
            first_ms: float | None = None
            async with hey.turn_session(request_messages) as (client, cid, reused):
                session_ms = (time.perf_counter() - started) * 1000
                text = await resolve_text((client, cid))
                yield chunk(role="assistant")
                seen_text: list[str] = []
                by_index: hey_module.Sources = {}
                async for kind, value in hey.events(text, (client, cid)):
                    if kind == "content":
                        if first_ms is None:
                            first_ms = (time.perf_counter() - started) * 1000
                        seen_text.append(value)
                        yield chunk(content=value)
                    elif kind == "sources":
                        by_index.update(value)
                section = hey_module.resolve_sources("".join(seen_text), by_index)
                if section:
                    yield chunk(content=hey_module.format_sources(section))
                yield chunk(finish="stop")
                yield "data: [DONE]\n\n"
            debug_log(
                mode="stream-live",
                in_messages=len(request_messages),
                hey_chars=len(text),
                reused=reused,
                session_ms=f"{session_ms:.0f}",
                ttft_ms=f"{first_ms:.0f}" if first_ms is not None else "n/a",
                total_ms=f"{(time.perf_counter() - started) * 1000:.0f}",
            )

        return StreamingResponse(generate(), media_type="text/event-stream")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Hey_ backend error: {e}") from e


def main() -> None:
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8787,
    )


if __name__ == "__main__":
    main()
