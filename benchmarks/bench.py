#!/usr/bin/env python3
"""Benchmarks for springerstiefel – verbose phase timing.

Sections:
  A. Python build phase (build_hey_message) with growing history – offline
  B. Live Hey_ phases: conversation POST, time-to-first-chunk, chunks, total
  C. Chunk queue with forced-small budget (sequential baseline)
  D. Concurrency: N parallel requests (Hey_-side behavior)
  E. Micro-benchmarks: parsing/building at scale – offline, best of N

Run from nix develop with the venv active, from the repo root:
    python3 benchmarks/bench.py            # everything (hits hey.bild.de)
    python3 benchmarks/bench.py --quick    # offline only (A + E)
"""
import asyncio
import json
import sys
import time

import httpx

from springerstiefel import config as config_module
from springerstiefel import language as language_module
from springerstiefel import messages as messages_module
from springerstiefel import tools as tools_module
from springerstiefel.hey import HeyClient, append_sources, extract_message_text

hey_client = HeyClient()

BASE = "https://hey.bild.de"
EXP_ID = "a5d82531-015a-46d0-9547-47602fe9b03e"


def log(section: str, **fields: object) -> None:
    print(f"[{section}]")
    for key, value in fields.items():
        print(f"    {key:<22} {value}")
    print()


def make_history(turns: int, tool_bytes: int = 0) -> list:
    messages: list = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"Question {i}: name a color."})
        messages.append({"role": "assistant", "content": f"Answer {i}: blue."})
    if tool_bytes:
        messages.append({"role": "user", "content": "Read the file."})
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": "call_1",
            "content": "x" * tool_bytes,
        })
    messages.append({"role": "user", "content": "Finally: say hello."})
    return messages


def bench_build() -> None:
    print("=" * 70)
    print("A. Python build phase (build_hey_message)")
    print("=" * 70)
    for turns, tool_bytes in [(2, 0), (10, 0), (15, 50_000), (15, 200_000)]:
        messages = make_history(turns, tool_bytes)
        t0 = time.perf_counter()
        text = messages_module.build_hey_message(messages)
        dt = (time.perf_counter() - t0) * 1000
        intermediates, final_parts = messages_module.build_hey_jobs(messages, None, "auto")
        log(
            f"turns={turns} tool_bytes={tool_bytes}",
            chars=len(text),
            build_ms=f"{dt:.2f}",
            jobs=len(intermediates) + 1,
        )


async def timed_flow(label: str, hey_message: str) -> dict:
    """Hey_ flow with phase timing (conversation, TTFT, chunks, total)."""
    print(f"--- {label} ---")
    phases: dict = {}
    t_total = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=BASE, headers=config_module.BROWSER_HEADERS, timeout=120.0
    ) as client:
        t0 = time.perf_counter()
        conv = await client.post("/api/conversations", json={"experienceId": EXP_ID})
        conv.raise_for_status()
        cid = conv.json()["conversationId"]
        phases["conversation_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        first_chunk_ms = None
        chunks = 0
        chars = 0
        async with client.stream(
            "POST", "/api/chat",
            json={"message": hey_message, "source": "custom"},
            headers={"Accept": "text/event-stream", "x-conversation-id": cid},
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
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        chunks += 1
                        chars += len(delta["content"])
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - t0) * 1000
                    full = choice.get("message") or {}
                    if isinstance(full.get("content"), str):
                        chars += len(full["content"])
        phases["ttft_ms"] = first_chunk_ms
        phases["stream_ms"] = (time.perf_counter() - t0) * 1000
        phases["chunks"] = chunks
        phases["chars"] = chars
    phases["total_ms"] = (time.perf_counter() - t_total) * 1000
    log(label, **{k: (f"{v:.0f}" if isinstance(v, float) else v) for k, v in phases.items()})
    return phases


async def bench_live() -> None:
    print("=" * 70)
    print("B. Live Hey_ phases (small / medium / large)")
    print("=" * 70)
    small = messages_module.build_hey_message([{"role": "user", "content": "Say hello."}])
    await timed_flow("small (1 msg)", small)

    medium = messages_module.build_hey_message(make_history(10))
    await timed_flow("medium (10 turns)", medium)

    big = messages_module.build_hey_message(make_history(10, tool_bytes=100_000))
    await timed_flow("large (10 turns + 100KB tool)", big)


async def bench_chunked() -> None:
    print("=" * 70)
    print("C. Chunk queue (forced small, via summarize_intermediates)")
    print("=" * 70)
    old_chars = config_module.settings.max_chunk_chars
    old_chunks = config_module.settings.max_chunks
    config_module.settings.max_chunk_chars = 400
    config_module.settings.max_chunks = 6
    try:
        messages = make_history(8)
        intermediates, final_parts = messages_module.build_hey_jobs(messages, None, "auto")
        assert final_parts is not None  # forced-small budget always splits
        log("jobs", intermediates=len(intermediates), final=True)
        t0 = time.perf_counter()
        summaries = await hey_client.summarize(intermediates)
        log("parallel-phase", ms=f"{(time.perf_counter() - t0) * 1000:.0f}")
        final = messages_module.render_final(final_parts, summaries)
        t1 = time.perf_counter()
        answer = await hey_client.full_text(final)
        log("final",
            ms=f"{(time.perf_counter() - t1) * 1000:.0f}",
            answer=answer[:120])
        log("chunked-total", ms=f"{(time.perf_counter() - t0) * 1000:.0f}")
    finally:
        config_module.settings.max_chunk_chars = old_chars
        config_module.settings.max_chunks = old_chunks


async def bench_concurrent() -> None:
    print("=" * 70)
    print("D. Concurrency (3 parallel small requests)")
    print("=" * 70)
    small = messages_module.build_hey_message([{"role": "user", "content": "Say hello."}])
    t0 = time.perf_counter()

    async def one(i: int) -> None:
        t1 = time.perf_counter()
        answer = await hey_client.full_text(small)
        log(f"request {i}", ms=f"{(time.perf_counter() - t1) * 1000:.0f}",
            answer=answer[:40])

    await asyncio.gather(one(1), one(2), one(3))
    log("parallel-total", ms=f"{(time.perf_counter() - t0) * 1000:.0f}")


def best_of(label: str, rounds: int, func, *args):
    """Run func N times, report best/worst in ms."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - t0) * 1000)
    log(label, best_ms=f"{min(times):.3f}", worst_ms=f"{max(times):.3f}")


def bench_micro() -> None:
    print("=" * 70)
    print("E. Micro-benchmarks (offline, best of 20)")
    print("=" * 70)
    history = make_history(30, tool_bytes=50_000)
    text = messages_module.build_hey_message(history)
    log("fixture", history_messages=len(history), message_chars=len(text))

    best_of("build_hey_message (30 turns)", 20, messages_module.build_hey_message, history)
    best_of("build_hey_jobs (30 turns)", 20, messages_module.build_hey_jobs,
            history, None, "auto")

    blocky = ("Text.\n<<TOOL_CALL>>\n"
              '{"name": "write", "arguments": {"path": "f.txt", "content": "x"}}'
              "\n<<END_TOOL_CALL>>\n") * 50
    best_of("extract_tool_calls (50 blocks)", 20, tools_module.extract_tool_calls, blocky)

    envelope = json.dumps({"answer": "A" * 5000, "suggestions": ["a", "b"]})
    best_of("extract_message_text (5KB envelope)", 20,
            extract_message_text, {"content": envelope})

    many_lines = [f"line {i}: {'y' * 200}" for i in range(500)]
    best_of("split_lines (500 lines)", 20, messages_module.split_lines, many_lines, 6000)

    best_of("detect_language (200 chars)", 20,
            language_module.detect_language,
            "Write a calculator in rust with tests and docs. " * 4)
    best_of("is_action_request", 20,
            tools_module.is_action_request, "Write the file f.")
    best_of("append_sources (5 markers)", 20,
            append_sources,
            "See [bild_0_0] and [bild_0_1].",
            {"bild_0_0": {"title": "A", "url": "https://x/0"},
             "bild_0_1": {"title": "B", "url": "https://x/1"}})


async def bench_experience() -> None:
    print("=" * 70)
    print("F. Experience auto-resolve (cold vs cached, live)")
    print("=" * 70)
    client = HeyClient()

    t0 = time.perf_counter()
    async with client.session() as (http_client, cid):
        cold_ms = (time.perf_counter() - t0) * 1000
    log("cold resolve + conversation", ms=f"{cold_ms:.0f}", cid=cid[:8])

    t0 = time.perf_counter()
    async with client.session() as (http_client2, cid2):
        warm_ms = (time.perf_counter() - t0) * 1000
    log("cached resolve + conversation", ms=f"{warm_ms:.0f}", cid=cid2[:8])
    log("resolve overhead (cold-warm)", ms=f"{cold_ms - warm_ms:.0f}")


async def bench_endpoint() -> None:
    print("=" * 70)
    print("G. Endpoint overhead with mocked backend (offline)")
    print("=" * 70)
    from fastapi.testclient import TestClient

    from springerstiefel.app import app

    history = make_history(30, tool_bytes=50_000)

    async def fake_events(self, message, session=None):
        yield ("final", "Hallo!")

    import unittest.mock

    with unittest.mock.patch.object(
        HeyClient, "events", fake_events
    ):
        test_client = TestClient(app)

        def post_plain():
            test_client.post(
                "/v1/chat/completions",
                json={"model": "hey", "messages": history},
            )

        def post_stream():
            response = test_client.post(
                "/v1/chat/completions",
                json={"model": "hey", "messages": history, "stream": True},
            )
            assert response.status_code == 200

        best_of_sync("POST non-stream (30 turns)", 5, post_plain)
        best_of_sync("POST stream (30 turns)", 5, post_stream)


def best_of_sync(label: str, rounds: int, func) -> None:
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    log(label, best_ms=f"{min(times):.1f}", worst_ms=f"{max(times):.1f}")


async def main() -> None:
    quick = "--quick" in sys.argv
    bench_build()
    bench_micro()
    if quick:
        print("(skipping live sections B/C/D/F/G – pass no flags for full run)")
        return
    await bench_live()
    await bench_chunked()
    await bench_concurrent()
    await bench_experience()
    await bench_endpoint()


if __name__ == "__main__":
    asyncio.run(main())
