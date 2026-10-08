import asyncio
import json
import time
import unittest.mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import proxy


client = TestClient(proxy.app)

MESSAGES = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Sag einfach hallo"},
]

FAKE_EVENTS = [
    ("content", "Hal"),
    ("content", "lo"),
    ("final", "Hallo"),
    ("done", None),
]


async def fake_hey_events(message: str, session=None):
    assert "Sag einfach hallo" in message
    for event in FAKE_EVENTS:
        yield event


@pytest.fixture(autouse=True)
def mock_hey_backend(monkeypatch):
    monkeypatch.setattr(proxy, "hey_events", fake_hey_events)


# Echte Implementierung (Importzeitpunkt, vor Fixture-Patches) für Tests,
# die den Session-Fluss bis _hey_events prüfen.
_REAL_HEY_EVENTS = proxy.hey_events


def test_models_lists_hey():
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"] == [
        {"id": "hey", "object": "model", "owned_by": "bild"}
    ]


def test_chat_completions_rejects_missing_messages():
    response = client.post("/v1/chat/completions", json={"model": "hey"})

    assert response.status_code == 400

    response = client.post(
        "/v1/chat/completions", json={"model": "hey", "messages": []}
    )

    assert response.status_code == 400


def test_chat_completions_prefers_final_text():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "hey", "messages": MESSAGES},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "hey"

    (choice,) = body["choices"]
    assert choice["message"] == {"role": "assistant", "content": "Hallo"}
    assert choice["finish_reason"] == "stop"


def _parse_sse(text: str):
    chunks = []
    done = False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, done


def test_chat_completions_stream_returns_sse():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "hey", "messages": MESSAGES, "stream": True},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    chunks, done = _parse_sse(response.text)

    assert done is True
    assert len(chunks) == 4
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"content": "Hal"}
    assert chunks[2]["choices"][0]["delta"] == {"content": "lo"}
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"


def test_last_user_text_takes_last_user_message():
    messages = [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "zweite Frage"},
    ]

    assert proxy.last_user_text(messages) == "zweite Frage"


def test_last_user_text_joins_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hallo "},
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": "Welt"},
            ],
        }
    ]

    assert proxy.last_user_text(messages) == "Hallo Welt"


def test_build_hey_message_embeds_system_and_history():
    messages = [
        {"role": "system", "content": "Sei knapp."},
        {"role": "user", "content": "Was ist 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Und 3+3?"},
    ]

    text = proxy.build_hey_message(messages)

    assert "[Systemanweisung]\nSei knapp." in text
    assert "Benutzer: Was ist 2+2?" in text
    assert "Assistent: 4" in text
    assert text.endswith("Aktuelle Anweisung:\nUnd 3+3?")


def test_build_hey_message_renders_tool_roundtrip():
    messages = [
        {"role": "user", "content": "Wie spät ist es?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        {"role": "user", "content": "Danke!"},
    ]

    text = proxy.build_hey_message(messages)

    assert "Assistent (Werkzeugaufruf get_time: {})" in text
    assert "Werkzeugergebnis (call_1): 12:00" in text
    assert text.endswith("Aktuelle Anweisung:\nDanke!")


def test_build_hey_message_continues_after_tool_result():
    messages = [
        {"role": "user", "content": "Wie spät ist es?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
    ]

    text = proxy.build_hey_message(messages)

    # Frage wird NICHT wiederholt (sonst Tool-Loop) …
    assert text.count("Wie spät ist es?") == 1
    # … stattdessen Fortsetzung mit Ergebnis am Ende.
    assert "Werkzeugergebnis (call_1): 12:00" in text
    assert text.endswith(
        "Aktuelle Anweisung:\nSetze die Bearbeitung fort. "
        "Das neueste Werkzeugergebnis steht am Ende des Verlaufs."
    )


def test_build_hey_message_lists_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Liest eine Datei",
                "parameters": {"type": "object"},
            },
        }
    ]

    text = proxy.build_hey_message(
        [{"role": "user", "content": "Lies foo.txt"}], tools
    )

    assert "read_file" in text
    assert "<<TOOL_CALL>>" in text


def test_build_hey_message_hides_tools_on_choice_none():
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file"},
        }
    ]

    text = proxy.build_hey_message(
        [{"role": "user", "content": "Hallo"}], tools, tool_choice="none"
    )

    assert "read_file" not in text
    assert "<<TOOL_CALL>>" not in text


def test_build_hey_message_marks_required_call():
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file"},
        }
    ]

    text = proxy.build_hey_message(
        [{"role": "user", "content": "Hallo"}], tools, tool_choice="required"
    )

    assert "MUSST" in text


def test_extract_tool_calls_parses_blocks():
    answer = (
        'Gerne!\n<<TOOL_CALL>>\n{"name": "read_file", '
        '"arguments": {"path": "foo.txt"}}\n<<END_TOOL_CALL>>'
    )

    clean, calls = proxy.extract_tool_calls(answer)

    assert clean == "Gerne!"
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "foo.txt"}',
            },
        }
    ]


def test_extract_tool_calls_keeps_broken_blocks():
    answer = "<<TOOL_CALL>>\nkein json\n<<END_TOOL_CALL>>"

    clean, calls = proxy.extract_tool_calls(answer)

    assert calls == []
    assert clean == answer


def test_extract_message_text_plain_content():
    assert proxy.extract_message_text({"content": "Hallo!"}) == "Hallo!"


def test_extract_message_text_json_envelope_string():
    message = {
        "content": json.dumps(
            {"answer": "Hier der Code.", "suggestions": ["Weiter"]}
        )
    }

    assert proxy.extract_message_text(message) == "Hier der Code."


def test_extract_message_text_json_envelope_dict():
    message = {"content": {"answer": "Antwort", "suggestions": []}}

    assert proxy.extract_message_text(message) == "Antwort"


def test_extract_message_text_parsed_fallback():
    message = {"content": "", "parsed": {"answer": "Fallback"}}

    assert proxy.extract_message_text(message) == "Fallback"


def test_extract_message_text_empty():
    assert proxy.extract_message_text({}) == ""
    assert proxy.extract_message_text({"content": ""}) == ""


def test_is_deflection_detects_refusals():
    assert proxy.is_deflection("Wobei soll ich im Projekt helfen?")
    assert proxy.is_deflection("Wobei kann ich Ihnen helfen?")
    assert proxy.is_deflection("Das kann ich hier nicht ausführen.")
    assert proxy.is_deflection("Nennen Sie bitte Pfad und Inhalt.")
    assert proxy.is_deflection("Wie kann ich Ihnen helfen?")
    assert proxy.is_deflection("Ich kann in diesem Schritt keine Datei anlegen.")


def test_is_deflection_accepts_normal_text():
    assert not proxy.is_deflection("Hier ist der Code:\n```rust\nfn main() {}```")
    assert not proxy.is_deflection("Es ist 14:37 Uhr.")


def test_hey_answer_retries_deflection_once():
    calls = {"n": 0}

    async def flaky_events(message: str, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("final", "Wobei soll ich im Projekt helfen?")
        else:
            assert proxy.TOOL_RETRY_NUDGE in message
            yield ("final", '<<TOOL_CALL>>\n{"name": "write", "arguments": {}}\n<<END_TOOL_CALL>>')
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(proxy, "hey_events", flaky_events):
            return await proxy.hey_answer("Mach X.", [{"type": "function"}])

    answer = asyncio.run(run())

    assert calls["n"] == 2
    assert "TOOL_CALL" in answer


def test_hey_answer_no_retry_without_tools():
    calls = {"n": 0}

    async def events(message: str, session=None):
        calls["n"] += 1
        yield ("final", "Wobei soll ich helfen?")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(proxy, "hey_events", events):
            return await proxy.hey_answer("Hallo.", None)

    answer = asyncio.run(run())

    assert calls["n"] == 1
    assert answer == "Wobei soll ich helfen?"


def test_tool_results_are_truncated(monkeypatch):
    monkeypatch.setattr(proxy, "HEY_MAX_TOOL_CHARS", 20)
    messages = [
        {"role": "user", "content": "Frage"},
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 100},
        {"role": "user", "content": "Und?"},
    ]

    text = proxy.build_hey_message(messages)

    assert ("x" * 100) not in text
    assert "[… Ausgabe gekürzt …]" in text
    assert text.endswith("Aktuelle Anweisung:\nUnd?")


def test_split_lines_keeps_wholes_lines_in_budget():
    lines = ["aaa", "bbb", "ccccc"]

    chunks = proxy.split_lines(lines, 8)

    assert chunks == [["aaa", "bbb"], ["ccccc"]]


def test_split_lines_empty():
    assert proxy.split_lines([], 100) == []


def test_build_hey_jobs_single_for_small_history():
    messages = [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = proxy.build_hey_jobs(messages, None, "auto")

    assert intermediates == []
    assert final_parts is None


def test_build_hey_jobs_splits_big_history(monkeypatch):
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNK_CHARS", 40)
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNKS", 5)
    messages = [
        {"role": "user", "content": "erste alte Frage"},
        {"role": "assistant", "content": "alte Antwort"},
        {"role": "user", "content": "noch eine Frage"},
        {"role": "assistant", "content": "noch eine Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = proxy.build_hey_jobs(messages, None, "auto")

    # Nichts geht verloren (statt Droppen wie früher) …
    assert len(intermediates) >= 1
    assert "Teil 1/" in intermediates[0]
    assert "Fasse in 2–3 Sätzen zusammen" in intermediates[0]
    # … aktuelle Anweisung nur im Finale.
    assert "Frage neu" not in "\n".join(intermediates)
    assert final_parts["current"] == "Frage neu"

    rendered = proxy.render_final(final_parts, ["Zwischenfazit"])
    assert "Zwischenfazit" in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nFrage neu")


def test_build_hey_jobs_caps_chunk_count(monkeypatch):
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNK_CHARS", 10)
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNKS", 2)
    messages = [
        {"role": "user", "content": f"Frage {i} mit viel Text dahinter"}
        for i in range(10)
    ]
    messages.append({"role": "user", "content": "Frage neu"})

    intermediates, final_parts = proxy.build_hey_jobs(messages, None, "auto")

    assert len(intermediates) + 1 <= 2
    assert final_parts["current"] == "Frage neu"


def test_chunked_flow_chains_summaries(monkeypatch):
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNK_CHARS", 40)
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNKS", 5)
    seen = []

    async def fake_full_text(message: str, session=None):
        seen.append(message)
        return f"Summary {len(seen)}"

    async def run():
        with unittest.mock.patch.object(proxy, "hey_full_text", fake_full_text):
            with unittest.mock.patch.object(
                proxy, "hey_events", fake_hey_events
            ):
                intermediates, final_parts = proxy.build_hey_jobs(
                    [
                        {"role": "user", "content": "erste alte Frage"},
                        {"role": "assistant", "content": "alte Antwort"},
                        {"role": "user", "content": "noch eine Frage"},
                        {"role": "assistant", "content": "noch eine Antwort"},
                        {"role": "user", "content": "Frage neu"},
                    ],
                    None,
                    "auto",
                )
                summaries = []
                for job in intermediates:
                    summaries.append(await proxy.hey_full_text(job))
                return intermediates, proxy.render_final(final_parts, summaries)

    intermediates, rendered = asyncio.run(run())

    assert len(seen) == len(intermediates)
    assert "Summary 1" in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nFrage neu")


def test_is_news_drift_detects_headline_dump():
    dump = "## Aktuelle BILD-Schlagzeilen von heute\nPolitik: ... [bild_0_1]"

    assert proxy.is_news_drift(dump, "Erstelle die Datei hello.rs.")
    assert not proxy.is_news_drift("Hier ist der Code.", "Erstelle hello.rs.")


def test_is_news_drift_skips_genuine_news_requests():
    dump = "Schlagzeilen: ... [bild_0_1]"

    assert not proxy.is_news_drift(dump, "Was sind die Nachrichten heute?")


def test_hey_answer_refocuses_news_drift():
    calls = {"n": 0}

    async def events(message: str, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("final", "Schlagzeilen des Tages [bild_0_1]")
        else:
            assert "Bearbeite nur diese Aufgabe" in message
            assert "hello.rs" in message
            yield ("final", "Erledigt.")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(proxy, "hey_events", events):
            return await proxy.hey_answer(
                "Mach X.", [{"type": "function"}], "Erstelle hello.rs."
            )

    answer = asyncio.run(run())

    assert calls["n"] == 2
    assert answer == "Erledigt."


def test_chat_completions_returns_tool_calls():
    async def tool_events(message: str, session=None):
        yield ("final", 'Bitte sehr:\n<<TOOL_CALL>>\n{"name": "get_time", "arguments": {}}\n<<END_TOOL_CALL>>')
        yield ("done", None)

    with unittest.mock.patch.object(proxy, "hey_events", tool_events):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hey",
                "messages": MESSAGES,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_time"},
                    }
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    (choice,) = body["choices"]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_time"
    assert choice["message"]["content"] == "Bitte sehr:"


def test_chat_completions_stream_with_tools_emits_tool_chunk():
    async def tool_events(message: str, session=None):
        yield ("content", "Moment…")
        yield ("final", '<<TOOL_CALL>>\n{"name": "get_time", "arguments": {}}\n<<END_TOOL_CALL>>')
        yield ("done", None)

    with unittest.mock.patch.object(proxy, "hey_events", tool_events):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hey",
                "messages": MESSAGES,
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_time"},
                    }
                ],
            },
        )

    assert response.status_code == 200
    chunks, done = _parse_sse(response.text)

    assert done is True
    tool_deltas = [
        c["choices"][0]["delta"]
        for c in chunks
        if "tool_calls" in c["choices"][0]["delta"]
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "get_time"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_summarize_intermediates_runs_parallel():
    async def slow_full_text(message: str, session=None):
        await asyncio.sleep(0.4)
        return f"Summary of {message}"

    async def run():
        with unittest.mock.patch.object(proxy, "hey_full_text", slow_full_text):
            start = time.perf_counter()
            result = await proxy.summarize_intermediates(["job-a", "job-b"])
            elapsed = time.perf_counter() - start
            return result, elapsed

    result, elapsed = asyncio.run(run())

    assert result == ["Summary of job-a", "Summary of job-b"]
    # Sequentiell wären es 0.8s – parallel deutlich darunter.
    assert elapsed < 0.6


def test_turn_shares_single_session_across_retry():
    chats = []
    sessions = {"entries": 0}

    class FakeSession:
        async def __aenter__(self):
            sessions["entries"] += 1
            return ("fake-client", "cid-1")

        async def __aexit__(self, *args):
            return False

    async def fake_inner_events(client, conversation_id, message):
        chats.append((client, conversation_id))
        if len(chats) == 1:
            yield ("final", "Wobei soll ich helfen?")
        else:
            assert proxy.TOOL_RETRY_NUDGE in message
            yield ("final", '<<TOOL_CALL>>\n{"name": "write", "arguments": {}}\n<<END_TOOL_CALL>>')

    with unittest.mock.patch.object(proxy, "hey_session", lambda: FakeSession()):
        with unittest.mock.patch.object(proxy, "_hey_events", fake_inner_events):
            with unittest.mock.patch.object(proxy, "hey_events", _REAL_HEY_EVENTS):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hey",
                        "messages": MESSAGES,
                        "tools": [{"type": "function", "function": {"name": "write"}}],
                    },
                )

    assert response.status_code == 200
    # Eine Session/Conversation für den Turn, zwei Chat-Calls darin.
    assert sessions["entries"] == 1
    assert chats == [("fake-client", "cid-1"), ("fake-client", "cid-1")]
    (choice,) = response.json()["choices"]
    assert choice["finish_reason"] == "tool_calls"


def test_summarize_intermediates_single_and_empty():
    async def run():
        assert await proxy.summarize_intermediates([]) == []
        with unittest.mock.patch.object(
            proxy, "hey_full_text", lambda m, session=None: asyncio.sleep(0, result=f"S({m})")
        ):
            assert await proxy.summarize_intermediates(["solo"]) == ["S(solo)"]

    asyncio.run(run())


def test_message_text_handles_odd_content():
    assert proxy.message_text({"role": "user"}) == ""
    assert proxy.message_text({"role": "user", "content": None}) == ""
    assert proxy.message_text({"role": "user", "content": 123}) == ""
    assert proxy.message_text({"role": "user", "content": [{"type": "x"}]}) == ""


def test_last_user_index_raises_without_user():
    with pytest.raises(HTTPException) as exc:
        proxy.last_user_index([{"role": "system", "content": "x"}])

    assert exc.value.status_code == 400


def test_news_refocus_nudge_truncates_long_task():
    nudge = proxy.news_refocus_nudge("x" * 1000)

    assert nudge.endswith("…]")
    assert "x" * 1000 not in nudge
    assert "Bearbeite nur diese Aufgabe" in nudge


def test_render_final_variants():
    bare = {"system": None, "lines": [], "current": "Hi", "tools": None}

    assert proxy.render_final(bare, []) == "Aktuelle Anweisung:\nHi"

    full = {
        "system": "[Systemanweisung]\nSei knapp.",
        "lines": ["Benutzer: Frage"],
        "current": "Antworte.",
        "tools": "[Tools]\n<<TOOL_CALL>>",
    }
    rendered = proxy.render_final(full, [])

    assert "[Systemanweisung]" in rendered
    assert "Benutzer: Frage" in rendered
    assert "Zusammenfassung" not in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nAntworte.\n\n[Tools]\n<<TOOL_CALL>>")


def test_render_intermediate_numbering():
    text = proxy.render_intermediate("[Systemanweisung]\nS.", ["a", "b"], 2, 5)

    assert "Teil 2/5" in text
    assert text.index("[Systemanweisung]") < text.index("Teil 2/5")
    assert text.endswith("Antworte nur mit der Zusammenfassung.")


def test_build_hey_jobs_tools_only_in_final(monkeypatch):
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNK_CHARS", 40)
    monkeypatch.setattr(proxy, "HEY_MAX_CHUNKS", 5)
    tools = [{"type": "function", "function": {"name": "write"}}]
    messages = [
        {"role": "user", "content": "erste alte Frage"},
        {"role": "assistant", "content": "alte Antwort"},
        {"role": "user", "content": "noch eine Frage"},
        {"role": "assistant", "content": "noch eine Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = proxy.build_hey_jobs(messages, tools, "auto")

    assert len(intermediates) >= 1
    assert "<<TOOL_CALL>>" not in "\n".join(intermediates)
    assert "<<TOOL_CALL>>" in final_parts["tools"]


def test_hey_session_creates_single_conversation():
    posts = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"conversationId": "cid-9"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            posts.append(url)
            return FakeResponse()

    async def run():
        with unittest.mock.patch.object(proxy.httpx, "AsyncClient", FakeClient):
            async with proxy.hey_session() as (client, cid):
                assert isinstance(client, FakeClient)
                return cid

    assert asyncio.run(run()) == "cid-9"
    assert posts == ["/api/conversations"]


def test_hey_events_skips_garbage_lines():
    lines = [
        "hello without prefix",
        "data: not json at all",
        'data: {"nochoices": true}',
        'data: {"choices": []}',
        'data: {"choices": [{"index": 0, "delta": {"content": "", "suggestions": ["a"]}}]}',
        'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}',
        "data: [DONE]",
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in lines:
                yield line

    class FakeStream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return False

    class FakeClient:
        def stream(self, *args, **kwargs):
            assert kwargs["headers"]["x-conversation-id"] == "cid-1"
            return FakeStream(FakeResponse())

    async def run():
        with unittest.mock.patch.object(proxy, "hey_events", _REAL_HEY_EVENTS):
            return [
                event
                async for event in proxy.hey_events("msg", session=(FakeClient(), "cid-1"))
            ]

    assert asyncio.run(run()) == [("content", "Hi"), ("done", None)]


def test_hey_full_text_falls_back_to_joined_deltas():
    async def deltas_only(message: str, session=None):
        yield ("content", "Hal")
        yield ("content", "lo")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(proxy, "hey_events", deltas_only):
            return await proxy.hey_full_text("msg")

    assert asyncio.run(run()) == "Hallo"


def test_tool_choice_none_hides_tools_end_to_end():
    seen = []

    async def recorder(message: str, session=None):
        seen.append(message)
        for event in FAKE_EVENTS:
            yield event

    with unittest.mock.patch.object(proxy, "hey_events", recorder):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hey",
                "messages": MESSAGES,
                "tools": [{"type": "function", "function": {"name": "write"}}],
                "tool_choice": "none",
            },
        )

    assert response.status_code == 200
    assert len(seen) == 1
    assert "<<TOOL_CALL>>" not in seen[0]
    assert "Sag einfach hallo" in seen[0]
