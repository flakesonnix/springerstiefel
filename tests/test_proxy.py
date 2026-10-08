import asyncio
import json
import time
import unittest.mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from springerstiefel import config as config_module
from springerstiefel import hey as hey_module
from springerstiefel import language as language_module
from springerstiefel import messages as messages_module
from springerstiefel import tools as tools_module
from springerstiefel.app import app
from springerstiefel.hey import HeyClient

client = TestClient(app)

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


async def fake_hey_events(self, message: str, session=None):
    assert "Sag einfach hallo" in message
    for event in FAKE_EVENTS:
        yield event


@pytest.fixture(autouse=True)
def mock_hey_backend(monkeypatch):
    monkeypatch.setattr(HeyClient, "events", fake_hey_events)


# Real implementation (import time, before fixture patches) for tests
# covering the session flow down to _chat_stream.
_REAL_HEY_EVENTS = HeyClient.events


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

    assert messages_module.last_user_text(messages) == "zweite Frage"


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

    assert messages_module.last_user_text(messages) == "Hallo Welt"


def test_build_hey_message_embeds_system_and_history():
    messages = [
        {"role": "system", "content": "Sei knapp."},
        {"role": "user", "content": "Was ist 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Und 3+3?"},
    ]

    text = messages_module.build_hey_message(messages)

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

    text = messages_module.build_hey_message(messages)

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

    text = messages_module.build_hey_message(messages)

    # Frage wird NICHT wiederholt (sonst Tool-Loop) …
    assert text.count("Wie spät ist es?") == 1
    # … continuation with the result at the end instead.
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

    text = messages_module.build_hey_message(
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

    text = messages_module.build_hey_message(
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

    text = messages_module.build_hey_message(
        [{"role": "user", "content": "Hallo"}], tools, tool_choice="required"
    )

    assert "per Aktion zu lösen" in text


def test_build_hey_message_escalates_action_requests():
    tools = [
        {
            "type": "function",
            "function": {"name": "write"},
        }
    ]

    text = messages_module.build_hey_message(
        [{"role": "user", "content": "Erstelle die Datei calc.rs."}], tools
    )

    assert "per Aktion zu lösen" in text

    calm = messages_module.build_hey_message(
        [{"role": "user", "content": "Was ist Rust?"}], tools
    )

    assert "per Aktion zu lösen" not in calm


def test_is_action_request():
    assert tools_module.is_action_request("Erstelle die Datei calc.rs.")
    assert tools_module.is_action_request("Write a program that adds numbers.")
    assert not tools_module.is_action_request("Was ist Rust?")
    assert not tools_module.is_action_request("Erkläre mir Ownership.")


def test_extract_tool_calls_parses_blocks():
    answer = (
        'Gerne!\n<<TOOL_CALL>>\n{"name": "read_file", '
        '"arguments": {"path": "foo.txt"}}\n<<END_TOOL_CALL>>'
    )

    clean, calls = tools_module.extract_tool_calls(answer)

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

    clean, calls = tools_module.extract_tool_calls(answer)

    assert calls == []
    assert clean == answer


def test_extract_message_text_plain_content():
    assert hey_module.extract_message_text({"content": "Hallo!"}) == "Hallo!"


def test_extract_message_text_json_envelope_string():
    message = {
        "content": json.dumps(
            {"answer": "Hier der Code.", "suggestions": ["Weiter"]}
        )
    }

    assert hey_module.extract_message_text(message) == "Hier der Code."


def test_extract_message_text_json_envelope_dict():
    message = {"content": {"answer": "Antwort", "suggestions": []}}

    assert hey_module.extract_message_text(message) == "Antwort"


def test_extract_message_text_parsed_fallback():
    message = {"content": "", "parsed": {"answer": "Fallback"}}

    assert hey_module.extract_message_text(message) == "Fallback"


def test_extract_message_text_empty():
    assert hey_module.extract_message_text({}) == ""
    assert hey_module.extract_message_text({"content": ""}) == ""


def test_is_deflection_detects_refusals():
    assert tools_module.is_deflection("Wobei soll ich im Projekt helfen?")
    assert tools_module.is_deflection("Wobei kann ich Ihnen helfen?")
    assert tools_module.is_deflection("Das kann ich hier nicht ausführen.")
    assert tools_module.is_deflection("Nennen Sie bitte Pfad und Inhalt.")
    assert tools_module.is_deflection("Wie kann ich Ihnen helfen?")
    assert tools_module.is_deflection("Ich kann in diesem Schritt keine Datei anlegen.")
    assert tools_module.is_deflection("Meinen Sie einen Rechner oder das Spiel Rust?")
    assert tools_module.is_deflection("Soll ich eine flake.nix hinzufügen?")
    assert tools_module.is_deflection("I can add a minimal flake.nix for this.")
    assert tools_module.is_deflection("Would you like me to create the file?")
    assert tools_module.is_deflection("Should I run the tests first?")


def test_is_deflection_accepts_normal_text():
    assert not tools_module.is_deflection("Hier ist der Code:\n```rust\nfn main() {}```")
    assert not tools_module.is_deflection("Es ist 14:37 Uhr.")


def test_hey_answer_retries_deflection_once():
    calls = {"n": 0}

    async def flaky_events(self, message: str, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("final", "Wobei soll ich im Projekt helfen?")
        else:
            assert tools_module.TOOL_RETRY_NUDGE in message
            yield ("final", '<<TOOL_CALL>>\n{"name": "write", "arguments": {}}\n<<END_TOOL_CALL>>')
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(HeyClient, "events", flaky_events):
            return await HeyClient().answer("Mach X.", [{"type": "function"}])

    answer, sources = asyncio.run(run())

    assert calls["n"] == 2
    assert "TOOL_CALL" in answer
    assert sources == {}


def test_hey_answer_no_retry_without_tools():
    calls = {"n": 0}

    async def events_no_retry(self, message: str, session=None):
        calls["n"] += 1
        yield ("final", "Wobei soll ich helfen?")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(HeyClient, "events", events_no_retry):
            return await HeyClient().answer("Hallo.", None)

    answer, sources = asyncio.run(run())

    assert calls["n"] == 1
    assert answer == "Wobei soll ich helfen?"
    assert sources == {}


def test_tool_results_are_truncated(monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_tool_chars", 20)
    messages = [
        {"role": "user", "content": "Frage"},
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 100},
        {"role": "user", "content": "Und?"},
    ]

    text = messages_module.build_hey_message(messages)

    assert ("x" * 100) not in text
    assert "[… Ausgabe gekürzt …]" in text
    assert text.endswith("Aktuelle Anweisung:\nUnd?")


def test_split_lines_keeps_wholes_lines_in_budget():
    lines = ["aaa", "bbb", "ccccc"]

    chunks = messages_module.split_lines(lines, 8)

    assert chunks == [["aaa", "bbb"], ["ccccc"]]


def test_split_lines_empty():
    assert messages_module.split_lines([], 100) == []


def test_build_hey_jobs_single_for_small_history():
    messages = [
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = messages_module.build_hey_jobs(messages, None, "auto")

    assert intermediates == []
    assert final_parts is None


def test_build_hey_jobs_splits_big_history(monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_chunk_chars", 40)
    monkeypatch.setattr(config_module.settings, "max_chunks", 5)
    messages = [
        {"role": "user", "content": "erste alte Frage"},
        {"role": "assistant", "content": "alte Antwort"},
        {"role": "user", "content": "noch eine Frage"},
        {"role": "assistant", "content": "noch eine Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = messages_module.build_hey_jobs(messages, None, "auto")

    # Nothing is lost (instead of dropping like before) …
    assert len(intermediates) >= 1
    assert "Teil 1/" in intermediates[0]
    assert "Fasse in 2–3 Sätzen zusammen" in intermediates[0]
    # … aktuelle Anweisung nur im Finale.
    assert "Frage neu" not in "\n".join(intermediates)
    assert final_parts["current"] == "Frage neu"

    rendered = messages_module.render_final(final_parts, ["Zwischenfazit"])
    assert "Zwischenfazit" in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nFrage neu")


def test_build_hey_jobs_caps_chunk_count(monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_chunk_chars", 10)
    monkeypatch.setattr(config_module.settings, "max_chunks", 2)
    messages = [
        {"role": "user", "content": f"Frage {i} mit viel Text dahinter"}
        for i in range(10)
    ]
    messages.append({"role": "user", "content": "Frage neu"})

    intermediates, final_parts = messages_module.build_hey_jobs(messages, None, "auto")

    assert len(intermediates) + 1 <= 2
    assert final_parts["current"] == "Frage neu"


def test_chunked_flow_chains_summaries(monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_chunk_chars", 40)
    monkeypatch.setattr(config_module.settings, "max_chunks", 5)
    seen = []

    async def fake_full_text(self, message: str, session=None):
        seen.append(message)
        return f"Summary {len(seen)}"

    async def run():
        hey_client = HeyClient()
        with unittest.mock.patch.object(HeyClient, "full_text", fake_full_text):
            with unittest.mock.patch.object(
                HeyClient, "events", fake_hey_events
            ):
                intermediates, final_parts = messages_module.build_hey_jobs(
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
                    summaries.append(await hey_client.full_text(job))
                return (
                    intermediates,
                    messages_module.render_final(final_parts, summaries),
                )

    intermediates, rendered = asyncio.run(run())

    assert len(seen) == len(intermediates)
    assert "Summary 1" in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nFrage neu")


def test_is_news_drift_detects_headline_dump():
    dump = "## Aktuelle BILD-Schlagzeilen von heute\nPolitik: ... [bild_0_1]"

    assert tools_module.is_news_drift(dump, "Erstelle die Datei hello.rs.")
    assert not tools_module.is_news_drift("Hier ist der Code.", "Erstelle hello.rs.")


def test_is_news_drift_skips_genuine_news_requests():
    dump = "Schlagzeilen: ... [bild_0_1]"

    assert not tools_module.is_news_drift(dump, "Was sind die Nachrichten heute?")


def test_hey_answer_refocuses_news_drift():
    calls = {"n": 0}

    async def events_news(self, message: str, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("final", "Schlagzeilen des Tages [bild_0_1]")
        else:
            assert "Bearbeite nur diese Aufgabe" in message
            assert "hello.rs" in message
            yield ("final", "Erledigt.")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(HeyClient, "events", events_news):
            return await HeyClient().answer(
                "Mach X.", [{"type": "function"}], "Erstelle hello.rs."
            )

    answer, _ = asyncio.run(run())

    assert calls["n"] == 2
    assert answer == "Erledigt."


def test_chat_completions_returns_tool_calls():
    async def tool_events_returns(self, message: str, session=None):
        yield (
            "final",
            'Bitte sehr:\n<<TOOL_CALL>>\n{"name": "get_time", "arguments": {}}\n'
            "<<END_TOOL_CALL>>",
        )
        yield ("done", None)

    with unittest.mock.patch.object(HeyClient, "events", tool_events_returns):
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
    async def tool_events_stream(self, message: str, session=None):
        yield ("content", "Moment…")
        yield ("final", '<<TOOL_CALL>>\n{"name": "get_time", "arguments": {}}\n<<END_TOOL_CALL>>')
        yield ("done", None)

    with unittest.mock.patch.object(HeyClient, "events", tool_events_stream):
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
    async def slow_full_text(self, message: str, session=None):
        await asyncio.sleep(0.4)
        return f"Summary of {message}"

    async def run():
        with unittest.mock.patch.object(HeyClient, "full_text", slow_full_text):
            start = time.perf_counter()
            result = await HeyClient().summarize(["job-a", "job-b"])
            elapsed = time.perf_counter() - start
            return result, elapsed

    result, elapsed = asyncio.run(run())

    assert result == ["Summary of job-a", "Summary of job-b"]
    # Sequential would be 0.8s – parallel well below that.
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

    async def fake_inner_events(self, client, conversation_id, message):
        chats.append((client, conversation_id))
        if len(chats) == 1:
            yield ("final", "Wobei soll ich helfen?")
        else:
            assert tools_module.TOOL_RETRY_NUDGE in message
            yield ("final", '<<TOOL_CALL>>\n{"name": "write", "arguments": {}}\n<<END_TOOL_CALL>>')

    with unittest.mock.patch.object(HeyClient, "session", lambda self: FakeSession()):
        with unittest.mock.patch.object(HeyClient, "_chat_stream", fake_inner_events):
            with unittest.mock.patch.object(HeyClient, "events", _REAL_HEY_EVENTS):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hey",
                        "messages": MESSAGES,
                        "tools": [{"type": "function", "function": {"name": "write"}}],
                    },
                )

    assert response.status_code == 200
    # One session/conversation for the turn, two chat calls inside.
    assert sessions["entries"] == 1
    assert chats == [("fake-client", "cid-1"), ("fake-client", "cid-1")]
    (choice,) = response.json()["choices"]
    assert choice["finish_reason"] == "tool_calls"


def test_summarize_intermediates_single_and_empty():
    async def run():
        hey_client = HeyClient()
        assert await hey_client.summarize([]) == []
        with unittest.mock.patch.object(
            HeyClient, "full_text",
            lambda self, m, session=None: asyncio.sleep(0, result=f"S({m})"),
        ):
            assert await hey_client.summarize(["solo"]) == ["S(solo)"]

    asyncio.run(run())


def test_message_text_handles_odd_content():
    assert messages_module.message_text({"role": "user"}) == ""
    assert messages_module.message_text({"role": "user", "content": None}) == ""
    assert messages_module.message_text({"role": "user", "content": 123}) == ""
    assert messages_module.message_text({"role": "user", "content": [{"type": "x"}]}) == ""


def test_last_user_index_raises_without_user():
    with pytest.raises(HTTPException) as exc:
        messages_module.last_user_index([{"role": "system", "content": "x"}])

    assert exc.value.status_code == 400


def test_news_refocus_nudge_truncates_long_task():
    nudge = tools_module.news_refocus_nudge("x" * 1000)

    assert nudge.endswith("…]")
    assert "x" * 1000 not in nudge
    assert "Bearbeite nur diese Aufgabe" in nudge


def test_render_final_variants():
    bare = {"system": None, "lines": [], "current": "Hi", "tools": None}

    assert messages_module.render_final(bare, []) == "Aktuelle Anweisung:\nHi"

    full = {
        "system": "[Systemanweisung]\nSei knapp.",
        "lines": ["Benutzer: Frage"],
        "current": "Antworte.",
        "tools": "[Tools]\n<<TOOL_CALL>>",
    }
    rendered = messages_module.render_final(full, [])

    assert "[Systemanweisung]" in rendered
    assert "Benutzer: Frage" in rendered
    assert "Zusammenfassung" not in rendered
    assert rendered.endswith("Aktuelle Anweisung:\nAntworte.\n\n[Tools]\n<<TOOL_CALL>>")


def test_render_intermediate_numbering():
    text = messages_module.render_intermediate("[Systemanweisung]\nS.", ["a", "b"], 2, 5)

    assert "Teil 2/5" in text
    assert text.index("[Systemanweisung]") < text.index("Teil 2/5")
    assert text.endswith("Antworte nur mit der Zusammenfassung.")


def test_build_hey_jobs_tools_only_in_final(monkeypatch):
    monkeypatch.setattr(config_module.settings, "max_chunk_chars", 40)
    monkeypatch.setattr(config_module.settings, "max_chunks", 5)
    tools = [{"type": "function", "function": {"name": "write"}}]
    messages = [
        {"role": "user", "content": "erste alte Frage"},
        {"role": "assistant", "content": "alte Antwort"},
        {"role": "user", "content": "noch eine Frage"},
        {"role": "assistant", "content": "noch eine Antwort"},
        {"role": "user", "content": "Frage neu"},
    ]

    intermediates, final_parts = messages_module.build_hey_jobs(messages, tools, "auto")

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
        with unittest.mock.patch.object(hey_module.httpx, "AsyncClient", FakeClient):
            async with HeyClient().session() as (client, cid):
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
        with unittest.mock.patch.object(HeyClient, "events", _REAL_HEY_EVENTS):
            return [
                event
                async for event in HeyClient().events("msg", session=(FakeClient(), "cid-1"))
            ]

    assert asyncio.run(run()) == [("content", "Hi"), ("done", None)]


def test_hey_full_text_falls_back_to_joined_deltas():
    async def deltas_only(self, message: str, session=None):
        yield ("content", "Hal")
        yield ("content", "lo")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(HeyClient, "events", deltas_only):
            return await HeyClient().full_text("msg")

    assert asyncio.run(run()) == "Hallo"


def test_tool_choice_none_hides_tools_end_to_end():
    seen = []

    async def recorder(self, message: str, session=None):
        seen.append(message)
        for event in FAKE_EVENTS:
            yield event

    with unittest.mock.patch.object(HeyClient, "events", recorder):
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


def test_is_weather_drift_detects_dump():
    dump = "Aktuell in Berlin: 14,5 °C, Regenwahrscheinlichkeit 68 %."

    assert tools_module.is_weather_drift(dump, "Erstelle die Datei hello.rs.")
    assert not tools_module.is_weather_drift("Hier ist der Code.", "Erstelle hello.rs.")


def test_is_weather_drift_skips_genuine_weather_requests():
    dump = "Aktuell in Berlin: 14,5 °C."

    assert not tools_module.is_weather_drift(dump, "Wie ist das Wetter heute?")
    assert not tools_module.is_weather_drift(dump, "Schreibe eine Wetter-App.")


def test_drift_kind_mapping():
    assert tools_module.drift_kind("Schlagzeilen [bild_0_1]", "Mach X.") == "news"
    assert tools_module.drift_kind("14 °C, Böen", "Mach X.") == "weather"
    assert tools_module.drift_kind("Hier der Code.", "Mach X.") is None
    assert tools_module.drift_kind("Schlagzeilen!", "Was gibt es Neues?") is None


def test_hey_answer_refocuses_weather_drift():
    calls = {"n": 0}

    async def events_weather(self, message: str, session=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield ("final", "Aktuell in Berlin: 14,5 °C, Regen.")
        else:
            assert "Wetter-Themen" in message
            assert "hello.rs" in message
            yield ("final", "Erledigt.")
        yield ("done", None)

    async def run():
        with unittest.mock.patch.object(HeyClient, "events", events_weather):
            return await HeyClient().answer(
                "Mach X.", [{"type": "function"}], "Erstelle hello.rs."
            )

    answer, _ = asyncio.run(run())

    assert calls["n"] == 2
    assert answer == "Erledigt."


def test_detect_language():
    assert language_module.detect_language("Write a calculator in rust.") == "en"
    assert language_module.detect_language("Erstelle die Datei calc.rs.") == "de"
    assert language_module.detect_language("") == "de"
    assert language_module.detect_language("fn main() {}") == "de"
    assert language_module.detect_language("Hallo Welt") == "de"


def test_build_hey_message_adds_directive_for_english():
    text = messages_module.build_hey_message(
        [{"role": "user", "content": "Write a calculator in rust."}]
    )

    assert "Reply in English" in text
    assert text.rstrip().endswith("Reply in English, including code comments.")


def test_build_hey_message_no_directive_for_german():
    text = messages_module.build_hey_message(
        [{"role": "user", "content": "Sag einfach hallo."}]
    )

    assert "Reply in English" not in text


def test_render_intermediate_english_instruction():
    text = messages_module.render_intermediate(None, ["a"], 1, 2, lang="en")

    assert "Summarize in 2-3 sentences" in text
    assert "Fasse in" not in text


def test_extract_sources_from_delta():
    delta = {
        "sources": [
            {"index": "bild_0_1", "title": "Titel", "url": "https://x.test/1"},
            {"index": "broken"},
            "nonsense",
        ]
    }

    assert hey_module.extract_sources(delta) == {
        "bild_0_1": {"title": "Titel", "url": "https://x.test/1"}
    }
    assert hey_module.extract_sources({}) == {}


def test_resolve_sources_order_and_fallback():
    sources = {
        "bild_0_0": {"title": "Erster", "url": "https://x.test/0"},
        "bild_0_1": {"title": "Zweiter", "url": "https://x.test/1"},
    }
    text = "Siehe [bild_0_1] und [bild_0_1] sowie [bild_0_0:image_0] und [web_9]."

    resolved = hey_module.resolve_sources(text, sources)

    assert resolved == [
        ("bild_0_1", "Zweiter", "https://x.test/1"),
        ("bild_0_0:image_0", "Erster", "https://x.test/0"),
    ]


def test_append_sources_only_when_resolved():
    assert hey_module.append_sources("Hallo!", {}) == "Hallo!"
    assert hey_module.append_sources("Siehe [web_9].", {}) == "Siehe [web_9]."

    text = hey_module.append_sources(
        "Siehe [bild_0_1].",
        {"bild_0_1": {"title": "Titel", "url": "https://x.test/1"}},
    )

    assert "Quellen:" in text
    assert "[bild_0_1] [Titel](https://x.test/1)" in text


def test_chat_completions_appends_quellen():
    async def sourced_events(self, message: str, session=None):
        yield ("sources", {"bild_0_1": {"title": "Titel", "url": "https://x.test/1"}})
        yield ("final", "Siehe [bild_0_1].")
        yield ("done", None)

    with unittest.mock.patch.object(HeyClient, "events", sourced_events):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "hey", "messages": MESSAGES},
        )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "[bild_0_1] [Titel](https://x.test/1)" in content


def test_pick_experience_prefers_chat_and_slug(monkeypatch):
    items = [
        {"experienceId": "id-disabled", "slug": "x", "isTextInputDisabled": True},
        {"experienceId": "id-chat", "slug": "toll", "interactionMode": "chat"},
        {"experienceId": "id-other", "slug": "y"},
    ]

    assert hey_module.pick_experience(items) == "id-chat"

    monkeypatch.setattr(config_module.settings, "experience_slug", "y")
    assert hey_module.pick_experience(items) == "id-other"

    assert hey_module.pick_experience([]) == config_module.DEFAULT_EXPERIENCE_ID
    assert hey_module.pick_experience([{"slug": "no-id"}]) == (
        config_module.DEFAULT_EXPERIENCE_ID
    )
