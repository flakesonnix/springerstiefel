# springerstiefel

A local OpenAI-compatible gateway that exposes **Hey_ (BILD)** as a model in
[OpenCode](https://opencode.ai):

```text
OpenCode
   │  OpenAI API (/v1/chat/completions)
   ▼
springerstiefel :8787
   │  existing browser session / cookies
   ▼
hey.bild.de
```

Hey_ has no public API (Azure OpenAI under the hood, reachable only through
the website). The proxy therefore talks to the website backend directly –
no browser, no login, plain HTTP with cookies.

> Status: the proxy is **fully wired up**. `POST /api/conversations`
> (returns `conversationId` + `anonid` cookie) → `POST /api/chat`
> (`message` + `source: custom`, `x-conversation-id` header) → Hey_ replies
> with OpenAI-style SSE, translated 1:1 to `/v1/chat/completions`.
> Reverse-engineered via a headless Firefox capture (`data/traffic-auto.jsonl`).

## Requirements

- NixOS with flakes enabled
- No Hey_ account needed (anonymous flow via `anonid` cookie)

## 1. Set up the environment

```bash
nix develop
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

The flake provides Python 3.11, `uv`, and Playwright browsers (Firefox)
patched from nixpkgs. Details:

- `PLAYWRIGHT_BROWSERS_PATH` points at the nixpkgs browsers – a manual
  `playwright install` is **not** needed on NixOS and doesn't work there
  anyway (dynamically linked binaries).
- Playwright's bundled `node` is automatically swapped for the nixpkgs build
  on shell entry (only once `.venv` exists – re-enter with `exit` +
  `nix develop` if needed).
- **Version pinning:** the browser revision must match the Playwright version
  in `pyproject.toml` (`playwright>=1.63,<1.64` ↔ `playwright-driver`
  from nixpkgs). After `nix flake update`, align `pyproject.toml` if needed.

## 2. Capturing the Hey_ requests (done, for reference)

```bash
hey-capture
```

This opens Firefox. Send a test message there (e.g. `Say hello`), then hit
Enter in the terminal.

Filtered traffic goes to `data/traffic.jsonl`, the browser profile to
`data/browser-firefox/` (both git-ignored). The reference capture lives at
`data/traffic-auto.jsonl` (created headless, cookie banner dismissed via
“Accept all”). Result:

```text
POST /api/conversations  {"experienceId": "a5d82531-…"}
  → 201 {"conversationId": "…"} + Set-Cookie: anonid=<JWT>
POST /api/chat  {"message": "…", "source": "custom"}
  Header: x-conversation-id: <conversationId>
  → 200 OpenAI-style SSE (model: gpt-5.4-mini-…, [DONE] at the end)
```

## 3. Start the proxy and plug it into OpenCode

```bash
hey-proxy   # listens on 127.0.0.1:8787
```

Optional environment variables:

```text
HEY_EXPERIENCE_ID       pin a Hey_ experience (default: auto-picked)
HEY_EXPERIENCE_SLUG     preferred experience slug for auto-pick (optional)
HEY_TIMEOUT             HTTP timeout in seconds (default: 120)
HEY_MAX_HISTORY         max history messages in the transcript (default: 30)
HEY_MAX_TOOL_CHARS      max chars per tool result (default: 4000)
HEY_MAX_CHUNK_CHARS     max chars per history chunk (default: 6000)
HEY_MAX_CHUNKS          max history chunks in the queue (default: 5)
HEY_TOOL_RETRY          1/0 – retries on deflection/news drift (default: 1)
```

Without `HEY_EXPERIENCE_ID`, the proxy picks a usable chat experience from
`GET /api/home` automatically (preferred slug first, then any enabled chat
experience), falling back to the last known default.

`opencode.json` (current format per the [provider docs](https://opencode.ai/docs/providers/)):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "hey/hey",
  "provider": {
    "hey": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hey_ BILD",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": { "hey": { "name": "Hey_" } }
    }
  }
}
```

Then verify:

```bash
opencode --model hey/hey
```

or `/models` inside OpenCode. Direct test without OpenCode:

```bash
curl -s http://127.0.0.1:8787/v1/models
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hey","messages":[{"role":"user","content":"Hello"}]}'
```

## 4. Tests

```bash
python3 -m pytest -q
python3 -m mypy src tests
```

Dev loop (reinstall, clean proxy restart, checks, live smoke test):

```bash
nix develop --command bash scripts/reload.sh
```

Benchmarks (verbose phase timing; live sections hit `hey.bild.de`):

```bash
python3 benchmarks/bench.py           # everything
python3 benchmarks/bench.py --quick   # offline only (build + micro)
```

Findings so far: Python overhead is negligible (build <0.1 ms, parsing
sub-millisecond); per Hey_ call the conversation POST costs ~170 ms and
time-to-first-chunk ~1.5 s – model generation dominates, prefill size barely
matters. Chunk intermediates run in parallel (max instead of sum), and Hey_
handles parallel requests cleanly.

- `tests/test_proxy.py` – `/v1/models`, validation (400 without messages),
  non-streaming shape (prefers final text), streaming SSE (`role` →
  `content` → `stop` + `[DONE]`), `last_user_text`, transcript building
  (system/history/tool round-trip/continuation), tool section + `tool_choice`,
  `<<TOOL_CALL>>` parsing, tool calls in non-streaming + streaming,
  session sharing, chunk splitting + parallel summaries, retry triggers.
  The Hey_ backend is mocked via `monkeypatch` – no network needed.
- `tests/test_capture.py` – URL filter (`is_interesting_url`): matches
  chat/API URLs, ignores static assets, case-insensitive.

## Runtime notes

- **Stateless:** every OpenAI request gets a fresh Hey_ conversation.
  All calls within one turn (chunks, retries) share **a single** session
  (one HTTP client + one conversation) – saves one conversation POST
  (~170 ms) per extra call.
  Hey_ accepts only a single message per request, so history, system prompt,
  and tool definitions are embedded as a transcript (`HEY_MAX_HISTORY`,
  default: 30 messages). After a tool round-trip the proxy asks to continue
  instead of repeating the question (no tool loop).
- **Chunk queue for oversized prompts:** if the history doesn't fit one
  request (`HEY_MAX_CHUNK_CHARS` per piece), it is split into a FIFO queue:
  older parts are condensed into 2–3 sentence summaries and embedded in the
  final call (instead of being dropped). The intermediate jobs are
  independent and run **in parallel** (`asyncio.gather`) – the phase costs
  max instead of sum. At most the newest `HEY_MAX_CHUNKS` history chunks go
  through – each chunk is its own Hey_ call, so large histories take time.
- **Tool calls:** `tools`/`tool_choice` are framed as an output protocol
  (`<<TOOL_CALL>>` blocks with JSON plus one example) and translated to
  OpenAI `tool_calls` (`finish_reason: tool_calls`). The tool section sits at
  the end of the message on purpose (recency effect).
  Keep the tone factual: aggressive imperatives (“NEVER”, “MUST”) trip Hey_'s
  guardrail against embedded third-party instructions, and the model then
  refuses the blocks.
  `tool_choice: "none"` hides the section; `"required"`/a specific function
  marks the call as mandatory. Explicit file/action requests
  (“create the file X”, “write the program …”, patterns in
  `ACTION_REQUEST_RES`) escalate to the same mandatory tone, since those
  can only be fulfilled via a tool call.
  With tools, streaming mode buffers (no live token stream); without tools
  the Hey_ stream passes through live.
- **Retry on deflection:** if tools were offered but no tool call comes back
  and the answer looks like a refusal/deflection (patterns in `REFUSAL_RES`),
  it is retried once with a nudge (`HEY_TOOL_RETRY=0` disables it).
- **Retry on news/weather drift:** if the proxy spots BILD markers (`[bild_0_1]`,
  “headlines”, “BILDplus” …) or weather markers (“rain probability”, “gusts”,
  “°C” …) although neither news nor weather was asked for, it retries once
  refocused on the task. Genuine news/weather questions (patterns in
  `NEWS_REQUEST_RES` / `WEATHER_REQUEST_RES`) are exempt.
- **Honest limitation:** Hey_ is a consumer news assistant, not an agent
  model. It sometimes calls tools reliably, sometimes deflects to questions,
  and occasionally drifts into news mode (headlines instead of action). The
  proxy turns that into best-effort agent behavior; for dependable agentic
  coding the backend is only partly suitable – for chat, explanations, and
  writing code as text it is solid.
- Backend errors surface as `502` with a short cause.
- Hey_ replies either with plain `content` or a JSON envelope
  `{"answer": ..., "suggestions": ...}` (string or dict, in `content`
  or `parsed`) – the proxy always extracts `answer`.
- **Reply language:** the prompt language is detected (English/German
  stopwords); English prompts get a reply-in-English directive (including
  code comments). German stays native.
- **Cited links:** Hey_ search answers carry `[bild_0_1]`/`[web_2]` markers
  plus `sources` (index → title/url) in the stream. The proxy resolves them
  and appends a `Quellen:` section with clickable links (image markers fall
  back to their parent article).

## Project layout

```text
├── flake.nix        # NixOS dev shell (Python, uv, Playwright browsers)
├── pyproject.toml   # package + dev extra (pytest)
├── src/springerstiefel/
│   ├── app.py       # OpenAI-compatible gateway (:8787)
│   ├── hey.py       # Hey_ backend client (HeyClient)
│   ├── messages.py  # transcript building + chunk queue
│   ├── tools.py     # tool-call protocol + drift detection
│   ├── config.py    # settings + constants
│   └── capture.py   # Firefox capture → data/traffic.jsonl
├── tests/           # pytest suite
└── data/            # browser profile + capture (local, ignored)
```

## Open items

- [ ] Try a real coding session (`opencode --model hey/hey` on a small
      refactoring, watch the tool loop over several rounds)
- [ ] `hey-capture`: dismiss the cookie banner automatically (like the
      auto-capture does) instead of a manual browser
