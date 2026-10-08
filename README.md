# springerstiefel

Lokaler OpenAI-kompatibler Gateway, um **Hey_ (BILD)** als Modell in
[OpenCode](https://opencode.ai) zu nutzen:

```text
OpenCode
   │  OpenAI API (/v1/chat/completions)
   ▼
hey-bild-proxy :8787
   │  bestehende Browser-Session / Cookies
   ▼
hey.bild.de
```

Hey_ bietet keine öffentliche API an (Backend: Azure OpenAI, aber nur über
die Website nutzbar). Deshalb spricht der Proxy direkt das Website-Backend
– ohne Browser, ohne Login, nur per HTTP mit Cookies.

> Stand: Der Proxy ist **live verdrahtet**. `POST /api/conversations`
> (liefert `conversationId` + `anonid`-Cookie) → `POST /api/chat`
> (`message` + `source: custom`, Header `x-conversation-id`) → Hey_ antwortet
> mit OpenAI-artigem SSE, das 1:1 nach `/v1/chat/completions` übersetzt wird.
> Ermittelt per headless Firefox-Mitschnitt (`data/traffic-auto.jsonl`).

## Voraussetzungen

- NixOS mit aktivierten Flakes
- Kein Hey_-Account nötig (anonymer Flow per `anonid`-Cookie)

## 1. Umgebung einrichten

```bash
nix develop
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

Die Flake liefert Python 3.11, `uv` sowie die Playwright-Browser (Firefox)
gepatcht aus nixpkgs. Details:

- `PLAYWRIGHT_BROWSERS_PATH` zeigt auf die nixpkgs-Browser – ein manuelles
  `playwright install` ist auf NixOS **nicht** nötig und funktioniert dort
  auch nicht (dynamisch gelinkte Binaries).
- Playwrights gebündeltes `node` wird beim Shell-Einstieg automatisch auf das
  nixpkgs-Node umgebogen (nur falls `.venv` schon existiert – ggf. einmal
  `exit` und erneut `nix develop`).
- **Version-Pinning:** Die Browser-Revision muss zur Playwright-Version in
  `pyproject.toml` passen (`playwright>=1.63,<1.64` ↔ `playwright-driver`
  aus nixpkgs). Nach einem `nix flake update` ggf. in `pyproject.toml`
  angleichen.

## 2. Hey_-Request mitschneiden (bereits erledigt, zum Nachvollziehen)

```bash
hey-capture
```

Es öffnet sich Firefox. Dort eine Testnachricht senden
(z. B. `Sag einfach hallo`), danach Enter im Terminal drücken.

Der gefilterte Traffic landet in `data/traffic.jsonl`, das Browser-Profil in
`data/browser-firefox/` (beides per `.gitignore` ausgenommen). Der
Referenz-Mitschnitt liegt unter `data/traffic-auto.jsonl` (headless erstellt,
Cookie-Banner per „Alle akzeptieren" weggeklickt). Ergebnis:

```text
POST /api/conversations  {"experienceId": "a5d82531-…"}
  → 201 {"conversationId": "…"} + Set-Cookie: anonid=<JWT>
POST /api/chat  {"message": "…", "source": "custom"}
  Header: x-conversation-id: <conversationId>
  → 200 OpenAI-artiges SSE (Modell: gpt-5.4-mini-…, [DONE] am Ende)
```

## 3. Proxy starten und in OpenCode einbinden

```bash
hey-proxy   # hört auf 127.0.0.1:8787
```

Umgebungsvariablen (optional):

```text
HEY_EXPERIENCE_ID       andere Hey_-Experience (Default: s. Mitschnitt)
HEY_TIMEOUT             HTTP-Timeout in Sekunden (Default: 120)
HEY_MAX_HISTORY         max. Verlauf-Messages im Transkript (Default: 30)
HEY_MAX_TOOL_CHARS      max. Zeichen pro Tool-Ergebnis (Default: 4000)
HEY_MAX_CHUNK_CHARS     max. Zeichen pro Verlaufschunk (Default: 6000)
HEY_MAX_CHUNKS          max. Verlaufschunks in der Queue (Default: 5)
HEY_TOOL_RETRY          1/0 – Retries bei Ausweichen/News-Drift (Default: 1)
```

`opencode.json` (aktuelles Format laut [Provider-Doku](https://opencode.ai/docs/providers/)):

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

Dann prüfen:

```bash
opencode --model hey/hey
```

bzw. in OpenCode `/models`. Direkt-Test ohne OpenCode:

```bash
curl -s http://127.0.0.1:8787/v1/models
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hey","messages":[{"role":"user","content":"Hallo"}]}'
```

## 4. Tests

```bash
python3 -m pytest -q
```

- `tests/test_proxy.py` – `/v1/models`, Validierung (400 ohne Messages),
  Non-Streaming-Shape (bevorzugt finalen Text), Streaming-SSE (`role` →
  `content` → `stop` + `[DONE]`), `last_user_text`, Transkript-Bau
  (System/Verlauf/Tool-Roundtrip/Fortsetzung), Tool-Sektion + `tool_choice`,
  `<<TOOL_CALL>>`-Parsing, Tool-Calls in Non-Streaming + Streaming.
  Das Hey_-Backend ist per `monkeypatch` gemockt – Tests brauchen kein Netz.
- `tests/test_capture.py` – URL-Filter (`is_interesting_url`): trifft
  Chat-/API-URLs, ignoriert Static Assets, case-insensitiv.

## Hinweise zum Betrieb

- **Stateless:** Jeder OpenAI-Request bekommt eine frische Hey_-Conversation.
  Alle Calls eines Turns (Chunks, Retries) teilen sich dabei **eine** Session
  (ein HTTP-Client + eine Conversation) – spart je Extra-Call einen
  Conversation-POST (~170 ms).
  Hey_ kennt pro Request nur eine einzelne Message – Verlauf, System-Prompt
  und Tool-Definitionen werden deshalb als Transkript eingebettet
  (`HEY_MAX_HISTORY`, Default: 30 Messages). Nach einem Tool-Roundtrip wird
  zur Fortsetzung aufgefordert statt die Frage zu wiederholen (kein Tool-Loop).
- **Chunk-Queue bei Oversize:** Passt der Verlauf nicht in einen Request
  (`HEY_MAX_CHUNK_CHARS` pro Stück), wird er in eine FIFO-Queue gesplittet:
  ältere Teile werden sequentiell zu 2–3-Satz-Zusammenfassungen verdichtet
  und ins Finale eingebettet (statt ersatzlos zu droppen). Die Zwischenjobs
  sind unabhängig und laufen **parallel** (`asyncio.gather`) – die Phase
  kostet max statt Summe. Es laufen höchstens die neuesten `HEY_MAX_CHUNKS`
  Verlaufschunks durch – jeder Chunk ist ein eigener Hey_-Call, große
  Verläufe kosten also Zeit.
- **Tool-Calls:** `tools`/`tool_choice` werden als Ausgabe-Protokoll
  (`<<TOOL_CALL>>`-Blöcke mit JSON + ein Beispiel) formuliert und zu
  OpenAI-`tool_calls` (`finish_reason: tool_calls`) übersetzt. Die
  Tool-Sektion steht absichtlich am Ende der Message (Recency-Effekt).
  Ton bewusst sachlich halten: aggressive Imperative („NIEMALS", „MUSST")
  triggern Hey_'s Guardrail gegen eingebettete Fremd-Anweisungen, dann
  verweigert das Modell die Blöcke.
  `tool_choice: "none"` blendet die Sektion aus, `"required"`/konkrete
  Funktion markiert Pflicht.
  Mit Tools wird im Streaming-Modus gepuffert (kein Live-Token-Stream),
  ohne Tools läuft der Hey_-Stream live durch.
- **Retry bei Ausweichen:** Kommt trotz angebotener Tools kein Tool-Call und
  sieht die Antwort nach Verweigerung/Rückfrage aus (Muster in `REFUSAL_RES`),
  wird einmalig mit Nudge wiederholt (`HEY_TOOL_RETRY=0` schaltet ab).
- **Retry bei News-Drift:** Erkennt der Proxy BILD-Marker (`[bild_0_1]`,
  „Schlagzeilen", „BILDplus" …) in der Antwort, obwohl keine News gefragt
  waren, wiederholt er einmal mit Refokus auf die Aufgabe. Echte
  News-Fragen (Muster in `NEWS_REQUEST_RES`) sind ausgenommen.
- **Ehrliche Grenze:** Hey_ ist ein News-Verbraucher-Assistent, kein
  Agent-Modell. Es ruft Tools mal zuverlässig auf, weicht mal auf Rückfragen
  aus und driftet gelegentlich in den News-Modus ab (Schlagzeilen statt
  Aktion). Der Proxy macht daraus Best-Effort-Agentenverhalten; für
  zuverlässiges Agentic-Coding ist das Backend nur bedingt geeignet –
  für Chat/Erklären/Code-Schreiben als Text ist es solide.
- Backend-Fehler kommen als `502` mit kurzer Ursache zurück.
- Hey_ antwortet mal mit plain `content`, mal mit JSON-Hülle
  `{"answer": ..., "suggestions": ...}` (String oder Dict, in `content`
  oder `parsed`) – der Proxy extrahiert jeweils `answer`.

## Projektstruktur

```text
├── flake.nix        # NixOS Dev-Shell (Python, uv, Playwright-Browser)
├── pyproject.toml   # Paket + dev-Extra (pytest)
├── capture.py       # Firefox-Mitschnitt → data/traffic.jsonl
├── proxy.py         # OpenAI-kompatibler Gateway (:8787)
├── tests/           # pytest-Suite
└── data/            # Browser-Profil + Mitschnitt (lokal, ignoriert)
```

## Offene Punkte

- [ ] Echten Coding-Einsatz testen (`opencode --model hey/hey` auf ein
      kleines Refactoring loslassen, Tool-Loop über mehrere Runden beobachten)
- [ ] `hey-capture`: Cookie-Banner automatisch wegklicken (wie im
      Auto-Mitschnitt) statt manuellem Browser
