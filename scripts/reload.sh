#!/usr/bin/env bash
# Dev loop for springerstiefel: reinstall, restart hey-proxy cleanly, verify.
#
# Run from the repo root inside nix develop:
#   nix develop --command bash scripts/reload.sh
#
# - kills the previous proxy via PID file (no pkill footguns)
# - reinstalls the editable package
# - starts hey-proxy detached, waits for :8787
# - runs pytest + mypy + a live smoke request against hey.bild.de
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=/tmp/springerstiefel-proxy.pid
LOG=/tmp/springerstiefel-proxy.log

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "stoppe alten Proxy (pid $OLD) ..."
    kill "$OLD"
    for _ in $(seq 1 15); do
      kill -0 "$OLD" 2>/dev/null || break
      sleep 1
    done
  fi
  rm -f "$PIDFILE"
fi

# Falls ein fremder Prozess den Port hält (kein PID-File, z.B. manuell
# gestartet): anhand der PID vom Listener beenden – nie per pkill-Muster,
# das trifft sonst die eigene Shell.
STALE=$(ss -tlnp 2>/dev/null | grep 8787 | grep -oP 'pid=\K[0-9]+' | head -n 1 || true)
if [ -n "${STALE:-}" ] && [ "$STALE" != "$$" ]; then
  echo "fremder Listener auf :8787 (pid $STALE), beende ..."
  kill "$STALE"
  for _ in $(seq 1 15); do
    ss -tln 2>/dev/null | grep -q 8787 || break
    sleep 1
  done
fi

echo "installiere Paket ..."
uv pip install -q -e ".[dev]"

echo "starte hey-proxy ..."
setsid nohup hey-proxy </dev/null >"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "proxy pid $(cat "$PIDFILE"), warte auf :8787 ..."
for _ in $(seq 1 30); do
  ss -tln 2>/dev/null | grep -q 8787 && break
  sleep 2
done
ss -tln | grep -q 8787 || {
  echo "FEHLER: kein Listener auf :8787"
  tail -n 20 "$LOG"
  exit 1
}

echo "--- pytest ---"
python3 -m pytest -q 2>&1 | tail -n 1
echo "--- mypy ---"
python3 -m mypy proxy.py capture.py benchmarks/bench.py 2>&1 | tail -n 1
echo "--- smoke (live) ---"
curl -s --max-time 90 -X POST http://127.0.0.1:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hey","messages":[{"role":"user","content":"Sag Hallo."}]}' \
  | head -c 300
echo
echo "OK – Proxy läuft (pid $(cat "$PIDFILE"), log $LOG)"
