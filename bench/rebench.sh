#!/usr/bin/env bash
# One-command Phase C re-benchmark: preflight + serve (continuous thermal /
# Xid / exact-GPU watch) + full suite; stops the owned server process tree.
# Usage: MODEL_DIR=... LLAMA_SERVER=... bench/rebench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the staged checkpoint directory}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
export MODEL_DIR LLAMA_SERVER PORT="${PORT:-8199}"
STOPFILE="${STOPFILE:-results/.rebench-stop-$$}"
rm -f "$STOPFILE"
cleanup() {
  [ -f "$STOPFILE" ] || touch "$STOPFILE"
  if [ -n "${WATCHER_PID:-}" ]; then
    wait "$WATCHER_PID" 2>/dev/null || true
  fi
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # Terminate the owned server process tree (server + children), not a
    # blanket pkill; the PID was captured at launch inside this script.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$STOPFILE"
}
trap cleanup EXIT
bench/thermal-guard.sh
bench/preflight.sh
bench/serve.sh > results/serve-rebench.log 2>&1 &
SERVER_PID=$!
# Continuous guard for the whole run: exact GPU count, thermal ceiling,
# Xid scan; exits when the stop file appears or the server dies.
bench/thermal-guard.sh --watch "$STOPFILE" &
WATCHER_PID=$!
deadline=$((SECONDS + 2400))
until curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; do
  kill -0 "$SERVER_PID" || { echo "server died; tail:"; tail -20 results/serve-rebench.log; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "health timeout"; exit 1; }
  sleep 5
done
bash bench/phase63-driver.sh
