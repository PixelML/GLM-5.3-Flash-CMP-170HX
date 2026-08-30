#!/usr/bin/env bash
# One-command Phase C re-benchmark: preflight + serve + full suite.
# Usage: MODEL_DIR=... LLAMA_SERVER=... bench/rebench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the staged checkpoint directory}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
export MODEL_DIR LLAMA_SERVER PORT="${PORT:-8199}"
bench/thermal-guard.sh
bench/preflight.sh
bench/serve.sh > results/serve-rebench.log 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
deadline=$((SECONDS + 2400))
until curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; do
  kill -0 "$SERVER_PID"; [ "$SECONDS" -lt "$deadline" ]
  sleep 5
done
bash bench/phase63-driver.sh
