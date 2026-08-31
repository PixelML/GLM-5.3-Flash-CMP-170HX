#!/usr/bin/env bash
# One-command baseline: preflight -> serve (bounded health wait) -> thermal
# guard -> speed sweep -> quality eval -> GPU snapshot -> manifest -> stop.
# The server stopped is always the one started here; there is no auto-trigger.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NAME:=runall}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
: "${MODEL_DIR:?set MODEL_DIR to the staged checkpoint directory}"
PORT="${PORT:-8199}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-1800}"
BASE="http://127.0.0.1:${PORT}"
PROMPT="$(cat bench/prompt-1k.txt)"
export LLAMA_SERVER MODEL_DIR PORT

mkdir -p results
LOG="${SERVE_LOG:-results/serve-${NAME}.log}"

echo "== thermal guard (pre-launch) =="
bench/thermal-guard.sh
echo "== preflight =="
bench/preflight.sh

echo "== serving (log: ${LOG}) =="
# setsid gives serve.sh its own process group; SERVER_PID is that group's
# PGID, so cleanup can always kill the whole owned tree (llama-server + tee).
setsid bash bench/serve.sh > "$LOG" 2>&1 &
SERVER_PID=$!
STOPFILE="${STOPFILE:-results/.thermal-stop-$$}"
rm -f "$STOPFILE"
WATCHER_PID=""
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  touch "$STOPFILE"
  [ -z "$WATCHER_PID" ] || wait "$WATCHER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$STOPFILE"
}
trap cleanup EXIT
WATCH_PID="$SERVER_PID" WATCH_PGID="$SERVER_PID" bench/thermal-guard.sh --watch "$STOPFILE" &
WATCHER_PID=$!
cleanup() {
  touch "$STOPFILE"
  wait "$WATCHER_PID" 2>/dev/null || true
  # UNCONDITIONAL owned-group kill: also works if the watcher killed the
  # group leader first; survivors must not be orphaned.
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$STOPFILE"
}
trap cleanup EXIT

deadline=$(( SECONDS + HEALTH_TIMEOUT_SECS ))
echo "== waiting /health (timeout ${HEALTH_TIMEOUT_SECS}s) =="
ok=""
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -sf "$BASE/health" 2>/dev/null | grep -Eq '"(ok|true)"|"status":"ok"'; then ok=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "== server died; log tail =="; tail -30 "$LOG"; exit 1; fi
  if ! kill -0 "$WATCHER_PID" 2>/dev/null; then echo '== safety watch exited early; failing run =='; exit 1; fi
  sleep 5
done
if [ -z "$ok" ]; then echo "== /health timeout after ${HEALTH_TIMEOUT_SECS}s =="; tail -30 "$LOG"; exit 1; fi

bench/thermal-guard.sh
echo "== warmup (1 run, discarded) =="
kill -0 "$WATCHER_PID" 2>/dev/null || { echo '== safety watch exited early =='; exit 1; }
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 --concurrency 1 \
  --runs 1 --out results/warmup.jsonl | tee "results/raw-${NAME}-warmup.jsonl"
bench/thermal-guard.sh

echo "== speed sweep =="
kill -0 "$WATCHER_PID" 2>/dev/null || { echo '== safety watch exited early =='; exit 1; }
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" \
  --max-tokens 256 --concurrency "${CONCURRENCY:-1}" --runs "${RUNS:-3}" \
  --out results/speed.jsonl | tee "results/raw-${NAME}.jsonl"
bench/thermal-guard.sh

echo "== quality eval =="
kill -0 "$WATCHER_PID" 2>/dev/null || { echo '== safety watch exited early =='; exit 1; }
python3 bench/eval.py --base "$BASE" --out results/quality.jsonl | tee "results/quality-${NAME}.log"

echo "== gpu snapshot =="
nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
  --format=csv,noheader | tee results/gpu-final.csv

echo "== manifest =="
python3 - <<'PY'
import json
import os
import subprocess
import time

rows = []
path = "results/speed.jsonl"
if os.path.exists(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip().startswith("{")]
manifest = {
    "generated_unix": int(time.time()),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(),
    "name": os.environ.get("NAME", "runall"),
    "model_file": os.environ.get("MODEL_FILE", "GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf"),
    "serve_cfg": {k: os.environ.get(k, "") for k in (
        "PORT", "CTX", "SPLIT", "TSPLIT", "PARALLEL", "THREADS",
        "FLASH_ATTN", "CACHE_K", "CACHE_V", "BATCH", "UBATCH", "NO_MMAP")},
    "speed_rows": len(rows),
    "quality_log": "results/quality-%s.log" % os.environ.get("NAME", "runall"),
}
with open("results/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(json.dumps(manifest))
PY
echo "== done =="
