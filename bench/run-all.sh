#!/usr/bin/env bash
# One-command baseline: serve + speed sweep + quality eval + GPU snapshot.
# Safety: abort at 80 °C core / 85 °C memory / any Xid. Stop the server when done.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the 5 UD-IQ4_XS shards}"
./preflight.sh   # refuses to start if vLLM occupies GPUs, shards incomplete, etc.
echo "== starting server =="
./serve.sh &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:8199/health" | grep -Eq 'ok|true'; then break; fi
  sleep 2
done
echo "== speed sweep =="
python3 measure.py --base http://127.0.0.1:8199 \
  --prompt "$(cat prompt-1k.txt)" \
  --max-tokens 256 --concurrency 1 --runs 3 --out results/speed.jsonl
echo "== quality eval =="
python3 eval.py --base http://127.0.0.1:8199 --out results/quality.jsonl
echo "== gpu snapshot =="
nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
  --format=csv,noheader | tee results/gpu-final.csv
echo "== done =="
