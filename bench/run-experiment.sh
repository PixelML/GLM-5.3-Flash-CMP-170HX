#!/usr/bin/env bash
# Run exactly ONE one-factor experiment end to end:
#   preflight -> serve -> /health -> thermal guard -> 1 warmup run
#   -> 3 measured runs (program command, verbatim) -> CSV row -> stop server.
# Safety: preflight refuses foreign GPU workloads; hard thermal stop at
# 80 C core / 85 C mem; the server stopped is always the one started here.
# Usage:  NAME=baseline LLAMA_SERVER=... MODEL_DIR=... [SPLIT=layer THREADS=8 ...] ./run-experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NAME:?experiment name, e.g. baseline}"
: "${LLAMA_SERVER:=/tmp/build-llamacpp/bin/llama-server}"
: "${MODEL_DIR:=/library/models/glm-5.3-flash/unsloth-UD-IQ4_XS}"
PORT="${PORT:-8199}"
BASE="http://127.0.0.1:${PORT}"
PROMPT='Write a 300-word technical summary of MoE routing.'
export LLAMA_SERVER MODEL_DIR PORT

mkdir -p results
LOG="/tmp/serve-${NAME}.log"

echo "== [${NAME}] preflight =="
bench/preflight.sh

# Record the config exactly as it will be served (semicolon-separated, no commas
# in field 1 so experiments.csv stays comma-clean).
ts="${TSPLIT:-auto}"
cfg="split=${SPLIT:-layer};ts=${ts};th=${THREADS:-8};par=${PARALLEL:-1};ctx=${CTX:-16384};fa=${FLASH_ATTN:-server-default};ctk=${CACHE_K:-server-default};ctv=${CACHE_V:-server-default};b=${BATCH:-server-default};ub=${UBATCH:-server-default};mmap=yes$([ "${NO_MMAP:-}" = 1 ] && echo -n no)"

thermal_guard() {
  # Hard stop: 80 C core / 85 C memory. Also fails if a GPU vanishes.
  python3 - <<'PY'
import subprocess, sys
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,temperature.gpu,temperature.memory",
                                   "--format=csv,noheader,nounits"], text=True)
except Exception as e:
    print(f"THERMAL GUARD: nvidia-smi failed ({e}); GPU loss?"); sys.exit(1)
for line in out.strip().splitlines():
    idx, tc, tm = [x.strip() for x in line.split(",")]
    if int(tc) >= 80: print(f"THERMAL STOP: GPU {idx} core {tc} C >= 80"); sys.exit(1)
    if int(tm) >= 85: print(f"THERMAL STOP: GPU {idx} mem {tm} C >= 85"); sys.exit(1)
sys.exit(0)
PY
}

echo "== [${NAME}] serving (log: ${LOG}) =="
bench/serve.sh > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== [${NAME}] waiting /health (mmap load of ~146 GiB can take minutes) =="
ok=""
for i in $(seq 1 360); do
  if curl -sf "$BASE/health" 2>/dev/null | grep -Eq '"(ok|true)"|"status":"ok"'; then ok=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "== [${NAME}] server died; log tail =="; tail -30 "$LOG"; exit 1; fi
  sleep 5
done
[ -n "$ok" ] || { echo "== [${NAME}] /health timeout =="; tail -30 "$LOG"; exit 1; }
echo "== [${NAME}] healthy after ~$((i*5))s =="
grep -m1 -iE 'flash_attn|flash attention' "$LOG" || true

thermal_guard
echo "== [${NAME}] warmup (1 run, discarded) =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 --concurrency 1 \
  --runs 1 --out results/warmup.jsonl | tee "results/raw-${NAME}-warmup.jsonl"

thermal_guard
echo "== [${NAME}] measured: 3 runs =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 --concurrency 1 \
  --runs 3 --out results/speed.jsonl | tee "results/raw-${NAME}.jsonl"

# Median over the 3 measured runs of per-request tok/s (usage-object tokens).
vals=$(python3 - "results/raw-${NAME}.jsonl" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{") and '"concurrency"' in l]
rows = [r for r in rows if r.get("ok")]
assert len(rows) == 3, f"expected 3 ok rows, got {len(rows)}"
med  = statistics.median(r["tok_per_s_per_req"] for r in rows)
tok  = round(statistics.mean(r["completion_tokens"] for r in rows))
ttft = statistics.median(r["ttft_s"] for r in rows) * 1000
p50  = statistics.median(r["itl_ms_p50"] for r in rows)
p95  = statistics.median(r["itl_ms_p95"] for r in rows)
print(round(med,2), tok, round(ttft), p50, p95)
PY
)
read -r med tok ttft p50 p95 <<< "$vals"

# GPU state from the measure.py gpu_after snapshot (last line of raw output).
snapshot=$(grep -F '{"gpu_after"' "results/raw-${NAME}.jsonl" | tail -1 | cut -c14-)
mem=$(echo "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); print(';'.join(r['mem_mib'] for r in d))")
tc=$( echo "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); print(';'.join(r['temp_core_c'] for r in d))")
tm=$( echo "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); print(';'.join(r['temp_mem_c'] for r in d))")
pw=$( echo "$snapshot" | python3 -c "import json,sys; d=json.load(sys.stdin); print(';'.join(r['power_w'] for r in d))")

echo "${cfg},${med},${ttft},${p50},${p95},${mem},${tc},${tm},${pw},measured" >> results/experiments.csv
echo "== [${NAME}] median ${med} tok/s over 3 runs (~${tok} gen tokens/run), ttft ${ttft} ms  (cfg: ${cfg}) =="
echo "== [${NAME}] stopping server (pid ${SERVER_PID}); trap cleans up =="
