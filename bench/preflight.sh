#!/usr/bin/env bash
# Read-only preflight for the llama.cpp benchmark loop. Starts nothing.
# Exit 0 = safe to serve; exit 1 = blocked (reasons printed). Never touches
# processes, GPUs, or storage outside checks.
set -u
fail=0
: "${MODEL_DIR:?set MODEL_DIR to the staged checkpoint directory}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
EXPECT_GPUS="${EXPECT_GPUS:-4}"
EXPECT_SHARDS="${EXPECT_SHARDS:-5}"
EXPECT_BYTES="${EXPECT_BYTES:-156822111075}"

echo "== preflight =="

# 1. nvidia-smi must succeed; exact GPU count; no foreign GPU workload.
gpu_q="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>&1)"
if [ $? -ne 0 ]; then
  echo "FAIL: nvidia-smi error: $gpu_q"
  fail=1
else
  gpus=$(echo "$gpu_q" | grep -c . || true)
  if [ "$gpus" -ne "$EXPECT_GPUS" ]; then
    echo "FAIL: $gpus visible GPUs != expected $EXPECT_GPUS"
    fail=1
  else
    echo "ok: $gpus GPUs visible"
  fi
fi
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  echo "FAIL: a vLLM server is running; per safety rules: stop, report, wait."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fail=1
else
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${apps:-0}" -gt 0 ]; then
    echo "FAIL: $apps compute process(es) hold GPU memory; stop, report, wait."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    fail=1
  else
    echo "ok: no compute processes on any GPU"
  fi
fi

# 2. Enough free VRAM on every card (weights ~36.5 GiB/card at 4-way split + ctx/KV).
need_mib=39000
while read -r idx free; do
  if [ "$free" -lt "$need_mib" ]; then
    echo "FAIL: GPU $idx has ${free} MiB free (< ${need_mib} MiB needed)"
    fail=1
  fi
# Guard: preflight aborts when nvidia-smi errored, so only rows with a valid
# numeric index exist here; any stray error text has no comma+index and is
# skipped by the awk filter.
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F", " '$1 ~ /^[0-9]+$/ {print}')

# 3. Model staging complete: shard count, per-shard size, exact byte total, and
#    full SHA256 verification (opt-out: PREFLIGHT_HASH=0).
count=$(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null | wc -l)
if [ "$count" -ne "$EXPECT_SHARDS" ]; then
  echo "FAIL: $count/$EXPECT_SHARDS GGUF shards staged in $MODEL_DIR"
  fail=1
else
  small=$(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -size -1G | wc -l)
  if [ "$small" -ne 0 ]; then
    echo "FAIL: $small shard(s) under 1 GiB (staging incomplete?)"
    fail=1
  else
    echo "ok: $count/$EXPECT_SHARDS shards staged, each >1 GiB"
  fi
  bytes=$(stat -c %s "$MODEL_DIR"/*.gguf | awk '{t+=$1} END{print t+0}')
  if [ "$bytes" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: staged bytes $bytes != expected $EXPECT_BYTES (download still running or corrupt)"
    fail=1
  else
    echo "ok: staged byte total is exact ($bytes)"
  fi
  if [ ! -f "$MODEL_DIR/SHA256SUMS" ]; then
    echo "FAIL: SHA256SUMS not present in $MODEL_DIR (staging not marked complete)"
    fail=1
  elif [ "${PREFLIGHT_HASH:-1}" = "0" ]; then
    echo "WARN: PREFLIGHT_HASH=0; skipping sha256 verification (size checks only)"
  else
    if (cd "$MODEL_DIR" && sha256sum -c SHA256SUMS --quiet >/dev/null 2>&1); then
      echo "ok: sha256 verification passed"
    else
      echo "FAIL: sha256 verification failed against SHA256SUMS"
      fail=1
    fi
  fi
fi

# 4. Runtime binary present and executable.
if [ -x "$LLAMA_SERVER" ]; then
  echo "ok: $LLAMA_SERVER"
else
  echo "FAIL: $LLAMA_SERVER missing or not executable"
  fail=1
fi

# 5. Default benchmark port free.
if ss -ltn 2>/dev/null | grep -q ":8199 "; then
  echo "FAIL: port 8199 already listening"
  fail=1
else
  echo "ok: port 8199 free"
fi

if [ "$fail" -ne 0 ]; then
  echo "== preflight: BLOCKED =="
  exit 1
fi
echo "== preflight: PASS =="
