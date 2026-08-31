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
    # Upstream-fact check: a tiny leading shard is legitimate when the
    # exact byte total and per-file sha256 match the pinned manifest.
    first_shard="$(find "$MODEL_DIR" -maxdepth 1 -name "*-00001-of-*.gguf" -printf "%f\n" | head -1)"
    first_size="$(stat -c %s "$MODEL_DIR/$first_shard" 2>/dev/null || echo 0)"
    if [ "$first_size" -lt 1048576 ]; then
      echo "FAIL: shard 1 is only $first_size bytes (staging stub?)"
      fail=1
    else
      echo "ok: tiny shard allowed ($first_shard = $first_size bytes; byte total + sha256 remain authoritative)"
    fi
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
  elif [ "${PREFLIGHT_HASH=1}" = "0" ]; then
    echo "WARN: PREFLIGHT_HASH=0; skipping sha256 verification (size checks only)"
  else
    if (cd "$MODEL_DIR" && sha256sum -c SHA256SUMS --quiet >/dev/null 2>&1); then
      echo "ok: sha256 verification passed"
    else
      echo "FAIL: sha256 verification failed against SHA256SUMS"
      fail=1
    fi
  fi
  # Upstream-fact cross-check: the staged manifest must match the committed
  # expected-hash pin, or a self-consistent but wrong SHA256SUMS could pass.
  if [ -f "$MODEL_DIR/SHA256SUMS" ]; then
    if ! diff -q "$MODEL_DIR/SHA256SUMS" results/expected-sha256.txt >/dev/null 2>&1; then
      echo "FAIL: staged SHA256SUMS differs from committed results/expected-sha256.txt pin"
      fail=1
    else
      echo "ok: staged SHA256SUMS matches the committed expected-hash pin"
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
