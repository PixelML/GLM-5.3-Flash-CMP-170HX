#!/usr/bin/env bash
# Read-only preflight for the llama.cpp benchmark loop. Starts nothing.
# Exit 0 = safe to serve; exit 1 = blocked (reasons printed). Never touches
# processes, GPUs, or storage outside checks.
set -u
fail=0

echo "== preflight =="

# 1. No foreign GPU workload (never kill; just refuse).
#    Catches both `vllm serve` and worker-style process names (VLLM::Worker_*).
if pgrep -f 'vllm serve' >/dev/null 2>&1; then
  echo "FAIL: a vLLM server is running; per safety rules: stop, report, wait."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fail=1
else
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${apps:-0}" -gt 0 ]; then
    echo "FAIL: $apps compute process(es) hold GPU memory; per safety rules: stop, report, wait."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    fail=1
  else
    echo "ok: no compute processes on any GPU"
  fi
fi

# 2. Enough free VRAM on every card (weights ~48.7 GiB/card at even split + ctx/KV).
need_mib=50000
while read -r idx free; do
  if [ "$free" -lt "$need_mib" ]; then
    echo "FAIL: GPU $idx has ${free} MiB free (< ${need_mib} MiB needed)"
    fail=1
  fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

# 3. Model staging complete: 5 shards, each non-trivially sized, total exact,
#    and SHA256SUMS present (the completion marker).
shard_dir="${MODEL_DIR:-/library/models/glm-5.3-flash/unsloth-UD-IQ4_XS}"
EXPECT_BYTES=156822111075
count=$(find "$shard_dir" -maxdepth 1 -name '*.gguf' | wc -l)
if [ "$count" -ne 5 ]; then
  echo "FAIL: $count/5 GGUF shards staged in $shard_dir"
  fail=1
else
  small=$(find "$shard_dir" -maxdepth 1 -name '*.gguf' -size -1G | wc -l)
  if [ "$small" -ne 0 ]; then
    echo "FAIL: $small shard(s) under 1 GiB (staging incomplete?)"
    fail=1
  else
    echo "ok: 5/5 shards staged, each >1 GiB"
  fi
  bytes=$(stat -c %s "$shard_dir"/*.gguf | awk '{t+=$1} END{print t+0}')
  if [ "$bytes" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: staged bytes $bytes != expected $EXPECT_BYTES (download still running or corrupt)"
    fail=1
  else
    echo "ok: staged byte total is exact ($bytes)"
  fi
fi
if [ ! -f "$shard_dir/SHA256SUMS" ]; then
  echo "FAIL: SHA256SUMS not present in $shard_dir (staging not marked complete)"
  fail=1
else
  echo "ok: SHA256SUMS present"
fi

# 4. Runtime binary present.
LLAMA_SERVER="${LLAMA_SERVER:-/tmp/build-llamacpp/bin/llama-server}"
if [ -x "$LLAMA_SERVER" ]; then
  echo "ok: $LLAMA_SERVER"
else
  echo "FAIL: $LLAMA_SERVER missing or not executable"
  fail=1
fi

# 5. Port 8199 free.
if ss -ltn 2>/dev/null | grep -q ':8199 '; then
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
