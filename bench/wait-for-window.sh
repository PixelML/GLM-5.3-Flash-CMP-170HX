#!/usr/bin/env bash
# Read-only waiter. Logs every 60 s: per-GPU used memory, shard staging state.
# NEVER starts, stops, or signals anything. Writes only to results/waitlog.csv.
# Window = every visible GPU < 2048 MiB used AND the staged checkpoint is
# complete (EXPECT_SHARDS shards, EXPECT_BYTES total, SHA256SUMS present).
# Proceed manually when the CSV shows complete=1; there is no auto-trigger.
set -u
cd "$(dirname "$0")/.."
: "${SHARD_DIR:?set SHARD_DIR to the staged checkpoint directory}"
OUT="${OUT:-results/waitlog.csv}"
EXPECT_BYTES="${EXPECT_BYTES:-156822111075}"
EXPECT_SHARDS="${EXPECT_SHARDS:-5}"
mkdir -p "$(dirname "$OUT")"

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used \
                      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ ! -f "$OUT" ]; then
    header="ts"
    for i in "${!used[@]}"; do header="$header,gpu_used_mib_$i"; done
    echo "$header,shards,sha256sums,total_bytes,complete" >> "$OUT"
  fi
  shards=$(find "$SHARD_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null | wc -l)
  if [ -f "$SHARD_DIR/SHA256SUMS" ]; then s=1; else s=0; fi
  bytes=$(stat -c %s "$SHARD_DIR"/*.gguf 2>/dev/null | awk '{t+=$1} END{print t+0}')
  gpus_ok=1
  [ "${#used[@]}" -eq 0 ] && gpus_ok=0
  for m in "${used[@]}"; do
    if [ -z "$m" ] || [ "$m" -ge 2048 ]; then gpus_ok=0; fi
  done
  shards_ok=0; [ "$shards" -eq "$EXPECT_SHARDS" ] && shards_ok=1
  bytes_ok=0;  [ "$bytes" -eq "$EXPECT_BYTES" ]   && bytes_ok=1
  complete=$(( gpus_ok && shards_ok && s == 1 && bytes_ok ))
  row="$(date -u +%FT%TZ)"
  for m in "${used[@]}"; do row="$row,$m"; done
  echo "$row,$shards,$s,$bytes,$complete" >> "$OUT"
  sleep 60
done
