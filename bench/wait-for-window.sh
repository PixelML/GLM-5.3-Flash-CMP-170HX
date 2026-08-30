#!/usr/bin/env bash
# Read-only waiter. Logs every 60 s: per-GPU used memory, shard staging state.
# NEVER starts, stops, or signals anything. Writes only to results/waitlog.csv.
# Program rule: proceed only when all 3 GPUs report <2048 MiB used AND the
# staging directory holds 5 complete shards marked by SHA256SUMS.
set -u
OUT="${OUT:-/tmp/glm-cmp-branch/results/waitlog.csv}"
SHARD_DIR="${SHARD_DIR:-/library/models/glm-5.3-flash/unsloth-UD-IQ4_XS}"
EXPECT_BYTES=156822111075
[ -f "$OUT" ] || echo "ts,gpu_used_mib_0,gpu_used_mib_1,gpu_used_mib_2,shards,sha256sums,total_bytes,complete" >> "$OUT"
while true; do
  read -r m0 m1 m2 <<<"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | paste -sd' ')"
  shards=$(find "$SHARD_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null | wc -l)
  if [ -f "$SHARD_DIR/SHA256SUMS" ]; then s=1; else s=0; fi
  bytes=$(stat -c %s "$SHARD_DIR"/*.gguf 2>/dev/null | awk '{t+=$1} END{print t+0}')
  gpus_ok=1
  for m in "$m0" "$m1" "$m2"; do [ -n "$m" ] && [ "$m" -lt 2048 ] || gpus_ok=0; done
  bytes_ok=0; [ "$bytes" -eq "$EXPECT_BYTES" ] && bytes_ok=1
  complete=$(( gpus_ok && shards==5 && s==1 && bytes_ok ))
  echo "$(date -u +%FT%TZ),$m0,$m1,$m2,$shards,$s,$bytes,$complete" >> "$OUT"
  sleep 60
done
