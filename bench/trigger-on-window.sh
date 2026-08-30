#!/usr/bin/env bash
# One-shot trigger: when the waiter logs complete=1 (staging done AND all GPUs
# free), run the baseline smoke once through the guarded runner. Before that,
# it only reads. If another process takes port 8199 first, preflight refuses
# and this exits without side effects.
cd /tmp/glm-cmp-branch
LOG=results/trigger.log
echo "$(date -u +%FT%TZ) trigger armed" >> "$LOG"
while true; do
  last=$(tail -1 results/waitlog.csv 2>/dev/null)
  if [[ "$last" == *,1 ]]; then
    echo "$(date -u +%FT%TZ) window open; starting baseline smoke" >> "$LOG"
    NAME=baseline LLAMA_SERVER=/tmp/build-llamacpp/bin/llama-server \
      MODEL_DIR=/library/models/glm-5.3-flash/unsloth-UD-IQ4_XS \
      bench/run-experiment.sh >> "$LOG" 2>&1
    echo "$(date -u +%FT%TZ) baseline finished rc=$?" >> "$LOG"
    break
  fi
  sleep 120
done
