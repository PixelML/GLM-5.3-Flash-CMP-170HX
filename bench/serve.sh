#!/usr/bin/env bash
# Serve GLM-5.3-Flash UD-IQ4_XS on the CMP node via llama.cpp.
# Requires: llama-server built from unslothai/llama.cpp @ 00699716c275498ff84d71e329178fe21cba56a6
# Usage: MODEL_DIR=/path/to/shards ./serve.sh
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the directory containing the 5 UD-IQ4_XS shards}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
MODEL_FILE="${MODEL_FILE:-GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}"
PORT="${PORT:-8199}"
CTX="${CTX:-16384}"
SPLIT="${SPLIT:-layer}"
TSPLIT="${TSPLIT:-1,1,1,1}"  # equal ratios, one per GPU; documented default
PARALLEL="${PARALLEL:-1}"  # concurrent slots; keep 1 for single-stream latency runs
THREADS="${THREADS:-8}"    # experiment factor 3
# Remaining permitted factors; empty = server default (record it from the
# startup log). Set exactly ONE knob differently from baseline per experiment.
FLASH_ATTN="${FLASH_ATTN:-}"   # on | off | auto (fork default: auto)
CACHE_K="${CACHE_K:-}"         # f16 | q8_0 | q4_0
CACHE_V="${CACHE_V:-}"         # f16 | q8_0 | q4_0
BATCH="${BATCH:-}"             # logical batch size (server default 2048)
UBATCH="${UBATCH:-}"           # physical ubatch size (server default 512)

extra_args=()
if [ -n "$FLASH_ATTN" ]; then extra_args+=(--flash-attn "$FLASH_ATTN"); fi
if [ -n "$CACHE_K" ];     then extra_args+=(--cache-type-k "$CACHE_K"); fi
if [ -n "$CACHE_V" ];     then extra_args+=(--cache-type-v "$CACHE_V"); fi
if [ -n "$BATCH" ];       then extra_args+=(--batch-size "$BATCH"); fi
if [ -n "$UBATCH" ];      then extra_args+=(--ubatch-size "$UBATCH"); fi
# NOTE: mmap by default. --no-mmap reads the full weight set into host RAM before
# offload; weights total 146.05 GiB (measured, HF blob sum) vs 94 GiB host RAM
# (measured, free -h), so --no-mmap is statically infeasible on this node.
# Inferred from hardware arithmetic, not measured — set NO_MMAP=1 to A/B it.

echo "serving on port $PORT, ctx=$CTX, split=$SPLIT, parallel=$PARALLEL"
exec "$LLAMA_SERVER" \
  --model "$MODEL_DIR/$MODEL_FILE" \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers 999 \
  --split-mode "$SPLIT" \
  --tensor-split "$TSPLIT" \
  --parallel "$PARALLEL" \
  --threads "$THREADS" \
  "${extra_args[@]}" \
  ${NO_MMAP:+--no-mmap} \
  2>&1 | tee "${SERVE_LOG:-results/serve.log}"
