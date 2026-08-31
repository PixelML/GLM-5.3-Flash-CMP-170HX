#!/usr/bin/env bash
# Phase C (#63) baseline driver: 3 warm-ups + 5 measured reps at concurrency 1,
# then quality pack, bounded concurrency ladder, and a time-boxed soak.
set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${OUTDIR:-results/phase63}"
BASE="${BASE:-http://127.0.0.1:8199}"
PROMPT="$(cat bench/prompt-1k.txt)"
GUARD=bench/thermal-guard.sh
mkdir -p "$OUTDIR"

echo "== thermal guard =="
"$GUARD"

echo "== 3 warm-up runs (discarded) =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1 --runs 3 --out ${OUTDIR}/warmups.jsonl \
  | tee ${OUTDIR}/warmups.log
if ! "$GUARD"; then
  # A guard breach is terminal: persist evidence, stop nonzero, and never
  # fall through to a successful exit or offline derivation.
  echo "GUARD_BREACH stage=warmup" > "${OUTDIR}/guard-breach.out"
  nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
    --format=csv,noheader | tee "${OUTDIR}/gpu-final.csv" || true
  exit 1
fi

echo "== 5 measured single-stream reps =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1 --runs 5 --out ${OUTDIR}/speed-c1.jsonl \
  | tee ${OUTDIR}/speed-c1.log
if ! "$GUARD"; then
  echo "GUARD_BREACH stage=speed-c1" > "${OUTDIR}/guard-breach.out"
  nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
    --format=csv,noheader | tee "${OUTDIR}/gpu-final.csv" || true
  exit 1
fi

echo "== quality / validity pack =="
python3 bench/eval.py --base "$BASE" --buckets math,instruction,coding,longctx,heldout \
  --out ${OUTDIR}/quality.jsonl | tee ${OUTDIR}/quality.log
if ! "$GUARD"; then
  echo "GUARD_BREACH stage=quality" > "${OUTDIR}/guard-breach.out"
  nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
    --format=csv,noheader | tee "${OUTDIR}/gpu-final.csv" || true
  exit 1
fi

echo "== bounded concurrency ladder (1,2,4 x 2 reps) =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1,2,4 --runs 2 --out ${OUTDIR}/ladder.jsonl \
  | tee ${OUTDIR}/ladder.log
if ! "$GUARD"; then
  echo "GUARD_BREACH stage=ladder" > "${OUTDIR}/guard-breach.out"
  nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
    --format=csv,noheader | tee "${OUTDIR}/gpu-final.csv" || true
  exit 1
fi

echo "== 20-min soak: sustained c=2, 256-tok completions =="
SOAK_END=$((SECONDS + 1200))
soak_n=0
: > ${OUTDIR}/soak.jsonl
while [ "$SECONDS" -lt "$SOAK_END" ]; do
  python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
    --concurrency 2 --runs 1 --out ${OUTDIR}/soak.jsonl \
    | tee -a ${OUTDIR}/soak.log
  soak_n=$((soak_n+1))
  if ! "$GUARD"; then
    echo "GUARD_BREACH stage=soak rep=$soak_n" > "${OUTDIR}/guard-breach.out"
    nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
      --format=csv,noheader | tee "${OUTDIR}/gpu-final.csv" || true
    exit 1
  fi
done
echo "soak reps completed: $soak_n"

echo "== gpu snapshot =="
nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
  --format=csv,noheader | tee ${OUTDIR}/gpu-final.csv
echo "== driver complete =="
