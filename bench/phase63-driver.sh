#!/usr/bin/env bash
# Phase C (#63) baseline driver: 3 warm-ups + 5 measured reps at concurrency 1,
# then quality pack, bounded concurrency ladder, and a time-boxed soak.
set -euo pipefail
cd "$(dirname "$0")/.."
OUTDIR="${OUTDIR:-results/phase63}"
BASE="${BASE:-http://127.0.0.1:8199}"
PROMPT="$(cat bench/prompt-1k.txt)"
# GUARD is env-overridable so the breach regression can drive THIS script
# against a bounded stub guard (no GPU); production default unchanged.
GUARD="${GUARD:-bench/thermal-guard.sh}"
mkdir -p "$OUTDIR"

# Terminal breach handler shared by every stage: persist breach evidence
# plus a final GPU snapshot, then exit nonzero so the driver can never
# reach a successful completion after a guard failure.
guard_breach() {
  local out="$1" stage="$2"
  echo "GUARD_BREACH stage=$stage" > "$out/guard-breach.out"
  nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
    --format=csv,noheader | tee "$out/gpu-final.csv" || true
  exit 1
}

echo "== thermal guard =="
"$GUARD" || guard_breach "${OUTDIR}" startup

echo "== 3 warm-up runs (discarded) =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1 --runs 3 --out ${OUTDIR}/warmups.jsonl \
  | tee ${OUTDIR}/warmups.log
"$GUARD" || guard_breach "${OUTDIR}" warmup

echo "== 5 measured single-stream reps =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1 --runs 5 --out ${OUTDIR}/speed-c1.jsonl \
  | tee ${OUTDIR}/speed-c1.log
"$GUARD" || guard_breach "${OUTDIR}" speed-c1

echo "== quality / validity pack =="
python3 bench/eval.py --base "$BASE" --buckets math,instruction,coding,longctx,heldout \
  --out ${OUTDIR}/quality.jsonl | tee ${OUTDIR}/quality.log
"$GUARD" || guard_breach "${OUTDIR}" quality

echo "== bounded concurrency ladder (1,2,4 x 2 reps) =="
python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
  --concurrency 1,2,4 --runs 2 --out ${OUTDIR}/ladder.jsonl \
  | tee ${OUTDIR}/ladder.log
"$GUARD" || guard_breach "${OUTDIR}" ladder

echo "== 20-min soak: sustained c=2, 256-tok completions =="
SOAK_END=$((SECONDS + 1200))
SOAK_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
soak_n=0
: > ${OUTDIR}/soak.jsonl
while [ "$SECONDS" -lt "$SOAK_END" ]; do
  python3 bench/measure.py --base "$BASE" --prompt "$PROMPT" --max-tokens 256 \
    --concurrency 2 --runs 1 --out ${OUTDIR}/soak.jsonl \
    | tee -a ${OUTDIR}/soak.log
  soak_n=$((soak_n+1))
  "$GUARD" || guard_breach "${OUTDIR}" "soak-rep-$soak_n"
done
SOAK_END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SOAK_START_EPOCH="$(date -u -d "$SOAK_START_UTC" +%s 2>/dev/null || echo 0)"
SOAK_END_EPOCH="$(date -u -d "$SOAK_END_UTC" +%s 2>/dev/null || echo 0)"
printf '{"start_utc": "%s", "end_utc": "%s", "window_s": %s}\n' \
  "$SOAK_START_UTC" "$SOAK_END_UTC" "$((SOAK_END_EPOCH - SOAK_START_EPOCH))" \
  > "${OUTDIR}/soak-window.json"
echo "soak reps completed: $soak_n"

echo "== gpu snapshot =="
nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu,temperature.memory,power.draw \
  --format=csv,noheader | tee ${OUTDIR}/gpu-final.csv
echo "== driver complete =="
