#!/usr/bin/env bash
# Regression: the PRODUCTION driver's soak breach path must fail closed.
# Drives bench/phase63-driver.sh with bounded shims: the stub guard passes
# the five pre-soak invocations (startup, warmup, speed-c1, quality,
# ladder) and FAILS the first soak guard; stub measure/eval succeed and
# leave an invocation marker. Asserts nonzero exit, stage=soak-rep-1
# evidence, persisted GPU snapshot, and that the workload was reached.
# Run: bash bench/test-driver-breach.sh   (SKIP/2 without GNU timeout)
set -eu
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
cleanup() {
  [ -n "${STOP_PID:-}" ] && kill -9 "$STOP_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

if ! command -v timeout >/dev/null 2>&1; then
  echo 'SKIP: GNU timeout unavailable; production-driver breach test cannot run (not a PASS)'
  exit 2
fi

mkdir -p "$TMP/bin"
# Stub nvidia-smi: guard always sees a breach-temperature GPU; the
# driver's evidence snapshot uses the same stub.
cat > "$TMP/bin/nvidia-smi" <<'STUB'
#!/bin/sh
echo "0, 81, 60, 100, 50, 100"
exit 0
STUB
# Stub python3: measure.py/eval.py succeed and mark that the workload was
# actually reached; anything else delegates to the real python3.
cat > "$TMP/bin/python3" <<'PYSTUB'
#!/bin/sh
case "$1" in
  bench/measure.py|bench/eval.py)
    echo "invoked: $*" >> "$MARKER_FILE"
    exit 0 ;;
  *)
    exec /usr/bin/python3 "$@" ;;
esac
PYSTUB
chmod +x "$TMP/bin/nvidia-smi" "$TMP/bin/python3"

# Stub guard: passes invocations 1-5 (startup, warmup, speed-c1, quality,
# ladder), fails invocation 6 (first soak guard) with a THERMAL marker.
cat > "$TMP/guard.sh" <<'GSTUB'
#!/bin/sh
COUNT_FILE="$GUARD_COUNT"
n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$COUNT_FILE"
if [ "$n" -ge 6 ]; then
  echo "THERMAL STOP: stub guard breach at soak (call $n)"
  exit 1
fi
exit 0
GSTUB
chmod +x "$TMP/guard.sh"
export GUARD_COUNT="$TMP/guard.count"
MARKER_FILE="$TMP/workload.marker"
export MARKER_FILE

OUT="$TMP/out"
mkdir -p "$OUT"
rc=0
GUARD="$TMP/guard.sh" OUTDIR="$OUT" PATH="$TMP/bin:$PATH" \
  timeout 60 bash bench/phase63-driver.sh >"$TMP/stdout.log" 2>&1 || rc=$?

FAIL=0
if [ "$rc" -eq 0 ]; then
  echo 'FAIL: production driver exited 0 despite a mid-soak guard breach'
  FAIL=1
fi
guard_calls=$(cat "$GUARD_COUNT")
if [ "$guard_calls" -lt 6 ]; then
  echo "FAIL: guard only invoked $guard_calls times; pre-soak path not exercised"
  FAIL=1
fi
if [ ! -s "$MARKER_FILE" ]; then
  echo 'FAIL: workload shims never invoked; soak path not reached'
  FAIL=1
fi
if ! grep -q 'THERMAL STOP' "$TMP/stdout.log"; then
  echo 'FAIL: no THERMAL_STOP message from guard'
  FAIL=1
fi
if [ ! -s "$OUT/guard-breach.out" ]; then
  echo 'FAIL: guard-breach.out missing or empty'
  FAIL=1
fi
grep -q 'stage=soak-rep-1' "$OUT/guard-breach.out" || { echo 'FAIL: breach stage is not soak-rep-1'; cat "$OUT/guard-breach.out"; FAIL=1; }
if [ ! -s "$OUT/gpu-final.csv" ]; then
  echo 'FAIL: breach GPU snapshot not persisted'
  FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then
  echo 'driver-breach regression (production driver, mid-soak): PASS'
fi
exit "$FAIL"
