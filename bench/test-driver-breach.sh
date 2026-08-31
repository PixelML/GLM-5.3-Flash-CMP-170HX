#!/usr/bin/env bash
# Regression: a guard breach mid-driver must terminate the driver NONZERO
# and persist evidence of the breach, so derive-row can never be reached
# with a successful exit. Uses a stub nvidia-smi shim; no GPU required.
# Run: bash bench/test-driver-breach.sh
set -eu
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
cleanup() {
  [ -n "${STOP_PID:-}" ] && kill -9 "$STOP_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

mkdir -p "$TMP/bin"
cat > "$TMP/bin/nvidia-smi" <<'STUB'
#!/bin/sh
# Always report a core temp at/above the 80 C ceiling -> any guard
# invocation (start, mid-driver, or watch) must STOP.
echo "0, 81, 60, 100, 50, 100"
exit 0
STUB
chmod +x "$TMP/bin/nvidia-smi"

cat > "$TMP/driver.sh" <<'DRIVER'
set -euo pipefail
GUARD="$1"; OUT="$2"
if "$GUARD"; then
  echo "phase simulated driver passed guard" > /dev/null
else
  echo "GUARD_FAILED" > "$OUT/breach.out"
  echo "gpu snapshot would be written here" > "$OUT/gpu-final.csv"
  exit 1
fi
echo "guard passed (unexpected with stub)"
DRIVER
chmod +x "$TMP/driver.sh"

if PATH="$TMP/bin:$PATH" bash "$TMP/driver.sh" "$(pwd)/bench/thermal-guard.sh" "$TMP" >"$TMP/stdout.log" 2>&1; then
  echo 'FAIL: driver exited 0 despite a guard breach'
  cat "$TMP/stdout.log"
  exit 1
fi
grep -q THERMAL "$TMP/stdout.log" || { echo 'FAIL: no THERMAL_STOP message'; cat "$TMP/stdout.log"; exit 1; }
grep -q GUARD_FAILED "$TMP/breach.out" || { echo 'FAIL: breach evidence not persisted'; exit 1; }
echo 'driver-breach regression: PASS'
