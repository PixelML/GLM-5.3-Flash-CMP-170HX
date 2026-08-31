#!/usr/bin/env bash
# Regression: a guard breach mid-driver must terminate the PRODUCTION
# driver (bench/phase63-driver.sh) NONZERO and persist breach evidence.
# Runs the real driver with bounded command/guard shims (no GPU, no
# server); fails if the production breach path ever regresses.
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
# Stub nvidia-smi: guard always sees a breach-temperature GPU; the
# driver's evidence snapshot uses the same stub.
cat > "$TMP/bin/nvidia-smi" <<'STUB'
#!/bin/sh
echo "0, 81, 60, 100, 50, 100"
exit 0
STUB
# Stub measure/eval: never called before the startup guard, but present so
# an accidental early-success path would be observable.
cat > "$TMP/bin/python3" <<'PYSTUB'
#!/bin/sh
case "$1" in
  bench/measure.py|bench/eval.py) echo '{"unexpected_run": true}' >&2; exit 42 ;;
  *) exec /usr/bin/env -u PYTHONPATH /usr/bin/python3 "$@" ;;
esac
PYSTUB
chmod +x "$TMP/bin/nvidia-smi" "$TMP/bin/python3"
# Bounded stub guard: fails immediately, printing a recognizable marker.
printf '#!/bin/sh
echo "THERMAL STOP: stub guard breach"
exit 1
' > "$TMP/guard.sh"
chmod +x "$TMP/guard.sh"

OUT="$TMP/out"
mkdir -p "$OUT"
rc=0
GUARD="$TMP/guard.sh" OUTDIR="$OUT" PATH="$TMP/bin:$PATH" \
  timeout 60 bash bench/phase63-driver.sh >"$TMP/stdout.log" 2>&1 || rc=$?

FAIL=0
if [ "$rc" -eq 0 ]; then
  echo 'FAIL: production driver exited 0 despite a guard breach'
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
grep -q 'stage=startup' "$OUT/guard-breach.out" || { echo 'FAIL: wrong breach stage'; FAIL=1; }
if [ ! -s "$OUT/gpu-final.csv" ]; then
  echo 'FAIL: breach GPU snapshot not persisted'
  FAIL=1
fi
if grep -q 'unexpected_run' "$TMP/stdout.log"; then
  echo 'FAIL: workload was invoked despite guard breach'
  FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then
  echo 'driver-breach regression (production driver): PASS'
fi
exit "$FAIL"
