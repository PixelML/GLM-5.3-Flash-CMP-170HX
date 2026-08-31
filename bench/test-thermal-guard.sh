#!/usr/bin/env bash
# Genuine leader-loss regression test for the thermal-guard watch mode.
# Setup contract (fail-closed): requires setsid/ps/timeout (else SKIP, exit 2);
# captures a real surviving child PID, verifies it shares the leader PGID and
# is alive after the leader is killed, and only then runs the watcher.
# Quoting is strict; any setup error aborts nonzero, never a false PASS.
# Run: bash bench/test-thermal-guard.sh
set -eu
cd "$(dirname "$0")/.."
STOPFILE=${STOPFILE:-/tmp/.tg-stop-$$}
export STOPFILE
cleanup() { rm -f "$STOPFILE" "$STOPFILE.cpid"; }
trap cleanup EXIT
for tool in setsid ps timeout; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "SKIP: $tool unavailable; leader-loss test cannot run (not a PASS)"
    exit 2
  fi
done

# Start an owned group: bash leader plus a sleep child that survives it.
# The inner shell records the sleep child's own PID ($!), not the leader
# PID, so the orphan precondition is genuinely provable.
setsid bash -c 'sleep 300 & echo $! > "$STOPFILE.cpid"; wait' &
LEADER=$!
for _ in $(seq 1 50); do
  if [ -s "$STOPFILE.cpid" ]; then break; fi
  sleep 0.1
done
CHILD=$(cat "$STOPFILE.cpid" 2>/dev/null || true)
rm -f "$STOPFILE.cpid"
if [ -z "$CHILD" ]; then
  echo 'FAIL: no child PID captured; setup broken'
  kill -9 "$LEADER" 2>/dev/null || true
  exit 1
fi
PGID=$(ps -o pgid= -p "$LEADER" | tr -d ' ')
CPGID=$(ps -o pgid= -p "$CHILD" | tr -d ' ')
if [ -z "$PGID" ] || [ "$PGID" != "$CPGID" ]; then
  echo 'FAIL: child PGID mismatch or missing; setup broken'
  kill -9 "$LEADER" 2>/dev/null || true
  exit 1
fi
if ! kill -0 "$CHILD" 2>/dev/null; then
  echo 'FAIL: child not alive before leader kill; setup broken'
  kill -9 "$LEADER" 2>/dev/null || true
  exit 1
fi

# Kill the leader; the child must survive as an orphan in the same group.
kill -9 "$LEADER" 2>/dev/null || true
sleep 0.2
if kill -0 "$LEADER" 2>/dev/null; then
  echo 'FAIL: leader did not die'
  kill -9 "$CHILD" 2>/dev/null || true
  exit 1
fi
sleep 0.1
if ! kill -0 "$CHILD" 2>/dev/null; then
  echo 'FAIL: no surviving child after leader kill; precondition unproven'
  exit 1
fi
sleep 0.4
if ! kill -0 "$CHILD" 2>/dev/null; then
  echo 'FAIL: child died shortly after leader kill; precondition unproven'
  exit 1
fi

# The watcher must fail closed AND kill the whole owned group.
rc=0
WATCH_PID="$LEADER" WATCH_PGID="$PGID" EXPECT_GPUS=4 timeout 30 bash bench/thermal-guard.sh --watch "$STOPFILE" >/dev/null 2>&1 || rc=$?
sleep 1
FAIL=0
if kill -0 "$CHILD" 2>/dev/null; then
  echo 'FAIL: surviving child still alive after leader-loss kill'
  FAIL=1
fi
if [ "$rc" -eq 0 ]; then
  echo 'FAIL: leader-loss with survivors returned rc=0'
  FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then
  echo 'thermal-guard leader-loss test: PASS'
fi
exit "$FAIL"
