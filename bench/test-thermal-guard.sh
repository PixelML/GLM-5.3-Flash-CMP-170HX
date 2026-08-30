#!/usr/bin/env bash
# Static reproduction of review 5062184381 finding 1: an absent watcher
# leader must terminate the owned group and return nonzero (never a silent
# rc=0 that orphans survivors). Run: bash bench/test-thermal-guard.sh
set -u
cd "$(dirname "$0")/.."
FAIL=0
# Case 1: absent leader, no owned group -> must exit nonzero.
WATCH_PID="999999" WATCH_PGID="999999" EXPECT_GPUS=4 timeout 30 bash bench/thermal-guard.sh --watch "/tmp/.tg-stop-$$" >/dev/null 2>&1
rc1=$?
[ "$rc1" -ne 0 ] || { echo "FAIL: absent leader returned rc=0"; FAIL=1; }
# Case 2: owned group with a sleeping survivor + absent leader -> the whole
# group must be gone after the watcher exits.
setsid sleep 300 &
LEADER=$!
PGID=$(ps -o pgid= -p "$LEADER" | tr -d ' ')
WATCH_PID="$LEADER" WATCH_PGID="$PGID" EXPECT_GPUS=4 timeout 30 bash bench/thermal-guard.sh --watch "/tmp/.tg-stop-$$" >/dev/null 2>&1
rc2=$?
sleep 1
if kill -0 -"$PGID" 2>/dev/null; then
  echo "FAIL: owned group $PGID still alive after leader-loss kill"
  FAIL=1
fi
[ "$rc2" -ne 0 ] || { echo "FAIL: leader-loss with survivors returned rc=0"; FAIL=1; }
rm -f "/tmp/.tg-stop-$$"
if [ "$FAIL" -eq 0 ]; then echo "thermal-guard leader-loss test: PASS"; fi
exit "$FAIL"
