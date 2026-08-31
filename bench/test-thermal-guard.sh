#!/usr/bin/env bash
# Genuine leader-loss reproduction for the thermal-guard watch mode: an
# absent leader with surviving children must terminate the owned group and
# return nonzero. Setup prerequisites are fail-closed: if setsid, ps, or
# timeout is unavailable the test reports SKIP (exit 2), never PASS.
# Run: bash bench/test-thermal-guard.sh
set -u
cd "$(dirname $0)/.."
FAIL=0
for tool in setsid ps timeout; do
  if ! command -v $tool >/dev/null 2>&1; then
    echo 'SKIP: tool unavailable; leader-loss test cannot run (not a PASS)'
    exit 2
  fi
done
STOPFILE=${STOPFILE:-/tmp/.tg-stop-$$}
rm -f $STOPFILE
setsid bash -c 'sleep 300 & wait' &
LEADER=$!
sleep 0.3
PGID=$(ps -o pgid= -p $LEADER | tr -d ' ')
[ -n $PGID ] || { echo 'FAIL: could not resolve PGID'; exit 1; }
kill -9 $LEADER 2>/dev/null || true
sleep 0.2
kill -0 $LEADER 2>/dev/null && { echo 'FAIL: leader did not die'; exit 1; }
WATCH_PID=$LEADER WATCH_PGID=$PGID EXPECT_GPUS=4 timeout 30 bash bench/thermal-guard.sh --watch $STOPFILE >/dev/null 2>&1
rc=$?
sleep 1
if kill -0 -$PGID 2>/dev/null; then
  echo 'FAIL: owned group $PGID still alive after leader-loss kill'
  FAIL=1
fi
[ $rc -ne 0 ] || { echo 'FAIL: leader-loss with survivors returned rc=0'; FAIL=1; }
WATCH_PID=999999 WATCH_PGID=999999 EXPECT_GPUS=4 timeout 30 bash bench/thermal-guard.sh --watch $STOPFILE >/dev/null 2>&1
rc1=$?
[ $rc1 -ne 0 ] || { echo 'FAIL: absent leader returned rc=0'; FAIL=1; }
rm -f $STOPFILE
if [ $FAIL -eq 0 ]; then echo 'thermal-guard leader-loss test: PASS'; fi
exit $FAIL
