#!/usr/bin/env bash
# Verifies the fail-closed leader-loss path of bench/thermal-guard.sh:
# when the watched group leader dies while a child survives in the same
# process group, the watcher must kill the whole owned group and exit
# nonzero. Run: bash bench/test-thermal-guard.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# Prerequisites: without these the test cannot exercise the path and must
# FAIL, not silently pass (review 5062227005 finding 1).
command -v setsid >/dev/null || { echo "FAIL: setsid unavailable"; exit 1; }
command -v ps >/dev/null    || { echo "FAIL: ps unavailable"; exit 1; }
STOPFILE="/tmp/.tg-stop-$$"
cleanup() { rm -f "$STOPFILE"; }
trap cleanup EXIT
FAIL=0
# Case 1: absent leader, no owned group -> must exit nonzero.
if WATCH_PID="999999" WATCH_PGID="999999" EXPECT_GPUS=4 timeout 30 \
   bash bench/thermal-guard.sh --watch "$STOPFILE" >/dev/null 2>&1; then
  echo "FAIL: absent leader returned rc=0"
  FAIL=1
fi
# Case 2: genuine leader loss with a surviving child in the same PGID.
# Start the group, spawn a child into it, then kill ONLY the leader so the
# child is orphaned but still a member of the owned group.
CHILD_FILE="/tmp/.tg-child-$$"
setsid bash -c 'sleep 300 & echo $! > "$0"; wait' "$CHILD_FILE" &
LEADER=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -s "$CHILD_FILE" ] && break
  sleep 0.1
done
CHILD="$(cat "$CHILD_FILE" 2>/dev/null || true)"
rm -f "$CHILD_FILE"
[ -n "$CHILD" ] || { echo "FAIL: could not start child in leader group"; kill -KILL -- "-$LEADER" 2>/dev/null || true; exit 1; }
PGID="$(ps -o pgid= -p "$LEADER" | tr -d ' ')"
[ -n "$PGID" ] || { echo "FAIL: could not resolve leader PGID"; kill -KILL "$LEADER" "$CHILD" 2>/dev/null || true; exit 1; }
# Sanity: the child must share the leader PGID before we kill the leader.
CHILD_PGID="$(ps -o pgid= -p "$CHILD" | tr -d ' ')"
[ "$CHILD_PGID" = "$PGID" ] || { echo "FAIL: child PGID mismatch"; kill -KILL -- "-$PGID" 2>/dev/null || true; exit 1; }
# Leader loss: kill the leader only; the child must survive that kill.
kill -KILL "$LEADER" 2>/dev/null || true
wait "$LEADER" 2>/dev/null || true
kill -0 "$CHILD" 2>/dev/null || { echo "FAIL: child died before watcher ran"; exit 1; }
# Watcher must see leader loss, kill the owned group, and exit nonzero.
if WATCH_PID="$LEADER" WATCH_PGID="$PGID" EXPECT_GPUS=4 timeout 30 \
   bash bench/thermal-guard.sh --watch "$STOPFILE" >/dev/null 2>&1; then
  echo "FAIL: leader-loss returned rc=0"
  FAIL=1
fi
sleep 1
if kill -0 -"$PGID" 2>/dev/null; then
  echo "FAIL: owned group $PGID still alive after leader-loss cleanup"
  kill -KILL -- "-$PGID" 2>/dev/null || true
  FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then echo "thermal-guard leader-loss test: PASS"; fi
exit "$FAIL"
