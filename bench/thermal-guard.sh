#!/usr/bin/env bash
# Read-only thermal/Xid guard. Exit 0 = safe to continue, exit 1 = STOP.
# Hard limits per safety policy: CORE_MAX (default 80 C), MEM_MAX (85 C).
# Also stops if nvidia-smi fails, no GPUs are visible, or an NVIDIA Xid
# appears in the kernel log. Never modifies GPU state.
set -u
CORE_MAX="${CORE_MAX:-80}"
MEM_MAX="${MEM_MAX:-85}"
WATCH_PID="${WATCH_PID:-}"
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"

# --watch <stop-file>: monitor until the stop file appears or WATCH_PID exits.
# On breach/GPU loss, kill only WATCH_PID (our server), never a foreign PID.
if [ "${1:-}" = "--watch" ]; then
  stop_file="${2:?usage: thermal-guard.sh --watch <stop-file-path>}"
  while true; do
    [ -f "$stop_file" ] && exit 0
    if [ -n "$WATCH_PID" ] && ! kill -0 "$WATCH_PID" 2>/dev/null; then
      echo "[thermal-watch] server process gone; watch exits"
      exit 0
    fi
    if ! out=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory \
                --format=csv,noheader,nounits 2>&1); then
      echo "[thermal-watch] nvidia-smi failed: $out"
      [ -z "$WATCH_PID" ] || kill "$WATCH_PID" 2>/dev/null || true
      exit 1
    fi
    breach=""
    while IFS= read -r line; do
      IFS=, read -r tc tm <<< "$(echo "$line" | tr -d ' ')"
      tc="${tc:-999}"; tm="${tm:-999}"
      if [ "$tc" -ge "$CORE_MAX" ] || [ "$tm" -ge "$MEM_MAX" ]; then
        breach="core=${tc}C mem=${tm}C (limits ${CORE_MAX}/${MEM_MAX})"
      fi
    done <<< "$out"
    if [ -n "$breach" ]; then
      echo "[thermal-watch] THERMAL BREACH: $breach"
      [ -z "$WATCH_PID" ] || kill "$WATCH_PID" 2>/dev/null || true
      exit 1
    fi
    sleep "$WATCH_INTERVAL"
  done
fi

if ! out=$(nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory \
            --format=csv,noheader,nounits 2>&1); then
  echo "THERMAL STOP: nvidia-smi failed: $out"
  exit 1
fi
if [ -z "$out" ]; then
  echo "THERMAL STOP: no GPUs visible"
  exit 1
fi
while IFS= read -r line; do
  IFS=, read -r idx tc tm <<< "$(echo "$line" | tr -d ' ')"
  tc="${tc:-999}"; tm="${tm:-999}"
  if [ "$tc" -ge "$CORE_MAX" ]; then
    echo "THERMAL STOP: GPU $idx core ${tc} C >= ${CORE_MAX} C"
    exit 1
  fi
  if [ "$tm" -ge "$MEM_MAX" ]; then
    echo "THERMAL STOP: GPU $idx memory ${tm} C >= ${MEM_MAX} C"
    exit 1
  fi
done <<< "$out"

# Best-effort Xid scan; dmesg may be restricted, in which case this is skipped.
xids=$(dmesg -T 2>/dev/null | grep -i xid | tail -5 || true)
if [ -n "$xids" ]; then
  echo "THERMAL STOP: NVIDIA Xid detected in kernel log:"
  echo "$xids"
  exit 1
fi
exit 0
