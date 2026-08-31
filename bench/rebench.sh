#!/usr/bin/env bash
# One-command Phase C re-benchmark: preflight + serve (continuous thermal /
# Xid / exact-GPU watch) + full suite; stops the owned server process tree.
# Usage: MODEL_DIR=... LLAMA_SERVER=... bench/rebench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_DIR:?set MODEL_DIR to the staged checkpoint directory}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the built llama-server binary}"
export MODEL_DIR LLAMA_SERVER PORT="${PORT:-8199}"
STOPFILE="${STOPFILE:-results/.rebench-stop-$$}"
rm -f "$STOPFILE"
cleanup() {
  [ -f "$STOPFILE" ] || touch "$STOPFILE"
  if [ -n "${WATCHER_PID:-}" ]; then
    wait "$WATCHER_PID" 2>/dev/null || true
  fi
  # Kill the owned group UNCONDITIONALLY: even if the leader died (watcher
  # kill during load), survivors like tee must not be orphaned.
  if [ -n "${SERVER_PID:-}" ]; then
    # serve.sh is launched under setsid below, so its PID is the PGID of an
    # owned process group (llama-server + tee); never a foreign group.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$STOPFILE"
}
trap cleanup EXIT
bench/thermal-guard.sh
bench/preflight.sh

# Capture harness identity BEFORE any run artifact (serve log, run dir)
# exists, so a clean checkout is not misreported as dirty.
HARNESS_REVISION="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain | head -1)" ]; then HARNESS_DIRTY=dirty; else HARNESS_DIRTY=clean; fi
export HARNESS_REVISION HARNESS_DIRTY
setsid bash bench/serve.sh > results/serve-rebench.log 2>&1 &
SERVER_PID=$!
# Continuous guard for the whole run: exact GPU count, thermal ceiling,
# Xid scan; on breach it kills the owned server group (WATCH_PID) itself.
WATCH_PID="$SERVER_PID" bench/thermal-guard.sh --watch "$STOPFILE" &
WATCHER_PID=$!
deadline=$((SECONDS + 2400))
until curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; do
  kill -0 "$SERVER_PID" || { echo "server died; tail:"; tail -20 results/serve-rebench.log; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "health timeout"; exit 1; }
  sleep 5
done
# One fresh run directory, created BEFORE launch; refuse to reuse one.
# Exactly ONE watched driver execution writes into it.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="results/run-${RUN_ID}"
[ ! -e "$OUTDIR" ] || { echo "refusing pre-existing output dir $OUTDIR"; exit 1; }
mkdir -p "$OUTDIR"

# Race the single watched driver against the safety watcher: any guard
# failure, watcher exit, or server death fails the whole run immediately.
OUTDIR="$OUTDIR" bash bench/phase63-driver.sh &
DRIVER_PID=$!
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
    wait "$WATCHER_PID"; wrc=$?
    echo "safety watch exited early (rc=$wrc); failing run"
    touch "$STOPFILE"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    kill -TERM "$DRIVER_PID" 2>/dev/null || true
    wait "$DRIVER_PID" 2>/dev/null || true
    exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died mid-run; failing run"
    touch "$STOPFILE"
    kill -TERM "$DRIVER_PID" 2>/dev/null || true
    wait "$DRIVER_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
wait "$DRIVER_PID"; driver_rc=$?
if [ "$driver_rc" -ne 0 ]; then echo "workload failed rc=$driver_rc"; exit 1; fi

# Stop the safety watcher and owned server group cleanly BEFORE offline
# derivation, and fail if the watcher was not healthy through the whole
# run window. Only then may "thermal breaches: none" be asserted.
touch "$STOPFILE"
wait "$WATCHER_PID"; watcher_rc=$?
if [ "$watcher_rc" -ne 0 ]; then
  echo "safety watch exited rc=$watcher_rc; failing run"
  exit 1
fi
kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
sleep 2
kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Every artifact of this run derives from this run's own outputs, with the
# watcher-verified thermal status and pre-run harness identity attached.
THERMAL_BREACHES=none bench/derive-row.sh "$OUTDIR"
python3 bench/summarize.py "$OUTDIR/experiments.csv" "$OUTDIR/quality.jsonl" "$OUTDIR/summary.csv" > "$OUTDIR/summary.json"
python3 bench/charts.py "$OUTDIR/summary.csv" "$OUTDIR/charts" "$OUTDIR/ladder.jsonl"
REBENCH_OUTDIR="$OUTDIR" python3 - <<'PY'
import json, os
from datetime import datetime, timezone
out = os.environ["REBENCH_OUTDIR"]
manifest = {
    "kind": "rebench-run",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": out,
    "artifacts": {
        "warmups": out + "/warmups.jsonl",
        "speed_c1": out + "/speed-c1.jsonl",
        "ladder": out + "/ladder.jsonl",
        "soak": out + "/soak.jsonl",
        "quality": out + "/quality.jsonl",
        "experiments_csv": out + "/experiments.csv",
        "summary_csv": out + "/summary.csv",
        "summary_json": out + "/summary.json",
        "charts_dir": out + "/charts",
        "gpu_snapshot": out + "/gpu-final.csv",
    },
}
with open(out + "/run-manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print("run manifest written")
PY
echo "rebench complete: $OUTDIR"
