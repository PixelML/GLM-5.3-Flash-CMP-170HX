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
  if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
    echo "safety watch exited during server load; failing run"
    exit 1
  fi
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
REBENCH_OUTDIR=$OUTDIR python3 - <<'PY'
import json, os, subprocess
from datetime import datetime, timezone
out = os.environ["REBENCH_OUTDIR"]
identity_path = out + "/run-identity.json"
if not os.path.exists(identity_path):
    raise SystemExit("missing required run-identity.json; refusing partial manifest")
with open(identity_path) as f:
    identity = json.load(f)
serve_cfg = {}
for key in ("PORT", "CTX", "SPLIT", "TSPLIT", "PARALLEL", "THREADS",
             "FLASH_ATTN", "CACHE_K", "CACHE_V", "BATCH", "UBATCH", "NO_MMAP",
             "MODEL_FILE"):
    serve_cfg[key] = os.environ.get(key, "default")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True)
if gpu.returncode != 0 or not gpu.stdout.strip():
    raise SystemExit("FATAL: nvidia-smi GPU metadata query failed; refusing to publish assumed GPU data")
gpu_meta = gpu.stdout.strip()
manifest = {
    "kind": "rebench-run",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": out,
    "identity": identity,
    "serve_config": serve_cfg,
    "serve_config_source": "launch environment; non-set keys record the serve.sh default",
    "serve_defaults_resolved": {
        "model_file": "GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf",
        "ctx": 16384,
        "split_mode": "layer",
        "tensor_split": "1,1,1,1",
        "parallel_slots": 1,
        "threads": 8,
        "flash_attn": "server default (auto)",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "batch": 2048,
        "ubatch": 512,
        "mmap": "on (--no-mmap statically infeasible: 146.05 GiB weights vs 94 GiB host RAM)",
        "gpu_layers": 999,
    },
    "model": {
        "checkpoint": "unsloth/GLM-5.3-Flash-GGUF UD-IQ4_XS",
        "file": serve_cfg.get("MODEL_FILE", "default"),
        "upstream_revision": "2975ab414d30340466d8c51533c6e91f0cca64c1",
        "total_bytes": 156822111075,
        "shards": 5,
        "quant": "UD-IQ4_XS (dynamic 4-bit k-quant GGUF)",
        "integrity": {
            "ran": os.environ.get("PREFLIGHT_HASH", "1") != "0",
            "method": "sha256sum -c SHA256SUMS in the model directory; preflight also fails if the staged SHA256SUMS differs from the committed results/expected-sha256.txt pin",
            "limitation": "SHA256SUMS is self-supplied by the staging process; the committed results/expected-sha256.txt pin is the upstream-fact cross-check",
        },
    },
    "runtime": {
        "server": "llama-server (unslothai/llama.cpp fork)",
        "pinned_revision": "00699716c275498ff84d71e329178fe21cba56a6",
        "build_flags": "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80",
        "binary_sha256": identity.get("runtime_binary_sha256", "unset"),
    },
    "environment": {
        "kernel": subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip() or "unknown",
        "os_release": subprocess.run(["sh", "-c", "head -2 /etc/os-release"], capture_output=True, text=True).stdout.replace(chr(10), ";").strip() or "unknown",
        "gpu_inventory": gpu_meta,
    },
    "evaluator": {
        "tool": "bench/eval.py + bench/measure.py (in-repo, deterministic)",
        "harness_revision": identity.get("harness_revision", "unknown"),
        "llm_judge": "none (exact-match / deterministic checks only)",
    },
    "protocol": {
        "speed_warmup_runs": 3,
        "speed_measured_runs_c1": 5,
        "ladder_concurrencies": [1, 2, 4],
        "ladder_reps_per_level": 2,
        "soak_seconds": 1200,
        "soak_concurrency": 2,
        "quality_tasks": 26,
        "seed": 42,
        "temperature": 0.0,
        "max_tokens": {
            "speed_warmup_c1_ladder_soak": 256,
            "quality_math": 512,
            "quality_instruction": 256,
            "quality_coding": 2048,
            "quality_longctx": 2048,
            "quality_heldout_math": 512,
            "quality_heldout_instruction": 256,
            "quality_heldout_code": 2048,
            "note": "derived from the evaluator source: math and held-out math use the eval.py default max_tokens=512; coding, longctx, and held-out code use 512+1536 reasoning margin = 2048; instruction and held-out instruction use 256",
        },
        "token_accounting": "speed/ladder/soak (measure.py): final usage object from streaming responses (stream_options include_usage=true), missing usage = failed sample; quality (eval.py): usage object from non-streaming responses",
    },
    "safety": {
        "core_limit_c": 80,
        "mem_limit_c": 85,
        "thermal_breaches": identity.get("thermal_breaches", "unverified"),
        "xid_check": "thermal-guard scans for Xid at start, during, and end; any Xid fails the run",
    },
    "artifacts": {},
}
for name, p in (("warmups","warmups.jsonl"),("speed_c1","speed-c1.jsonl"),
                ("ladder","ladder.jsonl"),("soak","soak.jsonl"),
                ("quality","quality.jsonl"),("experiments_csv","experiments.csv"),
                ("summary_csv","summary.csv"),("summary_json","summary.json"),
                ("gpu_snapshot","gpu-final.csv"),("run_identity","run-identity.json")):
    manifest["artifacts"][name] = out + "/" + p
manifest["artifacts"]["charts_dir"] = out + "/charts"
with open(out + "/run-manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print("run manifest written")
PY

echo "rebench complete: $OUTDIR"
