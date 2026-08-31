# Benchmark harness

One-command entry point for the UD-IQ4_XS llama.cpp attempt.

## Prereqs

1. `llama-server` built from unslothai/llama.cpp @ `00699716c275498ff84d71e329178fe21cba56a6` with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`.
2. 5 GGUF shards staged in a directory (not committed here).
3. Python deps: Python 3.10+ and `aiohttp` (`pip install aiohttp` or
   `uv pip install aiohttp`). `measure.py` imports it; `eval.py` uses only
   the standard library.

## Run

```bash
MODEL_DIR=/path/to/shards LLAMA_SERVER=/path/to/llama-server \
  bench/rebench.sh
```

Note: `run-all.sh` is a deprecated fail-closed shim that delegates to `bench/rebench.sh`

Produces one timestamped directory `results/run-<UTC timestamp>/`:

- `warmups.jsonl`, `speed-c1.jsonl`, `ladder.jsonl`, `soak.jsonl` — raw timing data
- `quality.jsonl` — per-task pass/fail + latency
- `experiments.csv`, `summary.csv`, `summary.json` — derived rows
- `charts/` — generated SVGs
- `gpu-final.csv` — end-of-run GPU snapshot
- `run-identity.json` — runtime binary fingerprint, harness revision,
  token-accounting endpoint, live power limits, thermal-watch status
- `run-manifest.json` — full sanitized run manifest (identity +
  serve config + GPU/protocol/safety metadata + artifact paths

Safety: abort at 80 °C core / 85 °C memory / any Xid. Stop the server when done.
