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
export LLAMA_SERVER=/path/to/llama-server
export MODEL_DIR=/path/to/shards
MODEL_DIR="$MODEL_DIR" LLAMA_SERVER="$LLAMA_SERVER" ./rebench.sh
```

`run-all.sh` is deprecated and must not be used for new evidence: it appends
into shared `results/*.jsonl` paths and predates the fresh-run, watched-driver,
and sanitized-manifest contract. Use `rebench.sh`, which creates one fresh
watched run directory per invocation.

Produces (inside `results/run-<id>/`):

- `experiments.csv` / `summary.csv` - derived row and summary
- `quality.jsonl` - per-task pass/fail + latency
- `run-manifest.json` - full reproducibility manifest (revisions, serve
  config, host, protocol, safety; private storage paths redacted)

Safety: abort at 80 °C core / 85 °C memory / any Xid. Stop the server when done.
