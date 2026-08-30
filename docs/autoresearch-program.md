# Autoresearch-style program — GLM-5.3-Flash decode optimization on CMP 170HX

Adapted from the [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
"program.md" pattern: a fixed protocol file drives an autonomous agent loop with
one primary metric, one factor change per experiment, and keep/revert discipline.
Paths are parameterized; fill the two environment values before starting.

## Current state (READ THIS FIRST)

The four-card node may be occupied by an **unrelated live serving workload**.
You MUST NOT kill, signal, or interfere with any process you did not start, and
you must not load your model while another workload holds the cards. First job:
WAIT — poll GPU memory every 10 minutes; when all 4 GPUs report <2 GiB used,
run bench/preflight.sh and proceed only on PASS.

## Objective

Maximize sustained single-stream decode throughput (256-token generations,
temperature 0, seed 42) for GLM-5.3-Flash GGUF UD-IQ4_XS on 4x CMP 170HX
(SM80, 64 GiB each). Metric: median tok/s over 3 measured runs after 1 warmup,
tokens from the final usage object. Quality floor: >=90% pass on
bench/eval.py tuning buckets. Safety: stop at 80 C core / 85 C memory / any
Xid / GPU loss.

## Allowed changes (exactly ONE factor per experiment)

1. --split-mode (layer / row)
2. --tensor-split ratio
3. --threads (4, 8, 12, 16)
4. --batch-size / --ubatch-size (128, 256, 512)
5. --cache-type-k / --cache-type-v (f16, q8_0, q4_0)
6. --flash-attn (on / off)
7. --parallel slots (1, 2, 4)
8. mmap vs --no-mmap ONLY if host RAM provably exceeds total weights (static
   arithmetic in the attempt README says it does not on the reference node)

## Fixed (never change in the loop)

- Checkpoint, quantization, runtime fork and pin (see
  [attempts/gguf-ud-iq4xs-llamacpp/README.md](../attempts/gguf-ud-iq4xs-llamacpp/README.md))
- Hardware, power limits, clocks, drivers, storage layout
- Measurement protocol: 1 warmup + 3 runs, 256 tokens, final-usage accounting
- Quality gate: bench/eval.py buckets before any config is declared final

## Loop

1. WAIT for GPUs (above). Verify staging: 5 GGUF shards, exact byte total from
   the attempt README, checksummed against results/expected-sha256.txt.
2. Pick ONE factor (experiment-plan.md factor order).
3. NAME=<factor> LLAMA_SERVER=<path> MODEL_DIR=<path> bench/run-experiment.sh
   (serves, waits /health, thermal-guards, 1 warmup + 3 runs, appends CSV).
4. Keep on median improvement >=1%; ties revert to the simpler config.
5. Repeat. Never declare a global optimum. Report best config + trace every
   30 minutes of active experiments.
6. When the sweep concludes: score ONLY the final config on the heldout bucket,
   run bench/summarize.py and bench/charts.py, update the attempt README with
   measured numbers, and move the release manifest forward with redacted evidence.

## Stopping

- Thermal/Xid/GPU-loss/staging-corruption: stop immediately, preserve evidence, report.
- Foreign workload appears mid-run: stop the server you started, save evidence, wait.

## Report format

Best-config table, tok/s trace per experiment, failed configs with error text,
thermal envelope, and the quality floor result for the final config.
