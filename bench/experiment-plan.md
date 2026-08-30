# Decode-speed experiment plan — UD-IQ4_XS llama.cpp on the four-card CMP node

Pin: unslothai/llama.cpp @ `00699716c275498ff84d71e329178fe21cba56a6`
(verified: local clone HEAD, 2026-08-30). Point `LLAMA_SERVER` at the built
`llama-server` binary; the path is node-local and not part of the recipe.
Metric: median tok/s of 3 measured runs after 1 warmup, single stream, 256-token
generations, temperature 0, seed 42; tokens taken from the final usage object
(measure.py already does this). Quality gate: >=90% pass on eval.py before any
config is declared final.

## Preconditions (bench/preflight.sh, read-only)

1. No vLLM workload on the GPUs (if present: stop, report, wait — never kill).
2. >=36000 MiB free VRAM per card (weights are ~36.5 GiB/card at an even 4-way
   split of the measured 146.05 GiB total, plus context/KV headroom).
3. 5/5 GGUF shards staged, each >1 GiB.
4. Port 8199 free, binary executable.

## Factor order (one factor per experiment; keep if median improves, else revert)

| # | Factor | Variants | Rationale / prior expectation |
|---|--------|----------|-------------------------------|
| 0 | Baseline | serve.sh defaults: mmap, layer split, tensor-split 1,1,1,1 (equal ratios across four GPUs, documented default), threads 8, ctx 16384, parallel 1, flash-attn per binary default (read from server startup log — record it) | reference point; also fixes the exact prompt-token count from the first usage object (prompt-1k.txt is ~1K tokens by construction, untokenized — do not claim an exact count) |
| 1 | --split-mode | row vs layer | MoE decode is weight-bandwidth-bound; row (tensor) split reads each weight once across cards instead of keeping whole layers per card. Community-reported as often faster for MoE when per-layer weights exceed per-card bandwidth benefit; label community-reported until measured here |
| 2 | --tensor-split | 1,1,1,1 (default) vs proportional-to-free-VRAM | cards are identical 64 GiB SKUs, so equal ratios are expected near-optimal; cheap A/B |
| 3 | --flash-attn | on vs off | affects KV-path FLOPs and, for some quantized KV types, is required; decode effect expected small at 16K ctx but measure |
| 4 | --cache-type-k / -v | f16 vs q8_0 (then q4_0) | KV is small for this MLA-style architecture (est. 2–4 GiB at 16K, untested); expect minor effect. If the fork rejects quantized KV for this arch, record as untested with the server's error text |
| 5 | --threads | 4, 8, 12, 16 | decode is GPU-bound with all layers offloaded; expect flat. Cheap sweep |
| 6 | --batch-size / --ubatch-size | 128, 256, 512 | prefill-side knobs; single-stream decode tok/s expected flat. Lower priority |
| 7 | --parallel slots | 2, 4 at fixed aggregate rate | task metric is single-stream median; parallel>1 splits KV pool and typically lowers per-stream tok/s. Keep only if the single-stream median does not regress |
| — | --no-mmap | (skipped) | statically infeasible: 146.05 GiB weights (measured HF blob sum) vs 94 GiB host RAM (measured free -h). Inferred, not measured; serve.sh keeps NO_MMAP=1 opt-out |

## Loop discipline

1. `./preflight.sh` must PASS before every server start.
2. Restart server with exactly one knob changed; wait for /health.
3. `python3 bench/measure.py --base http://127.0.0.1:8199 --prompt "$(cat bench/prompt-1k.txt)" --max-tokens 256 --concurrency 1 --runs 3 --out results/speed.jsonl`
4. Append one CSV row per experiment to `results/experiments.csv` (header fixed).
5. Thermal stop: 80 °C core, 85 °C memory, any Xid, GPU disappearance → stop,
   preserve logs, report. CMP 170HX cards idle at ~31 °C here, but the cards are
   passively cooled — confirm forced airflow before load.
6. Every 30 min: report current best config + median tok/s. Never claim a global
   optimum.

## Decision rule

- median tok/s improves → keep; ties (<1% delta) → revert to the simpler config.
- Any run with `ok: false`, a nonzero error list, or a missing final usage object
  → discard the run, note it, re-measure once; two failures → record as failed
  config with the error text.

## Methodology note (autoresearch adaptation)

This loop follows the [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
design (README @ HEAD, checked 2026-08-30): a human/agent-written program file
(a controller-local program file driving the Pi worker, mirroring autoresearch's
`program.md`), a fixed per-experiment protocol budget (here 1 warmup + 3
measured 256-token runs instead of its fixed 5-minute train budget, so configs
stay comparable), a single primary metric (median single-stream tok/s, the
analog of its val_bpb) gated by a hard quality floor (>=90% on eval.py) the
original does not need, and an append-only experiment log (`results/experiments.csv`)
with keep/revert decisions. Deviations from the original: our per-run cost is
minutes not seconds, so the sweep is bounded to the factors above rather than
open-ended code mutation, and every claim is hardware-local (same non-transferability
caveat autoresearch documents).
