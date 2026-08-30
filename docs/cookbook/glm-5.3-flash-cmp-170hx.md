# GLM-5.3-Flash on four-card CMP 170HX — recipe page (research preview)

> **Validation:** RESEARCH PREVIEW — no measured results yet; this page is the
> publication skeleton required before any claim moves in. Every number below is
> pending until the benchmark run completes. **Updated:** 2026-08-30 ·
> **Evidence:** results/ (this repository; immutable revision linked at publication)

## What will work (candidate, unverified)

 Serving [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
UD-IQ4_XS (146.05 GiB, MIT) on four 64 GiB CMP 170HX cards via the
[unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) DSA fork — the
first pairing where the quant fits (~48.7 GiB/card at an even split) and the
runtime builds for SM80. Static-fit labels are inferred; every measured value
lands here after the run.

## Choose a recipe

| Goal | Artifact | Runtime | Topology | Validation | Recipe |
| --- | --- | --- | --- | --- | --- |
| Max speed | UD-IQ4_XS GGUF | llama.cpp DSA fork | 4-card split (mode TBD) | untested | pending measurement |
| Balanced | UD-IQ4_XS GGUF | llama.cpp DSA fork | TBD | untested | pending measurement |
| Max quality | UD-IQ4_XS GGUF | llama.cpp DSA fork | TBD | untested | pending measurement |

## Run it (candidate baseline, unverified)

```bash
git clone https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX
cd GLM-5.3-Flash-CMP-170HX/bench
NAME=baseline LLAMA_SERVER=/path/to/llama-server MODEL_DIR=/path/to/staged-shards ./run-experiment.sh
```

Expected ready signal: `ok: 5/5 shards staged` then `preflight: PASS`, then
`healthy after ~Ns` in the runner log.

## Why these settings

Explained in [bench/experiment-plan.md](../bench/experiment-plan.md)
(factor order and rationale) and
[attempts/gguf-ud-iq4xs-llamacpp/README.md](../attempts/gguf-ud-iq4xs-llamacpp/README.md)
(static fit, mmap arithmetic).
Flags are added one at a time; nothing above is asserted to be optimal.

## Results

First measured baseline (phase C, concurrency 1, 400-token prompt /
256-token completion, c=1 median of 5): 17.73 tok/s aggregate decode,
14.44 s end-to-end per task. Aggregate throughput is ~flat across the
1/2/4 concurrency ladder (compute-bound); a 20-minute soak at c=2 held
17.5-17.7 tok/s with no throttling.

Corrected quality index 0.783 (mean bucket rate over math, instruction,
coding, long-context; small-sample gate, not a leaderboard). Two harness
defects were found and fixed during this phase — a sandbox wrapper that
dropped the candidate file from coding tasks, and completion budgets that
starved reasoning models before any answer bytes were produced. Raw JSONL,
the corrected pack, and the defect log are in results/phase63/ (README.md
inside documents the defects and the corrected per-bucket scores).

Charts (quality-vs-cost with GPU-only energy lower bound and disclosed
hypothetical price band, quality-vs-time, quality-vs-throughput,
concurrency-vs-throughput) are regenerated from results/summary.csv by
bench/charts.py into results/phase63/charts/.

## What failed (so far)

Five earlier attempts are recorded with evidence:
[NVFP4+vLLM](../attempts/nvfp4-vllm-sm121/README.md) (no SM80 runtime),
[EXL3/TR3 4bpw](../attempts/exl3-tr3-4bpw-exllamav3/README.md) (no SM80 ExLlamaV3),
[AWQ-INT4](../attempts/awq-int4-vllm/README.md) (does not fit),
[FP8/BF16](../attempts/fp8-bf16-reference/README.md) (does not fit),
and [EXL3 3.0bpw](../attempts/exl3-3bpw-0xsero/README.md) (quantization
artifact with no serving runtime). These shaped the current candidate.

## Reproduce the evaluation

Pins to link at publication: model revision `2975ab414d30340466d8c51533c6e91f0cca64c1`,
runtime `00699716c275498ff84d71e329178fe21cba56a6`, harness = this repo\'s bench/
at the published revision, redacted raw JSONL + tidy CSV + chart source in
results/.

## Limits

Single node, four-card only; small-sample quality gate (not a leaderboard);
CMP 170HX is a compute accelerator with passive cooling — results may not
transfer to graphics cards; all quality scores are local and task-specific;
GGUF is third-party-verified (tested in place, not mirrored or modified).

## Artifacts and evidence

- Upstream artifact: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (third-party; tested revision pinned above)
- Detailed GitHub repository: this repository
- Hardware club guidance: pending (two-PR rule; opens after measured results)
- Release manifest: releases/manifest.draft.json (draft, no-claim-ready)
