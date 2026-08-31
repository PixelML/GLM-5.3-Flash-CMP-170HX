# vLLM-native W4A16 lane — GLM-5.3-Flash on 4x CMP 170HX (SM80)

All results, candidates, and receipts for this lane live in this canonical repository. Nothing is duplicated in a runtime-specific repository.

## TL;DR (reporting contract)

**Current verdict — UNTESTED:** no vLLM-native candidate in this lane has been measured on the 4x CMP 170HX system yet.

| Candidate (pinned revision) | Result status | Aggregate input tok/s (benchmark wall time) | Aggregate output tok/s (benchmark wall time) | Per-request output tok/s (end-to-end, incl. prefill) | TTFT s p50/p95 (request-send to first streamed output) | ITL ms p50/p95 (gaps between streamed outputs) | Request success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [wtdcode GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16/tree/abd7b07719111f137e1de8a0c1b7e01c11b74d1a) @ `abd7b077` | UNTESTED | Pending | Pending | Pending | Pending | Pending | Pending |
| [cyankiwi GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4/tree/3999f9bf2c3e3064790af5a2d12d19090fc97f4d) @ `3999f9bf` | UNTESTED — deprioritized | Pending | Pending | Pending | Pending | Pending | Pending |
| PixelML mixed W4A16-expert / W8A16-dense | UNTESTED — not built | Pending | Pending | Pending | Pending | Pending | Pending |

Row aggregation semantics: each row reports the median across measured repetitions; each repetition aggregates across all requests over benchmark wall time. Per-request output tok/s is end-to-end (includes prefill), matching the vLLM per-request metric contract; pure-decode rate is reported separately via generation time / TPOT / ITL-derived throughput when isolated phase measurement is run.

### Metric definitions

- **Aggregate input/output tok/s** = total input/output tokens / benchmark wall time across all requests (service throughput, includes scheduling).
- **Per-request output tok/s** = output tokens / per-request end-to-end latency (includes prefill), per the [vLLM per-request metric definitions](https://github.com/vllm-project/vllm/blob/7292ee2791985ff3afe38850fb34125ad0853933/docs/features/per_request_metrics.md).
- **TTFT** = request-send to first streamed output token; **ITL** = gaps between streamed outputs, per the [vLLM benchmark definitions](https://github.com/vllm-project/vllm/blob/7292ee2791985ff3afe38850fb34125ad0853933/docs/benchmarking/cli.md).
- Token counts come from the runtime final usage object. Cold vs warm, repetitions, and output length are recorded per result.

### Prior baseline context (different runtime/protocol — not a vLLM result)

INFERRED label applies to all interpretation below; the numbers are MEASURED in the llama.cpp UD-IQ4_XS lane:

- Five-repetition C1 protocol: 17.73 completion tokens per end-to-end request second ([speed-c1.jsonl at 84b62bf](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/84b62bf0b21d6590e34cba9d87df3b8d5c5a84fb/results/phase63/speed-c1.jsonl)).
- Separate two-repetition concurrency ladder: C1 14.99/17.31, C2 17.50/17.53, C4 17.73/17.69 aggregate output tok/s ([ladder.jsonl at 84b62bf](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/84b62bf0b21d6590e34cba9d87df3b8d5c5a84fb/results/phase63/ladder.jsonl)). These are two different protocols with different sample counts and are not merged.
- INFERRED hypothesis (not established): the flat C2/C4 service throughput is consistent with a compute-bound dequant-kernel ceiling, but this measurement alone cannot distinguish dequant compute from memory bandwidth, scheduling, or host/PCIe bottlenecks. The vLLM compressed-tensors W4A16 candidate is the highest-leverage first test precisely because it changes the kernel class; its Ampere-kernel rationale is INFERRED until the executing kernel is confirmed on this topology.

### Runbook additions (from the SM80 transfer audit)

- PWRBRK#/B30 hardware-brake check before any run (verified clean on this node''s four cards).
- Never tree-kill a multi-GPU vLLM process (stranded CUDA contexts require host intervention).
- `--block-size 64` and `VLLM_USE_V2_MODEL_RUNNER=1` for MTP-under-PP paths.
- Quant selection rule: symmetric quantization with an unquantized MTP head is the validated pattern; asymmetric AWQ (candidate 2) is deprioritized unless probe evidence changes the picture.
- PP-first topology; mem-util 0.85 with a drafter; never `--enforce-eager` with a drafter.

### Candidate fit and runtime facts (ported from the superseded duplicate repository)

Externally reported sizes and inferred per-card fits, retained so the duplicate repo can be archived without losing source-labeled facts. None are measured on CMP.

- Candidate 1 (wtdcode AWQ W4A16 at rev abd7b077): total 190.81 GB / 177.71 GiB (externally reported); ~44.43 GiB weights/card ideal TP4 (inferred). Planned runtime wtdcode/vllm-backport v0.11.0 at commit 6048dbd07209a8fa8b24aeb761652af7143e31d8 (SM80 images; publisher reports GLM testing on 8xA6000 TP8 - externally reported). Our 4x CMP SM80 topology: UNTESTED.
- Candidate 2 (cyankiwi AWQ INT4 at rev 3999f9bf): 212,721,952,636 bytes = 198.11 GiB (externally reported); ~49.53 GiB/card ideal TP4 (inferred). Fit candidate only; serving not proven on CMP.
- Candidate 3 (PixelML mixed W4A16-expert / W8A16-dense): static fit in [PR #8](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/pull/8) - planned artifact 170,526,899,714 B = 170.5 GB = 158.8 GiB = 1.087x the GGUF baseline; runtime gate: vLLM PR 53906 open/merge-blocked (SM90/100/120, PP gated off), TP4 plausible/untested, custom all-reduce must be disabled on 4x PCIe without P2P. Supporting precedents: vLLM PR 48918 (CT WNA16 MoE, merged), llm-compressor #2940 (mixed-precision MoE).
- Order of operations: wait out the GPU work gate; re-check 53906 merge state at pin time; candidates 1 -> 2 -> 3, one at a time through the full gate chain, advancing only after merge-or-close of the current one.

### Evidence links (populated as this lane produces results)

Immutable pins land here: run manifest, raw C1 and ladder JSONL, and methodology at the exact evidence commit. Mutable links (open PRs, issue comments) are not used as evidence.
