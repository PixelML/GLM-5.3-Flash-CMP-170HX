# PP4 + native MTP k=3, vLLM sm80 — 4x CMP 170HX, 2026-09-05

The evidence set behind the **recipe of record** for GLM-5.3-Flash on a
four-card CMP 170HX node. Supersedes the 2026-09-03 TP4 result set in
`../2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/`, which is kept as history.

Club-side publication:
[club-170hx notebook](https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-09-05-glm-5.3-flash-4card-pp4-vllm.ipynb),
[guide page](https://github.com/PixelML/club-170hx/blob/main/docs/models/glm-5.3-flash.md).

## Run manifest

See [`run-manifest.json`](run-manifest.json) for the machine-readable version.

| Field | Value |
|---|---|
| Date | 2026-09-05 |
| Hardware | four-card CMP 170HX node, SM80, 64 GiB HBM2e per card, 180 W per-card cap, no NVLink, no P2P over PCIe |
| Checkpoint | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` @ `abd7b07719111f137e1de8a0c1b7e01c11b74d1a` — AWQ W4A16 (compressed-tensors), 24 files, 190,843,146,533 bytes, byte-verified |
| Image | `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905` |
| Image index digest | `sha256:62f612b49614523e6a46e1493d35d3efd1f363917129d38cc923a31053693bfb` |
| Runtime source | `PixelML/sm80vllm` branch `pp-dflash2/glm53-flash-487ecf187-20260905` — orphan overlay over `vllm/vllm-openai:glm53-flash` @ `487ecf187` plus 24 patches ported from [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) (Apache-2.0), each carrying an attribution trailer |
| Recipe script | `recipes/glm53-flash-4x170hx-pp4.sh` on that branch |
| Topology | PP4, `VLLM_PP_LAYER_PARTITION=14,12,12,7` — 45 hidden layers, sparse-MLA counts 3/3/3/2 per stage |
| Speculation | native MTP, `num_speculative_tokens=3` |
| Context | `--max-model-len 393216`; KV pool 1,194,627 tokens (3.04x a single max-length request) |
| Other flags | `--no-enable-prefix-caching`, `--gpu-memory-utilization 0.90`, `--max-num-seqs 8`, `--max-num-batched-tokens 4096`, `VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2`, `VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256`, `--limit-mm-per-prompt image:0,video:0` |
| Boot | 5/5 boots served for this recipe: 995, 1,029, 1,077, 1,263, 1,140 s to `Application startup complete`, plus 2/2 on the earlier port run; engine init 320.4 s on the measured boot |
| Memory after load | 45.45 GiB per pipeline stage; idle memory 51.6-54.2 GiB per card of 64 GiB |
| Device health | 4/4 cards at the expected PCI revision before and after every boot; zero Xid, zero ECC. Power cap 180 W verified on all four throughout |
| Temperature under load | untested — not sampled on this boot |

## Link state during measurement — read this before quoting any aggregate

Measured on a **degraded link**: one card trained at PCIe Gen1 x1 against a slot
ceiling of x8, one at x8, two at x16, no NVLink on any of them.

- **Aggregate, prefill and TTFT cells are lower bounds.** Do not compare them to
  a healthy-box number.
- **c=1 decode is link-insensitive under PP4** — one hidden-state hop of roughly
  50 KB per decode step — and carries the headline.

This is also why the topology changed. Tensor parallelism runs at the width of
its worst rank on a fabric with no NVLink: the identical TP4 recipe measured
70.5 tok/s before the link retrained and 14.4 tok/s after, under an unchanged
protocol, while PP4 held 60.8 tok/s on the same box. No decode number measured
across that boundary can be used to accept or reject a code change.

## Protocols

| | P1 | P2 |
|---|---|---|
| Harness | `scripts/bench.py` from [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8), used unmodified | this repository's `bench_glm53.py`, unchanged since the TP4 record |
| Sampling | temperature 0 | temperature 0.7, `ignore_eos` |
| Output tokens | 512 | 512 |
| Repetitions | 3, median | 5, median, first repetition cold |
| Workloads | code, json, counting, math, prose, plus a repetition diagnostic | one fixed long-form prompt |
| Degeneracy guard | the author's repeat guard flags repetition-collapsed completions; flagged cells are inflated and are never read as throughput | none |

Generated-token counts come from the final `usage` object of each response,
never from counting stream events. Acceptance counts come from the engine's
`vllm:spec_decode_*` counters differenced across each measurement window — exact
counts, not gauges.

## Results

### Draft-depth sweep, c=1, one boot per depth

| | k=2 | **k=3** | k=5 | k=7 |
|---|---:|---:|---:|---:|
| P1 counting | 81.89 | 75.69 | 63.73 | 53.40 |
| P1 json | 84.08 *(deg)* | 87.02 | 82.10 | 78.16 *(deg)* |
| P1 code | 60.52 | 58.37 *(deg)* | 40.65 | 34.64 |
| P1 math | 80.88 | **87.55** | 72.52 | 91.44 |
| P1 prose | 60.73 | 60.54 | 45.74 | 47.30 |
| P1 repetition *(diagnostic)* | 74.32 | 80.21 | 77.36 | 78.42 *(deg)* |
| **P1 headline, clean cells only** | 81.89 counting | **87.55 math** | 82.10 json | 91.44 math |
| **P2 median (5 reps)** | 56.54 | **67.91** | 51.26 | 38.13 |
| P2 peak / cold rep | 66.53 / 51.95 | 92.58 / 56.32 | 53.56 / 52.13 | 58.81 / 36.67 |
| Drafts | 7,352 | 3,921 | 5,422 | 5,106 |
| Draft tokens | 14,704 | 11,763 | 27,110 | 35,742 |
| Accepted tokens | 10,885 | 7,690 | 12,641 | 13,321 |
| Draft acceptance rate | 74.0% | 65.4% | 46.6% | 37.3% |
| Mean accepted length | 2.48 | 2.96 | 3.33 | 3.61 |

All values measured. `(deg)` marks a cell the repeat guard flagged.

**Reading.** Per-verified-token cost on this box is not flat, so depth does not
pay for itself. Acceptance rate halves from k=2 to k=7 while mean accepted
length rises only 1.46x, and each extra depth is another sequential MTP forward.
k=7 wins math alone and loses code, counting and prose; its P2 median is 38.13
against k=3's 67.91. **k=3 is the record. Acceptance, not depth, is the lever.**

### Decode vs context, c=1, k=3

512 output tokens, 3 repetitions plus a cold/warm pair per point, two thinking
arms alternated within one boot.

| Prompt tokens (actual) | Prompt tok/s † | Generation tok/s | ms/token | Cold / warm TTFT s † |
|---:|---:|---:|---:|---|
| 336 | 592.2 | 98.58 | 10.14 | 0.52 / 0.52 |
| 888 | 912.2 | 98.74 | 10.13 | 0.98 / 0.96 |
| 2,024 | 1,027.9 | 81.51 | 12.27 | 1.98 / 1.96 |
| 3,968 | 1,108.4 | 75.34 | 13.27 | 3.57 / 3.57 |
| 8,042 | 1,462.0 | 74.85 | 13.36 | 5.50 / 5.50 |
| 16,095 | 1,744.3 | 78.89 | 12.68 | 9.24 / 9.21 |
| 32,986 | 1,912.5 | 76.61 | 13.05 | 17.26 / 17.25 |
| 66,023 | 2,007.0 | 77.44 | 12.91 | 32.90 / 32.89 |
| 131,042 | 2,038.4 | 78.56 | 12.73 | 64.32 / 64.29 |
| 258,000 (target) | untested | untested | untested | untested — prompt calibration overshot the 393,216-token limit |

† lower bound, degraded link.

Decode is flat at 74.9-81.5 tok/s from 2,024 to 131,042 prompt tokens. Warm TTFT
equals cold TTFT at every length because the recipe runs with prefix caching
off. The two thinking arms are statistically identical because **the switch is
not switchable on this checkpoint** (`receipts/k3/thinking_probe.json`: no
`reasoning_content` under `enable_thinking` or `thinking`, true or false, or the
server default). They are one configuration measured twice, not an A/B, and are
reported as a single curve.

### Concurrency, k=3

4,096-token prompts, 256 output tokens, P2 sampling, uncached. **Every row is
link-bound.**

| Concurrency | Aggregate tok/s | Per-stream median tok/s | e2e p50 s | e2e p95 s | Success |
|---:|---:|---:|---:|---:|---|
| 1 | 30.21 | 30.22 | 8.47 | 8.47 | 1/1 |
| 2 | 49.80 | 25.04 | 10.28 | 10.28 | 2/2 |
| 4 | 44.38 | 11.12 | 23.05 | 23.07 | 4/4 |
| 8 | 75.51 | 9.48 | 27.03 | 27.11 | 8/8 |
| 16 | 78.36 | 7.02 | 51.97 | 52.19 | 16/16 |

### Prefill and TTFT, k=3

Uncached prefill with one output token; warm streaming TTFT with 32 output
tokens. **Link-bound.**

| Prompt tokens | Prefill tok/s | Prefill wall s | Warm streaming TTFT s | Reps |
|---:|---:|---:|---:|---:|
| 4,096 | 1,128.5 | 3.63 | 3.67 | 3 |
| 16,384 | 1,751.8 | 9.35 | 9.44 | 3 |

### Gates

| Gate | Result |
|---|---|
| `/v1/models` reachable | yes |
| Deterministic greedy repeat, 3x one token | PASS, all identical |
| Greedy sanity prompts, 64 tokens each | 3/3 correct and clean |
| Clean text across the P1 battery at k=3 | clean except the flagged cells above |

## Negative results

### DFlash2 k=7 on the AWQ W4A16 checkpoint — net loss

Receipt: `receipts/awq-dflash7/`. Drafter `incoai/GLM-5.3-Flash-DFlash2`
(cc-by-nc-nd-4.0), downloaded for measurement only, not redistributed.

| | MTP k=3 | DFlash2 k=7 on AWQ |
|---|---:|---:|
| P1 code (clean both) | 58.37 | 36.21 |
| P1 prose (clean both) | 60.54 | 32.45 |
| P1 counting | 75.69 | 129.71 |
| P1 json | 87.02 | 136.54 *(deg)* |
| P1 math | 87.55 | 110.81 |
| Draft acceptance rate | 65.4% | 41.6% |
| Mean accepted length | 2.96 | 3.91 |
| KV pool at 393,216 max len | 1,194,627 tokens (3.04x) | 523,657 tokens (1.33x) |
| Clean text on the code workload | yes | no — repeated, broken think tags |

This drafter is a large win on the upstream author's NVFP4 checkpoint and a loss
on ours. **This — not the card count, and not PP4 — is the main reason the
upstream headline figure does not reproduce here.** The decisive follow-up, the
same drafter on an NVFP4 checkpoint, separates drafter quality from
drafter/checkpoint mismatch and is untested (pending).

### PP4 before the patch set was ported

| Configuration | c=1 decode | Text |
|---|---:|---|
| PP4 + MTP k=5, stock build | 3.35 tok/s | degenerate, word-level repetition |
| PP4, speculation off, stock build | 6.11 tok/s (median of 3) | clean, 3/3 facts correct |
| PP4 + MTP k=3, ported patch set | 60.8 tok/s (P2) | clean |

Two separate faults: the degeneration was an MTP-under-PP artifact (the draft
head loaded random-init), and the throughput collapse was the base pipeline
hand-off on the stock build. The port fixes both. The first row is not an MTP
verdict.

### Block-drafter configurations that never reached a measurement

| Attempt | Outcome |
|---|---|
| PP + DFlash2, stock build | Refused at init: the drafter's auxiliary hidden-state layers (5, 14, 24, 33, 42 of 45) cannot all live on the last pipeline stage under any genuine four-way split, and the aux relay resolves layer names without forwarding hidden states across stages. Needs a real cross-stage relay upstream. Failed before any GPU memory allocation. |
| TP + DFlash2, stock build | Refused at KV-cache setup, identically with and without prefix caching: page size is not divisible by the maximum page size and cannot be padded for MLA attention layers. Prefix caching is refuted as the trigger. Needs padding support for MLA layers upstream. |

### NVFP4 checkpoint — not claim-ready

`LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `caca4e6a4ebb` (MIT) was attempted on
2026-09-05 and **did not boot**: out of memory during the mixture-of-experts
kernel-format conversion. No throughput, quality or acceptance number exists for
it. This blocks the decisive drafter test — the community block drafter on the
checkpoint it was tuned against — so the drafter/checkpoint-mismatch hypothesis
in the negative cell above remains a hypothesis.

### Thinking on vs off — not switchable on this checkpoint

Receipt: `receipts/k3/thinking_probe.json`. The probe sent
`chat_template_kwargs` with `enable_thinking` and with `thinking`, true and
false, and also sent no kwarg at all. **No case returned `reasoning_content`**;
every case returned a normal answer in `content`. There is no thinking toggle to
measure on this checkpoint, and the two arms of the context sweep are therefore
the same configuration measured twice.

### Do not combine autotune-off with MTP on the TP4 build

`--no-enable-flashinfer-autotune` together with MTP crashes at engine startup
with a CUDA launch failure, reproduced on a clean boot, and wedges a card at the
PCIe level. Autotune at its default is part of the measured recipe.

## Untested (pending), with reasons

| Cell | Reason |
|---|---|
| Lossless check, greedy, speculation on vs off, 20 fixed prompts | harness written; blocked by the node fault below |
| Sustained stability, 3 rounds of c=8 with health checks | harness written; blocked by the node fault below |
| Quality battery per bucket | harness and datasets staged; blocked by the node fault below |
| Power and temperature under load | not sampled on the measured boot |
| Accepted tokens per pass vs context length | no `SpecDecoding metrics` interval line fell inside a sample window during the sweep |
| 258k-token context point | the request exceeded the 393,216-token server limit after prompt calibration |
| BF16 parity | no host in this pool can load the BF16 checkpoint; quantization quality is compared only against the vendor's published numbers and the other quantizations measured here |
| Vision path | the recipe disables multimodal profiling as a node workaround |

## Receipt map

| Path | Cell |
|---|---|
| `receipts/cells.json` | sanity prompts and the k=3 gate battery |
| `receipts/k3/gate.json` | deterministic greedy repeat |
| `receipts/{k2,k3,k5,k7}/p1.json` | P1 per-workload decode, one boot per depth |
| `receipts/{k2,k3,k5,k7}/decode_c1.json` | P2 c=1 decode, one boot per depth |
| `receipts/accept_{before,after}.json` | `spec_decode` counters for the **k=3** window |
| `receipts/{k2,k5,k7}/accept_{before,after}.json` | the same counters for the other depths |
| `receipts/k3/sweep/context_sweep.json` | decode vs context, 327 to 131k tokens, both thinking arms, every raw sample |
| `receipts/k3/conc_sweep.json` | concurrency c=1,2,4,8,16 at 4k prompt |
| `receipts/k3/prefill_{4096,16384}/{prefill,ttft}.json` | uncached prefill and warm streaming TTFT |
| `receipts/k3/thinking_probe.json` | thinking-switch probe |
| `receipts/awq-dflash7/` | the DFlash2-on-AWQ negative cell |
| `commands.md` | the redacted commands that produced all of the above |

Every JSON is verbatim harness output with endpoints, filesystem paths, IP
literals and container names replaced by `<endpoint>`, `<path>`, `<addr>` and
`<container>`. Numbers, prompts and model text are unmodified.

## Licenses and attribution

| Source | License | Use |
|---|---|---|
| [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) | Apache-2.0 | 24 pipeline-parallel patches applied with per-patch attribution trailers; his `scripts/bench.py` and repeat guard are the P1 protocol, used unmodified; the KV-balancing rule behind the layer partition is his, the four-stage split is our adaptation. His published throughput figures are **community-reported** and are not mixed into the measured tables above. |
| [wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) | per the model card | The checkpoint, used as published at the pinned revision. Third-party verified, not re-quantized, not mirrored. The SM80 vLLM enablement this image lineage descends from is also wtdcode's. |
| [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | cc-by-nc-nd-4.0 | Block drafter, measurement only, not redistributed. |
| Base model | per the GLM-5.3-Flash model card | Served, not redistributed. |
