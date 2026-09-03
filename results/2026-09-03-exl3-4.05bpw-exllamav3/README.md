# Results — EXL3 4.05bpw / exllamav3 / TabbyAPI, 2026-09-03

Full narrative and evidence: [attempts/exl3-4.05bpw-exllamav3/README.md](../../attempts/exl3-4.05bpw-exllamav3/README.md).
This file collects the raw result tables for that attempt in one place.

## Run identity

- Checkpoint: `turboderp/GLM-5.3-Flash-exl3` @ branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`
- Weights: 165,151,555,504 bytes = 153.81 GiB, 31 manifest files (105 files on disk after re-layout)
- Runtime: exllamav3 1.4.6+cu128.torch2.10.0 via TabbyAPI
- Topology: `gpu_split: [48, 48, 48, 48]`, `tensor_parallel: false`
- Serving config: `max_seq_len: 32768`, `cache_size: 32768`, `cache_mode: Q8` (short-context only; see caveat), `reasoning: true`, drafting disabled

## Gates

| Gate | Result |
|---|---|
| `/v1/models` | 200 OK |
| Deterministic text (`max_tokens=32`, temp=0) | Insufficient (3/3): reasoning consumes full budget, `content: null`, `finish_reason: length` |
| Deterministic text (`max_tokens=128`, temp=0) | Passes: `content: "Paris"`, reasoning routed to `reasoning_content` |
| Golden corpus (20 prompts, `max_tokens=512`, keyword match) | 20/20 |

## Benchmark ladder (greedy, 400 completion tokens, 1 warmup + 3 measured reps)

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 26.9 | 26.9 |
| C2 | 31.1 | 15.6 |
| C4 | 41.7 | 10.5 |
| C8 | 44.8 | 8.3 |

Peak temperature through C8: 49 degrees C (GPU0). No OOM, no Xid/ECC events.

## Speculative decoding (n-gram), C1, 400 tokens, 3 reps

| Drafting | Tok/s |
|---|---:|
| Off | 26.9 |
| n-gram | 25.0 (-7.2%) |

No benefit found for this prompt shape; standing config ships with drafting disabled.

## Prefill / TTFT (2,941-token prompt, tokenizer-exact, FP16 cache for this measurement only)

| Condition | Prompt time | Throughput |
|---|---|---|
| Cold (first request after boot) | 5.57 s | — |
| Warm (reps 2-3, usage-reported mean) | 0.39 s | ~354 tok/s |

**Caveat (load-bearing):** `cache_mode: Q8` crashes any single request whose
context exceeds ~2,048 tokens on this runtime version, due to an explicit
`assert qc is None` guard against sparse DSA attention over a quantized MLA
cache in exllamav3 1.4.6's `mla_attn.py`. This measurement required
temporarily switching to `cache_mode: FP16`. See the attempt record for full
detail.

## Power (C4 load, 1s samples, 60s window)

| GPU | Mean W | Peak W |
|---|---:|---:|
| 0 | 65.4 | 120.0 |
| 1 | 63.0 | 101.4 |
| 2 | 56.2 | 97.4 |
| 3 | 50.1 | 78.9 |
| **Total** | **234.7** | **302.3** |

## Boot-attempt ladder (summary — full detail in the attempt record)

| # | `gpu_split` | Outcome |
|---|---|---|
| 1 | `[64, 64, 64, 64]` | Booted, OOM'd on first inference (no KV headroom, GPU3 unused) |
| 2 | `[40, 40, 40, 40]` | Immediate CUDA OOM in `touch_device`; coincided with NVRM driver corruption requiring a driver reload |
| 3 | `[40, 40, 40, 40]` (post-reload, Q8 cache) | Clean `RuntimeError: Insufficient VRAM in split for model and cache` at 46/50 modules loaded |
| 4 | `[48, 48, 48, 48]` | **Succeeded.** ~1.5-4 min boot from local NVMe, ~153 GiB total resident |
