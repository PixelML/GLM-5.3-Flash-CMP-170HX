# Results — EXL3 4.05bpw / exllamav3 / TabbyAPI, 2026-09-02 (250 W) / 2026-09-03 (180 W re-measure)

Full narrative and evidence: [attempts/exl3-4.05bpw-exllamav3/README.md](../../attempts/exl3-4.05bpw-exllamav3/README.md).
This file collects the raw result tables for that attempt in one place.

**Power cap correction (2026-09-03).** The tables below were originally
measured 2026-09-02 at the vBIOS default 250 W by accident. A same-server,
no-restart re-measure on 2026-09-03 at the verified 180 W club-standard
cap found no consistent throughput difference outside run-to-run noise.
**180 W is canonical**; the 250 W tables below are retained for
comparison, with the 180 W tables added alongside them.

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

**180 W (2026-09-03, verified cap, canonical):**

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 25.2 | 25.2 |
| C2 | 35.3 | 18.0 |
| C4 | 43.2 | 11.0 |
| C8 | 44.6 | 8.2 |

**250 W (2026-09-02, accidental default, retained for comparison):**

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 26.9 | 26.9 |
| C2 | 31.1 | 15.6 |
| C4 | 41.7 | 10.5 |
| C8 | 44.8 | 8.3 |

Delta (180 W vs 250 W): C1 -6.3%, C2 +13.5%, C4 +3.6%, C8 -0.4% — read as
run-to-run noise, not a directional power effect.

Peak temperature through C8: 51 degrees C (180 W) / 49 degrees C (250 W).
No OOM, no Xid/ECC events at either cap.

## Speculative decoding (n-gram), C1, 400 tokens, 3 reps

| Drafting | Tok/s |
|---|---:|
| Off | 26.9 |
| n-gram | 25.0 (-7.2%) |

No benefit found for this prompt shape; standing config ships with drafting disabled.

## Prefill / TTFT (2,941-token prompt, 2,954 tokens post chat-template, tokenizer-exact)

**180 W (2026-09-03, canonical):**

| Rep | Prompt time | Throughput |
|---|---|---:|
| 0 (cold) | 0.44 s | 313.6 tok/s |
| 1 (warm) | 0.39 s | 353.9 tok/s |
| 2 (warm) | 0.38 s | 363.2 tok/s |

Warm mean (reps 1-2): 358.5 tok/s. TTFT (3 reps, streaming): 0.73 s /
1.41 s / 1.78 s.

**250 W (2026-09-02, retained for comparison):**

| Condition | Prompt time | Throughput |
|---|---|---:|
| Cold (first request after boot) | 5.57 s | — |
| Warm (reps 2-3, usage-reported mean) | 0.39 s | ~354 tok/s |

**Caveat (load-bearing):** `cache_mode: Q8` crashes any single request whose
context exceeds ~2,048 tokens on this runtime version, due to an explicit
`assert qc is None` guard against sparse DSA attention over a quantized MLA
cache in exllamav3 1.4.6's `mla_attn.py`. The 2026-09-02 measurement
required temporarily switching to `cache_mode: FP16`; by the 2026-09-03
re-measure the standing config was already `FP16` for the 262k-context
work. See the attempt record for full detail.

## Power

**180 W (2026-09-03, verified cap, canonical), 1 Hz samples across the
full ladder + prefill/TTFT window (~10.5 min):**

| GPU | Mean W | Peak W | Peak temp (C) |
|---|---:|---:|---:|
| 0 | 56.7 | 139.5 | 51 |
| 1 | 58.8 | 172.6 | 49 |
| 2 | 56.4 | 168.4 | 49 |
| 3 | 49.8 | 102.5 | 45 |
| **Total** | **221.6** | **352.4** | — |

Peak per-card power (172.6 W) stayed under the 180 W cap at every sample;
total peak is a coincident-peak artifact, not a real simultaneous 4-card
draw.

**250 W (2026-09-02, accidental default, retained for comparison), C4
load, 1s samples, 60s window:**

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
