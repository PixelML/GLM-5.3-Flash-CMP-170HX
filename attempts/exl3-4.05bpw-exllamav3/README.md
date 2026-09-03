# Attempt — EXL3 4.05bpw on exllamav3 + TabbyAPI

Status: measured (served and benchmarked 2026-09-03)
Date: 2026-09-03

## Checkpoint

- Repository + exact revision: [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`
- Exact download bytes (download manifest): **165,151,555,504 bytes = 153.81 GiB**, 31 files in the manifest (shard weights plus config files); the served checkpoint directory holds 105 files after local re-layout and added config files
- Installed size: same weight bytes as download; file count grows only from re-layout and config additions, not from a conversion step
- Quantization: EXL3, 4.05 bits per weight (TurboDerp EXL3 codebook format)
- License: source-available per the EXL3 quant tooling; base model license applies to the underlying weights — record and check the checkpoint card before redistribution
- Architecture: `Glm5NextForConditionalGeneration` / `model_type: glm5_next`, DSA (DeepSeek Sparse Attention) with an indexer, `index_topk: 2048` (measured from `config.json`) — same MLA/DSA family recorded elsewhere in this repository
- Source of each number above: measured (download manifest, local file count)

## Runtime

- Engine: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) **1.4.6+cu128.torch2.10.0**, served via **TabbyAPI** (OpenAI-compatible server)
- SM80 support: **yes — measured**. This is the first row in this repository's comparison table to reach sustained, multi-request serving on the CMP 170HX (GA100, SM80) cards themselves, as opposed to a static-fit or compatibility-only finding. (A GGUF/llama.cpp pairing served successfully earlier on this node too; see the "Current conclusion" section of the top-level README — that result predates and is separate from this table.)
- Topology: manual `gpu_split: [48, 48, 48, 48]` (GiB per card), four cards. **`tensor_parallel: true` is not usable on this architecture** — it raises `NotImplementedError` for `Glm5NextForConditionalGeneration` in exllamav3 1.4.6. `gpu_split` with `tensor_parallel: false` is the only working topology.
- Source: measured (local build + serving on the four-card node, 2026-09-03)

## Static fit calculation

- Per-card budget: 64 GiB x 4 = 256 GiB aggregate
- Weights: 165,151,555,504 B = **153.81 GiB total**, fitting across the 256 GiB aggregate with room left for KV cache and activation overhead
- Working split `gpu_split: [48, 48, 48, 48]` (192 GiB nominal cap across 4 cards) leaves headroom versus the 153.81 GiB weight footprint for Q8 KV cache and MLA attention scratch
- Measured per-card memory after a successful load at this split: GPU0=48,468 MiB, GPU1=47,982 MiB, GPU2=47,982 MiB, GPU3=12,084-12,116 MiB (~153 GiB total). GPU3 stays underloaded relative to GPU0-2 at this split — see "Boot-attempt ladder," item 4, for why.

## Execution status and outcome

**Served and benchmarked (2026-09-03).** The server booted, all correctness
gates passed, and the full benchmark ladder ran to completion. See "Gates,"
"Benchmark ladder," "Speculative decoding," "Prefill / TTFT," and "Power"
below for full results.

### Boot-attempt ladder (failure history — record these; they are useful negative results)

exllamav3's `gpu_split` autosplit algorithm fills devices in sequential
order up to each device's stated cap; it does not reserve headroom for KV
cache or activation scratch, and it does not rebalance a card that is left
underfull. Four splits were tried before one worked:

1. **`gpu_split: [64, 64, 64, 64]` — booted, then OOM'd on first inference (not load).**
   The naive greedy layer-fill packed GPU0 and GPU1 near-full with weights
   alone, leaving no headroom for KV cache or attention scratch during the
   MLA attention prefill kernel. GPU3 was left completely unused (0 MiB) by
   the naive fill. Per-card memory at the OOM: GPU0=63,232 MiB,
   GPU1=62,510 MiB, GPU2=30,164 MiB, GPU3=6 MiB. Lesson: a split equal to
   the full per-card VRAM leaves no room for anything but weights, and the
   failure only shows up at first inference, not at load time.
2. **`gpu_split: [40, 40, 40, 40]` — crashed immediately.** `CUDA error: out
   of memory` inside `touch_device`'s trivial `torch.empty((32, 32))` call —
   a failure this small does not indicate a real VRAM shortage by itself.
   This run coincided with NVRM driver-level corruption: torch's CUDA init
   failed host-wide even though `nvidia-smi` reported 0 MiB used and a
   healthy driver state. Recovery required an out-of-band driver reload
   (`rmmod nvidia_uvm nvidia; modprobe nvidia nvidia_uvm`) by the operator
   before any further attempt could run. Lesson: a `torch.empty` failure at
   attach time can mean the driver state is corrupted, not that the split is
   too small — check `nvidia-smi` and consider a driver reload before
   re-tuning the split.
3. **`gpu_split: [40, 40, 40, 40]` again, post-driver-reload, checkpoint
   moved to local NVMe, `cache_mode: Q8`.** Clean failure at 46/50 modules
   loaded: `RuntimeError: Insufficient VRAM in split for model and cache`.
   153.81 GiB of weights plus a Q8 KV cache plus activation overhead does
   not fit inside a 160 GiB (4 x 40 GiB) budget. Lesson: this is a real,
   correctly-diagnosed capacity shortfall, distinct from item 2's driver
   corruption.
4. **`gpu_split: [48, 48, 48, 48]` — succeeded.** Boot time approximately
   1.5-4 minutes from local NVMe. Per-card memory after load: GPU0=48,468 MiB,
   GPU1=47,982 MiB, GPU2=47,982 MiB, GPU3=12,084-12,116 MiB (~153 GiB total).
   GPU3 remains notably underloaded relative to GPU0-2 even at this working
   split — this is a known characteristic of exllamav3's naive
   sequential-fill autosplit algorithm (it fills devices in declaration
   order until each hits its stated per-device cap; the last device just
   receives whatever layers are left), not a functional defect. A future
   attempt could try shaping the split unevenly (e.g. giving GPU3 a larger
   cap) to balance load, but this was not tested here.

### Gates (all passed)

- `/v1/models`: 200 OK.
- Deterministic text gate (`max_tokens=32`, `temp=0`, prompt "What is the
  capital of France? Answer with just the city name."): reproducibly (3/3)
  **insufficient at this budget** — with `reasoning: true`, the model spends
  the entire 32-token budget on chain-of-thought and returns `content: null`,
  `finish_reason: "length"`. This is a structural property of the reasoning
  template at this budget, not a bug. At `max_tokens=128` the same prompt
  cleanly returns `content: "Paris"`, with the chain-of-thought correctly
  routed to a separate `reasoning_content` field. **Recommendation:
  `max_tokens >= 128` for short factual answers whenever `reasoning: true`
  is set.**
- Golden corpus (20 prompts spanning short_factual, reasoning, code, json,
  and multilingual categories; keyword-match scoring; `max_tokens=512`):
  **20/20 passed.**

### Benchmark ladder

Greedy decoding, exactly 400 completion tokens per request, 1 warmup
repetition plus 3 measured repetitions, at concurrency 1/2/4/8:

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 26.9 | 26.9 |
| C2 | 31.1 | 15.6 |
| C4 | 41.7 | 10.5 |
| C8 | 44.8 | 8.3 |

Memory and temperature were sampled every 2 seconds continuously through
C8: no growth trend, peak observed temperature 49 degrees C on GPU0, well
under the 80 degrees C safety stop threshold used on this node. No OOM, no
Xid or ECC events at any concurrency tested.

### Speculative decoding (n-gram) — tested, no benefit found

Compared `draft_mode: ngram` against `draft_mode: model` with no draft
model configured (drafting effectively disabled), at concurrency 1, 400
completion tokens, 3 repetitions:

| Drafting | Tok/s |
|---|---:|
| Off (`draft_mode: model`, no draft model) | 26.9 |
| n-gram (`draft_mode: ngram`) | 25.0 (-7.2%, within noise / slight regression) |

No benefit was found for this prompt shape (a short technical-explanation
prompt, where a low n-gram hit rate is expected). The shipped/standing
config in this recipe uses drafting disabled. This result should not be
generalized to prompts with more repetitive or templated structure, where
n-gram drafting is more likely to help.

### Prefill / TTFT — load-bearing finding for this model + runtime combination

A 2,941-token prompt (token count verified against the model's own
tokenizer) was used for a prefill/time-to-first-token measurement. This
exposed a real exllamav3 1.4.6 behavior that governs safe production
configuration. A follow-up investigation on 2026-09-03 (see "Update
2026-09-03" below) found the exact root cause and lifted the limitation
for context up to 262,144 tokens.

> **GLM-5.3-Flash uses DSA (DeepSeek Sparse Attention) with an indexer.**
> Per the model's `config.json`, `index_topk: 2048`. In exllamav3's
> `mla_attn.py`, the sparse-attention path activates once
> `max(host_seqlens) + seqlen > index_topk` — in practice, once context
> passes roughly 2,048 tokens — and that sparse path carries an explicit
> assertion: `assert qc is None, "sparse DSA over a quantized MLA cache is
> not supported yet; use an fp16 cache"`.
>
> **Consequence: on exllamav3 1.4.6, any single request whose context
> exceeds roughly 2,048 tokens fails outright (a 503 on that one request,
> not a crash of the whole server) whenever `cache_mode: Q8` is set.** The
> standing/production config recommended by this recipe (`cache_mode: Q8`)
> is therefore **only safe for short-context serving (<= 2,048 tokens)**
> until this limitation is fixed upstream in exllamav3.

To obtain the 2,941-token prefill measurement at all, `cache_mode` was
temporarily switched to `FP16` for this one measurement, then reverted to
`Q8` immediately afterward. Results with an FP16 cache:

- Cold (first request after boot): 5.57 s prompt time for the full request.
- Warm (repetitions 2-3, usage-reported): mean `prompt_time` 0.39 s, which
  is approximately 354 tok/s prefill throughput.
- With `cache_mode: FP16`, the same `gpu_split: [48, 48, 48, 48]` still fits
  with headroom: GPU0=45,270 MiB, GPU1=48,078 MiB, GPU2=44,526 MiB,
  GPU3=19,316 MiB — but this configuration was **not** load-tested
  end-to-end at concurrency; this is flagged as future work, not a
  validated recommendation.

### Update 2026-09-03 — root cause found, 262,144-token context validated

A same-day follow-up investigation identified the exact root cause of the
~2,048-token cap above (it is DSA behavior colliding with a quantized
cache, not a `max_seq_len` or TabbyAPI chunk-size limit), then booted the
server with `cache_mode: FP16` and validated context up to 262,144 tokens.
**This limitation is now resolved.** The `cache_mode: FP16` config is the
recommended default for this recipe whenever long context matters; see
"Recommended configs" below.

#### Root cause

GLM-5.3-Flash uses DSA (DeepSeek Sparse Attention). The model's own
`config.json` sets `index_topk: 2048`. In exllamav3 1.4.6
(`exllamav3/modules/mla_attn.py`):

- The sparse-attention path activates once
  `max(host_seqlens) + seqlen > self.index_topk` (line 765) — this is
  intended DSA behavior, not a bug, on a model that natively supports up
  to 1,048,576 tokens (`max_position_embeddings` in `config.json`).
- That sparse-attention path asserts it cannot run against a quantized KV
  cache: `assert qc is None, "sparse DSA over a quantized MLA cache is not
  supported yet; use an fp16 cache"` (`_attend_sparse`, ~line 906).
- `qc` is only non-`None` when `cache_mode` is `Q8`/`Q6`/`Q4` (i.e.
  `CacheLayer_MLA_quant`). With `cache_mode: FP16`
  (`CacheLayer_MLA_fp16`), `qc = None` and the sparse path runs correctly.

**Conclusion: the cap came from `cache_mode: Q8` colliding with the DSA
indexer window — not from `max_seq_len`, and not from TabbyAPI's
chunk-size or max-input settings.**

#### The 262k-context boot (fix + validation)

Config changed from the short-context standing server documented above:

| Setting | Short-context (this attempt, above) | 262k-context (this update) |
|---|---|---|
| `cache_mode` | `Q8` | `FP16` |
| `max_seq_len` / `cache_size` | `32768` | `262144` |
| `chunk_size` | `2048` | `4096` |
| `gpu_split` | `[48, 48, 48, 48]` | `[48, 48, 48, 48]` (unchanged — no increase needed) |
| `reasoning` | `true` | `true` (unchanged) |
| Drafting | disabled | disabled (unchanged) |

Only one boot attempt was needed (succeeded first try, unlike the
four-attempt boot ladder recorded above for the original recipe). Boot
time approximately 2.5 minutes from local NVMe. Per-card memory after
load: GPU0=46,230 MiB, GPU1=45,456 MiB, GPU2=45,488 MiB, GPU3=23,638 MiB —
comfortably under each card's 64 GiB capacity (19-42 GiB headroom per
card).

#### Prefill ladder (max_tokens=1, tokenizer-exact prompt lengths, same booted server throughout, no restart between steps)

| Prompt tokens | Result | Prefill time | Peak mem (GPU0/1/2/3, MiB) |
|---:|---|---:|---|
| 2,941 | OK | 14.27 s | 48,308 / 47,518 / 47,550 / 25,662 |
| 16,000 | OK | 20.79 s | 49,076 / 48,322 / 48,354 / 26,496 |
| 32,000 | OK | 26.92 s | 49,076 / 48,356 / 48,356 / 26,496 |
| 65,000 | OK | 50.80 s | 49,142 / 48,388 / 48,420 / 26,624 |
| 131,000 | OK | 102.04 s | 49,144 / 48,390 / 48,422 / 26,626 |
| 200,000 | OK | 48.21 s | 49,176 / 48,390 / 48,422 / 26,626 |
| 250,000 | OK | 35.35 s | 49,176 / 48,390 / 48,422 / 26,626 |

**Largest verified prompt: 250,000 tokens.** No OOM and no crash at any
tested length.

Prefill time drops noticeably past 131k tokens, because DSA's indexer caps
attention cost at a fixed top-k=2048 token selection regardless of total
context length — so cost stops scaling linearly once the sparse path is
fully engaged. The 131k data point looks like a transitional or
less-optimized case rather than a regression; this is an open question for
further investigation, not a definitive explanation.

Memory stayed essentially flat from 16k tokens onward: going from 16k to
250k tokens of context cost only about 2 GiB total across all four cards.
The MLA/DSA cache is cheap per token, and the model's hybrid
linear-attention layers (KDA / gated-delta-net) carry O(1) state that does
not grow with context.

#### Needle-in-haystack retrieval test

| Context length | Result |
|---|---|
| 32k tokens | PASS — correctly retrieved a planted unique fact |
| 250k tokens | PASS — correctly retrieved a planted unique fact |

#### Health during the whole ladder

No Xid or ECC (including double-bit) events at any point, checked via
`nvidia-smi -q` and `dmesg`. Peak temperature 51 degrees C. The server
process stayed up for the entire ladder; no restart was needed between
steps, and the GPU driver was never reloaded (none was needed).

#### Scope note

Only prefill/context-length behavior was re-tested in this update. The
full C1/C2/C4/C8 throughput ladder was **not** re-run at 262k context —
the 26.9-44.8 tok/s figures above remain the short-context (`Q8` cache)
measurement. No throughput regression was observed at short context during
this update's testing, but that is not the same claim as a re-run ladder.

### Power

Measured during a C4 load run, 1-second samples over 60 seconds:

| GPU | Mean W | Peak W |
|---|---:|---:|
| 0 | 65.4 | 120.0 |
| 1 | 63.0 | 101.4 |
| 2 | 56.2 | 97.4 |
| 3 | 50.1 | 78.9 |
| **Total** | **234.7** | **302.3** |

No Xid or ECC events. Maximum temperature observed anywhere in this run:
49 degrees C.

## Recommended configs

- **Recommended default for long-context use (as of 2026-09-03):**
  `cache_mode: FP16`, `max_seq_len` / `cache_size: 262144`,
  `chunk_size: 4096`, `gpu_split: [48, 48, 48, 48]`,
  `tensor_parallel: false`, `reasoning: true`, drafting disabled. Validated
  up to 250,000 prompt tokens with no OOM, no crash, and passing
  needle-in-haystack retrieval at 32k and 250k tokens (see "Update
  2026-09-03" above). Costs almost nothing extra in VRAM versus the
  short-context config (19-42 GiB headroom per card) and showed no
  throughput regression at short context in this update's testing — but
  the full C1/C2/C4/C8 throughput ladder was not re-run at 262k context.
- **Short-context / lower-VRAM alternative (<= 2,048 tokens), standard
  chat use:** `cache_mode: Q8`, `max_seq_len` / `cache_size: 32768`,
  `chunk_size: 2048`, `gpu_split: [48, 48, 48, 48]`,
  `tensor_parallel: false`, `reasoning: true`, drafting disabled.
  Approximately 27-45 tok/s aggregate depending on concurrency (see the
  ladder above), approximately 235 W typical draw, approximately 153 GiB
  VRAM footprint. **Resolved as of 2026-09-03:** this config's context cap
  was originally believed to be a hard ~2,048-token limitation; it is now
  understood to be a `cache_mode: Q8` / DSA-indexer interaction, fixed by
  switching to `cache_mode: FP16` (see "Update 2026-09-03" above). Keep
  this `Q8` config only when the lower VRAM footprint matters more than
  long context.

## Scripts

The following scripts were used to produce the results above. Paths are
described generically; this repository does not reference internal
filesystem layout or private hostnames.

- **Concurrency-ladder script** (referred to above as `bench_ladder.py`):
  issues N parallel greedy, 400-completion-token requests against the
  TabbyAPI OpenAI-compatible endpoint, with one discarded warmup repetition
  followed by 3 measured repetitions per concurrency level (1/2/4/8).
  Aggregates completion-token counts from the response `usage` object (per
  this repository's evidence rules — never from stream-event counts) and
  reports both aggregate tok/s and mean per-request tok/s.
- **Golden-corpus keyword-match script**: sends the 20-prompt corpus
  (short_factual, reasoning, code, json, multilingual categories) at
  `max_tokens=512`, greedy decode, and scores each response by checking for
  one or more required keywords/substrings in the returned `content` field.
- **Tokenizer-exact prefill prompt**: a fixed prompt built and verified
  against the target model's own tokenizer to land at an exact token count
  (2,941 tokens for the prefill/TTFT measurement above), used to cross the
  `index_topk: 2048` DSA sparse-attention boundary deliberately and
  measure prefill throughput on both sides of it.

## Blocker

**Resolved (2026-09-03).** None for short-context serving. The former
long-context blocker — exllamav3 1.4.6's `assert qc is None` guard against
running sparse DSA attention over a quantized (Q8) MLA cache — is lifted
by switching to `cache_mode: FP16`, validated up to 262,144 tokens of
context (250,000 tokens of prompt actually tested) with no OOM and no
crash. See "Update 2026-09-03" above. The C1/C2/C4/C8 throughput ladder at
262k context remains future work.

## Evidence

- Download manifest: exact byte count and file count as reported by the
  checkpoint's own manifest at the pinned revision, 2026-09-03.
- Boot-attempt ladder: local server logs and `nvidia-smi` snapshots at each
  attempt (not committed; per-attempt memory figures recorded above).
- Gate and benchmark results: local harness output (`bench_ladder.py`,
  golden-corpus script), summarized in
  [results/2026-09-03-exl3-4.05bpw-exllamav3/](../../results/2026-09-03-exl3-4.05bpw-exllamav3/).
- exllamav3 sparse-DSA-over-quantized-cache assertion: `mla_attn.py` in
  [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) at
  the pinned 1.4.6 release.
- `tensor_parallel` `NotImplementedError` for `Glm5NextForConditionalGeneration`:
  observed directly when `tensor_parallel: true` was attempted on this
  architecture, 2026-09-03.

## Re-run instructions

1. Install exllamav3 1.4.6+cu128.torch2.10.0 and TabbyAPI, matched to the
   local CUDA/torch stack.
2. Fetch `turboderp/GLM-5.3-Flash-exl3` at branch `4.05bpw`, revision
   `2a30229e67012798ba9f0cd832bb78abf4c363d5`, into local checkpoint storage
   (not published here; see AGENTS.md).
3. Configure TabbyAPI with `gpu_split: [48, 48, 48, 48]`,
   `tensor_parallel: false`, `max_seq_len: 32768`, `cache_size: 32768`,
   `cache_mode: Q8`, `reasoning: true`, `draft_model.draft_mode: model`.
   Do not start from `gpu_split: [64, 64, 64, 64]` or an even smaller split
   than `[48, 48, 48, 48]` — see the boot-attempt ladder above for why both
   directions fail.
4. Boot and confirm `/v1/models` returns 200 OK, then run the deterministic
   text gate at `max_tokens >= 128` (not 32 — see "Gates") and the
   golden-corpus script before trusting the server for further measurement.
5. For any workload expected to exceed ~2,048 tokens of context, switch
   `cache_mode` to `FP16` before serving that traffic; do not serve
   long-context requests against a `Q8` cache with this runtime version.
6. Abort at 80 degrees C core / 85 degrees C memory temperature, or on any
   Xid or ECC event, per this repository's infrastructure-safety rules.
