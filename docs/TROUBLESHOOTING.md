# Troubleshooting

Known issues encountered while running GLM-5.3-Flash on CMP 170HX cards, and
their fixes. Each entry links back to the attempt record where it was
diagnosed. Entries are runtime-specific unless noted otherwise.

## exllamav3 / TabbyAPI (EXL3 4.05bpw recipe)

Full attempt record: [attempts/exl3-4.05bpw-exllamav3/README.md](../attempts/exl3-4.05bpw-exllamav3/README.md).

### Resolved: `Q8` KV cache crashes any request over ~2,048 tokens (DSA sparse attention)

**Status: resolved 2026-09-03.** This was previously an open blocker for
long-context serving; a follow-up investigation found the exact root cause
and validated a fix up to 262,144 tokens of context. Full data:
[attempts/exl3-4.05bpw-exllamav3/README.md](../attempts/exl3-4.05bpw-exllamav3/README.md#update-2026-09-03--root-cause-found-262144-token-context-validated).

**Symptom:** a single request with a long prompt fails with a 503; the
server itself stays up and keeps serving other requests.

**Root cause:** GLM-5.3-Flash uses DSA (DeepSeek Sparse Attention) with an
indexer. Per the model's `config.json`, `index_topk: 2048`. In exllamav3
1.4.6's `exllamav3/modules/mla_attn.py`:

- Sparse attention activates once `max(host_seqlens) + seqlen >
  self.index_topk` (line 765) — intended DSA behavior, not a bug, on a
  model that natively supports up to 1,048,576 tokens
  (`max_position_embeddings` in `config.json`).
- The sparse-attention code path asserts it cannot run against a quantized
  KV cache: `assert qc is None, "sparse DSA over a quantized MLA cache is
  not supported yet; use an fp16 cache"` (`_attend_sparse`, ~line 906).
- `qc` is only non-`None` when `cache_mode` is `Q8`/`Q6`/`Q4` (i.e.
  `CacheLayer_MLA_quant`). With `cache_mode: FP16`
  (`CacheLayer_MLA_fp16`), `qc = None` and the sparse path works fine.

**Conclusion: the cap came from `cache_mode: Q8` colliding with the DSA
indexer window — not from `max_seq_len`, and not from TabbyAPI's
chunk-size or max-input settings.**

**Fix:** set `cache_mode: FP16` (also raise `max_seq_len` / `cache_size`
to `262144` and `chunk_size` to `4096` for full long-context headroom;
`gpu_split` does not need to change). Validated up to 250,000 prompt
tokens with no OOM and no crash; needle-in-haystack retrieval passed at
32k and 250k tokens. Boot time approximately 2.5 minutes; per-card memory
after load stayed comfortably under 64 GiB (46,230 / 45,456 / 45,488 /
23,638 MiB across the four cards).

**Trade-off:** `cache_mode: FP16` increases per-request KV cache memory
versus `Q8`, but in practice this cost almost nothing extra — memory grew
only about 2 GiB total across all four cards going from 16k to 250k tokens
of context, because the MLA/DSA cache is cheap per token and this model's
hybrid linear-attention layers carry O(1) state that does not grow with
context. `cache_mode: FP16` / 262144-token context is now the recommended
default for this recipe when long context matters; keep the `Q8` /
32768-token config only when the lower VRAM footprint matters more.
**Not re-tested in this update:** the full C1/C2/C4/C8 throughput ladder
at 262k context — only prefill/context-length behavior was re-verified.

### Failure mode 1: even `gpu_split` at full per-card VRAM OOMs on first inference, not load

**Symptom:** the server boots successfully and reports a healthy model
load, then OOMs on the very first inference request.

**Cause:** `gpu_split: [64, 64, 64, 64]` (equal to the full 64 GiB per-card
budget) packs weights tightly enough that no headroom is left for KV cache
or attention scratch space, which is only allocated once inference starts
(specifically, during the MLA attention prefill kernel). exllamav3's
autosplit algorithm also fills devices in sequential order, so it left one
GPU (GPU3, in this run) at 0 MiB used while the earlier GPUs were packed
near-full.

**Fix:** never set a per-card `gpu_split` value at or near the card's full
VRAM budget. Leave room for KV cache and activation scratch. This recipe's
working value is `[48, 48, 48, 48]` on 64 GiB cards (~16 GiB/card
headroom against the raw VRAM ceiling, before weights).

### Failure mode 2: `torch.empty` OOM at attach time can mean driver corruption, not a real capacity shortfall

**Symptom:** the server crashes immediately on a small `gpu_split`
(`[40, 40, 40, 40]` in this run) with `CUDA error: out of memory` inside
`touch_device`'s trivial `torch.empty((32, 32))` call — an allocation far
too small to plausibly exhaust 40 GiB of VRAM.

**Cause (in this run):** this failure coincided with NVRM driver-level
corruption. `torch`'s CUDA initialization failed host-wide even though
`nvidia-smi` reported 0 MiB used per card and a nominally healthy driver
state.

**Fix:** if a trivially small CUDA allocation fails at attach time, check
`nvidia-smi` for a healthy state first. If `nvidia-smi` looks healthy but
`torch` still cannot initialize CUDA, the driver state itself may be
corrupted; recovering requires an out-of-band driver reload
(`rmmod nvidia_uvm nvidia; modprobe nvidia nvidia_uvm`) performed by a
human operator with appropriate access — do not treat this as a signal to
keep shrinking the split, since it is not a capacity problem.

### Failure mode 3: clean `Insufficient VRAM in split for model and cache` at a too-small split

**Symptom:** the server fails cleanly and legibly partway through module
loading (46/50 modules in this run) with
`RuntimeError: Insufficient VRAM in split for model and cache`.

**Cause:** this is a genuine, correctly diagnosed capacity shortfall,
distinct from failure mode 2 above (this run was retried post-driver-reload
and on local NVMe storage, with `cache_mode: Q8` set, and still failed at
this split). 153.81 GiB of weights plus a Q8 KV cache plus activation
overhead does not fit inside a 160 GiB (4 x 40 GiB) budget.

**Fix:** raise the `gpu_split` cap per card. This recipe found
`[48, 48, 48, 48]` (192 GiB nominal) sufficient.

### Note: `tensor_parallel: true` is not usable for this architecture

**Symptom:** setting `tensor_parallel: true` raises `NotImplementedError`.

**Cause:** exllamav3 1.4.6 does not implement tensor-parallel execution for
`Glm5NextForConditionalGeneration`.

**Fix:** use manual `gpu_split` with `tensor_parallel: false` instead. This
is the only working topology for this architecture on this runtime
version.

### Note: uneven GPU utilization across cards at a working split

**Symptom:** at a working split (e.g. `[48, 48, 48, 48]`), the last GPU in
the list ends up noticeably less loaded than the others (~12 GiB versus
~48 GiB in this run) even though the split values were identical.

**Cause:** exllamav3's autosplit algorithm fills devices in declaration
order until each hits its stated per-device cap; the last device simply
receives whatever layers are left over, rather than the algorithm
rebalancing to equalize load. This is a known characteristic of the
algorithm, not a functional defect — the server still works correctly.

**Possible improvement (untested):** shaping the split unevenly (for
example, giving the last device in the list a larger cap) may balance load
better. Not tested in this attempt; flagged as future work.

## vLLM (all rows using it in this repository)

**Terminal blocker, documented in prior attempt records — not re-diagnosed
here.** `glm5_next` is absent from upstream vLLM's model registry, on any
GPU architecture. See
[attempts/w4a16-autoround-vllm/README.md](../attempts/w4a16-autoround-vllm/README.md)
for the fullest writeup, including the specific open issues and the one
support PR ([vllm#53906](https://github.com/vllm-project/vllm/pull/53906),
open, SM90+ only).

## SGLang

No attempt record exists for SGLang in this repository as of 2026-09-03.
TODO: evaluate SGLang against `glm5_next` / DSA support on SM80 in a future
attempt record before this repository states a status for it.
