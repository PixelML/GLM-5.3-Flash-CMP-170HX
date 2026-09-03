# Changelog

Dated entries for publishable changes to this repository. See `attempts/`
for full per-attempt records and `results/` for raw evidence.

## 2026-09-03 (follow-up)

- **Resolved the ~2,048-token context cap** documented earlier the same
  day for row 7 (EXL3 4.05bpw / exllamav3 1.4.6 / TabbyAPI). Root cause:
  GLM-5.3-Flash's DSA (DeepSeek Sparse Attention) indexer
  (`index_topk: 2048` in `config.json`) triggers exllamav3's sparse
  attention path once context exceeds roughly 2,048 tokens, and that path
  asserts `qc is None` — it cannot run against a quantized (`Q8`/`Q6`/`Q4`)
  MLA cache. With `cache_mode: FP16`, `qc = None` and sparse attention
  works correctly. The cap was never a `max_seq_len` or TabbyAPI
  chunk/max-input limitation.
- Booted the server with `cache_mode: FP16`, `max_seq_len` /
  `cache_size: 262144`, `chunk_size: 4096`, `gpu_split` unchanged at
  `[48, 48, 48, 48]`. Boot succeeded on the first attempt, approximately
  2.5 minutes from local NVMe. Per-card memory after load: GPU0=46,230,
  GPU1=45,456, GPU2=45,488, GPU3=23,638 MiB (19-42 GiB headroom per card).
- Ran a prefill ladder (max_tokens=1, tokenizer-exact prompt lengths, one
  continuously-booted server, no restart between steps) at 2,941 / 16,000
  / 32,000 / 65,000 / 131,000 / 200,000 / 250,000 prompt tokens. All
  passed, no OOM, no crash. **Largest verified prompt: 250,000 tokens.**
  Prefill time drops past 131k tokens because DSA's fixed top-k=2048
  indexer caps attention cost regardless of total context length — flagged
  as an open question, not a definitively explained regression. Memory
  stayed essentially flat from 16k tokens onward (~2 GiB total growth
  across all four cards from 16k to 250k tokens).
- Ran needle-in-haystack retrieval at 32k and 250k tokens: both **PASS**.
- No Xid or ECC (including double-bit) events at any point in the ladder
  (`nvidia-smi -q` and `dmesg`); peak temperature 51 degrees C; server
  process stayed up throughout, no driver reload needed.
- **`cache_mode: FP16` / 262,144-token context is now the recommended
  default config** for this recipe whenever long context matters. The
  `cache_mode: Q8` / 32,768-token config remains documented as the
  lower-VRAM short-context alternative. The full C1/C2/C4/C8 throughput
  ladder was **not** re-run at 262k context in this update — only
  prefill/context-length behavior was re-tested.
- Updated
  [attempts/exl3-4.05bpw-exllamav3/README.md](attempts/exl3-4.05bpw-exllamav3/README.md),
  [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), and the row-7 entry
  in this repository's [README.md](README.md) comparison table to reflect
  the resolved status.

## 2026-09-03

- Added the first successful **comparison-table** row (row 7): GLM-5.3-Flash
  via exllamav3 1.4.6+cu128.torch2.10.0 + TabbyAPI, checkpoint
  `turboderp/GLM-5.3-Flash-exl3` (branch `4.05bpw`,
  `2a30229e67012798ba9f0cd832bb78abf4c363d5`, EXL3 4.05 bpw), served across
  all four CMP 170HX cards at manual `gpu_split: [48, 48, 48, 48]`
  (`tensor_parallel` is unimplemented for this architecture in exllamav3
  1.4.6). Full benchmark ladder completed: 26.9-44.8 tok/s aggregate across
  concurrency 1-8; 20/20 golden-corpus pass; deterministic-text gate needs
  `max_tokens >= 128` once `reasoning: true` is set. See
  [attempts/exl3-4.05bpw-exllamav3/README.md](attempts/exl3-4.05bpw-exllamav3/README.md)
  and [results/2026-09-03-exl3-4.05bpw-exllamav3/](results/2026-09-03-exl3-4.05bpw-exllamav3/README.md).
- **Key caveat carried forward from this recipe:** the standing/production
  `cache_mode: Q8` config is only safe for context lengths up to roughly
  2,048 tokens. Past that point, exllamav3 1.4.6's DSA sparse-attention
  path asserts against running over a quantized MLA cache, and any
  individual request over the threshold fails with a 503 (the server
  itself stays up). Long-context serving requires `cache_mode: FP16`,
  which fits the same `gpu_split` with headroom but is unvalidated at
  concurrency — flagged as future work, not a validated recommendation.
- Documented four prior boot-attempt failures on the way to the working
  `gpu_split` (naive full-VRAM split OOMing on first inference rather than
  load; a `torch.empty` OOM that turned out to be NVRM driver corruption
  requiring an out-of-band driver reload; a clean, correctly diagnosed
  "insufficient VRAM in split" failure at an undersized split) — preserved
  as negative-result documentation per this repository's evidence rules.
- Added `docs/API.md` (TabbyAPI OpenAI-compatible endpoint usage, the
  `reasoning: true` requirement and `reasoning_content` field, and the
  `stream_options.include_usage` requirement for usage accounting).
- Added `docs/TROUBLESHOOTING.md`, covering the four boot-attempt failure
  modes above plus the Q8-cache 2,048-token DSA limitation as a known
  issue, and the exllamav3 `tensor_parallel` `NotImplementedError`.
- Confirmed the vLLM lane remains terminal (unchanged: `glm5_next` absent
  from upstream vLLM's model registry on any GPU architecture; see rows 1,
  3, 6 and `attempts/w4a16-autoround-vllm/README.md`) — not re-run for
  this update. Confirmed no SGLang attempt record exists in this
  repository; left as an explicit TODO rather than an assumed status.
