# Changelog

Dated entries for publishable changes to this repository. See `attempts/`
for full per-attempt records and `results/` for raw evidence.

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
