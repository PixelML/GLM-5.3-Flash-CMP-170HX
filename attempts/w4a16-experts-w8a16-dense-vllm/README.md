# Attempt — W4A16 experts + W8A16 dense (vLLM compressed-tensors)

Status: static-fit-only
Date: 2026-08-31

## Checkpoint (planned)

- Base: zai-org/GLM-5.3-Flash on Hugging Face, Glm5NextForConditionalGeneration
  (config: hidden 4096, 45 layers, 288 routed experts, top-k routing, 1 shared
  expert, first 3 layers dense at intermediate 12288, MoE intermediate 2048,
  vocab 154880). Exact pinned quantization revision: TBD, none exists yet.
- Quantization (planned): compressed-tensors mixed scheme — routed-expert
  linear weights W4A16 (group-wise), dense linears (attention projections,
  dense MLP, router gate, embeddings, lm_head) W8A16 (group-wise). Activations
  stay 16-bit; KV cache stays fp8 as in the measured GGUF baseline.
- License: base-model license applies to any derived artifact; conversion
  would be local-only, no weights published.

## Runtime (static analysis, untested)

- vLLM Glm5Next exists only via PR 53906 (open, 17 commits, head
  7cf764c53ace7d83e1d237d87e8d3a1d9aac7b58, mergeable but mergeStateStatus
  blocked at time of writing). Architecture is NOT in vLLM main.
- Compressed-tensors loader in vLLM main documents per-target mixed schemes:
  config_groups is a map of N groups, each with its own target list and
  QuantizationArgs; get_scheme resolves per layer via find_matched_target
  (vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py).
- W4A16 and W8A16 both route through the WNA16 path
  (schemes/compressed_tensors_wNa16.py); MoE experts route through
  compressed_tensors_moe_wna16.py, which is wired to RoutedExperts via
  _add_fused_moe_to_target_scheme_map. Two different num_bits values in two
  config groups are the designed input to this dispatcher.
- llm-compressor mixed-precision infra exists (closed #1713, #2083; open
  #2940 shows a W4A16/W2A16 MoE precedent). vLLM PR 48918 (CT WNA16 MoE
  backend incl. Humming) is MERGED into main, so no additional serving-side
  patch is required beyond 53906 for the MoE wna16 path.

## Static fit calculation (planned artifact, not measured)

Per public config.json:

- Dense layers 1-3: 3 x (gate_up 4096x2x12288 + down 12288x4096) =
  452,984,832 params
- Shared expert (1 per MoE layer, 42 layers): 42 x (4096x2x2048 + 2048x4096)
  = 1,056,964,608 params
- Routed experts: 42 x 288 x 25,165,824 = 304,508,553,216 params
- Attention per layer Q,K,V,O at 64 heads x 128 head_dim: 4 x 4096 x 8192 =
  134,217,728; x45 = 6,039,797,760 params
- Embeddings + untied lm_head: 2 x 154880 x 4096 = 1,268,715,520 params

Weights at planned bit widths (group-scale overhead folded into a 1.0625x
packing factor; bf16 = 2 B/param):

- Routed experts at 4 bits: 304,508,553,216 x 0.5 x 1.0625 ~ 161.8 GB
- All other linears at 8 bits: 8,818,462,720 x 1.0 x 1.0625 ~ 9.4 GB
- Total weights ~ 171.2 GB (~159.4 GiB). Compare measured UD-IQ4_XS GGUF
  baseline in this repo: 156,822,111,075 B (146.05 GiB). The mixed CT
  artifact is ~9% larger and excludes fp8 KV and CUDA context. On a
  four-card 64 GiB CMP node (256 GiB VRAM), weights alone fit with ~85 GB
  headroom, but per-card balance under TP4 needs a live check (gate 3).
- Labels: inferred from public config.json; not downloaded, built, or served.

## Execution status and outcome

Nothing executed. This note is the static feasibility gate requested by
issue #66. No quantization run, no download, no GPU lease, no upload.

## Blocker (to unlock gates 2-5)

1. A working Glm5Next serving path requires vLLM PR 53906, currently open
   and merge-blocked. Open runtime bugs against it: #54317 (illegal memory
   access in KDA path on multi-GPU) and #54458 (hybrid KV page alignment
   inflates block usage). Re-check both at pin time.
2. No mixed W4A16/W8A16 Glm5Next CT checkpoint exists publicly; producing
   one requires an explicit quantization-resource lease (GPU/CPU time and
   roughly 340 GB scratch on shared storage) and is out of scope for this
   repo-only gate.

## Gates pending explicit resource lease

- Gate 2: pin 53906 head + llm-compressor revision; write mixed-scheme
  recipe; run compressor on a CPU/dense-only subset first.
- Gate 3: bounded TP=4 boot smoke on the four-card CMP lease with capped
  max-model-len, gpu-memory-utilization 0.85, stop on any Xid or thermal
  limit.
- Gate 4: quality spot-check (structured + prose buckets from the existing
  Phase C harness) versus the UD-IQ4_XS baseline receipts.
- Gate 5: only if gate 4 shows a correctness or latency win versus the
  baseline: full Phase-C-style measurement ladder.

## Evidence

- vLLM PR 53906 (Glm5Next, open/blocked): https://github.com/vllm-project/vllm/pull/53906
- vLLM PR 48918 (CT WNA16 MoE, merged): https://github.com/vllm-project/vllm/pull/48918
- vLLM issues #54317 and #54458 (open runtime bugs, linked by number)
- llm-compressor mixed-precision precedent: https://github.com/vllm-project/llm-compressor/pull/2940
- CT per-target scheme dispatch: vLLM main compressed_tensors.py,
  get_scheme_dict + _add_fused_moe_to_target_scheme_map (inspected 2026-08-31)
- Public config.json: https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json
- Measured size baseline: results/ in this repository (Phase C receipts,
  UD-IQ4_XS 156,822,111,075 B).

## Re-run instructions

Not applicable yet: static-only note. Gates 2-5 each require their own
pinned-revision runbook, written only when the matching resource lease is
granted on the tracking issue.
