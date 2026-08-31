# Attempt — W4A16 experts + W8A16 dense (vLLM compressed-tensors)

Status: static-fit-only. All sizes below are INFERRED/untested until an emitted checkpoint is measured.
Date: 2026-08-31

## Checkpoint (planned)

- Base: zai-org/GLM-5.3-Flash pinned at `04c4e9e95c5da8862dced7e5056455116f83a7e0`
  (Glm5NextForConditionalGeneration; hidden 4096, 45 layers + 1 MTP layer, 288
  routed experts, 1 shared expert per MoE layer, first 3 layers dense at
  intermediate 12288, MoE intermediate 2048, vocab 154880).
- Quantization (planned): compressed-tensors mixed scheme. Config to be pinned
  at emit time: routed-expert linears W4A16, symmetric, per-group-128
  (group size verified against the checkpoint''s existing fp8 scale layout:
  weight [2048,4096] with weight_scale_inv [16,32] => 128x128 groups), scale
  dtype float16, zero point none; all other linears W8A16, symmetric,
  per-group-128, scale dtype float16, zero point none. Target precedence:
  routed experts first, then "all other Linear" catch-all group. Untouched
  (stay BF16, excluded from quantized byte math): norms, router gate weights
  under the W8 catch-all only if Linear; layernorm-family and hc_* KDA
  coefficients are non-Linear and remain BF16/F32 as stored.
- KV cache: fp8 as in the measured GGUF baseline; excluded from weight math.
- License: base-model license applies to any derived artifact; conversion
  would be local-only, no weights published.

## Complete checkpoint inventory (from the pinned safetensors index, 76,108 tensors)

Official fp8 artifact totals 321,342,220,638 parameters / 328,337,455,672 B
(305.8 GiB) across 62 shards. Class breakdown (params; dominant dtype):

| class | params | stored dtype |
| --- | ---: | --- |
| routed experts (gate/up/down weights) | 311,653,564,416 | FP8 e4m3 |
| routed expert weight_scale_inv | 19,021,824 | F32 (g=128) |
| attention (MLA/KDA projections incl. q_a/q_b/kv_a/kv_b/o/b_proj/f_b/indexer) | 6,181,944,704 | BF16 + 1.2 B FP8 |
| shared expert | 1,082,196,480 | FP8 e4m3 |
| embeddings (input) | 634,388,480 | BF16 |
| lm_head (untied) | 634,388,480 | BF16 |
| vision tower | 563,627,008 | BF16 |
| dense MLP (layers 1-3) | 503,749,728 | BF16/FP8/F32 mix |
| MTP head (eh_proj + shared_head.head) | 33,558,528 | BF16 |
| KDA hc_* coefficients | 35,391,870 | F32/BF16 |
| norms | 389,120 | BF16 |

Total check: 321,342,220,638 = 311,653,564,416 + 19,021,824 + 6,181,944,704 +
1,082,196,480 + 634,388,480 + 634,388,480 + 563,627,008 + 503,749,728 +
33,558,528 + 35,391,870 + 389,120. The earlier 313.3 B count and the
4x4096x8192 attention shortcut in the first revision of this note were wrong;
the real attention block is the Glm5Next hybrid MLA/KDA set above (e.g. kv_a
[512,4096], kv_b [32768,512], o_proj [4096,16384], q_a [1536,4096], q_b
[16384,1536]).

## Planned-mixed-scheme byte arithmetic (INFERRED, g=128, fp16 scales, no zp)

- Routed experts W4: 311,653,564,416 x 0.5 B + (311,653,564,416 / 128) x 2 B
  = 160,706,177,280 B (160.7 GB).
- Everything else W8: 9,669,634,398 x 1 B + (9,669,634,398 / 128) x 2 B
  = 9,820,722,434 B (9.8 GB). This W8 target set includes the MTP head, vision
  tower, and dense MLPs; MTP/vision can also be left BF16 as a variant, which
  would add ~1.2 GB.
- Total planned artifact: ~170,526,899,714 B = 170.5 GB = 158.8 GiB
  (1.087x the measured UD-IQ4_XS GGUF at 156,822,111,075 B / 146.05 GiB).
  The single 1.0625x packing factor used in the first revision was wrong;
  per-scheme scale bytes are shown explicitly above.
- Labels: inferred from the pinned safetensors index headers; no checkpoint
  emitted, downloaded, or served under this scheme.

## Fit on the four-card 64 GiB CMP node (256 GiB aggregate)

- Weights alone: ~158.8 GiB, leaving ~97.2 GiB aggregate unallocated VRAM
  before runtime state. This is NOT serving headroom: rank imbalance under
  TP4, quant repacking buffers, CUDA graphs, NCCL/host-routed collectives,
  fp8 KV cache, and activations all consume from it. Per-card balance needs a
  live check (gate 3). No per-card claim is made from the aggregate.

## Runtime (static analysis, untested)

- vLLM Glm5Next support: upstream PR 53906 (open, merge-blocked) at its cited
  head implements SM90/SM100/SM120 paths; it does NOT make this SM80 node a
  supported serving target, and PP is explicitly gated off upstream, so PP4 is
  unsupported there. TP4 on this node is plausible/untested and carries a
  material performance risk: four PCIe GPUs without P2P cannot use vLLM custom
  all-reduce (unsupported for this topology; must be disabled), leaving
  host-routed TP collectives. Any community SM80 backport (e.g. the wtdcode
  backport image, which registers Glm5Next) is a separate third-party artifact
  and is not upstream vLLM; its own review/pin precedes any use.
- Compressed-tensors mixed schemes: per-target config_groups dispatch and the
  WNA16 MoE backend (vLLM PR 48918, merged) exist in main; loader references
  inspected in main compressed_tensors.py.

## Execution status and outcome

Nothing executed. Static feasibility gate only: no quantization run, no
download, no GPU lease, no upload.

## Blockers (to unlock gates 2-5)

1. Serving path: upstream 53906 is open/merge-blocked and does not cover SM80
   serving; a separately reviewed community backport is required, pinned and
   license-checked, before any gate 3 work.
2. No mixed W4A16/W8A16 Glm5Next CT checkpoint exists publicly; producing one
   requires an explicit quantization-resource lease. Peak host storage at
   emit time must plan for the pinned fp8 source (328,337,455,672 B /
   305.8 GiB) plus the output artifact (~170.5 GB) plus toolchain temporaries;
   a toolchain-specific peak estimate (host RAM + scratch) is produced at
   gate 2, not assumed here. The earlier ~340 GB scratch figure was
   unsupported.

## Gates pending explicit resource lease

- Gate 2: pin 53906 head (or chosen backport) + llm-compressor revision; write
  the mixed-scheme recipe with the exact config_groups from the paragraph
  above; run the compressor on a CPU/dense-only subset first; measure peak
  host RAM/disk.
- Gate 3: bounded TP=4 boot smoke on the four-card CMP lease, capped
  max-model-len, gpu-memory-utilization 0.85, stop on any Xid or thermal limit.
- Gate 4: quality spot-check (structured + prose buckets from the existing
  Phase C harness) versus the UD-IQ4_XS baseline receipts.
- Gate 5: only if gate 4 shows a correctness or latency win versus the
  baseline: full Phase-C-style measurement ladder.

## Evidence

- Official pinned model: https://huggingface.co/zai-org/GLM-5.3-Flash/tree/04c4e9e95c5da8862dced7e5056455116f83a7e0
- Official pinned config: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json
- vLLM PR 53906 (open/blocked; SM90/100/120 paths): https://github.com/vllm-project/vllm/pull/53906
- vLLM PR 48918 (CT WNA16 MoE, merged): https://github.com/vllm-project/vllm/pull/48918
- vLLM issue #54317: https://github.com/vllm-project/vllm/issues/54317
- vLLM issue #54458: https://github.com/vllm-project/vllm/issues/54458
- llm-compressor mixed-precision precedent: https://github.com/vllm-project/llm-compressor/pull/2940
- CT per-target scheme dispatch: vLLM main compressed_tensors.py, get_scheme_dict + _add_fused_moe_to_target_scheme_map (inspected 2026-08-31)
- Measured size baseline: results/ in this repository (Phase C receipts, UD-IQ4_XS 156,822,111,075 B).

## Re-run instructions

Not applicable yet: static-only note. Gates 2-5 each require their own
pinned-revision runbook, written only when the matching resource lease is
granted on the tracking issue.
