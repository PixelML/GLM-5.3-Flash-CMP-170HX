# Attempt — W4A16 AutoRound on upstream vLLM

Status: registry-blocked, not executed
Date: 2026-09-02

## Checkpoint

- [Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) @ `5eee1846f0321058ed73745f9aa16f2aaf0fc0a0` (last modified 2026-09-01)
- **181,505,393,058 bytes = 169.04 GiB** across 34 safetensors shards plus
  `model_extra_tensors.safetensors` (3.85 GiB) and `model_extra_conv.safetensors`
  (measured: HF API `?blobs=true` response, summed per-file)
- W4A16, AutoRound, 4-bit (`quantization_config.json`: `quant_method: auto-round`, `bits: 4`)
- License: MIT on this checkpoint's card; base model zai-org/GLM-5.3-Flash also
  cards `license: mit` (measured via HF API)
- Architecture: `Glm5NextForConditionalGeneration` / `model_type: glm5_next`,
  a hybrid of `linear_attention` and `deepseek_sparse_attention` (DSA/kpool
  indexer) blocks per layer (measured from `config.json`'s `layer_types`),
  `kv_lora_rank: 512`, `index_topk: 2048` — same MLA/DSA family as DeepSeek-V4
  plus a linear-attention layer type not previously served on this node.

## Runtime

Upstream vLLM's model registry (`vllm/model_executor/models/registry.py` on
main, checked 2026-09-02) has **zero** entries for `glm5_next` or `Glm5Next`.
The architecture is not registered in any released or main-branch vLLM.

- Support PR [vllm#53906](https://github.com/vllm-project/vllm/pull/53906)
  ("[Model] add GLM-5.3-Flash support") is open, unmerged, and targets
  **SM90+ only**.
- Open bugs against the same architecture, none scoped to SM80: linear
  attention backend gap (`Glm5NextTextLinearAttention not supported`,
  [#54062](https://github.com/vllm-project/vllm/issues/54062)); no sparse-MLA
  path on SM89/Ada ([#54059](https://github.com/vllm-project/vllm/issues/54059));
  no SM120 sparse-MLA path for rope-free MLA
  ([#53963](https://github.com/vllm-project/vllm/issues/53963)); recurring CUDA
  illegal memory access on 4xB200 across three kernels
  ([#54317](https://github.com/vllm-project/vllm/issues/54317)).

This is the same blocker already recorded for the AWQ INT4 row in this
repository's comparison table, confirmed here against the specific
W4A16/AutoRound artifact and against the current registry state. It is a
missing-model-class problem, not a version pin or a build target: no vLLM
release, old or new, implements `glm5_next` for any GPU architecture, and the
one open PR that adds it does not claim SM80.

## Static fit calculation (hypothetical — recorded for when upstream lands)

- Four-card node, TP=4 even split: 181,505,393,058 / 4 = 45,376,348,264 B =
  **42.26 GiB per card**, leaving ~21.7 GiB/card for CUDA context, KV cache,
  and activations out of the 64 GiB/card budget.
- PP2 x 2 data-parallel instances (vs one PP4, vs TP2) is the prior favored by
  this node's no-P2P, PCIe Gen1/Gen2 topology, per the pipeline-partition
  findings already recorded elsewhere on this node
  (`VLLM_PP_LAYER_PARTITION=11,11,11,10` worked for a comparably sized
  DeepSeek-family model; TP splits pay the interconnect cost on every layer).
  Untestable until the model boots at all.

## Speculative decoding / MTP

No `glm5_next` entry exists in vLLM's MTP model registry (only
`Glm4MoeMTPModel`, `Glm4MoeLiteMTPModel`, and `GlmOcrMTPModel`, all GLM-4
family). No DFlash2 or other third-party draft artifact was found for
GLM-5.3-Flash. Moot until base-model support exists.

## Execution status and outcome

Not executed. No GPU lease was taken for this attempt: the blocker is
architecture support, not fit, topology, or hardware, so a GPU run would not
produce a different answer. Killed by the registry check before any download.

## Blocker

Missing `glm5_next` support in upstream vLLM, for any GPU architecture. The
one open support PR (#53906) targets SM90+ and has open correctness bugs even
there; SM80 is out of scope of that PR entirely.

## Re-run instructions

Blocked until `glm5_next` lands in a released vLLM with an SM80-covering
kernel path, or until this checkpoint is served on a different SM80-capable
runtime. When either happens, use the methodology already recorded in the
AWQ INT4 attempt's re-run instructions (usage-object token counting, fixed
256-/900-token single-stream + concurrency sweep, cold-load/TTFT/peak
VRAM/power/temp capture, 80 C/85 C/Xid stop conditions).

## Evidence

- Byte total and file list: HF API `?blobs=true` response, 2026-09-02
- Config/quant config: HF `raw/main/config.json` and `quantization_config.json`, 2026-09-02
- Registry check: [vllm registry.py on main](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py), no `glm5` match, 2026-09-02
