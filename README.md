# GLM-5.3-Flash on CMP 170HX

Every attempt — successful, failed, or statically impossible — to run
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on
[NVIDIA CMP 170HX](https://github.com/PixelML/club-cmp170hx) (GA100, SM80, 64 GiB HBM2e) cards.

One repository per model family/workload: all quantizations, runtimes, and
attempt outcomes live here. See [AGENTS.md](AGENTS.md) for the publication
boundary and evidence rules. DGX Spark deployment is documented separately in
[PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark).

## Hardware target

Three-card CMP 170HX test node: 3 x 64 GiB = 192 GiB aggregate VRAM, SM80
(Ampere), PCIe Gen2 x4 per card in the current test guest, 180 W per-card
benchmark power policy with forced airflow. Generic labels only; see the club
repository for full node documentation.

## Comparison table

| # | Checkpoint (revision) | Exact bytes | Quantization | Runtime | SM80 support | Static fit (TP=3, 64 GiB/card) | Execution status | Blocker | Evidence |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) @ `11d73216cd636238e82e1d77fe1042ffab36e7fa` | ~194.4 GiB download (community-reported) | NVFP4 (ModelOpt, marlin MoE) | vLLM fork (SM121 image) | **No** — sparse-MLA path targets SM12x FlashInfer backends (measured: registry + PR review) | no fit check needed — runtime incompatible | **Failed** (compatibility-only, 2026-08-30) | `glm5_next` absent from upstream vLLM registry; support PR [vllm#53906](https://github.com/vllm-project/vllm/pull/53906) open, SM90+ only | [attempt record](attempts/nvfp4-vllm-sm121/README.md) |
| 2 | [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f` | ~171.8 GiB download / ~176 GB installed (community-reported) | EXL3/TR3 4 bpw (uniform K4, routed experts) | ExLlamaV3 via SM121 fork image | **Untested on SM80** — quant gate allows Ampere+ (inferred from `get_min_capability()`), but kernels ship as `sm_121a` cubins | ~64.5 GiB/card if TP=3 — over-budget once KV/CUDA context added (inferred) | **Failed** (compatibility-only, 2026-08-30) | SM121-only binary distribution; no SM80 build exists or is documented; NoPE-MLA attention needs a non-sparse SM80 backend that no runtime provides | [attempt record](attempts/exl3-tr3-4bpw-exllamav3/README.md) |
| 3 | [cyankiwi/GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4) @ `3999f9bf2c3e3064790af5a2d12d19090fc97f4d` | **212,721,952,636 B = 198.1 GiB** (measured, HF API blob sizes) | AWQ INT4 (pack-quantized, group 32, symmetric=false) | upstream vLLM (if `glm5_next` were supported) | **No** — same runtime blocker as row 1 | **66.0 GiB/card — exceeds 64 GiB physical by 2.04 GiB before any KV** (measured from blob sizes) | **Failed** (static fit, 2026-08-30) | Cannot fit 3 x 64 GiB at TP=3; pipeline parallelism does not rescue layer-wise splits | [attempt record](attempts/awq-int4-vllm/README.md) |
| 4 | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (official FP8) and [BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) | >= ~328 GB FP8; BF16 larger (community-reported) | FP8 / BF16 | any | n/a — memory | **No fit** — ~165+ GiB/card | **Failed** (static fit) | Size alone | [attempt record](attempts/fp8-bf16-reference/README.md) |
| 5 | Future: true 3-bit / W4A16-with-exclusions <= ~55 GiB per card | — | AWQ/GPTQ W4A16-class | upstream vLLM, pending `glm5_next` support | **Untested** — depends on upstream runtime work | Would fit (inferred) | **Not attempted** | No checkpoint meeting the budget existed as of 2026-08-30; HF search on that date found none | [attempt record](attempts/future-small-quant/README.md) |

Quantization legend: NVFP4 = NVIDIA 4-bit floating point; EXL3/TR3 = TurboDerp
3-bit-class quant with group-wise codebooks; AWQ = activation-aware weight
quantization. "Static fit" = weights-only budget per card at TP=3, before CUDA
context (~0.5-1 GiB), activations, and KV cache.

## Current conclusion (2026-08-30)

**GLM-5.3-Flash cannot run on CMP 170HX today.** Two independent blockers:

1. No SM80-capable runtime: `glm5_next` is absent from
   [upstream vLLM's model registry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
   (measured 2026-08-30), and the only known support PR
   ([vllm#53906](https://github.com/vllm-project/vllm/pull/53906)) is open,
   unmerged, and targets SM90+.
2. No checkpoint that fits: the smallest public quant is 198.1 GiB — over the
   192 GiB total VRAM of the three-card node before any runtime overhead.

Re-run the benchmark methodology in
[attempts/awq-int4-vllm/README.md](attempts/awq-int4-vllm/README.md) once either
blocker is resolved; the usage-token-counted harness design there is portable to
any future candidate.

## License

Apache-2.0 for this repository's own content. Checkpoints and drafts referenced
here carry their own licenses (base model, EXL3 source-available, DFlash2
CC BY-NC-ND); record and respect them per attempt.
