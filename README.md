# GLM-5.3-Flash on CMP 170HX

Every attempt — successful, failed, or statically impossible — to run
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on
[NVIDIA CMP 170HX](https://github.com/PixelML/club-170hx) (GA100, SM80, 64 GiB HBM2e) cards.

One repository per model family/workload: all quantizations, runtimes, and
attempt outcomes live here. See [AGENTS.md](AGENTS.md) for the publication
boundary and evidence rules. DGX Spark deployment is documented separately in
[PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark).

## Hardware target

Four-card CMP 170HX test node: 4 x 64 GiB = 256 GiB aggregate VRAM, SM80
(Ampere), PCIe Gen2 x4 per card in the current test guest, 180 W per-card
benchmark power policy with forced airflow. The node exposed three cards when
the earliest attempts below were evaluated (2026-08-30); those records keep
their three-card arithmetic as history. Generic labels only; see the club
repository for full node documentation.

## Comparison table

| # | Checkpoint (revision) | Exact bytes | Quantization | Runtime | SM80 support | Static fit (per card, current 4-card topology) | Execution status | Blocker | Evidence |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) @ `11d73216cd636238e82e1d77fe1042ffab36e7fa` | ~194.4 GiB download (community-reported) | NVFP4 (ModelOpt, marlin MoE) | vLLM fork (SM121 image) | **No** — sparse-MLA path targets SM12x FlashInfer backends (measured: registry + PR review) | ~48.6 GiB/card at TP=4 — fits on paper (inferred, community-reported size); three-card era fit was ~64.8 GiB/card at TP=3, over budget | **Failed** (compatibility-only, 2026-08-30) | `glm5_next` absent from upstream vLLM registry; support PR [vllm#53906](https://github.com/vllm-project/vllm/pull/53906) open, SM90+ only | [attempt record](attempts/nvfp4-vllm-sm121/README.md) |
| 2 | [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f` | 175,642,157,752 B = 175.64 GB = 163.58 GiB total (measured, HF blob sum at pinned rev) | EXL3/TR3 4 bpw (uniform K4, routed experts) | ExLlamaV3 via SM121 fork image | **Untested on SM80** — quant gate allows Ampere+ (inferred from `get_min_capability()`), but kernels ship as `sm_121a` cubins | ~40.9 GiB/card at TP=4 with ~23 GiB/card headroom (inferred; expert-layout skew unverified); three-card era fit was ~54.5 GiB/card at TP=3 | **Failed** (compatibility-only, 2026-08-30) | SM121-only binary distribution; no SM80 build exists or is documented; NoPE-MLA attention needs a non-sparse SM80 backend that no runtime provides | [attempt record](attempts/exl3-tr3-4bpw-exllamav3/README.md) |
| 3 | [cyankiwi/GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4) @ `3999f9bf2c3e3064790af5a2d12d19090fc97f4d` | **212,721,952,636 B = 198.1 GiB** (measured, HF API blob sizes) | AWQ INT4 (pack-quantized, group 32, symmetric=false) | upstream vLLM (if `glm5_next` were supported) | **No** — same runtime blocker as row 1 | ~49.5 GiB/card at TP=4 — fits on paper (measured blob sizes); three-card era fit was 66.0 GiB/card at TP=3, 2.04 GiB over budget | **Failed on the three-card node** (static fit, 2026-08-30); fit arithmetic cleared by the 4-card topology | Runtime blocker unchanged: `glm5_next` absent from upstream vLLM | [attempt record](attempts/awq-int4-vllm/README.md) |
| 4 | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (official FP8) and [BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) | >= ~328 GB FP8; BF16 larger (community-reported) | FP8 / BF16 | any | n/a — memory | **No fit** — ~76+ GiB/card at TP=4 (FP8; BF16 larger) | **Failed** (static fit) | Size alone, at either topology | [attempt record](attempts/fp8-bf16-reference/README.md) |
| 5 | Future: true 3-bit / W4A16-with-exclusions <= ~55 GiB per card | — | AWQ/GPTQ W4A16-class | upstream vLLM, pending `glm5_next` support | **Untested** — depends on upstream runtime work | Would fit at TP=4 (inferred); the AWQ row above already fits on paper at 4 cards | **Not attempted** | Runtime blocker, not size, on the 4-card node | [attempt record](attempts/future-small-quant/README.md) |

Quantization legend: NVFP4 = NVIDIA 4-bit floating point; EXL3/TR3 = TurboDerp
3-bit-class quant with group-wise codebooks; AWQ = activation-aware weight
quantization. "Static fit" = weights-only budget per card at the node's current
4-card topology (TP=4), before CUDA context (~0.5-1 GiB), activations, and KV
cache; rows retain their three-card-era TP=3 arithmetic as history where it
differs.

## Current conclusion (2026-08-30, four-card node)

The two blockers that killed every early attempt are resolved on paper as of
2026-08-30:

1. An SM80 runtime exists: the [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)
   DSA fork builds for `sm_80` (measured CMake configure; see the
   [GGUF attempt](attempts/gguf-ud-iq4xs-llamacpp/README.md)). vLLM paths
   remain blocked: `glm5_next` is absent from
   [upstream vLLM's model registry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
   (measured 2026-08-30); the only known support PR
   ([vllm#53906](https://github.com/vllm-project/vllm/pull/53906)) is open,
   unmerged, and targets SM90+.
2. Fitting checkpoints exist: the UD-IQ4_XS GGUF (146.05 GiB measured) fits
   4 x 64 GiB with ~27.5 GiB/card margin at an even split, and the EXL3/TR3
   4 bpw (163.58 GiB measured) and AWQ INT4 (198.1 GiB measured) also fit on
   paper at TP=4.

**Measured validation (Phase C, 2026-08-30, four-card node):** the GGUF pairing
served successfully. The UD-IQ4_XS GGUF on the sm_80 llama.cpp fork reached
17.73 tok/s median single-stream (5 reps), 17.71 tok/s aggregate at c=4
(two-run median), with a 41-repetition soak completing cleanly. The corrected 26-task
evaluation scored 21/26 overall — math 8/8, instruction 4/5, long-context 3/3
on the tuning set, plus 4/4 held-out math and 1/1 held-out code — after two
harness defects were found and fixed during the phase (a coding sandbox that
dropped the candidate module, and completion budgets that starved reasoning
output before any answer bytes). The four remaining misses (two coding, two held-out instruction) are
recorded as genuine failures at the evaluated budget; the committed receipts
capture tokens/latency/prediction only, so per-miss behavioral diagnoses are
not claimed from this data. Full receipt, per-request data, charts, and rebench entry:
[results/summary.csv](results/summary.csv),
[results/phase63/README.md](results/phase63/README.md),
[results/phase63/charts/](results/phase63/charts/),
[bench/rebench.sh](bench/rebench.sh). The EXL3/TR3, AWQ, NVFP4, and FP8/BF16
rows above remain blocked as recorded; all failed attempts stay in this
repository as history.

## License

Apache-2.0 for this repository's own content. Checkpoints and drafts referenced
here carry their own licenses (base model, EXL3 source-available, DFlash2
CC BY-NC-ND); record and respect them per attempt.
