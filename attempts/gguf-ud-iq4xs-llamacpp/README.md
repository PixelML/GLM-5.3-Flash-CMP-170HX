# Attempt — GGUF UD-IQ4_XS on llama.cpp (DSA branch)

Status: in-progress (weights staging, runtime building)
Date: 2026-08-30

## Checkpoint

- Repository + exact revision: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), `UD-IQ4_XS/` subfolder (5 shards, revision checked 2026-08-30)
- Exact download bytes (HF API blob sizes, summed): **156,822,111,075 bytes = 146.05 GiB**
- Installed size: same as download (GGUF is a flat container; no conversion step)
- Quantization: Unsloth Dynamic UD-IQ4_XS (importance-matrix 4-bit, per-layer mixed ``IQ4_XS``/``Q4_K``/``Q8_0``; quantization config embedded in GGUF metadata)
- License: **MIT** (from model card)
- Source: measured (HF API `?blobs=true`, sum of 5 shard sizes)

## Runtime

- Engine: llama.cpp fork [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) branch `glm5next/upstream` @ `00699716c275498ff84d71e329178fe21cba56a6`
- Upstream PR: [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) — **open, not merged** as of 2026-08-30 (community-reported status; fork head is the buildable implementation)
- SM80 support: **yes** — GGML CUDA backend compiles `sm_80` device code natively (measured: CMake configure with `-DCMAKE_CUDA_ARCHITECTURES=80` succeeded against CUDA 13.0.88)
- Topology: single-process `llama-server` with `--split-mode layer` across 3 cards, weight- and KV-split by per-card free VRAM
- Source: measured (local build), fork topology inferred from llama.cpp multi-GPU docs

## Static fit calculation

- Per-card budget: 64 GiB = 68,719,476,736 bytes
- Weights at TP=3 (even split): 156,822,111,075 / 3 = 52,274,037,025 B = **48.68 GiB per card**
- Per-card margin before context/KV: **~15.3 GiB** — enough for CUDA context (~0.6 GiB), a 16K-token KV pool for this MLA architecture (~2–4 GiB estimated, untested), and compute headroom
- Unlike the AWQ attempt, this checkpoint has real margin; the binding question is runtime behavior, not arithmetic

## Execution status and outcome

In progress. Runtime is compiling; weights are staging in shared model storage.
No serving or measurement has happened yet. This record will be updated with
measured load time, TTFT, throughput, and thermals once the model is served.

## Blocker

None so far. The two blockers that killed every earlier attempt (no fitting
quant, no SM80 runtime for the architecture) both clear with this pairing.

## Evidence

- HF API blob sizes: [api/models/unsloth/GLM-5.3-Flash-GGUF?blobs=true](https://huggingface.co/api/models/unsloth/GLM-5.3-Flash-GGUF?blobs=true) (checked 2026-08-30)
- llama.cpp GLM-DSA support: [`src/models/glm-dsa.cpp`](https://github.com/unslothai/llama.cpp/blob/glm5next/upstream/src/models/glm-dsa.cpp) on the fork head
- Fork branch head: `00699716c275498ff84d71e329178fe21cba56a6` (local clone verified)
- CUDA sm_80 compile: local build log (on the benchmark node, not committed)

## Re-run instructions

Once measured results land, this section will contain the exact pinned
`llama-server` command, environment variables, and stop conditions used for
the benchmark. Draft:

1. Build the pinned fork with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80`.
2. Stage the 5 GGUF shards in shared model storage (path not published here).
3. Launch `llama-server` with `--split-mode layer --tensor-split <free-VRAM-derived>`,
   `--ctx-size 16384` initially, `--n-gpu-layers 999`.
4. Abort at 80 °C core / 85 °C memory / any Xid / GPU disappearance.
