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

Blocked before first serve (2026-08-30). Runtime is built and verified
(pinned commit checked out clean; sm_80 binary present). Weights staging is
in progress: at 2026-08-30T05:08Z only 2 of 5 shards were staged and one was
actively growing (measured ~22.5 of ~29 GiB). No serving or measurement has
happened yet. This record will be updated with measured load time, TTFT,
throughput, and thermals once the model is served.

## Blocker

**GPU contention (measured, 2026-08-30T05:07Z):** the three-card node is
occupied by another workload — a vLLM server in pipeline-parallel mode across
all three cards (three worker processes, started ~02:47 local, root-owned,
serving a different model with speculative decoding). Free VRAM measured
~6.0 / 7.3 / 1.3 GiB per card against ~48.7 GiB/card needed for an even
3-way weight split. Per safety rules the workload is left untouched (never
kill processes we did not start); the attempt waits until the cards free up.
The two blockers that killed every earlier attempt (no fitting quant, no
SM80 runtime for the architecture) still clear with this pairing.

**Secondary (inferred, static arithmetic):** `--no-mmap` serving is infeasible
on this host — 146.05 GiB weights (measured HF blob sum) vs 94 GiB host RAM
(measured). Baseline must use mmap; see bench/experiment-plan.md.

Harness status: bench/ contains a read-only preflight guard
(preflight.sh: vLLM check, per-card free-VRAM check, shard-completeness
check), the fixed ~1K-token prompt, the experiments CSV, and the one-factor
experiment plan.

### Candidate matrix (source-cited, 2026-08-30)

| Candidate | Checkpoint size | Fit at 3 x 64 GiB | SM80 runtime | Status |
|---|---|---|---|---|
| GGUF UD-IQ4_XS (unsloth) @ HF repo rev `2975ab414d30340466d8c51533c6e91f0cca64c1` | 156,822,111,075 B = 146.05 GiB (measured, HF blob sum) | ~48.7 GiB/card, ~15.3 GiB margin (inferred, arithmetic) | llama.cpp fork @ `00699716c275498ff84d71e329178fe21cba56a6` builds sm_80 (measured) | **primary candidate**; staging, not yet served |
| EXL3/TR3 4 bpw (brandonmusic via Mia-AiLab mirror) | ~176 GB, 120 shards (community-reported) | ~54.6 GiB/card, ~9 GiB margin (inferred; expert skew unverified) | ExLlamaV3 has no SM80 build (verified via source review); SM121-only distribution | blocked before boot; see attempts/exl3-tr3-4bpw-exllamav3 |
| EXL3 3.0bpw (0xSero) @ `8b099bf276507a17faea920deff3f62d5597fb52` | 130-shard layers/ layout (community-reported size; not re-measured) | not established (no serving path) | `requires_custom_loader: true`; `runtime_status: pending_full_server`; kernel primitive only (community-reported repo metadata) | **not attempted**; not a servable artifact on llama.cpp or ExLlamaV3; quality gate FAIL per repo-reported KL 0.153 / ppl delta 0.093 / top-1 agree 0.873; missing tokenizer files |
| vLLM SM80 path | n/a | n/a | vLLM PR #53906 open, SM90+ only (verified upstream PR state) | blocked |

Revision note: GGUF repo revision pinned via HF API on 2026-08-30; shard count
and byte total re-measured during staging. EXL3 3.0bpw numbers are cited from
the checkpoint card and config at the pinned commit — community-reported, not
independently re-measured.

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
- shard 1 sha256 verified OK vs upstream LFS oid (read-only check, 2026-08-30); shards 3-5 verifiable only after staging completes; see results/expected-sha256.txt for the pinned hashes.
