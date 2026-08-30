# Attempt — GGUF UD-IQ4_XS on llama.cpp (DSA branch)

Status: staged/in-progress (checkpoint staging observed, not served)
Date: 2026-08-30

## Checkpoint

- Repository + exact revision: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), `UD-IQ4_XS/` subfolder (5 shards, revision checked 2026-08-30)
- Exact download bytes (HF API blob sizes, summed): **156,822,111,075 bytes = 146.05 GiB**
- Installed size: same as download (GGUF is a flat container; no conversion step)
- Quantization: Unsloth Dynamic UD-IQ4_XS (importance-matrix 4-bit, per-layer mixed ``IQ4_XS``/``Q4_K``/``Q8_0``; quantization config embedded in GGUF metadata)
- License: **MIT** (from model card)
- Source: measured (HF API `?blobs=true`, sum of 5 shard sizes)

## Runtime

- Engine: llama.cpp fork [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) @ commit `00699716c275498ff84d71e329178fe21cba56a6`
- Upstream PR: [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) — **open, not merged** as of 2026-08-30 (community-reported status; fork head is the buildable implementation)
- SM80 support: **unverified (inferred)** — the fork is pinned and its build flags are documented, but the local build log was never committed, so the earlier "measured build" claim is withdrawn; rebuild and commit a redacted log before serving
- Topology: single-process `llama-server` with `--split-mode layer` across 4 cards, weight- and KV-split by per-card free VRAM
- Source: measured (local build), fork topology inferred from llama.cpp multi-GPU docs

## Static fit calculation

- Per-card budget: 64 GiB = 68,719,476,736 bytes
- Weights at TP=4 (even split): 156,822,111,075 / 4 = 39,205,527,769 B = **36.51 GiB per card**
- Per-card margin before context/KV: **~27.5 GiB** — enough for CUDA context (~0.6 GiB), a 16K-token KV pool for this MLA architecture (~2–4 GiB estimated, untested), and compute headroom. (The three-card-era split was 48.68 GiB/card with ~15.3 GiB margin.)
- Unlike the AWQ attempt, this checkpoint has real margin; the binding question is runtime behavior, not arithmetic

## Execution status and outcome

Blocked before first serve (2026-08-30). Checkpoint staging is in progress:
4 of 5 shards were present at the time of the last read-only check (128 GiB of
the measured 146.05 GiB total), with the fifth shard absent. No serving or
measurement has happened yet. This record will be updated with measured load
time, TTFT, throughput, and thermals once the model is served.

## Blocker

Resource conflict (measured, redacted): an unrelated workload occupied the
cards, then the guest entered maintenance — no serving attempt has been made
yet. The attempt waits for full resource release (all four cards), a safe root
filesystem, and a preflight pass. The two blockers that killed every earlier
attempt (no fitting quant, no SM80 runtime for the architecture) clear with
this pairing on the 4-card node, pending runtime verification.

**Secondary (inferred, static arithmetic):** `--no-mmap` serving is infeasible
on this host — 146.05 GiB weights (measured HF blob sum) vs 94 GiB host RAM
(measured). Baseline must use mmap; see bench/experiment-plan.md.

Harness status: bench/ contains a read-only preflight guard
(preflight.sh: vLLM check, per-card free-VRAM check, shard-completeness
check), the fixed ~1K-token prompt, the experiments CSV, and the one-factor
experiment plan.

### Candidate matrix (source-cited, 2026-08-30)

| Candidate | Checkpoint size | Fit at 4 x 64 GiB | SM80 runtime | Status |
|---|---|---|---|---|
| GGUF UD-IQ4_XS (unsloth) @ HF repo rev `2975ab414d30340466d8c51533c6e91f0cca64c1` | 156,822,111,075 B = 146.05 GiB (measured, HF blob sum) | ~36.5 GiB/card, ~27.5 GiB margin (inferred, arithmetic) | llama.cpp fork @ `00699716c275498ff84d71e329178fe21cba56a6` builds sm_80 (measured) | **primary candidate**; staging, not yet served |
| EXL3/TR3 4 bpw (brandonmusic; source snapshot pinned in its attempt record) | 163.58 GiB weights, 120 shards (measured at pinned rev) | ~40.9 GiB/card, ~23 GiB margin (inferred; expert skew unverified) | ExLlamaV3 has no SM80 build (verified via source review); SM121-only distribution | blocked before boot; see attempts/exl3-tr3-4bpw-exllamav3 |
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
